"""
Conversion layer: Utilization Score -> Fantasy Points.

Primary usage is RB/WR/TE (required), with optional QB support when QB is trained
on utilization and needs owner-facing fantasy-point projections.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, POSITIONS

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# Efficiency features used for utilization -> FP conversion (per position)
EFFICIENCY_FEATURES = {
    "QB": ["completion_pct", "yards_per_attempt", "td_rate", "int_rate", "rushing_yards", "rushing_attempts"],
    "RB": ["yards_per_carry", "yards_per_target", "catch_rate", "total_touches", "snap_share"],
    "WR": ["yards_per_reception", "yards_per_target", "catch_rate", "targets", "snap_share"],
    "TE": ["yards_per_reception", "yards_per_target", "catch_rate", "targets", "snap_share"],
}


class UtilizationToFPConverter:
    """
    Converts predicted utilization score to fantasy points using efficiency metrics.
    Tree-based (RF/XGB) per requirements for bounded 0-100 utilization -> FP.
    """
    def __init__(self, position: str):
        self.position = position
        self.model = None
        self.xgb_model = None
        self.scaler = None
        self.feature_names = []
        self.util_col = "utilization_score"
        self.calibration = None
        self.is_fitted = False

    def _get_input_cols(self, df: pd.DataFrame) -> List[str]:
        base = [self.util_col]
        for c in EFFICIENCY_FEATURES.get(self.position, []):
            if c in df.columns:
                base.append(c)
        return base

    def fit(self, df: pd.DataFrame, target_col: str = "fantasy_points",
            oof_utilization: Optional[np.ndarray] = None) -> "UtilizationToFPConverter":
        """Fit the utilization-to-FP converter.

        Args:
            df: DataFrame with utilization_score, efficiency features, and target.
            target_col: Column with actual fantasy points.
            oof_utilization: Optional array of out-of-fold predicted utilization
                scores (same length as df). When provided, training uses these
                noisy predictions instead of actual utilization_score, which
                reduces train/serve distribution mismatch (the converter will
                see noisy inputs at serve time too).
        """
        if not HAS_SKLEARN:
            self.is_fitted = False
            return self
        cols = self._get_input_cols(df)
        if self.util_col not in df.columns or target_col not in df.columns:
            self.is_fitted = False
            return self
        work = df.copy()
        # If OOF utilization provided, substitute it for actual utilization during training
        # so the converter learns to handle noisy predicted inputs (reduces cascaded error)
        if oof_utilization is not None and len(oof_utilization) == len(work):
            work[self.util_col] = oof_utilization
        sort_cols = [c for c in ["season", "week"] if c in work.columns]
        if sort_cols:
            work = work.sort_values(sort_cols).reset_index(drop=True)
        X = work[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y = work[target_col].values
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < 30:
            self.is_fitted = False
            return self
        X, y = X[valid], y[valid]
        self.feature_names = cols
        self.scaler = StandardScaler()
        n = len(X)
        split_idx = int(n * 0.8)
        if split_idx < 30 or n - split_idx < 20:
            split_idx = max(30, n - 20)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        Xs = self.scaler.fit_transform(X_train)
        # Per requirements: tree-based models (RF/XGBoost) for bounded 0-100 -> FP conversion
        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
        self.model.fit(Xs, y_train)
        if HAS_XGB:
            try:
                self.xgb_model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.05,
                    subsample=0.8, random_state=42, n_jobs=1,
                )
                self.xgb_model.fit(Xs, y_train)
            except Exception:
                self.xgb_model = None
        self.calibration = None
        self._conversion_conformal_q = None
        if len(X_val) >= 20:
            Xv = self.scaler.transform(X_val)
            raw_val = self._predict_raw_from_scaled(Xv)
            finite = np.isfinite(raw_val) & np.isfinite(y_val)
            if finite.sum() >= 20:
                calibrator = Ridge(alpha=1.0)
                calibrator.fit(raw_val[finite].reshape(-1, 1), y_val[finite])
                self.calibration = {
                    "slope": float(calibrator.coef_[0]),
                    "intercept": float(calibrator.intercept_),
                }
                # Compute calibrated predictions for conformal residuals
                cal_pred = self.calibration["slope"] * raw_val[finite] + self.calibration["intercept"]
                conv_residuals = np.abs(y_val[finite] - cal_pred)
                self._conversion_conformal_q = {
                    0.80: float(np.quantile(conv_residuals, 0.80)),
                    0.95: float(np.quantile(conv_residuals, 0.95)),
                }
        self.is_fitted = True
        return self

    def _predict_raw_from_scaled(self, Xs: np.ndarray) -> np.ndarray:
        """Raw converter output before optional linear calibration."""
        rf_pred = self.model.predict(Xs)
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(Xs)
            return 0.5 * rf_pred + 0.5 * xgb_pred
        return rf_pred

    def predict(self, utilization: np.ndarray, efficiency_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            return utilization
        if efficiency_df is not None and len(self.feature_names) > 1:
            cols = [c for c in self.feature_names if c in efficiency_df.columns]
            if cols:
                X = efficiency_df[cols].reindex(columns=self.feature_names, fill_value=0).fillna(0).values
            else:
                X = np.column_stack([utilization, np.zeros((len(utilization), len(self.feature_names) - 1))])
        else:
            X = np.column_stack([utilization, np.zeros((len(utilization), len(self.feature_names) - 1))]) if len(self.feature_names) > 1 else utilization.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0)
        Xs = self.scaler.transform(X)
        pred = self._predict_raw_from_scaled(Xs)
        if isinstance(self.calibration, dict):
            slope = float(self.calibration.get("slope", 1.0))
            intercept = float(self.calibration.get("intercept", 0.0))
            pred = slope * pred + intercept
        return pred

    def save(self, path: Path = None):
        path = path or MODELS_DIR / f"util_to_fp_{self.position.lower()}.joblib"
        joblib.dump({
            "position": self.position,
            "model": self.model,
            "xgb_model": getattr(self, "xgb_model", None),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "calibration": self.calibration,
            "is_fitted": self.is_fitted,
            "_conversion_conformal_q": getattr(self, "_conversion_conformal_q", None),
        }, path)

    @classmethod
    def load(cls, position: str, path: Path = None) -> "UtilizationToFPConverter":
        path = path or MODELS_DIR / f"util_to_fp_{position.lower()}.joblib"
        c = cls(position)
        if path.exists():
            d = joblib.load(path)
            c.model = d.get("model")
            c.xgb_model = d.get("xgb_model")
            c.scaler = d.get("scaler")
            c.feature_names = d.get("feature_names", [])
            c.calibration = d.get("calibration")
            c.is_fitted = d.get("is_fitted", False)
            c._conversion_conformal_q = d.get("_conversion_conformal_q")
        return c


def _generate_oof_utilization(subset: pd.DataFrame) -> Optional[np.ndarray]:
    """Generate OOF utilization predictions via 5-fold time-series CV.

    This lets us train the UtilizationToFPConverter on *noisy predicted*
    utilization rather than actuals, reducing cascaded error at serve time.
    """
    if not HAS_SKLEARN:
        return None
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import GradientBoostingRegressor

    if "utilization_score" not in subset.columns:
        return None

    # Use the same lag/rolling features available during training
    feature_candidates = [c for c in subset.columns
                          if c not in ("player_id", "name", "team", "position", "season",
                                       "week", "fantasy_points", "utilization_score",
                                       "opponent", "home_away")
                          and not c.startswith("target_")
                          and subset[c].dtype in ("int64", "float64", "int32", "float32")]
    if len(feature_candidates) < 5:
        return None

    X = subset[feature_candidates].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = subset["utilization_score"].values
    valid = np.isfinite(y)
    if valid.sum() < 100:
        return None

    oof = np.full(len(y), np.nan)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X):
        if not valid[train_idx].any():
            continue
        t_mask = valid[train_idx]
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
        )
        model.fit(X[train_idx][t_mask], y[train_idx][t_mask])
        oof[val_idx] = model.predict(X[val_idx])

    # Fill early folds that have no OOF predictions with actual values
    # (these rows are only used in training, not serving)
    nan_mask = np.isnan(oof)
    oof[nan_mask] = y[nan_mask]
    return oof


def train_utilization_to_fp_per_position(
    train_data: pd.DataFrame, positions: Optional[List[str]] = None
) -> Dict[str, UtilizationToFPConverter]:
    """Train conversion model for requested positions (default RB/WR/TE).

    Uses OOF-predicted utilization scores to train the converter, reducing
    the train/serve distribution mismatch from cascaded prediction error.
    """
    converters = {}
    for pos in (positions or ["RB", "WR", "TE"]):
        subset = train_data[train_data["position"] == pos].copy()
        if "utilization_score" not in subset.columns or "fantasy_points" not in subset.columns:
            continue
        # Generate OOF utilization predictions for this position
        oof_util = _generate_oof_utilization(subset)
        if oof_util is not None:
            print(f"  {pos}: Training converter on OOF-predicted utilization (reduces cascaded error)")
        conv = UtilizationToFPConverter(pos)
        conv.fit(subset, target_col="fantasy_points", oof_utilization=oof_util)
        if conv.is_fitted:
            conv.save()
            converters[pos] = conv
    return converters

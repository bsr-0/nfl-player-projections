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

    def fit(self, df: pd.DataFrame, target_col: str = "fantasy_points") -> "UtilizationToFPConverter":
        if not HAS_SKLEARN:
            self.is_fitted = False
            return self
        cols = self._get_input_cols(df)
        if self.util_col not in df.columns or target_col not in df.columns:
            self.is_fitted = False
            return self
        work = df.copy()
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
        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.model.fit(Xs, y_train)
        if HAS_XGB:
            try:
                self.xgb_model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )
                self.xgb_model.fit(Xs, y_train)
            except Exception:
                self.xgb_model = None
        self.calibration = None
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
        return c


def train_utilization_to_fp_per_position(
    train_data: pd.DataFrame, positions: Optional[List[str]] = None
) -> Dict[str, UtilizationToFPConverter]:
    """Train conversion model for requested positions (default RB/WR/TE)."""
    converters = {}
    for pos in (positions or ["RB", "WR", "TE"]):
        subset = train_data[train_data["position"] == pos]
        if "utilization_score" not in subset.columns or "fantasy_points" not in subset.columns:
            continue
        conv = UtilizationToFPConverter(pos)
        conv.fit(subset, target_col="fantasy_points")
        if conv.is_fitted:
            conv.save()
            converters[pos] = conv
    return converters

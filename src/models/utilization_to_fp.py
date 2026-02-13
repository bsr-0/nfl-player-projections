"""
Conversion layer: Utilization Score -> Fantasy Points for RB/WR/TE.

Per requirements: secondary model that converts predicted utilization to fantasy
points using efficiency metrics (yards per touch, TD rate given opportunities).
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


# Efficiency features used for utilization -> FP conversion (per position)
EFFICIENCY_FEATURES = {
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
        self.scaler = None
        self.feature_names = []
        self.util_col = "utilization_score"
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
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        y = df[target_col].values
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < 30:
            self.is_fitted = False
            return self
        X, y = X[valid], y[valid]
        self.feature_names = cols
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

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
        return self.model.predict(Xs)

    def save(self, path: Path = None):
        path = path or MODELS_DIR / f"util_to_fp_{self.position.lower()}.joblib"
        joblib.dump({
            "position": self.position,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, position: str, path: Path = None) -> "UtilizationToFPConverter":
        path = path or MODELS_DIR / f"util_to_fp_{position.lower()}.joblib"
        c = cls(position)
        if path.exists():
            d = joblib.load(path)
            c.model = d.get("model")
            c.scaler = d.get("scaler")
            c.feature_names = d.get("feature_names", [])
            c.is_fitted = d.get("is_fitted", False)
        return c


def train_utilization_to_fp_per_position(train_data: pd.DataFrame) -> Dict[str, UtilizationToFPConverter]:
    """Train conversion model for each non-QB position."""
    converters = {}
    for pos in ["RB", "WR", "TE"]:
        subset = train_data[train_data["position"] == pos]
        if "utilization_score" not in subset.columns or "fantasy_points" not in subset.columns:
            continue
        conv = UtilizationToFPConverter(pos)
        conv.fit(subset, target_col="fantasy_points")
        if conv.is_fitted:
            conv.save()
            converters[pos] = conv
    return converters

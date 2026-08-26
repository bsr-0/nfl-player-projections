"""Component-based fantasy point prediction.

Council Phase 2 recommendation: predict stable stat components (yards,
receptions, TDs) separately, then assemble fantasy points from those
predictions.  Each component has higher autocorrelation and lower touchdown
contamination than raw fantasy points.

The key insight: a WR catching 6 passes for 70 yards scores ~10 PPR points.
Add one TD and it's 16.  That TD is nearly coin-flip predictable at the
individual game level.  By predicting yards and receptions (stable) separately
from TDs (volatile), we get a better overall prediction even though each
individual component model is simple.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ComponentPredictor:
    """Predict individual stat components per position, assemble fantasy points.

    For each position, trains a separate Ridge model for each stat component
    (e.g., rushing_yards, rushing_tds for RB).  At prediction time, predicts
    each component and multiplies by PPR scoring weights to get total fantasy
    points.

    Ridge was chosen over GradientBoosting after backtest evidence showed GBT
    produced negative R² for WR/TE/RB (worse than mean prediction), primarily
    due to systematic underprediction. Ridge generalizes better with the
    available training data sizes.
    """

    def __init__(self, position: str):
        self.position = position
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted = False

        from config.settings import COMPONENT_TARGETS, PPR_SCORING_WEIGHTS
        self.components = COMPONENT_TARGETS.get(position, [])
        self.scoring_weights = PPR_SCORING_WEIGHTS

    def _median_vector(self) -> Optional[np.ndarray]:
        """Train-fitted medians aligned to `feature_names`, or None if unfitted."""
        if not self.feature_names or not self.feature_medians:
            return None
        return np.array([self.feature_medians.get(c, 0.0) for c in self.feature_names],
                        dtype=np.float64)

    def _prepare_array(self, X_arr: np.ndarray,
                       medians: Optional[np.ndarray] = None) -> np.ndarray:
        """Clean feature array: inf -> NaN, then impute NaN.

        Ridge and StandardScaler cannot take NaN, so something has to fill it.
        This used to be a flat 0.0, which is a fabricated observation in raw
        units -- a 0.0 snap share or 0.0 separation is a confident wrong claim,
        and it lands BEFORE scaling, so it sits far from the neutral point
        rather than at it. Imputing the train-fitted median puts the value at
        roughly the scaled mean, which is the neutral choice for a linear model.

        Medians come from `fit` (train rows only) and are persisted in
        to_dict/from_dict, so predict-time imputation cannot see test data.
        Falls back to 0.0 only when called before fit.
        """
        X_arr = np.where(np.isfinite(X_arr), X_arr, np.nan)
        if medians is None:
            return np.nan_to_num(X_arr, nan=0.0)
        bad = np.isnan(X_arr)
        if bad.any():
            X_arr = X_arr.copy()
            X_arr[bad] = np.take(medians, np.where(bad)[1])
        return X_arr

    def _fit_component_models(
        self,
        X_arr: np.ndarray,
        y_components: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> tuple[Dict[str, Any], Dict[str, StandardScaler]]:
        """Train one Ridge model per component and return model/scaler dicts."""
        models: Dict[str, Any] = {}
        scalers: Dict[str, StandardScaler] = {}

        for component in self.components:
            if component not in y_components:
                continue

            y = np.asarray(y_components[component], dtype=np.float64)
            valid = np.isfinite(y) & np.isfinite(X_arr).all(axis=1)
            if valid.sum() < 30:
                logger.warning("Component %s has < 30 valid rows for %s, skipping",
                               component, self.position)
                continue

            X_v = X_arr[valid]
            y_v = y[valid]
            sw = sample_weight[valid] if sample_weight is not None else None

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_v)
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_v, sample_weight=sw)

            models[component] = model
            scalers[component] = scaler

        return models, scalers

    def _predict_total_fp(
        self,
        X_arr: np.ndarray,
        models: Optional[Dict[str, Any]] = None,
        scalers: Optional[Dict[str, StandardScaler]] = None,
    ) -> np.ndarray:
        models = models if models is not None else self.models
        scalers = scalers if scalers is not None else self.scalers
        total_fp = np.zeros(X_arr.shape[0], dtype=np.float64)

        for component, model in models.items():
            scaler = scalers.get(component)
            X_input = scaler.transform(X_arr) if scaler is not None else X_arr
            pred = model.predict(X_input)

            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)

            weight = self.scoring_weights.get(component, 0.0)
            total_fp += pred * weight

        return np.maximum(total_fp, 0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y_components: Dict[str, pd.Series],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ComponentPredictor":
        """Train one Ridge model per stat component."""
        self.feature_names = list(X.columns)
        self.feature_medians = {c: float(X[c].median()) for c in X.columns
                                if pd.api.types.is_numeric_dtype(X[c])}

        X_arr = X.values.astype(np.float64)
        X_arr = self._prepare_array(X_arr, self._median_vector())
        y_np = {k: v.values.astype(np.float64) for k, v in y_components.items()}
        self.models, self.scalers = self._fit_component_models(X_arr, y_np, sample_weight=sample_weight)

        self.is_fitted = len(self.models) > 0
        if self.is_fitted:
            logger.info("ComponentPredictor[%s]: trained %d/%d components",
                        self.position, len(self.models), len(self.components))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict fantasy points by predicting each component and assembling.

        Returns:
            Array of predicted fantasy points (same length as X).
        """
        if not self.is_fitted:
            raise ValueError(f"ComponentPredictor[{self.position}] not fitted")

        # Align features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        for col in self.feature_names:
            if col not in X_aligned.columns:
                X_aligned[col] = self.feature_medians.get(col, 0)
        X_arr = X_aligned.values.astype(np.float64)
        X_arr = self._prepare_array(X_arr, self._median_vector())
        return self._predict_total_fp(X_arr)

    def predict_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict individual stat components (for inspection/debugging)."""
        if not self.is_fitted:
            raise ValueError(f"ComponentPredictor[{self.position}] not fitted")

        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        X_arr = X_aligned.values.astype(np.float64)
        X_arr = self._prepare_array(X_arr, self._median_vector())

        result = {}
        for component, model in self.models.items():
            scaler = self.scalers.get(component)
            X_input = scaler.transform(X_arr) if scaler is not None else X_arr
            pred = model.predict(X_input)
            if component not in ("interceptions", "fumbles_lost"):
                pred = np.maximum(pred, 0.0)
            result[component] = pred
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for saving. Uses joblib for model-agnostic serialization."""
        models_b64 = {}
        for k, m in self.models.items():
            buf = io.BytesIO()
            joblib.dump(m, buf)
            models_b64[k] = base64.b64encode(buf.getvalue()).decode("ascii")

        scalers_b64 = {}
        for k, s in self.scalers.items():
            buf = io.BytesIO()
            joblib.dump(s, buf)
            scalers_b64[k] = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "position": self.position,
            "components": self.components,
            "feature_names": self.feature_names,
            "feature_medians": self.feature_medians,
            "model_type": "GradientBoostingRegressor",
            "serialization": "joblib_b64",
            "models": models_b64,
            "scalers": scalers_b64,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentPredictor":
        """Deserialize from saved dict."""
        cp = cls(data["position"])
        cp.feature_names = data.get("feature_names", [])
        cp.feature_medians = data.get("feature_medians", {})
        cp.components = data.get("components", cp.components)

        serialization = data.get("serialization", "ridge_json")

        if serialization == "joblib_b64":
            for comp_name, b64_str in data.get("models", {}).items():
                buf = io.BytesIO(base64.b64decode(b64_str))
                cp.models[comp_name] = joblib.load(buf)
            for comp_name, b64_str in data.get("scalers", {}).items():
                buf = io.BytesIO(base64.b64decode(b64_str))
                cp.scalers[comp_name] = joblib.load(buf)
        else:
            # Legacy Ridge JSON format
            from sklearn.linear_model import Ridge
            for comp_name, model_data in data.get("models", {}).items():
                model = Ridge(alpha=1.0)
                model.coef_ = np.array(model_data["coef"])
                model.intercept_ = model_data["intercept"]
                model.n_features_in_ = len(model.coef_)
                cp.models[comp_name] = model
            for comp_name, scaler_data in data.get("scalers", {}).items():
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_data["mean"])
                scaler.scale_ = np.array(scaler_data["scale"])
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = len(scaler.mean_)
                cp.scalers[comp_name] = scaler

        cp.is_fitted = len(cp.models) > 0
        return cp

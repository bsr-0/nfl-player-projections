"""Model architectures for Phase 2 (next_focus.md) single-week PPR comparison.

Six architectures (A-F) plus four naive baselines, all targeting raw
`fantasy_points` — no winsorization, no clipping (next_focus.md Phase 2 §5).

A/B/C are the same GBMRegressor wrapper with a different loss objective.
D/E/F are genuinely new (no prior hurdle/quantile-model/Yeo-Johnson code
existed in this repo — see GAPS.md Phase 2 notes).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

from src.evaluation.baselines import trailing_average_baseline

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - environment-dependent
    HAS_LIGHTGBM = False

DEFAULT_GBM_PARAMS = dict(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)

# sklearn GradientBoostingRegressor fallback loss names, keyed by our
# objective vocabulary (LightGBM naming).
_SKLEARN_LOSS_MAP = {
    "regression": "squared_error",
    "huber": "huber",
    "regression_l1": "absolute_error",
}


# ---------------------------------------------------------------------------
# Naive baselines (next_focus.md Phase 2 §3)
# ---------------------------------------------------------------------------

def position_role_baseline(df: pd.DataFrame, target_col: str = "fantasy_points") -> pd.Series:
    """Expanding position-average, shifted to avoid current-week leakage.

    Mirrors the expanding-mean pattern already used for defense rankings
    (src/data/external_data.py DefenseRankingsLoader.calculate_defense_rankings,
    lines ~187-194).
    """
    return (
        df.sort_values(["season", "week"])
        .groupby("position")[target_col]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )


def naive_baselines(combined_df: pd.DataFrame, target_col: str = "fantasy_points") -> Dict[str, pd.Series]:
    """Four naive baselines, computed over `combined_df` (train tail + test,
    sorted by season/week) so trailing/expanding windows see prior-season
    history at the test-season boundary. Callers should slice the returned
    Series down to test-row indices before scoring.
    """
    return {
        "baseline_recent_game": trailing_average_baseline(combined_df, n_weeks=1, target_col=target_col),
        "baseline_rolling3": trailing_average_baseline(combined_df, n_weeks=3, target_col=target_col),
        "baseline_rolling5": trailing_average_baseline(combined_df, n_weeks=5, target_col=target_col),
        "baseline_position_role": position_role_baseline(combined_df, target_col=target_col),
    }


# ---------------------------------------------------------------------------
# A / B / C: Gradient Boosting, MSE / Huber / MAE loss
# ---------------------------------------------------------------------------

class GBMRegressor:
    """Thin LightGBM wrapper (sklearn GradientBoostingRegressor fallback).

    objective: 'regression' (MSE, architecture A), 'huber' (architecture B,
    primary candidate per next_focus.md), or 'regression_l1' (MAE, architecture C).
    """

    def __init__(self, objective: str = "regression", **params):
        if objective not in _SKLEARN_LOSS_MAP:
            raise ValueError(f"Unknown objective: {objective!r}")
        self.objective = objective
        self.params = {**DEFAULT_GBM_PARAMS, **params}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "GBMRegressor":
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(objective=self.objective, verbosity=-1, **self.params)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                loss=_SKLEARN_LOSS_MAP[self.objective], **self.params
            )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# D: Two-stage / hurdle model
# ---------------------------------------------------------------------------

class HurdleModel:
    """P(PPR > threshold) x E[PPR | PPR > threshold].

    threshold=0 is the primary spec; next_focus.md also asks to test
    threshold=5 "if useful" — pass threshold=5 to get that variant.
    """

    def __init__(self, threshold: float = 0.0, **params):
        self.threshold = threshold
        self.params = {**DEFAULT_GBM_PARAMS, **params}
        self.classifier = None
        self.regressor = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "HurdleModel":
        positive = y > self.threshold
        if HAS_LIGHTGBM:
            self.classifier = lgb.LGBMClassifier(objective="binary", verbosity=-1, **self.params)
            self.regressor = lgb.LGBMRegressor(objective="regression", verbosity=-1, **self.params)
        else:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            self.classifier = GradientBoostingClassifier(**self.params)
            self.regressor = GradientBoostingRegressor(loss="squared_error", **self.params)

        self.classifier.fit(X, positive.astype(int), sample_weight=sample_weight)
        if positive.sum() < 2:
            raise ValueError("HurdleModel needs at least 2 positive-target rows to fit stage 2")
        positive_weight = None
        if sample_weight is not None:
            positive_weight = np.asarray(sample_weight)[np.asarray(positive)]
        self.regressor.fit(X[positive], y[positive], sample_weight=positive_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p_positive = self.classifier.predict_proba(X)[:, 1]
        e_given_positive = self.regressor.predict(X)
        return p_positive * e_given_positive


# ---------------------------------------------------------------------------
# E: Quantile models
# ---------------------------------------------------------------------------

class QuantileGBM:
    """Independent quantile regressors, mirroring UncertaintyQuantifier
    (src/models/advanced_ml_pipeline.py:505-538) but as a standalone
    prediction architecture rather than post-hoc uncertainty calibration.
    """

    QUANTILES = (0.25, 0.5, 0.75, 0.9)

    def __init__(self, **params):
        self.params = {**DEFAULT_GBM_PARAMS, **params}
        self.models: Dict[float, object] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "QuantileGBM":
        for q in self.QUANTILES:
            if HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(objective="quantile", alpha=q, verbosity=-1, **self.params)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(loss="quantile", alpha=q, **self.params)
            model.fit(X, y, sample_weight=sample_weight)
            self.models[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame with columns p25/p50/p75/p90."""
        return pd.DataFrame(
            {f"p{int(q * 100)}": self.models[q].predict(X) for q in self.QUANTILES},
            index=X.index,
        )

    def predict_median(self, X: pd.DataFrame) -> np.ndarray:
        """Point-prediction convenience accessor (p50) for uniform scoring."""
        return self.models[0.5].predict(X)


# ---------------------------------------------------------------------------
# F: Target transformation challenger
# ---------------------------------------------------------------------------

class YeoJohnsonHuber:
    """Yeo-Johnson-transformed target + Huber GBM, inverse-transformed at
    predict time. Yeo-Johnson (not log/Box-Cox) because PPR can be negative
    (fumbles/INTs can make single-game PPR go below zero).
    """

    def __init__(self, **params):
        self.transformer = PowerTransformer(method="yeo-johnson")
        self.model = GBMRegressor(objective="huber", **params)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> "YeoJohnsonHuber":
        y_t = self.transformer.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()
        self.model.fit(X, pd.Series(y_t, index=y.index), sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_t_pred = self.model.predict(X)
        return self.transformer.inverse_transform(y_t_pred.reshape(-1, 1)).ravel()

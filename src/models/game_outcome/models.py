"""Phase 1 models: LogisticRegression baseline, XGBoost tree-based arm.
Phase 2 models: Ridge baseline, XGBoost tree-based arm, for margin/total
regression -- same pipeline shape, different sklearn estimator.
Third arm (both phases): RandomForest -- a second tree family with a
different bias/variance tradeoff than boosting, wrapped with the same
median-imputation pipeline as the linear arms since (unlike XGBoost)
sklearn's RandomForest doesn't handle NaN natively.

All expose `fit(X, y)` / `predict_proba(X)` (classifiers) or `fit(X, y)` /
`predict(X)` (regressors) so the backtesters (src/evaluation/
game_outcome_backtester.py, game_margin_backtester.py) and the training
scripts treat every arm -- these models plus the baselines in baseline.py --
identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from config.settings import GAME_OUTCOME_MODEL_CONFIG


class GameOutcomeLogisticModel:
    """StandardScaler + LogisticRegression, fixed C (no inner-CV tuning in phase 1).

    Median-imputes missing feature values (cold-start rows with no prior-
    season fallback, or weather rows not yet backfilled) INSIDE the pipeline,
    so the imputer's medians are fit only on that call's training data --
    never on validation/test rows, which would otherwise be a subtle leak.
    """

    def __init__(self, C: Optional[float] = None) -> None:
        c = C if C is not None else GAME_OUTCOME_MODEL_CONFIG["logistic_C"]
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=c),
        )

    def fit(self, X: pd.DataFrame, y) -> "GameOutcomeLogisticModel":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class GameOutcomeXGBModel:
    """XGBClassifier with fixed, conservative hyperparameters (no Optuna in phase 1).

    Shallow depth (3) and min_child_weight=5 are deliberate: a game-outcome
    panel is a few thousand rows total (~285 games/season x N seasons), well
    below where XGBoost needs more capacity -- biasing toward under- rather
    than over-fitting. XGBoost handles NaN natively (missing feature values
    from cold-start/weather gaps need no imputer here, unlike the logistic arm).
    """

    def __init__(self, **overrides) -> None:
        params = dict(GAME_OUTCOME_MODEL_CONFIG["xgb_params"])
        params.update(overrides)
        self.model = XGBClassifier(**params)

    def fit(self, X: pd.DataFrame, y) -> "GameOutcomeXGBModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class GameMarginRidgeModel:
    """StandardScaler + Ridge, fixed alpha (no inner-CV tuning in phase 2).

    Used for both the margin and total targets -- same class, different `y`
    at fit time. Same median-imputation-inside-the-pipeline discipline as
    `GameOutcomeLogisticModel`.
    """

    def __init__(self, alpha: Optional[float] = None) -> None:
        a = alpha if alpha is not None else GAME_OUTCOME_MODEL_CONFIG["ridge_alpha"]
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=a),
        )

    def fit(self, X: pd.DataFrame, y) -> "GameMarginRidgeModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class GameRegressionXGBModel:
    """XGBRegressor with fixed, conservative hyperparameters (no Optuna in phase 2).

    Same shallow-depth reasoning as `GameOutcomeXGBModel`. Used for both the
    margin and total targets.
    """

    def __init__(self, **overrides) -> None:
        params = dict(GAME_OUTCOME_MODEL_CONFIG["xgb_regressor_params"])
        params.update(overrides)
        self.model = XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y) -> "GameRegressionXGBModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class GameOutcomeRFModel:
    """RandomForestClassifier, fixed conservative hyperparameters (no Optuna).

    No StandardScaler needed (trees are scale-invariant) -- just the median
    imputer, since sklearn's RandomForest can't handle NaN natively.
    """

    def __init__(self, **overrides) -> None:
        params = dict(GAME_OUTCOME_MODEL_CONFIG["rf_params"])
        params.update(overrides)
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(**params),
        )

    def fit(self, X: pd.DataFrame, y) -> "GameOutcomeRFModel":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class GameRegressionRFModel:
    """RandomForestRegressor, fixed conservative hyperparameters (no Optuna).
    Used for both the margin and total targets."""

    def __init__(self, **overrides) -> None:
        params = dict(GAME_OUTCOME_MODEL_CONFIG["rf_regressor_params"])
        params.update(overrides)
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(**params),
        )

    def fit(self, X: pd.DataFrame, y) -> "GameRegressionRFModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)

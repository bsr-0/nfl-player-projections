"""Share regressors for Plan A.

Same median-impute-inside-the-pipeline discipline as
src/models/game_outcome/models.py (the imputer's medians are fit only on
that call's training data, never on validation/test rows), applied to a
[0,1]-bounded share target instead of margin/total. No inner-CV
hyperparameter tuning in this first cut -- fixed, conservative defaults from
config.settings.TEAM_ALLOCATION_MODEL_CONFIG, same philosophy as
game_outcome's phase-1 fixed hyperparameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config.settings import TEAM_ALLOCATION_MODEL_CONFIG


class ShareRidgeModel:
    def __init__(self, alpha: Optional[float] = None) -> None:
        a = alpha if alpha is not None else TEAM_ALLOCATION_MODEL_CONFIG["ridge_alpha"]
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=a),
        )

    def fit(self, X: pd.DataFrame, y) -> "ShareRidgeModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


class ShareXGBModel:
    """XGBoost handles NaN natively -- no imputer needed, unlike the ridge arm."""

    def __init__(self, **overrides) -> None:
        params = dict(TEAM_ALLOCATION_MODEL_CONFIG["xgb_params"])
        params.update(overrides)
        self.model = XGBRegressor(**params)

    def fit(self, X: pd.DataFrame, y) -> "ShareXGBModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)

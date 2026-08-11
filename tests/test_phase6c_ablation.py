"""Unit tests for Phase 6c (next_focus.md follow-up) feature-count ablation.

Synthetic data only — no DB/network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.ablation import (
    rank_features_by_importance,
    top_n_features,
    _get_lgbm_model,
)
from src.models.single_week_ppr.architectures import GBMRegressor, YeoJohnsonHuber


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame({
        "strong_signal": rng.uniform(0, 10, n),
        "weak_signal": rng.uniform(0, 10, n),
        "pure_noise": rng.uniform(0, 10, n),
    })
    y = pd.Series(5 + 3 * X["strong_signal"] + 0.1 * X["weak_signal"] + rng.normal(0, 1, n))
    return X, y


class TestRankFeaturesByImportance:
    def test_gbm_regressor_ranks_strong_signal_first(self, synthetic_data):
        X, y = synthetic_data
        model = GBMRegressor(objective="regression_l1", n_estimators=50)
        model.fit(X, y)
        ranked = rank_features_by_importance(model, list(X.columns))
        assert ranked[0] == "strong_signal"
        assert set(ranked) == set(X.columns)

    def test_yeojohnson_huber_unwraps_correctly(self, synthetic_data):
        X, y = synthetic_data
        model = YeoJohnsonHuber(n_estimators=50)
        model.fit(X, y)
        ranked = rank_features_by_importance(model, list(X.columns))
        assert ranked[0] == "strong_signal"

    def test_unfitted_model_raises(self):
        model = GBMRegressor(objective="regression_l1")
        with pytest.raises(ValueError):
            _get_lgbm_model(model)


class TestTopNFeatures:
    def test_top_n_slices_correctly(self):
        ranked = ["a", "b", "c", "d", "e"]
        assert top_n_features(ranked, 3) == ["a", "b", "c"]

    def test_all_returns_everything(self):
        ranked = ["a", "b", "c"]
        assert top_n_features(ranked, "all") == ["a", "b", "c"]

    def test_n_larger_than_available_degrades_gracefully(self):
        ranked = ["a", "b"]
        assert top_n_features(ranked, 10) == ["a", "b"]

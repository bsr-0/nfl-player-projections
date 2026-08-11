"""Unit tests for Phase 5 (next_focus.md) nested-CV hyperparameter tuning.

Synthetic data only — no DB/network. Uses a small n_trials for the real
Optuna run so this stays fast.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.tuning import (
    inner_walk_forward_folds,
    tune_hyperparameters_for_fold,
    _build_model,
)


class TestInnerWalkForwardFolds:
    def test_expanding_window_with_purge_gap(self):
        # cv_gap_seasons defaults to 1 in config/settings.py MODEL_CONFIG.
        folds = inner_walk_forward_folds(list(range(2018, 2023)), n_folds=3)
        # Validation seasons should be the last 3: 2020, 2021, 2022.
        val_seasons = [v for _, v in folds]
        assert val_seasons == [2020, 2021, 2022]
        # Each inner_train must stop at least `gap` seasons before val.
        for inner_train, val in folds:
            assert max(inner_train) <= val - 2  # gap=1 means val-1-1

    def test_never_touches_seasons_after_val(self):
        folds = inner_walk_forward_folds([2018, 2019, 2020, 2021, 2022], n_folds=3)
        for inner_train, val in folds:
            assert all(s < val for s in inner_train)

    def test_degrades_gracefully_with_short_history(self):
        folds = inner_walk_forward_folds([2022, 2023], n_folds=3)
        assert isinstance(folds, list)  # doesn't error, may be short or empty

    def test_empty_history_returns_empty(self):
        assert inner_walk_forward_folds([], n_folds=3) == []


class TestTuneHyperparametersForFold:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(11)
        n = 400
        seasons = np.repeat(np.arange(2018, 2023), n // 5)
        X = pd.DataFrame({
            "targets_roll3_mean": rng.uniform(0, 10, n),
            "snap_share_pct_roll3_mean": rng.uniform(0, 1, n),
        })
        y = pd.Series(5 + 2 * X["targets_roll3_mean"] + rng.normal(0, 1, n))
        return X, y, seasons

    def test_returns_valid_params_for_c_gbm_mae(self, synthetic_data):
        X, y, seasons = synthetic_data
        best_params = tune_hyperparameters_for_fold(
            "C_gbm_mae", X, y, seasons, n_trials=5, n_inner_folds=2,
        )
        assert isinstance(best_params, dict)
        # A fresh model built with these params must fit without error.
        model = _build_model("C_gbm_mae", best_params)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.isfinite(preds).all()

    def test_returns_valid_params_for_f_yeojohnson_huber(self, synthetic_data):
        X, y, seasons = synthetic_data
        best_params = tune_hyperparameters_for_fold(
            "F_yeojohnson_huber", X, y, seasons, n_trials=5, n_inner_folds=2,
        )
        model = _build_model("F_yeojohnson_huber", best_params)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.isfinite(preds).all()

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            _build_model("Z_not_real", {})

    def test_insufficient_history_returns_empty_dict(self, synthetic_data):
        X, y, seasons = synthetic_data
        single_season = np.full(len(seasons), 2022)
        best_params = tune_hyperparameters_for_fold(
            "C_gbm_mae", X, y, single_season, n_trials=3, n_inner_folds=3,
        )
        assert best_params == {}

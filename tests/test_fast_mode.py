"""Tests for fast mode training — exercises code paths that caused segfaults.

Key regression tests:
- compute_vif() with zero-variance features (LAPACK segfault via NaN corr matrix)
- LightGBM tuning without cross_val_score (sklearn MRO segfault)
- DeepSeasonLongModel save/load (missing regression_to_mean_scale attribute)
- End-to-end PositionModel.fit() with fast config
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_CONFIG, FAST_MODEL_CONFIG


# ---------------------------------------------------------------------------
# VIF computation tests (primary segfault fix)
# ---------------------------------------------------------------------------

class TestComputeVifSafety:
    """Verify compute_vif handles degenerate inputs without crashing."""

    def test_zero_variance_features_no_crash(self):
        """Zero-variance columns must not cause LAPACK segfault."""
        from src.features.dimensionality_reduction import compute_vif

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "const_col": [5.0] * 200,           # zero variance
            "normal_a": rng.randn(200),
            "normal_b": rng.randn(200) * 2 + 1,
            "normal_c": rng.randn(200) * 0.5,
        })
        result = compute_vif(df)

        assert isinstance(result, dict)
        assert set(result.keys()) == set(df.columns)
        # Zero-variance feature should get inf VIF (will be pruned)
        assert result["const_col"] == np.inf
        # Normal features should have finite VIF >= 1
        for col in ["normal_a", "normal_b", "normal_c"]:
            assert np.isfinite(result[col])
            assert result[col] >= 1.0

    def test_all_zero_variance_no_crash(self):
        """All-constant columns should return without crashing."""
        from src.features.dimensionality_reduction import compute_vif

        df = pd.DataFrame({
            "a": [1.0] * 50,
            "b": [2.0] * 50,
            "c": [3.0] * 50,
        })
        result = compute_vif(df)
        assert isinstance(result, dict)
        assert len(result) == 3

    def test_nan_filled_features_no_crash(self):
        """Features that become constant after fillna(0) must not crash."""
        from src.features.dimensionality_reduction import compute_vif

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "all_nan": [np.nan] * 100,          # becomes all-zero after fillna
            "mostly_nan": [np.nan] * 99 + [1.0], # near-zero variance
            "normal": rng.randn(100),
        })
        result = compute_vif(df)
        assert isinstance(result, dict)
        assert len(result) == 3
        # all_nan becomes zero-variance after fillna → should be inf
        assert result["all_nan"] == np.inf

    def test_single_feature_no_crash(self):
        """Single-feature DataFrame should not crash."""
        from src.features.dimensionality_reduction import compute_vif

        df = pd.DataFrame({"only": np.random.randn(50)})
        result = compute_vif(df)
        assert isinstance(result, dict)
        assert len(result) == 1

    def test_two_identical_features(self):
        """Perfectly correlated features should get high VIF."""
        from src.features.dimensionality_reduction import compute_vif

        vals = np.random.randn(100)
        df = pd.DataFrame({"a": vals, "b": vals, "c": np.random.randn(100)})
        result = compute_vif(df)
        assert isinstance(result, dict)
        # a and b are perfectly correlated → very high or inf VIF
        assert result["a"] > 100 or result["a"] == np.inf
        assert result["b"] > 100 or result["b"] == np.inf


class TestPruneByVif:
    """Verify prune_by_vif handles degenerate inputs."""

    def test_prune_removes_zero_variance(self):
        """Zero-variance features should be pruned (VIF=inf > any threshold)."""
        from src.features.dimensionality_reduction import prune_by_vif

        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "const": [7.0] * 100,
            "a": rng.randn(100),
            "b": rng.randn(100),
        })
        result_df, removed = prune_by_vif(df, threshold=10.0)
        assert "const" in removed
        assert "const" not in result_df.columns


# ---------------------------------------------------------------------------
# LightGBM tuning test (secondary segfault fix)
# ---------------------------------------------------------------------------

class TestLightGBMTuningManualCV:
    """Verify LightGBM tuning uses manual CV instead of cross_val_score."""

    def test_source_does_not_use_cross_val_score(self):
        """_tune_lightgbm should not call cross_val_score (sklearn MRO segfault)."""
        import inspect
        from src.models.position_models import PositionModel

        source = inspect.getsource(PositionModel._tune_lightgbm)
        assert "cross_val_score" not in source, (
            "_tune_lightgbm still uses cross_val_score — this causes segfaults "
            "with sklearn>=1.6 via the __sklearn_tags__ MRO path"
        )

    def test_lightgbm_tuning_runs_without_crash(self):
        """LightGBM tuning with manual CV should complete without crash."""
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            pytest.skip("LightGBM not installed")

        from src.models.position_models import PositionModel

        rng = np.random.RandomState(42)
        n = 300
        X = rng.randn(n, 10)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + rng.randn(n) * 0.3
        seasons = np.repeat(np.arange(2018, 2024), n // 6)[:n]

        model = PositionModel("RB", n_weeks=1)
        # Use minimal trials for speed
        params = model._tune_lightgbm(X, y, n_trials=3, seasons=seasons)
        assert isinstance(params, dict)
        assert "n_estimators" in params


# ---------------------------------------------------------------------------
# DeepSeasonLongModel attribute test
# ---------------------------------------------------------------------------

class TestDeepModelAttributes:
    """Verify DeepSeasonLongModel has all required attributes."""

    def test_regression_to_mean_scale_initialized(self):
        """regression_to_mean_scale must be set in __init__ so save() works."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from src.models.horizon_models import DeepSeasonLongModel

        model = DeepSeasonLongModel("RB")
        assert hasattr(model, "regression_to_mean_scale")
        assert model.regression_to_mean_scale == 0.95

    def test_deep_model_save_load_roundtrip(self, tmp_path):
        """DeepSeasonLongModel should save and load without errors."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from src.models.horizon_models import DeepSeasonLongModel

        rng = np.random.RandomState(42)
        n, p = 200, 20
        X = rng.randn(n, p).astype(np.float32)
        y = (X[:, 0] * 3 + rng.randn(n) * 0.5).astype(np.float32)
        feature_names = [f"feat_{i}" for i in range(p)]

        model = DeepSeasonLongModel("RB", n_features=p)
        model.fit(X, y, feature_names=feature_names, epochs=5, batch_size=32)
        assert model.is_fitted

        save_path = tmp_path / "deep_18w_rb"
        model.save(path=save_path)
        assert (save_path / "config.joblib").exists()
        assert (save_path / "model.pt").exists()

        loaded = DeepSeasonLongModel.load("RB", path=save_path)
        assert loaded.is_fitted
        assert loaded.regression_to_mean_scale == 0.95


# ---------------------------------------------------------------------------
# End-to-end PositionModel.fit() with fast config
# ---------------------------------------------------------------------------

class TestFastModePositionModelFit:
    """End-to-end test: PositionModel.fit() with FAST_MODEL_CONFIG settings."""

    @pytest.fixture
    def fast_config(self):
        """Apply fast mode config and restore after test."""
        original = {k: MODEL_CONFIG.get(k) for k in FAST_MODEL_CONFIG}
        for k, v in FAST_MODEL_CONFIG.items():
            MODEL_CONFIG[k] = v
        yield
        for k, v in original.items():
            if v is None:
                MODEL_CONFIG.pop(k, None)
            else:
                MODEL_CONFIG[k] = v

    def test_position_model_fit_fast_mode(self, fast_config):
        """PositionModel should train successfully with fast config."""
        from src.models.position_models import PositionModel

        rng = np.random.RandomState(42)
        n_seasons = 6
        n_per_season = 100
        n = n_seasons * n_per_season

        # Create synthetic features including a zero-variance column
        feature_data = {
            f"feat_{i}": rng.randn(n) for i in range(15)
        }
        feature_data["const_feat"] = [1.0] * n  # zero variance

        X = pd.DataFrame(feature_data)
        y = pd.Series(X["feat_0"] * 2 + X["feat_1"] + rng.randn(n) * 0.5,
                       name="target")
        seasons = np.repeat(np.arange(2018, 2018 + n_seasons), n_per_season)

        model = PositionModel("RB", n_weeks=1)
        # Use minimal Optuna trials for speed
        model.fit(X, y, tune_hyperparameters=True, n_trials=3, seasons=seasons)

        assert model.is_fitted
        preds = model.predict(X.iloc[:10])
        assert len(preds) == 10
        assert np.all(np.isfinite(preds))

    def test_position_model_fit_with_many_zero_var_features(self, fast_config):
        """Model should handle datasets where many features are constant."""
        from src.models.position_models import PositionModel

        rng = np.random.RandomState(42)
        n = 500
        feature_data = {}
        # 5 useful features
        for i in range(5):
            feature_data[f"good_{i}"] = rng.randn(n)
        # 10 zero-variance features (e.g. missing data filled to 0)
        for i in range(10):
            feature_data[f"dead_{i}"] = [0.0] * n

        X = pd.DataFrame(feature_data)
        y = pd.Series(
            feature_data["good_0"] * 3 + feature_data["good_1"] + rng.randn(n) * 0.2,
            name="target",
        )
        seasons = np.repeat(np.arange(2018, 2023), n // 5)[:n]

        model = PositionModel("WR", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False, seasons=seasons)

        assert model.is_fitted
        preds = model.predict(X.iloc[:5])
        assert len(preds) == 5
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Random Forest tuning consistency
# ---------------------------------------------------------------------------

class TestRandomForestTuningManualCV:
    """Verify Random Forest tuning uses manual CV for consistency."""

    def test_source_does_not_use_cross_val_score(self):
        """_tune_random_forest should not call cross_val_score."""
        import inspect
        from src.models.position_models import PositionModel

        source = inspect.getsource(PositionModel._tune_random_forest)
        assert "cross_val_score" not in source


# ---------------------------------------------------------------------------
# SeasonAwareTimeSeriesSplit minimum fold size guard
# ---------------------------------------------------------------------------

class TestSeasonAwareTimeSeriesSplitSafety:
    """Verify CV splitter enforces minimum fold sizes."""

    def test_minimum_train_size_enforcement(self):
        """Folds with fewer than min_train_samples training rows are skipped."""
        from src.models.position_models import SeasonAwareTimeSeriesSplit

        # 4 seasons with 10 samples each, gap=1, 3 folds
        # First fold would have only 10 training samples (1 season)
        seasons = np.repeat([2020, 2021, 2022, 2023], 10)
        X = np.random.randn(40, 5)

        cv = SeasonAwareTimeSeriesSplit(
            n_splits=3, seasons=seasons, gap_seasons=1,
            min_train_samples=15,
        )
        folds = list(cv.split(X))

        for train_idx, test_idx in folds:
            assert len(train_idx) >= 15

    def test_default_min_train_samples(self):
        """Default min_train_samples should be 30."""
        from src.models.position_models import SeasonAwareTimeSeriesSplit

        cv = SeasonAwareTimeSeriesSplit(n_splits=3)
        assert cv.min_train_samples == 30

    def test_fast_mode_fold_count(self):
        """With enough data, 3 folds and gap=1 should produce reasonable folds."""
        from src.models.position_models import SeasonAwareTimeSeriesSplit

        seasons = np.repeat(np.arange(2014, 2025), 200)
        X = np.random.randn(len(seasons), 10)

        cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1)
        folds = list(cv.split(X))

        assert len(folds) == 3
        for train_idx, _ in folds:
            assert len(train_idx) >= 200

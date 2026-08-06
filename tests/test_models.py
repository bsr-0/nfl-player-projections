"""Tests for ML models."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.position_models import PositionModel, MultiWeekModel
from src.models.ensemble import EnsemblePredictor, ModelTrainer


class TestPositionModel:
    """Test suite for PositionModel."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate features
        X = pd.DataFrame({
            "rushing_yards_roll3_mean": np.random.uniform(30, 100, n_samples),
            "rushing_attempts_roll3_mean": np.random.uniform(8, 20, n_samples),
            "targets_roll3_mean": np.random.uniform(2, 8, n_samples),
            "utilization_score": np.random.uniform(30, 90, n_samples),
            "snap_share": np.random.uniform(0.3, 0.9, n_samples),
            "total_touches_lag1": np.random.uniform(10, 30, n_samples),
            "fantasy_points_lag1": np.random.uniform(5, 25, n_samples),
            "yards_per_carry": np.random.uniform(3, 6, n_samples),
        })
        
        # Generate target with some relationship to features
        y = (
            X["rushing_yards_roll3_mean"] * 0.1 +
            X["targets_roll3_mean"] * 1.5 +
            X["utilization_score"] * 0.1 +
            np.random.normal(0, 3, n_samples)
        )
        
        return X, pd.Series(y)
    
    def test_model_initialization(self):
        """Test model initializes correctly."""
        model = PositionModel("RB", n_weeks=1)
        
        assert model.position == "RB"
        assert model.n_weeks == 1
        assert not model.is_fitted
    
    def test_model_training_no_tune(self, sample_training_data):
        """Test model training without hyperparameter tuning."""
        X, y = sample_training_data
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        assert model.is_fitted
        assert "xgboost" in model.models
        assert "random_forest" in model.models
        assert "ridge" in model.models
    
    def test_model_prediction(self, sample_training_data):
        """Test model makes predictions."""
        X, y = sample_training_data
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))
    
    def test_prediction_with_uncertainty(self, sample_training_data):
        """Test prediction with uncertainty estimates."""
        X, y = sample_training_data
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        mean_pred, std_pred = model.predict_with_uncertainty(X)
        
        assert len(mean_pred) == len(X)
        assert len(std_pred) == len(X)
        assert np.all(std_pred >= 0)  # Std should be non-negative
    
    def test_model_evaluation(self, sample_training_data):
        """Test model evaluation metrics."""
        X, y = sample_training_data
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        metrics = model.evaluate(X, y)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["r2"] > 0  # Should be better than random
    
    def test_feature_importance(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == len(X.columns)
        assert "feature" in importance.columns
        assert "combined" in importance.columns


class TestMultiWeekModel:
    """Test suite for MultiWeekModel."""
    
    @pytest.fixture
    def sample_data_with_targets(self):
        """Generate sample data with multiple target horizons."""
        np.random.seed(42)
        n_samples = 300
        
        X = pd.DataFrame({
            "rushing_yards_roll3_mean": np.random.uniform(30, 100, n_samples),
            "utilization_score": np.random.uniform(30, 90, n_samples),
            "fantasy_points_lag1": np.random.uniform(5, 25, n_samples),
        })
        
        base_target = X["rushing_yards_roll3_mean"] * 0.1 + X["utilization_score"] * 0.1
        
        y_dict = {
            1: pd.Series(base_target + np.random.normal(0, 2, n_samples)),
            4: pd.Series(base_target * 4 + np.random.normal(0, 5, n_samples)),
            12: pd.Series(base_target * 12 + np.random.normal(0, 10, n_samples)),
        }
        
        return X, y_dict
    
    def test_multiweek_initialization(self):
        """Test multi-week model initialization."""
        model = MultiWeekModel("RB")
        
        assert model.position == "RB"
        assert "short" in model.horizon_groups
        assert "medium" in model.horizon_groups
        assert "long" in model.horizon_groups
    
    def test_multiweek_training(self, sample_data_with_targets):
        """Test multi-week model training."""
        X, y_dict = sample_data_with_targets
        
        model = MultiWeekModel("RB")
        model.fit(X, y_dict, tune_hyperparameters=False)
        
        # Should have models for different horizons
        assert len(model.models) > 0
    
    def test_multiweek_prediction(self, sample_data_with_targets):
        """Test predictions for different week horizons."""
        X, y_dict = sample_data_with_targets
        
        model = MultiWeekModel("RB")
        model.fit(X, y_dict, tune_hyperparameters=False)
        
        # Predict for different horizons
        pred_1w = model.predict(X, n_weeks=1)
        pred_4w = model.predict(X, n_weeks=4)
        pred_12w = model.predict(X, n_weeks=12)
        
        assert len(pred_1w) == len(X)
        assert len(pred_4w) == len(X)
        assert len(pred_12w) == len(X)
        
        # Longer horizon should predict higher total points
        assert pred_12w.mean() > pred_1w.mean()


class TestEnsemblePredictor:
    """Test suite for EnsemblePredictor."""

    def test_predictor_initialization(self):
        """Test ensemble predictor initialization."""
        predictor = EnsemblePredictor()

        assert not predictor.is_loaded
        assert len(predictor.position_models) == 0

    def test_predictor_without_models(self):
        """Test predictor raises error without loaded models."""
        predictor = EnsemblePredictor()

        data = pd.DataFrame({"position": ["RB"], "feature1": [1.0]})

        with pytest.raises(ValueError):
            predictor.predict(data, n_weeks=1)

    def test_multiweek_ci_uses_matching_horizon_model(self):
        """GAPS.md §7.4 follow-up regression test: the CI for an n_weeks
        prediction must come from the representative model that actually
        covers that horizon, not always the 1-week model.

        Builds a MultiWeekModel with distinguishable per-horizon
        uncertainty (via _uncertainty_scale_factor) on its 1w vs 4w
        sub-models, predicts at n_weeks=4 through EnsemblePredictor, and
        asserts the resulting prediction_std reflects the 4-week model's
        factor. Before the fix, `base_model = model.models.get(1)` meant
        this always came from the 1-week model regardless of n_weeks.
        """
        np.random.seed(0)
        n_samples = 300
        X = pd.DataFrame({
            "rushing_yards_roll3_mean": np.random.uniform(30, 100, n_samples),
            "utilization_score": np.random.uniform(30, 90, n_samples),
            "fantasy_points_lag1": np.random.uniform(5, 25, n_samples),
        })
        base_target = X["rushing_yards_roll3_mean"] * 0.1 + X["utilization_score"] * 0.1
        y_dict = {
            1: pd.Series(base_target + np.random.normal(0, 2, n_samples)),
            4: pd.Series(base_target * 4 + np.random.normal(0, 5, n_samples)),
            18: pd.Series(base_target * 18 + np.random.normal(0, 10, n_samples)),
        }

        mw_model = MultiWeekModel("RB")
        mw_model.fit(X, y_dict, tune_hyperparameters=False)

        # Force distinguishable, easily-identifiable uncertainty factors
        # on the two representative models under test.
        one_week_model = mw_model.models[1]
        four_week_model = mw_model.models[4]
        assert one_week_model is not four_week_model  # sanity: distinct model objects
        one_week_model._uncertainty_scale_factor = 1.0
        one_week_model._uncertainty_scale_factors_per_level = {0.80: 1.0, 0.95: 1.0}
        four_week_model._uncertainty_scale_factor = 9.0
        four_week_model._uncertainty_scale_factors_per_level = {0.80: 9.0, 0.95: 9.0}

        predictor = EnsemblePredictor()
        predictor.position_models["RB"] = mw_model
        predictor.is_loaded = True

        player_data = X.copy()
        player_data["position"] = "RB"

        result_1w = predictor.predict(player_data, n_weeks=1)
        result_4w = predictor.predict(player_data, n_weeks=4)

        std_1w = result_1w["prediction_std"].dropna()
        std_4w = result_4w["prediction_std"].dropna()
        assert len(std_1w) > 0 and len(std_4w) > 0

        # The 4-week prediction's std must reflect the 4-week model's much
        # larger scale factor (9x), not the 1-week model's (1x). Before the
        # fix, both would have used the 1-week model's factor and this
        # ratio would be ~1.0 instead of close to 9.0.
        ratio = std_4w.mean() / std_1w.mean()
        assert ratio > 5.0, (
            f"Expected 4-week CI to reflect the 4-week model's own "
            f"calibration (~9x wider), got ratio={ratio:.2f}"
        )


class TestModelTrainer:
    """Test suite for ModelTrainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer()
        
        assert len(trainer.trained_models) == 0
        assert len(trainer.training_metrics) == 0
    
    def test_training_summary(self):
        """Test training summary generation."""
        trainer = ModelTrainer()
        
        # Add mock metrics
        trainer.training_metrics = {
            "RB": {
                "1w": {"rmse": 5.0, "mae": 4.0, "r2": 0.6},
                "4w": {"rmse": 8.0, "mae": 6.0, "r2": 0.5},
            }
        }
        
        summary = trainer.get_training_summary()
        
        assert len(summary) == 2
        assert "position" in summary.columns
        assert "rmse" in summary.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

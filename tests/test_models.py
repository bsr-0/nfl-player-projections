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

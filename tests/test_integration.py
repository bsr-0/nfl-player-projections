"""Integration tests for the complete NFL prediction pipeline."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train import prepare_features
from src.utils.database import DatabaseManager
from src.features.utilization_score import calculate_utilization_scores
from src.features.feature_engineering import FeatureEngineer
from src.features.dimensionality_reduction import DimensionalityReducer
from src.models.position_models import PositionModel
from src.evaluation.metrics import ModelEvaluator, run_model_tests


class TestEndToEndPipeline:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Get real data from database for testing."""
        db = DatabaseManager()
        data = db.get_all_players_for_training(min_games=1)
        if data.empty:
            pytest.skip("No data in database. Run: python3 src/data/nfl_data_loader.py --seasons 2020-2024")
        return data
    
    def test_full_pipeline_rb(self, sample_data):
        """Test complete pipeline for RB position."""
        # Filter to RB
        rb_data = sample_data[sample_data["position"] == "RB"].copy()
        
        # Step 1: Calculate utilization scores
        rb_data = calculate_utilization_scores(rb_data)
        assert "utilization_score" in rb_data.columns
        
        # Step 2: Engineer features
        engineer = FeatureEngineer()
        rb_data = engineer.create_features(rb_data)
        
        # Step 3: Prepare training data
        X, y = engineer.prepare_training_data(rb_data, target_weeks=1)
        
        assert len(X) > 0
        assert len(y) == len(X)
        
        # Step 4: Train model (no tuning for speed)
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        assert model.is_fitted
        
        # Step 5: Make predictions
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert not np.any(np.isnan(predictions))
        
        # Step 6: Evaluate
        metrics = model.evaluate(X, y)
        
        assert metrics["r2"] > 0  # Should be better than random
        print(f"RB Model R²: {metrics['r2']:.4f}")
    
    def test_full_pipeline_all_positions(self, sample_data):
        """Test pipeline works for all positions."""
        positions = ["QB", "RB", "WR", "TE"]
        
        for position in positions:
            pos_data = sample_data[sample_data["position"] == position].copy()
            
            if len(pos_data) < 50:
                continue
            
            # Process
            pos_data = calculate_utilization_scores(pos_data)
            engineer = FeatureEngineer()
            pos_data = engineer.create_features(pos_data)
            X, y = engineer.prepare_training_data(pos_data, target_weeks=1)
            
            if len(X) < 20:
                continue
            
            # Train
            model = PositionModel(position, n_weeks=1)
            model.fit(X, y, tune_hyperparameters=False)
            
            # Verify
            assert model.is_fitted
            predictions = model.predict(X)
            assert len(predictions) == len(X)
            
            print(f"{position} model trained successfully")
    
    def test_multiweek_predictions(self, sample_data):
        """Test predictions for different week horizons."""
        rb_data = sample_data[sample_data["position"] == "RB"].copy()
        rb_data = calculate_utilization_scores(rb_data)
        
        engineer = FeatureEngineer()
        rb_data = engineer.create_features(rb_data)
        
        # Create targets for different horizons
        for n_weeks in [1, 4]:
            rb_data[f"target_{n_weeks}w"] = rb_data.groupby("player_id")["fantasy_points"].transform(
                lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
            )
        
        # Get features
        feature_cols = engineer.get_feature_columns()
        available_features = [c for c in feature_cols if c in rb_data.columns]
        
        X = rb_data[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y_1w = rb_data["target_1w"]
        
        # Remove NaN targets
        valid_mask = ~y_1w.isna()
        X = X[valid_mask]
        y_1w = y_1w[valid_mask]
        
        # Train
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y_1w, tune_hyperparameters=False)
        
        # Predict
        pred_1w = model.predict(X)
        
        assert len(pred_1w) == len(X)
        assert pred_1w.mean() > 0


class TestDimensionalityReduction:
    """Test dimensionality reduction integration."""
    
    @pytest.fixture
    def feature_data(self):
        """Generate feature data with many columns."""
        np.random.seed(42)
        n_samples = 200
        
        # Create correlated features
        base = np.random.randn(n_samples)
        
        X = pd.DataFrame({
            "feature_1": base + np.random.randn(n_samples) * 0.1,
            "feature_2": base + np.random.randn(n_samples) * 0.1,  # Correlated with 1
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples) * 0.01,  # Low variance
            "feature_6": base * 2 + np.random.randn(n_samples) * 0.2,  # Correlated
            "feature_7": np.random.randn(n_samples),
            "feature_8": np.random.randn(n_samples),
        })
        
        y = pd.Series(base * 2 + X["feature_3"] + np.random.randn(n_samples) * 0.5)
        
        return X, y
    
    def test_dimensionality_reduction(self, feature_data):
        """Test dimensionality reduction works."""
        X, y = feature_data
        
        reducer = DimensionalityReducer(
            variance_threshold=0.001,
            correlation_threshold=0.9,
            n_features_to_select=5
        )
        
        X_reduced = reducer.fit_transform(X, y)
        
        assert X_reduced.shape[1] <= X.shape[1]
        assert len(X_reduced) == len(X)
    
    def test_feature_importance_ranking(self, feature_data):
        """Test feature importance is calculated."""
        X, y = feature_data
        
        reducer = DimensionalityReducer()
        reducer.fit(X, y)
        
        top_features = reducer.get_top_features(n=5)
        
        assert len(top_features) <= 5
        assert all(isinstance(f[0], str) for f in top_features)
        assert all(isinstance(f[1], float) for f in top_features)


class TestModelEvaluation:
    """Test model evaluation integration."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Get a trained model and test data."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
        })
        
        y = pd.Series(X["feature_1"] * 2 + X["feature_2"] + np.random.randn(n_samples) * 0.5)
        
        model = PositionModel("RB", n_weeks=1)
        model.fit(X, y, tune_hyperparameters=False)
        
        return model, X, y
    
    def test_model_evaluator(self, trained_model_and_data):
        """Test model evaluator."""
        model, X, y = trained_model_and_data
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X, y)
        
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] > 0
    
    def test_cross_validation(self, trained_model_and_data):
        """Test cross-validation evaluation."""
        model, X, y = trained_model_and_data
        
        evaluator = ModelEvaluator()
        cv_metrics = evaluator.cross_validate(model.models["ridge"], X, y, n_splits=3)
        
        assert "cv_rmse_mean" in cv_metrics
        assert "cv_r2_mean" in cv_metrics
    
    def test_automated_model_tests(self, trained_model_and_data):
        """Test automated model testing."""
        model, X, y = trained_model_and_data
        
        passed = run_model_tests(model, X, y)
        
        # Model should pass basic tests
        assert passed


class TestDataFlow:
    """Test data flows correctly through the pipeline."""
    
    def test_real_data_loading(self):
        """Test real data is loaded correctly from database."""
        db = DatabaseManager()
        data = db.get_all_players_for_training(min_games=1)
        
        if data.empty:
            pytest.skip("No data in database. Run: python3 src/data/nfl_data_loader.py --seasons 2020-2024")
        
        assert len(data) > 0
        assert "player_id" in data.columns
        assert "position" in data.columns
        assert "fantasy_points" in data.columns
        
        # Check all positions present
        positions = data["position"].unique()
        assert "QB" in positions
        assert "RB" in positions
        assert "WR" in positions
        assert "TE" in positions
    
    def test_feature_preparation(self):
        """Test feature preparation from real data."""
        db = DatabaseManager()
        data = db.get_all_players_for_training(min_games=1)
        
        if data.empty:
            pytest.skip("No data in database. Run: python3 src/data/nfl_data_loader.py --seasons 2020-2024")
        
        # Use a subset for speed
        data = data.head(1000)
        prepared = prepare_features(data)
        
        assert "utilization_score" in prepared.columns
        
        # Should have rolling features
        rolling_cols = [c for c in prepared.columns if "roll" in c]
        assert len(rolling_cols) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

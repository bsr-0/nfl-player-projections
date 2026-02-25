"""Tests for uncertainty calibration in PositionModel.

Verifies that the conformal recalibration factor correctly adjusts
predicted standard deviations so nominal coverage matches actual coverage.
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load position_models directly from file to avoid __init__.py pulling
# in torch-dependent modules (horizon_models.py) that may not be installed.
_mod_path = str(Path(__file__).parent.parent / "src" / "models" / "position_models.py")
_spec = importlib.util.spec_from_file_location("position_models", _mod_path)
_pm = importlib.util.module_from_spec(_spec)
sys.modules["position_models"] = _pm
_spec.loader.exec_module(_pm)
PositionModel = _pm.PositionModel


class TestUncertaintyScaleFactor:
    """Test that _uncertainty_scale_factor is properly initialized and applied."""

    def test_default_scale_factor_is_one(self):
        model = PositionModel("QB")
        assert model._uncertainty_scale_factor == 1.0

    def test_scale_factor_attribute_exists_after_init(self):
        model = PositionModel("RB")
        assert hasattr(model, "_uncertainty_scale_factor")

    def test_scale_factor_applied_in_predict_with_uncertainty(self):
        """Verify that a non-unity scale factor actually scales the std output."""
        model = PositionModel("WR")
        model.is_fitted = True
        model.feature_names = ["f1", "f2", "f3"]
        model.feature_medians = {"f1": 0.0, "f2": 0.0, "f3": 0.0}
        from sklearn.preprocessing import StandardScaler
        model.scaler = StandardScaler()

        # Create dummy training data to fit scaler
        X_dummy = np.random.randn(100, 3)
        model.scaler.fit(X_dummy)

        # Create simple mock models that return constant predictions
        class _MockModel:
            def __init__(self, val):
                self._val = val
            def predict(self, X):
                return np.full(X.shape[0], self._val)

        model.models = {
            "random_forest": _MockModel(10.0),
            "xgboost": _MockModel(12.0),
            "ridge": _MockModel(11.0),
        }
        model._base_model_keys = ["random_forest", "xgboost", "ridge"]
        model.meta_learner = None
        model.ensemble_weights = {"random_forest": 0.3, "xgboost": 0.4, "ridge": 0.3}
        model.target_transformer = None
        model._conformal_residual_std = 2.0
        model._uncertainty_model = None

        import pandas as pd
        X_test = pd.DataFrame(np.random.randn(20, 3), columns=["f1", "f2", "f3"])

        # Get predictions with scale_factor = 1.0
        model._uncertainty_scale_factor = 1.0
        _, std_baseline = model.predict_with_uncertainty(X_test)

        # Get predictions with scale_factor = 2.0
        model._uncertainty_scale_factor = 2.0
        _, std_scaled = model.predict_with_uncertainty(X_test)

        # Scaled std should be exactly 2x the baseline
        np.testing.assert_allclose(std_scaled, std_baseline * 2.0, rtol=1e-6)

    def test_scale_factor_does_not_affect_mean_prediction(self):
        """The recalibration factor should only affect std, not mean predictions."""
        model = PositionModel("TE")
        model.is_fitted = True
        model.feature_names = ["f1", "f2"]
        model.feature_medians = {"f1": 0.0, "f2": 0.0}
        from sklearn.preprocessing import StandardScaler
        model.scaler = StandardScaler()
        model.scaler.fit(np.random.randn(50, 2))

        class _MockModel:
            def __init__(self, val):
                self._val = val
            def predict(self, X):
                return np.full(X.shape[0], self._val)

        model.models = {"random_forest": _MockModel(8.0), "xgboost": _MockModel(8.0), "ridge": _MockModel(8.0)}
        model._base_model_keys = ["random_forest", "xgboost", "ridge"]
        model.meta_learner = None
        model.ensemble_weights = {"random_forest": 0.3, "xgboost": 0.4, "ridge": 0.3}
        model.target_transformer = None
        model._conformal_residual_std = 1.5
        model._uncertainty_model = None

        import pandas as pd
        X_test = pd.DataFrame(np.random.randn(10, 2), columns=["f1", "f2"])

        model._uncertainty_scale_factor = 1.0
        mean_1, _ = model.predict_with_uncertainty(X_test)

        model._uncertainty_scale_factor = 3.0
        mean_2, _ = model.predict_with_uncertainty(X_test)

        np.testing.assert_allclose(mean_1, mean_2, rtol=1e-10)

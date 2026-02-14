"""Tests for horizon-model fallback behavior and availability reporting."""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import EnsemblePredictor
from src.models.horizon_models import Hybrid4WeekModel
from src.features.multiweek_features import MultiWeekFeatureEngineer, add_multiweek_features


def test_hybrid_predict_returns_fallback_when_unfitted():
    model = Hybrid4WeekModel("RB")
    model.is_fitted = False
    x = pd.DataFrame({"f1": [1.0, 2.0]})
    fallback = np.array([11.0, 12.0])
    out = model.predict(x, np.array(["p1", "p2"]), ["f1"], fallback)
    assert np.allclose(out, fallback)


def test_ensemble_horizon_availability_has_dependency_reasons(tmp_path, monkeypatch):
    predictor = EnsemblePredictor()
    # Avoid loading repository-persisted pickles in tests (version-mismatch warnings).
    from src.models import ensemble as ensemble_module
    monkeypatch.setattr(ensemble_module, "MODELS_DIR", tmp_path)
    predictor.load_models(positions=["QB"])
    status = predictor.get_horizon_availability()
    assert "QB" in status
    assert "hybrid_4w" in status["QB"]
    assert "deep_18w" in status["QB"]


def test_multiweek_feature_engineer_default_horizons_are_1_4_18():
    defaults = MultiWeekFeatureEngineer.add_multiweek_features.__defaults__
    assert defaults is not None
    assert list(defaults[0]) == [1, 4, 18]


def test_add_multiweek_features_default_horizons_are_1_4_18():
    defaults = add_multiweek_features.__defaults__
    assert defaults is not None
    assert list(defaults[0]) == [1, 4, 18]

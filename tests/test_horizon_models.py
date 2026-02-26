"""Tests for horizon-model fallback behavior, availability reporting, and critical fixes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import EnsemblePredictor
from src.models.horizon_models import Hybrid4WeekModel, _season_split
from src.features.multiweek_features import MultiWeekFeatureEngineer, add_multiweek_features


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# C1: Season-aware train/val split tests
# ---------------------------------------------------------------------------


def test_season_split_holds_out_last_season():
    """_season_split puts only the last season in validation."""
    seasons = np.array([2020] * 60 + [2021] * 60 + [2022] * 60 + [2023] * 60)
    train_idx, val_idx = _season_split(len(seasons), seasons, n_val_seasons=1, gap_seasons=0)
    assert set(seasons[val_idx]) == {2023}
    assert 2023 not in set(seasons[train_idx])


def test_season_split_respects_gap_seasons():
    """Purge gap removes seasons immediately before validation."""
    seasons = np.array([2020] * 60 + [2021] * 60 + [2022] * 60 + [2023] * 60)
    train_idx, val_idx = _season_split(len(seasons), seasons, n_val_seasons=1, gap_seasons=1)
    assert set(seasons[val_idx]) == {2023}
    assert 2022 not in set(seasons[train_idx]), "Gap season 2022 should be excluded from train"
    assert 2023 not in set(seasons[train_idx])
    assert {2020, 2021} == set(seasons[train_idx])


def test_season_split_fallback_when_no_seasons():
    """Falls back to 80/20 index split when seasons is None."""
    train_idx, val_idx = _season_split(100, None)
    assert len(train_idx) == 80
    assert len(val_idx) == 20
    assert train_idx[-1] < val_idx[0]


def test_season_split_fallback_with_insufficient_seasons():
    """Falls back to 80/20 when only 1 unique season exists."""
    seasons = np.array([2023] * 100)
    train_idx, val_idx = _season_split(100, seasons, n_val_seasons=1)
    assert len(train_idx) == 80
    assert len(val_idx) == 20


def test_lstm_sequences_return_season_labels():
    """_sequences returns season labels aligned with target rows."""
    from src.models.horizon_models import LSTM4WeekModel

    model = LSTM4WeekModel(sequence_length=3)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    pids = np.array(["p1"] * 20)
    seasons = np.array([2022] * 10 + [2023] * 10)

    X_seq, y_seq, seq_seasons = model._sequences(X, y, pids, seasons=seasons)
    assert seq_seasons is not None
    assert len(seq_seasons) == len(y_seq)
    # Sequences targeting rows 3-9 should have season 2022
    # Sequences targeting rows 10-16 should have season 2023
    for i, s in enumerate(seq_seasons):
        assert s in (2022, 2023)


def test_lstm_sequences_no_seasons_returns_none():
    """_sequences returns None for seq_seasons when seasons is not provided."""
    from src.models.horizon_models import LSTM4WeekModel

    model = LSTM4WeekModel(sequence_length=3)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randn(20).astype(np.float32)
    pids = np.array(["p1"] * 20)

    X_seq, y_seq, seq_seasons = model._sequences(X, y, pids)
    assert seq_seasons is None


def test_lstm_fit_accepts_seasons_param():
    """LSTM4WeekModel.fit() accepts seasons parameter without error."""
    from src.models.horizon_models import LSTM4WeekModel

    model = LSTM4WeekModel(sequence_length=3)
    n = 120
    X = np.random.randn(n, 5).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    pids = np.array(["p1"] * 60 + ["p2"] * 60)
    seasons = np.array([2021] * 30 + [2022] * 30 + [2021] * 30 + [2022] * 30)

    # Should not raise
    model.fit(X, y, pids, [f"f{i}" for i in range(5)], epochs=2, seasons=seasons)
    assert model.is_fitted


def test_deep_fit_accepts_seasons_param():
    """DeepSeasonLongModel.fit() accepts seasons parameter without error."""
    from src.models.horizon_models import DeepSeasonLongModel

    model = DeepSeasonLongModel("RB", n_features=5)
    n = 200
    X = np.random.randn(n, 5).astype(np.float64)
    y = np.random.randn(n).astype(np.float64)
    seasons = np.array([2020] * 50 + [2021] * 50 + [2022] * 50 + [2023] * 50)

    # Should not raise
    model.fit(X, y, [f"f{i}" for i in range(5)], epochs=2, seasons=seasons)
    assert model.is_fitted


# ---------------------------------------------------------------------------
# C2: LSTM DataLoader shuffle=True tests
# ---------------------------------------------------------------------------


def test_lstm_fit_code_has_shuffle_true():
    """Verify LSTM fit() source code uses shuffle=True in DataLoader."""
    import inspect
    from src.models.horizon_models import LSTM4WeekModel

    source = inspect.getsource(LSTM4WeekModel.fit)
    # Should contain shuffle=True, not shuffle=False
    assert "shuffle=True" in source, "LSTM fit() DataLoader must use shuffle=True"
    assert "shuffle=False" not in source, "LSTM fit() DataLoader must not use shuffle=False"


def test_lstm_tune_code_has_shuffle_true():
    """Verify LSTM tune_hyperparameters() source code uses shuffle=True."""
    import inspect
    from src.models.horizon_models import LSTM4WeekModel

    source = inspect.getsource(LSTM4WeekModel.tune_hyperparameters)
    assert "shuffle=True" in source, "LSTM tune DataLoader must use shuffle=True"
    assert "shuffle=False" not in source, "LSTM tune DataLoader must not use shuffle=False"


# ---------------------------------------------------------------------------
# C3: ARIMA dynamic predictions tests
# ---------------------------------------------------------------------------


def test_arima_predict_uses_y_hist_dict():
    """ARIMA prediction changes when provided recent data via y_hist dict."""
    from src.models.horizon_models import ARIMA4WeekModel

    arima = ARIMA4WeekModel(order=(1, 0, 0))
    # Create a simple trending series per player
    y = np.concatenate([
        np.arange(20, dtype=np.float64),
        np.arange(20, dtype=np.float64) * 0.5,
    ])
    pids = np.array(["p1"] * 20 + ["p2"] * 20)
    arima.fit(y, pids)

    X = np.zeros((2, 3))
    player_ids = np.array(["p1", "p2"])

    # Predict with no recent data (empty dict = use cached forecasts)
    pred_static = arima.predict(X, {}, player_ids)
    assert np.all(np.isfinite(pred_static))

    # Predict with recent high values for p1
    y_hist = {"p1": np.array([30.0, 35.0, 40.0, 45.0])}
    pred_updated = arima.predict(X, y_hist, player_ids)
    assert np.all(np.isfinite(pred_updated))

    # p1's prediction should change; p2 should stay the same
    assert pred_updated[0] != pred_static[0], \
        "ARIMA prediction should change when given recent y_hist data"
    assert pred_updated[1] == pred_static[1], \
        "p2 prediction should be unchanged when no y_hist provided"


def test_arima_predict_fallback_when_no_history():
    """ARIMA returns cached forecast when y_hist is empty dict or zeros array."""
    from src.models.horizon_models import ARIMA4WeekModel

    arima = ARIMA4WeekModel(order=(1, 0, 0))
    y = np.arange(20, dtype=np.float64)
    pids = np.array(["p1"] * 20)
    arima.fit(y, pids)

    X = np.zeros((1, 3))
    player_ids = np.array(["p1"])

    # Empty dict should use cached forecast
    pred_empty = arima.predict(X, {}, player_ids)
    assert np.isfinite(pred_empty[0])

    # Legacy zeros array should also work without crashing
    pred_zeros = arima.predict(X, np.zeros(1), player_ids)
    assert np.isfinite(pred_zeros[0])

    # Both should return the same cached value
    assert pred_empty[0] == pred_zeros[0]


def test_arima_predict_with_different_n_steps():
    """ARIMA with n_steps=8 produces different results than n_steps=4."""
    from src.models.horizon_models import ARIMA4WeekModel

    arima = ARIMA4WeekModel(order=(1, 0, 0))
    y = np.arange(30, dtype=np.float64) + np.random.randn(30) * 0.5
    pids = np.array(["p1"] * 30)
    arima.fit(y, pids)

    X = np.zeros((1, 3))
    player_ids = np.array(["p1"])

    # Provide y_hist so the online update path (which uses n_steps) is exercised.
    # With empty dict {}, the cached forecast is returned regardless of n_steps.
    y_hist = {"p1": np.array([25.0, 28.0, 30.0, 32.0])}
    pred_4 = arima.predict(X, y_hist, player_ids, n_steps=4)
    pred_8 = arima.predict(X, y_hist, player_ids, n_steps=8)
    assert np.isfinite(pred_4[0])
    assert np.isfinite(pred_8[0])
    # With trending data and different forecast horizons, predictions should differ
    # (unless the series is perfectly constant, which random noise prevents)
    assert pred_4[0] != pred_8[0], \
        "Different forecast horizons should generally produce different predictions"


def test_arima_stores_training_series():
    """ARIMA fit() stores per-player training series for online updating."""
    from src.models.horizon_models import ARIMA4WeekModel

    arima = ARIMA4WeekModel(order=(1, 0, 0))
    y = np.concatenate([np.arange(15, dtype=np.float64), np.arange(10, dtype=np.float64)])
    pids = np.array(["p1"] * 15 + ["p2"] * 10)
    arima.fit(y, pids)

    assert "p1" in arima._player_train_series
    assert "p2" in arima._player_train_series
    assert len(arima._player_train_series["p1"]) == 15
    assert len(arima._player_train_series["p2"]) == 10


def test_hybrid_predict_passes_n_weeks():
    """Hybrid4WeekModel.predict() accepts n_weeks parameter without error."""
    model = Hybrid4WeekModel("RB")
    model.is_fitted = False
    x = pd.DataFrame({"f1": [1.0, 2.0]})
    fallback = np.array([11.0, 12.0])
    # Should not raise with n_weeks parameter
    out = model.predict(x, np.array(["p1", "p2"]), ["f1"], fallback, n_weeks=6)
    assert np.allclose(out, fallback)


def test_hybrid_fit_with_seasons():
    """Hybrid4WeekModel.fit() accepts and uses seasons parameter."""
    model = Hybrid4WeekModel("RB")
    n = 200
    df = pd.DataFrame({f"f{i}": np.random.randn(n) for i in range(10)})
    y = pd.Series(np.random.randn(n))
    pids = np.array(["p1"] * 100 + ["p2"] * 100)
    seasons = np.array([2021] * 50 + [2022] * 50 + [2021] * 50 + [2022] * 50)

    # Should not raise
    model.fit(df, y, pids, [f"f{i}" for i in range(10)],
              epochs=2, tune_lstm=False, seasons=seasons)

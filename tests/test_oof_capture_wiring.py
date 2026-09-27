"""The capture must actually be reached by the fold runner.

`tests/test_oof_capture.py` proves the module is correct in isolation. That
is not the same as it being called: the artifact this exists to produce is
written from inside `_run_one_fold`, and a capture that is silently never
invoked looks exactly like a capture that works until someone goes looking
for the panel. This exercises the wiring with training stubbed out.
"""
import numpy as np
import pandas as pd
import pytest

import src.models.train as train_module
from src.models.oof_capture import ACTUAL_COLUMN, OOFLeakageError


class _StubMultiModel:
    """Stands in for a trained MultiWeekModel."""

    def __init__(self):
        self.models = {1: self}
        self.feature_names = []
        self.feature_medians = {}

    def predict(self, frame, n_weeks=1):
        return np.full(len(frame), 12.0)


def _test_frame(season=2024, n=24):
    """n players, each with one forecast origin and a season-final game.

    target_1w is built the way _create_horizon_targets builds it (next row's
    fantasy points within player-season), so the final game has none.
    """
    origin = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "season": season,
        "week": [1 + i % 4 for i in range(n)],
        "team": ["A", "B"] * (n // 2),
        "opponent": ["B", "A"] * (n // 2),
        "position": [["QB", "RB", "WR", "TE"][i % 4] for i in range(n)],
        "fantasy_points": np.linspace(8, 16, n),
    })
    final = origin.assign(week=origin["week"] + 10, fantasy_points=origin["fantasy_points"] + 1)
    frame = pd.concat([origin, final], ignore_index=True)
    frame["target_1w"] = frame.groupby(["player_id", "season"])["fantasy_points"].shift(-1)
    return frame


@pytest.fixture
def stubbed(monkeypatch):
    """Replace training and backtesting; keep the capture path real."""
    class _Trainer:
        trained_models = {p: _StubMultiModel() for p in ("QB", "RB", "WR", "TE")}
        training_metrics = {}

    def _prepare(train_data, test_data, positions, *a, **k):
        return train_data, test_data, _Trainer()

    monkeypatch.setattr(train_module, "_prepare_training_data", _prepare)
    monkeypatch.setattr(train_module, "_run_backtest_after_training",
                        lambda *a, **k: {"by_position": {}})
    monkeypatch.setattr(train_module, "_load_qb_target_choice", lambda: "fp")
    return _Trainer


def test_fold_runner_populates_the_collector(stubbed):
    collector = []
    train_module._run_one_fold(
        _test_frame(2023), _test_frame(2024), [2022, 2023], 2024,
        ["QB", "RB", "WR", "TE"], False, None, oof_collector=collector,
    )
    assert len(collector) == 1, "capture was never reached from _run_one_fold"
    rows = collector[0]
    assert len(rows) == 24
    assert {"player_id", "season", "week", "predicted_points",
            "actual_points", "residual"} <= set(rows.columns)
    assert (rows["season"] == 2024).all()
    # actual_points is the season-final game's outcome, keyed as such.
    assert (rows["target_week"] == rows["week"] + 10).all()


def test_fold_runner_works_unchanged_without_a_collector(stubbed):
    """The parameter is optional; existing callers must be unaffected."""
    trainer, results = train_module._run_one_fold(
        _test_frame(2023), _test_frame(2024), [2022, 2023], 2024,
        ["QB"], False, None,
    )
    assert results == {"by_position": {}}


def test_positions_below_the_minimum_test_size_are_absent_not_zeroed(stubbed):
    """_run_one_fold skips any position with <5 test rows, so those rows
    never receive a prediction. They must be absent from the panel rather
    than entering it with a null-or-zero prediction.
    """
    collector = []
    thin = _test_frame(2024, n=24)
    thin = thin[thin["position"] != "TE"].copy()          # WR/RB/QB keep 12 rows each
    te = _test_frame(2024, n=24).query("position == 'TE'")
    thin = pd.concat([thin, te[te["player_id"].isin(["p3"])]],   # 2 TE rows
                     ignore_index=True)
    train_module._run_one_fold(
        _test_frame(2023), thin, [2022, 2023], 2024,
        ["QB", "RB", "WR", "TE"], False, None, oof_collector=collector,
    )
    captured = collector[0]
    assert "TE" not in set(captured["position"]), "under-sized position leaked in"
    assert set(captured["position"]) == {"QB", "RB", "WR"}


def test_leakage_in_a_fold_aborts_rather_than_recording_it(stubbed):
    """Capture errors are swallowed as warnings, but a leakage error must
    propagate -- silently dropping a contaminated fold would leave a panel
    that is quietly missing a season instead of loudly wrong.
    """
    with pytest.raises(OOFLeakageError):
        train_module._run_one_fold(
            _test_frame(2024), _test_frame(2024), [2023, 2024], 2024,
            ["QB"], False, None, oof_collector=[],
        )

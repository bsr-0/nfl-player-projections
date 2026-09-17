"""The QB dual-target path must hand season labels to every model it fits.

Without them PositionModel.fit's SeasonAwareTimeSeriesSplit silently falls
back to a plain TimeSeriesSplit -- mid-season cuts, no purge gap -- so QB's
Optuna folds and OOF-stacking folds were leakier than RB/WR/TE's (which go
through the single-model path and do get the labels). Found 2026-09-16 by
dumping the matrix the tuner actually received: seasons=None.

Synthetic frame; MultiWeekModel is replaced by a recorder so no real
training happens.
"""
import numpy as np
import pandas as pd
import pytest

import src.models.ensemble as ens


class _Stop(Exception):
    pass


class _RecordingModel:
    calls = []

    def __init__(self, position):
        self.position = position
        self.models = {}

    def fit(self, X, y_dict, tune_hyperparameters=False, sample_weight=None, seasons=None):
        _RecordingModel.calls.append({"n_rows": len(X), "seasons": seasons, "sample_weight": sample_weight})
        if len(_RecordingModel.calls) == 3:
            raise _Stop()
        return self

    def predict(self, X, n_weeks=1):
        return np.zeros(len(X))


class _FakeConverter:
    def __init__(self, position):
        self.is_fitted = False

    def fit(self, df, target_col="fantasy_points"):
        self.is_fitted = False


@pytest.fixture
def qb_frame():
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2016, 2024):
        for week in range(1, 18):
            for p in range(4):
                rows.append({
                    "player_id": f"qb{p}", "position": "QB", "season": season, "week": week,
                    "team": "T", "fantasy_points": rng.normal(15, 8),
                    "passing_yards_lag1": rng.normal(250, 60), "snap_share_lag1": rng.uniform(0.5, 1),
                    "age": 25 + p, "target_util_1w": rng.normal(46, 17), "target_1w": rng.normal(15, 8),
                })
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def test_all_three_fits_receive_aligned_season_labels(monkeypatch, qb_frame):
    _RecordingModel.calls = []
    monkeypatch.setattr(ens, "MultiWeekModel", _RecordingModel)
    monkeypatch.setattr(ens, "UtilizationToFPConverter", _FakeConverter)
    monkeypatch.setitem(ens.MODEL_CONFIG, "validation_pct", 0.2)
    monkeypatch.setitem(ens.MODEL_CONFIG, "recency_decay_halflife", None)

    trainer = ens.ModelTrainer.__new__(ens.ModelTrainer)
    with pytest.raises(_Stop):
        trainer._train_qb_dual_and_pick(qb_frame, qb_frame.tail(40), [1], tune_hyperparameters=False)

    assert len(_RecordingModel.calls) == 3, "util candidate, fp candidate, winner"
    for i, call in enumerate(_RecordingModel.calls):
        s = call["seasons"]
        assert s is not None, f"fit #{i + 1} got seasons=None -> CV silently falls back to TimeSeriesSplit"
        assert len(s) == call["n_rows"], f"fit #{i + 1}: seasons misaligned with X"
        assert np.all(np.diff(s.astype(int)) >= 0), "season labels must be in temporal order"

    sel_util, sel_fp, winner = _RecordingModel.calls
    assert sel_util["n_rows"] == sel_fp["n_rows"] < winner["n_rows"]
    assert set(np.unique(winner["seasons"])) == set(range(2016, 2024))
    # The selection fits see the earlier 80%; the winner sees everything.
    assert winner["seasons"].max() >= sel_util["seasons"].max()

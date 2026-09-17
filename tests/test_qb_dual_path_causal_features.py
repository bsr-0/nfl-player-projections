"""The QB dual-target path must honour causal feature mode.

The single-model path (RB/WR/TE) restricts X to CAUSAL_FEATURES[position]
when FEATURE_MODE == "causal". The QB dual path relied on
select_features_simple instead -- which is a no-op in causal mode -- so
whenever QB had enough holdout rows to reach the dual path it trained on
every numeric column (465 in the 2026-09-16 retrain) while the other
positions got their ~65 causal features. The 09:27 models only had 62 QB
features because that run fell back to the single-model path
(qb_target_choice.json: fallback_no_qb_holdout).

Synthetic frame; MultiWeekModel is replaced by a recorder so no real
training happens.
"""
import numpy as np
import pandas as pd
import pytest

import config.settings as settings
import src.models.ensemble as ens


class _Stop(Exception):
    pass


class _RecordingModel:
    calls = []

    def __init__(self, position):
        self.position = position
        self.models = {}

    def fit(self, X, y_dict, tune_hyperparameters=False, sample_weight=None, seasons=None):
        _RecordingModel.calls.append(list(X.columns))
        raise _Stop()

    def predict(self, X, n_weeks=1):
        return np.zeros(len(X))


class _FakeConverter:
    def __init__(self, position):
        self.is_fitted = False

    def fit(self, df, target_col="fantasy_points"):
        self.is_fitted = False


CAUSAL = ["passing_yards_lag1", "snap_share_lag1"]
NOISE = [f"noise_{i}" for i in range(20)]


@pytest.fixture
def qb_frame():
    rng = np.random.default_rng(0)
    rows = []
    for season in range(2016, 2024):
        for week in range(1, 18):
            for p in range(4):
                row = {
                    "player_id": f"qb{p}", "position": "QB", "season": season, "week": week,
                    "team": "T", "fantasy_points": rng.normal(15, 8),
                    "passing_yards_lag1": rng.normal(250, 60), "snap_share_lag1": rng.uniform(0.5, 1),
                    "age": 25 + p, "target_util_1w": rng.normal(46, 17), "target_1w": rng.normal(15, 8),
                }
                row.update({c: rng.normal() for c in NOISE})
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


def _run(monkeypatch, qb_frame, mode):
    _RecordingModel.calls = []
    monkeypatch.setattr(ens, "MultiWeekModel", _RecordingModel)
    monkeypatch.setattr(ens, "UtilizationToFPConverter", _FakeConverter)
    monkeypatch.setitem(ens.MODEL_CONFIG, "validation_pct", 0.2)
    monkeypatch.setitem(ens.MODEL_CONFIG, "recency_decay_halflife", None)
    monkeypatch.setitem(ens.MODEL_CONFIG, "n_features_per_position", 50)
    monkeypatch.setattr(settings, "FEATURE_MODE", mode)
    monkeypatch.setitem(settings.CAUSAL_FEATURES, "QB", CAUSAL + ["not_in_frame"])

    trainer = ens.ModelTrainer.__new__(ens.ModelTrainer)
    with pytest.raises(_Stop):
        trainer._train_qb_dual_and_pick(qb_frame, qb_frame.tail(40), [1], tune_hyperparameters=False)
    return _RecordingModel.calls[0]


def test_causal_mode_restricts_qb_to_the_causal_list(monkeypatch, qb_frame):
    cols = _run(monkeypatch, qb_frame, "causal")
    assert cols == CAUSAL, cols


def test_full_mode_still_sees_the_wide_matrix(monkeypatch, qb_frame):
    # n_features_per_position (50) > candidate count, so no pruning happens
    # and every numeric non-target column must reach the model.
    cols = _run(monkeypatch, qb_frame, "full")
    assert set(CAUSAL + NOISE + ["age"]).issubset(cols), cols

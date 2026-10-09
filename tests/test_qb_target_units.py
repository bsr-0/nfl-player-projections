"""A QB model is only converted util->FP when position_target_type["QB"] is
"util" and a fitted converter exists. The dual path used to let a util model
win regardless, and the single-path fallback recorded "fp" regardless; either
served raw utilization as fantasy points (found 2026-10-07)."""
import json

import numpy as np
import pandas as pd
import pytest

import src.models.ensemble as ens


class _Model:
    fits = []

    def __init__(self, position):
        self.models = {}
        self.target = None

    def fit(self, X, y_dict, tune_hyperparameters=False, sample_weight=None, seasons=None):
        self.target = y_dict[1].name
        _Model.fits.append(self.target)
        return self

    def predict(self, X, n_weeks=1):
        # The util model's output, scaled by the unfitted-converter 0.25 (or
        # converted by _Converter), lands on the FP mean, so util always "wins".
        return np.full(len(X), 60.0 if self.target == "target_util_1w" else 0.0)

    def save(self, *a, **k):
        pass


def _frame():
    rng = np.random.default_rng(1)
    rows = [{"player_id": f"qb{p}", "position": "QB", "season": s, "week": w, "team": "T",
             "fantasy_points": 15.0, "passing_yards_lag1": rng.normal(250, 60),
             "age": 25 + p, "target_util_1w": 0.0, "target_1w": rng.normal(15, 8)}
            for s in range(2018, 2024) for w in range(1, 18) for p in range(4)]
    return pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)


class _Converter:
    fitted = True

    def __init__(self, position):
        self.is_fitted = _Converter.fitted

    def fit(self, df, target_col="fantasy_points"):
        self.is_fitted = _Converter.fitted

    def predict(self, preds, efficiency_df=None):
        # util predictions of 0 convert to a perfect-looking 15 -> util "wins"
        return np.full(len(preds), 15.0)

    def save(self):
        pass


@pytest.mark.parametrize("config_target,fitted,expected", [
    ("fp", True, "fp"), ("util", False, "fp"), ("util", True, "util")])
def test_dual_path_never_picks_an_unconvertible_util_model(
        monkeypatch, tmp_path, config_target, fitted, expected):
    _Model.fits = []
    _Converter.fitted = fitted
    monkeypatch.setattr(ens, "MultiWeekModel", _Model)
    monkeypatch.setattr(ens, "UtilizationToFPConverter", _Converter)
    monkeypatch.setattr(ens, "MODELS_DIR", tmp_path)
    monkeypatch.setitem(ens.MODEL_CONFIG, "position_target_type",
                        {"QB": config_target, "RB": "fp", "WR": "fp", "TE": "fp"})
    monkeypatch.setitem(ens.MODEL_CONFIG, "recency_decay_halflife", None)
    monkeypatch.setattr(ens.ModelTrainer, "_evaluate_model", lambda self, *a: {})
    frame = _frame()
    trainer = ens.ModelTrainer.__new__(ens.ModelTrainer)
    chosen = trainer._train_qb_dual_and_pick(frame, frame.tail(40), [1], tune_hyperparameters=False)
    assert chosen["qb_target"] == expected
    winner_target = _Model.fits[-1]
    assert winner_target == ("target_util_1w" if expected == "util" else "target_1w")
    assert chosen["util_vetoed"] == (expected == "fp")


def test_single_path_records_the_target_it_trained_on():
    import inspect
    source = inspect.getsource(ens.ModelTrainer.train_all_positions)
    fallback = source[source.index("QB fallback"):source.index("# Evaluate")]
    assert '"qb_target": target_type' in fallback

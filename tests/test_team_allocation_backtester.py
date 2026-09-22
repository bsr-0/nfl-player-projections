"""Walk-forward correctness for src/evaluation/team_share_backtester.py.

Mirrors tests/test_game_outcome_split.py's invariants for the sibling
game-outcome backtester, applied to Plan A's share models. Bypasses the
database entirely by monkeypatching `load_share_rows` -- the same technique
test_game_outcome_split.py uses on `build_game_outcome_rows` -- rather than
trying to point config.settings.DB_PATH at a temp file (features.py already
did `from config.settings import DB_PATH`, a local binding a DB_PATH
monkeypatch would not reach).
"""
import numpy as np
import pandas as pd
import pytest

import src.evaluation.team_share_backtester as bt
from src.evaluation.team_share_backtester import run_walk_forward_backtest
from src.models.team_allocation.features import ROLL_WINDOW, VOLUME_COLS

SEASONS = [2018, 2019, 2020, 2021, 2022]
WEEKS = [1, 2, 3, 4]


def _synthetic_rows() -> pd.DataFrame:
    rows = []
    rng = np.random.RandomState(0)
    for season in SEASONS:
        for week in WEEKS:
            for player, base in (("p1", 0.6), ("p2", 0.4)):
                row = {
                    "player_id": player, "season": season, "week": week,
                    "team": "AAA", "position": "WR",
                }
                for c in VOLUME_COLS:
                    share = base + rng.normal(scale=0.02)
                    row[c] = 5.0
                    row[f"team_{c}"] = 10.0
                    row[f"share_of_team_{c}"] = share
                    row[f"share_of_team_{c}_s2d"] = base
                    row[f"share_of_team_{c}_roll{ROLL_WINDOW}"] = base
                row["is_cold_start"] = 0
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def shares_db(monkeypatch):
    df = _synthetic_rows()
    monkeypatch.setattr(bt, "load_share_rows", lambda seasons=None: df)
    return df


def test_folds_never_let_test_season_leak_into_training(shares_db):
    report = run_walk_forward_backtest("targets", seasons=SEASONS, n_test_seasons=2)
    assert report["n_folds"] == 2
    for fold in report["folds"]:
        assert max(fold["train_seasons"]) < min(fold["test_seasons"])
        assert set(fold["train_seasons"]).isdisjoint(fold["test_seasons"])


def test_pooled_metrics_present_for_every_arm(shares_db):
    report = run_walk_forward_backtest("targets", seasons=SEASONS, n_test_seasons=2)
    assert set(report["pooled"]) == {"ridge", "xgboost", "rolling3"}
    for m in report["pooled"].values():
        assert m["mae"] >= 0
        assert m["n"] > 0


def test_rolling3_baseline_beats_nothing_but_is_well_formed(shares_db):
    """The synthetic data's rolling3 column is a near-perfect predictor by
    construction (same base value used for label and roll3), so this just
    checks the baseline arm runs and reports a small MAE -- not a real
    accuracy claim."""
    report = run_walk_forward_backtest("targets", seasons=SEASONS, n_test_seasons=2)
    assert report["pooled"]["rolling3"]["mae"] < 0.1


def test_strict_raises_when_too_few_seasons_for_requested_folds(shares_db):
    with pytest.raises(ValueError, match="strict=True"):
        run_walk_forward_backtest("targets", seasons=SEASONS, n_test_seasons=5)


def test_invalid_target_raises():
    with pytest.raises(ValueError, match="target must be one of"):
        run_walk_forward_backtest("not_a_real_target")

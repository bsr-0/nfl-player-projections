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
from src.evaluation.team_share_backtester import (
    _segment_metrics,
    bootstrap_mae_delta,
    run_walk_forward_backtest,
)
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


def _mixed_segment_rows() -> pd.DataFrame:
    """Same shape as _synthetic_rows but with two positions and both
    cold-start values represented, so segment-breakdown tests have more
    than one group to actually split on."""
    rows = []
    rng = np.random.RandomState(1)
    for season in SEASONS:
        for week in WEEKS:
            for player, base, position in (("p1", 0.6, "WR"), ("p2", 0.4, "RB")):
                row = {
                    "player_id": player, "season": season, "week": week,
                    "team": "AAA", "position": position,
                }
                for c in VOLUME_COLS:
                    share = base + rng.normal(scale=0.02)
                    row[c] = 5.0
                    row[f"team_{c}"] = 10.0
                    row[f"share_of_team_{c}"] = share
                    row[f"share_of_team_{c}_s2d"] = base
                    row[f"share_of_team_{c}_roll{ROLL_WINDOW}"] = base
                row["is_cold_start"] = int(week == 1)  # week 1 of every season is "cold"
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def mixed_shares(monkeypatch):
    df = _mixed_segment_rows()
    monkeypatch.setattr(bt, "load_share_rows", lambda seasons=None: df)
    return df


def test_pooled_includes_position_and_cold_start_breakdown(mixed_shares):
    report = run_walk_forward_backtest("rushing_attempts", seasons=SEASONS, n_test_seasons=2)
    for arm, m in report["pooled"].items():
        assert set(m["by_position"]) == {"WR", "RB"}
        assert set(m["by_cold_start"]) == {"0", "1"}
        # every segment's row count must add back up to the arm's total n
        assert sum(seg["n"] for seg in m["by_position"].values()) == m["n"]
        assert sum(seg["n"] for seg in m["by_cold_start"].values()) == m["n"]


def test_fold_level_arms_also_have_segment_breakdown(mixed_shares):
    report = run_walk_forward_backtest("rushing_attempts", seasons=SEASONS, n_test_seasons=2)
    for fold in report["folds"]:
        for arm, m in fold["arms"].items():
            assert "by_position" in m and "by_cold_start" in m


def test_pooled_tunable_arms_have_bootstrap_ci(mixed_shares):
    report = run_walk_forward_backtest("rushing_attempts", seasons=SEASONS, n_test_seasons=2)
    for arm in ("ridge", "xgboost"):
        boot = report["pooled"][arm]["vs_rolling3_bootstrap"]
        assert boot["ci_low"] <= boot["point_estimate"] <= boot["ci_high"]
        assert boot["n_bootstrap"] > 0
    assert "vs_rolling3_bootstrap" not in report["pooled"]["rolling3"]


def test_bootstrap_mae_delta_detects_a_strictly_better_candidate():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(0, 1, size=200)
    pred_baseline = y_true + rng.normal(scale=0.10, size=200)
    pred_candidate = y_true + rng.normal(scale=0.01, size=200)  # much closer to truth
    result = bootstrap_mae_delta(y_true, pred_candidate, pred_baseline, n_bootstrap=500, seed=1)
    assert result["point_estimate"] < 0
    assert result["ci_high"] < 0
    assert result["significant_improvement"] is True


def test_bootstrap_mae_delta_does_not_claim_significance_for_identical_predictions():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(0, 1, size=200)
    pred = y_true + rng.normal(scale=0.05, size=200)
    result = bootstrap_mae_delta(y_true, pred, pred, n_bootstrap=500, seed=1)
    assert result["point_estimate"] == pytest.approx(0.0)
    assert result["significant_improvement"] is False


def test_bootstrap_mae_delta_does_not_claim_significance_for_a_worse_candidate():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(0, 1, size=200)
    pred_baseline = y_true + rng.normal(scale=0.01, size=200)
    pred_candidate = y_true + rng.normal(scale=0.10, size=200)  # much worse
    result = bootstrap_mae_delta(y_true, pred_candidate, pred_baseline, n_bootstrap=500, seed=1)
    assert result["point_estimate"] > 0
    assert result["significant_improvement"] is False


def test_segment_metrics_splits_by_label_and_counts_add_up():
    y_true = np.array([0.1, 0.2, 0.3, 0.4])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array(["A", "A", "B", "B"])
    out = _segment_metrics(y_true, y_pred, labels, n_features=0)
    assert set(out) == {"A", "B"}
    assert out["A"]["n"] == 2
    assert out["B"]["n"] == 2
    assert out["A"]["mae"] == pytest.approx(0.0)

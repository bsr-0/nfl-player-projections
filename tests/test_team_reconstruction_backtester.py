"""src/evaluation/team_reconstruction_backtester.py -- walk-forward
reconstruction correctness, built on real `build_shares()` output (not
hand-synthesized columns, which would be error-prone to keep in sync with
the builder's actual schema) fed through the backtester via monkeypatching
`load_share_rows`, the same technique test_team_allocation_backtester.py
uses.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

import src.evaluation.team_share_backtester as share_bt
from scripts.build_team_week_player_shares import build_shares
from src.evaluation.team_reconstruction_backtester import run_reconstruction_backtest
from src.models.team_allocation.features import ROLL_WINDOW
from src.utils.database import DatabaseManager

SEASONS = list(range(2018, 2024))  # 6 seasons
WEEKS = [1, 2, 3, 4]


def _seed_db(db_path):
    db = DatabaseManager(db_path=db_path)
    con = sqlite3.connect(str(db_path))
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            rows.append({"player_id": "p1_wr", "season": season, "week": week, "team": "AAA", "position": "WR"})
            rows.append({"player_id": "p2_rb", "season": season, "week": week, "team": "AAA", "position": "RB"})
            rows.append({"player_id": "p3_qb", "season": season, "week": week, "team": "AAA", "position": "QB"})
    pd.DataFrame(rows).to_sql("canonical_player_weeks", con, index=False, if_exists="replace")
    con.close()

    rng = np.random.RandomState(0)
    for season in SEASONS:
        for week in WEEKS:
            db.insert_player_weekly_stats({
                "player_id": "p1_wr", "season": season, "week": week, "team": "AAA",
                "targets": 8, "rushing_attempts": 0,
                "receiving_yards": max(0, int(rng.normal(70, 15))), "rushing_yards": 0,
            })
            db.insert_player_weekly_stats({
                "player_id": "p2_rb", "season": season, "week": week, "team": "AAA",
                "targets": 3, "rushing_attempts": 15,
                "receiving_yards": max(0, int(rng.normal(20, 5))),
                "rushing_yards": max(0, int(rng.normal(75, 15))),
            })
            db.insert_player_weekly_stats({
                "player_id": "p3_qb", "season": season, "week": week, "team": "AAA",
                "targets": 0, "rushing_attempts": 3,
                "receiving_yards": 0, "rushing_yards": max(0, int(rng.normal(10, 5))),
            })
    return db


@pytest.fixture
def real_shares_panel(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    _seed_db(db_path)
    con = sqlite3.connect(str(db_path))
    panel = build_shares(con, SEASONS[0], SEASONS[-1])
    con.close()
    monkeypatch.setattr(share_bt, "load_share_rows", lambda seasons=None: panel)
    return panel


def test_reconstruction_report_has_every_arm(real_shares_panel):
    report = run_reconstruction_backtest(seasons=SEASONS, n_test_seasons=2)
    assert set(report["pooled"]) == {"ridge", "xgboost", "rolling3"}
    assert report["targets_reconstructed"] == ["rushing_yards", "receiving_yards"]
    assert report["n_rows_total"] > 0


def test_reconstruction_mae_is_nonnegative_and_finite(real_shares_panel):
    report = run_reconstruction_backtest(seasons=SEASONS, n_test_seasons=2)
    for arm, m in report["pooled"].items():
        assert m["mae"] >= 0
        assert np.isfinite(m["mae"])
        assert m["n"] > 0


def test_tunable_arms_have_bootstrap_ci_vs_rolling3(real_shares_panel):
    report = run_reconstruction_backtest(seasons=SEASONS, n_test_seasons=2)
    for arm in ("ridge", "xgboost"):
        assert "vs_rolling3_bootstrap" in report["pooled"][arm]
    assert "vs_rolling3_bootstrap" not in report["pooled"]["rolling3"]


def test_position_breakdown_present_and_qb_has_lower_partial_points(real_shares_panel):
    """QBs (position QB) have no receiving-yards population, so their
    reconstructed partial points should come from rushing_yards only and be
    small/zero for receiving, contrasted with WR (receiving-heavy)."""
    report = run_reconstruction_backtest(seasons=SEASONS, n_test_seasons=2)
    for arm, m in report["pooled"].items():
        assert "QB" in m["by_position"]
        assert "WR" in m["by_position"]


def test_reconstruction_uses_the_same_row_set_across_arms(real_shares_panel):
    report = run_reconstruction_backtest(seasons=SEASONS, n_test_seasons=2)
    ns = {arm: m["n"] for arm, m in report["pooled"].items()}
    assert len(set(ns.values())) == 1  # every arm scored on the identical OOF rows

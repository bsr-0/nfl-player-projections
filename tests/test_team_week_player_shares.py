"""Leakage-safety invariants for scripts/build_team_week_player_shares.py.

Mirrors tests/test_game_outcome_leakage.py's sentinel pattern: a value
planted in one week's player_weekly_stats must not appear in that same
week's OWN lagged share feature, only in a later week's.
"""
import sqlite3

import pandas as pd
import pytest

from scripts.build_team_week_player_shares import build_shares, validate_shares
from src.utils.database import DatabaseManager

SEASON = 2020


def _seed(db: DatabaseManager, n_weeks: int, sentinel_week: int = None, sentinel_targets: int = 900):
    """Two WRs (p1, p2) on team AAA sharing targets 6/4 each week, except
    p1 gets a sentinel spike in `sentinel_week`."""
    con = sqlite3.connect(str(db.db_path))
    rows = []
    for week in range(1, n_weeks + 1):
        rows.append({"player_id": "p1", "season": SEASON, "week": week, "team": "AAA", "position": "WR"})
        rows.append({"player_id": "p2", "season": SEASON, "week": week, "team": "AAA", "position": "WR"})
    pd.DataFrame(rows).to_sql("canonical_player_weeks", con, index=False, if_exists="replace")
    con.close()

    for week in range(1, n_weeks + 1):
        p1_targets = sentinel_targets if week == sentinel_week else 6
        db.insert_player_weekly_stats({
            "player_id": "p1", "season": SEASON, "week": week, "team": "AAA",
            "targets": p1_targets, "rushing_attempts": 0, "receiving_yards": 60, "rushing_yards": 0,
        })
        db.insert_player_weekly_stats({
            "player_id": "p2", "season": SEASON, "week": week, "team": "AAA",
            "targets": 4, "rushing_attempts": 0, "receiving_yards": 40, "rushing_yards": 0,
        })


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "test.db")


def _build(db):
    con = sqlite3.connect(str(db.db_path))
    try:
        return build_shares(con, SEASON, SEASON)
    finally:
        con.close()


def test_same_week_share_is_correct(db):
    _seed(db, n_weeks=4)
    panel = _build(db)
    row = panel[(panel.player_id == "p1") & (panel.week == 1)].iloc[0]
    assert row["share_of_team_targets"] == pytest.approx(0.6)


def test_shares_never_exceed_one_and_validate_passes(db):
    _seed(db, n_weeks=4)
    panel = _build(db)
    validate_shares(panel)


def test_sentinel_spike_does_not_leak_into_its_own_week(db):
    _seed(db, n_weeks=6, sentinel_week=3)
    panel = _build(db)
    panel = panel.sort_values("week").set_index("week")

    # Week 3's OWN same-week share reflects the spike (it's the label) --
    # but its LAGGED feature (computed from weeks < 3) must not.
    week3 = panel.loc[panel.index == 3]
    week3_p1 = week3[week3.player_id == "p1"].iloc[0]
    assert week3_p1["share_of_team_targets"] > 0.9  # same-week label: the spike itself
    assert week3_p1["share_of_team_targets_s2d"] == pytest.approx(0.6, abs=0.05)  # lag: weeks 1-2 only

    # Week 4 has weeks 1-3 as prior games, so ITS lagged feature must now
    # reflect the week-3 spike.
    week4_p1 = panel.loc[panel.index == 4]
    week4_p1 = week4_p1[week4_p1.player_id == "p1"].iloc[0]
    assert week4_p1["share_of_team_targets_s2d"] > 0.7


def test_duplicate_rows_are_rejected(db):
    _seed(db, n_weeks=2)
    panel = _build(db)
    dup = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_shares(dup)


def test_share_sum_over_one_is_rejected(db):
    """Each mutated value stays within [0, 1] individually, so this only
    trips the team-week sum check, not the per-row range check."""
    _seed(db, n_weeks=2)
    panel = _build(db)
    week1 = panel.index[panel["week"] == 1]
    panel.loc[week1, "share_of_team_targets"] = 0.7
    with pytest.raises(ValueError, match="sums to > 1"):
        validate_shares(panel)


def test_missing_canonical_table_raises(tmp_path):
    from scripts.build_team_week_player_shares import load_population
    con = sqlite3.connect(str(tmp_path / "empty.db"))
    try:
        with pytest.raises(ValueError, match="canonical_player_weeks table not found"):
            load_population(con, SEASON, SEASON)
    finally:
        con.close()

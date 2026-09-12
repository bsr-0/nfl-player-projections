"""Regression tests for the player_weekly_stats and schedule write paths.

Both used `INSERT OR REPLACE`, which deletes and recreates the row on a
conflict: any column not included in that INSERT's value list silently
reverts to its schema default. Two concrete failures this caused:

  - snap_count/snap_share/team_snaps: NaN ("no snap record") was coerced to
    0 before it ever reached the DB, and a re-ingest of a game already in
    player_weekly_stats reset those columns to 0 again even when a prior,
    real value was already stored. See tests/test_snap_null_semantics.py
    for the NULL-vs-zero distinction upstream in the aggregator.
  - spread_line/total_line: the routine nflverse schedule reload never
    supplies these, so every reload wiped out whatever backfill_vegas_lines.py
    had written.

Both write paths are now `INSERT ... ON CONFLICT DO UPDATE` with
COALESCE(excluded.col, table.col), so an absent/NULL incoming value keeps
what's already stored instead of erasing it.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "test.db")


def test_unknown_snaps_stay_null_through_reingest(db):
    db.insert_player({"player_id": "00-TEST1", "name": "Test Player", "position": "WR"})
    db.insert_player_weekly_stats({
        "player_id": "00-TEST1", "season": 2025, "week": 1, "team": "DEN",
        "snap_count": None, "snap_share": None, "team_snaps": None,
    })
    con = sqlite3.connect(db.db_path)
    row = con.execute(
        "SELECT snap_count FROM player_weekly_stats WHERE player_id='00-TEST1'"
    ).fetchone()
    assert row[0] is None

    # A later partial re-ingest that doesn't touch snaps must not fabricate 0.
    db.insert_player_weekly_stats({
        "player_id": "00-TEST1", "season": 2025, "week": 1, "team": "DEN",
        "passing_yards": 123,
    })
    row = con.execute(
        "SELECT snap_count, passing_yards FROM player_weekly_stats WHERE player_id='00-TEST1'"
    ).fetchone()
    assert row[0] is None
    assert row[1] == 123


def test_confirmed_zero_snaps_are_not_unknown(db):
    db.insert_player({"player_id": "00-TEST2", "name": "Benched", "position": "RB"})
    db.insert_player_weekly_stats({
        "player_id": "00-TEST2", "season": 2025, "week": 1, "team": "DEN",
        "snap_count": 0, "snap_share": 0.0, "team_snaps": 50,
    })
    con = sqlite3.connect(db.db_path)
    row = con.execute(
        "SELECT snap_count FROM player_weekly_stats WHERE player_id='00-TEST2'"
    ).fetchone()
    assert row[0] == 0


def test_reingest_does_not_wipe_prior_columns(db):
    """A re-ingest that omits a column must not blow away what's stored,
    the way INSERT OR REPLACE used to (e.g. data_source provenance)."""
    db.insert_player({"player_id": "00-TEST3", "name": "Test Player", "position": "QB"})
    db.insert_player_weekly_stats({
        "player_id": "00-TEST3", "season": 2025, "week": 1, "team": "KC",
        "passing_yards": 300, "data_source": "nflverse_stats",
    })
    db.insert_player_weekly_stats({
        "player_id": "00-TEST3", "season": 2025, "week": 1, "team": "KC",
        "rushing_yards": 10,
    })
    con = sqlite3.connect(db.db_path)
    row = con.execute(
        "SELECT passing_yards, rushing_yards, data_source "
        "FROM player_weekly_stats WHERE player_id='00-TEST3'"
    ).fetchone()
    assert row == (300, 10, "nflverse_stats")


def test_schedule_reload_preserves_vegas_lines(db):
    db.insert_schedule({
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BAL",
        "spread_line": -2.5, "total_line": 48.5,
    })
    # Routine schedule reload never carries Vegas lines.
    db.insert_schedule({
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BAL",
        "home_score": 24, "away_score": 20,
    })
    con = sqlite3.connect(db.db_path)
    row = con.execute(
        "SELECT spread_line, total_line, home_score, away_score "
        "FROM schedule WHERE season=2026 AND week=1"
    ).fetchone()
    assert row == (-2.5, 48.5, 24, 20)


def test_vegas_backfill_can_still_update_lines(db):
    db.insert_schedule({
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BAL",
        "spread_line": -2.5, "total_line": 48.5,
    })
    db.insert_schedule({
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BAL",
        "spread_line": -3.0, "total_line": 47.0,
    })
    con = sqlite3.connect(db.db_path)
    row = con.execute(
        "SELECT spread_line, total_line FROM schedule WHERE season=2026 AND week=1"
    ).fetchone()
    assert row == (-3.0, 47.0)

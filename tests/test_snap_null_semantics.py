"""snap_count / team_snaps / snap_share must distinguish zero from unknown.

The old code fillna(0)'d both columns, so "no snap record" and "took zero
snaps" became the same stored value -- the reason player_weekly_stats read
100% zero for 2006-2017 (GAPS.md 2026-08-19 audit). These pin the three
states apart.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.pbp_stats_aggregator import PBPStatsAggregator, compute_team_snaps


@pytest.fixture
def agg():
    return PBPStatsAggregator()


def _stats(rows):
    return pd.DataFrame(rows)


def _snaps(rows):
    return pd.DataFrame(rows)


def test_unmatched_row_gets_null_not_zero(agg):
    """The core regression: a player with no snap record is UNKNOWN."""
    agg.snap_data = _snaps([{
        "season": 2015, "week": 1, "team": "DEN", "player": "A Player",
        "player_id": "00-0000001", "pfr_player_id": None,
        "offense_snaps": 40, "offense_pct": 0.8,
    }])
    stats = _stats([
        {"player_id": "00-0000001", "name": "A Player", "season": 2015, "week": 1, "team": "DEN"},
        {"player_id": "00-0000002", "name": "Ghost", "season": 2015, "week": 1, "team": "DEN"},
    ])
    out = agg.merge_with_snaps(stats)
    ghost = out[out.player_id == "00-0000002"].iloc[0]

    assert pd.isna(ghost["snap_count"]), "unmatched row must be NULL, not 0"
    assert pd.isna(ghost["snap_share"])


def test_real_zero_snaps_stays_zero(agg):
    """A player the feed records at 0 snaps is a CONFIRMED zero and must
    survive as 0 -- not be confused with the unknown case."""
    agg.snap_data = _snaps([
        {"season": 2015, "week": 1, "team": "DEN", "player": "Starter",
         "player_id": "00-0000001", "pfr_player_id": None,
         "offense_snaps": 50, "offense_pct": 1.0},
        {"season": 2015, "week": 1, "team": "DEN", "player": "Benched",
         "player_id": "00-0000002", "pfr_player_id": None,
         "offense_snaps": 0, "offense_pct": 0.0},
    ])
    stats = _stats([
        {"player_id": "00-0000002", "name": "Benched", "season": 2015, "week": 1, "team": "DEN"},
    ])
    out = agg.merge_with_snaps(stats).iloc[0]

    assert out["snap_count"] == 0
    assert not pd.isna(out["snap_count"])
    assert out["snap_share"] == 0.0


def test_positive_snaps_pass_through(agg):
    agg.snap_data = _snaps([{
        "season": 2015, "week": 1, "team": "DEN", "player": "Starter",
        "player_id": "00-0000001", "pfr_player_id": None,
        "offense_snaps": 40, "offense_pct": 0.8,
    }])
    stats = _stats([
        {"player_id": "00-0000001", "name": "Starter", "season": 2015, "week": 1, "team": "DEN"},
    ])
    out = agg.merge_with_snaps(stats).iloc[0]

    assert out["snap_count"] == 40
    assert out["snap_share"] == pytest.approx(40 / 50)


def test_snap_columns_are_nullable_dtype(agg):
    """A plain int64 column cannot hold NA, so the dtype is load-bearing."""
    agg.snap_data = _snaps([{
        "season": 2015, "week": 1, "team": "DEN", "player": "Starter",
        "player_id": "00-0000001", "pfr_player_id": None,
        "offense_snaps": 40, "offense_pct": 0.8,
    }])
    stats = _stats([
        {"player_id": "00-0000009", "name": "Ghost", "season": 2015, "week": 1, "team": "DEN"},
    ])
    out = agg.merge_with_snaps(stats)

    assert out["snap_count"].dtype == "Int64"
    assert out["team_snaps"].dtype == "Int64"


def test_team_snaps_without_usable_rows_is_null_not_zero():
    """team_snaps == 0 is impossible -- a team always runs plays."""
    empty = pd.DataFrame(columns=["season", "week", "team", "offense_snaps", "offense_pct"])
    out = compute_team_snaps(empty)

    assert out.empty or out["team_snaps"].isna().all()


def test_team_snaps_never_silently_zero_for_a_real_team():
    snaps = pd.DataFrame([
        {"season": 2015, "week": 1, "team": "DEN", "offense_snaps": 50, "offense_pct": 1.0},
        {"season": 2015, "week": 1, "team": "DEN", "offense_snaps": 25, "offense_pct": 0.5},
    ])
    out = compute_team_snaps(snaps)

    assert (out["team_snaps"] > 0).all()
    assert out["team_snaps"].iloc[0] == 50


def test_null_survives_a_sqlite_roundtrip(agg, tmp_path):
    """The whole point is that NULL reaches the database."""
    import sqlite3

    agg.snap_data = _snaps([{
        "season": 2015, "week": 1, "team": "DEN", "player": "Starter",
        "player_id": "00-0000001", "pfr_player_id": None,
        "offense_snaps": 40, "offense_pct": 0.8,
    }])
    stats = _stats([
        {"player_id": "00-0000001", "name": "Starter", "season": 2015, "week": 1, "team": "DEN"},
        {"player_id": "00-0000009", "name": "Ghost", "season": 2015, "week": 1, "team": "DEN"},
    ])
    out = agg.merge_with_snaps(stats)[["player_id", "snap_count", "team_snaps"]]

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    out.to_sql("t", conn, index=False)
    nulls = conn.execute(
        "SELECT COUNT(*) FROM t WHERE snap_count IS NULL").fetchone()[0]
    zeros = conn.execute(
        "SELECT COUNT(*) FROM t WHERE snap_count = 0").fetchone()[0]
    conn.close()

    assert nulls == 1, "unknown row must persist as SQL NULL"
    assert zeros == 0

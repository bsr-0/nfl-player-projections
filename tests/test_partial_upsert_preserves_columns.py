"""A partial upsert must not destroy columns it does not mention.

`insert_team_stats` writes every column through
`COALESCE(excluded.<col>, team_stats.<col>)`, which reads as "keep the stored
value when the incoming one is absent". The parameter binding used
`stats.get(col, 0)`, which made that guarantee false -- the incoming value was
never absent, it was 0, and COALESCE(0, x) is 0.

So any caller passing a partial dict silently zeroed points_scored,
total_yards, turnovers and the rest on every row it touched. Found 2026-08-29
while writing the PBP backfill, which is exactly such a caller; the backfill
was routed around the upsert because of it.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.utils.database import DatabaseManager


FULL_ROW = dict(
    team="KC", season=2021, week=3, opponent="LAC", home_away="home",
    points_scored=30, points_allowed=24, total_yards=400, passing_yards=300,
    rushing_yards=100, turnovers=1, total_plays=65, pace_sec_per_play=28.5,
)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "t.db")


def _row(db):
    con = sqlite3.connect(db.db_path)
    cur = con.execute("SELECT * FROM team_stats")
    out = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    con.close()
    return out


def test_partial_upsert_preserves_unmentioned_columns(db):
    db.insert_team_stats(FULL_ROW)
    # The backfill scenario: only PBP-derived columns supplied.
    db.insert_team_stats(dict(team="KC", season=2021, week=3,
                              pace_sec_per_play=31.2, drive_count=11))
    row = _row(db)

    assert row["points_scored"] == 30, "partial upsert zeroed points_scored"
    assert row["total_yards"] == 400, "partial upsert zeroed total_yards"
    assert row["turnovers"] == 1, "partial upsert zeroed turnovers"
    assert row["passing_yards"] == 300, "partial upsert zeroed passing_yards"


def test_partial_upsert_still_applies_supplied_values(db):
    db.insert_team_stats(FULL_ROW)
    db.insert_team_stats(dict(team="KC", season=2021, week=3,
                              pace_sec_per_play=31.2, drive_count=11))
    row = _row(db)

    assert row["pace_sec_per_play"] == pytest.approx(31.2)
    assert row["drive_count"] == 11


def test_absent_pbp_columns_are_null_not_zero(db):
    """A column that was never computed must read as NULL.

    Storing 0 makes "not computed" indistinguishable from "measured zero",
    which is how eleven PBP columns sat at exactly 0.0 for 2006-2024 while
    passing every IS NOT NULL audit at 100%.
    """
    db.insert_team_stats(dict(team="SF", season=2019, week=1,
                              points_scored=20))
    row = _row(db)

    for col in ("pace_sec_per_play", "neutral_pass_rate_oe", "drive_count",
                "avg_drive_epa", "points_per_drive"):
        assert row[col] is None, (
            f"{col} was written as {row[col]!r} rather than NULL; a fabricated "
            f"zero is indistinguishable from a measurement"
        )

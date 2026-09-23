"""DataQualityGates must not mistake an in-progress week's trickle of stats
for a completed week's coverage.

The bug (found 2026-09-22 investigating why a retrain was blocked): a
Thursday game's box score lands in player_weekly_stats days before the rest
of that week's games, so the literal MAX(season, week) in the stats table
can be a single-game trickle. Comparing that against a full-week baseline
fails completeness/anomaly checks that have nothing wrong with them -- the
same class of bug nfl_calendar.season_has_completed_games() was already
written to fix at season granularity (GAPS.md 2026-09-03), recurring here at
week granularity in a function that never got that fix applied to it.
"""
import sqlite3

import pandas as pd
import pytest

from src.data.quality_gates import DataQualityGates
from src.utils.database import DatabaseManager

POSITIONS = ["QB", "RB", "WR", "TE"]


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(tmp_path / "t.db")


def _schedule_game(db, season, week, home, away, played):
    db.insert_schedule({
        "season": season, "week": week, "home_team": home, "away_team": away,
        "home_score": 24 if played else None, "away_score": 17 if played else None,
    })


def _full_week_rows(season, week, teams, n_players_per_team=12):
    rows = []
    for team in teams:
        for i in range(n_players_per_team):
            rows.append({
                "player_id": f"{team}_{i}", "season": season, "week": week,
                "team": team, "position": POSITIONS[i % 4],
                "fantasy_points": 5.0,
            })
    return rows


def test_in_progress_week_does_not_fail_completeness_or_anomalies(db):
    """Week 1: 4 teams, fully played, complete stats. Week 2: only the
    Thursday game (2 of the 4 teams) has been played and has stats; the
    other 2 teams' games (and therefore their stats) don't exist yet. The
    gate must validate against week 1, not treat week 2's 2-team trickle as
    a coverage collapse."""
    teams = ["AAA", "BBB", "CCC", "DDD"]
    _schedule_game(db, 2026, 1, "AAA", "BBB", played=True)
    _schedule_game(db, 2026, 1, "CCC", "DDD", played=True)
    _schedule_game(db, 2026, 2, "AAA", "BBB", played=True)   # Thursday game: done
    _schedule_game(db, 2026, 2, "CCC", "DDD", played=False)  # hasn't kicked off yet

    rows = _full_week_rows(2026, 1, teams) + _full_week_rows(2026, 2, ["AAA", "BBB"])
    df = pd.DataFrame(rows)

    gates = DataQualityGates(expected_positions=POSITIONS)
    result = gates.evaluate(df, db_path=db.db_path, check_freshness=False)

    assert result.passed, result.report
    assert result.report["checks"]["completeness"]["latest_window"] == {"season": 2026, "week": 1}
    assert result.report["checks"]["anomalies"]["passed"]
    # Raw observed state is still reported honestly, unaffected by the fix.
    assert result.report["latest_observed"] == {"season": 2026, "week": 2, "rows": len(df)}


def test_fully_played_latest_week_is_used_as_is(db):
    """No trailing in-progress window -- the fix must be a no-op here."""
    teams = ["AAA", "BBB", "CCC", "DDD"]
    _schedule_game(db, 2026, 1, "AAA", "BBB", played=True)
    _schedule_game(db, 2026, 1, "CCC", "DDD", played=True)
    _schedule_game(db, 2026, 2, "AAA", "BBB", played=True)
    _schedule_game(db, 2026, 2, "CCC", "DDD", played=True)

    rows = _full_week_rows(2026, 1, teams) + _full_week_rows(2026, 2, teams)
    df = pd.DataFrame(rows)

    gates = DataQualityGates(expected_positions=POSITIONS)
    result = gates.evaluate(df, db_path=db.db_path, check_freshness=False)

    assert result.passed, result.report
    assert result.report["checks"]["completeness"]["latest_window"] == {"season": 2026, "week": 2}


def test_missing_schedule_table_fails_open(tmp_path):
    """No schedule table at all (or unreadable) must not block the gate --
    same fail-open convention _load_scheduled_teams already uses."""
    teams = ["AAA", "BBB"]
    rows = _full_week_rows(2026, 1, teams)
    df = pd.DataFrame(rows)
    empty_db = tmp_path / "empty.db"
    sqlite3.connect(str(empty_db)).close()

    gates = DataQualityGates(expected_positions=POSITIONS)
    result = gates.evaluate(df, db_path=empty_db, check_freshness=False)

    assert result.report["checks"]["completeness"]["latest_window"] == {"season": 2026, "week": 1}

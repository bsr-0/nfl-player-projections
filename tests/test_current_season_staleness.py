"""Partial current-season weeks must not be served from a cache or counted as loaded."""
import sqlite3
from contextlib import contextmanager

import pandas as pd

from src.data.auto_refresh import newest_complete_week
from src.data.pbp_stats_aggregator import stale_cache_reason

SCHEDULED = {4: 32, 5: 30}


def _cache(rows):
    return pd.DataFrame(rows, columns=["week", "team"])


def _full(week, n):
    return [(week, f"T{i:02d}") for i in range(n)]


def test_a_complete_newest_week_at_the_calendar_week_is_fresh():
    assert stale_cache_reason(_cache(_full(4, 32) + _full(5, 30)), 5, SCHEDULED) is None


def test_a_cache_behind_the_calendar_is_stale():
    assert "current week 5" in stale_cache_reason(_cache(_full(4, 32)), 5, SCHEDULED)


def test_a_partly_played_newest_week_is_stale_even_at_the_calendar_week():
    reason = stale_cache_reason(_cache(_full(4, 32) + _full(5, 2)), 5, SCHEDULED)
    assert reason == "week 5 has 2 of 30 scheduled teams"


def test_without_a_schedule_only_the_calendar_rule_applies():
    assert stale_cache_reason(_cache(_full(4, 32) + _full(5, 2)), 5, {}) is None


def test_a_team_cache_stuck_at_week_one_is_stale():
    assert stale_cache_reason(_cache(_full(1, 32)), 5, SCHEDULED) == "cached week 1, current week 5"


class _DB:
    def __init__(self, con):
        self.con = con

    def get_latest_week_for_season(self, season):
        return self.con.execute("SELECT MAX(week) FROM player_weekly_stats WHERE season = ?", (season,)).fetchone()[0]

    @contextmanager
    def _get_connection(self):
        yield self.con


def _db(week5_teams, with_schedule=True):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE player_weekly_stats (season INT, week INT, team TEXT)")
    rows = [(2026, 4, f"T{i:02d}") for i in range(32)] + [(2026, 5, f"T{i:02d}") for i in range(week5_teams)]
    con.executemany("INSERT INTO player_weekly_stats VALUES (?, ?, ?)", rows)
    if with_schedule:
        con.execute("CREATE TABLE schedule (season INT, week INT, home_team TEXT, away_team TEXT)")
        con.executemany("INSERT INTO schedule VALUES (2026, ?, ?, ?)",
                        [(4, f"T{2 * i:02d}", f"T{2 * i + 1:02d}") for i in range(16)]
                        + [(5, f"T{2 * i:02d}", f"T{2 * i + 1:02d}") for i in range(15)])
    return _DB(con)


def test_a_partly_loaded_newest_week_does_not_count_as_loaded():
    assert newest_complete_week(_db(week5_teams=2), 2026) == 4


def test_a_fully_loaded_week_counts():
    assert newest_complete_week(_db(week5_teams=30), 2026) == 5


def test_without_a_schedule_the_stored_max_week_is_used():
    assert newest_complete_week(_db(week5_teams=2, with_schedule=False), 2026) == 5

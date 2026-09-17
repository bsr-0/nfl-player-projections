"""Historical odds snapshots must be taken BEFORE kickoff.

The scraper used to snapshot every event at the game date + 17:00 UTC. That
is 1pm EDT -- the standard early-window kickoff itself, not before it -- and
for London games (9:30am ET) it landed 3+ hours into the game. 6 events in
player_props_odds carry lines fetched at or after commence_time as a result.

Two guards: the scraper derives the snapshot from each event's own kickoff,
and compute_market_projections refuses rows fetched at/after kickoff so the
already-captured bad rows stay out of the market projection until re-fetched.
"""
import sqlite3

import pytest

from src.scrapers.odds_scraper import (
    PRE_KICKOFF_MARGIN_MINUTES,
    SNAPSHOT_TIME_SUFFIX,
    pre_kickoff_snapshot,
)


class TestPreKickoffSnapshot:
    def test_standard_early_window_is_before_kickoff(self):
        assert pre_kickoff_snapshot("2024-09-08T17:00:00Z") == "2024-09-08T16:30:00Z"

    def test_london_game_is_before_kickoff_not_three_hours_in(self):
        # 9:30am ET kickoff. The old fixed 17:00Z would be mid-4th-quarter.
        snap = pre_kickoff_snapshot("2024-10-06T13:30:00Z")
        assert snap == "2024-10-06T13:00:00Z"
        assert snap < "2024-10-06T13:30:00Z"

    def test_margin_is_applied(self):
        assert pre_kickoff_snapshot("2024-09-08T17:00:00Z", margin_minutes=5) == "2024-09-08T16:55:00Z"
        assert PRE_KICKOFF_MARGIN_MINUTES > 0

    def test_offset_aware_input_is_normalised_to_utc(self):
        assert pre_kickoff_snapshot("2024-09-08T13:00:00-04:00") == "2024-09-08T16:30:00Z"

    @pytest.mark.parametrize("bad", [None, "", "garbage", "2024-13-45T00:00:00Z"])
    def test_unparseable_returns_none_so_caller_can_fall_back(self, bad):
        assert pre_kickoff_snapshot(bad) is None

    def test_fallback_suffix_precedes_every_us_kickoff(self):
        # Earliest regular US window is 1pm ET = 17:00Z. Keep the fallback
        # clear of that so the day-level path is also pre-game.
        assert SNAPSHOT_TIME_SUFFIX < "T17:00:00Z"


class TestMarketProjectionGuard:
    def test_rows_fetched_at_or_after_kickoff_are_excluded(self, monkeypatch, tmp_path):
        import scripts.compute_market_projections as cmp

        db = tmp_path / "t.db"
        with sqlite3.connect(db) as con:
            con.execute("""
                CREATE TABLE player_props_odds (
                    player_name TEXT, market TEXT, season INT, week INT,
                    description TEXT, point REAL,
                    fetched_at TEXT, commence_time TEXT)
            """)
            kick = "2025-09-07T17:00:00Z"
            rows = [
                ("A", "player_rush_yds", 2025, 1, "Over", 60.5, "2025-09-07T16:30:00Z", kick),
                ("A", "player_rush_yds", 2025, 1, "Over", 99.5, kick, kick),               # at kickoff
                ("A", "player_rush_yds", 2025, 1, "Over", 99.5, "2025-09-07T19:00:00Z", kick),  # in-game
            ]
            con.executemany("INSERT INTO player_props_odds VALUES (?,?,?,?,?,?,?,?)", rows)
        monkeypatch.setattr(cmp, "DB_PATH", db)

        props = cmp._load_props(2025)
        assert props[("A", "player_rush_yds", 1)] == [60.5]

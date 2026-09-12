"""Regression test for NFL season-start calculation.

Pins _season_start against actual historical kickoff dates so the
Labor-day-vs-first-Thursday-of-September bug (2026 kickoff computed as
Sept 3 instead of the real Sept 10) can't silently come back.
"""
from datetime import datetime

from src.utils.nfl_calendar import _season_start, get_current_nfl_week

# Actual NFL Week 1 Thursday kickoff dates.
KNOWN_KICKOFFS = {
    2023: datetime(2023, 9, 7),
    2024: datetime(2024, 9, 5),
    2025: datetime(2025, 9, 4),
    2026: datetime(2026, 9, 10),
}


def test_season_start_matches_known_kickoffs():
    for season, kickoff in KNOWN_KICKOFFS.items():
        assert _season_start(season) == kickoff


def test_week1_2026_not_reached_until_kickoff():
    assert get_current_nfl_week(datetime(2026, 9, 9))["week_num"] == 0
    assert get_current_nfl_week(datetime(2026, 9, 10))["week_num"] == 1

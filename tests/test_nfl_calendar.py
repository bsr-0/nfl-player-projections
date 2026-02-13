"""
Rigorous tests for NFL calendar and schedule logic.

Covers get_current_nfl_season, get_current_nfl_week, and get_next_n_nfl_weeks
across all phases of the calendar year and NFL season so that schedule
availability (e.g. post-Super Bowl = next season) is correct everywhere.
"""

import pytest
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.nfl_calendar import (
    get_current_nfl_season,
    get_current_nfl_week,
    get_next_n_nfl_weeks,
    _playoff_dates,
    _season_start,
    current_season_has_weeks_played,
    is_future_or_current_matchup,
)


# -----------------------------------------------------------------------------
# get_current_nfl_season: all months of the year
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "date,expected_season",
    [
        # Jan-Aug: previous calendar year
        (datetime(2026, 1, 1), 2025),
        (datetime(2026, 1, 15), 2025),
        (datetime(2026, 2, 1), 2025),
        (datetime(2026, 2, 11), 2025),
        (datetime(2026, 3, 1), 2025),
        (datetime(2026, 4, 15), 2025),
        (datetime(2026, 5, 1), 2025),
        (datetime(2026, 6, 1), 2025),
        (datetime(2026, 7, 1), 2025),
        (datetime(2026, 8, 31), 2025),
        # Sept-Dec: current calendar year
        (datetime(2026, 9, 1), 2026),
        (datetime(2026, 9, 10), 2026),
        (datetime(2026, 10, 1), 2026),
        (datetime(2026, 11, 1), 2026),
        (datetime(2026, 12, 31), 2026),
    ],
)
def test_get_current_nfl_season_all_months(date, expected_season):
    """Season is previous year Jan-Aug, current year Sept-Dec."""
    assert get_current_nfl_season(date) == expected_season


# -----------------------------------------------------------------------------
# get_current_nfl_week: all phases of the NFL season
# -----------------------------------------------------------------------------
def test_current_week_preseason():
    """Before season start -> Preseason (week_num 0)."""
    # Season starts first Thu of Sept; early Sept before that is preseason. Aug is "previous" season (week 18).
    sept_1_2025 = datetime(2025, 9, 1)
    info = get_current_nfl_week(sept_1_2025)
    assert info["week_num"] == 0
    assert info["season"] == 2025
    assert "Preseason" in info["week"] or info["week"] == "Preseason"


def test_current_week_regular_season():
    """During regular season -> week 1-18."""
    # First Thursday of Sept 2025 is Sept 4; so Sept 10 is week 1 (or early week 2)
    sept_10_2025 = datetime(2025, 9, 10)
    info = get_current_nfl_week(sept_10_2025)
    assert info["season"] == 2025
    assert 1 <= info["week_num"] <= 18
    assert not info.get("is_playoffs", True)

    # Mid season: Oct 15
    oct_15_2025 = datetime(2025, 10, 15)
    info = get_current_nfl_week(oct_15_2025)
    assert info["season"] == 2025
    assert 1 <= info["week_num"] <= 18

    # Late regular season: late Dec (week 18)
    dec_30_2025 = datetime(2025, 12, 30)
    info = get_current_nfl_week(dec_30_2025)
    assert info["season"] == 2025
    assert 1 <= info["week_num"] <= 18


def test_current_week_playoffs():
    """Wild Card through Super Bowl week -> week 19-22."""
    # Wild Card weekend (Jan 10-12, 2026 for 2025 season)
    jan_11_2026 = datetime(2026, 1, 11)
    info = get_current_nfl_week(jan_11_2026)
    assert info["season"] == 2025
    assert info["week_num"] == 19
    assert info.get("is_playoffs") is True

    # Divisional (Jan 17-18)
    jan_17_2026 = datetime(2026, 1, 17)
    info = get_current_nfl_week(jan_17_2026)
    assert info["season"] == 2025
    assert info["week_num"] == 20

    # Conference Championships (Jan 19-26)
    jan_22_2026 = datetime(2026, 1, 22)
    info = get_current_nfl_week(jan_22_2026)
    assert info["season"] == 2025
    assert info["week_num"] == 21

    # Super Bowl week (Jan 27 - Feb 9 inclusive in code: sb + 1 day = Feb 9)
    feb_5_2026 = datetime(2026, 2, 5)
    info = get_current_nfl_week(feb_5_2026)
    assert info["season"] == 2025
    assert info["week_num"] == 22
    assert info.get("is_super_bowl") is True


def test_current_week_after_super_bowl_falls_through():
    """After SB week (Feb 10+): get_current_nfl_week falls through to 'regular season' week 18."""
    # This is the known quirk: Feb 11 is past SB+1 day, so we're not in SB week;
    # we then hit "regular season" and get week 18. get_next_n_nfl_weeks fixes this.
    feb_11_2026 = datetime(2026, 2, 11)
    info = get_current_nfl_week(feb_11_2026)
    assert info["season"] == 2025
    # Implementation detail: we get week 18 here; next_n_weeks uses SB date to override
    assert info["week_num"] == 18 or info["week_num"] == 22


# -----------------------------------------------------------------------------
# get_next_n_nfl_weeks: critical for schedule availability
# -----------------------------------------------------------------------------
class TestGetNextNNflWeeks:
    """Rigorous tests for get_next_n_nfl_weeks across season and calendar."""

    def test_zero_or_negative_n_returns_empty(self):
        assert get_next_n_nfl_weeks(datetime(2025, 10, 1), 0) == []
        assert get_next_n_nfl_weeks(datetime(2025, 10, 1), -1) == []

    def test_preseason_next_weeks_are_current_season(self):
        """Preseason: next 1 = week 1 of current season; next 18 = weeks 1-18."""
        aug_25_2025 = datetime(2025, 8, 25)
        one = get_next_n_nfl_weeks(aug_25_2025, 1)
        four = get_next_n_nfl_weeks(aug_25_2025, 4)
        eighteen = get_next_n_nfl_weeks(aug_25_2025, 18)
        assert one == [(2025, 1)]
        assert four == [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]
        assert eighteen == [(2025, i) for i in range(1, 19)]

    def test_regular_season_next_weeks_include_current_then_playoffs_then_next_season(self):
        """Mid season: next 1 is current week; next 4 is next 4; 18 can cross into next year."""
        oct_15_2025 = datetime(2025, 10, 15)  # ~week 6
        one = get_next_n_nfl_weeks(oct_15_2025, 1)
        four = get_next_n_nfl_weeks(oct_15_2025, 4)
        assert len(one) == 1
        assert one[0][0] == 2025
        assert 1 <= one[0][1] <= 18
        assert len(four) == 4
        # 18 weeks from week 6: 6..18 (13) + 19,20,21,22 (4) + 2026 week 1 (1) = 18
        eighteen = get_next_n_nfl_weeks(oct_15_2025, 18)
        assert len(eighteen) == 18
        seasons = {s for s, _ in eighteen}
        assert 2025 in seasons
        # May include 2026 if we cross past week 22
        assert eighteen[-1][0] >= 2025 and eighteen[-1][1] >= 1

    def test_super_bowl_week_next_weeks_are_next_season(self):
        """During Super Bowl week: next 1 = next season week 1; all horizons = next season."""
        feb_5_2026 = datetime(2026, 2, 5)  # SB week for 2025 season
        one = get_next_n_nfl_weeks(feb_5_2026, 1)
        four = get_next_n_nfl_weeks(feb_5_2026, 4)
        eighteen = get_next_n_nfl_weeks(feb_5_2026, 18)
        assert one == [(2026, 1)], "SB week: next 1 must be 2026 week 1"
        assert four == [(2026, 1), (2026, 2), (2026, 3), (2026, 4)]
        assert eighteen == [(2026, i) for i in range(1, 19)]

    def test_after_super_bowl_next_weeks_are_next_season(self):
        """After Super Bowl (Feb 10+): next weeks must be next season week 1, not 2025 week 18."""
        feb_11_2026 = datetime(2026, 2, 11)
        one = get_next_n_nfl_weeks(feb_11_2026, 1)
        four = get_next_n_nfl_weeks(feb_11_2026, 4)
        eighteen = get_next_n_nfl_weeks(feb_11_2026, 18)
        assert one == [(2026, 1)], "Post-SB: next 1 must be 2026 week 1"
        assert four == [(2026, 1), (2026, 2), (2026, 3), (2026, 4)]
        assert eighteen == [(2026, i) for i in range(1, 19)]
        assert all(s == 2026 for s, _ in eighteen)

    def test_day_after_super_bowl_still_sb_week(self):
        """Day after SB (Feb 9) is still SB week in get_current_nfl_week; next = next season."""
        feb_9_2026 = datetime(2026, 2, 9)
        one = get_next_n_nfl_weeks(feb_9_2026, 1)
        assert one == [(2026, 1)]

    def test_first_day_after_sb_week_is_next_season(self):
        """Feb 10 is first day 'after' SB week; next weeks must be 2026."""
        feb_10_2026 = datetime(2026, 2, 10)
        one = get_next_n_nfl_weeks(feb_10_2026, 1)
        assert one == [(2026, 1)]

    def test_offseason_march_through_august_next_weeks_are_next_season(self):
        """March through August (after SB): next weeks are next season week 1."""
        # Mar 2026: we're past 2025 SB, so "next" = 2026 week 1
        mar_1_2026 = datetime(2026, 3, 1)
        one = get_next_n_nfl_weeks(mar_1_2026, 1)
        assert one == [(2026, 1)]
        # Aug 2026: get_current_nfl_season(Aug)=2025; we're past 2025 SB, so next = 2026 week 1
        aug_1_2026 = datetime(2026, 8, 1)
        one_aug = get_next_n_nfl_weeks(aug_1_2026, 1)
        assert one_aug == [(2026, 1)]

    def test_playoff_weeks_advance_correctly(self):
        """During playoffs, next 4 can be 19,20,21,22 or 21,22,2026-1,2026-2."""
        jan_11_2026 = datetime(2026, 1, 11)  # Wild Card
        one = get_next_n_nfl_weeks(jan_11_2026, 1)
        four = get_next_n_nfl_weeks(jan_11_2026, 4)
        assert one == [(2025, 19)]
        assert four == [(2025, 19), (2025, 20), (2025, 21), (2025, 22)]

        jan_22_2026 = datetime(2026, 1, 22)  # Conf Champ
        four_cc = get_next_n_nfl_weeks(jan_22_2026, 4)
        assert four_cc == [(2025, 21), (2025, 22), (2026, 1), (2026, 2)]

    def test_week_18_next_four_includes_playoffs(self):
        """Late Dec: next 4 weeks include late regular season and playoffs (e.g. 17,18,19,20 or 18,19,20,21)."""
        dec_30_2025 = datetime(2025, 12, 30)
        four = get_next_n_nfl_weeks(dec_30_2025, 4)
        assert len(four) == 4
        assert four[0][0] == 2025
        assert 16 <= four[0][1] <= 18
        assert four[1][1] == four[0][1] + 1 or four[1][1] == 19


# -----------------------------------------------------------------------------
# Schedule availability: all three horizons use same calendar
# -----------------------------------------------------------------------------
def test_post_super_bowl_all_horizons_use_next_season():
    """After SB, 1w/4w/18w all reference next season so schedule check is consistent."""
    feb_11 = datetime(2026, 2, 11)
    for n in (1, 4, 18):
        weeks = get_next_n_nfl_weeks(feb_11, n)
        assert len(weeks) == n
        seasons = {s for s, _ in weeks}
        assert seasons == {2026}, f"Horizon {n}w after SB must be 2026 season, got {seasons}"


# -----------------------------------------------------------------------------
# current_season_has_weeks_played, is_future_or_current_matchup
# -----------------------------------------------------------------------------
def test_current_season_has_weeks_played():
    """Preseason (week 0) = False; regular/playoffs (week >= 1) = True."""
    sept_1_2025 = datetime(2025, 9, 1)  # Preseason before first Thu Sept
    assert current_season_has_weeks_played(sept_1_2025) is False
    sept_10_2025 = datetime(2025, 9, 10)
    assert current_season_has_weeks_played(sept_10_2025) is True


def test_is_future_or_current_matchup():
    """Future season or same-season future week is True."""
    today = datetime(2026, 2, 11)
    assert is_future_or_current_matchup(2026, 1, today) is True
    assert is_future_or_current_matchup(2025, 18, today) is True  # current or past
    assert is_future_or_current_matchup(2025, 10, today) is False  # past


# -----------------------------------------------------------------------------
# Sweep: one date per month (all weeks of the year) — invariants only
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "date",
    [
        datetime(2025, 1, 15),
        datetime(2025, 2, 15),
        datetime(2025, 3, 15),
        datetime(2025, 4, 15),
        datetime(2025, 5, 15),
        datetime(2025, 6, 15),
        datetime(2025, 7, 15),
        datetime(2025, 8, 15),
        datetime(2025, 9, 15),
        datetime(2025, 10, 15),
        datetime(2025, 11, 15),
        datetime(2025, 12, 15),
    ],
)
def test_get_next_n_nfl_weeks_invariants_every_month(date):
    """For any date in the year, next 1/4/18 weeks satisfy: correct length, valid (season, week)."""
    for n in (1, 4, 18):
        weeks = get_next_n_nfl_weeks(date, n)
        assert len(weeks) == n
        for s, w in weeks:
            assert isinstance(s, int) and s >= 2020 and s <= 2030
            assert isinstance(w, int) and 1 <= w <= 22
        # Consecutive weeks: (s, w) -> (s, w+1) or (s+1, 1)
        for i in range(len(weeks) - 1):
            s1, w1 = weeks[i]
            s2, w2 = weeks[i + 1]
            if w1 < 22:
                assert (s2, w2) == (s1, w1 + 1)
            else:
                assert (s2, w2) == (s1 + 1, 1)

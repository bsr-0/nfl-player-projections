"""A player traded mid-season must be projected against the schedule they
were actually on.

Callers used to pin the whole season to the player's FIRST team
(`g.sort_values("week")["team"].iloc[0]`), so after a trade the projection
used the old team's byes and opponents. Two real QBs surfaced this by
producing 13 real + 5 synthetic = 18 weeks against a 17-week schedule:
00-0033949/2023 (ARI->MIN, played ARI's week-14 bye for MIN) and
00-0026158/2025 (CLE->CIN, played CLE's week-9 bye for CIN). See GAPS.md.

Schedules are stubbed -- no DB.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.season_projection import (
    possible_weeks_for_player, REGULAR_SEASON_MAX_WEEK,
)


def _db(schedules):
    """schedules: {team: [weeks]} -> a db whose get_schedule honours it."""
    db = MagicMock()

    def get_schedule(season=None, team=None):
        weeks = schedules.get(team)
        if weeks is None:
            return pd.DataFrame()
        return pd.DataFrame({"week": weeks})

    db.get_schedule.side_effect = get_schedule
    return db


ALL = list(range(1, REGULAR_SEASON_MAX_WEEK + 1))


def _minus(*byes):
    return [w for w in ALL if w not in byes]


class TestTradedPlayerSchedule:
    def test_bye_weeks_follow_the_player_not_the_first_team(self):
        """ARI bye week 14, MIN bye week 13. A player who moves ARI->MIN at
        week 13 should be unplayable in ARI's week 13 bye... no: they are on
        MIN by then, so MIN's bye (13) applies and ARI's (14) does not."""
        db = _db({"ARI": _minus(14), "MIN": _minus(13)})
        real = {w: "ARI" for w in range(1, 13)}
        real[14] = "MIN"          # first game with the new team

        weeks, team_by_week = possible_weeks_for_player(db, real, 2023)

        assert team_by_week[14] == "MIN"
        assert team_by_week[15] == "MIN", "carried forward after the trade"
        assert team_by_week[1] == "ARI"
        assert 14 in weeks, "MIN plays week 14 -- ARI's bye is irrelevant now"

    def test_week_actually_played_is_always_playable(self):
        """The exact failure that surfaced this: the player played a week
        that is a bye on their FIRST team's schedule."""
        db = _db({"ARI": _minus(14), "MIN": _minus(13)})
        real = {w: "ARI" for w in range(1, 13)}
        real[14] = "MIN"
        weeks, _ = possible_weeks_for_player(db, real, 2023)
        assert 14 in weeks

    def test_no_more_weeks_than_the_season_has(self):
        db = _db({"ARI": _minus(14), "MIN": _minus(13)})
        real = {w: "ARI" for w in range(1, 13)}
        real[14] = "MIN"
        weeks, _ = possible_weeks_for_player(db, real, 2023)
        assert len(weeks) == len(set(weeks))
        assert len(weeks) <= REGULAR_SEASON_MAX_WEEK
        assert set(real).issubset(set(weeks)), "every played week must be projected"

    def test_weeks_before_debut_use_the_first_known_team(self):
        db = _db({"KC": _minus(6)})
        real = {10: "KC", 11: "KC"}
        weeks, team_by_week = possible_weeks_for_player(db, real, 2023)
        assert team_by_week[1] == "KC"
        assert 6 not in weeks, "KC's bye is still a bye before the player debuts"

    def test_single_team_player_matches_the_plain_team_lookup(self):
        """No trade: identical to the old behaviour."""
        from src.models.single_week_ppr.season_projection import possible_weeks_for_team
        db = _db({"KC": _minus(10)})
        real = {w: "KC" for w in _minus(10)}
        weeks, team_by_week = possible_weeks_for_player(db, real, 2023)
        assert weeks == possible_weeks_for_team(db, "KC", 2023)
        assert set(team_by_week.values()) == {"KC"}

    def test_empty_history_is_handled(self):
        weeks, team_by_week = possible_weeks_for_player(_db({}), {}, 2023)
        assert weeks == [] and team_by_week == {}

    def test_schedule_is_fetched_once_per_team(self):
        """The per-week loop must not hit the DB 18 times."""
        db = _db({"ARI": _minus(14), "MIN": _minus(13)})
        real = {w: "ARI" for w in range(1, 13)}
        real[14] = "MIN"
        possible_weeks_for_player(db, real, 2023)
        assert db.get_schedule.call_count == 2


class TestWeekAccountingIdentity:
    @pytest.mark.parametrize("real,schedules", [
        ({**{w: "ARI" for w in range(1, 13)}, 14: "MIN"},
         {"ARI": _minus(14), "MIN": _minus(13)}),
        ({**{w: "CLE" for w in range(1, 9)}, 9: "CIN"},
         {"CLE": _minus(9), "CIN": _minus(12)}),
    ])
    def test_real_weeks_never_exceed_possible(self, real, schedules):
        """The identity that caught the bug: a player cannot have played more
        weeks than the projection considers possible."""
        weeks, _ = possible_weeks_for_player(_db(schedules), real, 2023)
        assert len(real) <= len(weeks)
        assert set(real).issubset(set(weeks))

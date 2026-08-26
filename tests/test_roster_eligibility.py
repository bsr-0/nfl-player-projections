"""A synthetic week may only be generated for a player who could have played.

A roster audit of 644 pre-debut synthetic QB weeks found exactly ONE where
the player was a listed starter, and 45.7% of the manufactured fantasy
points came from players who were on IR, on the practice squad, declared
inactive, or not in the league at all (GAPS.md). These tests pin the
semantic contract so that can't drift back.

The contract is ACTIVE ROSTER, not starter. A rostered backup stays in the
forecast population -- being a backup is for the model's features to
express, not for the population definition to assume, and filtering to
known starters would leak the outcome into the population.

`active_roster_weeks` is monkeypatched; no DB.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.models.single_week_ppr.season_projection as sp
from src.models.single_week_ppr.season_projection import (
    possible_weeks_for_player, WeekSkipTracker, REGULAR_SEASON_MAX_WEEK,
)

ALL_WEEKS = list(range(1, REGULAR_SEASON_MAX_WEEK + 1))


@pytest.fixture
def db():
    d = MagicMock()
    d.get_schedule.side_effect = lambda season=None, team=None: pd.DataFrame(
        {"week": ALL_WEEKS})
    return d


def _played(*weeks, team="KC"):
    return {w: team for w in weeks}


class TestRosterEligibilityContract:
    """One case per real failure mode found in the audit."""

    @pytest.mark.parametrize("label,active_weeks,played,ineligible_weeks", [
        # Kyler Murray 2023: on IR weeks 1-9, active and playing from 10.
        ("on IR -> no synthetic row for the IR weeks",
         set(range(10, 19)), [10, 11, 12], list(range(1, 10))),
        # Desmond Ridder 2024: practice squad until week 9.
        ("practice squad -> no synthetic row while on the PS",
         set(range(9, 19)), [9, 10], list(range(1, 9))),
        # Philip Rivers 2025: retired, not acquired until week 15.
        ("not yet acquired -> no synthetic row before arrival",
         {15, 16, 17, 18}, [15, 16], list(range(1, 15))),
    ])
    def test_ineligible_weeks_are_never_synthesized(
            self, db, monkeypatch, label, active_weeks, played, ineligible_weeks):
        monkeypatch.setattr(sp, "active_roster_weeks",
                            lambda pid, season: set(active_weeks))
        real = _played(*played)
        weeks, _ = possible_weeks_for_player(db, real, 2023, player_id="p1")
        synthetic = [w for w in weeks if w not in real]
        leaked = sorted(set(synthetic) & set(ineligible_weeks))
        assert not leaked, f"{label}: ineligible weeks synthesized: {leaked}"
        assert set(synthetic).issubset(active_weeks), label

    def test_rostered_backup_stays_in_the_population(self, db, monkeypatch):
        """Marcus Mariota 2023: active all season, first real week 13. Every
        week he was rostered must still be synthesized -- being a backup is
        not a reason to remove him from the forecast population."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: set(ALL_WEEKS))
        real = _played(13)
        weeks, _ = possible_weeks_for_player(db, real, 2023, player_id="p1")
        synthetic = [w for w in weeks if w not in real]
        assert len(synthetic) == len(ALL_WEEKS) - 1

    def test_rostered_starter_is_allowed(self, db, monkeypatch):
        """Starters are in the population for the same reason backups are --
        the filter is eligibility, never role."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: set(ALL_WEEKS))
        weeks, _ = possible_weeks_for_player(db, _played(5), 2023, player_id="p1")
        assert len(weeks) == len(ALL_WEEKS)

    def test_role_is_never_consulted(self, db, monkeypatch):
        """Two players identical in roster status must get identical weeks
        regardless of depth chart -- otherwise the population definition has
        absorbed the outcome."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: {1, 2, 3, 4, 5})
        a, _ = possible_weeks_for_player(db, _played(5), 2023, player_id="starter")
        b, _ = possible_weeks_for_player(db, _played(5), 2023, player_id="backup")
        assert a == b


class TestWeeksActuallyPlayed:
    def test_played_weeks_survive_an_ineligible_status(self, db, monkeypatch):
        """If he played, he was eligible -- whatever the roster table says.
        Roster data must never delete a real observation."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: set())
        real = _played(3, 4, 5)
        weeks, _ = possible_weeks_for_player(db, real, 2023, player_id="p1")
        assert weeks == [3, 4, 5]


class TestFallbackWhenRosterDataMissing:
    def test_uncovered_season_permits_rather_than_drops(self, db, monkeypatch):
        """active_roster_weeks returns None for a season with no roster
        coverage. That must mean 'cannot check', not 'nobody is eligible'."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: None)
        weeks, _ = possible_weeks_for_player(db, _played(5), 2010, player_id="p1")
        # 2010 is a 17-week regular season -- the other tests here use 2023,
        # which is 18. This asserted 18 for 2010 and so encoded the flat-cap
        # bug: week 18 in 2010 is the wild-card round, not a playable week.
        assert len(weeks) == sp.regular_season_max_week(2010) == 17

    def test_filter_can_be_disabled(self, db, monkeypatch):
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: {1})
        weeks, _ = possible_weeks_for_player(
            db, _played(5), 2023, player_id="p1", require_active_roster=False)
        assert len(weeks) == len(ALL_WEEKS)

    def test_no_player_id_means_no_filter(self, db, monkeypatch):
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: {1})
        weeks, _ = possible_weeks_for_player(db, _played(5), 2023)
        assert len(weeks) == len(ALL_WEEKS)


class TestFunnelObservability:
    def test_every_stage_is_counted(self, db, monkeypatch):
        """candidate -> roster eligible must be countable, so an exclusion
        can never disappear into a `continue`."""
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: {1, 2, 3, 4, 5})
        t = WeekSkipTracker("test")
        real = _played(5)
        possible_weeks_for_player(db, real, 2023, player_id="p1", skip_tracker=t)

        candidates = len(ALL_WEEKS) - len(real)          # team played, he didn't
        assert t.funnel["candidate_weeks"] == candidates
        assert t.funnel["roster_eligible"] == 4          # weeks 1-4
        excluded = [s for s in t.skips
                    if s["reason"] == WeekSkipTracker.ROSTER_INELIGIBLE]
        assert len(excluded) == candidates - 4
        assert t.funnel["roster_eligible"] + len(excluded) == candidates

    def test_exclusions_name_the_reason_and_week(self, db, monkeypatch):
        monkeypatch.setattr(sp, "active_roster_weeks", lambda pid, s: {1, 2})
        t = WeekSkipTracker("test")
        possible_weeks_for_player(db, _played(1), 2023, player_id="p1", skip_tracker=t)
        weeks = {s["week"] for s in t.skips}
        assert 2 not in weeks, "week 2 was active -- must not be excluded"
        assert 3 in weeks and 18 in weeks
        assert all(s["reason"] == WeekSkipTracker.ROSTER_INELIGIBLE for s in t.skips)

    def test_funnel_report_prints_each_stage(self, capsys):
        t = WeekSkipTracker("phase X")
        t.count("candidate_weeks", 10)
        t.count("roster_eligible", 6)
        t.count("row_constructed", 5)
        t.count("predicted", 5)
        t.report_funnel()
        out = capsys.readouterr().out
        for stage in WeekSkipTracker.FUNNEL_STAGES:
            assert stage in out
        assert "(-4)" in out, "drop between candidate and eligible must be shown"

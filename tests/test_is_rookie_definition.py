"""Regression tests for the `is_rookie` definition.

The label has been wrong three separate ways, each silently:

  1. `games_count <= 8` in `_add_injury_features` -- labelled any veteran who
     missed half a season a rookie. Removed; `advanced_rookie_injury` is the
     single owner.
  2. Frame-relative debut -- derived from the minimum season PRESENT, so any
     windowed fold relabelled its own oldest veterans. Filtering to 2020+ made
     rookies of Frank Gore, Adrian Peterson and LeSean McCoy (127 of 292
     rookie player-seasons in that frame). Fixed with `first_nfl_season`,
     computed over the whole table.
  3. Data-floor censoring -- `first_nfl_season` cannot tell "debuted in 2006"
     from "our data starts in 2006". 548 players had first_nfl_season == 2006;
     of the 382 with draft records, 332 were drafted BEFORE 2006, some as far
     back as 1982. Only 50 were genuine. Fixed by requiring the draft year to
     agree at the floor.

All three were invisible in output: the column existed, was int-typed, and had
a plausible mean.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MIN_HISTORICAL_YEAR
from src.features.advanced_rookie_injury import AdvancedRookieProjector


def _frame(rows):
    base = {"name": "X", "position": "WR", "week": 1, "fantasy_points": 5.0,
            "draft_round": 3, "draft_pick": 80, "is_undrafted": 0}
    return pd.DataFrame([{**base, **r} for r in rows])


@pytest.fixture
def projector():
    return AdvancedRookieProjector()


class TestDataFloorCensoring:
    def test_veteran_censored_at_the_floor_is_not_a_rookie(self, projector):
        """The 332-player case: first VISIBLE season is the floor, but the
        draft year proves they were already in the league."""
        df = _frame([{"player_id": "VET", "season": MIN_HISTORICAL_YEAR,
                      "first_nfl_season": MIN_HISTORICAL_YEAR, "draft_season": 1999}])
        assert int(projector.add_advanced_rookie_features(df)["is_rookie"].iloc[0]) == 0

    def test_genuine_rookie_at_the_floor_is_kept(self, projector):
        """The 50-player case: draft year agrees with the visible debut, so the
        censoring guard must not throw them away too."""
        df = _frame([{"player_id": "ROOK", "season": MIN_HISTORICAL_YEAR,
                      "first_nfl_season": MIN_HISTORICAL_YEAR,
                      "draft_season": MIN_HISTORICAL_YEAR}])
        assert int(projector.add_advanced_rookie_features(df)["is_rookie"].iloc[0]) == 1

    def test_undrafted_at_the_floor_resolves_conservatively(self, projector):
        """Unknowable, so it must NOT claim rookie: handing a 10-year veteran
        rookie draft-capital priors is worse than omitting him from a
        subgroup."""
        df = _frame([{"player_id": "UDFA", "season": MIN_HISTORICAL_YEAR,
                      "first_nfl_season": MIN_HISTORICAL_YEAR, "draft_season": 0,
                      "is_undrafted": 1, "draft_pick": 400, "draft_round": 8}])
        assert int(projector.add_advanced_rookie_features(df)["is_rookie"].iloc[0]) == 0

    def test_debut_after_the_floor_is_unaffected_by_the_guard(self, projector):
        df = _frame([{"player_id": "R2020", "season": 2020,
                      "first_nfl_season": 2020, "draft_season": 2020}])
        assert int(projector.add_advanced_rookie_features(df)["is_rookie"].iloc[0]) == 1


class TestLateDebutIsDeliberate:
    def test_drafted_earlier_but_first_producing_season_counts_as_rookie(self, projector):
        """A DEFINITIONAL choice, not a bug -- 9.7% of drafted rookies.

        `is_rookie` here means "no prior NFL production to learn from", which
        is true for a 2010 pick whose first stats row is 2013. Keying on
        `season == draft_season` instead would label him a veteran while every
        history feature is NaN -- a contradiction the model cannot resolve.
        """
        df = _frame([{"player_id": "LATE", "season": 2013,
                      "first_nfl_season": 2013, "draft_season": 2010}])
        assert int(projector.add_advanced_rookie_features(df)["is_rookie"].iloc[0]) == 1


class TestFrameIndependence:
    def test_label_does_not_move_when_the_frame_is_filtered(self, projector):
        """Bug 2: a veteran must not become a rookie because a fold window
        starts at his earliest visible season."""
        rows = [{"player_id": "V", "season": s, "first_nfl_season": 2015,
                 "draft_season": 2015} for s in (2015, 2020, 2021)]
        full = projector.add_advanced_rookie_features(_frame(rows))
        sliced = projector.add_advanced_rookie_features(_frame(rows[1:]))
        assert list(full["is_rookie"]) == [1, 0, 0]
        assert list(sliced["is_rookie"]) == [0, 0], "filtering created a rookie"

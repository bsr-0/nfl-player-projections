"""Tests for the causal P(plays) estimators.

The critical property is causality: an estimator for week W may read weeks
< W of the current season (already-observed, legitimate information) but
must never read week W or later.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.availability import (
    AVAILABILITY_ESTIMATORS, PlayerAvailabilityHistory,
    prior_season_only, current_season_only, simple_blend,
    shrinkage_blend, recency_weighted, POSITION_AVG_FALLBACK,
)


def _hist(played_by_season, team="KC", seasons=(2022, 2023), n_weeks=17):
    rows = []
    for season, weeks in played_by_season.items():
        for w in weeks:
            rows.append({"player_id": "P1", "season": season, "week": w, "team": team})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_id", "season", "week", "team"])
    team_weeks = {(team, s): list(range(1, n_weeks + 1)) for s in seasons}
    return PlayerAvailabilityHistory(df, team_weeks)


class TestCausality:
    @pytest.mark.parametrize("name", list(AVAILABILITY_ESTIMATORS))
    def test_future_weeks_never_affect_the_estimate(self, name):
        """The whole point: weeks >= target must be invisible. A player who
        plays every remaining week must look identical, at week 10, to one
        who plays none of them."""
        est = AVAILABILITY_ESTIMATORS[name]
        past = [1, 2, 3]
        plays_future = _hist({2022: list(range(1, 18)), 2023: past + list(range(10, 18))})
        no_future = _hist({2022: list(range(1, 18)), 2023: past})
        assert est(plays_future, "P1", 2023, "KC", 10) == pytest.approx(
            est(no_future, "P1", 2023, "KC", 10))


class TestPriorSeasonOnly:
    def test_reproduces_prior_season_rate(self):
        h = _hist({2022: list(range(1, 18)), 2023: [1]})
        assert prior_season_only(h, "P1", 2023, "KC", 10) == pytest.approx(1.0)

    def test_ignores_current_season_collapse(self):
        """Documents the baseline's flaw: a player who has missed weeks 4-9
        still scores his healthy prior-season rate."""
        h = _hist({2022: list(range(1, 18)), 2023: [1, 2, 3]})
        assert prior_season_only(h, "P1", 2023, "KC", 10) == pytest.approx(1.0)

    def test_rookie_falls_back(self):
        h = _hist({2023: [1, 2]})
        assert prior_season_only(h, "P1", 2023, "KC", 10) == POSITION_AVG_FALLBACK


class TestCurrentSeasonOnly:
    def test_uses_only_elapsed_weeks(self):
        # played 3 of the 9 team games before week 10
        h = _hist({2022: list(range(1, 18)), 2023: [1, 2, 3]})
        assert current_season_only(h, "P1", 2023, "KC", 10) == pytest.approx(3 / 9)

    def test_week_one_has_no_evidence_so_falls_back_to_prior(self):
        h = _hist({2022: list(range(1, 18)), 2023: []})
        assert current_season_only(h, "P1", 2023, "KC", 1) == pytest.approx(1.0)


class TestBlends:
    def test_simple_blend_is_midpoint(self):
        h = _hist({2022: list(range(1, 18)), 2023: [1, 2, 3]})   # prior 1.0, current 3/9
        assert simple_blend(h, "P1", 2023, "KC", 10) == pytest.approx(0.5 * 1.0 + 0.5 * (3 / 9))

    def test_shrinkage_trusts_current_more_as_evidence_accumulates(self):
        """1 missed game after 2 games != 1 missed game after 14."""
        early = _hist({2022: list(range(1, 18)), 2023: [1]})       # 1 of 2 elapsed
        late = _hist({2022: list(range(1, 18)), 2023: list(range(1, 9))})  # 8 of 15 elapsed
        e = shrinkage_blend(early, "P1", 2023, "KC", 3)
        l = shrinkage_blend(late, "P1", 2023, "KC", 16)
        # both have current < prior(=1.0); the later one should sit closer to
        # its current-season rate, i.e. further below the prior
        assert l < e

    def test_shrinkage_equals_prior_when_no_evidence(self):
        h = _hist({2022: list(range(1, 18)), 2023: []})
        assert shrinkage_blend(h, "P1", 2023, "KC", 1) == pytest.approx(1.0)

    def test_recency_favors_recent_availability(self):
        """Missed early / played recently should beat played early / missed
        recently, even though both played the same number of games."""
        recent_good = _hist({2022: list(range(1, 18)), 2023: [6, 7, 8, 9]})
        recent_bad = _hist({2022: list(range(1, 18)), 2023: [1, 2, 3, 4]})
        assert recency_weighted(recent_good, "P1", 2023, "KC", 10) > \
               recency_weighted(recent_bad, "P1", 2023, "KC", 10)


class TestRangeSanity:
    @pytest.mark.parametrize("name", list(AVAILABILITY_ESTIMATORS))
    def test_estimates_stay_in_unit_interval(self, name):
        est = AVAILABILITY_ESTIMATORS[name]
        for played in ([], [1], [1, 2, 3], list(range(1, 10))):
            h = _hist({2022: list(range(1, 18)), 2023: played})
            v = est(h, "P1", 2023, "KC", 10)
            assert 0.0 <= v <= 1.0


class TestLeakageInvariant:
    """HARD INVARIANT, stated explicitly rather than left implicit:

        For a fixed player-week, no information dated at or after the target
        week may change the availability estimate.

    This protects the whole projection pipeline against a future edit that
    quietly starts consulting the outcome it is supposed to forecast. It is
    the property, not merely a test of today's implementations.
    """

    @pytest.mark.parametrize("name", list(AVAILABILITY_ESTIMATORS))
    @pytest.mark.parametrize("target_week", [1, 5, 10, 17])
    def test_arbitrary_future_rewrites_cannot_move_the_estimate(self, name, target_week):
        est = AVAILABILITY_ESTIMATORS[name]
        past = [w for w in [1, 2, 3, 6, 7] if w < target_week]
        baseline = _hist({2022: list(range(1, 18)), 2023: past})
        ref = est(baseline, "P1", 2023, "KC", target_week)

        # every possible future: none, all, and a scattered subset
        futures = [
            [],
            [w for w in range(target_week, 18)],
            [w for w in range(target_week, 18) if w % 2 == 0],
        ]
        for fut in futures:
            h = _hist({2022: list(range(1, 18)), 2023: past + fut})
            assert est(h, "P1", 2023, "KC", target_week) == pytest.approx(ref), (
                f"{name} leaked future weeks {fut} at target_week={target_week}")

    @pytest.mark.parametrize("name", list(AVAILABILITY_ESTIMATORS))
    def test_estimate_is_monotone_nondecreasing_in_past_availability(self, name):
        """Sanity on direction: having played MORE of the elapsed season can
        never lower the availability estimate."""
        est = AVAILABILITY_ESTIMATORS[name]
        fewer = _hist({2022: list(range(1, 18)), 2023: [1]})
        more = _hist({2022: list(range(1, 18)), 2023: [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        assert est(more, "P1", 2023, "KC", 10) >= est(fewer, "P1", 2023, "KC", 10) - 1e-9

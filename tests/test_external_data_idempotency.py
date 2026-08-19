"""Regression tests for re-running external enrichment on an enriched frame.

`build_synthetic_week_row` carries forward the player's last real row — which
`run_fold` already passed through `add_external_features` — and enriches it
again for the target week. That second pass used to raise
`ValueError: Columns must be same length as key`, because the rename in
`get_opponent_matchup_features` collided with the first pass's columns and
left duplicate labels behind. The error was swallowed, so every synthetic
week silently fell back to constant defaults.

No network or DB: `calculate_defense_rankings` is exercised directly on
synthetic frames.
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.external_data import DefenseRankingsLoader

MATCHUP_COLS = ['opp_defense_rank', 'opp_matchup_score',
                'opp_pts_allowed', 'defense_data_available']


def _frame(**overrides):
    base = dict(
        player_id=['a', 'a'], position=['QB', 'QB'], team=['KC', 'KC'],
        opponent=['DEN', 'LV'], season=[2023, 2023], week=[5, 6],
        fantasy_points=[20.0, 18.0],
    )
    base.update(overrides)
    return pd.DataFrame(base)


def _enrich(df):
    with redirect_stdout(io.StringIO()):
        return DefenseRankingsLoader().get_opponent_matchup_features(df)


class TestMatchupFeatureIdempotency:
    def test_reenrichment_does_not_raise(self):
        once = _enrich(_frame())
        _enrich(once)  # used to raise ValueError

    def test_no_duplicate_column_labels(self):
        twice = _enrich(_enrich(_frame()))
        dupes = [c for c in twice.columns if list(twice.columns).count(c) > 1]
        assert dupes == []

    def test_values_are_stable_across_passes(self):
        once = _enrich(_frame())
        twice = _enrich(once)
        thrice = _enrich(twice)
        pd.testing.assert_frame_equal(once[MATCHUP_COLS], twice[MATCHUP_COLS])
        pd.testing.assert_frame_equal(twice[MATCHUP_COLS], thrice[MATCHUP_COLS])

    def test_stale_carried_forward_values_are_recomputed(self):
        """A carried-forward row holds the PRIOR week's matchup numbers. The
        second pass must overwrite them, not preserve them."""
        stale = _frame()
        stale['opp_defense_rank'] = 3.0
        stale['opp_matchup_score'] = 0.99
        stale['opp_pts_allowed'] = 31.0
        stale['defense_data_available'] = 1

        out = _enrich(stale)
        assert (out['opp_defense_rank'] != 3.0).all()
        assert (out['opp_matchup_score'] != 0.99).all()
        assert (out['opp_pts_allowed'] != 31.0).all()

    def test_availability_flag_is_not_inherited(self):
        """defense_data_available must describe THIS pass. A frame arriving
        with the flag set, but no resolvable opponent history, must come out
        flagged 0 rather than keeping the inherited 1."""
        stale = _frame()
        stale['defense_data_available'] = 1
        out = _enrich(stale)
        assert (out['defense_data_available'] == 0).all()


class TestSeasonImportCache:
    def test_cache_returns_equal_but_independent_frames(self):
        """Callers mutate what they get back, so the cache must hand out
        copies — otherwise one caller's edit corrupts every later read."""
        from src.data import external_data

        external_data.clear_season_import_cache()
        sentinel = pd.DataFrame({'season': [2023], 'week': [1], 'value': [1.0]})
        calls = []

        def fetch():
            calls.append(1)
            return sentinel

        first = external_data._cached_import('unit-test', [2023], fetch)
        first.loc[0, 'value'] = 999.0
        second = external_data._cached_import('unit-test', [2023], fetch)

        assert len(calls) == 1, "second call should have been served from cache"
        assert second.loc[0, 'value'] == 1.0, "mutation leaked into the cache"

        external_data.clear_season_import_cache()
        external_data._cached_import('unit-test', [2023], fetch)
        assert len(calls) == 2, "clear_season_import_cache did not evict"

    def test_season_order_does_not_create_a_second_entry(self):
        from src.data import external_data

        external_data.clear_season_import_cache()
        calls = []

        def fetch():
            calls.append(1)
            return pd.DataFrame({'season': [2022, 2023]})

        external_data._cached_import('unit-test', [2022, 2023], fetch)
        external_data._cached_import('unit-test', [2023, 2022], fetch)
        assert len(calls) == 1

        external_data.clear_season_import_cache()

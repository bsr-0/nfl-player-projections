"""Guards for PRESERVE_HISTORY_MISSINGNESS.

A player's first-ever NFL week has no preceding weeks, so every rolling/lag
feature is genuinely undefined there. Median-filling it teaches the model that
a rookie's week 1 looks like an average veteran, and leaves LightGBM with no
missing-direction to route a real cold-start row into -- Phase 7's rookie bias
is -41.7 against Step 8A's -9.4, and Step 8A differs precisely by leaving its
lags NaN (measured: rookie lag NaN rate 1.000, zero rate 0.000).

FIVE separate fillers destroy that missingness, which is why the restore runs
once at the END of prepare_features rather than being exempted at each site:

  1. per-column `.fillna(0)` at creation sites
  2. the blanket numeric fill in utilization_score
  3. `_impute_missing`'s position-aware median
  4. the position-specific feature block, which runs AFTER (3)
  5. `advanced_rookie_injury`, measured wiping 959 restored NaNs back to 0

Zero-filling instead would be WORSE than the median: 0 is an occupied value in
these columns (59.4% of veteran snap_share_y1 sits below 0.5, min exactly
0.00), so a rookie would land on top of marginal veterans.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as settings
from src.features.feature_engineering import (
    FeatureEngineer,
    _structurally_missing_cols,
    history_missingness_cols,
)


@pytest.fixture
def flag(monkeypatch):
    def _set(v):
        monkeypatch.setattr(settings, "PRESERVE_HISTORY_MISSINGNESS", v)
    return _set


class TestFlagIsNotInert:
    def test_off_leaves_the_exempt_set_untouched(self, flag):
        flag(False)
        base = _structurally_missing_cols()
        flag(True)
        assert _structurally_missing_cols() != base, "flag does not change behaviour"

    def test_history_set_is_derived_not_hand_listed(self):
        cols = history_missingness_cols()
        assert len(cols) > 40, "history set suspiciously small"
        # Derived from CAUSAL_FEATURES, so a new rolling feature is picked up
        # automatically rather than silently escaping the exemption.
        assert any(c.endswith("_roll3_mean") for c in cols)
        assert "bayesian_prior_ppg" in cols
        assert "availability_3yr" in cols

    def test_career_static_columns_are_never_treated_as_history(self):
        """Draft capital exists for a rookie; blanking it would throw away the
        only signal a cold-start row has."""
        cols = history_missingness_cols()
        for c in ("draft_pick", "draft_pick_value", "is_undrafted", "combine_score",
                  "is_power5", "age_curve", "is_rookie"):
            assert c not in cols


class TestRestoreDebutHistoryNaN:
    def _frame(self):
        return pd.DataFrame({
            "player_id": ["A", "A", "A", "B", "B"],
            "season": [2020, 2020, 2021, 2020, 2020],
            "first_nfl_season": [2020, 2020, 2020, 2019, 2019],
            "week": [1, 2, 1, 1, 2],
            "bayesian_prior_ppg": [5.0, 5.0, 5.0, 5.0, 5.0],
            "availability_3yr": [0.9, 0.9, 0.9, 0.9, 0.9],
            "draft_pick": [10, 10, 10, 20, 20],
        })

    def test_blanks_only_the_players_first_ever_nfl_week(self):
        out = FeatureEngineer()._restore_debut_history_nan(self._frame())
        # A debuts 2020 wk1 -> blanked. A 2020 wk2 has one prior week, so
        # blanking it would FABRICATE missingness rather than preserve it.
        assert np.isnan(out.loc[0, "bayesian_prior_ppg"])
        assert not np.isnan(out.loc[1, "bayesian_prior_ppg"])
        # A's 2021 wk1 is not a debut season.
        assert not np.isnan(out.loc[2, "bayesian_prior_ppg"])

    def test_ignores_players_whose_debut_predates_the_frame(self):
        """B first played 2019, which is not in this frame. His 2020 week 1 is
        a continuation, not a debut, and must keep its history."""
        out = FeatureEngineer()._restore_debut_history_nan(self._frame())
        assert not np.isnan(out.loc[3, "bayesian_prior_ppg"])

    def test_career_static_survives_the_blanking(self):
        out = FeatureEngineer()._restore_debut_history_nan(self._frame())
        assert out["draft_pick"].notna().all(), "draft capital must survive"

    def test_missing_columns_do_not_raise(self):
        out = FeatureEngineer()._restore_debut_history_nan(
            pd.DataFrame({"player_id": ["A"], "season": [2020]}))
        assert len(out) == 1

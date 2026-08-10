"""Unit tests for Phase 3 (next_focus.md) window/weighting logic.

Synthetic data only — no DB/network. Tests window_to_season_list (season
selection, including the deliberate 2018-floor bypass) and
compute_recency_weights (none/linear/exponential formulas).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.windows import (
    WEIGHTING_SCHEMES,
    WINDOW_CANDIDATES,
    compute_recency_weights,
    window_to_season_list,
)
from config.settings import MIN_HISTORICAL_YEAR, MODEL_CONFIG


AVAILABLE_2006_2024 = list(range(2006, 2025))  # 2006..2024 inclusive


class TestWindowToSeasonList:
    @pytest.mark.parametrize("window,expected_years", [("3y", 3), ("5y", 5), ("7y", 7), ("10y", 10)])
    def test_fixed_windows_return_n_most_recent_seasons(self, window, expected_years):
        seasons = window_to_season_list(window, test_season=2023, available_seasons=AVAILABLE_2006_2024)
        assert len(seasons) == expected_years
        assert seasons == list(range(2023 - expected_years, 2023))

    def test_all_history_bypasses_2018_floor(self):
        seasons = window_to_season_list("all", test_season=2023, available_seasons=AVAILABLE_2006_2024)
        assert seasons[0] == MIN_HISTORICAL_YEAR
        assert seasons[-1] == 2022
        assert len(seasons) == 2023 - MIN_HISTORICAL_YEAR

    def test_10y_and_all_are_distinct_when_more_than_10_years_available(self):
        """The whole point of bypassing the 2018 floor: 10y != all when
        enough pre-2018 history exists (unlike under the production floor,
        where both would collapse to whatever's available since 2018)."""
        ten_year = window_to_season_list("10y", test_season=2023, available_seasons=AVAILABLE_2006_2024)
        all_history = window_to_season_list("all", test_season=2023, available_seasons=AVAILABLE_2006_2024)
        assert len(ten_year) != len(all_history)
        assert set(ten_year) < set(all_history)

    def test_degrades_gracefully_when_insufficient_history(self):
        """Requesting more years than exist shouldn't error, just return
        what's available."""
        short_history = [2020, 2021, 2022]
        seasons = window_to_season_list("10y", test_season=2023, available_seasons=short_history)
        assert seasons == [2020, 2021, 2022]

    def test_unknown_window_raises(self):
        with pytest.raises(ValueError):
            window_to_season_list("2y", test_season=2023, available_seasons=AVAILABLE_2006_2024)

    def test_never_includes_pre_min_historical_year(self):
        seasons = window_to_season_list("all", test_season=2023,
                                         available_seasons=[2001, 2002] + AVAILABLE_2006_2024)
        assert min(seasons) >= MIN_HISTORICAL_YEAR


class TestComputeRecencyWeights:
    @pytest.fixture
    def seasons(self):
        return pd.Series([2018, 2019, 2020, 2021, 2022])

    def test_none_returns_none(self, seasons):
        assert compute_recency_weights(seasons, "none") is None

    def test_linear_is_monotonic_increasing_and_maxes_at_one(self, seasons):
        w = compute_recency_weights(seasons, "linear")
        assert np.isclose(w.max(), 1.0)
        assert np.all(np.diff(w) > 0)  # strictly increasing with season

    def test_exponential_matches_production_formula(self, seasons):
        """Must reuse the exact formula already live in
        src/models/position_models.py:_horizon_recency_weights, not a
        reinvented one."""
        w = compute_recency_weights(seasons, "exponential")
        halflife = MODEL_CONFIG["horizon_recency_halflife"][1]
        seasons_arr = seasons.to_numpy(dtype=float)
        expected_decay = np.power(0.5, (seasons_arr.max() - seasons_arr) / halflife)
        expected = expected_decay / expected_decay.max()
        np.testing.assert_allclose(w, expected)

    def test_exponential_is_monotonic_increasing_and_maxes_at_one(self, seasons):
        w = compute_recency_weights(seasons, "exponential")
        assert np.isclose(w.max(), 1.0)
        assert np.all(np.diff(w) > 0)

    def test_single_season_returns_uniform_ones(self):
        w = compute_recency_weights(pd.Series([2022, 2022, 2022]), "exponential")
        np.testing.assert_allclose(w, [1.0, 1.0, 1.0])

    def test_unknown_scheme_raises(self, seasons):
        with pytest.raises(ValueError):
            compute_recency_weights(seasons, "quadratic")

    @pytest.mark.parametrize("scheme", WEIGHTING_SCHEMES)
    def test_all_schemes_covered(self, scheme):
        assert scheme in WEIGHTING_SCHEMES

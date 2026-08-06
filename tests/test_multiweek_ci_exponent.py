"""Tests for the multi-week CI-scaling exponent derivation (GAPS.md §7.4
follow-up: "n_weeks**0.4 multi-week CI-scaling exponent")."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.derive_multiweek_ci_exponent import fit_exponent, _forward_sum
from src.models.ensemble import _multi_week_ci_scale
from config.settings import MULTI_WEEK_CI_SCALE_EXPONENT, MULTI_WEEK_CI_SCALE_EXPONENT_DEFAULT


def _simulate_std_by_window(weekly_series: np.ndarray, windows, n_players=400, n_weeks=18, seed=0):
    """Build synthetic per-player weekly series from a generator and
    measure realized forward-sum std per window, mirroring what the real
    derivation script does against actual historical data."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_players):
        weekly = weekly_series(rng, n_weeks)
        for wk in range(n_weeks):
            rows.append((pid, wk, weekly[wk]))
    df = pd.DataFrame(rows, columns=["player_id", "week", "fantasy_points"])
    df = df.sort_values(["player_id", "week"])

    std_by_window = {}
    for window in windows:
        sums = df.groupby("player_id")["fantasy_points"].transform(
            lambda x, w=window: _forward_sum(x, w)
        )
        valid = sums.dropna()
        if len(valid) >= 30:
            std_by_window[window] = float(valid.std())
    return std_by_window


class TestFitExponent:
    def test_iid_weeks_recover_near_half(self):
        """Independent weekly noise: Var(sum of N) = N * Var(week), so the
        fitted exponent should land close to 0.5 (the textbook sqrt-law)."""
        def iid_weekly(rng, n_weeks):
            return rng.normal(15, 5, n_weeks)

        std_by_window = _simulate_std_by_window(iid_weekly, [1, 2, 4, 8, 13, 18])
        alpha, r2 = fit_exponent(std_by_window)

        assert alpha is not None
        assert 0.40 <= alpha <= 0.60, f"Expected alpha near 0.5 for i.i.d. weeks, got {alpha:.3f}"
        assert r2 > 0.95

    def test_autocorrelated_weeks_recover_above_half(self):
        """Strongly autocorrelated weekly performance (AR(1)-style hot/cold
        streaks) should push the fitted exponent well above the i.i.d. 0.5
        baseline."""
        def ar1_weekly(rng, n_weeks):
            phi = 0.85
            innovations = rng.normal(0, 5, n_weeks)
            level = np.zeros(n_weeks)
            level[0] = innovations[0]
            for t in range(1, n_weeks):
                level[t] = phi * level[t - 1] + innovations[t]
            return 15 + level

        std_by_window = _simulate_std_by_window(ar1_weekly, [1, 2, 4, 8, 13, 18])
        alpha, r2 = fit_exponent(std_by_window)

        assert alpha is not None
        assert alpha > 0.6, f"Expected alpha > 0.6 for strongly autocorrelated weeks, got {alpha:.3f}"

    def test_insufficient_windows_returns_none(self):
        alpha, r2 = fit_exponent({1: 5.0})
        assert alpha is None
        assert r2 is None


class TestMultiWeekCiScale:
    def test_scale_uses_configured_per_position_exponent(self):
        for position, alpha in MULTI_WEEK_CI_SCALE_EXPONENT.items():
            got = _multi_week_ci_scale(4, position)
            assert got == pytest.approx(4.0 ** alpha)

    def test_scale_falls_back_to_default_for_unknown_position(self):
        got = _multi_week_ci_scale(4, "UNKNOWN_POSITION")
        assert got == pytest.approx(4.0 ** MULTI_WEEK_CI_SCALE_EXPONENT_DEFAULT)

    def test_scale_is_one_at_n_weeks_one(self):
        for position in MULTI_WEEK_CI_SCALE_EXPONENT:
            assert _multi_week_ci_scale(1, position) == pytest.approx(1.0)

    def test_fitted_exponents_are_above_iid_baseline(self):
        """Real historical NFL fantasy production is positively
        autocorrelated week-to-week, so every fitted exponent should sit
        above the independent-weeks baseline of 0.5 (documents the actual
        finding, not just a property of the code)."""
        for position, alpha in MULTI_WEEK_CI_SCALE_EXPONENT.items():
            assert alpha > 0.5, f"{position}: expected alpha > 0.5, got {alpha}"

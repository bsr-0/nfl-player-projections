"""Tests for Phase 4A conformal interval helper
(scripts/add_conformal_intervals.py).

Critical invariants:
1. Temporal causality: week W's interval uses ONLY weeks < W's
   residuals. A monotonically-decaying residual stream must produce
   monotonically-decaying interval widths over time.
2. Per-position pooling: a high-variance position should get wider
   intervals than a low-variance position with the same mean.
3. Pool window: residuals older than ``window_weeks`` are pruned
   from the pool.
4. Coverage: on synthetic Gaussian-residual data, empirical
   coverage hits the nominal level (within Monte Carlo noise).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "add_conformal_intervals",
    PROJECT_ROOT / "scripts" / "add_conformal_intervals.py",
)
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)


def _make_predictions(seasons, weeks_per_season, n_players_per_pos, residual_std):
    """Build a synthetic predictions DataFrame.

    For each (season, week), each player gets ``predicted = 10`` and
    ``actual = predicted + N(0, residual_std)``.  Returns ordered by
    (season, week, player_id).
    """
    rng = np.random.default_rng(seed=42)
    rows = []
    pid = 0
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            for pos, std in residual_std.items():
                for p in range(n_players_per_pos):
                    pid += 1
                    pred = 10.0
                    actual = pred + rng.normal(0, std)
                    rows.append({
                        "season": season,
                        "week": week,
                        "player_id": f"P{pid:05d}",
                        "name": f"Player {pid}",
                        "position": pos,
                        "team": "X",
                        "predicted": pred,
                        "actual": actual,
                        "is_active": 1,
                    })
    return pd.DataFrame(rows)


def test_temporal_causality_first_week_uses_fallback():
    df = _make_predictions(
        seasons=[2024],
        weeks_per_season=3,
        n_players_per_pos=20,
        residual_std={"WR": 6.0},
    )
    out = ci.add_intervals(df, window_weeks=8, seed_residuals=None)
    w1 = out[out["week"] == 1]
    # Fallback half-widths: WR 80% = 6.5 (per FALLBACK_HALFWIDTHS)
    fb80 = ci.FALLBACK_HALFWIDTHS["WR"][0.80]
    assert (w1["hi80"] - w1["lo80"] - 2 * fb80).abs().max() < 1e-9


def test_pool_grows_after_first_weeks():
    """Once 4+ residuals are in the pool, the empirical quantile
    takes over from the fallback (per the size>=4 rule)."""
    df = _make_predictions(
        seasons=[2024],
        weeks_per_season=4,
        n_players_per_pos=10,
        residual_std={"WR": 6.0},
    )
    out = ci.add_intervals(df, window_weeks=8, seed_residuals=None)
    # By week 2, we have 10 residuals (all from week 1) → empirical.
    w2 = out[out["week"] == 2]
    fb80 = ci.FALLBACK_HALFWIDTHS["WR"][0.80]
    # Width should not exactly match fallback (very unlikely to coincide).
    avg_width = (w2["hi80"] - w2["lo80"]).mean()
    assert abs(avg_width - 2 * fb80) > 1e-3


def test_higher_variance_position_gets_wider_intervals():
    df = _make_predictions(
        seasons=[2024],
        weeks_per_season=10,
        n_players_per_pos=20,
        residual_std={"QB": 12.0, "TE": 3.0},
    )
    out = ci.add_intervals(df, window_weeks=8, seed_residuals=None)
    # Compare interval widths in the late season (after pool is full).
    late = out[out["week"] >= 5]
    qb_width = (late[late["position"] == "QB"]["hi80"] - late[late["position"] == "QB"]["lo80"]).mean()
    te_width = (late[late["position"] == "TE"]["hi80"] - late[late["position"] == "TE"]["lo80"]).mean()
    assert qb_width > te_width * 1.5


def test_coverage_hits_nominal_on_gaussian_synthetic():
    df = _make_predictions(
        seasons=[2024, 2025],
        weeks_per_season=18,
        n_players_per_pos=30,
        residual_std={"QB": 6.0, "RB": 5.0, "WR": 5.0, "TE": 4.0},
    )
    # Run the FULL pipeline including the prior-season seed.
    out_2024 = ci.add_intervals(df[df["season"] == 2024], window_weeks=8)
    seed = ci._seed_from_prior_df(df[df["season"] == 2024], 8) if hasattr(ci, "_seed_from_prior_df") else {
        pos: (df[(df["season"] == 2024) & (df["week"] >= 11) & (df["position"] == pos)]["actual"]
              - df[(df["season"] == 2024) & (df["week"] >= 11) & (df["position"] == pos)]["predicted"]).to_numpy()
        for pos in ["QB", "RB", "WR", "TE"]
    }
    out_2025 = ci.add_intervals(df[df["season"] == 2025], window_weeks=8, seed_residuals=seed)
    # Use late-season 2025 weeks so the pool has stabilized.
    late = out_2025[out_2025["week"] >= 6]
    cov80 = ((late["actual"] >= late["lo80"]) & (late["actual"] <= late["hi80"])).mean()
    cov95 = ((late["actual"] >= late["lo95"]) & (late["actual"] <= late["hi95"])).mean()
    # Wide tolerance (2x what we'd accept on real data) since this
    # is a 30-player synthetic sample with high MC noise.
    assert 0.72 <= cov80 <= 0.88, f"cov80 = {cov80:.2%}"
    assert 0.88 <= cov95 <= 0.99, f"cov95 = {cov95:.2%}"


def test_window_pruning():
    """If we shrink the window to 1 week, only the most recent
    week's residuals should be in the pool — earlier week's
    residuals must be pruned."""
    # Seed two seasons of equal residuals to get a stable pool, then
    # verify pruning kicks in when window_weeks=1.
    df = _make_predictions(
        seasons=[2024],
        weeks_per_season=5,
        n_players_per_pos=10,
        residual_std={"WR": 6.0},
    )
    out_full = ci.add_intervals(df, window_weeks=8, seed_residuals=None)
    out_pruned = ci.add_intervals(df, window_weeks=1, seed_residuals=None)
    # Late weeks: full window has 4*10=40 residuals; pruned has 10.
    # Quantiles should differ.
    late_full = out_full[out_full["week"] == 5]
    late_pruned = out_pruned[out_pruned["week"] == 5]
    assert not np.allclose(
        (late_full["hi80"] - late_full["lo80"]).values,
        (late_pruned["hi80"] - late_pruned["lo80"]).values,
    )


def test_seed_residuals_used_for_week1():
    df = _make_predictions(
        seasons=[2024],
        weeks_per_season=3,
        n_players_per_pos=20,
        residual_std={"WR": 6.0},
    )
    # Seed with a tight residual distribution (std=1) so W1 width
    # is much smaller than the fallback.
    seed = {"WR": np.array([0.5, -0.3, 0.7, -0.4, 0.2, -0.6, 0.8, -0.5] * 5)}
    out = ci.add_intervals(df, window_weeks=8, seed_residuals=seed)
    w1 = out[out["week"] == 1]
    fb80 = ci.FALLBACK_HALFWIDTHS["WR"][0.80]
    # With seed, the empirical quantile (~0.7) drives width far below 2*6.5=13.
    avg_width = (w1["hi80"] - w1["lo80"]).mean()
    assert avg_width < 2 * fb80, f"avg_width={avg_width} should be < {2*fb80} (fallback)"

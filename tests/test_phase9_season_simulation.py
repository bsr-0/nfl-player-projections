"""Unit tests for Phase 9 (next_focus.md) Monte Carlo season simulation.

Synthetic data / seeded RNG only -- no real DB/network, matching the
Phase 7 test file's pattern.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.season_simulation import (
    build_residual_donor_pools,
    simulate_player_season,
    _sample_block,
    _require_nonempty_donor_pool,
    EmptyDonorPoolError,
)


class TestBuildResidualDonorPools:
    def test_filters_by_position_model_and_strict_season(self, tmp_path):
        df = pd.DataFrame({
            "player": ["P1", "P1", "P2", "P3", "P4"],
            "position": ["QB", "QB", "QB", "RB", "QB"],
            "season": [2021, 2021, 2022, 2021, 2023],
            "week": [1, 2, 1, 1, 1],
            "actual_ppr": [10.0, 20.0, 5.0, 8.0, 12.0],
            "prediction": [8.0, 18.0, 6.0, 7.0, 11.0],
            "model": ["F_yeojohnson_huber"] * 4 + ["C_gbm_mae"],
        })
        csv_path = tmp_path / "phase4.csv"
        df.to_csv(csv_path, index=False)

        pools = build_residual_donor_pools(
            [csv_path], position="QB", architecture="F_yeojohnson_huber", before_season=2023,
        )
        # P4 excluded (wrong model AND season not < before_season).
        # P3 excluded (wrong position). P1 (2021) and P2 (2022) included --
        # both strictly before before_season=2023.
        assert ("P1", 2021) in pools
        assert ("P2", 2022) in pools
        assert ("P3", 2021) not in pools  # wrong position
        assert ("P4", 2023) not in pools  # wrong model AND season not < before_season

        p1 = pools[("P1", 2021)]
        assert p1 == [(1, 2.0), (2, 2.0)]  # (10-8), (20-18), sorted by week

    def test_excludes_season_not_strictly_before(self, tmp_path):
        df = pd.DataFrame({
            "player": ["P1", "P1"],
            "position": ["WR", "WR"],
            "season": [2023, 2024],
            "week": [1, 1],
            "actual_ppr": [10.0, 10.0],
            "prediction": [8.0, 8.0],
            "model": ["C_gbm_mae", "C_gbm_mae"],
        })
        csv_path = tmp_path / "phase4.csv"
        df.to_csv(csv_path, index=False)

        pools = build_residual_donor_pools([csv_path], "WR", "C_gbm_mae", before_season=2024)
        assert ("P1", 2023) in pools
        assert ("P1", 2024) not in pools  # not strictly before 2024

    def test_missing_file_returns_empty(self, tmp_path):
        pools = build_residual_donor_pools(
            [tmp_path / "does_not_exist.csv"], "QB", "F_yeojohnson_huber", before_season=2023,
        )
        assert pools == {}


class TestRequireNonemptyDonorPool:
    def test_raises_when_final_config_architecture_not_in_residual_csv(self, tmp_path):
        """Reproduces the real TE failure: FINAL_CONFIG selects an
        architecture (C_gbm_mae) newer than the Phase 4 residual CSV, which
        only has rows for the previously-selected architecture
        (B_gbm_huber). The donor pool silently comes back empty; the
        pipeline must raise instead of feeding it to `_sample_block`, which
        would otherwise fall back to all-zero residual blocks.
        """
        df = pd.DataFrame({
            "player": ["P1"],
            "position": ["TE"],
            "season": [2022],
            "week": [1],
            "actual_ppr": [10.0],
            "prediction": [8.0],
            "model": ["B_gbm_huber"],  # stale architecture, not the current FINAL_CONFIG choice
        })
        csv_path = tmp_path / "phase4.csv"
        df.to_csv(csv_path, index=False)

        donor_pool = build_residual_donor_pools(
            [csv_path], "TE", "C_gbm_mae", before_season=2023,
        )
        assert donor_pool == {}

        with pytest.raises(EmptyDonorPoolError):
            _require_nonempty_donor_pool(donor_pool, "TE", 2023, "C_gbm_mae", [csv_path])

    def test_does_not_raise_when_pool_nonempty(self):
        pool = {("P1", 2021): [(1, 1.0)]}
        _require_nonempty_donor_pool(pool, "TE", 2023, "C_gbm_mae", [])  # no raise


class TestSampleBlock:
    def test_exact_length_from_long_donor(self):
        pool = {("P1", 2021): [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)]}
        rng = np.random.default_rng(0)
        block = _sample_block(pool, length=2, rng=rng)
        assert len(block) == 2
        # Must be a contiguous sub-sequence of the donor's actual residuals.
        donor_vals = [1.0, 2.0, 3.0, 4.0]
        assert any(
            list(block) == donor_vals[i:i + 2] for i in range(len(donor_vals) - 1)
        )

    def test_falls_back_to_iid_draw_when_no_donor_long_enough(self):
        pool = {("P1", 2021): [(1, 5.0)]}
        rng = np.random.default_rng(0)
        block = _sample_block(pool, length=3, rng=rng)
        assert len(block) == 3
        assert all(v == 5.0 for v in block)  # only residual value available

    def test_empty_pool_returns_zeros(self):
        rng = np.random.default_rng(0)
        block = _sample_block({}, length=2, rng=rng)
        assert list(block) == [0.0, 0.0]


class TestSimulatePlayerSeason:
    def test_no_synthetic_weeks_returns_constant_known_total(self):
        week_predictions = [
            {"week": 1, "is_real": True, "data_source": "nflverse_stats", "actual_value": 10.0, "point_prediction": 9.0},
            {"week": 2, "is_real": True, "data_source": "inferred_snap_verified_zero", "actual_value": 0.0, "point_prediction": 2.0},
        ]
        rng = np.random.default_rng(0)
        totals = simulate_player_season(week_predictions, availability_rate=0.9, donor_pool={}, n_sims=100, rng=rng)
        assert len(totals) == 100
        assert np.all(totals == 10.0)  # 10.0 + 0.0, real values only

    def test_availability_rate_zero_never_plays_synthetic_weeks(self):
        week_predictions = [
            {"week": 1, "is_real": False, "data_source": None, "actual_value": None, "point_prediction": 15.0},
        ]
        pool = {("P1", 2021): [(1, 2.0), (2, -1.0), (3, 3.0)]}
        rng = np.random.default_rng(0)
        totals = simulate_player_season(week_predictions, availability_rate=0.0, donor_pool=pool, n_sims=50, rng=rng)
        assert np.all(totals == 0.0)

    def test_availability_rate_one_always_plays_synthetic_weeks(self):
        week_predictions = [
            {"week": 1, "is_real": False, "data_source": None, "actual_value": None, "point_prediction": 15.0},
        ]
        pool = {("P1", 2021): [(1, 0.0)]}  # zero residual -> exactly the point prediction
        rng = np.random.default_rng(0)
        totals = simulate_player_season(week_predictions, availability_rate=1.0, donor_pool=pool, n_sims=50, rng=rng)
        assert np.all(totals == 15.0)

    def test_mixed_real_and_synthetic_weeks_sum_correctly(self):
        week_predictions = [
            {"week": 1, "is_real": True, "data_source": "nflverse_stats", "actual_value": 20.0, "point_prediction": 18.0},
            {"week": 2, "is_real": False, "data_source": None, "actual_value": None, "point_prediction": 10.0},
        ]
        pool = {("P1", 2021): [(1, 0.0)]}
        rng = np.random.default_rng(0)
        totals = simulate_player_season(week_predictions, availability_rate=1.0, donor_pool=pool, n_sims=50, rng=rng)
        # Real week's 20.0 always included; synthetic week always plays (rate=1.0)
        # with a zero-residual donor -> exactly 10.0 added.
        assert np.all(totals == 30.0)

    def test_quantiles_are_monotonic(self):
        week_predictions = [
            {"week": w, "is_real": False, "data_source": None, "actual_value": None, "point_prediction": 12.0}
            for w in range(1, 6)
        ]
        pool = {
            (f"P{i}", 2021): [(w, float(np.sin(w + i))) for w in range(1, 6)]
            for i in range(20)
        }
        rng = np.random.default_rng(1)
        totals = simulate_player_season(week_predictions, availability_rate=0.85, donor_pool=pool, n_sims=500, rng=rng)
        p25, p50, p75, p90 = np.percentile(totals, [25, 50, 75, 90])
        assert p25 <= p50 <= p75 <= p90

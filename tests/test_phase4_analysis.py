"""Unit tests for Phase 4 (next_focus.md) tier/bucket/analysis logic.

Synthetic data only — no DB/network (except compute_tier_metrics's DB call,
tested separately by monkeypatching _load_full_history_for_tiers).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.tiers import (
    PLAYER_TIER_PPG_EDGES,
    assign_player_tier,
    assign_score_bucket,
    compute_prior_season_ppg,
)
from src.models.single_week_ppr import analysis


class TestAssignPlayerTier:
    def test_rookie_when_no_prior_ppg(self):
        assert assign_player_tier(None, "WR") == "rookie"
        assert assign_player_tier(np.nan, "WR") == "rookie"

    @pytest.mark.parametrize("position,ppg,expected", [
        ("QB", 20.0, "elite"),
        ("QB", 15.0, "starter"),
        ("QB", 8.0, "depth"),
        ("QB", 2.0, "waiver"),
        ("TE", 10.0, "elite"),  # TE elite threshold much lower than QB's
        ("TE", 1.0, "waiver"),
    ])
    def test_position_adjusted_edges(self, position, ppg, expected):
        assert assign_player_tier(ppg, position) == expected

    def test_qb_and_te_edges_differ(self):
        """The whole point of position-adjustment: same PPG, different tier."""
        assert assign_player_tier(10.0, "QB") != assign_player_tier(10.0, "TE")

    def test_unknown_position_is_rookie(self):
        assert assign_player_tier(15.0, "K") == "rookie"


class TestAssignScoreBucket:
    @pytest.mark.parametrize("value,expected", [
        (3.0, "0-5"), (7.0, "5-10"), (12.0, "10-15"), (18.0, "15-20"), (25.0, "20+"),
        (-2.0, "0-5"),  # negative single-game PPR floors into the bottom bucket
    ])
    def test_buckets(self, value, expected):
        assert assign_score_bucket(value) == expected

    def test_nan_is_unknown(self):
        assert assign_score_bucket(np.nan) == "unknown"


class TestComputePriorSeasonPpg:
    def test_leakage_safe_shift(self):
        df = pd.DataFrame({
            "player_id": ["P1"] * 4,
            "season": [2022, 2022, 2023, 2023],
            "fantasy_points": [10.0, 20.0, 100.0, 200.0],  # 2022 avg=15, 2023 avg=150
        })
        result = compute_prior_season_ppg(df)
        season_2023_rows = result[result["season"] == 2023]
        assert (season_2023_rows["prior_season_ppg"] == 15.0).all()
        # 2022 rows have no prior season in this data -> NaN, not leaked from 2023.
        season_2022_rows = result[result["season"] == 2022]
        assert season_2022_rows["prior_season_ppg"].isna().all()

    def test_never_uses_same_or_future_season(self):
        """A player who only ever appears in the season being evaluated
        must get NaN (rookie), never their own current-season average."""
        df = pd.DataFrame({
            "player_id": ["ROOKIE"],
            "season": [2024],
            "fantasy_points": [999.0],
        })
        result = compute_prior_season_ppg(df)
        assert result["prior_season_ppg"].isna().all()


class TestAnalysisFunctions:
    @pytest.fixture
    def sample_predictions(self):
        return pd.DataFrame({
            "player": ["P1", "P1", "P2", "P2"],
            "position": ["WR", "WR", "WR", "WR"],
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 2, 1, 2],
            "actual_ppr": [10.0, 12.0, 3.0, 4.0],
            "prediction": [9.0, 11.0, 4.0, 3.5],
            "model": ["C_gbm_mae"] * 4,
            "fold": [2024] * 4,
            "training_window": ["all"] * 4,
            "weighting_strategy": ["none"] * 4,
            "p25": [np.nan] * 4, "p50": [np.nan] * 4, "p75": [np.nan] * 4, "p90": [np.nan] * 4,
        })

    def test_summarize_shapes(self, sample_predictions):
        report = analysis.summarize(sample_predictions)
        assert set(report.keys()) == {"overall", "by_position", "by_season"}
        assert len(report["overall"]) == 1  # one model
        assert "mae" in report["overall"].columns

    def test_bucket_metrics(self, sample_predictions):
        result = analysis.compute_bucket_metrics(sample_predictions)
        assert "bucket" in result.columns
        # predictions 9,11,4,3.5 -> buckets 5-10, 10-15, 0-5, 0-5
        assert set(result["bucket"]) == {"5-10", "10-15", "0-5"}

    def test_tier_metrics_uses_full_history(self, sample_predictions, monkeypatch):
        """Monkeypatch the DB-backed history loader so this stays a unit
        test — verifies compute_tier_metrics wires prior-season PPG into
        tier assignment correctly."""
        fake_history = pd.DataFrame({
            "player_id": ["P1", "P2", "P1", "P2"],
            "season": [2023, 2023, 2024, 2024],
            "position": ["WR", "WR", "WR", "WR"],
            # 2023 rows set prior_season_ppg for the 2024 rows; 2024 values
            # here are irrelevant to tiering (only prior-season counts).
            "fantasy_points": [20.0, 1.0, 999.0, 999.0],
        })
        monkeypatch.setattr(analysis, "_load_full_history_for_tiers", lambda positions: fake_history)

        result = analysis.compute_tier_metrics(sample_predictions)
        tiers_seen = set(result["tier"])
        assert "elite" in tiers_seen or "starter" in tiers_seen
        assert "waiver" in tiers_seen

    def test_calibration_report_only_uses_quantile_rows(self, sample_predictions):
        qrow = sample_predictions.iloc[[0]].copy()
        qrow["model"] = "E_quantile_gbm"
        qrow[["p25", "p50", "p75", "p90"]] = [[5.0, 9.0, 12.0, 15.0]]
        combined = pd.concat([sample_predictions, qrow], ignore_index=True)

        result = analysis.compute_calibration_report(combined)
        assert len(result) == 1  # only the one (position, season) with quantile rows
        assert "p50_coverage" in result.columns

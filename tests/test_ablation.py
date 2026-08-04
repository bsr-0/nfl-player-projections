"""Tests for src/evaluation/ablation.py — ablation study utilities."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import (
    identify_utilization_columns,
    identify_rank_columns,
    _compute_ablation_summary,
    format_ablation_report,
)


class TestIdentifyUtilizationColumns:
    def test_matches_exact(self):
        cols = ["utilization_score", "rushing_yards", "fantasy_points"]
        result = identify_utilization_columns(cols)
        assert "utilization_score" in result
        assert "rushing_yards" not in result
        assert "fantasy_points" not in result

    def test_matches_derived_rolling(self):
        cols = [
            "utilization_score_roll3_mean",
            "utilization_score_roll5_mean",
            "utilization_score_lag_1",
            "rushing_yards_roll3_mean",
        ]
        result = identify_utilization_columns(cols)
        assert "utilization_score_roll3_mean" in result
        assert "utilization_score_roll5_mean" in result
        assert "utilization_score_lag_1" in result
        assert "rushing_yards_roll3_mean" not in result

    def test_matches_position_derived(self):
        cols = [
            "pos_util_expanding_mean",
            "player_util_ewm_5",
            "util_regression_to_mean",
            "total_yards",
        ]
        result = identify_utilization_columns(cols)
        assert "pos_util_expanding_mean" in result
        assert "player_util_ewm_5" in result
        assert "util_regression_to_mean" in result
        assert "total_yards" not in result

    def test_empty_input(self):
        assert identify_utilization_columns([]) == []


class TestIdentifyRankColumns:
    def test_matches_exact(self):
        cols = ["season_position_rank", "rushing_yards", "fantasy_points"]
        result = identify_rank_columns(cols)
        assert "season_position_rank" in result
        assert "rushing_yards" not in result

    def test_matches_adp_variants(self):
        cols = ["estimated_adp", "projected_adp", "position_rank", "target_share"]
        result = identify_rank_columns(cols)
        assert "estimated_adp" in result
        assert "projected_adp" in result
        assert "position_rank" in result
        assert "target_share" not in result

    def test_empty_input(self):
        assert identify_rank_columns([]) == []


class TestComputeAblationSummary:
    def _make_results(self, full_rmse=5.0, no_rank_rmse=5.5, no_util_rmse=6.0, no_both_rmse=7.0):
        return {
            "full": {"QB": {"rmse": full_rmse, "mae": 3.0, "r2": 0.5, "n_test": 100}},
            "no_rank": {"QB": {"rmse": no_rank_rmse, "mae": 3.5, "r2": 0.4, "n_test": 100}},
            "no_util": {"QB": {"rmse": no_util_rmse, "mae": 4.0, "r2": 0.3, "n_test": 100}},
            "no_rank_no_util": {"QB": {"rmse": no_both_rmse, "mae": 4.5, "r2": 0.2, "n_test": 100}},
        }

    def test_computes_deltas(self):
        results = self._make_results()
        summary = _compute_ablation_summary(results, ["QB"])
        assert "no_rank" in summary
        assert "no_util" in summary
        assert "no_rank_no_util" in summary
        assert summary["no_rank"]["QB"]["rmse_delta"] == pytest.approx(0.5, abs=0.01)
        assert summary["no_rank"]["QB"]["feature_contributes"] is True

    def test_large_degradation_verdict(self):
        # >15% degradation when both removed
        results = self._make_results(full_rmse=5.0, no_both_rmse=6.0)  # 20% degradation
        summary = _compute_ablation_summary(results, ["QB"])
        assert "DEPENDS HEAVILY" in summary["verdict"]

    def test_minimal_degradation_verdict(self):
        # <5% degradation when both removed
        results = self._make_results(full_rmse=5.0, no_both_rmse=5.1)  # 2% degradation
        summary = _compute_ablation_summary(results, ["QB"])
        assert "REAL EDGE" in summary["verdict"]

    def test_mixed_verdict(self):
        # 5-15% degradation
        results = self._make_results(full_rmse=5.0, no_both_rmse=5.5)  # 10% degradation
        summary = _compute_ablation_summary(results, ["QB"])
        assert "MIXED" in summary["verdict"]


class TestFormatAblationReport:
    def test_produces_string(self):
        results = {
            "full": {"QB": {"rmse": 5.0, "mae": 3.0, "r2": 0.5, "n_test": 100}},
            "no_rank": {"QB": {"rmse": 5.5, "mae": 3.5, "r2": 0.4, "n_test": 100}},
            "no_util": {"QB": {"rmse": 6.0, "mae": 4.0, "r2": 0.3, "n_test": 100}},
            "no_rank_no_util": {"QB": {"rmse": 7.0, "mae": 4.5, "r2": 0.2, "n_test": 100}},
            "summary": {
                "no_rank": {"QB": {"rmse_delta": 0.5, "rmse_pct_change": 10.0, "feature_contributes": True}},
                "no_util": {"QB": {"rmse_delta": 1.0, "rmse_pct_change": 20.0, "feature_contributes": True}},
                "no_rank_no_util": {"QB": {"rmse_delta": 2.0, "rmse_pct_change": 40.0, "feature_contributes": True}},
                "verdict": "MODEL DEPENDS HEAVILY ON RANK/UTIL",
            },
        }
        report = format_ablation_report(results, ["QB"])
        assert isinstance(report, str)
        assert "ABLATION" in report
        assert "QB" in report
        assert "VERDICT" in report

    def test_handles_missing_positions(self):
        results = {
            "full": {},
            "no_rank": {},
            "no_util": {},
            "no_rank_no_util": {},
            "summary": {"verdict": "No data available"},
        }
        report = format_ablation_report(results, ["QB"])
        assert "N/A" in report

"""Tests for confidence-interval behavior in backtester."""

from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.backtester import ModelBacktester, check_success_criteria


def test_confidence_interval_custom_column_names():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "position": ["RB", "RB", "WR", "WR"],
        "predicted_points": [10.0, 14.0, 9.0, 13.0],
        "fantasy_points": [12.0, 13.0, 8.5, 15.0],
    })
    out = backtester.calculate_confidence_intervals(
        df,
        pred_col="predicted_points",
        actual_col="fantasy_points",
        confidence=0.8,
        lower_col="prediction_ci80_lower",
        upper_col="prediction_ci80_upper",
    )
    assert "prediction_ci80_lower" in out.columns
    assert "prediction_ci80_upper" in out.columns
    assert (out["prediction_ci80_lower"] >= 0).all()


def test_success_criteria_prefers_explicit_ci_coverage():
    payload = {
        "metrics": {
            "spearman_rho": 0.7,
            "mape": 20.0,
            "within_7_pts_pct": 75.0,
            "within_10_pts_pct": 82.0,
            "tier_classification_accuracy": 0.8,
            "std_predicted": 7.0,
            "std_actual": 8.0,
            "mae_rmse_ratio": 0.77,
            "mae_rmse_healthy": True,
        },
        "by_week": {"1": {"rmse": 7.0}, "2": {"rmse": 7.4}},
        "by_position": {},
        "multiple_baseline_comparison": {
            "model_beats_all_by_20_pct": True,
            "model_beats_all_by_25_pct": True,
            "baseline_season_avg": {"improvement_pct": 26.0},
        },
        "confidence_band_coverage_10pt": 90.0,
    }
    sc = check_success_criteria(payload)
    assert sc["confidence_band_coverage_10pt"] == 90.0
    assert sc["confidence_band_target_882"] is True


def test_compare_to_multiple_baselines_returns_expected_keys():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
        "position": ["RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE"],
        "season": [2024] * 10,
        "week": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "fantasy_points": [10, 13, 9, 8, 14, 16, 6, 7, 11, 12],
        "predicted_points": [10.5, 12.5, 9.5, 8.5, 13.5, 15.0, 6.5, 7.0, 10.5, 11.5],
    })
    out = backtester.compare_to_multiple_baselines(df)
    assert "model" in out
    assert "baseline_persistence" in out
    assert "baseline_season_avg" in out
    assert "baseline_position_avg" in out
    assert "model_beats_all_by_20_pct" in out


def test_compare_to_expert_consensus_with_csv(tmp_path):
    backtester = ModelBacktester()
    preds = pd.DataFrame({
        "name": [f"Player{i}" for i in range(12)],
        "fantasy_points": np.linspace(8, 20, 12),
        "predicted_points": np.linspace(8.5, 19.5, 12),
    })
    expert_path = tmp_path / "expert.csv"
    expert = pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(12)],
        "proj_points": np.linspace(7.0, 18.0, 12),
    })
    expert.to_csv(expert_path, index=False)
    out = backtester.compare_to_expert_consensus(preds, str(expert_path))
    assert "model_rmse" in out
    assert "expert_rmse" in out
    assert "model_vs_expert_pct" in out

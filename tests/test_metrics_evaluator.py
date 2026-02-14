"""Tests for evaluation metrics correctness and report safety."""

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import ModelEvaluator, compare_to_expert_consensus


class _ArrayModel:
    """Simple deterministic model for metric tests."""

    def __init__(self, preds: np.ndarray):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X):
        return self._preds[: len(X)]


def test_evaluate_model_uses_boom_bust_and_vor_with_positions():
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6]})
    y = pd.Series([8.0, 12.0, 4.0, 22.0, 9.0, 15.0])
    preds = np.array([7.5, 11.0, 5.0, 21.0, 8.0, 14.0])
    positions = pd.Series(["RB", "RB", "WR", "WR", "TE", "TE"])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        _ArrayModel(preds),
        X,
        y,
        positions=positions,
    )

    assert "boom_precision" in metrics
    assert "bust_precision" in metrics
    assert "vor_accuracy" in metrics
    assert np.isfinite(metrics["vor_accuracy"])


def test_evaluate_model_expert_comparison_by_position_average():
    X = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y = pd.Series([10.0, 12.0, 14.0, 16.0])
    preds = np.array([10.5, 11.5, 14.2, 15.8])
    positions = pd.Series(["QB", "QB", "RB", "RB"])

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        _ArrayModel(preds),
        X,
        y,
        positions=positions,
        expert_rmse_by_position={"QB": 2.5, "RB": 2.0},
    )

    assert "improvement_over_expert_by_pos_avg_pct" in metrics
    assert np.isfinite(metrics["improvement_over_expert_by_pos_avg_pct"])


def test_generate_report_formats_tier_accuracy_as_percent():
    X = pd.DataFrame({"f1": [1, 2, 3, 4]})
    y = pd.Series([6.0, 10.0, 15.0, 20.0])
    preds = np.array([6.5, 10.5, 14.5, 19.5])
    positions = pd.Series(["WR", "WR", "TE", "TE"])

    evaluator = ModelEvaluator()
    report = evaluator.generate_report(
        _ArrayModel(preds),
        X,
        y,
        positions=positions,
        expert_rmse={"WR": 3.0, "TE": 3.5},
    )

    assert "Tier Classification Accuracy:" in report
    assert "Avg improvement vs expert by position:" in report


def test_evaluate_model_reports_core_error_metrics():
    X = pd.DataFrame({"f1": [1, 2, 3, 4, 5]})
    y = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    preds = np.array([9.0, 12.0, 15.0, 15.0, 19.0])
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(_ArrayModel(preds), X, y, model_name="core")
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert "mape" in metrics
    assert "r2" in metrics


def test_compare_to_expert_consensus_position_thresholds():
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    y_model = np.array([10.2, 12.1, 14.1, 16.2, 18.0])
    y_expert = np.array([9.0, 11.0, 13.0, 15.0, 17.0])
    out = compare_to_expert_consensus(y_true, y_model, y_expert, position="RB")
    assert out["position"] == "RB"
    assert "beat_expert_target_pct" in out
    assert "rmse_improvement_pct" in out

"""Tests for Directive V7 Section 13: Consolidated Evaluation Matrix."""
import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.evaluation_matrix import generate_evaluation_matrix


@pytest.fixture
def sample_metrics():
    return {
        "QB": {"rmse": 6.5, "mae": 4.2, "r2": 0.45, "mape": 32.1},
        "RB": {"rmse": 5.8, "mae": 3.9, "r2": 0.52, "mape": 28.5},
        "WR": {"rmse": 5.2, "mae": 3.5, "r2": 0.55, "mape": 26.0},
        "TE": {"rmse": 4.1, "mae": 2.8, "r2": 0.60, "mape": 22.0},
    }


class TestEvaluationMatrix:
    def test_generates_matrix(self, sample_metrics, tmp_path):
        output = tmp_path / "eval_matrix.json"
        result = generate_evaluation_matrix(
            training_metrics=sample_metrics,
            positions=["QB", "RB", "WR", "TE"],
            train_seasons=[2020, 2021, 2022, 2023],
            test_season=2024,
            feature_version="v8",
            experiment_id="exp_001",
            output_path=output,
        )
        assert "per_position" in result
        assert "QB" in result["per_position"]
        assert "cross_position_summary" in result
        assert output.exists()

    def test_per_position_structure(self, sample_metrics, tmp_path):
        result = generate_evaluation_matrix(
            training_metrics=sample_metrics,
            positions=["QB", "RB"],
            train_seasons=[2020],
            test_season=2024,
            output_path=tmp_path / "m.json",
        )
        qb = result["per_position"]["QB"]
        assert "predictive_accuracy" in qb
        assert "calibration" in qb
        assert "decision_utility" in qb
        assert "risk" in qb
        assert "stability" in qb
        assert qb["predictive_accuracy"]["rmse"] == 6.5

    def test_cross_position_summary(self, sample_metrics, tmp_path):
        result = generate_evaluation_matrix(
            training_metrics=sample_metrics,
            positions=["QB", "RB", "WR", "TE"],
            train_seasons=[2020],
            test_season=2024,
            output_path=tmp_path / "m.json",
        )
        summary = result["cross_position_summary"]
        assert summary["worst_rmse"] == 6.5  # QB
        assert summary["best_rmse"] == 4.1   # TE
        assert summary["positions_evaluated"] == 4

    def test_handles_missing_metrics(self, tmp_path):
        result = generate_evaluation_matrix(
            training_metrics={"QB": {}},
            positions=["QB"],
            train_seasons=[2020],
            test_season=2024,
            output_path=tmp_path / "m.json",
        )
        qb = result["per_position"]["QB"]
        assert qb["predictive_accuracy"]["rmse"] is None

    def test_writes_json(self, sample_metrics, tmp_path):
        output = tmp_path / "eval_matrix.json"
        generate_evaluation_matrix(
            training_metrics=sample_metrics,
            positions=["QB"],
            train_seasons=[2020],
            test_season=2024,
            output_path=output,
        )
        with open(output) as f:
            data = json.load(f)
        assert data["directive_section"] == "Section 13 — Required Evaluation Matrix"

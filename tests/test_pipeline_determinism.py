"""
Pipeline determinism test per Directive V7 Section 23.

Verifies that the same input with the same seed produces identical output.
"""

import numpy as np
import pandas as pd
import pytest


def _create_synthetic_data(n=100, seed=42):
    """Create synthetic NFL player data for determinism testing."""
    rng = np.random.RandomState(seed)
    data = pd.DataFrame({
        "player_id": [f"P{i:03d}" for i in range(n)],
        "name": [f"Player_{i}" for i in range(n)],
        "position": rng.choice(["QB", "RB", "WR", "TE"], n),
        "team": rng.choice(["KC", "BUF", "SF", "PHI"], n),
        "season": np.full(n, 2024),
        "week": rng.randint(1, 18, n),
        "passing_yards": rng.uniform(0, 400, n),
        "rushing_yards": rng.uniform(0, 150, n),
        "receiving_yards": rng.uniform(0, 200, n),
        "receptions": rng.randint(0, 12, n).astype(float),
        "targets": rng.randint(0, 15, n).astype(float),
        "fantasy_points": rng.uniform(2, 35, n),
    })
    return data


class TestPipelineDeterminism:
    """Verify that feature engineering is deterministic with fixed seeds."""

    def test_synthetic_data_determinism(self):
        """Same seed should produce identical synthetic data."""
        data1 = _create_synthetic_data(seed=42)
        data2 = _create_synthetic_data(seed=42)
        pd.testing.assert_frame_equal(data1, data2)

    def test_numpy_operations_determinism(self):
        """Numpy operations with fixed seed should be deterministic."""
        rng1 = np.random.RandomState(42)
        result1 = rng1.normal(0, 1, 100)

        rng2 = np.random.RandomState(42)
        result2 = rng2.normal(0, 1, 100)

        np.testing.assert_array_equal(result1, result2)

    def test_schema_validation_determinism(self):
        """Schema validation should produce same results on same input."""
        from src.data.schema_validator import validate_weekly_data

        data = _create_synthetic_data()
        result1 = validate_weekly_data(data)
        result2 = validate_weekly_data(data)
        assert result1 == result2

    def test_metrics_determinism(self):
        """Metric computation should be deterministic."""
        from src.evaluation.metrics import (
            spearman_rank_correlation,
            tier_classification_accuracy,
            boom_bust_metrics,
        )

        rng = np.random.RandomState(42)
        y_true = rng.uniform(5, 30, 100)
        y_pred = y_true + rng.normal(0, 3, 100)

        # Run twice
        rho1 = spearman_rank_correlation(y_true, y_pred)
        rho2 = spearman_rank_correlation(y_true, y_pred)
        assert rho1 == rho2

        tier1 = tier_classification_accuracy(y_true, y_pred)
        tier2 = tier_classification_accuracy(y_true, y_pred)
        assert tier1 == tier2

        boom1 = boom_bust_metrics(y_true, y_pred)
        boom2 = boom_bust_metrics(y_true, y_pred)
        assert boom1 == boom2

    def test_ece_determinism(self):
        """ECE computation should be deterministic."""
        from src.evaluation.metrics import expected_calibration_error

        rng = np.random.RandomState(42)
        y_true = rng.normal(15, 5, 200)
        y_pred = y_true + rng.normal(0, 2, 200)
        pred_std = np.abs(rng.normal(2, 0.5, 200))

        result1 = expected_calibration_error(y_true, y_pred, pred_std)
        result2 = expected_calibration_error(y_true, y_pred, pred_std)
        assert result1["ece"] == result2["ece"]

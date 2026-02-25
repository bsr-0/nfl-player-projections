"""Tests for distributional predictions (boom/bust probabilities)."""
import numpy as np
import pytest

from src.evaluation.distributional import (
    boom_bust_probabilities,
    classify_risk_tier,
    format_distributional_summary,
    BOOM_THRESHOLDS,
    BUST_THRESHOLDS,
)


class TestBoomBustProbabilities:
    def test_returns_expected_keys(self):
        preds = np.array([20.0, 10.0, 5.0])
        stds = np.array([3.0, 3.0, 3.0])
        positions = np.array(["QB", "RB", "WR"])
        result = boom_bust_probabilities(preds, stds, positions)
        assert "boom_prob" in result
        assert "bust_prob" in result
        assert "safe_floor" in result
        assert "upside_ceiling" in result

    def test_output_lengths_match_input(self):
        n = 10
        preds = np.random.uniform(5, 25, n)
        stds = np.full(n, 3.0)
        positions = np.array(["RB"] * n)
        result = boom_bust_probabilities(preds, stds, positions)
        for key in ("boom_prob", "bust_prob", "safe_floor", "upside_ceiling"):
            assert len(result[key]) == n

    def test_high_prediction_has_high_boom_prob(self):
        # QB predicted at 30 with std=3 should have high boom probability
        preds = np.array([30.0])
        stds = np.array([3.0])
        positions = np.array(["QB"])
        result = boom_bust_probabilities(preds, stds, positions)
        # P(X > 25) with mean=30, std=3 should be > 0.9
        assert result["boom_prob"][0] > 0.9

    def test_low_prediction_has_high_bust_prob(self):
        # RB predicted at 2 with std=2 should have high bust probability
        preds = np.array([2.0])
        stds = np.array([2.0])
        positions = np.array(["RB"])
        result = boom_bust_probabilities(preds, stds, positions)
        # P(X < 5) with mean=2, std=2 should be > 0.9
        assert result["bust_prob"][0] > 0.9

    def test_probabilities_are_valid(self):
        preds = np.random.uniform(5, 25, 20)
        stds = np.full(20, 4.0)
        positions = np.array(["QB", "RB", "WR", "TE"] * 5)
        result = boom_bust_probabilities(preds, stds, positions)
        assert np.all(result["boom_prob"] >= 0) and np.all(result["boom_prob"] <= 1)
        assert np.all(result["bust_prob"] >= 0) and np.all(result["bust_prob"] <= 1)

    def test_floor_below_ceiling(self):
        preds = np.array([15.0, 10.0])
        stds = np.array([3.0, 5.0])
        positions = np.array(["RB", "WR"])
        result = boom_bust_probabilities(preds, stds, positions)
        assert np.all(result["safe_floor"] < result["upside_ceiling"])


class TestClassifyRiskTier:
    def test_high_upside(self):
        # High boom, low bust
        tiers = classify_risk_tier(np.array([0.5]), np.array([0.05]))
        assert tiers[0] == "high_upside"

    def test_safe_floor(self):
        # Low boom, low bust
        tiers = classify_risk_tier(np.array([0.05]), np.array([0.05]))
        assert tiers[0] == "safe_floor"

    def test_volatile(self):
        # High boom AND high bust
        tiers = classify_risk_tier(np.array([0.3]), np.array([0.3]))
        assert tiers[0] == "volatile"

    def test_output_length(self):
        n = 15
        tiers = classify_risk_tier(np.random.uniform(0, 0.5, n), np.random.uniform(0, 0.5, n))
        assert len(tiers) == n


class TestFormatDistributionalSummary:
    def test_produces_string(self):
        preds = np.array([20.0, 15.0, 10.0])
        stds = np.array([3.0, 4.0, 5.0])
        positions = np.array(["QB", "RB", "WR"])
        names = np.array(["Player A", "Player B", "Player C"])
        report = format_distributional_summary(preds, stds, positions, names)
        assert isinstance(report, str)
        assert "Player A" in report
        assert "DISTRIBUTIONAL" in report

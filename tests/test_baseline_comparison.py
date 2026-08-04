"""Tests for baseline comparison integration in backtest reports.

Every backtest report must include baseline metrics so R² is contextualized
against what simple methods achieve. A model that can't beat a trailing
3-game average has no value.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.baselines import (
    trailing_average_baseline,
    season_average_baseline,
    expert_consensus_baseline,
)
from src.evaluation.ts_backtester import TimeSeriesBacktester, default_model_factory


# ---------------------------------------------------------------------------
# Baseline correctness tests
# ---------------------------------------------------------------------------

class TestBaselineCorrectness:
    """Verify that baseline implementations produce correct predictions."""

    @pytest.fixture()
    def sample_data(self):
        """Minimal player-week data for baseline testing."""
        rows = []
        for pid in ["p1", "p2"]:
            for week in range(1, 11):
                rows.append({
                    "player_id": pid,
                    "season": 2024,
                    "week": week,
                    "position": "RB",
                    "fantasy_points": float(10 + week + (5 if pid == "p1" else 0)),
                })
        return pd.DataFrame(rows)

    def test_trailing_average_uses_only_past(self, sample_data):
        """Trailing average must not use current or future week data."""
        preds = trailing_average_baseline(sample_data, n_weeks=3)
        # For player p1, week 5 (fp=20): trailing avg of weeks 2,3,4 = (17+18+19)/3 = 18
        p1_week5 = sample_data[
            (sample_data["player_id"] == "p1") & (sample_data["week"] == 5)
        ]
        idx = p1_week5.index[0]
        expected = (17 + 18 + 19) / 3  # weeks 2, 3, 4 for p1
        assert abs(preds.loc[idx] - expected) < 0.01, (
            f"Trailing avg for p1 week 5 should be {expected}, got {preds.loc[idx]}"
        )

    def test_trailing_average_no_future_leak(self, sample_data):
        """Week 1 prediction should only use week 0 data (none available),
        so it should be NaN or use expanding mean of available history."""
        preds = trailing_average_baseline(sample_data, n_weeks=3)
        p1_week1 = sample_data[
            (sample_data["player_id"] == "p1") & (sample_data["week"] == 1)
        ]
        idx = p1_week1.index[0]
        # Week 1 has no prior data, so prediction should be NaN
        assert pd.isna(preds.loc[idx]), (
            f"Week 1 should have no prior data for trailing avg, got {preds.loc[idx]}"
        )

    def test_season_average_shifted(self, sample_data):
        """Season average must be shifted by 1 (no current week)."""
        preds = season_average_baseline(sample_data)
        p1_week3 = sample_data[
            (sample_data["player_id"] == "p1") & (sample_data["week"] == 3)
        ]
        idx = p1_week3.index[0]
        # Season avg through week 2 for p1: (16 + 17) / 2 = 16.5
        expected = (16 + 17) / 2
        assert abs(preds.loc[idx] - expected) < 0.01

    def test_baselines_same_length_as_input(self, sample_data):
        """All baselines must return a Series aligned with input DataFrame."""
        for name, fn in [
            ("trailing", lambda df: trailing_average_baseline(df, n_weeks=3)),
            ("season", season_average_baseline),
            ("expert", expert_consensus_baseline),
        ]:
            result = fn(sample_data)
            assert len(result) == len(sample_data), (
                f"{name} baseline returned {len(result)} rows, "
                f"expected {len(sample_data)}"
            )


# ---------------------------------------------------------------------------
# Backtest report integration tests
# ---------------------------------------------------------------------------

class TestBacktestReportBaselines:
    """Verify that ts_backtester includes baselines in reports."""

    def test_compute_baseline_comparison_returns_dict(self):
        """compute_baseline_comparison should return a dict with baseline keys."""
        # Create minimal backtester with synthetic data
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "player_id": [f"p{i % 20}" for i in range(n)],
            "season": [2023] * 100 + [2024] * 100,
            "week": list(range(1, 11)) * 20,
            "position": ["RB"] * n,
            "fantasy_points": np.random.uniform(3, 25, n),
        })

        bt = TimeSeriesBacktester(
            data=data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["RB"],
            verbose=False,
        )
        # Simulate predictions (without actually running backtest)
        bt.predictions = []
        season_data = data[data["season"] == 2024]
        for _, row in season_data.iterrows():
            bt.predictions.append({
                "season": 2024,
                "week": int(row["week"]),
                "player_id": row["player_id"],
                "name": row["player_id"],
                "position": "RB",
                "team": "TST",
                "predicted": float(row["fantasy_points"]) + np.random.normal(0, 3),
                "actual": float(row["fantasy_points"]),
            })

        result = bt.compute_baseline_comparison()
        assert isinstance(result, dict), "compute_baseline_comparison must return a dict"
        assert "model" in result, "Result must include 'model' metrics"
        # At least one baseline should be present
        baseline_keys = [k for k in result.keys() if k != "model"]
        assert len(baseline_keys) > 0, (
            f"Result must include at least one baseline, got keys: {list(result.keys())}"
        )

    def test_baseline_metrics_have_expected_keys(self):
        """Each baseline entry should have rmse, mae, r2."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "player_id": [f"p{i % 20}" for i in range(n)],
            "season": [2023] * 100 + [2024] * 100,
            "week": list(range(1, 11)) * 20,
            "position": ["RB"] * n,
            "fantasy_points": np.random.uniform(3, 25, n),
        })

        bt = TimeSeriesBacktester(
            data=data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["RB"],
            verbose=False,
        )
        bt.predictions = []
        season_data = data[data["season"] == 2024]
        for _, row in season_data.iterrows():
            bt.predictions.append({
                "season": 2024,
                "week": int(row["week"]),
                "player_id": row["player_id"],
                "name": row["player_id"],
                "position": "RB",
                "team": "TST",
                "predicted": float(row["fantasy_points"]) + np.random.normal(0, 3),
                "actual": float(row["fantasy_points"]),
            })

        result = bt.compute_baseline_comparison()
        for key, metrics in result.items():
            if isinstance(metrics, dict) and "error" not in metrics:
                assert "rmse" in metrics, f"Baseline '{key}' missing 'rmse'"
                assert "mae" in metrics, f"Baseline '{key}' missing 'mae'"
                assert "r2" in metrics, f"Baseline '{key}' missing 'r2'"

    def test_results_dict_includes_baselines_key(self):
        """get_results_dict() must include a 'baselines' key."""
        np.random.seed(42)
        n = 200
        data = pd.DataFrame({
            "player_id": [f"p{i % 20}" for i in range(n)],
            "season": [2023] * 100 + [2024] * 100,
            "week": list(range(1, 11)) * 20,
            "position": ["RB"] * n,
            "fantasy_points": np.random.uniform(3, 25, n),
        })

        bt = TimeSeriesBacktester(
            data=data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["RB"],
            verbose=False,
        )
        # Simulate a minimal run by setting required attributes
        bt.predictions = []
        bt.weekly_metrics = {}
        bt.position_metrics = {}
        bt.overall_metrics = {"mae": 5.0, "rmse": 7.0, "r2": 0.1}
        bt.drawdown_metrics = {}
        bt._run_timestamp = "2024-01-01T00:00:00"

        season_data = data[data["season"] == 2024]
        for _, row in season_data.iterrows():
            bt.predictions.append({
                "season": 2024,
                "week": int(row["week"]),
                "player_id": row["player_id"],
                "name": row["player_id"],
                "position": "RB",
                "team": "TST",
                "predicted": float(row["fantasy_points"]) + np.random.normal(0, 3),
                "actual": float(row["fantasy_points"]),
            })

        results = bt.get_results_dict()
        assert "baselines" in results, (
            "get_results_dict() must include 'baselines' key for "
            "model-vs-baseline comparison"
        )

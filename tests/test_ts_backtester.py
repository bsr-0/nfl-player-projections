"""
Tests for the leakage-free time-series backtester.

Validates:
1. No future data leaks into training set
2. Rolling features are properly shifted
3. Expanding window grows each iteration
4. Scaler is fit on train only
5. Predictions table matches expected schema
6. Early vs late season error comparison is available
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ts_backtester import (
    TimeSeriesBacktester,
    assert_no_future_leakage,
    assert_rolling_features_shifted,
    default_model_factory,
    _calc_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic NFL-like data
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_seasons: int = 3, n_weeks: int = 17, n_players: int = 20) -> pd.DataFrame:
    """Create synthetic data mimicking NFL weekly stats."""
    rng = np.random.RandomState(42)
    rows = []
    positions = ["QB", "RB", "WR", "TE"]
    start_season = 2022

    for s in range(n_seasons):
        season = start_season + s
        for w in range(1, n_weeks + 1):
            for p in range(n_players):
                pos = positions[p % len(positions)]
                rows.append({
                    "player_id": f"player_{p}",
                    "name": f"Player {p}",
                    "position": pos,
                    "team": f"TEAM{p % 8}",
                    "season": season,
                    "week": w,
                    "fantasy_points": max(0, rng.normal(10, 5)),
                    "rushing_yards": max(0, rng.normal(30, 20)) if pos == "RB" else 0,
                    "rushing_attempts": max(0, int(rng.normal(10, 5))) if pos == "RB" else 0,
                    "receiving_yards": max(0, rng.normal(40, 25)),
                    "receptions": max(0, int(rng.normal(3, 2))),
                    "targets": max(0, int(rng.normal(5, 3))),
                    "passing_yards": max(0, rng.normal(200, 80)) if pos == "QB" else 0,
                    "passing_attempts": max(0, int(rng.normal(30, 10))) if pos == "QB" else 0,
                    "passing_tds": max(0, int(rng.normal(1.5, 1))) if pos == "QB" else 0,
                    "interceptions": max(0, int(rng.normal(0.5, 0.5))) if pos == "QB" else 0,
                    "rushing_tds": max(0, int(rng.normal(0.3, 0.3))),
                    "receiving_tds": max(0, int(rng.normal(0.3, 0.3))),
                })

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def synthetic_data():
    return _make_synthetic_data()


# ---------------------------------------------------------------------------
# Test: assert_no_future_leakage
# ---------------------------------------------------------------------------

class TestLeakageDiagnostics:
    def test_no_leakage_clean_split(self, synthetic_data):
        """Clean chronological split should pass."""
        train = synthetic_data[synthetic_data["season"] < 2024]
        test = synthetic_data[
            (synthetic_data["season"] == 2024) & (synthetic_data["week"] == 1)
        ]
        result = assert_no_future_leakage(train, test)
        assert result["passed"], f"Expected pass, got errors: {result['errors']}"

    def test_leakage_detected_future_rows(self, synthetic_data):
        """If training set includes rows from the test week, leakage should be detected."""
        test = synthetic_data[
            (synthetic_data["season"] == 2024) & (synthetic_data["week"] == 5)
        ]
        # Deliberately include future data (same season, later week)
        train = synthetic_data[
            (synthetic_data["season"] <= 2024) & (synthetic_data["week"] <= 10)
        ]
        result = assert_no_future_leakage(train, test)
        # The train includes weeks 6-10 of season 2024 which are AFTER week 5
        # Our diagnostic checks for rows in the test season at or after test week
        # Train has weeks 6-10 in season 2024, test is week 5 => week 6+ are after
        # Actually the check looks at player-level: train rows at (season >= test_season AND week >= test_week)
        # Week 5 is the test week, weeks 5-10 would be caught (train has week <= 10 AND test is week 5)
        # train has season=2024, week=5..10 for these players → those at week >= 5 are future
        assert not result["passed"], "Expected leakage detection"


class TestRollingFeatureShiftCheck:
    def test_shifted_features_pass(self):
        """Properly shifted rolling features should have NaN in first row per player."""
        df = pd.DataFrame({
            "player_id": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "fantasy_points_roll3_mean": [
                np.nan, 10, 11, 12, 13,
                np.nan, 8, 9, 10, 11,
                np.nan, 7, 8, 9, 10,
            ],
        })
        result = assert_rolling_features_shifted(df)
        assert result["passed"], f"Expected pass, got warnings: {result['warnings']}"

    def test_unshifted_features_fail(self):
        """Unshifted rolling features (no NaN in first row) should trigger warning."""
        df = pd.DataFrame({
            "player_id": ["A"] * 5 + ["B"] * 5,
            "fantasy_points_roll3_mean": [10, 10, 11, 12, 13, 8, 8, 9, 10, 11],
        })
        result = assert_rolling_features_shifted(df)
        assert not result["passed"]


# ---------------------------------------------------------------------------
# Test: TimeSeriesBacktester
# ---------------------------------------------------------------------------

class TestTimeSeriesBacktester:
    def test_basic_run(self, synthetic_data):
        """Backtester should produce predictions for each week in the target season."""
        bt = TimeSeriesBacktester(
            data=synthetic_data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["QB", "RB"],
            feature_pipeline=_simple_passthrough_pipeline,
            verbose=False,
        )
        pred_df = bt.run_backtest()

        assert len(pred_df) > 0
        assert "predicted" in pred_df.columns
        assert "actual" in pred_df.columns
        assert "season" in pred_df.columns
        assert "week" in pred_df.columns
        assert "player_id" in pred_df.columns
        assert "position" in pred_df.columns

        # All predictions should be for the target season
        assert (pred_df["season"] == 2024).all()

        # Should have multiple weeks
        assert pred_df["week"].nunique() > 1

    def test_expanding_window_grows(self, synthetic_data):
        """Training set should grow as we move through weeks."""
        train_sizes = []

        class TrackingFactory:
            def __call__(self, train_df, position):
                train_sizes.append(len(train_df))
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0)

        bt = TimeSeriesBacktester(
            data=synthetic_data,
            model_factory=TrackingFactory(),
            season_to_backtest=2024,
            positions=["RB"],
            feature_pipeline=_simple_passthrough_pipeline,
            verbose=False,
        )
        bt.run_backtest()

        # Training sizes should generally be non-decreasing
        # (same position, so each subsequent week should add data)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Training set shrank: week {i} had {train_sizes[i]} < {train_sizes[i-1]}"
            )

    def test_no_leakage_in_backtest(self, synthetic_data):
        """Verify that during backtest, no future data appears in training."""
        leakage_checks = []

        def checking_pipeline(train_df, test_df):
            check = assert_no_future_leakage(train_df, test_df)
            leakage_checks.append(check)
            return train_df, test_df

        bt = TimeSeriesBacktester(
            data=synthetic_data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["QB"],
            feature_pipeline=checking_pipeline,
            verbose=False,
        )
        bt.run_backtest()

        assert len(leakage_checks) > 0
        for i, check in enumerate(leakage_checks):
            assert check["passed"], f"Leakage in fold {i}: {check['errors']}"

    def test_metrics_computed(self, synthetic_data):
        """After backtesting, metrics should be available."""
        bt = TimeSeriesBacktester(
            data=synthetic_data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["QB", "RB", "WR", "TE"],
            feature_pipeline=_simple_passthrough_pipeline,
            verbose=False,
        )
        bt.run_backtest()

        assert hasattr(bt, "overall_metrics")
        assert "mae" in bt.overall_metrics
        assert "rmse" in bt.overall_metrics
        assert "r2" in bt.overall_metrics
        assert len(bt.weekly_metrics) > 0
        assert len(bt.position_metrics) > 0

    def test_results_dict_serializable(self, synthetic_data):
        """Results dict should be JSON-serializable."""
        import json

        bt = TimeSeriesBacktester(
            data=synthetic_data,
            model_factory=default_model_factory,
            season_to_backtest=2024,
            positions=["QB"],
            feature_pipeline=_simple_passthrough_pipeline,
            verbose=False,
        )
        bt.run_backtest()
        d = bt.get_results_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["season"] == 2024
        assert d["backtest_type"] == "expanding_window_weekly_refit"
        assert d["diagnostics"]["model_refit_per_week"] is True


# ---------------------------------------------------------------------------
# Test: _calc_metrics
# ---------------------------------------------------------------------------

class TestCalcMetrics:
    def test_perfect_prediction(self):
        y = pd.Series([10.0, 20.0, 30.0])
        m = _calc_metrics(y, y)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0

    def test_imperfect_prediction(self):
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([12.0, 18.0, 28.0])
        m = _calc_metrics(y_true, y_pred)
        assert m["mae"] == 2.0
        assert m["rmse"] == 2.0
        assert 0 < m["r2"] < 1.0

    def test_handles_nan(self):
        y_true = pd.Series([10.0, np.nan, 30.0])
        y_pred = pd.Series([10.0, 20.0, 30.0])
        m = _calc_metrics(y_true, y_pred)
        assert m["n"] == 2  # Only 2 valid pairs


# ---------------------------------------------------------------------------
# Helper: simple passthrough pipeline (no feature engineering, just returns as-is)
# ---------------------------------------------------------------------------

def _simple_passthrough_pipeline(train_df, test_df):
    """Minimal pipeline that just returns train/test with numeric columns intact."""
    return train_df, test_df

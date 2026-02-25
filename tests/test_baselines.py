"""Tests for src/evaluation/baselines.py — strong baseline strategies."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (
    trailing_average_baseline,
    season_average_baseline,
    positional_rank_baseline,
    compare_model_to_baselines,
    format_baseline_report,
)


def _make_weekly_data(n_players: int = 5, n_weeks: int = 10, seasons: int = 2) -> pd.DataFrame:
    """Create synthetic weekly player data for testing baselines."""
    rows = []
    for season in range(2022, 2022 + seasons):
        for pid in range(1, n_players + 1):
            for week in range(1, n_weeks + 1):
                rows.append({
                    "player_id": pid,
                    "season": season,
                    "week": week,
                    "position": "RB" if pid <= n_players // 2 else "WR",
                    "fantasy_points": float(10 + pid + np.sin(week) + season * 0.1),
                })
    return pd.DataFrame(rows)


class TestTrailingAverageBaseline:
    def test_returns_series(self):
        df = _make_weekly_data()
        result = trailing_average_baseline(df, n_weeks=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_no_future_leakage(self):
        """Trailing average must not use the current week's value."""
        df = _make_weekly_data(n_players=1, n_weeks=5, seasons=1)
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        result = trailing_average_baseline(df, n_weeks=3)
        # Week 5 (idx=4) prediction should be average of weeks 2,3,4 (shifted by 1)
        expected = df["fantasy_points"].iloc[1:4].mean()
        assert abs(result.iloc[4] - expected) < 0.01

    def test_handles_nan_gracefully(self):
        df = _make_weekly_data(n_players=1, n_weeks=3, seasons=1)
        df.loc[0, "fantasy_points"] = np.nan
        result = trailing_average_baseline(df, n_weeks=3)
        assert len(result) == len(df)


class TestSeasonAverageBaseline:
    def test_returns_series(self):
        df = _make_weekly_data()
        result = season_average_baseline(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_no_future_leakage(self):
        """Season average must not include current week."""
        df = _make_weekly_data(n_players=1, n_weeks=5, seasons=1)
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        result = season_average_baseline(df)
        # Week 4 (idx=3): average of weeks 1,2,3 (shifted)
        expected = df["fantasy_points"].iloc[0:3].mean()
        assert abs(result.iloc[3] - expected) < 0.01


class TestPositionalRankBaseline:
    def test_returns_series(self):
        df = _make_weekly_data(seasons=2)
        result = positional_rank_baseline(df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_uses_prior_season(self):
        """Second season predictions should be based on first season averages."""
        df = _make_weekly_data(n_players=1, n_weeks=5, seasons=2)
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        result = positional_rank_baseline(df)
        # For second season (rows 5-9), prediction should be first season average
        first_season_avg = df[df["season"] == 2022]["fantasy_points"].mean()
        second_season_preds = result.iloc[5:10]
        # All predictions for the second season should equal first season avg
        for val in second_season_preds:
            if np.isfinite(val):
                assert abs(val - first_season_avg) < 0.5


class TestCompareModelToBaselines:
    def test_returns_comparison_dict(self):
        df = _make_weekly_data(n_players=10, n_weeks=15, seasons=2)
        # Simple model predictions: just the trailing average + small noise
        model_preds = pd.Series(
            df["fantasy_points"].values + np.random.normal(0, 0.5, len(df)),
            index=df.index,
        )
        result = compare_model_to_baselines(df, model_preds)
        assert isinstance(result, dict)
        # Should have at least trailing_3g_avg, trailing_5g_avg, season_avg
        assert len(result) >= 3

    def test_contains_required_keys(self):
        df = _make_weekly_data(n_players=10, n_weeks=15, seasons=2)
        model_preds = pd.Series(df["fantasy_points"].values, index=df.index)
        result = compare_model_to_baselines(df, model_preds)
        for name, metrics in result.items():
            assert "baseline_rmse" in metrics
            assert "model_rmse" in metrics
            assert "rmse_improvement_pct" in metrics
            assert "model_beats_baseline" in metrics
            assert "n_compared" in metrics

    def test_perfect_model_beats_baselines(self):
        """A model that perfectly predicts should beat all baselines."""
        df = _make_weekly_data(n_players=10, n_weeks=15, seasons=2)
        # Perfect predictions
        model_preds = pd.Series(df["fantasy_points"].values, index=df.index)
        result = compare_model_to_baselines(df, model_preds)
        for name, metrics in result.items():
            assert metrics["model_beats_baseline"], f"Perfect model should beat {name}"


class TestFormatBaselineReport:
    def test_produces_string(self):
        comparison = {
            "trailing_3g_avg": {
                "baseline_rmse": 5.0,
                "baseline_mae": 3.5,
                "model_rmse": 4.0,
                "model_mae": 2.8,
                "rmse_improvement_pct": 20.0,
                "mae_improvement_pct": 20.0,
                "model_beats_baseline": True,
                "n_compared": 100,
            }
        }
        report = format_baseline_report(comparison)
        assert isinstance(report, str)
        assert "BEATS" in report
        assert "trailing_3g_avg" in report

    def test_warning_when_model_loses(self):
        comparison = {
            "season_avg": {
                "baseline_rmse": 3.0,
                "baseline_mae": 2.0,
                "model_rmse": 4.0,
                "model_mae": 3.0,
                "rmse_improvement_pct": -33.33,
                "mae_improvement_pct": -50.0,
                "model_beats_baseline": False,
                "n_compared": 100,
            }
        }
        report = format_baseline_report(comparison)
        assert "LOSES TO" in report

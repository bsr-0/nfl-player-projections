"""Tests for LOYO walk-forward backtest orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.backtester import run_loyo_backtest
from config.settings import LOYO_CONFIG


def _make_fold_result(season, rmse, mae, r2, corr, n_predictions=500):
    """Build a synthetic per-fold result dict matching backtest_season() schema."""
    return {
        "season": season,
        "n_predictions": n_predictions,
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": corr,
            "spearman_rho": corr - 0.05,
            "within_5_pts_pct": 40.0 + r2 * 20,
            "within_7_pts_pct": 55.0 + r2 * 15,
            "tier_classification_accuracy": 0.3 + r2 * 0.1,
            "mape": 100.0 - r2 * 50,
        },
        "by_position": {
            "QB": {"rmse": rmse - 1, "mae": mae - 0.5, "r2": r2 + 0.05, "correlation": corr + 0.1},
            "RB": {"rmse": rmse + 2, "mae": mae + 1, "r2": r2 - 0.1, "correlation": corr - 0.05},
            "WR": {"rmse": rmse + 1, "mae": mae + 0.5, "r2": r2 - 0.15, "correlation": corr - 0.1},
            "TE": {"rmse": rmse, "mae": mae, "r2": r2 - 0.05, "correlation": corr},
        },
        "by_week": {},
        "ranking_accuracy": {},
    }


# Simulated per-season quality (rmse, mae, r2, correlation)
_SEASON_QUALITY = {
    2018: (8.5, 6.0, 0.15, 0.40),
    2019: (8.0, 5.8, 0.20, 0.45),
    2020: (9.5, 7.0, 0.10, 0.35),
    2021: (7.8, 5.5, 0.25, 0.50),
    2022: (7.5, 5.3, 0.28, 0.52),
    2023: (8.2, 5.9, 0.18, 0.42),
    2024: (9.0, 6.5, 0.12, 0.38),
}


def _mock_fold_runner(td, td_test, tr_ss, ts, positions, tune, n_trials):
    """Mock _run_one_fold: returns a synthetic result dict for the given test season."""
    rmse, mae, r2, corr = _SEASON_QUALITY.get(ts, (8.0, 6.0, 0.20, 0.45))
    return MagicMock(), _make_fold_result(ts, rmse, mae, r2, corr)


def _mock_data_loader(positions, test_season=None, optimize_training_years=False,
                      strict_requirements=None):
    """Mock load_training_data: returns minimal DataFrames."""
    train = pd.DataFrame({"season": [2010, 2011], "player_id": ["a", "b"]})
    test = pd.DataFrame({
        "season": [test_season] * 30,
        "player_id": [f"p{i}" for i in range(30)],
    })
    train_seasons = list(range(2006, test_season))
    return train, test, train_seasons, test_season


def _run(test_seasons, positions=None, **kwargs):
    """Helper: run LOYO with mocked fold runner and data loader."""
    return run_loyo_backtest(
        test_seasons=test_seasons,
        positions=positions or ["QB"],
        tune_hyperparameters=False,
        _fold_runner=kwargs.pop("_fold_runner", _mock_fold_runner),
        _data_loader=kwargs.pop("_data_loader", _mock_data_loader),
        **kwargs,
    )


class TestLOYOSeasonRange:
    def test_default_range_respects_config(self):
        assert LOYO_CONFIG["default_test_seasons_start"] == 2020
        assert LOYO_CONFIG["min_train_seasons"] == 5
        assert LOYO_CONFIG["purge_gap"] == 1

    def test_explicit_seasons_override(self):
        result = _run([2022, 2023])
        assert result["test_seasons"] == [2022, 2023]
        assert result["n_folds"] == 2

    def test_skips_fold_with_insufficient_test_data(self):
        def load_small(positions, test_season=None, optimize_training_years=False,
                       strict_requirements=None):
            if test_season == 2020:
                return (
                    pd.DataFrame({"season": [2019]}),
                    pd.DataFrame({"season": [2020] * 5, "player_id": ["a"] * 5}),
                    list(range(2006, 2020)),
                    2020,
                )
            return _mock_data_loader(positions, test_season=test_season)

        result = _run([2020, 2021], _data_loader=load_small)
        assert result["n_folds"] == 1
        assert result["test_seasons"] == [2021]


class TestLOYOAggregation:
    def test_aggregate_metrics_computed(self):
        result = _run([2021, 2022, 2023], positions=["QB", "RB"])
        agg = result["aggregate"]["overall"]
        assert "rmse" in agg
        assert "mean" in agg["rmse"]
        assert "std" in agg["rmse"]
        assert "min" in agg["rmse"]
        assert "max" in agg["rmse"]
        assert agg["rmse"]["min"] <= agg["rmse"]["mean"] <= agg["rmse"]["max"]

    def test_per_position_aggregation(self):
        result = _run([2021, 2022], positions=["QB", "RB", "WR", "TE"])
        by_pos = result["aggregate"]["by_position"]
        assert "QB" in by_pos
        assert "RB" in by_pos
        assert "rmse" in by_pos["QB"]

    def test_total_predictions_summed(self):
        result = _run([2021, 2022, 2023])
        expected = sum(f["n_predictions"] for f in result["per_fold"])
        assert result["n_total_predictions"] == expected


class TestLOYOStability:
    def test_stability_diagnostics_present(self):
        result = _run([2018, 2019, 2020, 2021, 2022])
        stab = result["stability"]
        assert "rmse_cv" in stab
        assert "rmse_cv_flag" in stab
        assert "r2_trend_rho" in stab
        assert "worst_fold" in stab
        assert "best_fold" in stab
        assert isinstance(stab["rmse_cv"], float)

    def test_worst_best_fold_identified(self):
        result = _run([2020, 2021, 2022])
        # 2020 has highest RMSE (9.5), 2022 has lowest (7.5)
        assert result["stability"]["worst_fold"]["season"] == 2020
        assert result["stability"]["best_fold"]["season"] == 2022

    def test_r2_trend_requires_3_folds(self):
        result = _run([2021, 2022])
        assert result["stability"]["r2_trend_rho"] is None


class TestLOYOOutputSchema:
    def test_output_is_json_serializable(self):
        result = _run([2021, 2022])
        serialized = json.dumps(result, default=str)
        assert len(serialized) > 100

    def test_output_has_required_top_level_keys(self):
        result = _run([2021])
        required_keys = [
            "backtest_type", "backtest_date", "test_seasons", "n_folds",
            "n_total_predictions", "per_fold", "aggregate", "stability", "config",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        assert result["backtest_type"] == "loyo_walk_forward"

    def test_config_captures_settings(self):
        result = _run([2021], fast=True, n_trials=10)
        cfg = result["config"]
        assert cfg["tune_hyperparameters"] is False
        assert cfg["fast"] is True
        assert cfg["n_trials"] == 10


class TestLOYOEdgeCases:
    def test_empty_seasons_returns_empty(self):
        result = _run([])
        assert result == {}

    def test_all_folds_fail_returns_empty(self):
        def failing_loader(*args, **kwargs):
            raise Exception("DB unavailable")

        result = _run(
            [2021, 2022],
            _data_loader=failing_loader,
        )
        assert result == {}

    def test_fold_exception_is_caught(self):
        def failing_runner(*args, **kwargs):
            raise RuntimeError("OOM")

        result = _run(
            [2021, 2022],
            _fold_runner=failing_runner,
        )
        assert result == {}

"""Backtest metrics must survive the strict JSON writer.

train.py's walk-forward writes each fold's `by_position` metrics with
atomic_write_json (allow_nan=False, no numpy fallback). `mae_rmse_healthy`
came out as numpy.bool_, so that write raised TypeError after every fold had
trained -- and train.py only catches OSError/ValueError there, so the run
died before writing the OOF panel (GAPS.md 2026-09-28).
"""
import numpy as np
import pandas as pd

from src.evaluation.backtester import ModelBacktester
from src.utils.atomic_io import atomic_write_json


def test_position_metrics_round_trip_through_atomic_write_json(tmp_path):
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.gamma(2.0, 5.0, 200))
    predicted = actual + rng.normal(0.0, 4.0, 200)
    metrics = ModelBacktester()._calculate_metrics(actual, predicted)

    assert type(metrics["mae_rmse_healthy"]) is bool
    atomic_write_json({"folds": [{"test_season": 2024, "by_position": {"QB": metrics}}]},
                      tmp_path / "walk_forward_fold_metrics.json")

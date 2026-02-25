"""Tests for centralized leakage guards."""
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.leakage import filter_feature_columns, drop_leakage_columns


def test_filter_feature_columns_removes_leakage():
    cols = [
        "predicted_points",
        "projection_1w",
        "target_1w",
        "target_util_4w",
        "utilization_score",
        "baseline_model",
        "actual_for_backtest",
        "targets_rolling_3",
        "target_share_pct",
    ]
    filtered = filter_feature_columns(cols)
    assert "predicted_points" not in filtered
    assert "projection_1w" not in filtered
    assert "target_1w" not in filtered
    assert "target_util_4w" not in filtered
    assert "utilization_score" not in filtered
    assert "actual_for_backtest" not in filtered
    assert "baseline_model" not in filtered
    # Non-target share features should remain
    assert "targets_rolling_3" in filtered
    assert "target_share_pct" in filtered


def test_drop_leakage_columns_dataframe():
    df = pd.DataFrame({
        "predicted_points": [1, 2],
        "projection_1w": [3, 4],
        "target_1w": [5, 6],
        "utilization_score": [7, 8],
        "targets_rolling_3": [1, 1],
    })
    out = drop_leakage_columns(df)
    assert "predicted_points" not in out.columns
    assert "projection_1w" not in out.columns
    assert "target_1w" not in out.columns
    assert "utilization_score" not in out.columns
    assert "targets_rolling_3" in out.columns

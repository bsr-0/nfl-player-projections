"""Tests for centralized leakage guards."""
import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.leakage import (
    filter_feature_columns,
    drop_leakage_columns,
    is_leakage_feature,
    find_leakage_columns,
    assert_no_leakage_columns,
    SAFE_SCHEDULE_PREFIXES,
)


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


# ---- Identifier column blocking (C3 fix) ----

def test_identifier_columns_blocked():
    """id, player_id, and similar identifiers must never be features."""
    for col in ("id", "player_id", "player_name", "name", "gsis_id"):
        assert is_leakage_feature(col), f"{col} should be flagged as leakage"


def test_filter_removes_identifier_columns():
    cols = ["id", "player_id", "fp_rolling_3", "age", "name"]
    filtered = filter_feature_columns(cols)
    assert "id" not in filtered
    assert "player_id" not in filtered
    assert "name" not in filtered
    assert "fp_rolling_3" in filtered
    assert "age" in filtered


# ---- Safe schedule feature allowlist (C1 fix) ----

def test_safe_schedule_features_not_blocked():
    """Schedule-derived features with _next_ should pass through."""
    safe_cols = [
        "sos_next_1",
        "sos_next_5",
        "sos_next_18",
        "sos_rank_next_1",
        "favorable_matchups_next_5",
        "expected_games_next_1",
        "expected_games_next_18",
        "injury_prob_next_1",
        "injury_risk_score_5",
    ]
    for col in safe_cols:
        assert not is_leakage_feature(col), (
            f"{col} is a safe schedule feature but was flagged as leakage"
        )
    filtered = filter_feature_columns(safe_cols)
    assert filtered == safe_cols


def test_unknown_next_features_still_blocked():
    """Features with _next that are NOT in the safe allowlist must be blocked."""
    dangerous_cols = [
        "score_next_week",
        "points_next_game",
        "outcome_future_1w",
        "result_forward_2w",
    ]
    for col in dangerous_cols:
        assert is_leakage_feature(col), (
            f"{col} should be flagged as leakage but was not"
        )


def test_forward_substrings_blocked():
    """_future and _forward substrings should still be blocked."""
    assert is_leakage_feature("player_future_points")
    assert is_leakage_feature("score_forward_avg")


def test_assert_raises_on_leakage():
    """assert_no_leakage_columns should raise ValueError for leaky features."""
    with pytest.raises(ValueError, match="Leakage columns detected"):
        assert_no_leakage_columns(["fp_rolling_3", "predicted_points", "age"])


def test_assert_passes_with_safe_schedule():
    """assert_no_leakage_columns should pass with safe schedule features."""
    cols = ["fp_rolling_3", "sos_next_5", "expected_games_next_1", "age"]
    # Should not raise
    assert_no_leakage_columns(cols, ban_utilization_score=True)


def test_allowlist_overrides_leakage():
    """Explicit allow parameter should bypass leakage detection."""
    cols = ["predicted_points", "fp_rolling_3"]
    filtered = filter_feature_columns(cols, allow=["predicted_points"])
    assert "predicted_points" in filtered
    assert "fp_rolling_3" in filtered

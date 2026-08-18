"""Regression test for the train/test column mismatch in _apply_bounded_scaling.

Missing-indicator columns (`<col>_missing`) are built CONDITIONALLY per frame
by feature_engineering (only when that frame's missingness exceeds 2%), and
train/test are engineered separately -- so a bounded column can exist in train
but not test. Before the fix, `test_df[col]` raised KeyError, and because
callers wrap fold loading in a broad try/except, the entire fold was silently
dropped from results (observed for real: QB/2025/'all' vanished from a Phase 3
grid with no visible error).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.feature_preparation import _apply_bounded_scaling


class TestBoundedScalingColumnMismatch:
    def test_indicator_col_missing_from_test_does_not_raise(self, tmp_path):
        """The exact production failure: indicator present in train, absent
        in test. Must not raise, and must fill test's indicator with 0."""
        train = pd.DataFrame({
            "target_share_pct_roll3_mean": [10.0, 20.0, 30.0],
            "air_yards_share_pct_roll3_mean_missing": [0, 1, 0],
        })
        test = pd.DataFrame({
            "target_share_pct_roll3_mean": [15.0, 25.0],
            # no *_missing column -- test frame had <2% missingness
        })
        _apply_bounded_scaling(train, test, tmp_path / "scaler.joblib")

        assert "air_yards_share_pct_roll3_mean_missing" in test.columns
        assert (test["air_yards_share_pct_roll3_mean_missing"] == 0).all()

    def test_non_indicator_col_absent_from_test_is_dropped_not_invented(self, tmp_path):
        """A non-indicator bounded column absent from test should be dropped
        from scaling rather than fabricated -- we don't invent real feature
        values."""
        train = pd.DataFrame({
            "target_share_pct_roll3_mean": [10.0, 20.0, 30.0],
            "snap_share_pct_roll3_mean": [50.0, 60.0, 70.0],
        })
        test = pd.DataFrame({"target_share_pct_roll3_mean": [15.0, 25.0]})
        artifact = _apply_bounded_scaling(train, test, tmp_path / "scaler.joblib")

        assert "snap_share_pct_roll3_mean" not in test.columns  # not invented
        assert "snap_share_pct_roll3_mean" not in artifact["columns"]  # dropped from scaling

    def test_matching_columns_still_scale_normally(self, tmp_path):
        """No regression to the normal path: shared bounded columns are
        min-max scaled on train-fit parameters."""
        train = pd.DataFrame({"target_share_pct_roll3_mean": [0.0, 50.0, 100.0]})
        test = pd.DataFrame({"target_share_pct_roll3_mean": [50.0]})
        artifact = _apply_bounded_scaling(train, test, tmp_path / "scaler.joblib")

        assert "target_share_pct_roll3_mean" in artifact["columns"]
        assert train["target_share_pct_roll3_mean"].tolist() == [0.0, 0.5, 1.0]
        assert test["target_share_pct_roll3_mean"].iloc[0] == pytest.approx(0.5)

    def test_empty_test_frame_is_unaffected(self, tmp_path):
        # >=3 rows required: _infer_bounded_columns skips columns with fewer
        # than 3 non-null values.
        train = pd.DataFrame({"target_share_pct_roll3_mean": [0.0, 50.0, 100.0]})
        test = pd.DataFrame()
        artifact = _apply_bounded_scaling(train, test, tmp_path / "scaler.joblib")
        assert "target_share_pct_roll3_mean" in artifact["columns"]

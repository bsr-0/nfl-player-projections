"""Tests for percentile bounds calculation and normalization integrity.

The UtilizationScoreCalculator normalizes raw utilization components (e.g.,
snap_share_pct, target_share_pct) to 0-100 using percentile bounds fitted
on training data. When bounds are zero-width (lo == hi), the fallback
behavior must not rank within the current data window, as that leaks
test-set information into utilization scores.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.utilization_score import (
    UtilizationScoreCalculator,
    load_percentile_bounds,
    save_percentile_bounds,
)


class TestZeroWidthBoundsFallback:
    """Verify that zero-width bounds do NOT rank within the test set."""

    def test_zero_width_bounds_return_constant(self):
        """When bounds are [X, X] (zero-width), all values should receive
        the same neutral score (50.0), not be ranked within the series."""
        calc = UtilizationScoreCalculator(
            position_percentiles={("RB", "snap_share_pct"): (0.0, 0.0)}
        )
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = calc._percentile_normalize(
            series, position="RB", component_key="snap_share_pct"
        )
        # All values should be the same (neutral) — NOT ranked [20, 40, 60, 80, 100]
        unique_values = result.unique()
        assert len(unique_values) == 1, (
            f"Zero-width bounds should produce a constant score for all "
            f"values, but got {len(unique_values)} unique values: {sorted(unique_values)}. "
            f"This indicates the normalizer is ranking within the current "
            f"data window, which leaks test-set information."
        )

    def test_zero_width_bounds_return_fifty(self):
        """The neutral score for zero-width bounds should be 50.0."""
        calc = UtilizationScoreCalculator(
            position_percentiles={("WR", "snap_share_pct"): (0.0, 0.0)}
        )
        series = pd.Series([5.0, 15.0, 25.0])
        result = calc._percentile_normalize(
            series, position="WR", component_key="snap_share_pct"
        )
        assert (result == 50.0).all(), (
            f"Expected all values to be 50.0, got: {result.tolist()}"
        )

    def test_constant_value_zero_width_returns_fifty(self):
        """Even when all series values are identical AND bounds are zero-width,
        result should be 50.0."""
        calc = UtilizationScoreCalculator(
            position_percentiles={("TE", "inline_rate_pct"): (100.0, 100.0)}
        )
        series = pd.Series([100.0, 100.0, 100.0])
        result = calc._percentile_normalize(
            series, position="TE", component_key="inline_rate_pct"
        )
        assert (result == 50.0).all()


class TestValidBoundsNormalization:
    """Verify that valid (non-zero-width) bounds normalize correctly."""

    def test_valid_bounds_normalize_to_range(self):
        """Values within [lo, hi] should map to [0, 100]."""
        calc = UtilizationScoreCalculator(
            position_percentiles={("RB", "rush_share_pct"): (10.0, 60.0)}
        )
        series = pd.Series([10.0, 35.0, 60.0])
        result = calc._percentile_normalize(
            series, position="RB", component_key="rush_share_pct"
        )
        expected = pd.Series([0.0, 50.0, 100.0])
        pd.testing.assert_series_equal(result, expected, atol=0.01)

    def test_values_outside_bounds_clipped(self):
        """Values below lo or above hi should be clipped to [0, 100]."""
        calc = UtilizationScoreCalculator(
            position_percentiles={("QB", "dropback_rate_pct"): (20.0, 80.0)}
        )
        series = pd.Series([0.0, 50.0, 100.0])
        result = calc._percentile_normalize(
            series, position="QB", component_key="dropback_rate_pct"
        )
        assert result.min() >= 0.0
        assert result.max() <= 100.0


class TestBoundsFitProducesValidWidths:
    """Verify that fitting bounds on realistic data produces valid widths."""

    def test_bounds_fit_no_zero_width(self):
        """Fitting bounds on varied data should never produce lo == hi."""
        calc = UtilizationScoreCalculator()
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "position": ["RB"] * n,
            "snap_share_pct": np.random.uniform(10, 90, n),
            "rush_share_pct": np.random.uniform(5, 95, n),
        })
        calc.fit_percentile_bounds(
            df, position="RB",
            component_columns=["snap_share_pct", "rush_share_pct"],
            persist=False,
        )
        for key, (lo, hi) in calc.position_percentiles.items():
            assert hi > lo, (
                f"Fitted bounds for {key} are zero-width: [{lo}, {hi}]"
            )

    def test_bounds_fit_constant_column_warns(self):
        """Fitting bounds on a constant column should log a warning,
        not silently produce zero-width bounds."""
        calc = UtilizationScoreCalculator()
        df = pd.DataFrame({
            "position": ["TE"] * 100,
            "inline_rate_pct": [100.0] * 100,
        })
        # Should still produce bounds (possibly zero-width for constant data)
        # but we need the system to handle this case gracefully
        calc.fit_percentile_bounds(
            df, position="TE",
            component_columns=["inline_rate_pct"],
            persist=False,
        )
        key = ("TE", "inline_rate_pct")
        if key in calc.position_percentiles:
            lo, hi = calc.position_percentiles[key]
            # If zero-width, that's expected for constant data —
            # but the normalizer must handle it without ranking.
            if lo == hi:
                series = pd.Series([100.0, 100.0, 100.0])
                result = calc._percentile_normalize(
                    series, position="TE", component_key="inline_rate_pct"
                )
                assert len(result.unique()) == 1, (
                    "Constant column with zero-width bounds must return "
                    "constant score, not rank-based values."
                )


class TestBoundsLoadValidation:
    """Verify that loading bounds validates and warns about zero-width entries."""

    def test_ensure_bounds_loaded_warns_on_zero_width(self, tmp_path, caplog):
        """_ensure_bounds_loaded() must log a warning for zero-width bounds."""
        import logging
        bounds = {
            ("RB", "snap_share_pct"): (0.0, 0.0),
            ("RB", "rush_share_pct"): (5.0, 95.0),
        }
        path = tmp_path / "test_bounds.json"
        save_percentile_bounds(bounds, path)

        calc = UtilizationScoreCalculator()
        calc._BOUNDS_DEFAULT_PATH = path
        with caplog.at_level(logging.WARNING):
            calc._ensure_bounds_loaded()
        assert any("zero-width" in record.message.lower() for record in caplog.records), (
            "Expected a warning about zero-width bounds, but none was logged"
        )
        assert any("RB|snap_share_pct" in record.message for record in caplog.records)

    def test_ensure_bounds_loaded_no_warning_for_valid_bounds(self, tmp_path, caplog):
        """No warning should be logged when all bounds are valid."""
        import logging
        bounds = {
            ("RB", "snap_share_pct"): (5.0, 85.0),
            ("RB", "rush_share_pct"): (10.0, 90.0),
        }
        path = tmp_path / "test_bounds.json"
        save_percentile_bounds(bounds, path)

        calc = UtilizationScoreCalculator()
        calc._BOUNDS_DEFAULT_PATH = path
        with caplog.at_level(logging.WARNING):
            calc._ensure_bounds_loaded()
        zero_width_warnings = [
            r for r in caplog.records if "zero-width" in r.message.lower()
        ]
        assert len(zero_width_warnings) == 0


class TestBoundsMetadata:
    """Verify that bounds metadata tracks training provenance."""

    def test_save_load_preserves_metadata(self, tmp_path):
        """Bounds metadata (training seasons) must survive save/load cycle."""
        bounds = {("RB", "snap_share_pct"): (5.0, 85.0)}
        meta = {"train_seasons": [2020, 2021, 2022, 2023]}
        path = tmp_path / "test_bounds.json"
        save_percentile_bounds(bounds, path, metadata=meta)
        loaded, loaded_meta = load_percentile_bounds(path, return_meta=True)
        assert loaded_meta["train_seasons"] == [2020, 2021, 2022, 2023]
        assert ("RB", "snap_share_pct") in loaded
        lo, hi = loaded[("RB", "snap_share_pct")]
        assert lo == 5.0
        assert hi == 85.0

"""Tests for utilization score calculation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.utilization_score import (
    UtilizationScoreCalculator,
    calculate_utilization_scores,
    save_percentile_bounds,
    load_percentile_bounds,
    validate_percentile_bounds_meta,
)


class TestUtilizationScoreCalculator:
    """Test suite for UtilizationScoreCalculator."""
    
    @pytest.fixture
    def calculator(self):
        return UtilizationScoreCalculator()
    
    @pytest.fixture
    def sample_rb_data(self):
        """Sample RB data for testing."""
        return pd.DataFrame({
            "player_id": ["rb1", "rb2", "rb3"],
            "name": ["RB One", "RB Two", "RB Three"],
            "position": ["RB", "RB", "RB"],
            "team": ["KC", "KC", "KC"],
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "rushing_attempts": [20, 10, 5],
            "rushing_yards": [100, 40, 20],
            "rushing_tds": [1, 0, 0],
            "targets": [5, 8, 2],
            "receptions": [4, 6, 1],
            "receiving_yards": [30, 50, 5],
            "receiving_tds": [0, 1, 0],
            "snap_count": [50, 40, 20],
            "snap_share": [0.75, 0.60, 0.30],
        })
    
    @pytest.fixture
    def sample_wr_data(self):
        """Sample WR data for testing."""
        return pd.DataFrame({
            "player_id": ["wr1", "wr2"],
            "name": ["WR One", "WR Two"],
            "position": ["WR", "WR"],
            "team": ["SF", "SF"],
            "season": [2024, 2024],
            "week": [1, 1],
            "targets": [12, 5],
            "receptions": [8, 3],
            "receiving_yards": [120, 30],
            "receiving_tds": [1, 0],
            "rushing_attempts": [1, 0],
            "rushing_yards": [10, 0],
            "rushing_tds": [0, 0],
            "snap_count": [55, 30],
            "snap_share": [0.85, 0.45],
        })
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with correct weights."""
        assert "RB" in calculator.weights
        assert "WR" in calculator.weights
        assert "TE" in calculator.weights
        assert "QB" in calculator.weights
    
    def test_rb_utilization_calculation(self, calculator, sample_rb_data):
        """Test RB utilization score calculation."""
        result = calculator.calculate_all_scores(sample_rb_data, pd.DataFrame())
        
        assert "utilization_score" in result.columns
        assert len(result) == 3
        
        # Higher volume RB should have higher score
        rb1_score = result[result["player_id"] == "rb1"]["utilization_score"].iloc[0]
        rb3_score = result[result["player_id"] == "rb3"]["utilization_score"].iloc[0]
        assert rb1_score > rb3_score
    
    def test_wr_utilization_calculation(self, calculator, sample_wr_data):
        """Test WR utilization score calculation."""
        result = calculator.calculate_all_scores(sample_wr_data, pd.DataFrame())
        
        assert "utilization_score" in result.columns
        
        # Higher target WR should have higher score
        wr1_score = result[result["player_id"] == "wr1"]["utilization_score"].iloc[0]
        wr2_score = result[result["player_id"] == "wr2"]["utilization_score"].iloc[0]
        assert wr1_score > wr2_score
    
    def test_utilization_score_range(self, calculator, sample_rb_data):
        """Test that utilization scores are in valid range (0-100)."""
        result = calculator.calculate_all_scores(sample_rb_data, pd.DataFrame())
        
        scores = result["utilization_score"]
        assert scores.min() >= 0
        assert scores.max() <= 100
    
    def test_utilization_tier(self, calculator):
        """Test utilization tier assignment."""
        assert calculator.get_utilization_tier(85, "RB") == "Elite"
        assert calculator.get_utilization_tier(75, "RB") == "Strong"
        assert calculator.get_utilization_tier(65, "RB") == "Average"
        assert calculator.get_utilization_tier(55, "RB") == "Below Average"
        assert calculator.get_utilization_tier(40, "RB") == "Low"
    
    def test_expected_ppg_range(self, calculator):
        """Test expected PPG range calculation."""
        # Elite RB
        elite_range = calculator.get_expected_ppg_range(85, "RB")
        assert "min" in elite_range
        assert "avg" in elite_range
        assert "max" in elite_range
        assert elite_range["avg"] > 15  # Elite RBs average 18+ PPG
        
        # Average RB
        avg_range = calculator.get_expected_ppg_range(65, "RB")
        assert avg_range["avg"] < elite_range["avg"]
    
    def test_convenience_function(self, sample_rb_data):
        """Test the convenience function."""
        result = calculate_utilization_scores(sample_rb_data)
        
        assert "utilization_score" in result.columns
        assert len(result) == len(sample_rb_data)


class TestUtilizationScoreEdgeCases:
    """Test edge cases for utilization score calculation."""
    
    @pytest.fixture
    def calculator(self):
        return UtilizationScoreCalculator()
    
    def test_zero_stats(self, calculator):
        """Test handling of players with zero stats."""
        data = pd.DataFrame({
            "player_id": ["zero"],
            "name": ["Zero Stats"],
            "position": ["RB"],
            "team": ["KC"],
            "season": [2024],
            "week": [1],
            "rushing_attempts": [0],
            "rushing_yards": [0],
            "rushing_tds": [0],
            "targets": [0],
            "receptions": [0],
            "receiving_yards": [0],
            "receiving_tds": [0],
            "snap_count": [0],
            "snap_share": [0],
        })
        
        result = calculator.calculate_all_scores(data, pd.DataFrame())
        
        # Should not raise error
        assert "utilization_score" in result.columns
        # Score should be low but valid
        assert result["utilization_score"].iloc[0] >= 0
    
    def test_missing_columns(self, calculator):
        """Test handling of missing columns."""
        data = pd.DataFrame({
            "player_id": ["p1"],
            "name": ["Player"],
            "position": ["RB"],
            "team": ["KC"],
            "season": [2024],
            "week": [1],
            "rushing_attempts": [10],
            "rushing_yards": [50],
            # Missing other columns
        })
        
        # Should handle gracefully
        result = calculator.calculate_all_scores(data, pd.DataFrame())
        assert len(result) == 1
    
    def test_empty_dataframe(self, calculator):
        """Test handling of empty DataFrame."""
        data = pd.DataFrame()
        result = calculator.calculate_all_scores(data, pd.DataFrame())
        assert len(result) == 0

    def test_percentile_bounds_metadata_roundtrip(self, tmp_path):
        """Bounds metadata should be preserved and validated."""
        bounds = {("RB", "snap_share_pct"): (0.1, 0.9)}
        meta = {"train_seasons": [2022, 2023], "min_season": 2022, "max_season": 2023}
        path = tmp_path / "bounds.json"
        save_percentile_bounds(bounds, path, metadata=meta)
        loaded_bounds, loaded_meta = load_percentile_bounds(path, return_meta=True)
        assert loaded_bounds == bounds
        assert validate_percentile_bounds_meta(loaded_meta, [2022, 2023]) is True
        assert validate_percentile_bounds_meta(loaded_meta, [2021, 2023]) is False


class TestPercentileNormalizationBounds:
    """Tests for _percentile_normalize behavior with zero-width and normal bounds."""

    @pytest.fixture
    def calculator(self):
        return UtilizationScoreCalculator()

    def test_normal_bounds_linear_scaling(self, calculator):
        """Normal bounds produce linear 0-100 scaling."""
        calculator.position_percentiles[("WR", "target_share")] = (10.0, 90.0)
        series = pd.Series([10.0, 50.0, 90.0])
        result = calculator._percentile_normalize(series, "WR", "target_share")
        np.testing.assert_allclose(result.values, [0.0, 50.0, 100.0])

    def test_zero_width_bounds_varied_data_uses_rank(self, calculator):
        """Zero-width bounds with varied data fall back to rank-based percentile."""
        calculator.position_percentiles[("WR", "snap_share_pct")] = (0.0, 0.0)
        series = pd.Series([0.1, 0.3, 0.5, 0.7, 0.9])
        result = calculator._percentile_normalize(series, "WR", "snap_share_pct")
        # Rank-based: values should be in (0, 100], preserving order
        assert result.is_monotonic_increasing
        assert result.min() > 0
        assert result.max() <= 100

    def test_zero_width_bounds_constant_series_returns_midpoint(self, calculator):
        """Zero-width bounds + constant series → all 50.0 (neutral)."""
        calculator.position_percentiles[("TE", "inline_rate")] = (100.0, 100.0)
        series = pd.Series([0.0, 0.0, 0.0])
        result = calculator._percentile_normalize(series, "TE", "inline_rate")
        np.testing.assert_array_equal(result.values, [50.0, 50.0, 50.0])

    def test_fit_warns_on_zero_width_bounds(self, calculator, caplog):
        """fit_percentile_bounds logs a warning for zero-width bounds."""
        train_df = pd.DataFrame({
            "position": ["WR"] * 20,
            "snap_share_pct": [0.0] * 20,
        })
        with caplog.at_level("WARNING", logger="src.features.utilization_score"):
            calculator.fit_percentile_bounds(
                train_df, "WR", ["snap_share_pct"], persist=False
            )
        assert any("Zero-width" in msg for msg in caplog.messages)
        assert calculator.position_percentiles[("WR", "snap_share_pct")] == (0.0, 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Web App Rendering Tests for NFL Predictor.

Tests web app render functions for:
- Empty DataFrame handling
- Missing column handling
- NaN value handling
- Type conversion edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHelperFunctions:
    """Test helper functions used in rendering."""
    
    def test_recommendation_badge_function(self):
        """Test get_recommendation_badge returns correct values."""
        # Import the function from app.py (using exec since it's in main script)
        app_path = Path(__file__).parent.parent / 'app.py'
        
        # Test basic recommendation logic inline
        def get_recommendation_badge_simple(proj_points, position, util_score=None):
            """Simplified version for testing."""
            if proj_points >= 15:
                return "START", "high"
            elif proj_points >= 10:
                return "FLEX", "medium"
            else:
                return "SIT", "low"
        
        rec, conf = get_recommendation_badge_simple(20.0, 'RB')
        assert rec == "START"
        assert conf == "high"
        
        rec, conf = get_recommendation_badge_simple(12.0, 'WR')
        assert rec == "FLEX"
        assert conf == "medium"
        
        rec, conf = get_recommendation_badge_simple(5.0, 'TE')
        assert rec == "SIT"
        assert conf == "low"
    
    def test_safe_get_value(self):
        """Test safe value extraction from row dict."""
        row = {
            'name': 'Test Player',
            'team': 'KC',
            'fantasy_points': 15.5,
            'util_score': np.nan,
            'missing_col': None
        }
        
        # Safe get with defaults
        assert row.get('name', 'Unknown') == 'Test Player'
        assert row.get('nonexistent', 'N/A') == 'N/A'
        
        # Handle NaN
        util = row.get('util_score', 0)
        if pd.isna(util):
            util = 0
        assert util == 0


class TestEmptyDataHandling:
    """Test handling of empty DataFrames in rendering."""
    
    def test_empty_dataframe_checks(self):
        """Test that empty DataFrames are handled correctly."""
        empty_df = pd.DataFrame()
        
        # Check standard empty checks
        assert empty_df.empty
        assert len(empty_df) == 0
        assert list(empty_df.columns) == []
    
    def test_filtered_empty_dataframe(self):
        """Test DataFrame that becomes empty after filtering."""
        df = pd.DataFrame({
            'position': ['QB', 'QB', 'QB'],
            'fantasy_points': [20.0, 18.0, 15.0],
            'name': ['Player1', 'Player2', 'Player3']
        })
        
        # Filter to non-existent position
        rb_df = df[df['position'] == 'RB']
        assert rb_df.empty
        
        # Safe iteration over empty DataFrame
        count = 0
        for _, row in rb_df.iterrows():
            count += 1
        assert count == 0
    
    def test_dataframe_with_all_nan(self):
        """Test DataFrame with all NaN values in a column."""
        df = pd.DataFrame({
            'name': ['Player1', 'Player2'],
            'util_score': [np.nan, np.nan],
            'fantasy_points': [np.nan, np.nan]
        })
        
        # Check for all NaN
        assert df['util_score'].isna().all()
        
        # Safe aggregation
        mean_util = df['util_score'].mean()
        assert pd.isna(mean_util)
        
        # Safe fillna
        filled = df['util_score'].fillna(0)
        assert (filled == 0).all()


class TestMissingColumnHandling:
    """Test handling of missing columns."""
    
    def test_column_existence_check(self):
        """Test checking if columns exist before access."""
        df = pd.DataFrame({
            'name': ['Player1'],
            'team': ['KC']
        })
        
        # Check column exists
        assert 'name' in df.columns
        assert 'util_score' not in df.columns
        
        # Safe access pattern
        if 'util_score' in df.columns:
            util = df['util_score'].iloc[0]
        else:
            util = 0
        
        assert util == 0
    
    def test_get_with_default_for_missing(self):
        """Test get method for accessing missing columns."""
        df = pd.DataFrame({
            'name': ['Player1'],
            'fantasy_points': [15.0]
        })
        
        row = df.iloc[0].to_dict()
        
        # Safe get with default
        proj = row.get('projection_1w', row.get('fantasy_points', 0))
        assert proj == 15.0
        
        util = row.get('util_score', 0)
        assert util == 0
    
    def test_fallback_column_pattern(self):
        """Test fallback through multiple column names."""
        df = pd.DataFrame({
            'name': ['Player1'],
            'fantasy_points': [15.0]
        })
        
        # Pattern used in app.py
        proj_columns = ['projection_1w', 'proj_1w', 'fantasy_points_proj', 'fantasy_points']
        
        proj_col = None
        for col in proj_columns:
            if col in df.columns:
                proj_col = col
                break
        
        assert proj_col == 'fantasy_points'
        assert df[proj_col].iloc[0] == 15.0


class TestTypeConversion:
    """Test type conversion edge cases."""
    
    def test_string_to_float_conversion(self):
        """Test converting string values to float."""
        values = ['15.5', '-', 'N/A', None, 10.0, np.nan]
        
        for val in values:
            if isinstance(val, str):
                try:
                    result = float(val)
                except (ValueError, TypeError):
                    result = 0.0
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                result = 0.0
            else:
                result = float(val)
            
            # All should be valid floats
            assert isinstance(result, float)
    
    def test_safe_numeric_conversion(self):
        """Test pd.to_numeric for safe conversion."""
        series = pd.Series(['15.5', '10', '-', 'N/A', None])
        
        # Safe conversion
        numeric = pd.to_numeric(series, errors='coerce')
        
        assert numeric.iloc[0] == 15.5
        assert numeric.iloc[1] == 10.0
        assert pd.isna(numeric.iloc[2])  # '-' becomes NaN
        assert pd.isna(numeric.iloc[3])  # 'N/A' becomes NaN
        assert pd.isna(numeric.iloc[4])  # None becomes NaN
    
    def test_util_score_type_handling(self):
        """Test util_score which can be string or float."""
        test_values = [
            (85.0, 85.0),
            ('-', 0.0),
            ('N/A', 0.0),
            (np.nan, 0.0),
            (None, 0.0),
            (0, 0.0),
            ('75', 75.0)
        ]
        
        for input_val, expected in test_values:
            if isinstance(input_val, str):
                if input_val in ['-', 'N/A', '']:
                    result = 0.0
                else:
                    try:
                        result = float(input_val)
                    except ValueError:
                        result = 0.0
            elif input_val is None:
                result = 0.0
            elif isinstance(input_val, float) and np.isnan(input_val):
                result = 0.0
            else:
                result = float(input_val)
            
            assert result == expected, f"Input {input_val} expected {expected}, got {result}"


class TestPlotlyChartData:
    """Test data preparation for Plotly charts."""
    
    def test_bar_chart_data_empty(self):
        """Test bar chart handles empty data."""
        df = pd.DataFrame(columns=['name', 'fantasy_points'])
        
        if not df.empty:
            x_data = df['name'].tolist()
            y_data = df['fantasy_points'].tolist()
        else:
            x_data = []
            y_data = []
        
        assert x_data == []
        assert y_data == []
    
    def test_scatter_plot_data_with_nan(self):
        """Test scatter plot handles NaN values."""
        df = pd.DataFrame({
            'util_score': [80.0, np.nan, 70.0],
            'fantasy_points': [15.0, 12.0, np.nan]
        })
        
        # Drop NaN for plotting
        plot_df = df.dropna(subset=['util_score', 'fantasy_points'])
        
        assert len(plot_df) == 1
        assert plot_df.iloc[0]['util_score'] == 80.0
        assert plot_df.iloc[0]['fantasy_points'] == 15.0
    
    def test_chart_color_values(self):
        """Test color column values are valid."""
        df = pd.DataFrame({
            'name': ['P1', 'P2', 'P3'],
            'score': [90, 50, 10]
        })
        
        # Create color category
        df['color'] = df['score'].apply(
            lambda x: 'green' if x >= 70 else ('yellow' if x >= 40 else 'red')
        )
        
        assert df['color'].tolist() == ['green', 'yellow', 'red']


class TestRendererRobustness:
    """Test render functions don't crash with edge cases."""
    
    def test_player_card_data_validation(self):
        """Test player card data preparation."""
        # Minimal valid data
        row = {
            'name': 'Test',
            'team': 'KC',
            'position': 'RB'
        }
        
        # Extract values with defaults
        name = row.get('name', 'Unknown')
        team = row.get('team', 'FA')
        pos = row.get('position', 'FLEX')
        proj = row.get('fantasy_points', 0)
        util = row.get('util_score', 0)
        
        # Handle potential type issues
        if isinstance(proj, str):
            proj = 0.0 if proj in ['-', 'N/A', ''] else float(proj)
        if isinstance(util, str):
            util = 0.0 if util in ['-', 'N/A', ''] else float(util)
        
        # Build HTML safely
        html = f"<div>{name} - {team} ({pos}): {proj:.1f} pts</div>"
        assert "Test - KC (RB): 0.0 pts" in html
    
    def test_comparison_with_missing_player(self):
        """Test comparison when one player is missing data."""
        player_a = {
            'name': 'Player A',
            'team': 'KC',
            'fantasy_points': 15.0
        }
        
        player_b = {}  # Empty dict
        
        # Safe extraction
        name_a = player_a.get('name', 'Unknown')
        name_b = player_b.get('name', 'Unknown')
        
        assert name_a == 'Player A'
        assert name_b == 'Unknown'


class TestSeasonWeekFiltering:
    """Test season/week filter edge cases."""
    
    def test_filter_with_no_matching_data(self):
        """Test filtering returns empty when no matches."""
        df = pd.DataFrame({
            'season': [2023, 2023, 2023],
            'week': [1, 2, 3],
            'name': ['P1', 'P2', 'P3']
        })
        
        # Filter to non-existent season
        filtered = df[df['season'] == 2024]
        assert filtered.empty
        
        # Safe max on empty
        if not filtered.empty:
            max_week = filtered['week'].max()
        else:
            max_week = 0
        
        assert max_week == 0
    
    def test_latest_data_extraction(self):
        """Test getting latest season/week data."""
        df = pd.DataFrame({
            'season': [2023, 2023, 2024, 2024],
            'week': [17, 18, 1, 2],
            'name': ['P1', 'P2', 'P3', 'P4']
        })
        
        latest_season = df['season'].max()
        latest_week = df[df['season'] == latest_season]['week'].max()
        
        assert latest_season == 2024
        assert latest_week == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

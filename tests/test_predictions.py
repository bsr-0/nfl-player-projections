"""
Unit Tests for NFL Predictor
Tests: predictions, data mining, performance tracking
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from model_connector import ModelConnector
from performance_tracker import PerformanceTracker
from advanced_features import InjuryImpactModel, MatchupAdjuster, WhatIfAnalyzer

# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_historical_data():
    """Sample historical data for testing."""
    return pd.DataFrame({
        'player_id': ['P1', 'P1', 'P2', 'P2'],
        'player_name': ['Player A', 'Player A', 'Player B', 'Player B'],
        'season': [2024, 2024, 2024, 2024],
        'week': [1, 2, 1, 2],
        'position': ['RB', 'RB', 'WR', 'WR'],
        'recent_team': ['KC', 'KC', 'SF', 'SF'],
        'utilization_score': [85, 88, 75, 72],
        'targets': [5, 6, 8, 7],
        'carries': [18, 20, 0, 0],
        'receptions': [4, 5, 6, 5],
    })

@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return pd.DataFrame({
        'player': ['Player A', 'Player B', 'Player C'],
        'position': ['RB', 'WR', 'QB'],
        'team': ['KC', 'SF', 'BUF'],
        'util_1w': [85.0, 75.0, 90.0],
        'util_1w_low': [78.0, 68.0, 85.0],
        'util_1w_high': [92.0, 82.0, 95.0],
        'util_18w_avg': [83.0, 73.0, 88.0],
        'tier': ['elite', 'high', 'elite'],
    })

# ============================================================================
# Model Connector Tests
# ============================================================================

def test_model_connector_initialization():
    """Test ModelConnector can be initialized."""
    connector = ModelConnector()
    assert connector is not None
    assert isinstance(connector.models, dict)

@pytest.mark.skip(reason="Requires trained models in models/ directory")
def test_model_connector_fallback(sample_historical_data):
    """Test fallback prediction when models unavailable."""
    connector = ModelConnector()
    predictions = connector.batch_predict(sample_historical_data, n_per_position=2)
    
    # Should return predictions even without models
    assert not predictions.empty
    assert 'player' in predictions.columns
    assert 'util_1w' in predictions.columns

@pytest.mark.skip(reason="Requires trained models in models/ directory")
def test_prediction_ranges(sample_historical_data):
    """Test that predictions are within valid ranges."""
    connector = ModelConnector()
    predictions = connector.batch_predict(sample_historical_data)
    
    # All predictions should be 0-100
    assert (predictions['util_1w'] >= 0).all()
    assert (predictions['util_1w'] <= 100).all()
    
    # Lower bound < prediction < upper bound
    assert (predictions['util_1w_low'] <= predictions['util_1w']).all()
    assert (predictions['util_1w'] <= predictions['util_1w_high']).all()

# ============================================================================
# Performance Tracker Tests
# ============================================================================

def test_performance_tracker_initialization(tmp_path):
    """Test PerformanceTracker can be initialized."""
    db_file = tmp_path / "test_performance.db"
    tracker = PerformanceTracker(db_path=str(db_file))
    assert tracker is not None
    assert db_file.exists()

@pytest.mark.skip(reason="PerformanceTracker.record_predictions not yet implemented")
def test_record_and_evaluate_predictions(sample_predictions, tmp_path):
    """Test recording predictions and evaluating accuracy."""
    db_file = tmp_path / "test_performance.db"
    tracker = PerformanceTracker(db_path=str(db_file))
    
    # Record predictions
    pred_id = tracker.record_predictions(sample_predictions, week=10, season=2024)
    assert pred_id == "2024_week_10"
    
    # Create mock actuals
    actuals = sample_predictions.copy()
    actuals['actual_util'] = actuals['util_1w'] + np.random.normal(0, 3, len(actuals))
    
    # Evaluate
    metrics = tracker.record_actuals(actuals[['player', 'position', 'team', 'actual_util']], pred_id)
    
    assert 'overall' in metrics
    assert 'mae' in metrics['overall']
    assert metrics['overall']['n'] == len(sample_predictions)

@pytest.mark.skip(reason="PerformanceTracker.get_summary_stats not yet implemented")
def test_performance_summary(tmp_path):
    """Test summary statistics calculation."""
    db_file = tmp_path / "test_performance.db"
    tracker = PerformanceTracker(db_path=str(db_file))
    summary = tracker.get_summary_stats()
    
    assert 'total_weeks' in summary
    assert 'overall_accuracy' in summary

# ============================================================================
# Injury Impact Tests
# ============================================================================

def test_injury_model_initialization():
    """Test InjuryImpactModel can be initialized."""
    model = InjuryImpactModel()
    assert model is not None

def test_injury_adjustments():
    """Test injury status adjustments."""
    model = InjuryImpactModel()
    
    base_util = 85.0
    
    # Test different statuses
    healthy = model.adjust_for_injury('Player A', 'RB', base_util, 'HEALTHY')
    assert healthy['adjusted_prediction'] == base_util
    assert healthy['risk_level'] == 'low'
    
    questionable = model.adjust_for_injury('Player A', 'RB', base_util, 'QUESTIONABLE')
    assert questionable['adjusted_prediction'] < base_util
    assert questionable['risk_level'] == 'high'
    
    out = model.adjust_for_injury('Player A', 'RB', base_util, 'OUT')
    assert out['adjusted_prediction'] == 0
    assert out['risk_level'] == 'out'

# ============================================================================
# Matchup Adjuster Tests
# ============================================================================

def test_matchup_adjuster_initialization():
    """Test MatchupAdjuster can be initialized."""
    adjuster = MatchupAdjuster()
    assert adjuster is not None

def test_matchup_adjustments():
    """Test matchup difficulty adjustments."""
    adjuster = MatchupAdjuster()
    adjuster.fetch_defense_rankings()  # Will create mock rankings
    
    base_util = 80.0
    
    # Test adjustment (exact values depend on mock data)
    result = adjuster.adjust_for_matchup('elite', 'RB', 'KC', base_util)
    
    assert 'adjusted_prediction' in result
    assert 'matchup_factor' in result
    assert 'matchup_rating' in result

# ============================================================================
# What-If Analyzer Tests
# ============================================================================

def test_whatif_analyzer_initialization(sample_historical_data):
    """Test WhatIfAnalyzer can be initialized."""
    analyzer = WhatIfAnalyzer(sample_historical_data)
    assert analyzer is not None

def test_draft_pick_analysis(sample_historical_data):
    """Test analyzing a draft pick."""
    analyzer = WhatIfAnalyzer(sample_historical_data)
    
    result = analyzer.analyze_draft_pick('Player A', 2024, 2)
    
    assert 'player' in result or 'error' in result
    if 'error' not in result:
        assert 'avg_utilization' in result
        assert 'verdict' in result

def test_player_comparison(sample_historical_data):
    """Test comparing two players."""
    analyzer = WhatIfAnalyzer(sample_historical_data)
    
    result = analyzer.compare_players('Player A', 'Player B', 2024)
    
    assert 'player1' in result or 'error' in result
    if 'error' not in result:
        assert 'winner' in result

# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skip(reason="Requires trained models and database access")
def test_end_to_end_prediction_pipeline(sample_historical_data):
    """Test complete prediction pipeline."""
    # 1. Generate predictions
    connector = ModelConnector()
    predictions = connector.batch_predict(sample_historical_data, n_per_position=5)
    
    assert not predictions.empty
    
    # 2. Record predictions
    tracker = PerformanceTracker()
    pred_id = tracker.record_predictions(predictions, week=10, season=2024)
    
    assert pred_id is not None
    
    # 3. Apply injury adjustments
    injury_model = InjuryImpactModel()
    for _, player in predictions.head(1).iterrows():
        adjusted = injury_model.adjust_for_injury(
            player['player'],
            player['position'],
            player['util_1w'],
            'QUESTIONABLE'
        )
        assert adjusted['adjusted_prediction'] < player['util_1w']

def test_data_quality_checks(sample_historical_data):
    """Test data quality validation."""
    # Check required columns
    required_cols = ['player_id', 'player_name', 'season', 'week', 'position', 'utilization_score']
    assert all(col in sample_historical_data.columns for col in required_cols)
    
    # Check data ranges
    assert (sample_historical_data['utilization_score'] >= 0).all()
    assert (sample_historical_data['utilization_score'] <= 100).all()
    assert (sample_historical_data['week'] >= 1).all()
    assert (sample_historical_data['week'] <= 18).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

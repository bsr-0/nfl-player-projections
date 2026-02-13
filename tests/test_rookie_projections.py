"""
Unit Tests for Rookie Projections

Tests for:
- Draft tier classification
- Draft capital interpolation
- Rookie PPG projection
- Breakout/bust probability bounds
- Bayesian rookie prior
- Combine score calculation
- Comparable player matching
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.advanced_rookie_injury import AdvancedRookieProjector, RookieProfile
from src.features.season_long_features import RookieProjector, DraftDataLoader


class TestDraftTierClassification:
    """Test draft position to tier classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
    
    def test_top_10_pick_is_round1_top10(self):
        """Top 10 picks should be classified as round_1_top10."""
        for pick in [1, 5, 10]:
            tier = self.projector.get_draft_tier(draft_round=1, draft_pick=pick)
            assert tier == 'round_1_top10'
    
    def test_late_first_round_is_round1(self):
        """Picks 11-32 should be classified as round_1."""
        for pick in [11, 20, 32]:
            tier = self.projector.get_draft_tier(draft_round=1, draft_pick=pick)
            assert tier == 'round_1'
    
    def test_second_round_is_round2(self):
        """Round 2 picks should be classified as round_2."""
        tier = self.projector.get_draft_tier(draft_round=2, draft_pick=45)
        assert tier == 'round_2'
    
    def test_late_round_is_round3_plus(self):
        """Round 3+ picks should be classified as round_3_plus."""
        for round_num in [3, 4, 5, 6, 7]:
            tier = self.projector.get_draft_tier(draft_round=round_num, draft_pick=100)
            assert tier == 'round_3_plus'


class TestDraftCapitalInterpolation:
    """Test draft capital value curve interpolation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
    
    def test_pick_1_has_highest_value(self):
        """Pick #1 should have the highest draft capital value."""
        value = self.projector.calculate_draft_capital_value(1)
        assert value == 1.0
    
    def test_value_decreases_with_pick(self):
        """Draft capital value should decrease with later picks."""
        value_1 = self.projector.calculate_draft_capital_value(1)
        value_32 = self.projector.calculate_draft_capital_value(32)
        value_100 = self.projector.calculate_draft_capital_value(100)
        value_250 = self.projector.calculate_draft_capital_value(250)
        
        assert value_1 > value_32 > value_100 > value_250
    
    def test_value_bounds(self):
        """Draft capital value should be between 0 and 1."""
        for pick in [1, 50, 100, 200, 260]:
            value = self.projector.calculate_draft_capital_value(pick)
            assert 0 <= value <= 1
    
    def test_interpolation_between_known_picks(self):
        """Values should interpolate smoothly between known picks."""
        value_10 = self.projector.calculate_draft_capital_value(10)
        value_15 = self.projector.calculate_draft_capital_value(15)
        value_12 = self.projector.calculate_draft_capital_value(12)
        
        # Pick 12 value should be between pick 10 and pick 15
        assert value_10 >= value_12 >= value_15


class TestRookiePPGProjection:
    """Test rookie PPG projection by archetype."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
        self.basic_projector = RookieProjector()
    
    def test_all_positions_have_archetypes(self):
        """All positions should have archetype definitions."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            assert position in self.projector.ROOKIE_PPG_BY_DRAFT
    
    def test_elite_higher_than_low(self):
        """Elite archetype should project higher than low."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            elite_tier = self.projector.get_draft_tier(1, 1)  # Pick #1
            low_tier = self.projector.get_draft_tier(7, 230)  # Late round
            
            elite_stats = self.projector.ROOKIE_PPG_BY_DRAFT[position].get(elite_tier, {})
            low_stats = self.projector.ROOKIE_PPG_BY_DRAFT[position].get(low_tier, {})
            
            if elite_stats and low_stats:
                assert elite_stats.get('mean', 0) >= low_stats.get('mean', 0)
    
    def test_projections_are_positive(self):
        """All projections should be positive."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            for tier, stats in self.projector.ROOKIE_PPG_BY_DRAFT[position].items():
                assert stats['mean'] > 0
                assert stats['games'] > 0
    
    def test_project_rookie_returns_profile(self):
        """project_rookie should return a RookieProfile."""
        profile = self.projector.project_rookie(
            player_id='test_player',
            name='Test Player',
            position='RB',
            draft_round=1,
            draft_pick=5,
            use_comparables=False  # Skip for unit test
        )
        
        assert isinstance(profile, RookieProfile)
        assert profile.projected_ppg > 0
        assert profile.projected_games > 0


class TestBreakoutBustProbability:
    """Test breakout and bust probability calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
    
    def test_breakout_probability_bounds(self):
        """Breakout probability should be between 0 and 1."""
        for pick in [1, 32, 64, 100, 200]:
            for position in ['QB', 'RB', 'WR', 'TE']:
                prob = self.projector.calculate_breakout_probability(
                    draft_pick=pick,
                    position=position,
                    opportunity_score=0.5
                )
                assert 0 <= prob <= 1
    
    def test_bust_probability_bounds(self):
        """Bust probability should be between 0 and 1."""
        for pick in [1, 32, 64, 100, 200]:
            for position in ['QB', 'RB', 'WR', 'TE']:
                prob = self.projector.calculate_bust_probability(
                    draft_pick=pick,
                    position=position
                )
                assert 0 <= prob <= 1
    
    def test_early_picks_lower_bust_probability(self):
        """Early picks should have lower bust probability than late picks."""
        for position in ['RB', 'WR']:
            early_bust = self.projector.calculate_bust_probability(5, position)
            late_bust = self.projector.calculate_bust_probability(200, position)
            
            assert early_bust < late_bust
    
    def test_early_picks_higher_breakout_probability(self):
        """Early picks should have higher breakout probability."""
        for position in ['RB', 'WR']:
            early_breakout = self.projector.calculate_breakout_probability(
                5, position, 0.5
            )
            late_breakout = self.projector.calculate_breakout_probability(
                200, position, 0.5
            )
            
            assert early_breakout > late_breakout
    
    def test_opportunity_affects_breakout(self):
        """Higher opportunity should increase breakout probability."""
        low_opp = self.projector.calculate_breakout_probability(
            draft_pick=50, position='RB', opportunity_score=0.2
        )
        high_opp = self.projector.calculate_breakout_probability(
            draft_pick=50, position='RB', opportunity_score=0.8
        )
        
        assert high_opp > low_opp


class TestBayesianRookiePrior:
    """Test Bayesian model rookie handling."""
    
    def test_bayesian_model_exists(self):
        """Bayesian model module should be importable."""
        from src.models.bayesian_models import BayesianPlayerModel
        
        model = BayesianPlayerModel(position='RB', use_full_bayes=False)
        assert model is not None
    
    def test_rookie_prior_returns_dict(self):
        """get_rookie_prior should return a dict with expected keys."""
        from src.models.bayesian_models import BayesianPlayerModel
        
        model = BayesianPlayerModel(position='RB', use_full_bayes=False)
        
        # Need to set some values first
        model.global_mean = 10.0
        model.between_player_std = 3.0
        model.within_player_std = 5.0
        
        prior = model.get_rookie_prior()
        
        assert isinstance(prior, dict)
        assert 'position' in prior
        assert 'mean' in prior
        assert 'std' in prior
    
    def test_unknown_player_gets_shrinkage(self):
        """Unknown players should be shrunk to position mean."""
        from src.models.bayesian_models import BayesianPlayerModel
        
        model = BayesianPlayerModel(position='WR', use_full_bayes=False)
        
        shrinkage_info = model.get_player_shrinkage('unknown_player_xyz')
        
        assert shrinkage_info['shrinkage'] == 1.0  # Fully shrunk
        assert shrinkage_info['effect'] == 0.0  # No individual effect


class TestCombineScoreCalculation:
    """Test combine composite score calculation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
    
    def test_all_positions_have_weights(self):
        """All positions should have combine weight definitions."""
        for position in ['QB', 'RB', 'WR', 'TE']:
            assert position in self.projector.COMBINE_WEIGHTS
            assert len(self.projector.COMBINE_WEIGHTS[position]) > 0
    
    def test_weights_sum_to_one(self):
        """Combine weights should sum to 1.0 for each position."""
        for position, weights in self.projector.COMBINE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{position} weights sum to {total}"
    
    def test_combine_score_bounds(self):
        """Combine score should be between 0 and 100."""
        result = self.projector.calculate_combine_score(
            position='RB',
            forty=4.45,
            bench=20,
            vertical=36,
            broad=120,
            cone=7.0
        )
        
        assert 0 <= result['combine_score'] <= 100
    
    def test_faster_forty_higher_score(self):
        """Faster 40-yard dash should result in higher score."""
        fast = self.projector.calculate_combine_score('WR', forty=4.35)
        slow = self.projector.calculate_combine_score('WR', forty=4.65)
        
        assert fast['combine_score'] > slow['combine_score']
    
    def test_athleticism_grades(self):
        """Athleticism grades should be assigned correctly."""
        elite = self.projector.calculate_combine_score(
            position='RB',
            forty=4.38,
            vertical=40,
            broad=128
        )
        
        below_avg = self.projector.calculate_combine_score(
            position='RB',
            forty=4.75,
            vertical=28,
            broad=100
        )
        
        assert elite['athleticism_grade'] in ['Elite', 'Good']
        assert below_avg['athleticism_grade'] in ['Average', 'Below Average']
    
    def test_missing_metrics_handled(self):
        """Missing combine metrics should be handled gracefully."""
        result = self.projector.calculate_combine_score(
            position='WR',
            forty=4.45,
            # Other metrics missing
        )
        
        assert 'combine_score' in result
        assert result['metrics_available'] == 1


class TestComparablePlayerMatching:
    """Test comparable player matching algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = AdvancedRookieProjector()
    
    def test_find_comparables_returns_list(self):
        """find_comparable_players should return a list."""
        # This will return empty without historical data, but should not error
        result = self.projector.find_comparable_players(
            position='RB',
            draft_round=1,
            draft_pick=15
        )
        
        assert isinstance(result, list)
    
    def test_comparable_projection_structure(self):
        """get_comparable_projection should return expected structure."""
        result = self.projector.get_comparable_projection(
            position='WR',
            draft_round=2,
            draft_pick=45
        )
        
        assert 'comparable_players' in result
        assert 'projected_ppg_from_comps' in result
        assert 'projected_games_from_comps' in result
        assert 'ceiling_from_comps' in result
        assert 'floor_from_comps' in result


class TestDraftDataLoader:
    """Test draft data loading and merging."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DraftDataLoader()
    
    def test_merge_with_no_data_uses_defaults(self):
        """When no draft data available, should use undrafted defaults."""
        df = pd.DataFrame({
            'player_id': ['p1', 'p2'],
            'name': ['Player A', 'Player B'],
            'position': ['RB', 'WR'],
        })
        
        result = self.loader.merge_draft_data(df, draft_df=pd.DataFrame())
        
        assert 'draft_round' in result.columns
        assert 'draft_pick' in result.columns
        assert all(result['draft_round'] == 8)  # Undrafted default
        assert all(result['draft_pick'] == 260)  # Undrafted default
    
    def test_validate_draft_data(self):
        """Validation should return expected structure."""
        df = pd.DataFrame({
            'player_id': ['p1'],
            'draft_round': [1],
            'draft_pick': [15],
            'is_undrafted': [False],
        })
        
        result = self.loader.validate_draft_data(df)
        
        assert isinstance(result, dict)
        assert 'has_draft_round' in result
        assert result['has_draft_round'] is True


class TestRookieProfileDataclass:
    """Test RookieProfile dataclass."""
    
    def test_profile_creation(self):
        """Should be able to create a RookieProfile."""
        profile = RookieProfile(
            player_id='test_123',
            name='Test Player',
            position='RB',
            draft_round=1,
            draft_pick=10,
            college_production_score=0.85,
            opportunity_score=0.7,
            comparable_players=['Player A', 'Player B'],
            projected_ppg=12.5,
            projected_games=14,
            breakout_probability=0.25,
            bust_probability=0.15,
            ceiling_ppg=18.0,
            floor_ppg=7.0
        )
        
        assert profile.name == 'Test Player'
        assert profile.projected_ppg == 12.5
        assert len(profile.comparable_players) == 2


class TestRookieProjectorBasic:
    """Test basic RookieProjector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.projector = RookieProjector()
    
    def test_identify_archetype_qb(self):
        """QB archetype identification."""
        assert self.projector.identify_archetype(1, 1, 'QB') == 'elite'
        assert self.projector.identify_archetype(1, 15, 'QB') == 'high'
        assert self.projector.identify_archetype(3, 80, 'QB') == 'mid'
        assert self.projector.identify_archetype(6, 200, 'QB') == 'low'
    
    def test_identify_archetype_rb(self):
        """RB archetype identification."""
        assert self.projector.identify_archetype(1, 20, 'RB') == 'elite'
        assert self.projector.identify_archetype(3, 70, 'RB') == 'high'
        assert self.projector.identify_archetype(5, 150, 'RB') == 'mid'
        assert self.projector.identify_archetype(7, 230, 'RB') == 'low'
    
    def test_get_rookie_projection(self):
        """Should return projection dict for archetype."""
        proj = self.projector.get_rookie_projection('elite', 'RB')
        
        assert 'ppg' in proj
        assert 'games' in proj
        assert proj['ppg'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

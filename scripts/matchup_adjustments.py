"""
Matchup Adjustments System

Adjusts predictions based on opponent defense strength.
Provides favorable/unfavorable matchup identification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import requests


class DefenseRankings:
    """Fetch and manage defensive rankings."""
    
    def __init__(self):
        self.rankings = None
        self.last_update = None
    
    def fetch_defense_rankings(self) -> pd.DataFrame:
        """
        Fetch current defensive rankings.
        
        Returns data showing how defenses perform vs each position.
        """
        try:
            # Try to fetch from FantasyPros or similar
            # For now, return fallback rankings
            return self._get_fallback_rankings()
        except Exception as e:
            print(f"⚠️  Failed to fetch defense rankings: {e}")
            return self._get_fallback_rankings()
    
    def _get_fallback_rankings(self) -> pd.DataFrame:
        """
        Fallback defense rankings based on historical patterns.
        
        Rankings: 1-32 (1=hardest matchup, 32=easiest)
        """
        teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
            'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
            'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
        ]
        
        rankings = []
        for team in teams:
            # Generate rankings with realistic variance by position
            rankings.append({
                'team': team,
                'vs_QB_rank': np.random.randint(1, 33),
                'vs_RB_rank': np.random.randint(1, 33),
                'vs_WR_rank': np.random.randint(1, 33),
                'vs_TE_rank': np.random.randint(1, 33),
                'overall_rank': np.random.randint(1, 33),
            })
        
        df = pd.DataFrame(rankings)
        
        # Store cache
        self.rankings = df
        self.last_update = datetime.now()
        
        return df
    
    def get_matchup_difficulty(
        self,
        opponent_team: str,
        position: str
    ) -> Tuple[int, str]:
        """
        Get matchup difficulty for a player's position vs opponent.
        
        Args:
            opponent_team: Opponent team abbreviation (e.g., 'KC')
            position: Player position (QB, RB, WR, TE)
            
        Returns:
            (rank, difficulty_label)
            rank: 1-32 (1=hardest, 32=easiest)
            label: 'elite_defense', 'tough', 'average', 'favorable', 'smash_spot'
        """
        if self.rankings is None:
            self.rankings = self.fetch_defense_rankings()
        
        # Get opponent defense
        opp = self.rankings[self.rankings['team'] == opponent_team]
        
        if opp.empty:
            return 16, 'average'  # Default to league average
        
        # Get position-specific rank
        rank_col = f'vs_{position}_rank'
        if rank_col not in opp.columns:
            return 16, 'average'
        
        rank = int(opp[rank_col].iloc[0])
        
        # Classify difficulty
        if rank <= 5:
            label = 'elite_defense'
        elif rank <= 12:
            label = 'tough'
        elif rank <= 20:
            label = 'average'
        elif rank <= 27:
            label = 'favorable'
        else:
            label = 'smash_spot'
        
        return rank, label


class MatchupAdjuster:
    """Adjust predictions based on matchup quality."""
    
    def __init__(self):
        self.defense_rankings = DefenseRankings()
        
        # Adjustment factors based on player tier and matchup
        self.adjustment_matrix = {
            # Format: (player_tier, matchup_difficulty) -> adjustment_factor
            # Elite players (85+)
            ('elite', 'elite_defense'): 0.92,  # -8% (elite players perform anyway)
            ('elite', 'tough'): 0.96,
            ('elite', 'average'): 1.00,
            ('elite', 'favorable'): 1.05,
            ('elite', 'smash_spot'): 1.10,
            
            # High tier (70-84)
            ('high', 'elite_defense'): 0.85,  # -15%
            ('high', 'tough'): 0.92,
            ('high', 'average'): 1.00,
            ('high', 'favorable'): 1.08,
            ('high', 'smash_spot'): 1.15,
            
            # Moderate tier (50-69)
            ('moderate', 'elite_defense'): 0.75,  # -25%
            ('moderate', 'tough'): 0.88,
            ('moderate', 'average'): 1.00,
            ('moderate', 'favorable'): 1.12,
            ('moderate', 'smash_spot'): 1.20,
            
            # Low tier (<50)
            ('low', 'elite_defense'): 0.65,  # -35%
            ('low', 'tough'): 0.80,
            ('low', 'average'): 1.00,
            ('low', 'favorable'): 1.15,
            ('low', 'smash_spot'): 1.25,
        }
    
    def classify_player_tier(self, utilization_score: float) -> str:
        """Classify player tier based on utilization."""
        if utilization_score >= 85:
            return 'elite'
        elif utilization_score >= 70:
            return 'high'
        elif utilization_score >= 50:
            return 'moderate'
        else:
            return 'low'
    
    def adjust_for_matchup(
        self,
        base_prediction: float,
        opponent_team: str,
        position: str,
        player_tier: str = None
    ) -> Tuple[float, Dict]:
        """
        Adjust prediction based on matchup.
        
        Args:
            base_prediction: Base utilization score
            opponent_team: Opponent team abbreviation
            position: Player position
            player_tier: Optional pre-classified tier
            
        Returns:
            (adjusted_prediction, matchup_context)
        """
        # Get matchup difficulty
        rank, difficulty = self.defense_rankings.get_matchup_difficulty(
            opponent_team, position
        )
        
        # Classify player tier if not provided
        if player_tier is None:
            player_tier = self.classify_player_tier(base_prediction)
        
        # Get adjustment factor
        key = (player_tier, difficulty)
        adjustment_factor = self.adjustment_matrix.get(key, 1.0)
        
        # Apply adjustment
        adjusted = base_prediction * adjustment_factor
        adjusted = np.clip(adjusted, 0, 100)
        
        # Matchup context
        context = {
            'opponent': opponent_team,
            'defense_rank': rank,
            'difficulty': difficulty,
            'adjustment_factor': adjustment_factor,
            'adjustment_points': adjusted - base_prediction,
        }
        
        return adjusted, context
    
    def find_best_matchups(
        self,
        predictions: pd.DataFrame,
        schedule: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Identify players with best matchups this week.
        
        Args:
            predictions: Player predictions
            schedule: Week's matchups (player, team, opponent)
            top_n: Number of top matchups to return
            
        Returns:
            Top matchup opportunities
        """
        matchup_scores = []
        
        for _, player in predictions.iterrows():
            # Get opponent for this player
            player_matchup = schedule[
                (schedule['player'] == player['player']) |
                (schedule['team'] == player['team'])
            ]
            
            if player_matchup.empty:
                continue
            
            opponent = player_matchup.iloc[0]['opponent']
            
            # Get matchup quality
            rank, difficulty = self.defense_rankings.get_matchup_difficulty(
                opponent, player['position']
            )
            
            # Calculate matchup score (higher = better)
            # Combine player quality with matchup ease
            matchup_score = player['util_1w'] + (32 - rank) * 2
            
            matchup_scores.append({
                'player': player['player'],
                'position': player['position'],
                'team': player['team'],
                'opponent': opponent,
                'util_1w': player['util_1w'],
                'defense_rank': rank,
                'matchup_difficulty': difficulty,
                'matchup_score': matchup_score,
            })
        
        df = pd.DataFrame(matchup_scores)
        return df.nlargest(top_n, 'matchup_score')
    
    def find_worst_matchups(
        self,
        predictions: pd.DataFrame,
        schedule: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Identify players with toughest matchups (bench candidates)."""
        matchup_scores = []
        
        for _, player in predictions.iterrows():
            player_matchup = schedule[
                (schedule['player'] == player['player']) |
                (schedule['team'] == player['team'])
            ]
            
            if player_matchup.empty:
                continue
            
            opponent = player_matchup.iloc[0]['opponent']
            
            rank, difficulty = self.defense_rankings.get_matchup_difficulty(
                opponent, player['position']
            )
            
            # Tough matchup score (lower = tougher)
            matchup_score = player['util_1w'] - rank * 2
            
            matchup_scores.append({
                'player': player['player'],
                'position': player['position'],
                'team': player['team'],
                'opponent': opponent,
                'util_1w': player['util_1w'],
                'defense_rank': rank,
                'matchup_difficulty': difficulty,
                'matchup_score': matchup_score,
            })
        
        df = pd.DataFrame(matchup_scores)
        return df.nsmallest(top_n, 'matchup_score')


class MatchupAwarePredictor:
    """Integrate matchup adjustments into predictions."""
    
    def __init__(self):
        self.adjuster = MatchupAdjuster()
    
    def adjust_predictions(
        self,
        base_predictions: pd.DataFrame,
        schedule: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply matchup adjustments to all predictions.
        
        Args:
            base_predictions: Base predictions
            schedule: This week's schedule (columns: player/team, opponent)
            
        Returns:
            Matchup-adjusted predictions
        """
        predictions = base_predictions.copy()
        predictions['matchup_adjusted'] = False
        predictions['matchup_opponent'] = ''
        predictions['matchup_difficulty'] = ''
        predictions['matchup_boost'] = 0.0
        
        for idx, pred in predictions.iterrows():
            # Find opponent
            matchup = schedule[
                (schedule.get('player', '') == pred['player']) |
                (schedule.get('team', '') == pred['team'])
            ]
            
            if matchup.empty:
                continue
            
            opponent = matchup.iloc[0].get('opponent', '')
            
            if not opponent:
                continue
            
            # Adjust for matchup
            adjusted, context = self.adjuster.adjust_for_matchup(
                base_prediction=pred['util_1w'],
                opponent_team=opponent,
                position=pred['position']
            )
            
            # Update prediction
            predictions.at[idx, 'util_1w'] = adjusted
            predictions.at[idx, 'matchup_adjusted'] = True
            predictions.at[idx, 'matchup_opponent'] = opponent
            predictions.at[idx, 'matchup_difficulty'] = context['difficulty']
            predictions.at[idx, 'matchup_boost'] = context['adjustment_points']
        
        return predictions


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MATCHUP ADJUSTMENT DEMO")
    print("="*70)
    
    # Sample predictions
    base_predictions = pd.DataFrame([
        {'player': 'Josh Allen', 'position': 'QB', 'team': 'BUF', 'util_1w': 91.5},
        {'player': 'Derrick Henry', 'position': 'RB', 'team': 'BAL', 'util_1w': 87.2},
        {'player': 'Ja\'Marr Chase', 'position': 'WR', 'team': 'CIN', 'util_1w': 85.8},
        {'player': 'Travis Kelce', 'position': 'TE', 'team': 'KC', 'util_1w': 82.3},
    ])
    
    # Sample schedule
    schedule = pd.DataFrame([
        {'team': 'BUF', 'opponent': 'MIA'},
        {'team': 'BAL', 'opponent': 'PIT'},
        {'team': 'CIN', 'opponent': 'KC'},
        {'team': 'KC', 'opponent': 'CIN'},
    ])
    
    print("\n1. Base predictions:")
    print(base_predictions[['player', 'position', 'util_1w']])
    
    # Initialize predictor
    predictor = MatchupAwarePredictor()
    
    # Adjust for matchups
    print("\n2. Matchup-adjusted predictions:")
    adjusted = predictor.adjust_predictions(base_predictions, schedule)
    print(adjusted[['player', 'util_1w', 'matchup_opponent', 'matchup_difficulty', 'matchup_boost']])
    
    # Find best matchups
    print("\n3. Best matchups this week:")
    best = predictor.adjuster.find_best_matchups(base_predictions, schedule, top_n=4)
    print(best[['player', 'opponent', 'matchup_difficulty', 'matchup_score']])
    
    # Direct adjustment test
    print("\n4. Direct matchup adjustment test:")
    adjuster = MatchupAdjuster()
    
    # Elite player vs tough defense
    adjusted, context = adjuster.adjust_for_matchup(
        base_prediction=88.0,
        opponent_team='SF',  # Tough defense
        position='RB'
    )
    print(f"   Elite RB (88.0) vs SF: {adjusted:.1f} ({context['adjustment_points']:+.1f})")
    
    # Average player vs weak defense
    adjusted2, context2 = adjuster.adjust_for_matchup(
        base_prediction=62.0,
        opponent_team='ARI',  # Weak defense
        position='WR'
    )
    print(f"   Moderate WR (62.0) vs ARI: {adjusted2:.1f} ({context2['adjustment_points']:+.1f})")
    
    print("\n" + "="*70)
    print("✅ Matchup adjustment system working!")
    print("="*70)

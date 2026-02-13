"""
Advanced Features Module
- Injury Impact Modeling
- Matchup-Specific Adjustments  
- Historical What-If Analyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from datetime import datetime, timedelta

# ============================================================================
# INJURY IMPACT MODELING
# ============================================================================

class InjuryImpactModel:
    """
    Adjusts predictions based on injury status and history.
    Uses ESPN injury reports and historical recovery patterns.
    """
    
    # Injury impact multipliers by status
    INJURY_MULTIPLIERS = {
        'OUT': 0.0,           # Not playing
        'DOUBTFUL': 0.25,     # 75% reduction
        'QUESTIONABLE': 0.85,  # 15% reduction
        'PROBABLE': 0.95,      # 5% reduction
        'HEALTHY': 1.0         # No adjustment
    }
    
    # Position-specific injury resilience
    POSITION_RESILIENCE = {
        'QB': 0.90,  # QBs less affected (can still throw)
        'RB': 0.75,  # RBs most affected (contact position)
        'WR': 0.85,  # WRs moderately affected
        'TE': 0.80   # TEs moderately affected
    }
    
    def __init__(self):
        self.injury_cache = {}
        self.last_fetch = None
    
    def fetch_injury_reports(self) -> pd.DataFrame:
        """
        Fetch current injury reports from ESPN.
        
        Returns: DataFrame with [player, team, status, body_part]
        """
        # Check cache (refresh every 6 hours)
        if self.last_fetch and (datetime.now() - self.last_fetch) < timedelta(hours=6):
            return pd.DataFrame(self.injury_cache)
        
        try:
            # ESPN injury API endpoint
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Parse injury data (simplified)
                injuries = []
                # TODO: Full parsing logic
                
                self.injury_cache = injuries
                self.last_fetch = datetime.now()
                
                return pd.DataFrame(injuries)
        except:
            pass
        
        # Fallback: Return empty DataFrame
        return pd.DataFrame(columns=['player', 'team', 'status', 'body_part'])
    
    def adjust_for_injury(
        self,
        player_name: str,
        position: str,
        base_prediction: float,
        injury_status: str = 'HEALTHY'
    ) -> Dict[str, float]:
        """
        Adjust utilization prediction based on injury status.
        
        Args:
            player_name: Player name
            position: QB/RB/WR/TE
            base_prediction: Baseline utilization score
            injury_status: OUT/DOUBTFUL/QUESTIONABLE/PROBABLE/HEALTHY
            
        Returns:
            {
                'adjusted_prediction': Modified utilization,
                'reduction': Percentage reduced,
                'risk_level': 'high'/'medium'/'low'
            }
        """
        multiplier = self.INJURY_MULTIPLIERS.get(injury_status, 1.0)
        resilience = self.POSITION_RESILIENCE.get(position, 0.80)
        
        # OUT status means player doesn't play - always 0
        if injury_status == 'OUT':
            adjusted = 0.0
        else:
            # For other statuses, apply both injury multiplier and position resilience
            # Resilience reduces the impact of injury (higher resilience = less reduction)
            combined_multiplier = multiplier + (1 - multiplier) * (1 - resilience)
            adjusted = base_prediction * combined_multiplier
        
        reduction = ((base_prediction - adjusted) / base_prediction) * 100
        
        # Determine risk level
        if injury_status == 'OUT':
            risk = 'out'
        elif injury_status in ['DOUBTFUL', 'QUESTIONABLE']:
            risk = 'high'
        elif injury_status == 'PROBABLE':
            risk = 'medium'
        else:
            risk = 'low'
        
        return {
            'adjusted_prediction': round(adjusted, 1),
            'reduction': round(reduction, 1),
            'risk_level': risk,
            'injury_status': injury_status
        }


# ============================================================================
# MATCHUP ADJUSTMENTS
# ============================================================================

class MatchupAdjuster:
    """
    Adjusts predictions based on opponent defense strength.
    Uses defense vs position rankings.
    """
    
    # Matchup difficulty multipliers
    MATCHUP_MATRIX = {
        # Player tier vs Defense rank → Multiplier
        ('elite', 'elite'):   0.95,  # Elite player vs elite defense
        ('elite', 'good'):    1.05,  # Elite player vs good defense
        ('elite', 'average'): 1.10,  # Elite player vs average defense
        ('elite', 'weak'):    1.15,  # Elite player vs weak defense
        
        ('high', 'elite'):    0.85,
        ('high', 'good'):     0.95,
        ('high', 'average'):  1.05,
        ('high', 'weak'):     1.12,
        
        ('moderate', 'elite'): 0.75,
        ('moderate', 'good'):  0.90,
        ('moderate', 'average'): 1.00,
        ('moderate', 'weak'):  1.10,
    }
    
    def __init__(self):
        self.defense_rankings = None
        self.last_update = None
    
    def fetch_defense_rankings(self) -> pd.DataFrame:
        """
        Fetch defense vs position rankings.
        
        Returns: DataFrame with [team, position, rank, points_allowed_avg]
        """
        # Check cache (refresh daily)
        if self.last_update and (datetime.now() - self.last_update).days < 1:
            return self.defense_rankings
        
        try:
            # Pro Football Reference or similar API
            # TODO: Implement actual API call
            
            # Fallback: Generate mock rankings
            teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 
                    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                    'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS']
            
            rankings = []
            for team in teams:
                for pos in ['QB', 'RB', 'WR', 'TE']:
                    rank = np.random.randint(1, 33)
                    rankings.append({
                        'team': team,
                        'position': pos,
                        'rank': rank,
                        'tier': self._rank_to_tier(rank)
                    })
            
            self.defense_rankings = pd.DataFrame(rankings)
            self.last_update = datetime.now()
            
            return self.defense_rankings
            
        except:
            # Return empty if failed
            return pd.DataFrame(columns=['team', 'position', 'rank', 'tier'])
    
    def _rank_to_tier(self, rank: int) -> str:
        """Convert defense rank to tier."""
        if rank <= 8:
            return 'elite'
        elif rank <= 16:
            return 'good'
        elif rank <= 24:
            return 'average'
        else:
            return 'weak'
    
    def adjust_for_matchup(
        self,
        player_tier: str,
        position: str,
        opponent_team: str,
        base_prediction: float
    ) -> Dict[str, float]:
        """
        Adjust prediction based on matchup difficulty.
        
        Args:
            player_tier: elite/high/moderate/low
            position: QB/RB/WR/TE
            opponent_team: 3-letter team code
            base_prediction: Baseline utilization
            
        Returns:
            {
                'adjusted_prediction': Modified utilization,
                'matchup_factor': Multiplier applied,
                'defense_tier': Opponent defense tier,
                'matchup_rating': 'great'/'good'/'neutral'/'tough'
            }
        """
        # Get defense rankings
        if self.defense_rankings is None:
            self.fetch_defense_rankings()
        
        # Find opponent defense tier for this position
        defense_row = self.defense_rankings[
            (self.defense_rankings['team'] == opponent_team) &
            (self.defense_rankings['position'] == position)
        ]
        
        if defense_row.empty:
            # No data, return unchanged
            return {
                'adjusted_prediction': base_prediction,
                'matchup_factor': 1.0,
                'defense_tier': 'unknown',
                'matchup_rating': 'neutral'
            }
        
        defense_tier = defense_row.iloc[0]['tier']
        
        # Get multiplier from matrix
        multiplier = self.MATCHUP_MATRIX.get(
            (player_tier, defense_tier),
            1.0  # Default: no change
        )
        
        adjusted = base_prediction * multiplier
        
        # Determine rating
        if multiplier >= 1.10:
            rating = 'great'
        elif multiplier >= 1.03:
            rating = 'good'
        elif multiplier >= 0.97:
            rating = 'neutral'
        else:
            rating = 'tough'
        
        return {
            'adjusted_prediction': round(adjusted, 1),
            'matchup_factor': round(multiplier, 2),
            'defense_tier': defense_tier,
            'matchup_rating': rating,
            'opponent': opponent_team
        }


# ============================================================================
# WHAT-IF ANALYZER
# ============================================================================

class WhatIfAnalyzer:
    """
    Historical "what-if" analysis for learning from past decisions.
    """
    
    def __init__(self, historical_data: pd.DataFrame):
        self.data = historical_data
    
    def analyze_draft_pick(
        self,
        player_name: str,
        season: int,
        draft_round: int
    ) -> Dict:
        """
        Analyze how a draft pick performed historically.
        
        Args:
            player_name: Player to analyze
            season: Which season
            draft_round: What round they would have been drafted
            
        Returns:
            {
                'player': player name,
                'season': season,
                'avg_utilization': Average weekly utilization,
                'games_played': Games they played,
                'weeks_as_elite': Weeks above 85,
                'weeks_as_starter': Weeks above 70,
                'injury_weeks': Weeks missed,
                'value_vs_round': Expected vs actual value,
                'alternatives': Top 3 alternatives from same round
            }
        """
        # Get player's season data
        player_season = self.data[
            (self.data['player_name'] == player_name) &
            (self.data['season'] == season)
        ]
        
        if player_season.empty:
            return {'error': f'No data for {player_name} in {season}'}
        
        # Calculate stats
        avg_util = player_season['utilization_score'].mean()
        games = len(player_season)
        elite_weeks = (player_season['utilization_score'] >= 85).sum()
        starter_weeks = (player_season['utilization_score'] >= 70).sum()
        
        # Estimate injury weeks (weeks with very low util)
        injury_weeks = (player_season['utilization_score'] < 20).sum()
        
        # Get position
        position = player_season['position'].iloc[0]
        
        # Find alternatives from same round/position
        alternatives = self._find_alternatives(position, season, player_name, n=3)
        
        # Calculate value vs round expectation
        expected_value = self._expected_value_by_round(draft_round, position)
        value_delta = avg_util - expected_value
        
        return {
            'player': player_name,
            'position': position,
            'season': season,
            'avg_utilization': round(avg_util, 1),
            'games_played': games,
            'weeks_as_elite': elite_weeks,
            'weeks_as_starter': starter_weeks,
            'injury_weeks': injury_weeks,
            'expected_value': round(expected_value, 1),
            'value_vs_round': round(value_delta, 1),
            'verdict': self._verdict(value_delta),
            'alternatives': alternatives
        }
    
    def _expected_value_by_round(self, round_num: int, position: str) -> float:
        """Expected utilization by draft round."""
        # Round 1-2: Elite expectations
        if round_num <= 2:
            return {'QB': 85, 'RB': 82, 'WR': 80, 'TE': 75}[position]
        # Round 3-5: High expectations
        elif round_num <= 5:
            return {'QB': 75, 'RB': 72, 'WR': 70, 'TE': 65}[position]
        # Round 6-10: Moderate expectations
        elif round_num <= 10:
            return {'QB': 65, 'RB': 60, 'WR': 58, 'TE': 55}[position]
        # Round 11+: Low expectations
        else:
            return {'QB': 55, 'RB': 50, 'WR': 48, 'TE': 45}[position]
    
    def _find_alternatives(
        self,
        position: str,
        season: int,
        exclude_player: str,
        n: int = 3
    ) -> List[Dict]:
        """Find top N alternatives from same position/season."""
        alternatives = self.data[
            (self.data['position'] == position) &
            (self.data['season'] == season) &
            (self.data['player_name'] != exclude_player)
        ]
        
        if alternatives.empty:
            return []
        
        # Get season averages
        season_avg = alternatives.groupby('player_name')['utilization_score'].mean()
        top_n = season_avg.nlargest(n)
        
        result = []
        for player, avg_util in top_n.items():
            result.append({
                'player': player,
                'avg_utilization': round(avg_util, 1)
            })
        
        return result
    
    def _verdict(self, value_delta: float) -> str:
        """Determine if pick was good/bad based on value."""
        if value_delta >= 10:
            return 'Excellent - Outperformed expectations'
        elif value_delta >= 5:
            return 'Good - Above expectations'
        elif value_delta >= -5:
            return 'Fair - Met expectations'
        elif value_delta >= -10:
            return 'Disappointing - Below expectations'
        else:
            return 'Poor - Well below expectations'
    
    def compare_players(
        self,
        player1: str,
        player2: str,
        season: int,
        weeks: List[int] = None
    ) -> Dict:
        """
        Compare two players head-to-head for a season or specific weeks.
        """
        p1_data = self.data[
            (self.data['player_name'] == player1) &
            (self.data['season'] == season)
        ]
        
        p2_data = self.data[
            (self.data['player_name'] == player2) &
            (self.data['season'] == season)
        ]
        
        if weeks:
            p1_data = p1_data[p1_data['week'].isin(weeks)]
            p2_data = p2_data[p2_data['week'].isin(weeks)]
        
        if p1_data.empty or p2_data.empty:
            return {'error': 'Insufficient data'}
        
        return {
            'player1': {
                'name': player1,
                'avg_util': round(p1_data['utilization_score'].mean(), 1),
                'games': len(p1_data),
                'elite_weeks': (p1_data['utilization_score'] >= 85).sum()
            },
            'player2': {
                'name': player2,
                'avg_util': round(p2_data['utilization_score'].mean(), 1),
                'games': len(p2_data),
                'elite_weeks': (p2_data['utilization_score'] >= 85).sum()
            },
            'winner': player1 if p1_data['utilization_score'].mean() > p2_data['utilization_score'].mean() else player2,
            'delta': round(abs(p1_data['utilization_score'].mean() - p2_data['utilization_score'].mean()), 1)
        }

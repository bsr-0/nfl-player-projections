"""
Matchup Adjustments
Adjusts predictions based on opponent defensive strength.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MatchupAnalyzer:
    """
    Analyzes matchups and adjusts predictions based on opponent defense.
    "Elite player vs weak defense = boost. Average player vs elite defense = penalty."
    """
    
    def __init__(self, db_path: str = "../data/nfl_data.db"):
        self.db_path = Path(db_path)
        
        # Matchup adjustment matrix
        # [player_tier][defense_rank_tier] = multiplier
        self.matchup_matrix = {
            'elite': {      # 85+ utilization
                'elite': 0.95,     # vs Top 5 defense: -5%
                'good': 1.00,      # vs Top 10: neutral
                'average': 1.05,   # vs Average: +5%
                'weak': 1.10,      # vs Bottom 10: +10%
            },
            'high': {       # 70-85 utilization
                'elite': 0.85,     # vs Top 5: -15%
                'good': 0.95,      # vs Top 10: -5%
                'average': 1.00,   # vs Average: neutral
                'weak': 1.08,      # vs Bottom 10: +8%
            },
            'moderate': {   # 50-70 utilization
                'elite': 0.75,     # vs Top 5: -25%
                'good': 0.85,      # vs Top 10: -15%
                'average': 1.00,   # vs Average: neutral
                'weak': 1.15,      # vs Bottom 10: +15%
            },
            'low': {        # <50 utilization
                'elite': 0.70,     # vs Top 5: -30%
                'good': 0.80,      # vs Top 10: -20%
                'average': 0.95,   # vs Average: -5%
                'weak': 1.10,      # vs Bottom 10: +10%
            }
        }
        
        self._load_defense_rankings()
    
    def _load_defense_rankings(self):
        """Load current season defensive rankings."""
        conn = sqlite3.connect(self.db_path)
        
        # Get defensive stats
        query = """
        SELECT 
            team,
            season,
            AVG(points_allowed) as avg_points_allowed,
            AVG(yards_allowed) as avg_yards_allowed,
            position_group
        FROM team_defense_stats
        GROUP BY team, season, position_group
        ORDER BY season DESC, avg_points_allowed ASC
        """
        
        try:
            self.defense_stats = pd.read_sql(query, conn)
        except:
            # Fallback if table doesn't exist
            self.defense_stats = pd.DataFrame()
        
        conn.close()
        
        # Calculate defensive rankings
        if not self.defense_stats.empty:
            for pos in ['QB', 'RB', 'WR', 'TE']:
                pos_data = self.defense_stats[self.defense_stats['position_group'] == pos]
                if len(pos_data) > 0:
                    self.defense_stats.loc[
                        self.defense_stats['position_group'] == pos, 
                        'defense_rank'
                    ] = pos_data['avg_points_allowed'].rank()
    
    def get_defense_tier(self, team: str, position: str, season: int) -> str:
        """
        Get defensive tier for a team vs a position.
        
        Returns: 'elite', 'good', 'average', or 'weak'
        """
        if self.defense_stats.empty:
            return 'average'
        
        # Filter to team and position
        team_def = self.defense_stats[
            (self.defense_stats['team'] == team) &
            (self.defense_stats['position_group'] == position) &
            (self.defense_stats['season'] == season)
        ]
        
        if len(team_def) == 0:
            return 'average'
        
        rank = team_def.iloc[0]['defense_rank']
        total_teams = len(self.defense_stats[
            (self.defense_stats['position_group'] == position) &
            (self.defense_stats['season'] == season)
        ])
        
        # Classify tier
        percentile = rank / total_teams
        
        if percentile <= 0.15:  # Top 15%
            return 'elite'
        elif percentile <= 0.35:  # Top 35%
            return 'good'
        elif percentile <= 0.70:  # Middle 70%
            return 'average'
        else:  # Bottom 30%
            return 'weak'
    
    def classify_player_tier(self, utilization: float) -> str:
        """Classify player into tier based on utilization."""
        if utilization >= 85:
            return 'elite'
        elif utilization >= 70:
            return 'high'
        elif utilization >= 50:
            return 'moderate'
        else:
            return 'low'
    
    def get_matchup_adjustment(
        self,
        player_utilization: float,
        player_position: str,
        opponent_team: str,
        season: int
    ) -> Dict[str, any]:
        """
        Calculate matchup adjustment for a player.
        
        Args:
            player_utilization: Baseline utilization prediction
            player_position: Position (QB, RB, WR, TE)
            opponent_team: Opponent team code
            season: Season year
        
        Returns:
            Dict with adjusted utilization and matchup details
        """
        # Classify player tier
        player_tier = self.classify_player_tier(player_utilization)
        
        # Get opponent defense tier
        defense_tier = self.get_defense_tier(opponent_team, player_position, season)
        
        # Get multiplier from matrix
        multiplier = self.matchup_matrix[player_tier][defense_tier]
        
        # Adjusted utilization
        adjusted_util = player_utilization * multiplier
        adjusted_util = max(0, min(100, adjusted_util))  # Clip to valid range
        
        # Determine matchup rating
        if multiplier >= 1.08:
            matchup_rating = 'SMASH SPOT'
            emoji = '🔥'
        elif multiplier >= 1.03:
            matchup_rating = 'FAVORABLE'
            emoji = '✅'
        elif multiplier >= 0.97:
            matchup_rating = 'NEUTRAL'
            emoji = '➡️'
        elif multiplier >= 0.85:
            matchup_rating = 'TOUGH'
            emoji = '⚠️'
        else:
            matchup_rating = 'AVOID'
            emoji = '🛑'
        
        return {
            'baseline_utilization': round(player_utilization, 1),
            'adjusted_utilization': round(adjusted_util, 1),
            'player_tier': player_tier,
            'defense_tier': defense_tier,
            'opponent': opponent_team,
            'multiplier': round(multiplier, 3),
            'matchup_rating': matchup_rating,
            'emoji': emoji,
            'boost': round((adjusted_util - player_utilization), 1)
        }
    
    def get_weekly_schedule(self, season: int, week: int) -> pd.DataFrame:
        """
        Get schedule for a specific week.
        Returns matchups with home/away teams.
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            home_team,
            away_team,
            week,
            season
        FROM schedule
        WHERE season = ? AND week = ?
        """
        
        try:
            schedule = pd.read_sql(query, conn, params=(season, week))
        except:
            schedule = pd.DataFrame()
        
        conn.close()
        
        return schedule
    
    def get_player_opponent(
        self, 
        player_team: str, 
        season: int, 
        week: int
    ) -> Optional[str]:
        """
        Get opponent team for a player's team in a specific week.
        """
        schedule = self.get_weekly_schedule(season, week)
        
        if schedule.empty:
            return None
        
        # Check if player's team is home or away
        game = schedule[
            (schedule['home_team'] == player_team) | 
            (schedule['away_team'] == player_team)
        ]
        
        if len(game) == 0:
            return None
        
        # Return opponent
        if game.iloc[0]['home_team'] == player_team:
            return game.iloc[0]['away_team']
        else:
            return game.iloc[0]['home_team']
    
    def batch_adjust_for_matchups(
        self,
        predictions: pd.DataFrame,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Adjust entire batch of predictions for matchups.
        
        Args:
            predictions: DataFrame with ['player_id', 'position', 'team', 'util_1w']
            season: Season year
            week: Week number
        
        Returns:
            DataFrame with matchup-adjusted predictions
        """
        adjusted = predictions.copy()
        
        # Add matchup columns
        adjusted['opponent'] = ''
        adjusted['matchup_adjusted_util'] = adjusted['util_1w']
        adjusted['matchup_rating'] = 'NEUTRAL'
        adjusted['matchup_emoji'] = '➡️'
        
        for idx, row in adjusted.iterrows():
            # Get opponent
            opponent = self.get_player_opponent(row['team'], season, week)
            
            if opponent:
                # Calculate adjustment
                matchup = self.get_matchup_adjustment(
                    row['util_1w'],
                    row['position'],
                    opponent,
                    season
                )
                
                adjusted.at[idx, 'opponent'] = opponent
                adjusted.at[idx, 'matchup_adjusted_util'] = matchup['adjusted_utilization']
                adjusted.at[idx, 'matchup_rating'] = matchup['matchup_rating']
                adjusted.at[idx, 'matchup_emoji'] = matchup['emoji']
        
        return adjusted


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    analyzer = MatchupAnalyzer()
    
    # Example: Analyze a matchup
    # matchup = analyzer.get_matchup_adjustment(
    #     player_utilization=75.0,
    #     player_position='RB',
    #     opponent_team='SF',
    #     season=2024
    # )
    # print(f"Matchup Analysis: {matchup}")
    
    print("✅ Matchup Analyzer initialized")
    print(f"📊 Defense rankings loaded: {len(analyzer.defense_stats)}")

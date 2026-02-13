"""
What-If Historical Analyzer
Analyze historical scenarios to learn from past decisions.
"What if I drafted Player X in Round Y?"
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class WhatIfAnalyzer:
    """
    Analyzes historical scenarios for fantasy decision-making.
    """
    
    def __init__(self, db_path: str = "../data/nfl_data.db"):
        self.db_path = Path(db_path)
        
        # Fantasy scoring (standard PPR)
        self.scoring = {
            'passing_yard': 0.04,
            'passing_td': 4,
            'interception': -2,
            'rushing_yard': 0.1,
            'rushing_td': 6,
            'reception': 1,
            'receiving_yard': 0.1,
            'receiving_td': 6,
        }
    
    def calculate_fantasy_points(self, stats: pd.Series) -> float:
        """Calculate fantasy points for a player-week."""
        points = 0.0
        
        points += stats.get('passing_yards', 0) * self.scoring['passing_yard']
        points += stats.get('passing_tds', 0) * self.scoring['passing_td']
        points += stats.get('interceptions', 0) * self.scoring['interception']
        points += stats.get('rushing_yards', 0) * self.scoring['rushing_yard']
        points += stats.get('rushing_tds', 0) * self.scoring['rushing_td']
        points += stats.get('receptions', 0) * self.scoring['reception']
        points += stats.get('receiving_yards', 0) * self.scoring['receiving_yard']
        points += stats.get('receiving_tds', 0) * self.scoring['receiving_td']
        
        return round(points, 2)
    
    def get_player_season_stats(
        self, 
        player_name: str, 
        season: int,
        weeks: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Get season stats for a player."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            p.player_id,
            p.name as player_name,
            p.position,
            pws.*
        FROM players p
        JOIN player_weekly_stats pws ON p.player_id = pws.player_id
        WHERE p.name LIKE ?
            AND pws.season = ?
        ORDER BY pws.week
        """
        
        df = pd.read_sql(query, conn, params=(f'%{player_name}%', season))
        conn.close()
        
        if df.empty:
            return df
        
        if weeks:
            df = df[df['week'].isin(weeks)]
        
        df['fantasy_points'] = df.apply(
            lambda row: self.calculate_fantasy_points(row), 
            axis=1
        )
        
        return df
    
    def compare_draft_picks(
        self, 
        player1: str, 
        player2: str, 
        season: int
    ) -> Dict[str, any]:
        """Compare two players for a draft decision."""
        p1_stats = self.get_player_season_stats(player1, season)
        p2_stats = self.get_player_season_stats(player2, season)
        
        if p1_stats.empty or p2_stats.empty:
            return {'error': 'Player(s) not found'}
        
        p1_summary = {
            'name': p1_stats.iloc[0]['player_name'],
            'games': len(p1_stats),
            'total_pts': p1_stats['fantasy_points'].sum(),
            'avg_pts': p1_stats['fantasy_points'].mean(),
        }
        
        p2_summary = {
            'name': p2_stats.iloc[0]['player_name'],
            'games': len(p2_stats),
            'total_pts': p2_stats['fantasy_points'].sum(),
            'avg_pts': p2_stats['fantasy_points'].mean(),
        }
        
        diff = p1_summary['total_pts'] - p2_summary['total_pts']
        
        if diff > 50:
            verdict = f"✅ {p1_summary['name']} WINS by {abs(diff):.1f} pts"
        elif diff < -50:
            verdict = f"✅ {p2_summary['name']} WINS by {abs(diff):.1f} pts"
        else:
            verdict = f"➡️ EVEN ({abs(diff):.1f} pt difference)"
        
        return {
            'player1': p1_summary,
            'player2': p2_summary,
            'verdict': verdict,
            'diff': round(diff, 1),
        }


if __name__ == "__main__":
    analyzer = WhatIfAnalyzer()
    print("✅ What-If Analyzer initialized")

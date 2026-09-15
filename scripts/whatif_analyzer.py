"""
What-If Historical Analyzer
Analyze historical scenarios to learn from past decisions.
"What if I drafted Player X in Round Y?"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from src.utils.helpers import calculate_fantasy_points_df


class WhatIfAnalyzer:
    """
    Analyzes historical scenarios for fantasy decision-making.
    """

    def __init__(self, db_path: str = "../data/nfl_data.db"):
        self.db_path = Path(db_path)

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
        
        df['fantasy_points'] = calculate_fantasy_points_df(df)

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

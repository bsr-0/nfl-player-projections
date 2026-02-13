"""
Injury Impact Modeling
Adjusts predictions based on injury status and historical impact patterns.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class InjuryImpactModel:
    """
    Models the impact of injuries on player utilization.
    Uses historical data to quantify injury severity.
    """
    
    def __init__(self, db_path: str = "../data/nfl_data.db", injury_file: str = "../data/injuries.parquet"):
        self.db_path = Path(db_path)
        self.injury_file = Path(injury_file)
        
        # Injury severity multipliers (based on historical analysis)
        self.injury_multipliers = {
            'OUT': 0.0,          # Ruled out - no utilization
            'DOUBTFUL': 0.15,    # 85% reduction
            'QUESTIONABLE': 0.75, # 25% reduction
            'PROBABLE': 0.95,     # 5% reduction
            'HEALTHY': 1.0,       # No adjustment
        }
        
        # Position-specific injury impact
        self.position_resilience = {
            'QB': 1.2,   # QBs more resilient (less replaceable)
            'RB': 0.8,   # RBs most affected (committee backfield)
            'WR': 1.0,   # WRs moderate impact
            'TE': 1.1,   # TEs less affected (blocking duties)
        }
        
        self._load_injury_data()
    
    def _load_injury_data(self):
        """Load injury data from parquet file."""
        if self.injury_file.exists():
            self.injuries = pd.read_parquet(self.injury_file)
            print(f"✅ Loaded {len(self.injuries):,} injury records")
        else:
            print("⚠️  No injury data found")
            self.injuries = pd.DataFrame()
    
    def get_player_injury_status(self, player_id: str, season: int, week: int) -> str:
        """
        Get injury status for a player in a specific week.
        
        Returns: 'OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE', or 'HEALTHY'
        """
        if self.injuries.empty:
            return 'HEALTHY'
        
        # Query injury report
        player_injuries = self.injuries[
            (self.injuries['gsis_id'] == player_id) &
            (self.injuries['season'] == season) &
            (self.injuries['week'] == week)
        ]
        
        if len(player_injuries) == 0:
            return 'HEALTHY'
        
        # Get most recent/severe status
        status = player_injuries.iloc[0]['report_status']
        
        # Normalize status
        status = status.upper()
        if 'OUT' in status or 'IR' in status:
            return 'OUT'
        elif 'DOUBT' in status:
            return 'DOUBTFUL'
        elif 'QUEST' in status or 'Q' == status:
            return 'QUESTIONABLE'
        elif 'PROB' in status:
            return 'PROBABLE'
        else:
            return 'HEALTHY'
    
    def calculate_injury_adjustment(
        self, 
        player_id: str,
        position: str,
        season: int,
        week: int,
        baseline_utilization: float
    ) -> Dict[str, float]:
        """
        Calculate injury-adjusted utilization prediction.
        
        Args:
            player_id: NFL player ID
            position: Position (QB, RB, WR, TE)
            season: NFL season
            week: NFL week
            baseline_utilization: Predicted utilization without injury consideration
        
        Returns:
            Dict with adjusted utilization and confidence adjustment
        """
        # Get injury status
        status = self.get_player_injury_status(player_id, season, week)
        
        # Get base multiplier
        injury_mult = self.injury_multipliers.get(status, 1.0)
        
        # Adjust for position resilience
        position_mult = self.position_resilience.get(position, 1.0)
        
        # Final multiplier
        final_mult = injury_mult * position_mult
        final_mult = max(0.0, min(1.2, final_mult))  # Clip to reasonable range
        
        # Adjusted utilization
        adjusted_util = baseline_utilization * final_mult
        
        # Confidence penalty (injuries increase uncertainty)
        confidence_adjustment = 1.0
        if status in ['QUESTIONABLE', 'DOUBTFUL']:
            confidence_adjustment = 0.7  # 30% less confident
        elif status == 'OUT':
            confidence_adjustment = 1.0  # Fully confident they won't play
        
        return {
            'injury_status': status,
            'baseline_utilization': round(baseline_utilization, 1),
            'adjusted_utilization': round(adjusted_util, 1),
            'injury_multiplier': round(final_mult, 3),
            'confidence_adjustment': confidence_adjustment,
            'warning': status not in ['HEALTHY', 'PROBABLE']
        }
    
    def get_injury_history(self, player_id: str, last_n_weeks: int = 16) -> pd.DataFrame:
        """
        Get recent injury history for a player.
        Useful for identifying injury-prone players.
        """
        if self.injuries.empty:
            return pd.DataFrame()
        
        history = self.injuries[
            self.injuries['gsis_id'] == player_id
        ].sort_values(['season', 'week'], ascending=False).head(last_n_weeks)
        
        return history[['season', 'week', 'report_status', 'report_primary_injury', 'report_secondary_injury']]
    
    def calculate_injury_risk_score(self, player_id: str) -> Dict[str, any]:
        """
        Calculate injury risk score (0-100, higher = more risky).
        Based on frequency and severity of past injuries.
        """
        history = self.get_injury_history(player_id, last_n_weeks=52)  # Last ~3 seasons
        
        if len(history) == 0:
            return {
                'risk_score': 0,
                'risk_level': 'LOW',
                'injuries_last_year': 0,
                'avg_severity': 0
            }
        
        # Count injuries by severity
        injury_counts = history['report_status'].value_counts()
        
        # Weight by severity
        severity_weights = {'OUT': 5, 'DOUBTFUL': 3, 'QUESTIONABLE': 2, 'PROBABLE': 1}
        
        weighted_score = sum(
            injury_counts.get(status, 0) * weight
            for status, weight in severity_weights.items()
        )
        
        # Normalize to 0-100
        risk_score = min(100, weighted_score * 2)
        
        # Classify risk level
        if risk_score >= 70:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': round(risk_score, 1),
            'risk_level': risk_level,
            'injuries_last_year': len(history[history['season'] == history['season'].max()]),
            'total_injury_reports': len(history)
        }
    
    def batch_adjust_predictions(
        self, 
        predictions: pd.DataFrame,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Adjust an entire batch of predictions for injuries.
        
        Args:
            predictions: DataFrame with columns ['player_id', 'position', 'util_1w', ...]
            season: Season year
            week: Week number
        
        Returns:
            DataFrame with injury-adjusted predictions
        """
        adjusted = predictions.copy()
        
        # Add injury columns
        adjusted['injury_status'] = 'HEALTHY'
        adjusted['injury_adjusted_util'] = adjusted['util_1w']
        adjusted['injury_warning'] = False
        
        for idx, row in adjusted.iterrows():
            adjustment = self.calculate_injury_adjustment(
                row['player_id'],
                row['position'],
                season,
                week,
                row['util_1w']
            )
            
            adjusted.at[idx, 'injury_status'] = adjustment['injury_status']
            adjusted.at[idx, 'injury_adjusted_util'] = adjustment['adjusted_utilization']
            adjusted.at[idx, 'injury_warning'] = adjustment['warning']
        
        return adjusted


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    model = InjuryImpactModel()
    
    # Example: Check injury risk for a player
    # player_id = "00-0036355"  # Example ID
    # risk = model.calculate_injury_risk_score(player_id)
    # print(f"Injury Risk: {risk}")
    
    # Example: Adjust prediction
    # adjustment = model.calculate_injury_adjustment(
    #     player_id="00-0036355",
    #     position="RB",
    #     season=2024,
    #     week=10,
    #     baseline_utilization=75.0
    # )
    # print(f"Injury Adjustment: {adjustment}")
    
    print("✅ Injury Impact Model initialized")
    print(f"📊 Injury records loaded: {len(model.injuries):,}")

"""
QB-Specific Feature Engineering

QBs are harder to predict because:
1. Their production depends heavily on game script (blowouts = less passing)
2. Rushing upside varies dramatically between mobile/pocket QBs
3. Interceptions and fumbles are high-variance events
4. Offensive line quality affects everything

This module adds QB-specific features to improve prediction accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QBFeatureEngineer:
    """
    Engineer QB-specific features for improved prediction.
    
    Key features:
    - Rushing upside (rushing attempts, yards per carry)
    - Efficiency metrics (completion %, yards per attempt)
    - Turnover tendency (INT rate, fumble rate)
    - Game script indicators (pass/run ratio trends)
    - Pressure/sack metrics (if available)
    """
    
    def __init__(self):
        pass
    
    def engineer_qb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add QB-specific features to the dataframe."""
        result = df.copy()
        
        # Only process QBs
        qb_mask = result['position'] == 'QB'
        
        if qb_mask.sum() == 0:
            return result
        
        # Sort for proper lag/rolling calculations
        result = result.sort_values(['player_id', 'season', 'week'])
        
        # ===== Efficiency Metrics =====
        # Completion percentage
        result.loc[qb_mask, 'completion_pct'] = np.where(
            result.loc[qb_mask, 'passing_attempts'] > 0,
            result.loc[qb_mask, 'passing_completions'] / result.loc[qb_mask, 'passing_attempts'],
            0
        )
        
        # Yards per attempt
        result.loc[qb_mask, 'yards_per_attempt'] = np.where(
            result.loc[qb_mask, 'passing_attempts'] > 0,
            result.loc[qb_mask, 'passing_yards'] / result.loc[qb_mask, 'passing_attempts'],
            0
        )
        
        # TD rate (TDs per attempt)
        result.loc[qb_mask, 'td_rate'] = np.where(
            result.loc[qb_mask, 'passing_attempts'] > 0,
            result.loc[qb_mask, 'passing_tds'] / result.loc[qb_mask, 'passing_attempts'],
            0
        )
        
        # INT rate (INTs per attempt)
        result.loc[qb_mask, 'int_rate'] = np.where(
            result.loc[qb_mask, 'passing_attempts'] > 0,
            result.loc[qb_mask, 'interceptions'] / result.loc[qb_mask, 'passing_attempts'],
            0
        )
        
        # ===== Rushing Upside =====
        # Rushing yards per carry
        result.loc[qb_mask, 'rush_yards_per_carry'] = np.where(
            result.loc[qb_mask, 'rushing_attempts'] > 0,
            result.loc[qb_mask, 'rushing_yards'] / result.loc[qb_mask, 'rushing_attempts'],
            0
        )
        
        # Rushing attempt rate (rushing attempts as % of total plays)
        total_plays = result['passing_attempts'] + result['rushing_attempts']
        result.loc[qb_mask, 'rush_attempt_rate'] = np.where(
            total_plays[qb_mask] > 0,
            result.loc[qb_mask, 'rushing_attempts'] / total_plays[qb_mask],
            0
        )
        
        # Mobile QB indicator (high rushing attempts)
        result.loc[qb_mask, 'is_mobile_qb'] = (result.loc[qb_mask, 'rushing_attempts'] >= 4).astype(int)
        
        # Rushing TD rate
        result.loc[qb_mask, 'rush_td_rate'] = np.where(
            result.loc[qb_mask, 'rushing_attempts'] > 0,
            result.loc[qb_mask, 'rushing_tds'] / result.loc[qb_mask, 'rushing_attempts'],
            0
        )
        
        # ===== Volume Metrics =====
        # Total plays (passing + rushing)
        result.loc[qb_mask, 'total_plays'] = (
            result.loc[qb_mask, 'passing_attempts'] + 
            result.loc[qb_mask, 'rushing_attempts']
        )
        
        # Total yards
        result.loc[qb_mask, 'total_yards'] = (
            result.loc[qb_mask, 'passing_yards'] + 
            result.loc[qb_mask, 'rushing_yards']
        )
        
        # Total TDs
        result.loc[qb_mask, 'total_tds'] = (
            result.loc[qb_mask, 'passing_tds'] + 
            result.loc[qb_mask, 'rushing_tds']
        )
        
        # ===== Turnover Metrics =====
        # Turnover rate (INTs + fumbles lost per play)
        turnovers = result['interceptions'] + result['fumbles_lost']
        result.loc[qb_mask, 'turnover_rate'] = np.where(
            total_plays[qb_mask] > 0,
            turnovers[qb_mask] / total_plays[qb_mask],
            0
        )
        
        # ===== Passer Rating (simplified) =====
        # NFL Passer Rating formula (simplified)
        result.loc[qb_mask, 'passer_rating'] = self._calculate_passer_rating(
            result.loc[qb_mask, 'passing_completions'],
            result.loc[qb_mask, 'passing_attempts'],
            result.loc[qb_mask, 'passing_yards'],
            result.loc[qb_mask, 'passing_tds'],
            result.loc[qb_mask, 'interceptions']
        )
        
        # ===== Rolling/Lag Features for QB-specific metrics =====
        qb_df = result[qb_mask].copy()
        
        for col in ['completion_pct', 'yards_per_attempt', 'td_rate', 'int_rate',
                    'rush_attempt_rate', 'passer_rating', 'turnover_rate', 'total_plays']:
            if col in qb_df.columns:
                # Rolling 3-game average (shifted to avoid leakage)
                qb_df[f'{col}_rolling_3'] = qb_df.groupby('player_id')[col].transform(
                    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
                )
                
                # Lag 1 (previous game)
                qb_df[f'{col}_lag_1'] = qb_df.groupby('player_id')[col].shift(1)
        
        # Merge back
        for col in qb_df.columns:
            if col not in result.columns:
                result[col] = np.nan
            result.loc[qb_mask, col] = qb_df[col].values
        
        return result
    
    def _calculate_passer_rating(self, completions: pd.Series, attempts: pd.Series,
                                  yards: pd.Series, tds: pd.Series, 
                                  ints: pd.Series) -> pd.Series:
        """
        Calculate NFL Passer Rating.
        
        Formula:
        a = ((Comp/Att) - 0.3) * 5
        b = ((Yards/Att) - 3) * 0.25
        c = (TD/Att) * 20
        d = 2.375 - ((Int/Att) * 25)
        
        Rating = ((a + b + c + d) / 6) * 100
        
        Each component is capped between 0 and 2.375
        """
        # Avoid division by zero
        att = attempts.replace(0, 1)
        
        a = ((completions / att) - 0.3) * 5
        b = ((yards / att) - 3) * 0.25
        c = (tds / att) * 20
        d = 2.375 - ((ints / att) * 25)
        
        # Cap each component
        a = a.clip(0, 2.375)
        b = b.clip(0, 2.375)
        c = c.clip(0, 2.375)
        d = d.clip(0, 2.375)
        
        rating = ((a + b + c + d) / 6) * 100
        
        # Handle cases where attempts = 0
        rating = rating.where(attempts > 0, 0)
        
        return rating


def add_qb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add QB features."""
    engineer = QBFeatureEngineer()
    return engineer.engineer_qb_features(df)


# Additional QB-specific allowed features for training
QB_FEATURE_PATTERNS = [
    'completion_pct',
    'yards_per_attempt',
    'td_rate',
    'int_rate',
    'rush_yards_per_carry',
    'rush_attempt_rate',
    'is_mobile_qb',
    'rush_td_rate',
    'total_plays',
    'total_yards',
    'total_tds',
    'turnover_rate',
    'passer_rating',
]

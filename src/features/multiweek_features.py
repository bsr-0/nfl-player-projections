"""
Multi-Week Prediction Features

Advanced features for multi-week (5-week, 18-week) prediction horizons:
1. Schedule Strength Analysis - Difficulty of upcoming opponents
2. Multi-Week Aggregation - Proper modeling of multi-week totals (not just weekly * N)
3. Injury Probability Modeling - Risk of missing games based on position/usage

These features are designed to work across all prediction horizons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# SCHEDULE STRENGTH ANALYSIS
# =============================================================================

class ScheduleStrengthAnalyzer:
    """
    Analyze strength of schedule for upcoming games.
    
    Calculates:
    - Defense strength of upcoming opponents
    - Aggregate schedule difficulty for N-week windows
    - Favorable/unfavorable matchup streaks
    """
    
    def __init__(self):
        self.defense_rankings = {}
    
    def calculate_defense_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defense strength for each team by position.
        
        Defense strength = average fantasy points allowed to position
        Higher = weaker defense (allows more points)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Calculate points allowed by each defense to each position
        defense_pts = df.groupby(['opponent', 'season', 'position']).agg({
            'fantasy_points': ['mean', 'std', 'count']
        }).reset_index()
        
        defense_pts.columns = ['team', 'season', 'position', 
                               'pts_allowed_mean', 'pts_allowed_std', 'games']
        
        # Only use teams with enough games
        defense_pts = defense_pts[defense_pts['games'] >= 4]
        
        # Calculate league average by position
        league_avg = df.groupby(['season', 'position'])['fantasy_points'].mean().reset_index()
        league_avg.columns = ['season', 'position', 'league_avg']
        
        defense_pts = defense_pts.merge(league_avg, on=['season', 'position'], how='left')
        
        # Defense strength score: pts allowed / league avg
        # > 1 = weak defense (allows more than avg)
        # < 1 = strong defense (allows less than avg)
        defense_pts['defense_strength'] = (
            defense_pts['pts_allowed_mean'] / defense_pts['league_avg']
        ).clip(0.5, 1.5)
        
        # Rank defenses (1 = weakest = allows most points)
        defense_pts['defense_rank'] = defense_pts.groupby(
            ['season', 'position']
        )['pts_allowed_mean'].rank(ascending=False)
        
        return defense_pts
    
    def get_schedule_strength(self, df: pd.DataFrame, 
                              n_weeks: int = 5) -> pd.DataFrame:
        """
        Calculate schedule strength for next N weeks.
        
        For each player-week, calculates:
        - sos_next_N: Average defense strength of next N opponents
        - sos_rank_next_N: Rank of schedule difficulty
        - favorable_matchups_next_N: Count of favorable matchups
        """
        if df.empty:
            return df
        
        print(f"Calculating schedule strength for {n_weeks}-week windows...")
        
        result = df.copy()
        
        # Get defense strength
        defense_strength = self.calculate_defense_strength(df)
        
        if defense_strength.empty:
            result[f'sos_next_{n_weeks}'] = 1.0
            result[f'sos_rank_next_{n_weeks}'] = 16
            result[f'favorable_matchups_next_{n_weeks}'] = n_weeks // 2
            return result
        
        # Create lookup for defense strength
        defense_lookup = defense_strength.set_index(
            ['team', 'season', 'position']
        )['defense_strength'].to_dict()
        
        # For each player, calculate upcoming schedule strength
        # This requires knowing the schedule - we'll estimate from historical patterns
        
        # Group by player and calculate rolling opponent strength
        result = result.sort_values(['player_id', 'season', 'week'])
        
        # Get opponent defense strength for each game
        result['opp_defense_strength'] = result.apply(
            lambda row: defense_lookup.get(
                (row['opponent'], row['season'], row['position']), 1.0
            ) if pd.notna(row.get('opponent')) else 1.0,
            axis=1
        )
        
        # Calculate forward-looking schedule strength (next N weeks)
        # We use a forward rolling window
        result[f'sos_next_{n_weeks}'] = result.groupby(
            ['player_id', 'season']
        )['opp_defense_strength'].transform(
            lambda x: x.shift(-1).rolling(n_weeks, min_periods=1).mean()
        )
        
        # Fill NaN with average
        result[f'sos_next_{n_weeks}'] = result[f'sos_next_{n_weeks}'].fillna(1.0)
        
        # Count favorable matchups (defense strength > 1.1)
        result['is_favorable'] = (result['opp_defense_strength'] > 1.1).astype(int)
        result[f'favorable_matchups_next_{n_weeks}'] = result.groupby(
            ['player_id', 'season']
        )['is_favorable'].transform(
            lambda x: x.shift(-1).rolling(n_weeks, min_periods=1).sum()
        ).fillna(n_weeks // 2)
        
        # Rank schedule strength within position
        result[f'sos_rank_next_{n_weeks}'] = result.groupby(
            ['season', 'week', 'position']
        )[f'sos_next_{n_weeks}'].rank(ascending=False)
        
        # Clean up
        result = result.drop(columns=['is_favorable'], errors='ignore')
        
        print(f"  Added schedule strength features for {n_weeks}-week horizon")
        
        return result


# =============================================================================
# MULTI-WEEK AGGREGATION MODEL
# =============================================================================

class MultiWeekAggregator:
    """
    Proper multi-week aggregation that accounts for:
    - Variance reduction over multiple weeks
    - Regression to mean over time
    - Games played probability
    - Bye weeks
    
    Instead of just multiplying weekly projection by N weeks,
    this models the actual distribution of multi-week totals.
    """
    
    # Historical games played rates by position
    GAMES_PLAYED_RATES = {
        'QB': 0.92,   # QBs play ~15.6 games on average
        'RB': 0.82,   # RBs play ~14 games (higher injury rate)
        'WR': 0.88,   # WRs play ~15 games
        'TE': 0.90,   # TEs play ~15.3 games
    }
    
    # Regression to mean factor (how much to regress over time)
    REGRESSION_FACTORS = {
        1: 0.0,    # No regression for 1 week
        5: 0.15,   # 15% regression for 5 weeks
        10: 0.25,  # 25% regression for 10 weeks
        18: 0.35,  # 35% regression for full season
    }
    
    def __init__(self):
        pass
    
    def calculate_expected_games(self, df: pd.DataFrame, 
                                  n_weeks: int) -> pd.DataFrame:
        """
        Calculate expected games played over N weeks.
        
        Based on:
        - Position-specific injury rates
        - Player's historical games played rate
        - Current injury status
        """
        result = df.copy()
        
        # Get player's historical games played rate
        player_games = df.groupby(['player_id', 'season']).size().reset_index(name='games_in_season')
        player_games['games_rate'] = player_games['games_in_season'] / 17  # 17-game season
        
        # Merge back
        result = result.merge(
            player_games[['player_id', 'season', 'games_rate']],
            on=['player_id', 'season'],
            how='left',
            suffixes=('', '_calc')
        )
        
        # Handle column naming from merge
        if 'games_rate_calc' in result.columns:
            result['games_rate'] = result['games_rate_calc']
            result = result.drop(columns=['games_rate_calc'])
        
        # Use position default if no history
        result['games_rate'] = result['games_rate'].fillna(
            result['position'].map(self.GAMES_PLAYED_RATES)
        ).fillna(0.85)
        
        # Adjust for current injury status if available
        if 'injury_score' in result.columns:
            result['games_rate'] = result['games_rate'] * result['injury_score']
        
        # Expected games in next N weeks
        result[f'expected_games_next_{n_weeks}'] = (
            result['games_rate'] * n_weeks
        ).clip(0, n_weeks)
        
        return result
    
    def calculate_multiweek_projection(self, df: pd.DataFrame,
                                        n_weeks: int,
                                        weekly_col: str = 'fp_rolling_3') -> pd.DataFrame:
        """
        Calculate proper multi-week projection.
        
        Multi-week total = weekly_avg * expected_games * regression_factor
        
        Also calculates:
        - Multi-week floor/ceiling
        - Variance of multi-week total
        """
        result = df.copy()
        
        # Get weekly projection
        if weekly_col not in result.columns:
            weekly_col = 'fantasy_points'
        
        result['weekly_projection'] = result[weekly_col].fillna(
            result['fantasy_points']
        )
        
        # Calculate expected games
        result = self.calculate_expected_games(result, n_weeks)
        
        # Get regression factor
        regression = self.REGRESSION_FACTORS.get(
            n_weeks, 
            0.1 * np.log(n_weeks + 1)  # Log-based for other horizons
        )
        
        # Calculate position average for regression target
        pos_avg = result.groupby('position')['weekly_projection'].transform('mean')
        
        # Regressed weekly projection
        result['weekly_regressed'] = (
            result['weekly_projection'] * (1 - regression) +
            pos_avg * regression
        )
        
        # Multi-week projection
        result[f'projection_{n_weeks}w'] = (
            result['weekly_regressed'] * result[f'expected_games_next_{n_weeks}']
        )
        
        # Calculate variance for multi-week total
        # Variance of sum = sum of variances (assuming independence)
        # But weeks aren't fully independent, so we use a correlation factor
        correlation_factor = 0.3  # Moderate week-to-week correlation
        
        if 'weekly_volatility' in result.columns:
            weekly_var = result['weekly_volatility'] ** 2
        else:
            weekly_var = result.groupby('position')['fantasy_points'].transform('std') ** 2
        
        # Variance of N-week sum with correlation
        n_games = result[f'expected_games_next_{n_weeks}']
        result[f'variance_{n_weeks}w'] = (
            weekly_var * n_games * (1 + (n_games - 1) * correlation_factor)
        )
        result[f'std_{n_weeks}w'] = np.sqrt(result[f'variance_{n_weeks}w'])
        
        # Floor and ceiling (10th and 90th percentile)
        result[f'floor_{n_weeks}w'] = (
            result[f'projection_{n_weeks}w'] - 1.28 * result[f'std_{n_weeks}w']
        ).clip(0)
        result[f'ceiling_{n_weeks}w'] = (
            result[f'projection_{n_weeks}w'] + 1.28 * result[f'std_{n_weeks}w']
        )
        
        print(f"  Added multi-week projections for {n_weeks}-week horizon")
        
        return result


# =============================================================================
# INJURY PROBABILITY MODELING
# =============================================================================

class InjuryProbabilityModel:
    """
    Model probability of injury/missed games based on:
    - Position (RBs have highest injury rate)
    - Usage/workload (high-touch players more at risk)
    - Age (older players more injury-prone)
    - Injury history
    - Recovery trajectory modeling
    
    Returns probability of missing games over N-week horizon.
    """
    
    # Base injury rates by position (probability of missing at least 1 game per season)
    BASE_INJURY_RATES = {
        'QB': 0.25,   # 25% chance of missing a game
        'RB': 0.45,   # 45% - highest injury rate
        'WR': 0.30,   # 30%
        'TE': 0.28,   # 28%
    }
    
    # Usage impact on injury risk (per touch above average)
    USAGE_INJURY_FACTOR = 0.002  # 0.2% increase per touch above avg
    
    # Recovery trajectory by injury type
    # Values represent % of baseline performance in weeks 1, 2, 3, 4+ after return
    RECOVERY_TRAJECTORIES = {
        'hamstring': {
            'weeks_out_typical': 2,
            'performance_on_return': [0.75, 0.85, 0.92, 1.0],
            'reinjury_risk': 0.25,
        },
        'ankle': {
            'weeks_out_typical': 1,
            'performance_on_return': [0.85, 0.92, 0.98, 1.0],
            'reinjury_risk': 0.15,
        },
        'knee': {
            'weeks_out_typical': 3,
            'performance_on_return': [0.70, 0.80, 0.88, 0.95],
            'reinjury_risk': 0.20,
        },
        'concussion': {
            'weeks_out_typical': 1,
            'performance_on_return': [0.90, 0.95, 1.0, 1.0],
            'reinjury_risk': 0.10,
        },
        'shoulder': {
            'weeks_out_typical': 2,
            'performance_on_return': [0.80, 0.90, 0.95, 1.0],
            'reinjury_risk': 0.15,
        },
        'back': {
            'weeks_out_typical': 2,
            'performance_on_return': [0.75, 0.85, 0.90, 0.95],
            'reinjury_risk': 0.30,
        },
        'acl': {
            'weeks_out_typical': 40,  # Season-ending
            'performance_on_return': [0.60, 0.70, 0.80, 0.90],
            'reinjury_risk': 0.15,
        },
        'unknown': {
            'weeks_out_typical': 2,
            'performance_on_return': [0.80, 0.90, 0.95, 1.0],
            'reinjury_risk': 0.20,
        },
    }
    
    def __init__(self):
        pass
    
    def get_recovery_trajectory(self, injury_type: str, weeks_since_return: int = 0) -> Dict:
        """
        Get expected recovery trajectory for an injury type.
        
        Args:
            injury_type: Type of injury (hamstring, ankle, knee, etc.)
            weeks_since_return: Weeks since player returned to action
            
        Returns:
            Dict with recovery information
        """
        # Normalize injury type
        injury_key = injury_type.lower().strip() if injury_type else 'unknown'
        
        # Find matching trajectory
        trajectory = None
        for key in self.RECOVERY_TRAJECTORIES:
            if key in injury_key:
                trajectory = self.RECOVERY_TRAJECTORIES[key]
                break
        
        if trajectory is None:
            trajectory = self.RECOVERY_TRAJECTORIES['unknown']
        
        # Get performance multiplier based on weeks since return
        perf_curve = trajectory['performance_on_return']
        if weeks_since_return < 0:
            # Still injured
            performance_pct = 0.0
        elif weeks_since_return < len(perf_curve):
            performance_pct = perf_curve[weeks_since_return]
        else:
            # Fully recovered
            performance_pct = perf_curve[-1]
        
        return {
            'injury_type': injury_key,
            'weeks_out_typical': trajectory['weeks_out_typical'],
            'performance_pct': performance_pct,
            'reinjury_risk': trajectory['reinjury_risk'],
            'weeks_since_return': weeks_since_return,
            'fully_recovered': performance_pct >= 0.98,
        }
    
    def add_recovery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add recovery trajectory features to player DataFrame.
        
        For players returning from injury, adds:
        - recovery_performance_pct: Expected % of baseline performance
        - recovery_weeks_remaining: Estimated weeks to full recovery
        - reinjury_risk: Probability of reinjury
        
        Args:
            df: Player DataFrame with injury information
            
        Returns:
            DataFrame with recovery features added
        """
        result = df.copy()
        
        # Default values
        result['recovery_performance_pct'] = 1.0
        result['recovery_weeks_remaining'] = 0
        result['reinjury_risk'] = 0.0
        result['is_returning_from_injury'] = False
        
        # Check if we have injury data
        has_injury_type = 'injury_type' in result.columns
        has_weeks_missed = 'weeks_missed' in result.columns or 'games_missed' in result.columns
        has_return_status = 'returning_from_injury' in result.columns
        
        if not (has_injury_type or has_return_status):
            return result
        
        # Process each player
        for idx, row in result.iterrows():
            injury_type = row.get('injury_type', 'unknown')
            
            # Determine if player is returning from injury
            is_returning = False
            weeks_since_return = 0
            
            if has_return_status and row.get('returning_from_injury', False):
                is_returning = True
                weeks_since_return = row.get('weeks_since_return', 0)
            elif has_injury_type and pd.notna(injury_type) and injury_type not in ['', 'None', 'healthy']:
                # Check if current status indicates returning
                status = str(row.get('injury_status', row.get('status', ''))).upper()
                if status in ['PROBABLE', 'QUESTIONABLE']:
                    is_returning = True
                    weeks_since_return = 0
            
            if is_returning:
                recovery_info = self.get_recovery_trajectory(
                    injury_type=injury_type,
                    weeks_since_return=weeks_since_return
                )
                
                result.at[idx, 'recovery_performance_pct'] = recovery_info['performance_pct']
                result.at[idx, 'reinjury_risk'] = recovery_info['reinjury_risk']
                result.at[idx, 'is_returning_from_injury'] = True
                
                # Estimate weeks to full recovery
                perf_pct = recovery_info['performance_pct']
                if perf_pct < 1.0:
                    # Roughly estimate weeks remaining
                    result.at[idx, 'recovery_weeks_remaining'] = max(0, int((1.0 - perf_pct) * 4))
        
        return result
    
    def discount_prediction_for_recovery(
        self, 
        predicted_points: float,
        recovery_performance_pct: float,
        reinjury_risk: float = 0.0
    ) -> float:
        """
        Discount a prediction based on recovery status.
        
        Args:
            predicted_points: Base prediction
            recovery_performance_pct: Expected % of baseline (0-1)
            reinjury_risk: Probability of reinjury
            
        Returns:
            Adjusted prediction
        """
        # Apply performance discount
        adjusted = predicted_points * recovery_performance_pct
        
        # Apply reinjury risk discount (expected value reduction)
        adjusted *= (1 - reinjury_risk * 0.5)  # Assume 50% of points lost if reinjured
        
        return adjusted
    
    def add_injury_history_features(
        self, 
        df: pd.DataFrame, 
        historical_injuries: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Add injury history features to enhance injury probability modeling.
        
        Features added:
        - injury_history_count: Total prior injuries in career
        - injury_history_2yr: Injuries in last 2 seasons
        - same_bodypart_injuries: Count of injuries to same body part
        - days_since_last_injury: Days since last injury event
        - injury_prone_flag: True if 3+ injuries in 2 years
        - avg_games_missed: Average games missed per injury
        
        Args:
            df: Player DataFrame with player_id column
            historical_injuries: Historical injury data (optional, will estimate if not provided)
            
        Returns:
            DataFrame with injury history features added
        """
        result = df.copy()
        
        # Initialize default values
        result['injury_history_count'] = 0
        result['injury_history_2yr'] = 0
        result['same_bodypart_injuries'] = 0
        result['days_since_last_injury'] = 365  # Default: 1 year (healthy)
        result['injury_prone_flag'] = False
        result['avg_games_missed'] = 0.0
        result['injury_history_risk_multiplier'] = 1.0
        
        if historical_injuries is None or historical_injuries.empty:
            # Try to estimate from current data if possible
            return self._estimate_injury_history_from_games(result)
        
        # Get current season for recency calculations
        if 'season' in result.columns:
            current_season = result['season'].max()
        else:
            from src.utils.nfl_calendar import get_current_nfl_season
            current_season = get_current_nfl_season()
        
        # Process historical injuries for each player
        player_id_col = 'player_id' if 'player_id' in result.columns else None
        name_col = 'name' if 'name' in result.columns else 'player_name' if 'player_name' in result.columns else None
        
        if player_id_col is None and name_col is None:
            return result
        
        # Build injury history by player
        for idx, row in result.iterrows():
            player_key = row.get(player_id_col) if player_id_col else row.get(name_col)
            if pd.isna(player_key):
                continue
            
            # Find player's injury history
            hist_col = 'player_id' if 'player_id' in historical_injuries.columns else 'player_name'
            player_injuries = historical_injuries[
                historical_injuries[hist_col].astype(str).str.contains(str(player_key), case=False, na=False)
            ]
            
            if player_injuries.empty:
                continue
            
            # Count total injuries
            total_injuries = len(player_injuries)
            result.at[idx, 'injury_history_count'] = total_injuries
            
            # Count recent injuries (last 2 seasons)
            if 'season' in player_injuries.columns:
                recent = player_injuries[player_injuries['season'] >= current_season - 2]
                result.at[idx, 'injury_history_2yr'] = len(recent)
            
            # Count same body part injuries
            current_injury_type = str(row.get('injury_type', '')).lower()
            if current_injury_type and 'injury_type' in player_injuries.columns:
                same_type = player_injuries[
                    player_injuries['injury_type'].astype(str).str.lower().str.contains(
                        current_injury_type[:4], na=False  # Match first 4 chars
                    )
                ]
                result.at[idx, 'same_bodypart_injuries'] = len(same_type)
            
            # Days since last injury (estimate from season/week)
            if 'season' in player_injuries.columns and 'week' in player_injuries.columns:
                latest = player_injuries.sort_values(['season', 'week'], ascending=False).iloc[0]
                latest_season = latest['season']
                latest_week = latest.get('week', 1)
                current_week = row.get('week', 1)
                
                # Rough estimate: 7 days per week, 17 weeks per season
                season_diff = current_season - latest_season
                week_diff = current_week - latest_week if season_diff == 0 else current_week + (17 - latest_week)
                days_approx = season_diff * 365 + week_diff * 7
                result.at[idx, 'days_since_last_injury'] = max(0, days_approx)
            
            # Average games missed
            if 'games_missed' in player_injuries.columns:
                avg_missed = player_injuries['games_missed'].mean()
                result.at[idx, 'avg_games_missed'] = avg_missed if pd.notna(avg_missed) else 2.0
            
            # Injury prone flag
            injuries_2yr = result.at[idx, 'injury_history_2yr']
            result.at[idx, 'injury_prone_flag'] = injuries_2yr >= 3
        
        # Calculate injury history risk multiplier
        result['injury_history_risk_multiplier'] = self._calculate_history_risk_multiplier(result)
        
        return result
    
    def _estimate_injury_history_from_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate injury history from games played when historical data unavailable.
        
        Players with fewer games played than expected are assumed to have injury history.
        """
        result = df.copy()
        
        if 'games_played' not in result.columns:
            return result
        
        # Expected games by position (from healthy seasons)
        expected_games = {'QB': 16, 'RB': 14, 'WR': 15, 'TE': 15}
        
        for idx, row in result.iterrows():
            position = row.get('position', 'WR')
            games = row.get('games_played', 16)
            expected = expected_games.get(position, 15)
            
            # Estimate missed games
            missed = max(0, expected - games)
            
            if missed >= 4:
                result.at[idx, 'injury_history_count'] = 1
                result.at[idx, 'injury_prone_flag'] = missed >= 8
                result.at[idx, 'avg_games_missed'] = missed
        
        # Recalculate risk multiplier
        result['injury_history_risk_multiplier'] = self._calculate_history_risk_multiplier(result)
        
        return result
    
    def _calculate_history_risk_multiplier(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate injury risk multiplier based on history.
        
        Formula:
        - Each prior injury adds 10% risk
        - Same body part adds 30% additional risk
        - Injury prone flag adds 20% risk
        - Recent injury (< 60 days) adds 25% risk
        """
        multiplier = pd.Series(1.0, index=df.index)
        
        # Prior injury count effect
        if 'injury_history_count' in df.columns:
            multiplier += df['injury_history_count'] * 0.10
        
        # Same body part effect
        if 'same_bodypart_injuries' in df.columns:
            multiplier += df['same_bodypart_injuries'] * 0.30
        
        # Injury prone effect
        if 'injury_prone_flag' in df.columns:
            multiplier += df['injury_prone_flag'].astype(float) * 0.20
        
        # Recent injury effect
        if 'days_since_last_injury' in df.columns:
            recent_mask = df['days_since_last_injury'] < 60
            multiplier = multiplier.where(~recent_mask, multiplier + 0.25)
        
        # Cap multiplier at 3.0x
        return multiplier.clip(upper=3.0)
    
    def calculate_injury_probability(self, df: pd.DataFrame,
                                      n_weeks: int) -> pd.DataFrame:
        """
        Calculate probability of missing games over N weeks.
        
        Returns:
        - injury_prob_next_N: Probability of any injury
        - expected_missed_games_N: Expected games missed
        - injury_risk_score: 0-1 risk score
        """
        result = df.copy()
        
        print(f"Calculating injury probability for {n_weeks}-week horizon...")
        
        # Base injury rate by position
        result['base_injury_rate'] = result['position'].map(self.BASE_INJURY_RATES)
        
        # Adjust for usage (touches per game)
        if 'rushing_attempts' in result.columns and 'targets' in result.columns:
            result['touches'] = (
                result['rushing_attempts'].fillna(0) + 
                result['targets'].fillna(0)
            )
            
            # Calculate position average touches
            pos_avg_touches = result.groupby('position')['touches'].transform('mean')
            
            # Usage adjustment
            result['usage_adjustment'] = (
                (result['touches'] - pos_avg_touches) * self.USAGE_INJURY_FACTOR
            ).clip(-0.1, 0.2)
        else:
            result['usage_adjustment'] = 0
        
        # Adjust for current injury status
        if 'is_injured' in result.columns:
            result['current_injury_adjustment'] = result['is_injured'] * 0.3
        else:
            result['current_injury_adjustment'] = 0
        
        # Calculate per-week injury probability
        result['weekly_injury_prob'] = (
            result['base_injury_rate'] / 17 +  # Spread season rate over weeks
            result['usage_adjustment'] / 17 +
            result['current_injury_adjustment']
        ).clip(0, 0.5)
        
        # Probability of at least one injury over N weeks
        # P(at least 1) = 1 - P(none) = 1 - (1-p)^n
        result[f'injury_prob_next_{n_weeks}'] = (
            1 - (1 - result['weekly_injury_prob']) ** n_weeks
        )
        
        # Expected missed games
        # E[missed] = sum of probabilities
        result[f'expected_missed_games_{n_weeks}'] = (
            result['weekly_injury_prob'] * n_weeks
        )
        
        # Injury risk score (0-1, higher = more risk)
        max_prob = result[f'injury_prob_next_{n_weeks}'].max()
        if max_prob > 0:
            result[f'injury_risk_score_{n_weeks}'] = (
                result[f'injury_prob_next_{n_weeks}'] / max_prob
            )
        else:
            result[f'injury_risk_score_{n_weeks}'] = 0
        
        # Clean up intermediate columns
        result = result.drop(columns=[
            'base_injury_rate', 'usage_adjustment', 
            'current_injury_adjustment', 'weekly_injury_prob', 'touches'
        ], errors='ignore')
        
        print(f"  Added injury probability features for {n_weeks}-week horizon")
        
        return result


# =============================================================================
# COMBINED MULTI-WEEK FEATURE ENGINEERING
# =============================================================================

class MultiWeekFeatureEngineer:
    """
    Main class to add all multi-week features.
    
    Combines:
    - Schedule strength analysis
    - Multi-week aggregation
    - Injury probability modeling
    """
    
    def __init__(self):
        self.schedule_analyzer = ScheduleStrengthAnalyzer()
        self.aggregator = MultiWeekAggregator()
        self.injury_model = InjuryProbabilityModel()
    
    def add_multiweek_features(self, df: pd.DataFrame,
                                horizons: List[int] = [1, 5, 18]) -> pd.DataFrame:
        """
        Add all multi-week features for specified horizons.
        
        Args:
            df: Player data DataFrame
            horizons: List of week horizons to calculate (default: 1, 5, 18)
            
        Returns:
            DataFrame with multi-week features added
        """
        if df.empty:
            return df
        
        print("\n" + "="*60)
        print("Adding Multi-Week Features")
        print("="*60)
        
        result = df.copy()
        
        for n_weeks in horizons:
            print(f"\n--- {n_weeks}-Week Horizon ---")
            
            # 1. Schedule strength
            result = self.schedule_analyzer.get_schedule_strength(result, n_weeks)
            
            # 2. Multi-week aggregation
            result = self.aggregator.calculate_multiweek_projection(result, n_weeks)
            
            # 3. Injury probability
            result = self.injury_model.calculate_injury_probability(result, n_weeks)
        
        # Summary of new features
        new_features = [c for c in result.columns if c not in df.columns]
        print(f"\n✅ Added {len(new_features)} multi-week features")
        
        return result


def add_multiweek_features(df: pd.DataFrame, 
                           horizons: List[int] = [1, 5, 18]) -> pd.DataFrame:
    """Convenience function to add all multi-week features."""
    engineer = MultiWeekFeatureEngineer()
    return engineer.add_multiweek_features(df, horizons)

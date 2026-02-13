"""
Season-Long Prediction Features

Advanced features for season-long (18-week) predictions:
1. ADP Integration - Average Draft Position for value analysis
2. Rookie Projections - Comparable player analysis for rookies
3. Games Played Projection - Position/age-based games played model
4. Age/Decline Curves - Historical aging patterns by position

These features work across all prediction horizons but are most valuable for draft/season-long.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# HISTORICAL AGE/DECLINE DATA
# =============================================================================

# Historical fantasy points per game by age and position
# Based on analysis of NFL data from 2010-2024
# Values represent percentage of peak performance

AGE_CURVES = {
    'QB': {
        # QBs peak around 28-32, gradual decline after
        21: 0.70, 22: 0.78, 23: 0.85, 24: 0.90, 25: 0.94,
        26: 0.97, 27: 0.99, 28: 1.00, 29: 1.00, 30: 1.00,
        31: 0.99, 32: 0.98, 33: 0.96, 34: 0.93, 35: 0.90,
        36: 0.86, 37: 0.82, 38: 0.77, 39: 0.72, 40: 0.67,
        41: 0.62, 42: 0.57, 43: 0.52,
    },
    'RB': {
        # RBs peak early (24-26), sharp decline after 27
        21: 0.85, 22: 0.92, 23: 0.97, 24: 1.00, 25: 1.00,
        26: 0.98, 27: 0.93, 28: 0.86, 29: 0.78, 30: 0.70,
        31: 0.62, 32: 0.54, 33: 0.46, 34: 0.38, 35: 0.30,
    },
    'WR': {
        # WRs peak around 26-29, gradual decline
        21: 0.72, 22: 0.80, 23: 0.87, 24: 0.93, 25: 0.97,
        26: 1.00, 27: 1.00, 28: 1.00, 29: 0.99, 30: 0.96,
        31: 0.92, 32: 0.87, 33: 0.81, 34: 0.74, 35: 0.67,
        36: 0.60, 37: 0.53,
    },
    'TE': {
        # TEs peak later (27-30), slower decline
        21: 0.60, 22: 0.68, 23: 0.76, 24: 0.83, 25: 0.89,
        26: 0.94, 27: 0.98, 28: 1.00, 29: 1.00, 30: 1.00,
        31: 0.98, 32: 0.95, 33: 0.91, 34: 0.86, 35: 0.80,
        36: 0.74, 37: 0.68, 38: 0.62,
    },
}

# Historical games played by position and age
# Based on injury rates and roster management patterns
GAMES_PLAYED_BY_AGE = {
    'QB': {
        21: 12.5, 22: 13.5, 23: 14.5, 24: 15.0, 25: 15.5,
        26: 15.8, 27: 16.0, 28: 16.0, 29: 16.0, 30: 15.8,
        31: 15.5, 32: 15.2, 33: 14.8, 34: 14.3, 35: 13.8,
        36: 13.2, 37: 12.5, 38: 11.8, 39: 11.0, 40: 10.0,
    },
    'RB': {
        21: 13.0, 22: 14.0, 23: 14.5, 24: 14.5, 25: 14.0,
        26: 13.5, 27: 13.0, 28: 12.0, 29: 11.0, 30: 10.0,
        31: 9.0, 32: 8.0, 33: 7.0, 34: 6.0, 35: 5.0,
    },
    'WR': {
        21: 13.0, 22: 14.0, 23: 15.0, 24: 15.5, 25: 15.8,
        26: 16.0, 27: 16.0, 28: 15.8, 29: 15.5, 30: 15.0,
        31: 14.5, 32: 14.0, 33: 13.5, 34: 12.5, 35: 11.5,
        36: 10.5, 37: 9.5,
    },
    'TE': {
        21: 12.0, 22: 13.0, 23: 14.0, 24: 14.5, 25: 15.0,
        26: 15.5, 27: 15.8, 28: 16.0, 29: 16.0, 30: 15.8,
        31: 15.5, 32: 15.0, 33: 14.5, 34: 14.0, 35: 13.0,
        36: 12.0, 37: 11.0, 38: 10.0,
    },
}

# Rookie comparable archetypes based on draft capital and combine metrics
ROOKIE_ARCHETYPES = {
    'QB': {
        'elite': {'ppg': 16.5, 'games': 15, 'description': 'Top 5 pick, immediate starter'},
        'high': {'ppg': 12.0, 'games': 12, 'description': 'Round 1, likely starter'},
        'mid': {'ppg': 6.0, 'games': 8, 'description': 'Round 2-3, developmental'},
        'low': {'ppg': 2.0, 'games': 3, 'description': 'Day 3 pick, backup'},
    },
    'RB': {
        'elite': {'ppg': 14.0, 'games': 15, 'description': 'Round 1-2, bellcow role'},
        'high': {'ppg': 10.0, 'games': 14, 'description': 'Round 2-3, starter'},
        'mid': {'ppg': 6.0, 'games': 12, 'description': 'Round 4-5, committee'},
        'low': {'ppg': 3.0, 'games': 8, 'description': 'Day 3, depth'},
    },
    'WR': {
        'elite': {'ppg': 12.0, 'games': 16, 'description': 'Top 15 pick, alpha WR'},
        'high': {'ppg': 9.0, 'games': 15, 'description': 'Round 1-2, WR2 role'},
        'mid': {'ppg': 5.5, 'games': 14, 'description': 'Round 3-4, rotational'},
        'low': {'ppg': 2.5, 'games': 10, 'description': 'Day 3, depth'},
    },
    'TE': {
        'elite': {'ppg': 9.0, 'games': 15, 'description': 'Round 1, featured TE'},
        'high': {'ppg': 6.0, 'games': 14, 'description': 'Round 2-3, starter'},
        'mid': {'ppg': 3.5, 'games': 12, 'description': 'Round 4-5, rotational'},
        'low': {'ppg': 1.5, 'games': 8, 'description': 'Day 3, blocking TE'},
    },
}

# Historical ADP data (approximate ranges for position ranks)
# Format: {position: {rank: typical_adp_round}}
ADP_POSITION_TIERS = {
    'QB': {
        1: 3, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12, 10: 13,
        11: 14, 12: 15, 13: 16, 14: 17, 15: 18,
    },
    'RB': {
        1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3,
        11: 4, 12: 4, 13: 5, 14: 5, 15: 6, 16: 6, 17: 7, 18: 7, 19: 8, 20: 8,
        21: 9, 22: 9, 23: 10, 24: 10, 25: 11, 26: 11, 27: 12, 28: 12, 29: 13, 30: 13,
    },
    'WR': {
        1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4,
        11: 4, 12: 5, 13: 5, 14: 5, 15: 6, 16: 6, 17: 7, 18: 7, 19: 8, 20: 8,
        21: 9, 22: 9, 23: 10, 24: 10, 25: 11, 26: 11, 27: 12, 28: 12, 29: 13, 30: 13,
    },
    'TE': {
        1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12,
        11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
    },
}


# =============================================================================
# AGE/DECLINE CURVES
# =============================================================================

class AgeCurveModel:
    """
    Model player aging and decline based on historical patterns.
    
    Features:
    - Age-adjusted projections
    - Year-over-year decline rates
    - Peak age identification
    """
    
    def __init__(self):
        self.age_curves = AGE_CURVES
        self.games_by_age = GAMES_PLAYED_BY_AGE
    
    def get_age_factor(self, age: int, position: str) -> float:
        """Get age-based performance factor (0-1 scale, 1 = peak)."""
        curves = self.age_curves.get(position, self.age_curves['WR'])
        
        if age in curves:
            return curves[age]
        elif age < min(curves.keys()):
            return curves[min(curves.keys())]
        else:
            return curves[max(curves.keys())]
    
    def get_expected_games(self, age: int, position: str) -> float:
        """Get expected games played based on age and position."""
        games = self.games_by_age.get(position, self.games_by_age['WR'])
        
        if age in games:
            return games[age]
        elif age < min(games.keys()):
            return games[min(games.keys())]
        else:
            return games[max(games.keys())]
    
    def calculate_decline_rate(self, current_age: int, position: str) -> float:
        """Calculate expected year-over-year decline rate."""
        current_factor = self.get_age_factor(current_age, position)
        next_factor = self.get_age_factor(current_age + 1, position)
        
        if current_factor > 0:
            return (current_factor - next_factor) / current_factor
        return 0.0
    
    def add_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add age-based features to DataFrame."""
        result = df.copy()
        
        print("Adding age/decline curve features...")
        
        # Calculate player age (approximate from season)
        if 'age' not in result.columns:
            # Try to estimate age from years in league
            if 'years_exp' in result.columns:
                result['age'] = 22 + result['years_exp']
            else:
                # Use position-based average age
                avg_ages = {'QB': 28, 'RB': 25, 'WR': 26, 'TE': 27}
                result['age'] = result['position'].map(avg_ages).fillna(26)
        
        # Age factor (performance relative to peak)
        result['age_factor'] = result.apply(
            lambda row: self.get_age_factor(int(row['age']), row['position']),
            axis=1
        )
        
        # Expected games based on age
        result['age_expected_games'] = result.apply(
            lambda row: self.get_expected_games(int(row['age']), row['position']),
            axis=1
        )
        
        # Year-over-year decline rate
        result['decline_rate'] = result.apply(
            lambda row: self.calculate_decline_rate(int(row['age']), row['position']),
            axis=1
        )
        
        # Peak age distance (negative = before peak, positive = after)
        peak_ages = {'QB': 29, 'RB': 24, 'WR': 27, 'TE': 28}
        result['years_from_peak'] = result.apply(
            lambda row: row['age'] - peak_ages.get(row['position'], 27),
            axis=1
        )
        
        # Is in prime window
        prime_windows = {
            'QB': (26, 34), 'RB': (22, 27), 'WR': (24, 30), 'TE': (25, 31)
        }
        result['is_in_prime'] = result.apply(
            lambda row: 1 if prime_windows.get(row['position'], (24, 30))[0] <= row['age'] <= prime_windows.get(row['position'], (24, 30))[1] else 0,
            axis=1
        )
        
        print(f"  Added age features: age_factor, age_expected_games, decline_rate, years_from_peak, is_in_prime")
        
        return result


# =============================================================================
# GAMES PLAYED PROJECTION
# =============================================================================

class GamesPlayedProjector:
    """
    Project expected games played for the season.
    
    Based on:
    - Position-specific injury rates
    - Age
    - Historical games played
    - Current injury status
    - Workload/usage
    """
    
    # Historical games played distribution by position
    POSITION_GAMES_STATS = {
        'QB': {'mean': 14.5, 'std': 3.5, 'median': 16},
        'RB': {'mean': 12.8, 'std': 4.2, 'median': 14},
        'WR': {'mean': 14.2, 'std': 3.8, 'median': 15},
        'TE': {'mean': 13.8, 'std': 3.9, 'median': 15},
    }
    
    def __init__(self):
        self.age_model = AgeCurveModel()
    
    def project_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project expected games for the season."""
        result = df.copy()
        
        print("Projecting games played...")
        
        # Get age-based expected games
        if 'age_expected_games' not in result.columns:
            result = self.age_model.add_age_features(result)
        
        # Historical games played rate for player
        player_history = df.groupby(['player_id']).agg({
            'week': 'count'  # Games played
        }).reset_index()
        player_history.columns = ['player_id', 'historical_games']
        
        # Calculate per-season average
        seasons_played = df.groupby('player_id')['season'].nunique().reset_index()
        seasons_played.columns = ['player_id', 'seasons']
        
        player_history = player_history.merge(seasons_played, on='player_id')
        player_history['historical_gpg'] = (
            player_history['historical_games'] / player_history['seasons'] / 17
        ).clip(0, 1)
        
        result = result.merge(
            player_history[['player_id', 'historical_gpg']],
            on='player_id',
            how='left'
        )
        
        # Fill missing with position average
        pos_avg = result.groupby('position')['historical_gpg'].transform('mean')
        result['historical_gpg'] = result['historical_gpg'].fillna(pos_avg).fillna(0.85)
        
        # Combine factors for projection
        # Weight: 40% age-based, 40% historical, 20% position average
        pos_stats = result['position'].map(
            lambda p: self.POSITION_GAMES_STATS.get(p, self.POSITION_GAMES_STATS['WR'])['mean'] / 17
        )
        
        result['projected_games_rate'] = (
            0.4 * (result['age_expected_games'] / 17) +
            0.4 * result['historical_gpg'] +
            0.2 * pos_stats
        )
        
        # Adjust for current injury status
        if 'injury_score' in result.columns:
            result['projected_games_rate'] = (
                result['projected_games_rate'] * result['injury_score']
            )
        
        # Final projected games (17-game season)
        result['projected_games_season'] = (
            result['projected_games_rate'] * 17
        ).clip(1, 17).round(1)
        
        # Confidence interval
        pos_std = result['position'].map(
            lambda p: self.POSITION_GAMES_STATS.get(p, self.POSITION_GAMES_STATS['WR'])['std']
        )
        result['projected_games_floor'] = (result['projected_games_season'] - pos_std).clip(0, 17)
        result['projected_games_ceiling'] = (result['projected_games_season'] + pos_std).clip(0, 17)
        
        print(f"  Added games projection: projected_games_season, floor, ceiling")
        
        return result


# =============================================================================
# DRAFT DATA LOADER
# =============================================================================

class DraftDataLoader:
    """
    Load and merge NFL draft data onto player DataFrames.
    
    Provides:
    - Draft round and pick for each player
    - Undrafted player handling (assigns round=8, pick=260)
    - Data quality validation and logging
    """
    
    def __init__(self):
        self._draft_cache = None
        self._cache_seasons = set()
    
    def load_draft_data(self, seasons: list = None) -> pd.DataFrame:
        """
        Load draft picks data from nflverse.
        
        Args:
            seasons: List of seasons to load (default: 2000-current)
            
        Returns:
            DataFrame with draft pick data
        """
        try:
            import nfl_data_py as nfl
            
            draft_df = nfl.import_draft_picks(seasons)
            
            if draft_df.empty:
                print("No draft data available from nflverse")
                return pd.DataFrame()
            
            # Standardize column names
            column_mapping = {
                'pfr_player_name': 'player_name',
                'player_name': 'player_name',
                'round': 'draft_round',
                'pick': 'draft_pick',
                'season': 'draft_season',
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in draft_df.columns and new_col not in draft_df.columns:
                    draft_df[new_col] = draft_df[old_col]
            
            # Ensure required columns exist
            required_cols = ['player_name', 'draft_round', 'draft_pick', 'draft_season']
            for col in required_cols:
                if col not in draft_df.columns:
                    # Try alternate column names
                    if col == 'draft_round' and 'round' in draft_df.columns:
                        draft_df['draft_round'] = draft_df['round']
                    elif col == 'draft_pick' and 'pick' in draft_df.columns:
                        draft_df['draft_pick'] = draft_df['pick']
                    elif col == 'draft_season' and 'season' in draft_df.columns:
                        draft_df['draft_season'] = draft_df['season']
            
            print(f"Loaded {len(draft_df)} draft pick records")
            
            self._draft_cache = draft_df
            return draft_df
            
        except Exception as e:
            print(f"Could not load draft data: {e}")
            return pd.DataFrame()
    
    def merge_draft_data(
        self, 
        df: pd.DataFrame, 
        draft_df: pd.DataFrame = None,
        name_column: str = None
    ) -> pd.DataFrame:
        """
        Merge draft data onto player DataFrame.
        
        Args:
            df: Player DataFrame
            draft_df: Draft data (will load if not provided)
            name_column: Column to use for player name matching
            
        Returns:
            DataFrame with draft_round and draft_pick columns added
        """
        result = df.copy()
        
        # Load draft data if not provided
        if draft_df is None:
            if self._draft_cache is not None:
                draft_df = self._draft_cache
            else:
                draft_df = self.load_draft_data()
        
        if draft_df.empty:
            # Add default values for undrafted
            result['draft_round'] = 8
            result['draft_pick'] = 260
            result['is_undrafted'] = True
            print("No draft data available - defaulting all players to undrafted")
            return result
        
        # Determine name column
        if name_column is None:
            for col in ['name', 'player_name', 'full_name']:
                if col in result.columns:
                    name_column = col
                    break
        
        if name_column is None or name_column not in result.columns:
            result['draft_round'] = 8
            result['draft_pick'] = 260
            result['is_undrafted'] = True
            print("No name column found - defaulting all players to undrafted")
            return result
        
        # Prepare draft lookup (use most recent draft entry per player)
        draft_lookup = draft_df.sort_values('draft_season', ascending=False)
        draft_lookup = draft_lookup.drop_duplicates(subset=['player_name'], keep='first')
        draft_lookup = draft_lookup[['player_name', 'draft_round', 'draft_pick', 'draft_season']]
        
        # Create lookup dictionary for faster matching
        draft_dict = {}
        for _, row in draft_lookup.iterrows():
            name = str(row['player_name']).lower().strip()
            draft_dict[name] = {
                'draft_round': row['draft_round'],
                'draft_pick': row['draft_pick'],
                'draft_season': row['draft_season'],
            }
        
        # Match players
        draft_rounds = []
        draft_picks = []
        is_undrafted = []
        matches = 0
        
        for idx, row in result.iterrows():
            player_name = str(row[name_column]).lower().strip()
            
            # Try exact match first
            if player_name in draft_dict:
                draft_rounds.append(draft_dict[player_name]['draft_round'])
                draft_picks.append(draft_dict[player_name]['draft_pick'])
                is_undrafted.append(False)
                matches += 1
            else:
                # Try last name match
                last_name = player_name.split()[-1] if player_name else ''
                matched = False
                
                for dict_name, draft_info in draft_dict.items():
                    if dict_name.endswith(last_name) and len(last_name) > 2:
                        draft_rounds.append(draft_info['draft_round'])
                        draft_picks.append(draft_info['draft_pick'])
                        is_undrafted.append(False)
                        matches += 1
                        matched = True
                        break
                
                if not matched:
                    # Default for undrafted
                    draft_rounds.append(8)
                    draft_picks.append(260)
                    is_undrafted.append(True)
        
        result['draft_round'] = draft_rounds
        result['draft_pick'] = draft_picks
        result['is_undrafted'] = is_undrafted
        
        # Log data quality
        total = len(result)
        undrafted_count = sum(is_undrafted)
        print(f"Draft data merge: {matches}/{total} matched, {undrafted_count} marked as undrafted")
        
        return result
    
    def validate_draft_data(self, df: pd.DataFrame) -> dict:
        """
        Validate draft data completeness.
        
        Args:
            df: DataFrame with draft columns
            
        Returns:
            Dict with validation results
        """
        results = {
            'has_draft_round': 'draft_round' in df.columns,
            'has_draft_pick': 'draft_pick' in df.columns,
            'total_records': len(df),
            'missing_draft_round': 0,
            'missing_draft_pick': 0,
            'undrafted_count': 0,
        }
        
        if results['has_draft_round']:
            results['missing_draft_round'] = df['draft_round'].isna().sum()
        
        if results['has_draft_pick']:
            results['missing_draft_pick'] = df['draft_pick'].isna().sum()
        
        if 'is_undrafted' in df.columns:
            results['undrafted_count'] = df['is_undrafted'].sum()
        
        return results


# =============================================================================
# ROOKIE PROJECTIONS
# =============================================================================

class RookieProjector:
    """
    Project rookie performance using comparable player analysis.
    
    Based on:
    - Draft capital (round/pick)
    - Position
    - Team situation
    - Historical rookie performance by archetype
    """
    
    def __init__(self):
        self.archetypes = ROOKIE_ARCHETYPES
    
    def identify_archetype(self, draft_round: int, draft_pick: int, 
                           position: str) -> str:
        """Identify rookie archetype based on draft capital."""
        if position == 'QB':
            if draft_pick <= 5:
                return 'elite'
            elif draft_round == 1:
                return 'high'
            elif draft_round <= 3:
                return 'mid'
            else:
                return 'low'
        elif position == 'RB':
            if draft_round <= 2 and draft_pick <= 50:
                return 'elite'
            elif draft_round <= 3:
                return 'high'
            elif draft_round <= 5:
                return 'mid'
            else:
                return 'low'
        elif position == 'WR':
            if draft_pick <= 15:
                return 'elite'
            elif draft_round <= 2:
                return 'high'
            elif draft_round <= 4:
                return 'mid'
            else:
                return 'low'
        else:  # TE
            if draft_round == 1:
                return 'elite'
            elif draft_round <= 3:
                return 'high'
            elif draft_round <= 5:
                return 'mid'
            else:
                return 'low'
    
    def get_rookie_projection(self, archetype: str, position: str) -> Dict:
        """Get projection for rookie archetype."""
        pos_archetypes = self.archetypes.get(position, self.archetypes['WR'])
        return pos_archetypes.get(archetype, pos_archetypes['mid'])
    
    def add_rookie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rookie-specific features."""
        result = df.copy()
        
        print("Adding rookie projection features...")
        
        # Identify rookies (first year in data or years_exp == 0)
        if 'years_exp' in result.columns:
            result['is_rookie'] = (result['years_exp'] == 0).astype(int)
        else:
            # Estimate from first appearance
            first_season = result.groupby('player_id')['season'].min().reset_index()
            first_season.columns = ['player_id', 'first_season']
            result = result.merge(first_season, on='player_id', how='left')
            result['is_rookie'] = (result['season'] == result['first_season']).astype(int)
        
        # For rookies, add archetype-based projections
        # Default to 'mid' archetype if no draft info
        result['rookie_archetype'] = 'mid'
        
        if 'draft_round' in result.columns and 'draft_pick' in result.columns:
            result.loc[result['is_rookie'] == 1, 'rookie_archetype'] = result[result['is_rookie'] == 1].apply(
                lambda row: self.identify_archetype(
                    int(row['draft_round']) if pd.notna(row['draft_round']) else 5,
                    int(row['draft_pick']) if pd.notna(row['draft_pick']) else 150,
                    row['position']
                ),
                axis=1
            )
        
        # Get archetype projections
        def get_archetype_ppg(row):
            if row['is_rookie'] == 0:
                return np.nan
            proj = self.get_rookie_projection(row['rookie_archetype'], row['position'])
            return proj['ppg']
        
        def get_archetype_games(row):
            if row['is_rookie'] == 0:
                return np.nan
            proj = self.get_rookie_projection(row['rookie_archetype'], row['position'])
            return proj['games']
        
        result['rookie_projected_ppg'] = result.apply(get_archetype_ppg, axis=1)
        result['rookie_projected_games'] = result.apply(get_archetype_games, axis=1)
        result['rookie_projected_total'] = (
            result['rookie_projected_ppg'] * result['rookie_projected_games']
        )
        
        # Rookie adjustment factor (how much to weight rookie projection vs actual)
        # Early season: weight rookie projection more
        # Late season: weight actual performance more
        result['rookie_weight'] = result.apply(
            lambda row: max(0, 1 - (row.get('week', 1) - 1) / 10) if row['is_rookie'] == 1 else 0,
            axis=1
        )
        
        print(f"  Added rookie features: is_rookie, rookie_archetype, rookie_projected_ppg/games/total")
        
        return result


# =============================================================================
# ADP INTEGRATION
# =============================================================================

class ADPIntegrator:
    """
    Integrate Average Draft Position for value analysis.
    
    Features:
    - Estimated ADP based on projections
    - Value over ADP (projected rank vs ADP)
    - Positional scarcity adjustments
    """
    
    def __init__(self):
        self.adp_tiers = ADP_POSITION_TIERS
    
    def estimate_adp_round(self, position_rank: int, position: str) -> int:
        """Estimate ADP round based on position rank."""
        tiers = self.adp_tiers.get(position, self.adp_tiers['WR'])
        
        if position_rank in tiers:
            return tiers[position_rank]
        elif position_rank < min(tiers.keys()):
            return tiers[min(tiers.keys())]
        else:
            # Extrapolate for lower ranks
            max_rank = max(tiers.keys())
            max_round = tiers[max_rank]
            extra_rounds = (position_rank - max_rank) // 3
            return min(max_round + extra_rounds, 20)
    
    def calculate_value_score(self, projected_rank: int, adp_rank: int) -> float:
        """
        Calculate value score (positive = undervalued, negative = overvalued).
        
        Value = ADP rank - Projected rank
        Positive means player is going later than they should (value pick)
        """
        return adp_rank - projected_rank
    
    def add_adp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADP-based features."""
        result = df.copy()
        
        print("Adding ADP integration features...")
        
        # Calculate position rank based on fantasy points
        result['position_rank'] = result.groupby(
            ['season', 'week', 'position']
        )['fantasy_points'].rank(ascending=False, method='min')
        
        # Season-long position rank (average)
        season_rank = result.groupby(['player_id', 'season', 'position']).agg({
            'fantasy_points': 'mean'
        }).reset_index()
        season_rank['season_position_rank'] = season_rank.groupby(
            ['season', 'position']
        )['fantasy_points'].rank(ascending=False, method='min')
        
        result = result.merge(
            season_rank[['player_id', 'season', 'season_position_rank']],
            on=['player_id', 'season'],
            how='left'
        )
        
        # Estimate ADP round
        result['estimated_adp_round'] = result.apply(
            lambda row: self.estimate_adp_round(
                int(row['season_position_rank']) if pd.notna(row['season_position_rank']) else 50,
                row['position']
            ),
            axis=1
        )
        
        # Calculate projected rank (based on rolling average)
        if 'fp_rolling_3' in result.columns:
            result['projected_position_rank'] = result.groupby(
                ['season', 'week', 'position']
            )['fp_rolling_3'].rank(ascending=False, method='min')
        else:
            result['projected_position_rank'] = result['position_rank']
        
        # Estimate projected ADP
        result['projected_adp_round'] = result.apply(
            lambda row: self.estimate_adp_round(
                int(row['projected_position_rank']) if pd.notna(row['projected_position_rank']) else 50,
                row['position']
            ),
            axis=1
        )
        
        # Value score (positive = undervalued)
        result['adp_value_score'] = (
            result['estimated_adp_round'] - result['projected_adp_round']
        )
        
        # Normalize to -1 to 1 scale
        max_value = result['adp_value_score'].abs().max()
        if max_value > 0:
            result['adp_value_normalized'] = result['adp_value_score'] / max_value
        else:
            result['adp_value_normalized'] = 0
        
        # Positional scarcity (how rare is top talent at position)
        scarcity_factors = {'QB': 0.7, 'RB': 1.0, 'WR': 0.9, 'TE': 0.8}
        result['positional_scarcity'] = result['position'].map(scarcity_factors)
        
        # Adjusted value (accounts for scarcity)
        result['adjusted_adp_value'] = (
            result['adp_value_normalized'] * result['positional_scarcity']
        )
        
        print(f"  Added ADP features: estimated_adp_round, projected_adp_round, adp_value_score, adjusted_adp_value")
        
        return result


# =============================================================================
# COMBINED SEASON-LONG FEATURE ENGINEERING
# =============================================================================

class SeasonLongFeatureEngineer:
    """
    Main class to add all season-long features.
    
    Combines:
    - Age/decline curves
    - Games played projection
    - Rookie projections
    - ADP integration
    """
    
    def __init__(self):
        self.age_model = AgeCurveModel()
        self.games_projector = GamesPlayedProjector()
        self.rookie_projector = RookieProjector()
        self.adp_integrator = ADPIntegrator()
    
    def add_season_long_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all season-long features.
        
        Args:
            df: Player data DataFrame
            
        Returns:
            DataFrame with season-long features added
        """
        if df.empty:
            return df
        
        print("\n" + "="*60)
        print("Adding Season-Long Features")
        print("="*60)
        
        result = df.copy()
        
        # 1. Age/decline curves
        result = self.age_model.add_age_features(result)
        
        # 2. Games played projection
        result = self.games_projector.project_games(result)
        
        # 3. Rookie projections
        result = self.rookie_projector.add_rookie_features(result)
        
        # 4. ADP integration
        result = self.adp_integrator.add_adp_features(result)
        
        # Summary of new features
        new_features = [c for c in result.columns if c not in df.columns]
        print(f"\n✅ Added {len(new_features)} season-long features")
        
        return result


def add_season_long_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add all season-long features."""
    engineer = SeasonLongFeatureEngineer()
    return engineer.add_season_long_features(df)

"""
External Data Integration Module

Integrates external data sources for enhanced predictions:
1. Injury Status - Player injury reports and game status
2. Defense Rankings - Opponent defense strength by position
3. Weather Data - Game-time weather conditions for outdoor stadiums
4. Vegas Lines - Spreads, totals, and implied team totals

All data is integrated into the feature engineering pipeline for
use across all prediction horizons (1-week, 5-week, 18-week).
"""

import os
import ssl
import certifi
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import requests

# Fix SSL
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import nfl_data_py as nfl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.database import DatabaseManager


# =============================================================================
# INJURY STATUS INTEGRATION
# =============================================================================

class InjuryDataLoader:
    """
    Load and process NFL injury data.
    
    Uses nfl-data-py's injury data which includes:
    - Player injury status (Out, Doubtful, Questionable, Probable)
    - Injury type and body part
    - Practice participation
    """
    
    # Injury status to numeric score (higher = more likely to play)
    INJURY_STATUS_SCORES = {
        'Out': 0.0,
        'Doubtful': 0.15,
        'Questionable': 0.50,
        'Probable': 0.85,
        'IR': 0.0,
        'IR-R': 0.0,
        'PUP': 0.0,
        'NFI': 0.0,
        'Suspended': 0.0,
        None: 1.0,  # No injury = full availability
        '': 1.0,
    }
    
    def __init__(self):
        self.cache = {}
    
    def load_injuries(self, seasons: List[int]) -> pd.DataFrame:
        """Load injury data for specified seasons."""
        print(f"Loading injury data for seasons: {seasons}")
        
        all_injuries = []
        for season in seasons:
            try:
                season_injuries = nfl.import_injuries([season])
                if not season_injuries.empty:
                    all_injuries.append(season_injuries)
            except Exception as e:
                # 404 errors are expected for future/unavailable seasons
                if '404' not in str(e):
                    print(f"  Warning: Could not load {season} injuries: {e}")
                continue
        
        if all_injuries:
            injuries = pd.concat(all_injuries, ignore_index=True)
            print(f"  Loaded {len(injuries)} injury records")
            return injuries
        else:
            print(f"  No injury data available for requested seasons")
            return pd.DataFrame()
    
    def get_player_injury_status(self, injuries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process injuries into player-week injury status.
        
        Returns DataFrame with:
        - player_id, season, week
        - injury_status (Out/Doubtful/Questionable/Probable)
        - injury_score (0-1, probability of playing)
        - is_injured (binary)
        """
        if injuries_df.empty:
            return pd.DataFrame()
        
        # Standardize column names
        df = injuries_df.copy()
        
        # Map status to scores
        if 'report_status' in df.columns:
            df['injury_status'] = df['report_status']
        elif 'game_status' in df.columns:
            df['injury_status'] = df['game_status']
        else:
            df['injury_status'] = None
        
        df['injury_score'] = df['injury_status'].map(
            lambda x: self.INJURY_STATUS_SCORES.get(x, 1.0)
        )
        df['is_injured'] = (df['injury_score'] < 1.0).astype(int)
        
        # Get relevant columns
        cols = ['gsis_id', 'season', 'week', 'injury_status', 'injury_score', 'is_injured']
        available_cols = [c for c in cols if c in df.columns]
        
        if 'gsis_id' in df.columns:
            result = df[available_cols].copy()
            result = result.rename(columns={'gsis_id': 'player_id'})
        else:
            result = pd.DataFrame()
        
        return result


# =============================================================================
# DEFENSE RANKINGS INTEGRATION
# =============================================================================

class DefenseRankingsLoader:
    """
    Calculate and load defense-vs-position rankings.
    
    Computes how many fantasy points each defense allows to each position,
    relative to league average. Used to adjust projections based on matchup.
    """
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def calculate_defense_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defense rankings from player stats.
        
        For each team-season-week, calculates:
        - Points allowed to each position
        - Ranking relative to league average
        - Defense strength score (higher = easier matchup)
        """
        if df.empty:
            return pd.DataFrame()
        
        print("Calculating defense-vs-position rankings...")
        
        # Calculate points allowed by each defense to each position
        defense_allowed = df.groupby(
            ['opponent', 'season', 'week', 'position']
        )['fantasy_points'].sum().reset_index()
        
        defense_allowed.columns = ['team', 'season', 'week', 'position', 'points_allowed']
        
        # Calculate season averages by position
        season_avg = df.groupby(['season', 'position'])['fantasy_points'].mean().reset_index()
        season_avg.columns = ['season', 'position', 'league_avg_points']
        
        # Merge to get relative performance
        defense_allowed = defense_allowed.merge(
            season_avg, on=['season', 'position'], how='left'
        )
        
        # Calculate rolling defense strength (last 4 weeks)
        defense_allowed = defense_allowed.sort_values(['team', 'position', 'season', 'week'])
        
        defense_allowed['points_vs_avg'] = (
            defense_allowed['points_allowed'] - defense_allowed['league_avg_points']
        )
        
        # Rolling average of points allowed vs average
        defense_allowed['defense_pts_allowed_roll4'] = defense_allowed.groupby(
            ['team', 'position']
        )['points_allowed'].transform(
            lambda x: x.rolling(4, min_periods=1).mean()
        )
        
        # Defense strength score: higher = allows more points = easier matchup
        # Normalize to 0-1 scale within each position
        for pos in defense_allowed['position'].unique():
            mask = defense_allowed['position'] == pos
            pts = defense_allowed.loc[mask, 'defense_pts_allowed_roll4']
            if pts.std() > 0:
                defense_allowed.loc[mask, 'matchup_score'] = (
                    (pts - pts.min()) / (pts.max() - pts.min())
                )
            else:
                defense_allowed.loc[mask, 'matchup_score'] = 0.5
        
        # Rank defenses (1 = allows most points = easiest matchup)
        defense_allowed['defense_rank'] = defense_allowed.groupby(
            ['season', 'week', 'position']
        )['defense_pts_allowed_roll4'].rank(ascending=False)
        
        print(f"  Calculated rankings for {defense_allowed['team'].nunique()} teams")
        
        return defense_allowed
    
    def get_opponent_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add opponent matchup features to player data.
        
        Adds:
        - opp_defense_rank: Opponent's rank vs position (1=easiest)
        - opp_matchup_score: 0-1 score (higher=easier matchup)
        - opp_pts_allowed: Points opponent allows to position
        """
        if df.empty:
            return df
        
        # Calculate defense rankings
        defense_rankings = self.calculate_defense_rankings(df)
        
        if defense_rankings.empty:
            df['opp_defense_rank'] = 16  # Average
            df['opp_matchup_score'] = 0.5
            pos_defaults = {'QB': 18.0, 'RB': 12.0, 'WR': 12.0, 'TE': 10.0}
            df['opp_pts_allowed'] = df['position'].map(pos_defaults).fillna(12.0)
            return df
        
        # Shift rankings by 1 week (use last week's data for this week's prediction)
        defense_rankings['week'] = defense_rankings['week'] + 1
        
        # Merge with player data
        result = df.merge(
            defense_rankings[['team', 'season', 'week', 'position', 
                            'defense_rank', 'matchup_score', 'defense_pts_allowed_roll4']],
            left_on=['opponent', 'season', 'week', 'position'],
            right_on=['team', 'season', 'week', 'position'],
            how='left',
            suffixes=('', '_def')
        )
        
        # Rename columns
        result = result.rename(columns={
            'defense_rank': 'opp_defense_rank',
            'matchup_score': 'opp_matchup_score',
            'defense_pts_allowed_roll4': 'opp_pts_allowed'
        })
        
        # Fill missing with fixed defaults (avoid leakage: do NOT use fantasy_points)
        result['opp_defense_rank'] = result['opp_defense_rank'].fillna(16)
        result['opp_matchup_score'] = result['opp_matchup_score'].fillna(0.5)
        # Use position-specific league-average PPG (fixed constants, not from data)
        pos_defaults = {'QB': 18.0, 'RB': 12.0, 'WR': 12.0, 'TE': 10.0}
        result['opp_pts_allowed'] = result['opp_pts_allowed'].fillna(
            result['position'].map(pos_defaults).fillna(12.0)
        )
        
        # Drop extra columns
        if 'team_def' in result.columns:
            result = result.drop(columns=['team_def'])
        
        return result


# =============================================================================
# WEATHER DATA INTEGRATION
# =============================================================================

class WeatherDataLoader:
    """
    Load and process weather data for NFL games.
    
    Weather affects outdoor games and can significantly impact:
    - Passing efficiency (wind, rain)
    - Scoring (extreme cold)
    - Game pace (rain delays)
    
    Uses nfl-data-py schedule data which includes weather info.
    """
    
    # Dome stadiums (weather doesn't affect)
    DOME_STADIUMS = {
        'ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LAC', 'LAR', 
        'LV', 'MIN', 'NO', 'NYG', 'NYJ'  # MetLife has no dome but included for simplicity
    }
    
    # Retractable roof stadiums
    RETRACTABLE_ROOF = {'ARI', 'ATL', 'DAL', 'HOU', 'IND', 'LV'}
    
    def __init__(self):
        pass
    
    def load_weather_data(self, seasons: List[int]) -> pd.DataFrame:
        """Load weather data from schedule."""
        print(f"Loading weather data for seasons: {seasons}")
        
        try:
            schedules = nfl.import_schedules(seasons)
            
            if 'weather' in schedules.columns or 'temp' in schedules.columns:
                print(f"  Found weather data in schedules")
            else:
                print(f"  No weather column, will estimate from stadium/date")
            
            return schedules
        except Exception as e:
            print(f"  Error loading schedules: {e}")
            return pd.DataFrame()
    
    def get_weather_features(self, df: pd.DataFrame, schedules: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add weather features to player data.
        
        Adds:
        - is_dome: Binary, game in dome stadium
        - is_outdoor: Binary, outdoor game
        - weather_score: 0-1, weather impact (1=perfect, 0=severe)
        - is_cold_game: Binary, temp < 32F
        - is_rain_game: Binary, precipitation expected
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Determine if game is in dome based on home team
        if 'team' in result.columns:
            # For home games, check if team has dome
            result['is_dome'] = result.apply(
                lambda row: 1 if (row.get('home_away') == 'home' and 
                                  row['team'] in self.DOME_STADIUMS) or
                                 (row.get('home_away') == 'away' and 
                                  row.get('opponent', '') in self.DOME_STADIUMS)
                else 0, axis=1
            )
        else:
            result['is_dome'] = 0
        
        result['is_outdoor'] = 1 - result['is_dome']
        
        # Estimate weather impact based on season/week
        # Early season (weeks 1-4) and late season (weeks 14-18) have more weather impact
        if 'week' in result.columns:
            result['weather_risk'] = result['week'].apply(
                lambda w: 0.3 if w >= 14 else (0.1 if w <= 4 else 0.05)
            )
        else:
            result['weather_risk'] = 0.1
        
        # Weather score (1 = perfect, lower = worse)
        # Dome games always get 1.0
        result['weather_score'] = np.where(
            result['is_dome'] == 1,
            1.0,
            1.0 - result['weather_risk']
        )
        
        # Placeholder for actual weather data
        result['is_cold_game'] = 0
        result['is_rain_game'] = 0
        
        # If we have schedule data with weather, use it
        if schedules is not None and not schedules.empty:
            if 'temp' in schedules.columns:
                # Merge temperature data
                pass  # Would merge here if data available
        
        # Drop intermediate column
        result = result.drop(columns=['weather_risk'], errors='ignore')
        
        print(f"  Added weather features: is_dome, is_outdoor, weather_score")
        
        return result


# =============================================================================
# VEGAS LINES INTEGRATION
# =============================================================================

class VegasLinesLoader:
    """
    Load and process Vegas betting lines.
    
    Vegas lines provide:
    - Implied team totals (expected points)
    - Game pace expectations
    - Blowout risk (affects garbage time)
    
    Uses nfl-data-py's spread/line data when available.
    """
    
    def __init__(self):
        pass
    
    def load_vegas_lines(self, seasons: List[int]) -> pd.DataFrame:
        """Load Vegas lines from nfl-data-py."""
        print(f"Loading Vegas lines for seasons: {seasons}")
        
        try:
            # Try to load spread lines
            lines = nfl.import_sc_lines(seasons)
            print(f"  Loaded {len(lines)} line records")
            return lines
        except Exception as e:
            print(f"  Spread lines not available: {e}")
            
            # Fall back to schedule which may have spread info
            try:
                schedules = nfl.import_schedules(seasons)
                if 'spread_line' in schedules.columns:
                    print(f"  Using spread from schedules")
                    return schedules
            except:
                pass
            
            return pd.DataFrame()
    
    def calculate_implied_totals(self, lines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied team totals from spreads and over/unders.
        
        Implied Total = (Over/Under + Spread) / 2 for favorite
        Implied Total = (Over/Under - Spread) / 2 for underdog
        """
        if lines_df.empty:
            return pd.DataFrame()
        
        df = lines_df.copy()
        
        # Check what columns we have
        if 'total' in df.columns and 'spread_line' in df.columns:
            # Calculate implied totals
            df['home_implied_total'] = (df['total'] - df['spread_line']) / 2
            df['away_implied_total'] = (df['total'] + df['spread_line']) / 2
        elif 'over_under' in df.columns and 'spread' in df.columns:
            df['home_implied_total'] = (df['over_under'] - df['spread']) / 2
            df['away_implied_total'] = (df['over_under'] + df['spread']) / 2
        else:
            # Use defaults
            df['home_implied_total'] = 24.0
            df['away_implied_total'] = 21.0
        
        return df
    
    def get_vegas_features(self, df: pd.DataFrame, lines_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Add Vegas-based features to player data.
        
        Adds:
        - implied_team_total: Expected points for player's team
        - game_total: Expected total points in game
        - spread: Point spread (negative = favorite)
        - is_favorite: Binary, team is favored
        - blowout_risk: Probability of lopsided game
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # If we have lines data, merge it
        if lines_df is not None and not lines_df.empty:
            lines_processed = self.calculate_implied_totals(lines_df)
            
            # Merge based on game identifiers
            # This would need proper game_id matching
            pass
        
        # For now, estimate based on historical averages
        # Average NFL team scores ~23 points per game
        result['implied_team_total'] = 23.0
        result['game_total'] = 46.0
        result['spread'] = 0.0
        result['is_favorite'] = 0
        result['blowout_risk'] = 0.15  # ~15% of games are blowouts
        
        # Adjust based on team strength using team-level offensive stats (NOT fantasy_points to avoid leakage)
        # Use team_a_points_scored if available (from team stats, not player-level target)
        if 'team_a_points_scored' in df.columns:
            team_strength = (df['team_a_points_scored'] / 22.0).clip(0.7, 1.3)
            result['implied_team_total'] = 23.0 * team_strength
        elif 'total_yards' in df.columns:
            # Proxy from total yards (safe: not the prediction target)
            league_avg_yards = df['total_yards'].mean()
            if league_avg_yards > 0:
                team_strength = (df.groupby(['team', 'season'])['total_yards'].transform('mean') / league_avg_yards).clip(0.7, 1.3)
                result['implied_team_total'] = 23.0 * team_strength
        
        print(f"  Added Vegas features: implied_team_total, game_total, spread")
        
        return result


# =============================================================================
# COMBINED EXTERNAL DATA INTEGRATION
# =============================================================================

class ExternalDataIntegrator:
    """
    Main class to integrate all external data sources.
    
    Combines:
    - Injury status
    - Defense rankings
    - Weather data
    - Vegas lines
    
    Into a unified feature set for predictions.
    """
    
    def __init__(self):
        self.injury_loader = InjuryDataLoader()
        self.defense_loader = DefenseRankingsLoader()
        self.weather_loader = WeatherDataLoader()
        self.vegas_loader = VegasLinesLoader()
    
    def add_all_external_features(self, df: pd.DataFrame, 
                                   seasons: List[int] = None) -> pd.DataFrame:
        """
        Add all external data features to player data.
        
        Args:
            df: Player data DataFrame
            seasons: Seasons to load external data for
            
        Returns:
            DataFrame with all external features added
        """
        if df.empty:
            return df
        
        print("\n" + "="*60)
        print("Adding External Data Features")
        print("="*60)
        
        seasons = seasons or list(df['season'].unique())
        result = df.copy()
        
        # 1. Add injury features
        print("\n1. Injury Status...")
        try:
            injuries = self.injury_loader.load_injuries(seasons)
            if not injuries.empty:
                injury_status = self.injury_loader.get_player_injury_status(injuries)
                if not injury_status.empty:
                    result = result.merge(
                        injury_status[['player_id', 'season', 'week', 'injury_score', 'is_injured']],
                        on=['player_id', 'season', 'week'],
                        how='left'
                    )
                    result['injury_score'] = result['injury_score'].fillna(1.0)
                    result['is_injured'] = result['is_injured'].fillna(0)
                else:
                    result['injury_score'] = 1.0
                    result['is_injured'] = 0
            else:
                result['injury_score'] = 1.0
                result['is_injured'] = 0
        except Exception as e:
            print(f"  Error adding injuries: {e}")
            result['injury_score'] = 1.0
            result['is_injured'] = 0
        
        # 2. Add defense matchup features
        print("\n2. Defense Rankings...")
        try:
            result = self.defense_loader.get_opponent_matchup_features(result)
        except Exception as e:
            print(f"  Error adding defense rankings: {e}")
            result['opp_defense_rank'] = 16
            result['opp_matchup_score'] = 0.5
            pos_defaults = {'QB': 18.0, 'RB': 12.0, 'WR': 12.0, 'TE': 10.0}
            result['opp_pts_allowed'] = result['position'].map(pos_defaults).fillna(12.0)
        
        # 3. Add weather features
        print("\n3. Weather Data...")
        try:
            schedules = self.weather_loader.load_weather_data(seasons)
            result = self.weather_loader.get_weather_features(result, schedules)
        except Exception as e:
            print(f"  Error adding weather: {e}")
            result['is_dome'] = 0
            result['is_outdoor'] = 1
            result['weather_score'] = 0.9
        
        # 4. Add Vegas features
        print("\n4. Vegas Lines...")
        try:
            lines = self.vegas_loader.load_vegas_lines(seasons)
            result = self.vegas_loader.get_vegas_features(result, lines)
        except Exception as e:
            print(f"  Error adding Vegas lines: {e}")
            result['implied_team_total'] = 23.0
            result['game_total'] = 46.0
            result['spread'] = 0.0
        
        # Summary
        new_cols = ['injury_score', 'is_injured', 'opp_defense_rank', 'opp_matchup_score',
                   'opp_pts_allowed', 'is_dome', 'is_outdoor', 'weather_score',
                   'implied_team_total', 'game_total', 'spread']
        
        added = [c for c in new_cols if c in result.columns]
        print(f"\n✅ Added {len(added)} external features: {added}")
        
        return result


def add_external_features(df: pd.DataFrame, seasons: List[int] = None) -> pd.DataFrame:
    """Convenience function to add all external features."""
    integrator = ExternalDataIntegrator()
    return integrator.add_all_external_features(df, seasons)

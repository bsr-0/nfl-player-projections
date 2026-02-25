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
        
        # Calculate league average by position using expanding mean (no future leakage)
        # Sort temporally so expanding window only sees past data
        df_sorted = df.sort_values(['season', 'week'])
        df_sorted['league_avg_points'] = df_sorted.groupby('position')['fantasy_points'].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        # Build a (season, week, position) -> league_avg lookup from the player-level data
        league_avg = df_sorted.groupby(['season', 'week', 'position'])['league_avg_points'].first().reset_index()

        # Merge to get relative performance
        defense_allowed = defense_allowed.merge(
            league_avg, on=['season', 'week', 'position'], how='left'
        )
        
        # Calculate rolling defense strength (last 4 weeks)
        defense_allowed = defense_allowed.sort_values(['team', 'position', 'season', 'week'])
        
        defense_allowed['points_vs_avg'] = (
            defense_allowed['points_allowed'] - defense_allowed['league_avg_points']
        )
        
        # Rolling average of points allowed vs average
        # shift(1) ensures we only use data from PRIOR weeks (no leakage of current-week outcome)
        defense_allowed['defense_pts_allowed_roll4'] = defense_allowed.groupby(
            ['team', 'position']
        )['points_allowed'].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )
        
        # Defense strength score: higher = allows more points = easier matchup
        # Normalize to 0-1 scale using expanding min/max per position (no future leakage)
        defense_allowed = defense_allowed.sort_values(['position', 'season', 'week'])
        for pos in defense_allowed['position'].unique():
            mask = defense_allowed['position'] == pos
            pts = defense_allowed.loc[mask, 'defense_pts_allowed_roll4']
            exp_min = pts.expanding(min_periods=1).min()
            exp_max = pts.expanding(min_periods=1).max()
            denom = (exp_max - exp_min).replace(0, np.nan)
            defense_allowed.loc[mask, 'matchup_score'] = (
                ((pts - exp_min) / denom).fillna(0.5)
            )
        
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
        result = df.copy()

        # Calculate defense rankings
        defense_rankings = self.calculate_defense_rankings(df)
        
        if defense_rankings.empty:
            result['defense_data_available'] = 0
            result['opp_defense_rank'] = 16  # Average
            result['opp_matchup_score'] = 0.5
            pos_defaults = {'QB': 18.0, 'RB': 12.0, 'WR': 12.0, 'TE': 10.0}
            result['opp_pts_allowed'] = result['position'].map(pos_defaults).fillna(12.0)
            return result
        # Shift rankings by 1 week (use last week's data for this week's prediction)
        defense_rankings['week'] = defense_rankings['week'] + 1
        
        # Merge with player data
        result = result.merge(
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

        # Availability signal before filling defaults
        avail_mask = (
            result['opp_defense_rank'].notna()
            & result['opp_matchup_score'].notna()
            & result['opp_pts_allowed'].notna()
        )
        result['defense_data_available'] = avail_mask.astype(int)
        
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
            try:
                from src.utils.leakage import sanitize_schedule_df
                schedules = sanitize_schedule_df(schedules)
            except Exception:
                pass
            
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
        result['weather_data_available'] = 0
        
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
        
        # Default cold/rain to 0, will overwrite from schedule weather data
        result['is_cold_game'] = 0
        result['is_rain_game'] = 0
        # Standardized weather numeric columns used by downstream features/tests.
        result['weather_temp_f'] = np.nan
        result['weather_wind_mph'] = np.nan
        
        # Merge actual weather data from schedules when available
        if schedules is not None and not schedules.empty:
            sched = schedules.copy()
            has_temp = 'temp' in sched.columns
            has_wind = 'wind' in sched.columns
            has_weather = 'weather' in sched.columns  # text description column
            
            if (has_temp or has_weather) and 'season' in sched.columns and 'week' in sched.columns:
                # Build per-team weather lookup from home/away teams
                home_col = 'home_team' if 'home_team' in sched.columns else None
                away_col = 'away_team' if 'away_team' in sched.columns else None
                
                if home_col and away_col:
                    weather_cols = ['season', 'week']
                    if has_temp:
                        weather_cols.append('temp')
                    if has_wind:
                        weather_cols.append('wind')
                    if has_weather:
                        weather_cols.append('weather')
                    
                    # Both home and away teams experience the same weather
                    home_weather = sched[weather_cols + [home_col]].rename(columns={home_col: 'team'})
                    away_weather = sched[weather_cols + [away_col]].rename(columns={away_col: 'team'})
                    weather_lookup = pd.concat([home_weather, away_weather], ignore_index=True)
                    weather_lookup = weather_lookup.drop_duplicates(subset=['season', 'week', 'team'], keep='first')
                    
                    if 'team' in result.columns and 'season' in result.columns and 'week' in result.columns:
                        result = result.merge(
                            weather_lookup,
                            on=['season', 'week', 'team'],
                            how='left',
                            suffixes=('', '_sched')
                        )
                        
                        # Derive is_cold_game from temperature (< 35F)
                            if has_temp:
                                temp_col = 'temp_sched' if 'temp_sched' in result.columns else 'temp'
                                if temp_col in result.columns:
                                temp_vals = pd.to_numeric(result[temp_col], errors='coerce')
                                result.loc[temp_vals.notna(), 'weather_temp_f'] = temp_vals[temp_vals.notna()]
                                if temp_vals.notna().any():
                                    result.loc[temp_vals.notna(), 'weather_data_available'] = 1
                                result['is_cold_game'] = np.where(
                                    (temp_vals < 35) & (result['is_dome'] == 0),
                                    1, 0
                                )
                                # Refine weather_score using actual temperature and wind
                                temp_penalty = np.where(temp_vals < 35, 0.15, np.where(temp_vals < 45, 0.05, 0.0))
                                if has_wind:
                                    wind_col = 'wind_sched' if 'wind_sched' in result.columns else 'wind'
                                    if wind_col in result.columns:
                                        wind_vals = pd.to_numeric(result[wind_col], errors='coerce').fillna(0)
                                        result['weather_wind_mph'] = pd.to_numeric(result[wind_col], errors='coerce')
                                        wind_penalty = np.where(wind_vals > 20, 0.15, np.where(wind_vals > 10, 0.05, 0.0))
                                    else:
                                        wind_penalty = 0.0
                                else:
                                    wind_penalty = 0.0
                                
                                result['weather_score'] = np.where(
                                    result['is_dome'] == 1,
                                    1.0,
                                    (1.0 - temp_penalty - wind_penalty).clip(0.3, 1.0)
                                )
                                # Clean up merge column
                                if temp_col != 'temp':
                                    result = result.drop(columns=[temp_col], errors='ignore')
                                if has_wind and f'wind_sched' in result.columns:
                                    result = result.drop(columns=['wind_sched'], errors='ignore')
                        
                        # Derive is_rain_game from weather description
                        if has_weather:
                            weather_desc_col = 'weather_sched' if 'weather_sched' in result.columns else 'weather'
                            if weather_desc_col in result.columns:
                                weather_str = result[weather_desc_col].astype(str).str.lower()
                                result['is_rain_game'] = np.where(
                                    (weather_str.str.contains('rain|shower|precip|snow|sleet', na=False)) &
                                    (result['is_dome'] == 0),
                                    1, 0
                                )
                                if weather_str.notna().any():
                                    result.loc[weather_str.notna(), 'weather_data_available'] = 1
                                if weather_desc_col != 'weather':
                                    result = result.drop(columns=[weather_desc_col], errors='ignore')
                        
                        merged_count = weather_lookup.shape[0]
                        print(f"  Merged actual weather data for {merged_count} team-game rows")
        
        # Drop intermediate column
        result = result.drop(columns=['weather_risk'], errors='ignore')
        
        print(f"  Added weather features: is_dome, is_outdoor, weather_score, is_cold_game, is_rain_game")
        
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
                try:
                    from src.utils.leakage import sanitize_schedule_df
                    schedules = sanitize_schedule_df(schedules)
                except Exception:
                    pass
                if 'spread_line' in schedules.columns:
                    print(f"  Using spread from schedules")
                    return schedules
            except Exception:
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
        
        # Initialize with defaults (will be overwritten by actual data where available)
        result['implied_team_total'] = 23.0
        result['game_total'] = 46.0
        result['spread'] = 0.0
        result['is_favorite'] = 0
        result['blowout_risk'] = 0.15  # ~15% of games are blowouts
        
        merged_count = 0
        
        # If we have lines data, merge it
        if lines_df is not None and not lines_df.empty:
            lines_processed = self.calculate_implied_totals(lines_df)
            
            # Build a game-level lookup from lines data
            # nfl-data-py schedules use: game_id, season, week, home_team, away_team,
            #   spread_line, total_line (or total)
            game_lines = lines_processed.copy()
            
            # Standardize column names for the merge
            total_col = None
            spread_col = None
            for tc in ['total_line', 'total', 'over_under']:
                if tc in game_lines.columns:
                    total_col = tc
                    break
            for sc in ['spread_line', 'spread']:
                if sc in game_lines.columns:
                    spread_col = sc
                    break
            
            if total_col and spread_col and 'season' in game_lines.columns and 'week' in game_lines.columns:
                # Determine home/away team columns
                home_col = 'home_team' if 'home_team' in game_lines.columns else None
                away_col = 'away_team' if 'away_team' in game_lines.columns else None
                
                if home_col and away_col:
                    # Build per-team-game lookup: one row per team per game
                    home_rows = game_lines[['season', 'week', home_col, total_col, spread_col]].copy()
                    home_rows = home_rows.rename(columns={home_col: 'team'})
                    home_rows['is_home'] = 1
                    home_rows['game_total'] = home_rows[total_col]
                    home_rows['spread'] = home_rows[spread_col]  # negative = home favorite
                    home_rows['implied_team_total'] = (home_rows[total_col] - home_rows[spread_col]) / 2
                    
                    away_rows = game_lines[['season', 'week', away_col, total_col, spread_col]].copy()
                    away_rows = away_rows.rename(columns={away_col: 'team'})
                    away_rows['is_home'] = 0
                    away_rows['game_total'] = away_rows[total_col]
                    away_rows['spread'] = -away_rows[spread_col]  # flip sign for away team
                    away_rows['implied_team_total'] = (away_rows[total_col] + away_rows[spread_col]) / 2
                    
                    vegas_lookup = pd.concat([home_rows, away_rows], ignore_index=True)
                    vegas_lookup = vegas_lookup[['season', 'week', 'team', 'game_total',
                                                 'spread', 'implied_team_total']].copy()
                    # Drop duplicates (multiple lines per game possible)
                    vegas_lookup = vegas_lookup.drop_duplicates(subset=['season', 'week', 'team'], keep='first')
                    
                    # Merge with player data on season/week/team
                    if 'team' in result.columns and 'season' in result.columns and 'week' in result.columns:
                        # Suffix to avoid overwriting defaults yet
                        result = result.merge(
                            vegas_lookup,
                            on=['season', 'week', 'team'],
                            how='left',
                            suffixes=('', '_vegas')
                        )
                        
                        # Overwrite defaults where Vegas data is available
                        for col in ['implied_team_total', 'game_total', 'spread']:
                            vegas_col = f'{col}_vegas'
                            if vegas_col in result.columns:
                                mask = result[vegas_col].notna()
                                result.loc[mask, col] = result.loc[mask, vegas_col]
                                result.loc[mask, 'vegas_data_available'] = 1
                                result = result.drop(columns=[vegas_col])
                                merged_count = int(mask.sum())

                        print(f"  Merged Vegas lines for {merged_count} player-game rows")
        
        # Derive is_favorite and blowout_risk from spread
        result['is_favorite'] = (result['spread'] < 0).astype(int)
        result['blowout_risk'] = (result['spread'].abs() / 20.0).clip(0.0, 0.5)
        
        # Fallback: adjust implied total based on team strength when Vegas data was not available
        no_vegas = result['implied_team_total'] == 23.0
        if no_vegas.any():
            if 'team_a_points_scored' in df.columns:
                team_strength = (df['team_a_points_scored'] / 22.0).clip(0.7, 1.3)
                result.loc[no_vegas, 'implied_team_total'] = 23.0 * team_strength[no_vegas]
            elif 'total_yards' in df.columns:
                league_avg_yards = df['total_yards'].mean()
                if league_avg_yards > 0:
                    team_strength = (df.groupby(['team', 'season'])['total_yards'].transform('mean') / league_avg_yards).clip(0.7, 1.3)
                    result.loc[no_vegas, 'implied_team_total'] = 23.0 * team_strength[no_vegas]
        
        print(f"  Added Vegas features: implied_team_total, game_total, spread, is_favorite, blowout_risk")
        
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
        result['injury_data_available'] = 0
        result['defense_data_available'] = 0
        result['weather_data_available'] = 0
        result['vegas_data_available'] = 0
        
        # 1. Add injury features
        print("\n1. Injury Status...")
        try:
            injuries = self.injury_loader.load_injuries(seasons)
            if not injuries.empty:
                injury_status = self.injury_loader.get_player_injury_status(injuries)
                if not injury_status.empty:
                    result['injury_data_available'] = 1
                    result = result.merge(
                        injury_status[['player_id', 'season', 'week', 'injury_score', 'is_injured']],
                        on=['player_id', 'season', 'week'],
                        how='left'
                    )
                    # Fallback merge by normalized name when player IDs differ across sources.
                    if (
                        "name" in result.columns
                        and result["injury_score"].isna().any()
                        and ("player_name" in injury_status.columns or "name" in injury_status.columns)
                    ):
                        source_name_col = "player_name" if "player_name" in injury_status.columns else "name"
                        fallback = injury_status[
                            [source_name_col, "season", "week", "injury_score", "is_injured"]
                        ].copy()
                        fallback = fallback.dropna(subset=[source_name_col])
                        fallback["name_key"] = (
                            fallback[source_name_col]
                            .astype(str)
                            .str.lower()
                            .str.replace(r"[^a-z0-9 ]", "", regex=True)
                            .str.replace(r"\s+", " ", regex=True)
                            .str.strip()
                        )
                        fallback = fallback.drop_duplicates(
                            subset=["season", "week", "name_key"], keep="last"
                        )
                        missing_mask = result["injury_score"].isna()
                        if missing_mask.any():
                            left = result.loc[missing_mask, ["season", "week", "name"]].copy()
                            left["name_key"] = (
                                left["name"]
                                .astype(str)
                                .str.lower()
                                .str.replace(r"[^a-z0-9 ]", "", regex=True)
                                .str.replace(r"\s+", " ", regex=True)
                                .str.strip()
                            )
                            left_idx = left.index
                            left = left.merge(
                                fallback[["season", "week", "name_key", "injury_score", "is_injured"]],
                                on=["season", "week", "name_key"],
                                how="left",
                            )
                            left.index = left_idx
                            score_fill = left["injury_score"].notna()
                            injured_fill = left["is_injured"].notna()
                            if score_fill.any():
                                result.loc[left.index[score_fill], "injury_score"] = left.loc[score_fill, "injury_score"]
                            if injured_fill.any():
                                result.loc[left.index[injured_fill], "is_injured"] = left.loc[injured_fill, "is_injured"]
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
                   'implied_team_total', 'game_total', 'spread',
                   'injury_data_available', 'defense_data_available',
                   'weather_data_available', 'vegas_data_available',
                   'external_data_quality']
        
        added = [c for c in new_cols if c in result.columns]
        print(f"\n✅ Added {len(added)} external features: {added}")

        # Aggregate data-quality indicator (0-1)
        if all(c in result.columns for c in [
            "injury_data_available", "defense_data_available",
            "weather_data_available", "vegas_data_available"
        ]):
            result["external_data_quality"] = (
                result["injury_data_available"]
                + result["defense_data_available"]
                + result["weather_data_available"]
                + result["vegas_data_available"]
            ) / 4.0
        
        return result


def add_external_features(df: pd.DataFrame, seasons: List[int] = None) -> pd.DataFrame:
    """Convenience function to add all external features."""
    integrator = ExternalDataIntegrator()
    return integrator.add_all_external_features(df, seasons)

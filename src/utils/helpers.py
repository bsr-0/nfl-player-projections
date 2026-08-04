"""Helper utility functions."""
from typing import Dict, Any
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_scoring():
    """Lazy import to avoid circular import with config.settings."""
    from config.settings import SCORING
    return SCORING


def calculate_fantasy_points(stats: Dict[str, Any]) -> float:
    """Calculate PPR fantasy points from player stats."""
    SCORING = _get_scoring()
    points = 0.0

    points += stats.get("passing_yards", 0) * SCORING["passing_yards"]
    points += stats.get("passing_tds", 0) * SCORING["passing_tds"]
    points += stats.get("interceptions", 0) * SCORING["interceptions"]
    points += stats.get("rushing_yards", 0) * SCORING["rushing_yards"]
    points += stats.get("rushing_tds", 0) * SCORING["rushing_tds"]
    points += stats.get("receptions", 0) * SCORING["receptions"]
    points += stats.get("receiving_yards", 0) * SCORING["receiving_yards"]
    points += stats.get("receiving_tds", 0) * SCORING["receiving_tds"]
    points += stats.get("fumbles_lost", 0) * SCORING["fumbles_lost"]
    points += stats.get("two_point_conversions", 0) * SCORING["two_point_conversions"]
    
    return round(points, 2)


def calculate_fantasy_points_df(df: pd.DataFrame) -> pd.Series:
    """Calculate fantasy points for a DataFrame of stats."""
    SCORING = _get_scoring()
    points = pd.Series(0.0, index=df.index)

    if "passing_yards" in df.columns:
        points += df["passing_yards"].fillna(0) * SCORING["passing_yards"]
    if "passing_tds" in df.columns:
        points += df["passing_tds"].fillna(0) * SCORING["passing_tds"]
    if "interceptions" in df.columns:
        points += df["interceptions"].fillna(0) * SCORING["interceptions"]
    if "rushing_yards" in df.columns:
        points += df["rushing_yards"].fillna(0) * SCORING["rushing_yards"]
    if "rushing_tds" in df.columns:
        points += df["rushing_tds"].fillna(0) * SCORING["rushing_tds"]
    if "receptions" in df.columns:
        points += df["receptions"].fillna(0) * SCORING["receptions"]
    if "receiving_yards" in df.columns:
        points += df["receiving_yards"].fillna(0) * SCORING["receiving_yards"]
    if "receiving_tds" in df.columns:
        points += df["receiving_tds"].fillna(0) * SCORING["receiving_tds"]
    if "fumbles_lost" in df.columns:
        points += df["fumbles_lost"].fillna(0) * SCORING["fumbles_lost"]
    if "two_point_conversions" in df.columns:
        points += df["two_point_conversions"].fillna(0) * SCORING["two_point_conversions"]
    
    return points.round(2)


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching across sources."""
    if not name:
        return ""
    
    # Remove common suffixes
    suffixes = [" Jr.", " Sr.", " III", " II", " IV", " V"]
    normalized = name.strip()
    for suffix in suffixes:
        normalized = normalized.replace(suffix, "")
    
    # Lowercase and remove extra spaces
    normalized = " ".join(normalized.lower().split())
    
    return normalized


def generate_player_id(name: str, position: str = None, team: str = None) -> str:
    """Generate a consistent player ID from name and optional metadata."""
    normalized = normalize_player_name(name)
    parts = normalized.split()
    
    if len(parts) >= 2:
        player_id = f"{parts[0][:3]}{parts[-1][:4]}"
    else:
        player_id = normalized[:7]
    
    if position:
        player_id += f"_{position.lower()}"
    
    return player_id


def safe_divide(numerator, denominator, default: float = 0.0):
    """
    Safely divide handling zeros, NaN, and inf.
    
    Works with scalars, numpy arrays, and pandas Series.
    This is the canonical implementation - use this instead of 
    duplicating safe_divide logic elsewhere.
    
    Args:
        numerator: Numerator (scalar, array, or Series)
        denominator: Denominator (scalar, array, or Series)
        default: Value to use when division is undefined
        
    Returns:
        Division result with same type as inputs
    """
    # Handle scalar case
    if isinstance(numerator, (int, float)) and isinstance(denominator, (int, float)):
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    
    # Handle array/Series case (vectorized)
    # Convert to numeric to avoid object dtype issues
    if isinstance(numerator, pd.Series):
        numerator_arr = pd.to_numeric(numerator, errors='coerce').values
    else:
        numerator_arr = np.asarray(numerator, dtype=float)
    
    if isinstance(denominator, pd.Series):
        denominator_arr = pd.to_numeric(denominator, errors='coerce').values
    else:
        denominator_arr = np.asarray(denominator, dtype=float)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            (denominator_arr == 0) | np.isnan(denominator_arr),
            default,
            np.divide(numerator_arr, denominator_arr)
        )
        # Handle any inf values that slipped through
        result = np.where(np.isinf(result), default, result)
        # Handle any NaN values from numeric conversion
        result = np.where(np.isnan(result), default, result)
    
    # Return same type as input
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


def rolling_average(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Calculate rolling average with shift to avoid data leakage."""
    return series.shift(1).rolling(window=window, min_periods=min_periods).mean()


def exponential_weighted_average(series: pd.Series, span: int) -> pd.Series:
    """Calculate exponentially weighted average with shift."""
    return series.shift(1).ewm(span=span, adjust=False).mean()


def create_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    """Create lag features for specified columns."""
    result = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                result[f"{col}_lag{lag}"] = df.groupby("player_id")[col].shift(lag)
    
    return result


def clip_outliers(series: pd.Series, lower_percentile: float = 0.01, 
                  upper_percentile: float = 0.99) -> pd.Series:
    """Clip outliers based on percentiles."""
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower=lower, upper=upper)


def get_season_week_from_date(date_str: str) -> tuple:
    """Extract NFL season and week from a date string."""
    from datetime import datetime
    
    date = datetime.strptime(date_str, "%Y-%m-%d")
    year = date.year
    month = date.month
    
    # NFL season starts in September
    if month >= 9:
        season = year
    else:
        season = year - 1
    
    # Approximate week calculation (would need actual schedule for precision)
    if month >= 9:
        week = (date.isocalendar()[1] - 35) % 18 + 1
    else:
        week = (date.isocalendar()[1] + 17) % 18 + 1
    
    return season, max(1, min(18, week))


def team_abbreviation_map() -> Dict[str, str]:
    """Return mapping of team name variations to standard abbreviations."""
    return {
        "arizona cardinals": "ARI", "arizona": "ARI", "cardinals": "ARI",
        "atlanta falcons": "ATL", "atlanta": "ATL", "falcons": "ATL",
        "baltimore ravens": "BAL", "baltimore": "BAL", "ravens": "BAL",
        "buffalo bills": "BUF", "buffalo": "BUF", "bills": "BUF",
        "carolina panthers": "CAR", "carolina": "CAR", "panthers": "CAR",
        "chicago bears": "CHI", "chicago": "CHI", "bears": "CHI",
        "cincinnati bengals": "CIN", "cincinnati": "CIN", "bengals": "CIN",
        "cleveland browns": "CLE", "cleveland": "CLE", "browns": "CLE",
        "dallas cowboys": "DAL", "dallas": "DAL", "cowboys": "DAL",
        "denver broncos": "DEN", "denver": "DEN", "broncos": "DEN",
        "detroit lions": "DET", "detroit": "DET", "lions": "DET",
        "green bay packers": "GB", "green bay": "GB", "packers": "GB",
        "houston texans": "HOU", "houston": "HOU", "texans": "HOU",
        "indianapolis colts": "IND", "indianapolis": "IND", "colts": "IND",
        "jacksonville jaguars": "JAX", "jacksonville": "JAX", "jaguars": "JAX",
        "kansas city chiefs": "KC", "kansas city": "KC", "chiefs": "KC",
        "las vegas raiders": "LV", "las vegas": "LV", "raiders": "LV", "oakland raiders": "LV",
        "los angeles chargers": "LAC", "la chargers": "LAC", "chargers": "LAC",
        "los angeles rams": "LAR", "la rams": "LAR", "rams": "LAR",
        "miami dolphins": "MIA", "miami": "MIA", "dolphins": "MIA",
        "minnesota vikings": "MIN", "minnesota": "MIN", "vikings": "MIN",
        "new england patriots": "NE", "new england": "NE", "patriots": "NE",
        "new orleans saints": "NO", "new orleans": "NO", "saints": "NO",
        "new york giants": "NYG", "ny giants": "NYG", "giants": "NYG",
        "new york jets": "NYJ", "ny jets": "NYJ", "jets": "NYJ",
        "philadelphia eagles": "PHI", "philadelphia": "PHI", "eagles": "PHI",
        "pittsburgh steelers": "PIT", "pittsburgh": "PIT", "steelers": "PIT",
        "san francisco 49ers": "SF", "san francisco": "SF", "49ers": "SF",
        "seattle seahawks": "SEA", "seattle": "SEA", "seahawks": "SEA",
        "tampa bay buccaneers": "TB", "tampa bay": "TB", "buccaneers": "TB", "bucs": "TB",
        "tennessee titans": "TEN", "tennessee": "TEN", "titans": "TEN",
        "washington commanders": "WAS", "washington": "WAS", "commanders": "WAS",
        "washington football team": "WAS", "redskins": "WAS",
    }


def standardize_team_name(team: str) -> str:
    """Convert team name to standard abbreviation."""
    if not team:
        return ""
    
    team_map = team_abbreviation_map()
    team_lower = team.lower().strip()
    
    # Check if already an abbreviation
    if team.upper() in team_map.values():
        return team.upper()
    
    return team_map.get(team_lower, team.upper()[:3])

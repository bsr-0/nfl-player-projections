"""
Utilization Score and Advanced Feature Engineering.

Core utilization score is computed by utilization_score.calculate_utilization_scores
(single source of truth). This module adds:
- WOPR (Weighted Opportunity Rating)
- Expected Fantasy Points
- Weekly Volatility and Consistency Scores
- Uncertainty Quantification
- Rolling/lag features and playoff/week context
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.utilization_score import calculate_utilization_scores
from config.settings import UTILIZATION_WEIGHTS


class UtilizationCalculator:
    """
    Backward-compatible wrapper: core utilization from utilization_score module,
    plus WOPR and expected_fp. Single source of truth: utilization_score.calculate_utilization_scores.
    """

    WEIGHTS = UTILIZATION_WEIGHTS

    def __init__(self, weights: Optional[Dict] = None, percentile_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None):
        self.weights = weights
        self.percentile_bounds = percentile_bounds

    def calculate_all_scores(self, player_df: pd.DataFrame, team_df: pd.DataFrame = None) -> pd.DataFrame:
        """Compute utilization scores (delegate to utilization_score) then add WOPR and expected_fp."""
        team_df = team_df if team_df is not None else pd.DataFrame()
        df = calculate_utilization_scores(
            player_df, team_df=team_df, weights=self.weights, percentile_bounds=self.percentile_bounds
        )
        df = self._add_wopr(df)
        df = self._add_expected_fp(df)
        return df

    def _add_wopr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add WOPR = 1.5 * target_share + 0.7 * air_yards_share (from utilization_score components if present)."""
        result = df.copy()
        if "target_share_pct" in result.columns and "air_yards_share_pct" in result.columns:
            # Use pct columns from utilization_score (0-100 scale)
            result["wopr"] = (1.5 * (result["target_share_pct"] / 100) + 0.7 * (result["air_yards_share_pct"] / 100))
        elif "target_share" in result.columns and "air_yards_share" in result.columns:
            result["wopr"] = 1.5 * result["target_share"] + 0.7 * result["air_yards_share"]
        else:
            result["wopr"] = 0.0
        result["wopr_normalized"] = (result["wopr"] * 100 / 2.2).clip(0, 100)
        return result

    def _add_expected_fp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expected_fp and fp_over_expected from opportunities."""
        result = df.copy()
        PTS_PER_TARGET, PTS_PER_CARRY, PTS_PER_PASS_ATT = 1.5, 0.7, 0.35
        result["expected_fp"] = 0.0
        ra = result.get("rushing_attempts", pd.Series(0, index=result.index))
        tg = result.get("targets", pd.Series(0, index=result.index))
        pa = result.get("passing_attempts", pd.Series(0, index=result.index))
        rb = result["position"] == "RB"
        wr = result["position"] == "WR"
        te = result["position"] == "TE"
        qb = result["position"] == "QB"
        result.loc[rb, "expected_fp"] = ra[rb].fillna(0) * PTS_PER_CARRY + tg[rb].fillna(0) * PTS_PER_TARGET
        result.loc[wr, "expected_fp"] = tg[wr].fillna(0) * PTS_PER_TARGET
        result.loc[te, "expected_fp"] = tg[te].fillna(0) * PTS_PER_TARGET * 0.9
        result.loc[qb, "expected_fp"] = pa[qb].fillna(0) * PTS_PER_PASS_ATT + ra[qb].fillna(0) * PTS_PER_CARRY * 1.2
        if "fantasy_points" in result.columns:
            result["fp_over_expected"] = result["fantasy_points"] - result["expected_fp"]
        return result


class VolatilityCalculator:
    """
    Calculate player volatility and consistency metrics.
    
    Weekly Volatility measures week-to-week variance in fantasy production.
    High volatility = boom/bust player (good for best ball, risky for weekly)
    Low volatility = consistent floor (safer for weekly leagues)
    """
    
    def calculate_weekly_volatility(self, df: pd.DataFrame,
                                    min_games: int = 3) -> pd.DataFrame:
        """
        Calculate weekly volatility for each player using expanding windows.

        Uses shift(1) + expanding to avoid leaking current/future game data.
        Volatility = Standard deviation of fantasy points (prior games only)
        Consistency = 1 / (1 + CV) where CV = std/mean
        """
        result = df.copy()
        result = result.sort_values(['player_id', 'season', 'week'])

        # Expanding stats per player using only prior games (shift(1) prevents current-game leakage)
        result['_fp_shifted'] = result.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1)
        )
        result['_fp_exp_mean'] = result.groupby('player_id')['_fp_shifted'].transform(
            lambda x: x.expanding(min_periods=min_games).mean()
        )
        result['_fp_exp_std'] = result.groupby('player_id')['_fp_shifted'].transform(
            lambda x: x.expanding(min_periods=min_games).std()
        )
        result['_fp_exp_min'] = result.groupby('player_id')['_fp_shifted'].transform(
            lambda x: x.expanding(min_periods=min_games).min()
        )
        result['_fp_exp_max'] = result.groupby('player_id')['_fp_shifted'].transform(
            lambda x: x.expanding(min_periods=min_games).max()
        )

        result['weekly_volatility'] = result['_fp_exp_std']
        result['coefficient_of_variation'] = np.where(
            result['_fp_exp_mean'] > 0,
            result['_fp_exp_std'] / result['_fp_exp_mean'],
            0
        )
        result['consistency_score'] = 1 / (1 + result['coefficient_of_variation'])
        result['boom_bust_range'] = result['_fp_exp_max'] - result['_fp_exp_min']

        # Clean up temp columns
        result = result.drop(columns=['_fp_shifted', '_fp_exp_mean', '_fp_exp_std',
                                       '_fp_exp_min', '_fp_exp_max'])

        return result
    
    def calculate_rolling_volatility(self, df: pd.DataFrame,
                                     window: int = 4) -> pd.DataFrame:
        """Calculate rolling volatility over recent games.

        Uses shift(1) to exclude the current game from the rolling window,
        preventing look-ahead leakage.
        """
        result = df.copy()

        # Sort by player and time
        result = result.sort_values(['player_id', 'season', 'week'])

        # Rolling std with shift(1) to exclude current game
        result['rolling_volatility'] = result.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        )

        # Rolling consistency with shift(1) to exclude current game
        rolling_mean = result.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        result['rolling_consistency'] = np.where(
            rolling_mean > 0,
            1 / (1 + result['rolling_volatility'] / rolling_mean),
            0.5
        )

        return result


class UncertaintyQuantifier:
    """
    Quantify prediction uncertainty for fantasy football forecasts.
    
    Provides:
    - Prediction intervals (e.g., 80% confidence bounds)
    - Confidence scores based on data quality
    - Risk-adjusted projections
    """
    
    def __init__(self):
        self.historical_errors = {}
    
    def calculate_prediction_intervals(self, predictions: np.ndarray,
                                       historical_std: float,
                                       confidence: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals based on historical error distribution.
        
        Args:
            predictions: Point predictions
            historical_std: Historical standard deviation of errors
            confidence: Confidence level (e.g., 0.80 for 80% interval)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats
        
        # Z-score for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)
        
        lower = predictions - z * historical_std
        upper = predictions + z * historical_std
        
        # Floor at 0 (can't have negative fantasy points)
        lower = np.maximum(lower, 0)
        
        return lower, upper
    
    def calculate_confidence_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate confidence score for each prediction based on data quality.
        
        Factors:
        - Sample size (more games = higher confidence)
        - Recency (recent data = higher confidence)
        - Consistency (lower volatility = higher confidence)
        - Injury status (healthy = higher confidence)
        """
        result = df.copy()
        result = result.sort_values(['player_id', 'season', 'week'])

        # Sample size factor using expanding count (only prior games, no future leakage)
        games_played = result.groupby('player_id').cumcount()  # 0-indexed count up to current row
        sample_factor = 1 - np.exp(-games_played / 8)  # ~0.63 at 8 games, ~0.86 at 16

        # Consistency factor
        if 'consistency_score' in result.columns:
            consistency_factor = result['consistency_score'].fillna(0.5)
        else:
            consistency_factor = 0.5

        # Recency factor: use per-player cumulative game index (no global max leakage)
        # More recent games get higher confidence. We use the cumcount itself
        # so each row only knows how many games the player has played so far.
        if 'season' in result.columns and 'week' in result.columns:
            # games_played is already the cumcount (0-indexed), use it directly
            # Scale: at game 0 recency is ~0.53, at game 16 it's ~0.86
            recency_factor = (1 - np.exp(-games_played / 8)).values
        else:
            recency_factor = 0.8
        
        # Combine factors
        result['confidence_score'] = (
            0.4 * sample_factor +
            0.3 * consistency_factor +
            0.3 * recency_factor
        ).clip(0, 1)
        
        return result
    
    def calculate_risk_adjusted_projection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk-adjusted projections.
        
        Risk-adjusted = projection * (1 - risk_factor * volatility)
        
        This gives a more conservative estimate for volatile players.
        """
        result = df.copy()
        
        if 'weekly_volatility' not in result.columns:
            vol_calc = VolatilityCalculator()
            result = vol_calc.calculate_weekly_volatility(result)
        
        # Normalize volatility to 0-1 scale using expanding quantile (no future leakage)
        result = result.sort_values(['season', 'week']) if 'season' in result.columns else result
        max_vol = result['weekly_volatility'].expanding(min_periods=10).quantile(0.95)
        max_vol = max_vol.clip(lower=0.1)  # avoid division by zero
        normalized_vol = (result['weekly_volatility'] / max_vol).clip(0, 1)
        
        # Risk adjustment factor (0.1 = 10% discount for max volatility)
        risk_factor = 0.15
        
        result['risk_adjusted_projection'] = result['fantasy_points'] * (1 - risk_factor * normalized_vol)
        
        return result


def engineer_all_features(df: pd.DataFrame, percentile_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    Apply all feature engineering to a player DataFrame.
    Core utilization from utilization_score (single source); then WOPR, expected_fp, volatility, uncertainty, rolling/lag.
    When percentile_bounds is None, loads from MODELS_DIR/utilization_percentile_bounds.json if present (serve consistency).
    """
    print("Engineering advanced features...")
    if percentile_bounds is None:
        try:
            from config.settings import MODELS_DIR
            from src.features.utilization_score import load_percentile_bounds
            bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
            if bounds_path.exists():
                percentile_bounds = load_percentile_bounds(bounds_path)
        except Exception:
            percentile_bounds = None
    util_calc = UtilizationCalculator(percentile_bounds=percentile_bounds)
    df = util_calc.calculate_all_scores(df, pd.DataFrame())
    # Backfill target_share for rolling features if only _pct exists
    if "target_share_pct" in df.columns and "target_share" not in df.columns:
        df["target_share"] = df["target_share_pct"] / 100.0
    
    # Volatility metrics
    vol_calc = VolatilityCalculator()
    df = vol_calc.calculate_weekly_volatility(df)
    df = vol_calc.calculate_rolling_volatility(df)
    
    # Uncertainty metrics
    uncertainty = UncertaintyQuantifier()
    df = uncertainty.calculate_confidence_score(df)
    df = uncertainty.calculate_risk_adjusted_projection(df)
    
    # Rolling averages for trend detection
    df = df.sort_values(['player_id', 'season', 'week'])
    
    for window in [3, 4, 5, 8]:
        # Rolling fantasy points
        df[f'fp_rolling_{window}'] = df.groupby('player_id')['fantasy_points'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        
        # Rolling utilization
        if 'utilization_score' in df.columns:
            df[f'util_rolling_{window}'] = df.groupby('player_id')['utilization_score'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Rolling target share
        if 'target_share' in df.columns:
            df[f'target_share_rolling_{window}'] = df.groupby('player_id')['target_share'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Rolling usage rate (touches, targets, snaps) per requirements III.A
        if 'total_touches' in df.columns:
            df[f'touches_rolling_{window}'] = df.groupby('player_id')['total_touches'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        if 'targets' in df.columns:
            df[f'targets_rolling_{window}'] = df.groupby('player_id')['targets'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Rolling efficiency metrics (yards per touch, catch rate) per requirements III.A
        if 'yards_per_carry' in df.columns:
            df[f'ypc_rolling_{window}'] = df.groupby('player_id')['yards_per_carry'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        if 'catch_rate' in df.columns:
            df[f'catch_rate_rolling_{window}'] = df.groupby('player_id')['catch_rate'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
    
    # Lag features (previous game stats) - requirements: LAG_WEEKS=[1,2,3,4]
    for lag in [1, 2, 3, 4]:
        df[f'fp_lag_{lag}'] = df.groupby('player_id')['fantasy_points'].shift(lag)
        df[f'targets_lag_{lag}'] = df.groupby('player_id')['targets'].shift(lag)
        df[f'rush_att_lag_{lag}'] = df.groupby('player_id')['rushing_attempts'].shift(lag)
        if 'utilization_score' in df.columns:
            df[f'util_lag_{lag}'] = df.groupby('player_id')['utilization_score'].shift(lag)
        if 'snap_share' in df.columns:
            df[f'snap_share_lag_{lag}'] = df.groupby('player_id')['snap_share'].shift(lag)
    
    # Trend features
    df['fp_trend'] = df['fp_rolling_3'] - df['fp_rolling_5']  # Positive = improving
    df['util_trend'] = df.get('util_rolling_3', 0) - df.get('util_rolling_5', 0)
    
    # Season-to-date averages
    df['fp_season_avg'] = df.groupby(['player_id', 'season'])['fantasy_points'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # =================================================================
    # PLAYOFF/SUPER BOWL WEEK FEATURES
    # =================================================================
    # Binary indicators for playoff context
    df['is_playoff_week'] = (df['week'] > 18).astype(int)
    df['is_wild_card'] = (df['week'] == 19).astype(int)
    df['is_divisional'] = (df['week'] == 20).astype(int)
    df['is_conference_championship'] = (df['week'] == 21).astype(int)
    df['is_super_bowl'] = (df['week'] == 22).astype(int)
    
    # Season phase categorical (grouped weeks to avoid overfitting)
    # Early (1-6), Mid (7-12), Late (13-18), Playoff (19+)
    def get_season_phase(week):
        if week <= 6:
            return 0  # Early
        elif week <= 12:
            return 1  # Mid
        elif week <= 18:
            return 2  # Late
        else:
            return 3  # Playoff
    
    df['season_phase'] = df['week'].apply(get_season_phase)
    
    # Week position within season (normalized 0-1)
    df['week_normalized'] = df['week'].clip(upper=18) / 18.0
    
    # Late season indicator (weeks 15-18 when teams may rest starters)
    df['is_late_season'] = ((df['week'] >= 15) & (df['week'] <= 18)).astype(int)
    
    # High stakes games (playoff + late season with playoff implications)
    df['is_high_stakes'] = ((df['week'] >= 15) | (df['week'] > 18)).astype(int)
    
    print(f"  Added playoff/week features: is_playoff_week, is_super_bowl, season_phase, etc.")
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"  Created {len(df.columns)} total features")
    
    return df


def add_external_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add external matchup features from the external_data module.
    
    This is a convenience wrapper that adds:
    - Injury status
    - Defense rankings
    - Weather data
    - Vegas lines
    """
    try:
        from src.data.external_data import add_external_features
        return add_external_features(df)
    except ImportError:
        print("  Warning: external_data module not available")
        return df


# Convenience function for quick utilization score calculation
def quick_utilization_score(targets: int, team_targets: int,
                           rush_att: int, team_rush_att: int,
                           snaps: int, team_snaps: int,
                           position: str = 'RB') -> float:
    """
    Quick calculation of utilization score from basic stats.
    Uses config UTILIZATION_WEIGHTS (single source).
    """
    target_share = targets / team_targets if team_targets > 0 else 0
    rush_share = rush_att / team_rush_att if team_rush_att > 0 else 0
    snap_share = snaps / team_snaps if team_snaps > 0 else 0
    weights = UTILIZATION_WEIGHTS.get(position, UTILIZATION_WEIGHTS["RB"])
    score = (
        weights.get("target_share", 0) * target_share +
        weights.get("rush_share", 0) * rush_share +
        weights.get("snap_share", 0) * snap_share
    )
    return score * 100

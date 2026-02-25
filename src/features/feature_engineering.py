"""Feature engineering pipeline for NFL player prediction.

Missing data root causes (and how we handle them):
- LEFT JOINs in get_all_players_for_training: team_stats, utilization_scores,
  team_defense_stats can be NULL when not populated. We backfill team_stats from
  player aggregates; utilization is computed in pipeline; defense defaults applied.
- Schedule/StrengthOfSchedule can fail or be missing: opponent_rating,
  matchup_difficulty, team_sos get neutral defaults (50.0).
- Rolling/lag features produce NaN for early rows per player: we impute in
  prepare_training_data and in a final _impute_missing step so model never sees NaN/inf.
- Injury/rookie: optional columns injury_score, is_injured, is_rookie get safe
  defaults (1.0, 0, 0) when not provided so they can act as utilization predictors.

Data quality (per requirements): max 5%% missing per feature is acceptable. We flag
features exceeding 5%% missing (logged); imputation strategy: column median, then 0.
Features with >5%% missing are still used but may reduce model reliability.
"""
import pandas as pd
import numpy as np
import os
import warnings
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    ROLLING_WINDOWS, LAG_WEEKS, POSITIONS, MOMENTUM_WEIGHTS,
    BOOM_BUST_THRESHOLDS, BOOM_BUST_DEFAULT,
    AGE_CURVE_PARAMS, AGE_CURVE_DEFAULT,
)
from src.utils.helpers import (
    rolling_average, exponential_weighted_average,
    create_lag_features, safe_divide
)

# Rolling/aggregation on sparse early-season windows can emit this benign warning.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


class FeatureEngineer:
    """Feature engineering for NFL player performance prediction."""
    
    def __init__(self):
        self.rolling_windows = ROLLING_WINDOWS
        self.lag_weeks = LAG_WEEKS
        self.feature_columns = []
    
    def create_features(self, df: pd.DataFrame, 
                        include_target: bool = True) -> pd.DataFrame:
        """
        Create all features for model training/prediction.
        
        Args:
            df: DataFrame with player weekly stats
            include_target: Whether to include target variable
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Sort by player and time
        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        
        # Create base features
        df = self._create_base_features(df)
        
        # Create rolling features (historical averages)
        df = self._create_rolling_features(df)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create trend features
        df = self._create_trend_features(df)
        
        # Create opponent features
        df = self._create_opponent_features(df)
        
        # Create situational features
        df = self._create_situational_features(df)

        # Team-change and scheme-fit features (proactive context adjustment)
        df = self._create_team_change_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Advanced requirement features (boom/bust, season phase, experience, classification, workload risk)
        df = self._create_advanced_requirement_features(df)

        # Return-from-injury production patterns (first 3 games back)
        df = self._create_return_from_injury_features(df)

        # Vegas game script predictors (spread, over/under, implied team total)
        df = self._create_vegas_game_script_features(df)

        # Optional injury/rookie predictors for utilization (defaults when missing)
        df = self._ensure_injury_rookie_features(df)
        
        # Outlier detection per requirements Section VI.C (>3 sigma flagged, not removed)
        df = self._flag_outliers(df, sigma_threshold=3.0)
        
        # Check missing rate per feature (requirement: max 5% acceptable); log exceedances
        self._check_missing_rate(df, threshold_pct=5.0)
        # Final imputation: no NaN/inf in numeric columns so pipelines are robust
        df = self._impute_missing(df)
        
        # Store feature column names
        self._update_feature_columns(df)
        
        return df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features."""
        # Efficiency metrics
        df["yards_per_carry"] = safe_divide(
            df["rushing_yards"], df["rushing_attempts"]
        )
        df["yards_per_target"] = safe_divide(
            df["receiving_yards"], df["targets"]
        )
        df["yards_per_reception"] = safe_divide(
            df["receiving_yards"], df["receptions"]
        )
        df["catch_rate"] = safe_divide(df["receptions"], df["targets"]) * 100

        # Advanced PBP efficiency (EPA/WPA per opportunity)
        pass_plays = df.get("pass_plays", df.get("passing_attempts", pd.Series(0, index=df.index)))
        rush_plays = df.get("rush_plays", df.get("rushing_attempts", pd.Series(0, index=df.index)))
        recv_targets = df.get("recv_targets", df.get("targets", pd.Series(0, index=df.index)))
        if "pass_epa" in df.columns:
            df["pass_epa_per_play"] = safe_divide(df["pass_epa"], pass_plays)
        if "rush_epa" in df.columns:
            df["rush_epa_per_play"] = safe_divide(df["rush_epa"], rush_plays)
        if "recv_epa" in df.columns:
            df["recv_epa_per_target"] = safe_divide(df["recv_epa"], recv_targets)
        if "pass_wpa" in df.columns:
            df["pass_wpa_per_play"] = safe_divide(df["pass_wpa"], pass_plays)
        if "rush_wpa" in df.columns:
            df["rush_wpa_per_play"] = safe_divide(df["rush_wpa"], rush_plays)
        if "recv_wpa" in df.columns:
            df["recv_wpa_per_target"] = safe_divide(df["recv_wpa"], recv_targets)
        
        # QB-specific (only if columns exist)
        if "passing_completions" in df.columns and "passing_attempts" in df.columns:
            df["completion_pct"] = safe_divide(
                df["passing_completions"], df["passing_attempts"]
            ) * 100
            df["yards_per_attempt"] = safe_divide(
                df.get("passing_yards", 0), df["passing_attempts"]
            )
            df["td_rate"] = safe_divide(
                df.get("passing_tds", 0), df["passing_attempts"]
            ) * 100
            df["int_rate"] = safe_divide(
                df.get("interceptions", 0), df["passing_attempts"]
            ) * 100
        
        # Volume metrics (with safe defaults)
        rushing_attempts = df.get("rushing_attempts", pd.Series(0, index=df.index))
        receptions = df.get("receptions", pd.Series(0, index=df.index))
        rushing_yards = df.get("rushing_yards", pd.Series(0, index=df.index))
        receiving_yards = df.get("receiving_yards", pd.Series(0, index=df.index))
        rushing_tds = df.get("rushing_tds", pd.Series(0, index=df.index))
        receiving_tds = df.get("receiving_tds", pd.Series(0, index=df.index))
        targets = df.get("targets", pd.Series(0, index=df.index))
        
        df["total_touches"] = rushing_attempts + receptions
        df["total_yards"] = rushing_yards + receiving_yards
        df["total_tds"] = rushing_tds + receiving_tds
        df["opportunities"] = rushing_attempts + targets
        
        # Weighted opportunities (rushing worth more than targets)
        df["weighted_opportunities"] = rushing_attempts * 2 + targets
        
        # TD dependency
        df["yards_per_td"] = safe_divide(
            df["total_yards"], df["total_tds"].replace(0, np.nan)
        )
        
        # QB advanced: air yards per attempt, TD/INT ratio, deep ball attempts
        if "air_yards" in df.columns and "passing_attempts" in df.columns:
            df["air_yards_per_attempt"] = safe_divide(df["air_yards"], df["passing_attempts"])
        if "passing_tds" in df.columns and "interceptions" in df.columns:
            df["td_int_ratio"] = safe_divide(
                df["passing_tds"], df["interceptions"].replace(0, 0.5)
            )
        if "deep_pass_attempts" in df.columns:
            df["deep_ball_pct"] = safe_divide(
                df["deep_pass_attempts"], df.get("passing_attempts", pd.Series(1, index=df.index))
            ) * 100

        # RB advanced: yards after contact, broken tackles
        if "yards_after_contact" in df.columns:
            df["yac_per_carry"] = safe_divide(df["yards_after_contact"], df.get("rushing_attempts", pd.Series(1, index=df.index)))
        if "broken_tackles" in df.columns:
            df["broken_tackle_rate"] = safe_divide(
                df["broken_tackles"], df.get("rushing_attempts", pd.Series(1, index=df.index))
            )

        # WR/TE advanced: average depth of target, yards after catch, contested catch rate
        if "average_depth_of_target" in df.columns:
            pass  # already present as aDOT
        elif "air_yards" in df.columns and "targets" in df.columns:
            df["average_depth_of_target"] = safe_divide(df["air_yards"], df["targets"])
        if "yards_after_catch" in df.columns and "receptions" in df.columns:
            df["yac_per_reception"] = safe_divide(df["yards_after_catch"], df["receptions"])
        if "contested_catches" in df.columns and "contested_targets" in df.columns:
            df["contested_catch_rate"] = safe_divide(df["contested_catches"], df["contested_targets"])
        if "slot_snaps" in df.columns and "snap_count" in df.columns:
            df["slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100

        # Route participation rate for RB receiving work
        if "routes_run" in df.columns and "snap_count" in df.columns:
            df["route_participation_rate"] = safe_divide(df["routes_run"], df["snap_count"])

        # Game script indicators
        if "home_away" in df.columns:
            df["is_home"] = (df["home_away"] == "home").astype(int)
        else:
            df["is_home"] = 0

        # Season progress
        df["season_week_pct"] = df["week"] / 18

        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling average features for key metrics."""
        rolling_cols = [
            "fantasy_points", "rushing_yards", "rushing_attempts", "rushing_tds",
            "receiving_yards", "receptions", "targets", "receiving_tds",
            "passing_yards", "passing_attempts", "passing_tds", "interceptions",
            "total_touches", "total_yards", "opportunities", "utilization_score",
            "yards_per_carry", "yards_per_target", "catch_rate"
        ]
        
        # Filter to columns that exist
        rolling_cols = [c for c in rolling_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        
        for window in self.rolling_windows:
            for col in rolling_cols:
                # Rolling mean (shifted to avoid leakage)
                new_cols[f"{col}_roll{window}_mean"] = df.groupby("player_id")[col].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std for volatility (per requirements: utilization_score volatility too)
                if col in ["fantasy_points", "total_yards", "total_touches", "utilization_score"]:
                    new_cols[f"{col}_roll{window}_std"] = df.groupby("player_id")[col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
                    )
        
        # Exponential weighted averages (more weight on recent games)
        # Per requirements: heavy EWMA weighting for 4-week model; multiple spans
        ewm_cols = ["fantasy_points", "total_yards", "opportunities", "utilization_score"]
        ewm_cols = [c for c in ewm_cols if c in df.columns]
        
        for col in ewm_cols:
            for span in [3, 5, 8]:
                new_cols[f"{col}_ewm{span}"] = df.groupby("player_id")[col].transform(
                    lambda x, s=span: x.shift(1).ewm(span=s, adjust=False).mean()
                )
        
        # Regression-to-mean features for long-horizon (18-week) model
        # Per requirements: 18-week model should heavily weight regression to mean
        # IMPORTANT: Use expanding (causal) position-level aggregates to avoid lookahead bias.
        if "fantasy_points" in df.columns:
            if "position" in df.columns:
                # Expanding position-level mean/std: only data available up to each row
                pos_expanding_mean = df.groupby("position")["fantasy_points"].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )
                player_ewm = df.groupby("player_id")["fantasy_points"].transform(
                    lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
                )
                new_cols["fp_deviation_from_pos_mean"] = player_ewm - pos_expanding_mean
                pos_expanding_std = df.groupby("position")["fantasy_points"].transform(
                    lambda x: x.shift(1).expanding(min_periods=2).std()
                ).clip(lower=1.0)
                new_cols["fp_regression_to_mean_z"] = (player_ewm - pos_expanding_mean) / pos_expanding_std
            # Season-level mean for same player: use expanding mean within each
            # (player, season) group to avoid using future games within the season.
            if "season" in df.columns:
                season_expanding_ppg = df.groupby(["player_id", "season"])["fantasy_points"].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )
                df["_tmp_season_ppg"] = season_expanding_ppg
                new_cols["prev_season_ppg"] = df.groupby("player_id")["_tmp_season_ppg"].shift(1)
                df.drop(columns=["_tmp_season_ppg"], inplace=True, errors="ignore")
        
        if "utilization_score" in df.columns and "position" in df.columns:
            pos_util_expanding_mean = df.groupby("position")["utilization_score"].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            player_util_ewm = df.groupby("player_id")["utilization_score"].transform(
                lambda x: x.shift(1).ewm(span=8, adjust=False).mean()
            )
            new_cols["util_deviation_from_pos_mean"] = player_util_ewm - pos_util_expanding_mean
        
        # Add all new columns at once using pd.concat to avoid fragmentation
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for recent performance."""
        lag_cols = [
            "fantasy_points", "total_yards", "total_touches", "total_tds",
            "utilization_score", "snap_share"
        ]
        
        lag_cols = [c for c in lag_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        for lag in self.lag_weeks:
            for col in lag_cols:
                new_cols[f"{col}_lag{lag}"] = df.groupby("player_id")[col].shift(lag)
        
        # Add all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend features to capture momentum."""
        trend_cols = ["fantasy_points", "total_yards", "utilization_score"]
        trend_cols = [c for c in trend_cols if c in df.columns]
        
        # Collect all new columns in a dictionary to avoid fragmentation
        new_cols = {}
        
        for col in trend_cols:
            # Short-term trend (last 3 games slope)
            new_cols[f"{col}_trend3"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 3)
            )
            
            # Medium-term trend (last 5 games slope)
            new_cols[f"{col}_trend5"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 5)
            )
            
            # Long-term trend (last 8 games slope) per requirements III.A
            new_cols[f"{col}_trend8"] = df.groupby("player_id")[col].transform(
                lambda x: self._calculate_trend(x, 8)
            )
        
        # Week-over-week change (shift(1) to avoid leakage - use prior week's change only)
        for col in ["fantasy_points", "total_yards"]:
            if col in df.columns:
                shifted = df.groupby("player_id")[col].shift(1)
                new_cols[f"{col}_wow_change"] = shifted.diff()
                new_cols[f"{col}_wow_pct_change"] = shifted.pct_change(fill_method=None)
        
        # Add all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)
        
        return df
    
    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend (slope) over a rolling window."""
        def slope(x):
            if len(x) < 2:
                return 0
            x_clean = x.dropna()
            if len(x_clean) < 2:
                return 0
            return np.polyfit(range(len(x_clean)), x_clean, 1)[0]
        
        return series.shift(1).rolling(window=window, min_periods=2).apply(slope, raw=False)
    
    def _create_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on opponent strength."""
        # Fantasy points allowed by opponent (if available)
        opp_cols = [
            "fantasy_points_allowed_qb", "fantasy_points_allowed_rb",
            "fantasy_points_allowed_wr", "fantasy_points_allowed_te"
        ]
        
        for col in opp_cols:
            if col in df.columns:
                # Normalize to z-score using expanding (causal) mean/std.
                # shift(1) excludes the current row to prevent self-referential leakage.
                # Sort by (season, week) first so expanding windows respect temporal order.
                if "season" in df.columns and "week" in df.columns:
                    sorted_df = df.sort_values(["season", "week"])
                    shifted = sorted_df[col].shift(1)
                    expanding_mean = shifted.expanding(min_periods=1).mean()
                    expanding_std = shifted.expanding(min_periods=2).std().clip(lower=1e-6)
                    df[f"{col}_zscore"] = ((sorted_df[col] - expanding_mean) / expanding_std).reindex(df.index)
                else:
                    shifted = df[col].shift(1)
                    expanding_mean = shifted.expanding(min_periods=1).mean()
                    expanding_std = shifted.expanding(min_periods=2).std().clip(lower=1e-6)
                    df[f"{col}_zscore"] = (df[col] - expanding_mean) / expanding_std
        
        # Create position-specific opponent feature
        if "position" in df.columns:
            df["opp_fpts_allowed"] = np.nan
            
            for pos in POSITIONS:
                col = f"fantasy_points_allowed_{pos.lower()}"
                if col in df.columns:
                    mask = df["position"] == pos
                    # Convert values to float to avoid dtype incompatibility
                    values = pd.to_numeric(df.loc[mask, col], errors='coerce')
                    df.loc[mask, "opp_fpts_allowed"] = values.values
        
        # Add comprehensive team-level features (TeamA = player's team, TeamB = opponent)
        df = self._add_team_matchup_features(df)
        
        return df
    
    def _add_team_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add team-level features for both player's team (TeamA) and opponent (TeamB).
        
        This includes offensive/defensive metrics for game context prediction.
        """
        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            
            # Get all team stats
            all_team_stats = db.get_team_stats()
            if all_team_stats.empty:
                return df
            
            # Calculate rolling team averages (last 3 games)
            team_metrics = ['points_scored', 'points_allowed', 'total_yards',
                           'passing_yards', 'rushing_yards', 'turnovers',
                           'pass_attempts', 'rush_attempts', 'redzone_scores',
                           'neutral_pass_plays', 'neutral_run_plays',
                           'neutral_pass_rate', 'neutral_pass_rate_lg', 'neutral_pass_rate_oe',
                           'drive_count', 'drive_success_rate', 'avg_drive_epa',
                           'points_per_drive', 'pace_sec_per_play']
            
            # Create team averages lookup using PRIOR season's data to avoid
            # lookahead bias (current season avg includes future weeks).
            team_avgs = all_team_stats.groupby(['team', 'season']).agg({
                col: 'mean' for col in team_metrics if col in all_team_stats.columns
            }).reset_index()
            # Shift season forward by 1 so that a row for season S uses season S-1 averages
            team_avgs['season'] = team_avgs['season'] + 1
            
            # Rename for TeamA (player's team - offensive context)
            team_a_cols = {col: f'team_a_{col}' for col in team_metrics if col in team_avgs.columns}
            team_a_avgs = team_avgs.rename(columns=team_a_cols)
            team_a_avgs = team_a_avgs.rename(columns={'team': 'team', 'season': 'season'})

            # Merge TeamA prior-season stats
            if 'team' in df.columns and 'season' in df.columns:
                merge_cols = ['team', 'season']
                df = df.merge(
                    team_a_avgs,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_team_a')
                )

            # Create TeamB (opponent) averages
            team_b_cols = {col: f'team_b_{col}' for col in team_metrics if col in team_avgs.columns}
            team_b_avgs = team_avgs.rename(columns=team_b_cols)
            team_b_avgs = team_b_avgs.rename(columns={'team': 'opponent', 'season': 'season'})

            # Merge TeamB prior-season stats (opponent)
            if 'opponent' in df.columns and 'season' in df.columns:
                merge_cols = ['opponent', 'season']
                df = df.merge(
                    team_b_avgs,
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_team_b')
                )

            # In-season rolling team averages (4-week causal window) to supplement
            # stale prior-season averages.  shift(1) excludes current week.
            # Blend: 60% in-season / 40% prior-season when week >= 5, else prior-season only.
            if 'team' in df.columns and 'season' in df.columns and 'week' in df.columns:
                avail_metrics = [m for m in team_metrics if m in all_team_stats.columns]
                if avail_metrics:
                    ts_sorted = all_team_stats.sort_values(['team', 'season', 'week'])
                    for metric in avail_metrics:
                        ts_sorted[f'_inseason_{metric}'] = ts_sorted.groupby(
                            ['team', 'season']
                        )[metric].transform(
                            lambda x: x.shift(1).rolling(4, min_periods=2).mean()
                        )
                    inseason_cols = ['team', 'season', 'week'] + [f'_inseason_{m}' for m in avail_metrics]
                    inseason_df = ts_sorted[inseason_cols].drop_duplicates()
                    df = df.merge(inseason_df, on=['team', 'season', 'week'], how='left')
                    # Blend in-season rolling with prior-season for TeamA columns
                    for metric in avail_metrics:
                        ta_col = f'team_a_{metric}'
                        is_col = f'_inseason_{metric}'
                        if ta_col in df.columns and is_col in df.columns:
                            has_inseason = df[is_col].notna() & (df['week'] >= 5)
                            df.loc[has_inseason, ta_col] = (
                                0.6 * df.loc[has_inseason, is_col] +
                                0.4 * df.loc[has_inseason, ta_col]
                            )
                    # Drop temporary in-season columns
                    df = df.drop(columns=[f'_inseason_{m}' for m in avail_metrics], errors='ignore')

            # Create matchup differential features
            matchup_pairs = [
                ('team_a_points_scored', 'team_b_points_allowed', 'matchup_scoring_edge'),
                ('team_a_total_yards', 'team_b_total_yards', 'matchup_yards_diff'),
                ('team_a_passing_yards', 'team_b_passing_yards', 'matchup_pass_diff'),
                ('team_a_rushing_yards', 'team_b_rushing_yards', 'matchup_rush_diff'),
            ]
            
            for col_a, col_b, new_col in matchup_pairs:
                if col_a in df.columns and col_b in df.columns:
                    df[new_col] = df[col_a] - df[col_b]
            
            # Game script prediction features
            if 'team_a_points_scored' in df.columns and 'team_b_points_scored' in df.columns:
                # Expected game total
                df['expected_game_total'] = df['team_a_points_scored'] + df['team_b_points_scored']
                # Expected point differential (positive = player's team favored)
                df['expected_point_diff'] = df['team_a_points_scored'] - df['team_b_points_allowed']
            
            # Pace features (plays per game)
            if 'team_a_pass_attempts' in df.columns and 'team_a_rush_attempts' in df.columns:
                df['team_a_plays_per_game'] = df['team_a_pass_attempts'] + df['team_a_rush_attempts']
            if 'team_b_pass_attempts' in df.columns and 'team_b_rush_attempts' in df.columns:
                df['team_b_plays_per_game'] = df['team_b_pass_attempts'] + df['team_b_rush_attempts']
            
            # Pass/rush tendency
            if 'team_a_pass_attempts' in df.columns and 'team_a_rush_attempts' in df.columns:
                total = df['team_a_pass_attempts'] + df['team_a_rush_attempts']
                df['team_a_pass_rate'] = df['team_a_pass_attempts'] / total.replace(0, 1)

            # Offensive momentum score per requirements III.B: weighted combination of
            # team offensive EPA trend (proxied by points_scored), pass/rush success rate
            # trends (passing_yards, rushing_yards), and scoring efficiency (turnovers).
            # Time-weighted: recent 4 weeks = 60%, weeks 5-8 = 30%, weeks 9+ = 10%.
            if 'week' in all_team_stats.columns and 'points_scored' in all_team_stats.columns:
                ts = all_team_stats.sort_values(['team', 'season', 'week'])

                def _momentum_60_30_10(grp: pd.Series) -> pd.Series:
                    """Time-weighted momentum: 60% last 4w, 30% weeks 5-8, 10% weeks 9+."""
                    out = pd.Series(index=grp.index, dtype=float)
                    for i in range(len(grp)):
                        hist = grp.iloc[:i]  # past weeks only (no leakage)
                        if len(hist) == 0:
                            out.iloc[i] = np.nan
                            continue
                        vals = hist.values[::-1]  # most recent first
                        n = len(vals)
                        w = np.zeros(n)
                        w[: min(4, n)] = 0.6 / min(4, n)
                        if n > 4:
                            w[4: min(8, n)] = 0.3 / min(4, n - 4)
                        if n > 8:
                            w[8:] = 0.1 / (n - 8)
                        out.iloc[i] = np.nansum(vals * w)
                    return out

                # Primary component: scoring (proxy for offensive EPA)
                ts["_mom_pts"] = ts.groupby(
                    ['team', 'season'], group_keys=False
                )['points_scored'].transform(_momentum_60_30_10)

                # Secondary components: passing yards, rushing yards, turnover trend
                # (available from team_stats; use when present for richer composite)
                composite_parts = [("_mom_pts", 0.50)]  # 50% scoring efficiency
                if 'passing_yards' in ts.columns:
                    ts["_mom_pass"] = ts.groupby(
                        ['team', 'season'], group_keys=False
                    )['passing_yards'].transform(_momentum_60_30_10)
                    composite_parts.append(("_mom_pass", 0.20))  # 20% pass success
                if 'rushing_yards' in ts.columns:
                    ts["_mom_rush"] = ts.groupby(
                        ['team', 'season'], group_keys=False
                    )['rushing_yards'].transform(_momentum_60_30_10)
                    composite_parts.append(("_mom_rush", 0.15))  # 15% rush success
                if 'turnovers' in ts.columns:
                    # Turnovers are negative: invert so fewer turnovers = higher momentum
                    ts["_mom_to"] = -ts.groupby(
                        ['team', 'season'], group_keys=False
                    )['turnovers'].transform(_momentum_60_30_10)
                    composite_parts.append(("_mom_to", 0.15))  # 15% ball security

                # Normalize each component to z-scores using EXPANDING (causal)
                # mean/std within each team-season. shift(1) excludes the current
                # row from its own z-score to prevent self-referential leakage.
                for col, _ in composite_parts:
                    exp_mean = ts.groupby(['team', 'season'])[col].transform(
                        lambda x: x.shift(1).expanding(min_periods=1).mean()
                    )
                    exp_std = ts.groupby(['team', 'season'])[col].transform(
                        lambda x: x.shift(1).expanding(min_periods=2).std()
                    ).clip(lower=1e-6)
                    ts[col + "_z"] = ((ts[col] - exp_mean) / exp_std).fillna(0.0)

                # Redistribute weight if some components are missing
                total_weight = sum(w for _, w in composite_parts)
                ts["offensive_momentum_score"] = sum(
                    ts[col + "_z"] * (w / total_weight) for col, w in composite_parts
                )
                # Rescale from z-score space to interpretable ~0-44 range (mean ~22)
                # Use expanding stats with shift(1) to avoid self-referential leakage.
                exp_mom_mean = ts.groupby(['team', 'season'])["offensive_momentum_score"].transform(
                    lambda x: x.shift(1).expanding(min_periods=1).mean()
                )
                exp_mom_std = ts.groupby(['team', 'season'])["offensive_momentum_score"].transform(
                    lambda x: x.shift(1).expanding(min_periods=2).std()
                ).clip(lower=1e-6)
                ts["offensive_momentum_score"] = (
                    22.0 + 8.0 * (ts["offensive_momentum_score"] - exp_mom_mean) / exp_mom_std
                ).fillna(22.0).clip(0, 44)

                # Drop temp columns
                temp_cols = [c for c in ts.columns if c.startswith("_mom_")]
                ts = ts.drop(columns=temp_cols, errors="ignore")

                mom = ts[['team', 'season', 'week', 'offensive_momentum_score']].drop_duplicates()
                if 'team' in df.columns and 'season' in df.columns and 'week' in df.columns:
                    df = df.merge(mom, on=['team', 'season', 'week'], how='left')
                    df['offensive_momentum_score'] = df['offensive_momentum_score'].fillna(22.0)
        except Exception as e:
            # Team features are optional - don't fail if unavailable
            pass

        # --- Divisional game and prime-time game indicators (per requirements III.A) ---
        # Populate from nfl-data-py schedule data when available, otherwise keep defaults.
        if 'is_divisional' not in df.columns or 'is_primetime' not in df.columns:
            try:
                import nfl_data_py as nfl
                seasons = sorted(df["season"].unique()) if "season" in df.columns else []
                if seasons:
                    sched = nfl.import_schedules([int(s) for s in seasons])
                    if not sched.empty:
                        # Build a lookup: (season, week, team) -> (div_game, primetime)
                        # nfl-data-py schedules have div_game (bool) and gametime columns.
                        sched_rows = []
                        for _, row in sched.iterrows():
                            s = int(row.get("season", 0))
                            w = int(row.get("week", 0))
                            home = row.get("home_team", "")
                            away = row.get("away_team", "")
                            # Divisional: div_game column if present, else 0
                            is_div = int(row.get("div_game", 0)) if pd.notna(row.get("div_game")) else 0
                            # Prime-time: gametime 20:00+ (8pm+ starts), or game_type indicators
                            gametime = str(row.get("gametime", ""))
                            is_prime = 0
                            if gametime and gametime != "nan":
                                try:
                                    hour = int(gametime.split(":")[0])
                                    is_prime = 1 if hour >= 20 else 0
                                except (ValueError, IndexError):
                                    pass
                            # Also treat Thursday/Monday/Saturday night as primetime
                            gameday = str(row.get("gameday", ""))
                            weekday = str(row.get("weekday", row.get("day_of_week", "")))
                            if weekday.lower() in ("thursday", "monday"):
                                is_prime = 1
                            for team in [home, away]:
                                if team:
                                    sched_rows.append({
                                        "season": s, "week": w, "team": team,
                                        "_is_divisional": is_div, "_is_primetime": is_prime,
                                    })
                        if sched_rows:
                            sched_lookup = pd.DataFrame(sched_rows).drop_duplicates(
                                subset=["season", "week", "team"]
                            )
                            if "team" in df.columns and "season" in df.columns and "week" in df.columns:
                                df = df.merge(
                                    sched_lookup, on=["season", "week", "team"],
                                    how="left", suffixes=("", "_sched")
                                )
                                if "_is_divisional" in df.columns:
                                    if "is_divisional" not in df.columns:
                                        df["is_divisional"] = df["_is_divisional"].fillna(0).astype(int)
                                    else:
                                        df["is_divisional"] = df["is_divisional"].fillna(df["_is_divisional"]).fillna(0).astype(int)
                                    df = df.drop(columns=["_is_divisional"])
                                if "_is_primetime" in df.columns:
                                    if "is_primetime" not in df.columns:
                                        df["is_primetime"] = df["_is_primetime"].fillna(0).astype(int)
                                    else:
                                        df["is_primetime"] = df["is_primetime"].fillna(df["_is_primetime"]).fillna(0).astype(int)
                                    df = df.drop(columns=["_is_primetime"])
            except Exception:
                pass  # Schedule data is optional; fall through to defaults

        if 'offensive_momentum_score' not in df.columns:
            df['offensive_momentum_score'] = 22.0
        if 'is_divisional' not in df.columns:
            df['is_divisional'] = 0
        if 'is_primetime' not in df.columns:
            df['is_primetime'] = 0
        
        # Fill missing team features with league averages
        team_feature_defaults = {
            'team_a_points_scored': 22.0, 'team_a_points_allowed': 22.0,
            'team_a_total_yards': 340.0, 'team_a_passing_yards': 220.0,
            'team_a_rushing_yards': 120.0, 'team_a_turnovers': 1.5,
            'team_b_points_scored': 22.0, 'team_b_points_allowed': 22.0,
            'team_b_total_yards': 340.0, 'team_b_passing_yards': 220.0,
            'team_b_rushing_yards': 120.0, 'team_b_turnovers': 1.5,
            'matchup_scoring_edge': 0.0, 'matchup_yards_diff': 0.0,
            'matchup_pass_diff': 0.0, 'matchup_rush_diff': 0.0,
            'expected_game_total': 44.0, 'expected_point_diff': 0.0,
            'team_a_plays_per_game': 65.0, 'team_b_plays_per_game': 65.0,
            'team_a_pass_rate': 0.55,
            'team_a_neutral_pass_rate': 0.55, 'team_a_neutral_pass_rate_oe': 0.0,
            'team_b_neutral_pass_rate': 0.55, 'team_b_neutral_pass_rate_oe': 0.0,
            'team_a_drive_success_rate': 0.50, 'team_b_drive_success_rate': 0.50,
            'team_a_points_per_drive': 1.8, 'team_b_points_per_drive': 1.8,
            'team_a_avg_drive_epa': 0.0, 'team_b_avg_drive_epa': 0.0,
            'team_a_pace_sec_per_play': 28.0, 'team_b_pace_sec_per_play': 28.0,
        }
        
        for col, default_val in team_feature_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)
        
        return df
    
    def _create_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create situational/contextual features."""
        # Team offensive context
        if "team_plays" in df.columns:
            df["plays_share"] = safe_divide(
                df["opportunities"], df["team_plays"]
            )
        
        if "team_pass_attempts" in df.columns:
            df["team_pass_rate"] = safe_divide(
                df["team_pass_attempts"], 
                df.get("team_plays", df["team_pass_attempts"] + df.get("team_rush_attempts", 0))
            )

        # Situation-specific usage rates (advanced PBP)
        neutral_targets = df.get("neutral_targets", pd.Series(0, index=df.index))
        team_neutral_pass_plays = df.get("team_neutral_pass_plays", pd.Series(0, index=df.index))
        df["neutral_target_share"] = safe_divide(neutral_targets, team_neutral_pass_plays)

        df["third_down_target_rate"] = safe_divide(
            df.get("third_down_targets", pd.Series(0, index=df.index)),
            df.get("targets", pd.Series(0, index=df.index))
        )
        df["short_yardage_touch_rate"] = safe_divide(
            df.get("short_yardage_rushes", pd.Series(0, index=df.index)),
            df.get("rushing_attempts", pd.Series(0, index=df.index))
        )
        df["two_minute_target_rate"] = safe_divide(
            df.get("two_minute_targets", pd.Series(0, index=df.index)),
            df.get("targets", pd.Series(0, index=df.index))
        )
        total_touches = df.get("rushing_attempts", pd.Series(0, index=df.index)) + df.get(
            "targets", pd.Series(0, index=df.index)
        )
        df["high_leverage_touch_rate"] = safe_divide(
            df.get("high_leverage_touches", pd.Series(0, index=df.index)),
            total_touches
        )
        
        # Bye week indicator (no game previous week)
        df["post_bye"] = df.groupby("player_id")["week"].transform(
            lambda x: (x.diff() > 1).astype(int)
        )
        # Days since last game (approx: 7 per week gap; required for data spec)
        df["days_since_last_game"] = df.groupby("player_id")["week"].transform(
            lambda x: (x.diff().fillna(1).clip(lower=1) * 7).astype(float)
        )
        df["days_since_last_game"] = df["days_since_last_game"].fillna(7.0)

        # Short week indicator: Thursday games after Sunday (3 rest days vs 6-7 normal)
        # Detectable from game_time if available, or inferred from week spacing
        df["short_week"] = (df["days_since_last_game"] <= 4.0).astype(int)
        # Rest advantage: positive = more rest than typical 7 days, negative = short week
        df["rest_advantage"] = (df["days_since_last_game"] - 7.0).clip(-4.0, 7.0)
        # Post-bye boost interaction: bye week * recent production trend
        if "fantasy_points_roll3_mean" in df.columns:
            df["post_bye_x_recent_form"] = df["post_bye"] * df["fantasy_points_roll3_mean"].fillna(0)
        # Short week penalty interaction
        if "fantasy_points_roll3_mean" in df.columns:
            df["short_week_x_recent_form"] = df["short_week"] * df["fantasy_points_roll3_mean"].fillna(0)

        # Add schedule-based features if available
        df = self._add_schedule_features(df)
        
        # Add game script / garbage time adjustment
        df = self._add_game_script_adjustment(df)
        
        return df

    def _create_team_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features to proactively adjust for team changes and scheme fit.

        Captures:
        - team_changed: whether player changed teams since last appearance
        - weeks_on_team: number of weeks since joining current team (resets on change or season boundary)
        - team_change_pass_rate_delta: change in team pass rate vs previous team context
        - scheme_fit_score: positional fit to team pass rate (higher = better fit)
        - scheme_mismatch_on_change: mismatch severity when team_changed
        - team_change_recent_util: recent utilization prior to change (lagged)
        """
        if df.empty or "player_id" not in df.columns or "team" not in df.columns:
            return df

        df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
        grp = df.groupby("player_id", sort=False)
        prev_team = grp["team"].shift(1)
        prev_season = grp["season"].shift(1)

        season_changed = (df["season"] != prev_season) & prev_season.notna()
        team_changed = (df["team"] != prev_team) & prev_team.notna()

        df["team_changed"] = team_changed.astype(int)
        df["new_season"] = season_changed.astype(int)

        # Weeks on current team (reset on team change or season boundary)
        change_flag = (team_changed | season_changed).astype(int)
        df["_team_change_flag"] = change_flag
        df["_team_stint_id"] = grp["_team_change_flag"].cumsum()
        df["weeks_on_team"] = df.groupby(["player_id", "_team_stint_id"]).cumcount() + 1

        # Team pass rate delta (use prior row for same player as a proxy)
        if "team_a_pass_rate" in df.columns:
            prev_pass = grp["team_a_pass_rate"].shift(1)
            delta = (df["team_a_pass_rate"] - prev_pass).fillna(0.0)
            df["team_pass_rate_delta"] = delta
            df["team_change_pass_rate_delta"] = (delta * df["team_changed"]).fillna(0.0)
        else:
            df["team_pass_rate_delta"] = 0.0
            df["team_change_pass_rate_delta"] = 0.0

        # Scheme fit: position-specific preferred pass rate
        if "team_a_pass_rate" in df.columns and "position" in df.columns:
            pass_pref = df["position"].map({
                "QB": 0.60,
                "WR": 0.60,
                "TE": 0.58,
                "RB": 0.45,
            }).fillna(0.52)
            mismatch = (df["team_a_pass_rate"] - pass_pref).abs()
            # Normalize mismatch to [0, 1] with 0.6 as a wide max band
            mismatch_norm = (mismatch / 0.6).clip(0.0, 1.0)
            df["scheme_fit_score"] = (1.0 - mismatch_norm).fillna(0.5)
            df["scheme_mismatch"] = mismatch_norm.fillna(0.5)
            df["scheme_mismatch_on_change"] = (df["scheme_mismatch"] * df["team_changed"]).fillna(0.0)
        else:
            df["scheme_fit_score"] = 0.5
            df["scheme_mismatch"] = 0.5
            df["scheme_mismatch_on_change"] = 0.0

        # Recent utilization prior to change (lagged utilization only)
        util_col = None
        for cand in ["utilization_score_roll4_mean", "utilization_score_lag1", "utilization_score_roll3_mean"]:
            if cand in df.columns:
                util_col = cand
                break
        if util_col:
            df["team_change_recent_util"] = np.where(df["team_changed"] == 1,
                                                     df[util_col].fillna(0.0), 0.0)
        else:
            df["team_change_recent_util"] = 0.0

        df = df.drop(columns=["_team_change_flag", "_team_stint_id"], errors="ignore")
        return df
    
    def _add_game_script_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add game script and garbage time adjustment features.
        
        Garbage time stats (when win probability is very low or very high) are
        less meaningful for projecting future performance. A backup RB getting
        15 carries in garbage time doesn't indicate elite usage going forward.
        
        We use score differential and game clock to estimate game script context
        and create adjustment factors for utilization metrics.
        
        Features created:
        - garbage_time_pct: Estimated % of stats accumulated in garbage time
        - game_script_factor: Adjustment factor (0.5-1.0) for utilization metrics
        - competitive_snaps_pct: % of snaps in competitive game situations
        """
        # Check if we have the necessary columns
        has_score_diff = 'score_differential' in df.columns or 'point_differential' in df.columns
        has_win_prob = 'win_probability' in df.columns or 'wp' in df.columns
        
        if has_score_diff:
            score_diff_col = 'score_differential' if 'score_differential' in df.columns else 'point_differential'
            df = self._calculate_garbage_time_from_score(df, score_diff_col)
        elif has_win_prob:
            wp_col = 'win_probability' if 'win_probability' in df.columns else 'wp'
            df = self._calculate_garbage_time_from_wp(df, wp_col)
        else:
            # Estimate from final score if available
            df = self._estimate_garbage_time_from_result(df)
        
        # Create game script adjustment factor
        if 'garbage_time_pct' in df.columns:
            # Discount utilization by garbage time percentage
            # e.g., if 30% of stats came in garbage time, adjustment = 0.85
            df['game_script_factor'] = 1.0 - (0.5 * df['garbage_time_pct'])
            df['game_script_factor'] = df['game_script_factor'].clip(0.5, 1.0)
        else:
            df['game_script_factor'] = 1.0
        
        # Calculate competitive snaps percentage if possible
        if 'competitive_snaps' in df.columns and 'snap_count' in df.columns:
            df['competitive_snaps_pct'] = safe_divide(
                df['competitive_snaps'], df['snap_count']
            )
        else:
            df['competitive_snaps_pct'] = 1.0 - df.get('garbage_time_pct', 0)
        
        return df
    
    def _calculate_garbage_time_from_score(
        self, 
        df: pd.DataFrame, 
        score_diff_col: str
    ) -> pd.DataFrame:
        """
        Calculate garbage time percentage from score differential.
        
        Garbage time is defined as:
        - Leading by 17+ points in 4th quarter
        - Trailing by 17+ points in 4th quarter
        - Leading by 24+ points in 3rd quarter
        - Trailing by 24+ points in 3rd quarter
        """
        # If we have quarter-level data
        if 'quarter' in df.columns:
            garbage_conditions = (
                # 4th quarter blowouts
                ((df['quarter'] == 4) & (df[score_diff_col].abs() >= 17)) |
                # 3rd quarter blowouts
                ((df['quarter'] == 3) & (df[score_diff_col].abs() >= 24))
            )
            df['is_garbage_time'] = garbage_conditions.astype(int)
            
            # Calculate percentage of plays in garbage time
            df['garbage_time_pct'] = df.groupby(['player_id', 'season', 'week'])['is_garbage_time'].transform('mean')
        else:
            # Estimate from final score differential
            # Games with 17+ point differential likely had significant garbage time
            df['garbage_time_pct'] = np.where(
                df[score_diff_col].abs() >= 24, 0.30,
                np.where(
                    df[score_diff_col].abs() >= 17, 0.20,
                    np.where(
                        df[score_diff_col].abs() >= 10, 0.10,
                        0.0
                    )
                )
            )
        
        return df
    
    def _calculate_garbage_time_from_wp(
        self, 
        df: pd.DataFrame, 
        wp_col: str
    ) -> pd.DataFrame:
        """
        Calculate garbage time percentage from win probability.
        
        Garbage time is when win probability is < 10% or > 90%.
        This is the most accurate method when play-by-play data is available.
        """
        # Binary garbage time indicator
        df['is_garbage_time'] = (
            (df[wp_col] < 0.10) | (df[wp_col] > 0.90)
        ).astype(int)
        
        # Calculate percentage of plays in garbage time by game
        df['garbage_time_pct'] = df.groupby(['player_id', 'season', 'week'])['is_garbage_time'].transform('mean')
        
        return df
    
    def _estimate_garbage_time_from_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate garbage time from final game result when detailed data unavailable.
        
        Uses final score differential to estimate how much garbage time likely occurred.
        This is a rough heuristic but better than ignoring game script entirely.
        """
        # Calculate or get final score differential
        if 'team_score' in df.columns and 'opponent_score' in df.columns:
            df['final_margin'] = df['team_score'] - df['opponent_score']
        elif 'margin' in df.columns:
            df['final_margin'] = df['margin']
        else:
            # No score data available
            df['garbage_time_pct'] = 0.0
            return df
        
        # Estimate garbage time based on final margin
        # Larger margins = more likely there was garbage time
        df['garbage_time_pct'] = np.select(
            [
                df['final_margin'].abs() >= 28,  # Blowout: ~35% garbage time
                df['final_margin'].abs() >= 21,  # Big win: ~25% garbage time
                df['final_margin'].abs() >= 14,  # Comfortable: ~15% garbage time
                df['final_margin'].abs() >= 7,   # Close-ish: ~5% garbage time
            ],
            [0.35, 0.25, 0.15, 0.05],
            default=0.0
        )
        
        return df
    
    def create_adjusted_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game-script-adjusted utilization metrics.
        
        Multiplies raw utilization metrics by game_script_factor to discount
        stats accumulated in garbage time.
        
        Args:
            df: DataFrame with utilization metrics and game_script_factor
            
        Returns:
            DataFrame with adjusted utilization columns
        """
        util_cols = [col for col in df.columns if 'utilization' in col.lower() or 'share' in col.lower()]
        
        if 'game_script_factor' not in df.columns:
            return df
        
        for col in util_cols:
            # Create adjusted version
            adj_col = f"{col}_adj"
            df[adj_col] = df[col] * df['game_script_factor']
        
        return df
    
    def _add_schedule_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add schedule and strength of schedule features."""
        try:
            from src.utils.database import DatabaseManager
            from src.scrapers.schedule_scraper import StrengthOfScheduleCalculator
            
            db = DatabaseManager()
            
            # Get unique seasons in data
            seasons = df['season'].unique() if 'season' in df.columns else []
            
            for season in seasons:
                schedule = db.get_schedule(season=int(season))
                if schedule.empty:
                    continue
                
                # Get team stats for SOS calculation
                team_stats = db.get_team_stats(season=int(season) - 1)  # Prior year
                
                sos_calc = StrengthOfScheduleCalculator(team_stats)
                sos_calc.calculate_team_rankings(int(season))
                
                # Calculate SOS for each team
                all_sos = sos_calc.get_all_teams_sos(schedule)
                sos_map = dict(zip(all_sos['team'], all_sos['sos_rating']))
                
                # Add team SOS to player data
                season_mask = df['season'] == season
                if 'team' in df.columns:
                    df.loc[season_mask, 'team_sos'] = df.loc[season_mask, 'team'].map(sos_map)
                
                # Add weekly matchup difficulty
                for team in df.loc[season_mask, 'team'].unique():
                    if pd.isna(team):
                        continue
                    matchups = sos_calc.calculate_weekly_matchup_difficulty(schedule, team)
                    if not matchups.empty:
                        matchup_map = dict(zip(matchups['week'], matchups['matchup_difficulty']))
                        team_mask = season_mask & (df['team'] == team)
                        df.loc[team_mask, 'matchup_difficulty'] = df.loc[team_mask, 'week'].map(matchup_map)
                        
                        # Add opponent rating
                        opp_map = dict(zip(matchups['week'], matchups['opponent_rating']))
                        df.loc[team_mask, 'opponent_rating'] = df.loc[team_mask, 'week'].map(opp_map)
        except Exception as e:
            # Schedule features are optional - don't fail if unavailable
            pass
        
        # Fill missing schedule features with neutral values
        if 'team_sos' not in df.columns:
            df['team_sos'] = 50.0
        if 'matchup_difficulty' not in df.columns:
            df['matchup_difficulty'] = 50.0
        if 'opponent_rating' not in df.columns:
            df['opponent_rating'] = 50.0
        
        df['team_sos'] = df['team_sos'].fillna(50.0)
        df['matchup_difficulty'] = df['matchup_difficulty'].fillna(50.0)
        df['opponent_rating'] = df['opponent_rating'].fillna(50.0)
        
        return df
    
    def refresh_matchup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute schedule- and opponent-dependent features on a dataframe that
        already has team, season, week, opponent, home_away set (e.g. prediction
        input for the upcoming week). Use after overwriting those columns so the
        model sees the correct matchup.
        """
        if df.empty or 'team' not in df.columns or 'season' not in df.columns:
            return df
        df = self._add_schedule_features(df)
        df = self._add_team_matchup_features(df)
        # Ensure neutral defaults for any matchup columns the model might expect
        for col, default in [
            ('team_sos', 50.0), ('matchup_difficulty', 50.0), ('opponent_rating', 50.0),
        ]:
            if col in df.columns:
                df[col] = df[col].fillna(default)
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key metrics."""
        # Utilization x Efficiency interactions
        # Use LAGGED utilization (not current-week) to avoid leakage.
        # The raw utilization_score is excluded from features, so derived
        # features from it would carry the same leakage risk.
        util_lagged = None
        if "utilization_score_lag_1" in df.columns:
            util_lagged = df["utilization_score_lag_1"]
        elif "utilization_score_roll3_mean" in df.columns:
            util_lagged = df["utilization_score_roll3_mean"]
        if util_lagged is not None:
            if "yards_per_carry" in df.columns:
                df["util_x_ypc"] = util_lagged * df["yards_per_carry"]
            if "yards_per_target" in df.columns:
                df["util_x_ypt"] = util_lagged * df["yards_per_target"]
        
        # Volume x Efficiency
        df["touches_x_ypc"] = df["total_touches"] * df.get("yards_per_carry", 0)
        
        # Opportunity x TD rate
        if "total_tds" in df.columns and "opportunities" in df.columns:
            td_rate = safe_divide(df["total_tds"], df["opportunities"])
            df["opp_x_td_rate"] = df["opportunities"] * td_rate
        
        # --- Matchup Quality Indicator (requirements III.B) ---
        # Composite score combining opponent defense weakness, game script favorability,
        # and pace environment. Higher = better matchup for fantasy production.
        # Use expanding mean/std to avoid future data leakage in z-score components.
        mqi_components = []

        # Helper: compute causal z-scores using expanding mean/std (no future leakage).
        # shift(1) ensures the current row's value is excluded from its own z-score.
        def _causal_zscore(series: pd.Series) -> pd.Series:
            if "season" in df.columns and "week" in df.columns:
                sorted_vals = series.reindex(df.sort_values(["season", "week"]).index)
                shifted = sorted_vals.shift(1)
                exp_mean = shifted.expanding(min_periods=1).mean()
                exp_std = shifted.expanding(min_periods=2).std().clip(lower=0.1)
                z = ((sorted_vals - exp_mean) / exp_std).reindex(df.index)
            else:
                shifted = series.shift(1)
                exp_mean = shifted.expanding(min_periods=1).mean()
                exp_std = shifted.expanding(min_periods=2).std().clip(lower=0.1)
                z = (series - exp_mean) / exp_std
            return z

        # Component 1: Opponent points allowed (position-specific when available)
        if "opp_fpts_allowed" in df.columns:
            opp_z = _causal_zscore(df["opp_fpts_allowed"])
            mqi_components.append(opp_z.fillna(0) * 0.35)
        elif "matchup_difficulty" in df.columns:
            # matchup_difficulty: higher = harder opponent, so invert
            md_z = -(df["matchup_difficulty"] - 50.0) / 25.0
            mqi_components.append(md_z.fillna(0) * 0.35)

        # Component 2: Game script favorability (implied team total or expected point diff)
        if "implied_team_total" in df.columns:
            itt_z = _causal_zscore(df["implied_team_total"])
            mqi_components.append(itt_z.fillna(0) * 0.30)
        elif "expected_point_diff" in df.columns:
            epd_z = _causal_zscore(df["expected_point_diff"])
            mqi_components.append(epd_z.fillna(0) * 0.30)

        # Component 3: Pace environment (team plays per game)
        if "team_a_plays_per_game" in df.columns and "team_b_plays_per_game" in df.columns:
            combined_pace = df["team_a_plays_per_game"] + df["team_b_plays_per_game"]
            pace_z = _causal_zscore(combined_pace)
            mqi_components.append(pace_z.fillna(0) * 0.20)
        
        # Component 4: Home field advantage
        if "is_home" in df.columns:
            mqi_components.append(df["is_home"].fillna(0) * 0.15)
        
        if mqi_components:
            df["matchup_quality_indicator"] = sum(mqi_components)
            # Normalize to 0-100 scale using expanding min/max with shift(1)
            # to exclude current row from its own normalization bounds
            mqi = df["matchup_quality_indicator"]
            if "season" in df.columns and "week" in df.columns:
                # Sort by time FIRST so expanding windows are causal
                sort_order = df.sort_values(["season", "week"]).index
                mqi_sorted = mqi.reindex(sort_order)
                shifted = mqi_sorted.shift(1)
                expanding_min = shifted.expanding(min_periods=1).min()
                expanding_max = shifted.expanding(min_periods=1).max()
                denom = (expanding_max - expanding_min).replace(0, np.nan)
                normalized = (((mqi_sorted - expanding_min) / denom) * 100).fillna(50.0).clip(0, 100)
                df["matchup_quality_indicator"] = normalized.reindex(df.index)
            else:
                mqi_min, mqi_max = mqi.min(), mqi.max()
                if mqi_max > mqi_min:
                    df["matchup_quality_indicator"] = ((mqi - mqi_min) / (mqi_max - mqi_min) * 100).clip(0, 100)
                else:
                    df["matchup_quality_indicator"] = 50.0
        else:
            df["matchup_quality_indicator"] = 50.0
        
        return df
    
    def _create_advanced_requirement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features required by the comprehensive rubric but not yet present.

        Adds:
        - Boom/bust rates (% weeks >20 pts, <5 pts) per player (rolling)
        - Season phase indicators (early/mid/late season)
        - Divisional and prime-time game indicators (placeholders if not filled upstream)
        - NFL experience (years in league) and age-adjusted performance curves
        - Player usage classification (workhorse/committee RB, WR1/2/3 designation)
        - Contract year indicator (heuristic from NFL experience when no contract data)
        - Cumulative workload injury risk (per requirements Section II.C)
        """
        if df.empty:
            return df

        new_cols = {}

        # --- Boom / Bust rates (rolling window, shifted to avoid leakage) ---
        # Position-specific thresholds: QB scores higher so needs higher boom threshold, etc.
        if "fantasy_points" in df.columns:
            shifted_fp = df.groupby("player_id")["fantasy_points"].shift(1)
            if "position" in df.columns:
                boom_thresh = df["position"].map(
                    {p: t["boom"] for p, t in BOOM_BUST_THRESHOLDS.items()}
                ).fillna(BOOM_BUST_DEFAULT["boom"])
                bust_thresh = df["position"].map(
                    {p: t["bust"] for p, t in BOOM_BUST_THRESHOLDS.items()}
                ).fillna(BOOM_BUST_DEFAULT["bust"])
            else:
                boom_thresh = BOOM_BUST_DEFAULT["boom"]
                bust_thresh = BOOM_BUST_DEFAULT["bust"]
            boom_flag = (shifted_fp >= boom_thresh).astype(float)
            bust_flag = (shifted_fp < bust_thresh).astype(float)
            for window in [4, 8]:
                new_cols[f"boom_rate_roll{window}"] = boom_flag.groupby(
                    df["player_id"]
                ).transform(lambda x: x.rolling(window, min_periods=1).mean())
                new_cols[f"bust_rate_roll{window}"] = bust_flag.groupby(
                    df["player_id"]
                ).transform(lambda x: x.rolling(window, min_periods=1).mean())

        # --- Season phase indicators (early wk 1-6, mid 7-12, late 13-18) ---
        if "week" in df.columns:
            new_cols["is_early_season"] = (df["week"] <= 6).astype(int)
            new_cols["is_mid_season"] = ((df["week"] >= 7) & (df["week"] <= 12)).astype(int)
            new_cols["is_late_season"] = (df["week"] >= 13).astype(int)

        # --- NFL experience (years in league) ---
        if "season" in df.columns and "player_id" in df.columns:
            first_season = df.groupby("player_id")["season"].transform("min")
            new_cols["nfl_experience_years"] = (df["season"] - first_season).clip(lower=0)

        # --- Age-adjusted performance curve ---
        # Position-specific: RBs peak earlier (~25) and decline faster; QBs/TEs peak later (~28).
        if "age" in df.columns:
            age = df["age"].fillna(26)
        elif "season" in df.columns and "birth_year" in df.columns:
            age = df["season"] - df["birth_year"]
        else:
            age = None
        if age is not None:
            if "position" in df.columns:
                peak = df["position"].map(
                    {p: c["peak"] for p, c in AGE_CURVE_PARAMS.items()}
                ).fillna(AGE_CURVE_DEFAULT["peak"])
                coeff = df["position"].map(
                    {p: c["coefficient"] for p, c in AGE_CURVE_PARAMS.items()}
                ).fillna(AGE_CURVE_DEFAULT["coefficient"])
            else:
                peak = AGE_CURVE_DEFAULT["peak"]
                coeff = AGE_CURVE_DEFAULT["coefficient"]
            new_cols["age_curve"] = 1.0 - coeff * ((age - peak) ** 2)

        # --- Player usage classification (RB: workhorse/committee; WR: WR1/2/3) ---
        if "position" in df.columns and "total_touches" in df.columns:
            # RB workhorse: >15 touches/game rolling average = workhorse
            rb_mask = df["position"] == "RB"
            touches_roll4 = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )
            new_cols["is_workhorse_rb"] = ((touches_roll4 >= 15) & rb_mask).astype(int)
            new_cols["is_committee_rb"] = ((touches_roll4 < 15) & (touches_roll4 >= 5) & rb_mask).astype(int)

        if "position" in df.columns and "targets" in df.columns:
            # WR designation based on rolling target share within team
            tgt_roll4 = df.groupby("player_id")["targets"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )
            wr_mask = df["position"] == "WR"
            new_cols["is_wr1"] = ((tgt_roll4 >= 7) & wr_mask).astype(int)
            new_cols["is_wr2"] = ((tgt_roll4 >= 4) & (tgt_roll4 < 7) & wr_mask).astype(int)
            new_cols["is_wr3"] = ((tgt_roll4 < 4) & (tgt_roll4 >= 1) & wr_mask).astype(int)

            # TE: red zone specialist indicator
            te_mask = df["position"] == "TE"
            if "receiving_tds" in df.columns:
                td_roll4 = df.groupby("player_id")["receiving_tds"].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
                new_cols["is_rz_specialist_te"] = ((td_roll4 >= 0.3) & te_mask).astype(int)

        # --- Three-down back indicator (high snap share + receiving work) ---
        if "snap_share" in df.columns and "receptions" in df.columns:
            snap_roll = df.groupby("player_id")["snap_share"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            rec_roll = df.groupby("player_id")["receptions"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            rb_mask2 = df["position"] == "RB" if "position" in df.columns else pd.Series(False, index=df.index)
            new_cols["is_three_down_back"] = ((snap_roll >= 0.5) & (rec_roll >= 1.5) & rb_mask2).astype(int)

        # --- Contract year indicator (per requirements III.C) ---
        # When an explicit contract_year column is available, use it directly.
        # Otherwise, use NFL experience as a heuristic: players in years 4-5 are
        # typically in or approaching the end of their rookie contract; players in
        # years 8-9 are often approaching a second contract expiration. This is an
        # imperfect proxy but captures the known "contract year bump" effect.
        if "contract_year" in df.columns:
            new_cols["is_contract_year"] = df["contract_year"].fillna(0).astype(int)
        elif "nfl_experience_years" in new_cols:
            exp = new_cols["nfl_experience_years"]
            new_cols["is_contract_year"] = (
                ((exp == 3) | (exp == 4) | (exp == 7) | (exp == 8))
            ).astype(int)
        elif "season" in df.columns and "player_id" in df.columns:
            first_season = df.groupby("player_id")["season"].transform("min")
            exp = (df["season"] - first_season).clip(lower=0)
            new_cols["is_contract_year"] = (
                ((exp == 3) | (exp == 4) | (exp == 7) | (exp == 8))
            ).astype(int)
        else:
            new_cols["is_contract_year"] = 0

        # --- Cumulative workload injury risk (per requirements Section II.C) ---
        # Higher cumulative recent touches = higher injury probability for RBs
        # Age-adjusted: older players have higher risk at same workload level
        # Position-specific thresholds: RB (high risk at 80+ touches/4w),
        # WR/TE (high risk at 40+ touches/4w), QB (sack-based, separate below)
        if "total_touches" in df.columns:
            cum_touches_3w = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).sum()
            )
            cum_touches_4w = df.groupby("player_id")["total_touches"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).sum()
            )
            new_cols["cumulative_workload_3w"] = cum_touches_3w.fillna(0)
            new_cols["cumulative_workload_4w"] = cum_touches_4w.fillna(0)

            # Position-specific workload thresholds (touches per 4 weeks)
            pos_thresholds = {"RB": 80.0, "WR": 40.0, "TE": 35.0, "QB": 120.0}
            base_risk = cum_touches_4w.fillna(0).copy()
            if "position" in df.columns:
                for pos, threshold in pos_thresholds.items():
                    pos_mask = df["position"] == pos
                    base_risk[pos_mask] = (cum_touches_4w[pos_mask].fillna(0) / threshold).clip(0, 1)
            else:
                base_risk = (cum_touches_4w.fillna(0) / 80.0).clip(0, 1)

            # Age multiplier: risk increases ~3% per year above age 27
            # (peak athletic years); younger players get slight discount
            age_multiplier = pd.Series(1.0, index=df.index)
            if "age" in df.columns:
                age = df["age"].fillna(26)
                age_multiplier = (1.0 + 0.03 * (age - 27).clip(lower=-3)).clip(0.9, 1.5)
            elif "age_curve" in new_cols:
                # Invert age_curve: lower curve = older = higher risk
                age_multiplier = (2.0 - new_cols["age_curve"]).clip(0.9, 1.5)

            new_cols["workload_injury_risk"] = (base_risk * age_multiplier).clip(0, 1)
            # Raw (non-age-adjusted) for comparison
            new_cols["workload_injury_risk_raw"] = (cum_touches_4w.fillna(0) / 100.0).clip(0, 1)

        # --- QB-specific: sack rate based injury risk ---
        if "sacks" in df.columns and "passing_attempts" in df.columns:
            sack_roll = df.groupby("player_id")["sacks"].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            ).fillna(0)
            new_cols["qb_sack_injury_risk"] = (sack_roll / 5.0).clip(0, 1)

        # Assign all new columns at once
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

        return df

    def _create_return_from_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create return-from-injury production pattern features.

        Per requirements: track first 3 games back from injury.
        Features:
        - games_since_injury: 0 = just returned, NaN = no recent injury
        - is_first_3_games_back: binary flag for first 3 games after missing time
        - return_from_injury_discount: performance discount factor (0.7-1.0)
        """
        if df.empty or "player_id" not in df.columns:
            df["games_since_injury"] = np.nan
            df["is_first_3_games_back"] = 0
            df["return_from_injury_discount"] = 1.0
            return df

        new_cols = {"games_since_injury": pd.Series(np.nan, index=df.index),
                    "is_first_3_games_back": pd.Series(0, index=df.index, dtype=int),
                    "return_from_injury_discount": pd.Series(1.0, index=df.index)}

        # Detect missed weeks per player (gap > 1 week between consecutive rows)
        for pid, grp in df.groupby("player_id"):
            if len(grp) < 2 or "week" not in grp.columns:
                continue
            idx = grp.index
            weeks = grp["week"].values
            gaps = np.diff(weeks)
            games_since = 999
            for i in range(1, len(idx)):
                if gaps[i - 1] > 1:
                    # Player missed at least one week
                    games_since = 0
                elif games_since < 999:
                    games_since += 1
                if games_since <= 2:
                    row_label = idx[i]
                    new_cols["games_since_injury"].at[row_label] = games_since
                    new_cols["is_first_3_games_back"].at[row_label] = 1
                    # Discount: 0.70 first game back, 0.85 second, 0.95 third
                    discount = [0.70, 0.85, 0.95][min(games_since, 2)]
                    new_cols["return_from_injury_discount"].at[row_label] = discount

        for col_name, series in new_cols.items():
            df[col_name] = series
        df["games_since_injury"] = df["games_since_injury"].fillna(99.0)
        return df

    def _create_vegas_game_script_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Vegas game script predictors from schedule data.

        Per requirements: spread, over/under, implied team total, win probability.
        These come from nfl-data-py schedule data which has spread_line and total_line.
        """
        # Check if vegas features already added by external_data.py
        if "spread" in df.columns and "game_total" in df.columns:
            # Ensure implied team total exists
            if "implied_team_total" not in df.columns:
                df["implied_team_total"] = (df["game_total"] + df["spread"]) / 2
            # Win probability proxy from spread
            if "win_probability" not in df.columns:
                # Rough conversion: spread of -7 ~ 70% win probability
                df["win_probability"] = 0.5 + df["spread"].clip(-14, 14) / 28.0 * (-1)
                df["win_probability"] = df["win_probability"].clip(0.05, 0.95)
            return df

        # Try to load from schedule data
        try:
            import nfl_data_py as nfl
            seasons = sorted(df["season"].unique()) if "season" in df.columns else []
            if not seasons:
                raise ValueError("No seasons")
            schedules = nfl.import_schedules([int(s) for s in seasons])
            if schedules.empty or "spread_line" not in schedules.columns:
                raise ValueError("No spread data in schedules")

            # Build lookup: home_team + away_team + season + week -> spread, total
            sched = schedules.copy()
            sched = sched.rename(columns={"gameday": "game_date"}, errors="ignore")
            if "total_line" in sched.columns:
                sched = sched.rename(columns={"total_line": "total"})
            elif "total" not in sched.columns:
                sched["total"] = 46.0

            # Create home and away lookups
            home_lookup = sched[["season", "week", "home_team", "spread_line", "total"]].copy()
            home_lookup = home_lookup.rename(columns={"home_team": "team"})
            home_lookup["spread"] = -home_lookup["spread_line"]  # negative spread = home favored
            home_lookup["game_total"] = home_lookup["total"]
            home_lookup["implied_team_total"] = (home_lookup["game_total"] - home_lookup["spread"]) / 2

            away_lookup = sched[["season", "week", "away_team", "spread_line", "total"]].copy()
            away_lookup = away_lookup.rename(columns={"away_team": "team"})
            away_lookup["spread"] = away_lookup["spread_line"]  # positive spread = away underdog
            away_lookup["game_total"] = away_lookup["total"]
            away_lookup["implied_team_total"] = (away_lookup["game_total"] + away_lookup["spread"]) / 2

            vegas = pd.concat([home_lookup, away_lookup], ignore_index=True)
            vegas = vegas[["season", "week", "team", "spread", "game_total", "implied_team_total"]].drop_duplicates()

            # Merge
            before_len = len(df)
            df = df.merge(vegas, on=["season", "week", "team"], how="left", suffixes=("", "_vegas"))
            # Prefer existing columns if already present
            for col in ["spread", "game_total", "implied_team_total"]:
                vegas_col = f"{col}_vegas"
                if vegas_col in df.columns:
                    if col not in df.columns:
                        df[col] = df[vegas_col]
                    else:
                        df[col] = df[col].fillna(df[vegas_col])
                    df = df.drop(columns=[vegas_col])

        except Exception:
            pass

        # Defaults for missing values
        for col, default in [("spread", 0.0), ("game_total", 46.0), ("implied_team_total", 23.0)]:
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = df[col].fillna(default)

        # Win probability from spread
        if "win_probability" not in df.columns:
            df["win_probability"] = 0.5 + df["spread"].clip(-14, 14) / 28.0 * (-1)
            df["win_probability"] = df["win_probability"].clip(0.05, 0.95)

        # Is favorite
        if "is_favorite" not in df.columns:
            df["is_favorite"] = (df["spread"] < 0).astype(int)

        return df

    def _ensure_injury_rookie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure injury and rookie predictor columns exist with safe defaults.
        Used as predictors for utilization: injury_score (0-1 availability),
        is_injured (0/1), is_rookie (limited sample = higher uncertainty).
        When external injury data is absent, defaults assume full availability.
        """
        if df.empty:
            return df
        # Injury: from external_data when available; else full availability
        if "injury_score" not in df.columns:
            df["injury_score"] = 1.0
        else:
            df["injury_score"] = df["injury_score"].fillna(1.0).clip(0.0, 1.0)
        if "is_injured" not in df.columns:
            df["is_injured"] = 0
        else:
            df["is_injured"] = df["is_injured"].fillna(0).astype(int).clip(0, 1)
        # Rookie: first season or very few games in current season (improves utilization prediction)
        if "games_count" in df.columns:
            df["is_rookie"] = (df["games_count"] <= 8).astype(int)
        else:
            # Approximate: count rows per player up to this row (no full history in single df)
            games_per_player = df.groupby("player_id").cumcount() + 1
            df["is_rookie"] = (games_per_player <= 8).astype(int)
        return df
    
    def _flag_outliers(self, df: pd.DataFrame, sigma_threshold: float = 3.0) -> pd.DataFrame:
        """
        Flag statistical outliers (>3 standard deviations) per requirements Section VI.C.
        
        Legitimate outliers (record-breaking performances) are kept but flagged.
        Injury-impacted games get special handling via is_outlier_injury flag.
        Creates 'is_statistical_outlier' column (0/1) for model awareness.
        """
        if df.empty:
            return df
        key_cols = ["fantasy_points", "total_yards", "total_touches", "utilization_score"]
        key_cols = [c for c in key_cols if c in df.columns]
        if not key_cols:
            df["is_statistical_outlier"] = 0
            return df
        
        outlier_mask = pd.Series(False, index=df.index)

        # Use expanding mean/std to avoid future data leakage in outlier thresholds
        has_temporal = "season" in df.columns and "week" in df.columns
        if has_temporal:
            sort_idx = df.sort_values(["season", "week"]).index
        for col in key_cols:
            if col not in df.columns:
                continue
            if has_temporal:
                col_sorted = df[col].reindex(sort_idx)
                shifted = col_sorted.shift(1)
                mean_val = shifted.expanding(min_periods=10).mean().reindex(df.index)
                std_val = shifted.expanding(min_periods=10).std().reindex(df.index)
            else:
                mean_val = df[col].mean()
                std_val = df[col].std()
            if isinstance(std_val, (int, float)) and (pd.isna(std_val) or std_val == 0):
                continue
            col_outlier = (df[col] - mean_val).abs() > sigma_threshold * std_val
            outlier_mask = outlier_mask | col_outlier.fillna(False)

        df["is_statistical_outlier"] = outlier_mask.astype(int)
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  Outlier detection: {n_outliers} rows flagged (>{sigma_threshold}σ in {key_cols})")

        # Injury-impacted outlier flag: low performance + injured (expanding stats)
        if "injury_score" in df.columns and "fantasy_points" in df.columns:
            if has_temporal:
                fp_sorted = df["fantasy_points"].reindex(sort_idx)
                fp_shifted = fp_sorted.shift(1)
                fp_mean = fp_shifted.expanding(min_periods=10).mean().reindex(df.index)
                fp_std = fp_shifted.expanding(min_periods=10).std().reindex(df.index)
            else:
                fp_mean = df["fantasy_points"].mean()
                fp_std = df["fantasy_points"].std()
            if isinstance(fp_std, (int, float)) and (pd.isna(fp_std) or fp_std == 0):
                df["is_outlier_injury"] = 0
            else:
                low_perf = df["fantasy_points"] < (fp_mean - 2 * fp_std)
                injured = df["injury_score"].fillna(1.0) < 0.7
                df["is_outlier_injury"] = (low_perf & injured).fillna(False).astype(int)
        else:
            df["is_outlier_injury"] = 0
        
        return df

    def _check_missing_rate(self, df: pd.DataFrame, threshold_pct: float = 5.0) -> None:
        """
        Log features with missing rate above threshold (requirement: max 5% per feature).
        Does not drop columns; call before _impute_missing for visibility.
        """
        if df.empty or len(df) == 0:
            return
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_rows = len(df)
        over = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            missing = df[col].isna().sum()
            if missing == 0:
                continue
            pct = 100.0 * missing / n_rows
            if pct > threshold_pct:
                over.append((col, round(pct, 1)))
        if over:
            over.sort(key=lambda x: -x[1])
            # In test runs this warning is noisy and expected due to synthetic/sparse fixtures.
            if os.getenv("PYTEST_CURRENT_TEST"):
                return
            if os.getenv("NFL_FEATURE_WARN_MISSINGNESS", "1") != "1":
                return
            import warnings
            warnings.warn(
                f"Feature engineering: {len(over)} features exceed {threshold_pct}% missing "
                f"(requirement guideline). Consider reviewing data or dropping: {[x[0] for x in over[:10]]}"
                + (f" ... and {len(over) - 10} more" if len(over) > 10 else ""),
                UserWarning,
                stacklevel=2,
            )

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace inf with nan and impute NaN in numeric columns so model never sees missing/inf.
        Root cause: LEFT JOINs (team_stats, utilization, defense), rolling/lag NaNs, failed schedule.

        Imputation strategy (missingness-aware):
        1. For rolling/lag features with >5% missing, add a binary ``_missing`` indicator
           column so the model can distinguish "no prior data" from "low prior performance."
        2. Impute with column median (avoids distorting distribution), fallback 0.

        Per requirements, features with >5% missing are flagged in _check_missing_rate;
        we still impute so pipelines run.
        """
        if df.empty:
            return df
        # Replace inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Add binary missing indicators for rolling/lag features with meaningful missingness.
        # These let the model learn that early-season NaN != zero performance.
        n_rows = len(df)
        missing_indicator_cols = {}
        rolling_lag_tokens = ("_roll", "_lag", "_ewm", "_trend")
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if not any(tok in col for tok in rolling_lag_tokens):
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            miss_pct = n_missing / n_rows
            # Only add indicator when missingness is structurally meaningful (>2%)
            if miss_pct > 0.02:
                indicator_name = f"{col}_missing"
                if indicator_name not in df.columns:
                    missing_indicator_cols[indicator_name] = df[col].isna().astype(np.int8)

        if missing_indicator_cols:
            indicator_df = pd.DataFrame(missing_indicator_cols, index=df.index)
            df = pd.concat([df, indicator_df], axis=1)

        # Fill NaN: median per column (avoids distorting distribution), fallback 0
        # Re-fetch numeric cols since we may have added indicator columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in df.columns or not df[col].isna().any():
                continue
            med = df[col].median()
            if pd.isna(med):
                med = 0.0
            df[col] = df[col].fillna(med)
        return df
    
    def _update_feature_columns(self, df: pd.DataFrame):
        """Update list of feature columns.
        
        Excludes identifiers, raw targets, and leakage-prone columns. This
        guard is intentionally conservative to prevent model-output or target
        leakage in downstream training/evaluation pipelines.
        """
        exclude_cols = {
            "player_id", "name", "season", "week", "team", "opponent",
            "home_away", "position", "fantasy_points", "id", "created_at",
            "games_played",
        }
        # Also exclude any target columns that might have been added before
        # feature selection (e.g. during training pipeline)
        exclude_prefixes = ("target_", "actual_for_backtest", "predicted_", "baseline_")
        
        self.feature_columns = [
            col for col in df.columns 
            if col not in exclude_cols
            and not any(col.startswith(p) for p in exclude_prefixes)
            and df[col].dtype in [np.float64, np.int64, float, int]
        ]
        try:
            from src.utils.leakage import filter_feature_columns
            self.feature_columns = filter_feature_columns(self.feature_columns)
        except Exception:
            pass
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_columns
    
    def prepare_training_data(self, df: pd.DataFrame, 
                              target_weeks: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with features
            target_weeks: Number of weeks ahead to predict (1-18)
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = df.copy()
        
        # Create target: fantasy points N weeks ahead
        if target_weeks == 1:
            df["target"] = df.groupby("player_id")["fantasy_points"].shift(-1)
        else:
            # For multi-week prediction, use sum of next N weeks
            df["target"] = df.groupby("player_id")["fantasy_points"].transform(
                lambda x: x.shift(-1).rolling(window=target_weeks, min_periods=1).sum()
            )
        
        # Remove rows without target
        df = df.dropna(subset=["target"])
        
        # Get features
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in df.columns]
        
        X = df[available_features].copy()
        y = df["target"]
        
        # Clean inf values and fill NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X, y
    
    def prepare_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for making predictions.
        
        Args:
            df: DataFrame with player stats
            
        Returns:
            DataFrame with features ready for prediction
        """
        # Create features
        df = self.create_features(df, include_target=False)
        
        # Get most recent row per player
        latest = df.groupby("player_id").last().reset_index()
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        available_features = [c for c in feature_cols if c in latest.columns]
        
        return latest[["player_id", "name", "position", "team"] + available_features]


class PositionFeatureEngineer(FeatureEngineer):
    """Position-specific feature engineering."""
    
    def __init__(self, position: str):
        super().__init__()
        self.position = position
    
    def create_features(self, df: pd.DataFrame, 
                        include_target: bool = True) -> pd.DataFrame:
        """Create position-specific features."""
        # Filter to position
        df = df[df["position"] == self.position].copy()
        
        # Create base features
        df = super().create_features(df, include_target)
        
        # Add position-specific features
        if self.position == "QB":
            df = self._create_qb_features(df)
        elif self.position == "RB":
            df = self._create_rb_features(df)
        elif self.position == "WR":
            df = self._create_wr_features(df)
        elif self.position == "TE":
            df = self._create_te_features(df)
        
        return df
    
    def _create_qb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """QB-specific features per requirements: pass attempts, completion %, air yards,
        TD/INT ratio, rushing, sacks, time in pocket, deep ball, red zone efficiency."""
        new_cols = {}
        new_cols["rush_pct_of_plays"] = safe_divide(
            df["rushing_attempts"],
            df["passing_attempts"] + df["rushing_attempts"]
        ) * 100
        new_cols["yards_per_completion"] = safe_divide(
            df["passing_yards"], df["passing_completions"]
        )
        if "sacks" in df.columns:
            adj = df["passing_attempts"] + df["sacks"]
            new_cols["adj_pass_attempts"] = adj
            new_cols["sack_rate"] = safe_divide(df["sacks"], adj) * 100
        # Time in pocket (when available from PBP data)
        if "time_in_pocket" in df.columns:
            new_cols["avg_time_in_pocket"] = df["time_in_pocket"]
        # Red zone efficiency
        if "redzone_attempts" in df.columns and "redzone_completions" in df.columns:
            new_cols["rz_completion_pct"] = safe_divide(
                df["redzone_completions"], df["redzone_attempts"]
            ) * 100
        # Deep ball attempts (20+ yards)
        if "deep_pass_attempts" in df.columns:
            new_cols["deep_pass_rate"] = safe_divide(
                df["deep_pass_attempts"], df["passing_attempts"]
            ) * 100
            if "deep_pass_completions" in df.columns:
                new_cols["deep_pass_accuracy"] = safe_divide(
                    df["deep_pass_completions"], df["deep_pass_attempts"]
                ) * 100
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_rb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RB-specific features per requirements: rush attempts, receptions, red zone/goal line,
        snap share, route participation, yards after contact, broken tackles."""
        new_cols = {}
        new_cols["receiving_pct"] = safe_divide(
            df["receptions"], df["total_touches"]
        ) * 100
        new_cols["td_per_touch"] = safe_divide(
            df["total_tds"], df["total_touches"]
        )
        if "total_touches_roll5_mean" in df.columns:
            new_cols["workload_trend"] = df["total_touches"] - df["total_touches_roll5_mean"]
        # Goal line carries (inside 5-yard line)
        if "goal_line_carries" in df.columns:
            new_cols["goal_line_carry_rate"] = safe_divide(
                df["goal_line_carries"], df["rushing_attempts"]
            )
        elif "rush_inside_10" in df.columns:
            new_cols["goal_line_carry_rate"] = safe_divide(
                df["rush_inside_10"], df["rushing_attempts"]
            )
        # Red zone carries and targets
        if "redzone_carries" in df.columns:
            new_cols["rz_carry_rate"] = safe_divide(df["redzone_carries"], df["rushing_attempts"])
        if "redzone_targets" in df.columns:
            new_cols["rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_wr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """WR-specific features per requirements: targets, aDOT, YAC, target share,
        red zone targets, contested catch rate, route diversity, slot vs outside."""
        new_cols = {}
        # Target quality
        new_cols["yards_per_route"] = safe_divide(
            df["receiving_yards"], df.get("routes_run", df.get("snap_count", pd.Series(1, index=df.index)))
        )

        # Contested catch proxy (low catch rate but high yards)
        new_cols["contested_proxy"] = safe_divide(
            df["receiving_yards"], df["targets"]
        ) * (1 - df["catch_rate"] / 100)

        # Deep threat indicator
        new_cols["yards_per_catch"] = safe_divide(
            df["receiving_yards"], df["receptions"]
        )

        # Red zone target rate
        if "redzone_targets" in df.columns and "targets" in df.columns:
            new_cols["rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])

        # Slot vs outside alignment
        if "slot_pct" not in df.columns and "slot_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100
        if "outside_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["outside_pct"] = safe_divide(df["outside_snaps"], df["snap_count"]) * 100

        # Route tree diversity: number of different route types (if available)
        if "route_diversity_score" in df.columns:
            new_cols["route_tree_diversity"] = df["route_diversity_score"]

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
    
    def _create_te_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """TE-specific features per requirements: targets, receptions, receiving yards, TDs,
        air yards, aDOT, YAC, target share, red zone targets, contested catch rate,
        route tree diversity, slot vs. outside alignment."""
        new_cols = {}

        # Red zone specialist (high TD rate relative to volume)
        new_cols["rz_specialist_score"] = safe_divide(
            df["receiving_tds"], df["targets"]
        ) * 100

        # Seam threat (yards per target)
        new_cols["seam_threat"] = df.get("yards_per_target", pd.Series(0, index=df.index))

        # Red zone target rate
        if "redzone_targets" in df.columns and "targets" in df.columns:
            new_cols["te_rz_target_rate"] = safe_divide(df["redzone_targets"], df["targets"])

        # Contested catch proxy (low catch rate but high yards - same approach as WR)
        if "catch_rate" in df.columns and "receiving_yards" in df.columns and "targets" in df.columns:
            new_cols["te_contested_proxy"] = safe_divide(
                df["receiving_yards"], df["targets"]
            ) * (1 - df["catch_rate"].fillna(0) / 100)

        # aDOT / air yards per target (depth of target)
        if "air_yards" in df.columns and "targets" in df.columns:
            new_cols["te_adot"] = safe_divide(df["air_yards"], df["targets"])
        elif "air_yards_share" in df.columns:
            new_cols["te_adot"] = df["air_yards_share"]

        # Yards after catch (YAC)
        if "yards_after_catch" in df.columns:
            new_cols["te_yac"] = df["yards_after_catch"]
        elif "receiving_yards" in df.columns and "air_yards" in df.columns:
            new_cols["te_yac"] = (df["receiving_yards"] - df["air_yards"]).clip(lower=0)

        # Yards per route (efficiency per snap involvement)
        new_cols["te_yards_per_route"] = safe_divide(
            df["receiving_yards"],
            df.get("routes_run", df.get("snap_count", pd.Series(1, index=df.index)))
        )

        # Route participation rate
        if "routes_run" in df.columns and "snap_count" in df.columns:
            new_cols["te_route_participation"] = safe_divide(df["routes_run"], df["snap_count"]) * 100
        elif "route_participation" in df.columns:
            new_cols["te_route_participation"] = df["route_participation"]

        # Inline blocking rate proxy: snaps NOT running routes as % of total snaps
        if "routes_run" in df.columns and "snap_count" in df.columns:
            new_cols["te_inline_block_rate"] = (
                1.0 - safe_divide(df["routes_run"], df["snap_count"])
            ).clip(0, 1) * 100

        # Slot vs. outside alignment
        if "slot_snaps" in df.columns and "snap_count" in df.columns:
            new_cols["te_slot_pct"] = safe_divide(df["slot_snaps"], df["snap_count"]) * 100

        # Route tree diversity (when available)
        if "route_diversity_score" in df.columns:
            new_cols["te_route_diversity"] = df["route_diversity_score"]

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df


# NOTE: safe_divide is imported from src.utils.helpers - do not redefine here

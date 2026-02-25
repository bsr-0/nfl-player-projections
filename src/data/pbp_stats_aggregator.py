"""
Play-by-Play Stats Aggregator

Derives weekly player stats from nfl-data-py play-by-play data.
This is used when weekly stats are not directly available (e.g., current season).
"""
import os
import ssl
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RAW_DATA_DIR,
    PBP_ADVANCED_FEATURES_ENABLED,
    NEUTRAL_SCORE_DIFF,
    SHORT_YARDAGE_YDSTOGO,
    RED_ZONE_YARDLINE,
    GOAL_LINE_YARDLINE,
    TWO_MINUTE_SECONDS,
    PROE_FALLBACK_LG_NEUTRAL_RATE,
)


class PBPStatsAggregator:
    """Aggregate play-by-play data into weekly player stats."""
    
    def __init__(self):
        self.pbp_data = None
        self.snap_data = None
    
    def load_data(self, season: int) -> "PBPStatsAggregator":
        """Load PBP (required) and snap count (optional) data for a season. PBP-only path when snaps unavailable."""
        print(f"Loading play-by-play data for {season}...")
        self.pbp_data = nfl.import_pbp_data([season])
        print(f"  Loaded {len(self.pbp_data)} plays")
        
        self.snap_data = None
        try:
            snap_df = nfl.import_snap_counts([season])
            if snap_df is not None and not snap_df.empty:
                self.snap_data = snap_df
                print(f"  Loaded {len(self.snap_data)} snap records")
            else:
                print(f"  Snap counts for {season}: not available (using position inference)")
        except Exception as e:
            print(f"  Snap counts for {season}: not available ({e}); using position inference")
        
        return self

    def _prepare_pbp_features(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Prepare play-level flags and standard columns used by advanced aggregation."""
        if pbp is None or pbp.empty:
            return pd.DataFrame()
        df = pbp.copy()

        # Ensure required columns exist
        for col in [
            "score_differential", "qtr", "ydstogo", "yardline_100",
            "game_seconds_remaining", "half_seconds_remaining", "wp",
            "epa", "wpa", "success", "down"
        ]:
            if col not in df.columns:
                df[col] = np.nan

        score_diff = pd.to_numeric(df["score_differential"], errors="coerce").fillna(0.0)
        qtr = pd.to_numeric(df["qtr"], errors="coerce").fillna(4.0)
        ydstogo = pd.to_numeric(df["ydstogo"], errors="coerce").fillna(99.0)
        yardline = pd.to_numeric(df["yardline_100"], errors="coerce").fillna(99.0)
        down = pd.to_numeric(df["down"], errors="coerce").fillna(0.0)

        # Base advanced columns
        df["epa"] = pd.to_numeric(df["epa"], errors="coerce").fillna(0.0)
        df["wpa"] = pd.to_numeric(df["wpa"], errors="coerce").fillna(0.0)
        df["success"] = pd.to_numeric(df["success"], errors="coerce")

        # Situational flags
        df["is_neutral"] = (score_diff.abs() <= NEUTRAL_SCORE_DIFF) & (qtr <= 3)
        df["is_short_yardage"] = ydstogo <= SHORT_YARDAGE_YDSTOGO
        df["is_redzone"] = yardline <= RED_ZONE_YARDLINE
        df["is_goal_line"] = yardline <= GOAL_LINE_YARDLINE
        df["is_third_down"] = down == 3

        # Two-minute window (prefer half_seconds_remaining when available)
        half_sec = (
            pd.to_numeric(df["half_seconds_remaining"], errors="coerce")
            if "half_seconds_remaining" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        gsr = (
            pd.to_numeric(df["game_seconds_remaining"], errors="coerce")
            if "game_seconds_remaining" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        sec_remaining = half_sec if half_sec.notna().any() else gsr
        df["is_two_minute"] = (qtr.isin([2, 4])) & (sec_remaining <= TWO_MINUTE_SECONDS)

        # High leverage: use WP when available, otherwise late & close proxy
        wp = (
            pd.to_numeric(df["wp"], errors="coerce")
            if "wp" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        if wp.notna().any():
            hl = (wp >= 0.2) & (wp <= 0.8)
            fallback = (score_diff.abs() <= NEUTRAL_SCORE_DIFF) & (qtr >= 4)
            df["is_high_leverage"] = np.where(wp.notna(), hl, fallback)
        else:
            df["is_high_leverage"] = (score_diff.abs() <= NEUTRAL_SCORE_DIFF) & (qtr >= 4)

        # Cast to int for aggregation speed
        for col in ["is_neutral", "is_short_yardage", "is_redzone", "is_goal_line",
                    "is_two_minute", "is_high_leverage", "is_third_down"]:
            df[col] = df[col].astype(int)

        return df
    
    def aggregate_passing_stats(self, pbp: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate passing stats by player/week."""
        pbp = pbp if pbp is not None else self.pbp_data
        if pbp is None or pbp.empty:
            return pd.DataFrame()

        # Filter to pass plays
        pass_plays = pbp[pbp['play_type'] == 'pass'].copy()
        if 'passer_player_id' in pass_plays.columns:
            pass_plays = pass_plays[pass_plays['passer_player_id'].notna()]
        
        # Aggregate by passer
        agg = {
            'passing_yards': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum',
            'complete_pass': 'sum',
            'play_id': 'count',  # attempts
        }
        if 'epa' in pass_plays.columns:
            agg['epa'] = 'sum'
        if 'wpa' in pass_plays.columns:
            agg['wpa'] = 'sum'
        if 'success' in pass_plays.columns:
            agg['success'] = 'mean'

        passing = pass_plays.groupby(
            ['season', 'week', 'passer_player_id', 'passer_player_name', 'posteam']
        ).agg(agg).reset_index()
        
        passing = passing.rename(columns={
            'passer_player_id': 'player_id',
            'passer_player_name': 'name',
            'posteam': 'team',
            'play_id': 'passing_attempts',
            'complete_pass': 'completions',
            'pass_touchdown': 'passing_tds',
            'interception': 'interceptions',
            'epa': 'pass_epa',
            'wpa': 'pass_wpa',
            'success': 'pass_success_rate',
        })
        if 'passing_attempts' in passing.columns:
            passing['pass_plays'] = passing['passing_attempts']
        else:
            passing['pass_plays'] = 0
        
        passing['position'] = 'QB'
        return passing
    
    def aggregate_rushing_stats(self, pbp: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate rushing stats by player/week."""
        pbp = pbp if pbp is not None else self.pbp_data
        if pbp is None or pbp.empty:
            return pd.DataFrame()
        
        # Filter to rush plays
        rush_plays = pbp[pbp['play_type'] == 'run'].copy()
        if 'rusher_player_id' in rush_plays.columns:
            rush_plays = rush_plays[rush_plays['rusher_player_id'].notna()]

        # Goal-line / high-value rushing opportunities
        # nflverse convention: yardline_100 is distance to opponent endzone.
        # inside 10 => yardline_100 <= 10, inside 5 => yardline_100 <= 5.
        if 'yardline_100' in rush_plays.columns:
            rush_plays['is_rush_inside_10'] = (rush_plays['yardline_100'] <= 10).astype(int)
            rush_plays['is_rush_inside_5'] = (rush_plays['yardline_100'] <= 5).astype(int)
        else:
            rush_plays['is_rush_inside_10'] = 0
            rush_plays['is_rush_inside_5'] = 0

        # Aggregate by rusher
        agg = {
            'rushing_yards': 'sum',
            'rush_touchdown': 'sum',
            'play_id': 'count',  # attempts
            'is_rush_inside_10': 'sum',
            'is_rush_inside_5': 'sum',
        }
        for col in ['epa', 'wpa']:
            if col in rush_plays.columns:
                agg[col] = 'sum'
        if 'success' in rush_plays.columns:
            agg['success'] = 'mean'
        if 'is_neutral' in rush_plays.columns:
            agg['is_neutral'] = 'sum'
        if 'is_short_yardage' in rush_plays.columns:
            agg['is_short_yardage'] = 'sum'
        if 'is_goal_line' in rush_plays.columns:
            agg['is_goal_line'] = 'sum'
        if 'is_high_leverage' in rush_plays.columns:
            agg['is_high_leverage'] = 'sum'

        rushing = rush_plays.groupby(
            ['season', 'week', 'rusher_player_id', 'rusher_player_name', 'posteam']
        ).agg(agg).reset_index()
        
        rushing = rushing.rename(columns={
            'rusher_player_id': 'player_id',
            'rusher_player_name': 'name',
            'posteam': 'team',
            'play_id': 'rushing_attempts',
            'rush_touchdown': 'rushing_tds',
            'is_rush_inside_10': 'rush_inside_10',
            'is_rush_inside_5': 'rush_inside_5',
            'epa': 'rush_epa',
            'wpa': 'rush_wpa',
            'success': 'rush_success_rate',
            'is_neutral': 'neutral_rushes',
            'is_short_yardage': 'short_yardage_rushes',
            'is_goal_line': 'goal_line_rushes',
            'is_high_leverage': 'high_leverage_rushes',
        })
        if 'rushing_attempts' in rushing.columns:
            rushing['rush_plays'] = rushing['rushing_attempts']
        else:
            rushing['rush_plays'] = 0
        
        return rushing
    
    def aggregate_receiving_stats(self, pbp: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate receiving stats by player/week."""
        pbp = pbp if pbp is not None else self.pbp_data
        if pbp is None or pbp.empty:
            return pd.DataFrame()
        
        # Filter to pass plays with a receiver
        rec_plays = pbp[(pbp['play_type'] == 'pass') & (pbp['receiver_player_id'].notna())].copy()

        # Air yards + deep target proxies (15+ air yards)
        if 'air_yards' in rec_plays.columns:
            rec_plays['air_yards_filled'] = rec_plays['air_yards'].fillna(0)
            rec_plays['is_target_15_plus'] = (rec_plays['air_yards_filled'] >= 15).astype(int)
        else:
            rec_plays['air_yards_filled'] = 0.0
            rec_plays['is_target_15_plus'] = 0
        
        # Aggregate by receiver
        agg = {
            'receiving_yards': 'sum',
            'complete_pass': 'sum',  # receptions
            'pass_touchdown': 'sum',
            'play_id': 'count',  # targets
            'air_yards_filled': 'sum',
            'is_target_15_plus': 'sum',
        }
        for col in ['epa', 'wpa']:
            if col in rec_plays.columns:
                agg[col] = 'sum'
        if 'success' in rec_plays.columns:
            agg['success'] = 'mean'
        if 'is_neutral' in rec_plays.columns:
            agg['is_neutral'] = 'sum'
        if 'is_third_down' in rec_plays.columns:
            agg['is_third_down'] = 'sum'
        if 'is_redzone' in rec_plays.columns:
            agg['is_redzone'] = 'sum'
        if 'is_goal_line' in rec_plays.columns:
            agg['is_goal_line'] = 'sum'
        if 'is_two_minute' in rec_plays.columns:
            agg['is_two_minute'] = 'sum'
        if 'is_high_leverage' in rec_plays.columns:
            agg['is_high_leverage'] = 'sum'

        receiving = rec_plays.groupby(
            ['season', 'week', 'receiver_player_id', 'receiver_player_name', 'posteam']
        ).agg(agg).reset_index()
        
        receiving = receiving.rename(columns={
            'receiver_player_id': 'player_id',
            'receiver_player_name': 'name',
            'posteam': 'team',
            'play_id': 'targets',
            'complete_pass': 'receptions',
            'pass_touchdown': 'receiving_tds',
            'air_yards_filled': 'air_yards',
            'is_target_15_plus': 'targets_15_plus',
            'epa': 'recv_epa',
            'wpa': 'recv_wpa',
            'success': 'recv_success_rate',
            'is_neutral': 'neutral_targets',
            'is_third_down': 'third_down_targets',
            'is_redzone': 'redzone_targets',
            'is_goal_line': 'goal_line_targets',
            'is_two_minute': 'two_minute_targets',
            'is_high_leverage': 'high_leverage_targets',
        })
        if 'targets' in receiving.columns:
            receiving['recv_targets'] = receiving['targets']
        else:
            receiving['recv_targets'] = 0
        
        return receiving
    
    def merge_with_snaps(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stats with snap count data; set snap_count (offense_snaps) and team_snaps for utilization_score/DB."""
        if self.snap_data is None:
            return stats_df
        
        snap_pos = self.snap_data[['season', 'week', 'player', 'team', 'position', 'offense_snaps', 'offense_pct']].copy()
        snap_pos = snap_pos.rename(columns={'player': 'name'})
        snap_pos['snap_count'] = snap_pos['offense_snaps'].fillna(0).astype(int)
        team_snaps = (
            snap_pos.groupby(['season', 'week', 'team'], as_index=False)['offense_snaps']
            .sum()
            .rename(columns={'offense_snaps': 'team_snaps'})
        )
        team_snaps['team_snaps'] = team_snaps['team_snaps'].fillna(0).astype(int)
        snap_pos = snap_pos.merge(team_snaps, on=['season', 'week', 'team'], how='left')
        snap_pos['team_snaps'] = snap_pos['team_snaps'].fillna(0).astype(int)
        merged = stats_df.merge(
            snap_pos[['season', 'week', 'name', 'team', 'position', 'snap_count', 'team_snaps', 'offense_pct']],
            on=['season', 'week', 'name', 'team'],
            how='left'
        )
        if 'snap_count' not in merged.columns:
            merged['snap_count'] = 0
        if 'team_snaps' not in merged.columns:
            merged['team_snaps'] = 0
        merged['snap_count'] = merged['snap_count'].fillna(0).astype(int)
        merged['team_snaps'] = merged['team_snaps'].fillna(0).astype(int)
        merged['snap_share'] = np.where(
            merged['team_snaps'] > 0,
            merged['snap_count'] / merged['team_snaps'],
            0.0
        )
        return merged
    
    def calculate_fantasy_points(self, df: pd.DataFrame, ppr: float = 1.0) -> pd.DataFrame:
        """Calculate fantasy points (PPR scoring)."""
        df = df.copy()
        
        # Initialize
        df['fantasy_points'] = 0.0
        
        # Passing: 0.04 per yard, 4 per TD, -2 per INT
        if 'passing_yards' in df.columns:
            df['fantasy_points'] += df['passing_yards'].fillna(0) * 0.04
        if 'passing_tds' in df.columns:
            df['fantasy_points'] += df['passing_tds'].fillna(0) * 4
        if 'interceptions' in df.columns:
            df['fantasy_points'] -= df['interceptions'].fillna(0) * 2
        
        # Rushing: 0.1 per yard, 6 per TD
        if 'rushing_yards' in df.columns:
            df['fantasy_points'] += df['rushing_yards'].fillna(0) * 0.1
        if 'rushing_tds' in df.columns:
            df['fantasy_points'] += df['rushing_tds'].fillna(0) * 6
        
        # Receiving: 0.1 per yard, 6 per TD, PPR per reception
        if 'receiving_yards' in df.columns:
            df['fantasy_points'] += df['receiving_yards'].fillna(0) * 0.1
        if 'receiving_tds' in df.columns:
            df['fantasy_points'] += df['receiving_tds'].fillna(0) * 6
        if 'receptions' in df.columns:
            df['fantasy_points'] += df['receptions'].fillna(0) * ppr
        
        df['fantasy_points'] = df['fantasy_points'].round(1)
        
        return df
    
    def aggregate_all_stats(self, season: int = None,
                            include_advanced: bool = True,
                            pbp_df: pd.DataFrame = None,
                            snap_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Aggregate all player stats from PBP data for a season or provided DataFrame.

        Returns DataFrame with weekly player stats similar to nfl.import_weekly_data()
        """
        if pbp_df is not None:
            self.pbp_data = pbp_df
            if snap_df is not None:
                self.snap_data = snap_df
        else:
            if season is None:
                raise ValueError("season is required when pbp_df is not provided")
            self.load_data(season)
        if self.pbp_data is None or self.pbp_data.empty:
            return pd.DataFrame()

        # Prepare PBP features
        pbp = self._prepare_pbp_features(self.pbp_data) if include_advanced else self.pbp_data

        # Get passing stats
        passing = self.aggregate_passing_stats(pbp)
        
        # Get rushing stats
        rushing = self.aggregate_rushing_stats(pbp)
        
        # Get receiving stats
        receiving = self.aggregate_receiving_stats(pbp)
        
        # Merge rushing and receiving for skill players
        skill = rushing.merge(
            receiving,
            on=['season', 'week', 'player_id', 'name', 'team'],
            how='outer'
        )

        # Derived situational touch composites
        gl_rush = skill.get('goal_line_rushes', pd.Series(0, index=skill.index))
        gl_tgt = skill.get('goal_line_targets', pd.Series(0, index=skill.index))
        skill['goal_line_touches'] = gl_rush.fillna(0) + gl_tgt.fillna(0)
        hl_rush = skill.get('high_leverage_rushes', pd.Series(0, index=skill.index))
        hl_tgt = skill.get('high_leverage_targets', pd.Series(0, index=skill.index))
        skill['high_leverage_touches'] = hl_rush.fillna(0) + hl_tgt.fillna(0)

        # Combine with passing (QBs)
        all_stats = pd.concat([passing, skill], ignore_index=True)
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = [
            'passing_yards', 'passing_tds', 'interceptions', 'completions', 'passing_attempts',
            'rushing_yards', 'rushing_tds', 'rushing_attempts',
            'receiving_yards', 'receiving_tds', 'receptions', 'targets',
            'pass_plays', 'rush_plays', 'recv_targets',
            'pass_epa', 'rush_epa', 'recv_epa',
            'pass_wpa', 'rush_wpa', 'recv_wpa',
            'pass_success_rate', 'rush_success_rate', 'recv_success_rate',
            'neutral_targets', 'neutral_rushes', 'third_down_targets', 'short_yardage_rushes',
            'redzone_targets', 'goal_line_touches', 'two_minute_targets', 'high_leverage_touches',
        ]
        for col in numeric_cols:
            if col in all_stats.columns:
                all_stats[col] = all_stats[col].fillna(0)
        
        # Merge with snap data for positions
        all_stats = self.merge_with_snaps(all_stats)
        
        # Infer position if not from snaps
        if 'position' not in all_stats.columns or all_stats['position'].isna().any():
            all_stats['position'] = all_stats.apply(self._infer_position, axis=1)
        
        # Calculate fantasy points
        all_stats = self.calculate_fantasy_points(all_stats)
        
        # Filter to fantasy-relevant positions
        all_stats = all_stats[all_stats['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB', 'HB'])]
        all_stats.loc[all_stats['position'].isin(['FB', 'HB']), 'position'] = 'RB'
        
        return all_stats
    
    def _infer_position(self, row) -> str:
        """Infer position from stats if not available."""
        if pd.notna(row.get('position')):
            return row['position']
        
        # If has passing attempts, likely QB
        if row.get('passing_attempts', 0) > 0:
            return 'QB'
        
        # If has rushing attempts but few/no targets, likely RB
        if row.get('rushing_attempts', 0) > row.get('targets', 0):
            return 'RB'
        
        # Default to WR for receivers
        return 'WR'


def _drive_result_to_points(result) -> int:
    """Map drive_result text to points for drive-level scoring."""
    if result is None or (isinstance(result, float) and np.isnan(result)):
        return 0
    res = str(result).lower()
    if "touchdown" in res:
        return 7
    if "field goal" in res:
        return 3
    if "safety" in res:
        return 2
    return 0


def _compute_league_neutral_pass_rate(team_week: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-safe league neutral pass rate (expanding, shifted) by season/week."""
    if team_week.empty:
        return pd.DataFrame(columns=["season", "week", "neutral_pass_rate_lg"])
    league_week = team_week.groupby(["season", "week"], as_index=False)[
        ["neutral_pass_plays", "neutral_run_plays"]
    ].sum()
    denom = league_week["neutral_pass_plays"] + league_week["neutral_run_plays"]
    league_week["neutral_pass_rate"] = np.where(
        denom > 0, league_week["neutral_pass_plays"] / denom, np.nan
    )
    league_week = league_week.sort_values(["season", "week"])
    league_week["neutral_pass_rate_lg"] = league_week.groupby("season")[
        "neutral_pass_rate"
    ].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    return league_week[["season", "week", "neutral_pass_rate_lg"]]

    def aggregate_team_stats(self, pbp: pd.DataFrame = None) -> pd.DataFrame:
        """Aggregate team-level neutral pass rate and drive metrics from PBP."""
        pbp = pbp if pbp is not None else self.pbp_data
        if pbp is None or pbp.empty:
            return pd.DataFrame()

        # Ensure advanced flags exist
        pbp = self._prepare_pbp_features(pbp)
        if "posteam" not in pbp.columns:
            return pd.DataFrame()

        plays = pbp[pbp["posteam"].notna()].copy()
        if plays.empty:
            return pd.DataFrame()

        is_pass = plays["play_type"] == "pass"
        is_run = plays["play_type"] == "run"
        plays["is_pass"] = is_pass.astype(int)
        plays["is_run"] = is_run.astype(int)

        # Exclude spikes/kneels from neutral pass rate if available
        spike = plays.get("qb_spike", 0)
        kneel = plays.get("qb_kneel", 0)
        if isinstance(spike, pd.Series):
            spike = spike.fillna(0)
        if isinstance(kneel, pd.Series):
            kneel = kneel.fillna(0)
        exclude = (spike == 1) | (kneel == 1)

        plays["neutral_pass_play"] = (
            (plays["is_neutral"] == 1) & is_pass & ~exclude
        ).astype(int)
        plays["neutral_run_play"] = (
            (plays["is_neutral"] == 1) & is_run & ~exclude
        ).astype(int)

        agg = {
            "neutral_pass_play": "sum",
            "neutral_run_play": "sum",
            "is_pass": "sum",
            "is_run": "sum",
        }
        if "passing_yards" in plays.columns:
            agg["passing_yards"] = "sum"
        if "rushing_yards" in plays.columns:
            agg["rushing_yards"] = "sum"
        if "interception" in plays.columns:
            agg["interception"] = "sum"
        if "fumble_lost" in plays.columns:
            agg["fumble_lost"] = "sum"

        team_week = plays.groupby(["season", "week", "posteam"]).agg(agg).reset_index()
        team_week = team_week.rename(columns={
            "posteam": "team",
            "neutral_pass_play": "neutral_pass_plays",
            "neutral_run_play": "neutral_run_plays",
            "is_pass": "pass_attempts",
            "is_run": "rush_attempts",
        })

        # Basic totals
        team_week["total_plays"] = team_week["pass_attempts"] + team_week["rush_attempts"]
        if "passing_yards" in team_week.columns:
            team_week["passing_yards"] = team_week["passing_yards"].fillna(0)
        else:
            team_week["passing_yards"] = 0
        if "rushing_yards" in team_week.columns:
            team_week["rushing_yards"] = team_week["rushing_yards"].fillna(0)
        else:
            team_week["rushing_yards"] = 0
        team_week["total_yards"] = team_week["passing_yards"] + team_week["rushing_yards"]
        if "interception" in team_week.columns:
            interceptions = team_week["interception"].fillna(0)
        else:
            interceptions = 0
        if "fumble_lost" in team_week.columns:
            fumbles = team_week["fumble_lost"].fillna(0)
        else:
            fumbles = 0
        turnovers = interceptions + fumbles
        team_week["turnovers"] = turnovers

        denom = team_week["neutral_pass_plays"] + team_week["neutral_run_plays"]
        team_week["neutral_pass_rate"] = np.where(denom > 0, team_week["neutral_pass_plays"] / denom, np.nan)

        # Opponent (mode of defteam)
        if "defteam" in plays.columns:
            opp = plays.groupby(["season", "week", "posteam"])["defteam"].agg(
                lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) else None
            ).reset_index()
            opp = opp.rename(columns={"posteam": "team", "defteam": "opponent"})
            team_week = team_week.merge(opp, on=["season", "week", "team"], how="left")
        else:
            team_week["opponent"] = None

        # League neutral pass rate (leak-safe)
        league_rates = _compute_league_neutral_pass_rate(team_week)
        team_week = team_week.merge(league_rates, on=["season", "week"], how="left")
        team_week["neutral_pass_rate_lg"] = team_week["neutral_pass_rate_lg"].fillna(PROE_FALLBACK_LG_NEUTRAL_RATE)
        team_week["neutral_pass_rate"] = team_week["neutral_pass_rate"].fillna(team_week["neutral_pass_rate_lg"])
        team_week["neutral_pass_rate_oe"] = team_week["neutral_pass_rate"] - team_week["neutral_pass_rate_lg"]

        # Drive metrics
        drive_col = "drive" if "drive" in plays.columns else ("drive_id" if "drive_id" in plays.columns else None)
        if drive_col:
            drive_df = plays[plays[drive_col].notna()].copy()
            if "drive_result" in drive_df.columns:
                drive_df["drive_points"] = drive_df["drive_result"].map(_drive_result_to_points).fillna(0)
            else:
                drive_df["drive_points"] = 0
            drive_metrics = drive_df.groupby(["season", "week", "posteam", drive_col]).agg(
                drive_epa=("epa", "sum"),
                drive_points=("drive_points", "max"),
            ).reset_index()
            drive_metrics["drive_success"] = (
                (drive_metrics["drive_epa"] > 0) | (drive_metrics["drive_points"] > 0)
            ).astype(int)
            team_drive = drive_metrics.groupby(["season", "week", "posteam"]).agg(
                drive_count=("drive_epa", "size"),
                avg_drive_epa=("drive_epa", "mean"),
                drive_success_rate=("drive_success", "mean"),
                points_per_drive=("drive_points", "mean"),
            ).reset_index().rename(columns={"posteam": "team"})
            team_week = team_week.merge(team_drive, on=["season", "week", "team"], how="left")
        else:
            team_week["drive_count"] = 0
            team_week["avg_drive_epa"] = 0.0
            team_week["drive_success_rate"] = 0.0
            team_week["points_per_drive"] = 0.0

        # Pace (seconds per play) by team/week
        if {"game_id", "play_id", "game_seconds_remaining"}.issubset(plays.columns):
            pace_df = plays[["season", "week", "posteam", "game_id", "play_id", "game_seconds_remaining"]].copy()
            pace_df = pace_df.sort_values(["game_id", "play_id"])
            pace_df["sec_per_play"] = pace_df.groupby(["game_id", "posteam"])["game_seconds_remaining"].diff() * -1
            pace_df["sec_per_play"] = pace_df["sec_per_play"].clip(lower=0, upper=90)
            pace_team = pace_df.groupby(["season", "week", "posteam"])["sec_per_play"].mean().reset_index()
            pace_team = pace_team.rename(columns={"posteam": "team", "sec_per_play": "pace_sec_per_play"})
            team_week = team_week.merge(pace_team, on=["season", "week", "team"], how="left")
        else:
            team_week["pace_sec_per_play"] = np.nan

        # Ensure required columns exist
        for col in [
            "drive_count", "drive_success_rate", "avg_drive_epa", "points_per_drive",
            "pace_sec_per_play"
        ]:
            if col not in team_week.columns:
                team_week[col] = 0.0

        return team_week


def _ensure_store_weekly_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has all columns required by NFLDataLoader._store_weekly_data.

    Adds defaults for opponent, home_away, fumbles_lost; renames completions -> passing_completions;
    ensures all stat columns exist so PBP-only output is valid for DB insert.
    """
    if df.empty:
        return df
    df = df.copy()
    if "completions" in df.columns and "passing_completions" not in df.columns:
        df["passing_completions"] = df["completions"]
    for col, default in [("opponent", ""), ("home_away", "unknown"), ("fumbles_lost", 0)]:
        if col not in df.columns:
            df[col] = default
    if "games_played" not in df.columns:
        df["games_played"] = 1
    stat_cols = [
        "passing_attempts", "passing_completions", "passing_yards", "passing_tds",
        "interceptions", "rushing_attempts", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # High-value touch source columns (optional; default to 0 when unavailable)
    for col, default in [
        ("rush_inside_10", 0),
        ("rush_inside_5", 0),
        ("targets_15_plus", 0),
        ("air_yards", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    # Advanced PBP columns (optional; default to 0)
    adv_defaults = {
        "pass_plays": 0,
        "rush_plays": 0,
        "recv_targets": 0,
        "pass_epa": 0.0,
        "rush_epa": 0.0,
        "recv_epa": 0.0,
        "pass_wpa": 0.0,
        "rush_wpa": 0.0,
        "recv_wpa": 0.0,
        "pass_success_rate": 0.0,
        "rush_success_rate": 0.0,
        "recv_success_rate": 0.0,
        "neutral_targets": 0,
        "neutral_rushes": 0,
        "third_down_targets": 0,
        "short_yardage_rushes": 0,
        "redzone_targets": 0,
        "goal_line_touches": 0,
        "two_minute_targets": 0,
        "high_leverage_touches": 0,
    }
    if "pass_plays" not in df.columns and "passing_attempts" in df.columns:
        df["pass_plays"] = df["passing_attempts"].fillna(0)
    if "rush_plays" not in df.columns and "rushing_attempts" in df.columns:
        df["rush_plays"] = df["rushing_attempts"].fillna(0)
    if "recv_targets" not in df.columns and "targets" in df.columns:
        df["recv_targets"] = df["targets"].fillna(0)
    for col, default in adv_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    return df


def _pbp_cache_paths(season: int) -> tuple:
    """Return cache paths for player and team advanced PBP aggregates."""
    player_cache = RAW_DATA_DIR / f"pbp_advanced_{season}.parquet"
    team_cache = RAW_DATA_DIR / f"pbp_team_advanced_{season}.parquet"
    return player_cache, team_cache


def get_pbp_advanced_stats(season: int,
                           include_team: bool = True,
                           include_advanced: bool = None,
                           use_cache: bool = True) -> tuple:
    """
    Load or compute advanced PBP-derived player/team stats with caching.
    Returns (player_weekly_df, team_weekly_df).
    """
    if include_advanced is None:
        include_advanced = PBP_ADVANCED_FEATURES_ENABLED

    player_cache, team_cache = _pbp_cache_paths(season)
    player_df = None
    team_df = None

    if use_cache and player_cache.exists():
        try:
            player_df = pd.read_parquet(player_cache)
        except Exception:
            player_df = None
    if include_team and use_cache and team_cache.exists():
        try:
            team_df = pd.read_parquet(team_cache)
        except Exception:
            team_df = None

    if player_df is not None and (not include_team or team_df is not None):
        return player_df, team_df

    aggregator = PBPStatsAggregator()
    aggregator.load_data(season)
    if player_df is None:
        player_df = aggregator.aggregate_all_stats(
            pbp_df=aggregator.pbp_data,
            snap_df=aggregator.snap_data,
            include_advanced=include_advanced,
        )
        if not player_df.empty:
            player_df = _ensure_store_weekly_schema(player_df)
    if include_team and team_df is None:
        team_df = aggregator.aggregate_team_stats(aggregator.pbp_data)

    if use_cache:
        if player_df is not None and not player_df.empty:
            try:
                player_df.to_parquet(player_cache, index=False)
            except Exception:
                pass
        if include_team and team_df is not None and not team_df.empty:
            try:
                team_df.to_parquet(team_cache, index=False)
            except Exception:
                pass

    return player_df, team_df


def get_weekly_stats_from_pbp(season: int,
                              include_advanced: bool = None,
                              use_cache: bool = True) -> pd.DataFrame:
    """
    Return weekly player stats from PBP data in same shape as nfl.import_weekly_data.

    Used by nfl_data_loader and auto_refresh when weekly data is missing or incomplete.
    Column names and defaults match NFLDataLoader._store_weekly_data. Works for current
    season (current/next) when import_weekly_data is empty; snap counts are optional.
    """
    stats, _ = get_pbp_advanced_stats(
        season,
        include_team=False,
        include_advanced=include_advanced,
        use_cache=use_cache,
    )
    if stats is None or stats.empty:
        return pd.DataFrame()
    return _ensure_store_weekly_schema(stats)


def get_team_stats_from_pbp(season: int,
                            use_cache: bool = True) -> pd.DataFrame:
    """Return team-level advanced stats from PBP data (neutral pass rate, drive metrics)."""
    _, team_stats = get_pbp_advanced_stats(
        season,
        include_team=True,
        include_advanced=True,
        use_cache=use_cache,
    )
    if team_stats is None or team_stats.empty:
        return pd.DataFrame()
    return team_stats


def load_current_season_stats_from_pbp() -> pd.DataFrame:
    """
    Load current NFL season stats from PBP data (convenience wrapper).
    
    Returns DataFrame compatible with nfl_data_loader schema.
    """
    from src.utils.nfl_calendar import get_current_nfl_season
    season = get_current_nfl_season()
    stats = get_weekly_stats_from_pbp(season)
    if not stats.empty:
        print(f"\n{season} Season Stats Summary (from PBP):")
        print(f"  Total player-weeks: {len(stats)}")
        print(f"  Weeks: {sorted(stats['week'].unique())}")
        print(f"  Positions: {stats['position'].value_counts().to_dict()}")
    return stats


if __name__ == "__main__":
    stats = load_current_season_stats_from_pbp()
    
    # Show top players from latest week
    latest_week = stats['week'].max()
    latest = stats[stats['week'] == latest_week].sort_values('fantasy_points', ascending=False)
    
    print(f"\nTop 10 Players Week {latest_week}:")
    print(latest[['name', 'team', 'position', 'fantasy_points']].head(10).to_string())

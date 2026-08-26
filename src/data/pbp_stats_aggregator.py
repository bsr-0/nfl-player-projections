"""
Play-by-Play Stats Aggregator

Derives weekly player stats from nfl-data-py play-by-play data.
This is used when weekly stats are not directly available (e.g., current season).
"""
import os
import ssl

try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    certifi = None

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.entity_resolver import resolver
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


def compute_team_snaps(snaps: pd.DataFrame) -> pd.DataFrame:
    """Team offensive play count per (season, week, team) from per-player snaps.

    NOT a sum over players: ~11 offensive players are each credited with most
    of the same ~55-75 plays, so summing inflates the total roughly 11x and
    crushes every derived snap_share feature (GAPS.md 2026-08-07, "team_snaps
    inflation"). Prefer the total implied by each player's own share --
    offense_snaps / offense_pct, nflverse's own verified per-player fraction --
    taking the median across the team-week to shrug off rounding noise. Fall
    back to max(offense_snaps), the every-down player, when offense_pct is
    absent.

    Extracted so the ingest path and any backfill share one implementation;
    a re-derived copy is how a fixed formula and a stale table drift apart.
    """
    if 'offense_pct' in snaps.columns:
        implied = np.where(
            snaps['offense_pct'] > 0,
            snaps['offense_snaps'] / snaps['offense_pct'],
            np.nan,
        )
        team_snaps = (
            pd.DataFrame({
                'season': snaps['season'], 'week': snaps['week'],
                'team': snaps['team'], '_implied': implied,
            })
            .groupby(['season', 'week', 'team'], as_index=False)['_implied']
            .median()
            .rename(columns={'_implied': 'team_snaps'})
        )
    else:
        team_snaps = pd.DataFrame(columns=['season', 'week', 'team', 'team_snaps'])
    max_snaps = (
        snaps.groupby(['season', 'week', 'team'], as_index=False)['offense_snaps']
        .max()
        .rename(columns={'offense_snaps': 'team_snaps_max'})
    )
    team_snaps = max_snaps.merge(team_snaps, on=['season', 'week', 'team'], how='left')
    team_snaps['team_snaps'] = team_snaps['team_snaps'].fillna(team_snaps['team_snaps_max'])
    team_snaps = team_snaps.drop(columns=['team_snaps_max'])
    # Nullable: a team-week with no usable snap rows has an UNKNOWN play
    # count, and 0 is never a true one -- every team runs plays. The old
    # fillna(0) made that impossible value look measured.
    team_snaps['team_snaps'] = team_snaps['team_snaps'].round().astype('Int64')
    return team_snaps


class PBPStatsAggregator:
    """Aggregate play-by-play data into weekly player stats."""
    
    def __init__(self):
        self.pbp_data = None
        self.snap_data = None
    
    def load_data(self, season: int) -> "PBPStatsAggregator":
        """Load PBP (required) and snap count (optional) data for a season. PBP-only path when snaps unavailable."""
        import nfl_data_py as nfl
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
        resolved = resolver.build_keys(passing, source="pbp_passing", name_col="name")
        passing = resolved.dataframe
        passing['player_id'] = passing['canonical_player_id'].where(passing['canonical_player_id'] != '', passing['player_id'])
        passing['team'] = passing['team_norm'].where(passing['team_norm'] != '', passing['team'])
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
        
        resolved = resolver.build_keys(rushing, source="pbp_rushing", name_col="name")
        rushing = resolved.dataframe
        rushing['player_id'] = rushing['canonical_player_id'].where(rushing['canonical_player_id'] != '', rushing['player_id'])
        rushing['team'] = rushing['team_norm'].where(rushing['team_norm'] != '', rushing['team'])
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
        
        resolved = resolver.build_keys(receiving, source="pbp_receiving", name_col="name")
        receiving = resolved.dataframe
        receiving['player_id'] = receiving['canonical_player_id'].where(receiving['canonical_player_id'] != '', receiving['player_id'])
        receiving['team'] = receiving['team_norm'].where(receiving['team_norm'] != '', receiving['team'])
        return receiving
    
    def merge_with_snaps(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stats with snap count data; set snap_count (offense_snaps) and team_snaps for utilization_score/DB.

        Uses pfr_player_id → GSIS mapping as primary join key. Falls back to
        name+team when the mapping is unavailable.
        """
        if self.snap_data is None:
            return stats_df

        required = {'season', 'week', 'player', 'team', 'offense_snaps'}
        available = set(self.snap_data.columns)
        if not required.issubset(available):
            print(f"  WARNING: snap data missing columns {required - available}; skipping merge")
            return stats_df

        keep_cols = ['season', 'week', 'player', 'team', 'offense_snaps']
        if 'pfr_player_id' in self.snap_data.columns:
            keep_cols.append('pfr_player_id')
        if 'position' in self.snap_data.columns:
            keep_cols.append('position')
        if 'offense_pct' in self.snap_data.columns:
            keep_cols.append('offense_pct')
        snap_pos = self.snap_data[keep_cols].copy()

        # Map PFR player IDs to GSIS IDs (the format used in player_weekly_stats)
        if 'pfr_player_id' in snap_pos.columns:
            from src.data.nfl_data_loader import get_pfr_to_gsis_map
            pfr_to_gsis = get_pfr_to_gsis_map()
            snap_pos['player_id'] = snap_pos['pfr_player_id'].map(pfr_to_gsis)
        else:
            snap_pos['player_id'] = np.nan

        snap_pos['name'] = snap_pos['player']
        # A missing offense_snaps means the feed didn't record this player's
        # participation, not that he took zero snaps -- keep it nullable.
        snap_pos['snap_count'] = snap_pos['offense_snaps'].astype('Int64')

        # team_snaps = the team's actual offensive play count for that game.
        # Previously computed via groupby(...)['offense_snaps'].sum() --
        # summing every individual player's own snap count instead of the
        # team's actual play count. Since ~11+ offensive players are each
        # credited with most of the same ~55-75 snaps that game, this
        # inflated team_snaps to roughly (avg snaps per player * num
        # credited players), ~10x the real total -- crushing every derived
        # snap_share/snap_share_pct feature for whichever season is
        # ingested via this PBP path (GAPS.md, 2026-08-07, "team_snaps
        # inflation" entry). Prefer the team total implied by each
        # player's own offense_pct (nflverse's own verified per-player
        # share: offense_snaps / offense_pct); the median across the
        # team/week is robust to individual rounding noise. Fall back to
        # max(offense_snaps) -- the single most-used player, a reasonable
        # proxy for total team plays -- when offense_pct is unavailable.
        team_snaps = compute_team_snaps(snap_pos)
        snap_pos = snap_pos.merge(team_snaps, on=['season', 'week', 'team'], how='left')
        snap_pos['team_snaps'] = snap_pos['team_snaps'].astype('Int64')

        # Build merge-ready snap frame
        snap_merge_cols = ['snap_count', 'team_snaps']
        if 'offense_pct' in snap_pos.columns:
            snap_merge_cols.append('offense_pct')

        # Primary join: player_id (GSIS) + season + week
        has_player_id = 'player_id' in snap_pos.columns and 'player_id' in stats_df.columns
        if has_player_id and snap_pos['player_id'].notna().any():
            join_keys = ['player_id', 'season', 'week']
            snap_for_merge = snap_pos.dropna(subset=['player_id'])[join_keys + snap_merge_cols].drop_duplicates(subset=join_keys)
            merged = stats_df.merge(snap_for_merge, on=join_keys, how='left', suffixes=('', '_snap'))
        else:
            # Fallback: name + team + season + week
            join_keys = ['season', 'week', 'name', 'team']
            snap_for_merge = snap_pos[join_keys + snap_merge_cols].drop_duplicates(subset=join_keys)
            merged = stats_df.merge(snap_for_merge, on=join_keys, how='left', suffixes=('', '_snap'))

        # Coalesce: prefer snap-side values (original may be 0-filled placeholder)
        for col in snap_merge_cols:
            snap_col = f'{col}_snap'
            if snap_col in merged.columns:
                merged[col] = merged[snap_col].fillna(merged[col])
                merged = merged.drop(columns=[snap_col])

        # NULL, not 0, when a row found no snap record. These used to be
        # fillna(0), which asserted "this player took zero snaps" for every
        # unmatched row -- indistinguishable from a real zero, and the reason
        # player_weekly_stats carried 100% zeros for 2006-2017 (GAPS.md
        # 2026-08-19 audit). Int64 is the nullable integer dtype, so the
        # missingness survives to SQLite as NULL instead of collapsing.
        if 'snap_count' not in merged.columns:
            merged['snap_count'] = pd.NA
        if 'team_snaps' not in merged.columns:
            merged['team_snaps'] = pd.NA
        merged['snap_count'] = merged['snap_count'].astype('Int64')
        merged['team_snaps'] = merged['team_snaps'].astype('Int64')
        # snap_share is unknown when either side is unknown; it is a genuine
        # 0.0 only when the player is known to have taken 0 of a known team
        # total. Division alone propagates NA; `.where` handles team_snaps==0,
        # which would otherwise be an infinity.
        merged['snap_share'] = (
            (merged['snap_count'] / merged['team_snaps'])
            .where(merged['team_snaps'] > 0)
        )
        return merged
    
    def calculate_fantasy_points(self, df: pd.DataFrame, ppr: float = 1.0) -> pd.DataFrame:
        """Calculate fantasy points (PPR scoring).

        Uses the same formula as NFLDataLoader._calculate_fantasy_points and
        config.settings.SCORING so PBP-sourced and weekly-sourced rows produce
        consistent target values.
        """
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

        # Fumbles lost: -2 per fumble (was missing — caused PBP targets to
        # systematically overstate fantasy points vs weekly-sourced data)
        if 'fumbles_lost' in df.columns:
            df['fantasy_points'] -= df['fumbles_lost'].fillna(0) * 2

        # Two-point conversions: +2 each
        if 'two_point_conversions' in df.columns:
            df['fantasy_points'] += df['two_point_conversions'].fillna(0) * 2

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

        # Players who both passed AND rushed/received in the same game appear
        # in both the passing and skill groups — merge duplicate rows by
        # summing numeric stats. String columns default to "first", but
        # `position` must NOT: `aggregate_passing_stats` hardcodes
        # position='QB' (line ~197) for anyone with a pass attempt, including
        # a single trick-play pass thrown by a real RB/WR (Derrick Henry,
        # Christian McCaffrey, etc.) — since the passing row is concatenated
        # first (`all_stats = pd.concat([passing, skill], ...)` above),
        # "first" always picks 'QB' over the skill row's real position for
        # these dual-source rows, permanently corrupting `players.position`
        # on the next ingestion of that week (INSERT OR REPLACE, no
        # correction mechanism -- see GAPS.md, 2026-08-08). Instead, drop
        # `position` from this collapse entirely so `_infer_position` (called
        # below on the full frame) decides it from the SUMMED stats plus the
        # players-table lookup, not an arbitrary row-concat order.
        _key = ["player_id", "season", "week"]
        if all_stats.duplicated(subset=_key, keep=False).any():
            valid_id = all_stats["player_id"].astype(str).str.strip() != ""
            to_collapse = all_stats[valid_id & all_stats.duplicated(subset=_key, keep=False)]
            keep = all_stats[~valid_id | ~all_stats.duplicated(subset=_key, keep=False)]
            if not to_collapse.empty:
                _num = to_collapse.select_dtypes(include="number").columns.difference(_key).tolist()
                _str = to_collapse.columns.difference(_key).difference(_num).tolist()
                _str_no_pos = [c for c in _str if c != "position"]
                _agg = {c: "sum" for c in _num}
                _agg.update({c: "first" for c in _str_no_pos})
                to_collapse = to_collapse.groupby(_key, as_index=False, sort=False).agg(_agg)
                to_collapse["position"] = np.nan
                all_stats = pd.concat([keep, to_collapse], ignore_index=True)

        # Extract opponent from PBP data
        if self.pbp_data is not None and 'defteam' in self.pbp_data.columns and 'posteam' in self.pbp_data.columns:
            opp_map = self.pbp_data.groupby(['season', 'week', 'posteam'])['defteam'].agg(
                lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) else ''
            ).reset_index()
            opp_map = opp_map.rename(columns={'posteam': 'team', 'defteam': 'opponent'})
            if 'opponent' in all_stats.columns:
                all_stats = all_stats.drop(columns=['opponent'])
            all_stats = all_stats.merge(opp_map, on=['season', 'week', 'team'], how='left')
            all_stats['opponent'] = all_stats['opponent'].fillna('')

        # Extract home_away from PBP data
        if self.pbp_data is not None and 'home_team' in self.pbp_data.columns and 'posteam' in self.pbp_data.columns:
            ha_map = self.pbp_data.groupby(['season', 'week', 'posteam'])['home_team'].agg(
                lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) else ''
            ).reset_index()
            ha_map['home_away'] = np.where(ha_map['posteam'] == ha_map['home_team'], 'home', 'away')
            ha_map = ha_map.rename(columns={'posteam': 'team'})[['season', 'week', 'team', 'home_away']]
            if 'home_away' in all_stats.columns:
                all_stats = all_stats.drop(columns=['home_away'])
            all_stats = all_stats.merge(ha_map, on=['season', 'week', 'team'], how='left')
            all_stats['home_away'] = all_stats['home_away'].fillna('unknown')

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
        """Infer position from stats if not available.

        Heuristic order: QB (passing attempts) > RB (rush-heavy) > TE vs WR
        (receiving yards threshold).  Returns empty string when position cannot
        be determined so callers can detect and handle unknown positions.

        Lookup order (2026-04-25 fix): row['position'] → players table →
        heuristic.  The players table is the post-load authoritative source
        (kept in sync with nflverse weekly_rosters via the layer-1 UPDATE
        and ongoing maintenance).  The heuristic is only used for genuinely
        unknown players to avoid the WR-default that misclassified TEs like
        Bowers, McBride, Pitts (productive TEs whose yds/target > 8 fell
        through to the WR fallback).  See
        docs/POSITION_CLASSIFICATION_AUDIT.md for the RCA.
        """
        if pd.notna(row.get('position')):
            return row['position']

        # Manual overrides take highest priority (covers rookies missing
        # from weekly_rosters whose DB position is wrong).
        from config.settings import POSITION_OVERRIDES
        pid = row.get('player_id')
        if pid and pid in POSITION_OVERRIDES:
            return POSITION_OVERRIDES[pid]

        # Players-table lookup.  This is a per-row query but cached at
        # class-level after first lookup.
        if pid:
            cached = self._players_position_lookup().get(pid)
            if cached:
                return cached

        # If passing is the dominant activity, likely QB. A bare nonzero
        # passing_attempts (typically exactly 1) is common for trick-play
        # passes thrown by RBs/WRs (Derrick Henry, Christian McCaffrey,
        # etc.) -- require passing to meaningfully dominate rush/target
        # volume, not just be present, or a single-trick-play game
        # misclassifies a real skill player as QB (GAPS.md, 2026-08-08).
        pass_att = row.get('passing_attempts', 0) or 0
        rush_att = row.get('rushing_attempts', 0) or 0
        tgt = row.get('targets', 0) or 0
        if pass_att >= 5 and pass_att >= (rush_att + tgt):
            return 'QB'

        # If has rushing attempts but few/no targets, likely RB
        if rush_att > tgt:
            return 'RB'

        # Distinguish TE from WR: TEs typically have fewer receiving yards per
        # game than WRs.  This is imperfect but avoids systematically
        # misclassifying all TEs as WR (the previous default).
        recv_yards = row.get('receiving_yards', 0) or 0
        targets = row.get('targets', 0) or 0
        if targets > 0 and recv_yards / max(targets, 1) < 8:
            return 'TE'

        return 'WR'

    _PLAYERS_POSITION_CACHE = None

    @classmethod
    def _players_position_lookup(cls) -> dict:
        """Cached {player_id: position}, ROSTERS FIRST, players table second.

        This used to read `players` only, which made the corruption
        self-sustaining: `aggregate_passing_stats` stamps position='QB' on
        anyone with a pass attempt, that lands in `players` via
        INSERT OR REPLACE, and then this lookup hands the same wrong value
        back to `_infer_position` on every subsequent ingest. Nulling
        `position` during the duplicate-row collapse (the 2026-08-08
        mitigation) could not help, because the fallback it deferred to was
        reading the poisoned value.

        Weekly roster snapshots are reported, not derived, so they break the
        loop. Verified: `weekly_rosters` has McCaffrey/Henry as RB and Kupp
        as WR for every week they played, while `players` said QB for all
        three (GAPS.md 2026-08-20).

        Returns {} if DB or tables are missing (preserves heuristic-only
        behavior for first-ever loads).
        """
        if cls._PLAYERS_POSITION_CACHE is not None:
            return cls._PLAYERS_POSITION_CACHE
        cls._PLAYERS_POSITION_CACHE = {}
        try:
            import sqlite3
            from pathlib import Path
            db = Path(__file__).resolve().parents[2] / "data" / "nfl_data.db"
            if not db.exists():
                return cls._PLAYERS_POSITION_CACHE
            from src.utils.database import DatabaseManager
            lookup = dict(DatabaseManager(str(db)).get_authoritative_player_positions())
            with sqlite3.connect(str(db)) as conn:
                rows = conn.execute(
                    "SELECT player_id, position FROM players "
                    "WHERE position IN ('QB','RB','WR','TE')"
                ).fetchall()
            for pid, pos in rows:
                lookup.setdefault(pid, pos)
            cls._PLAYERS_POSITION_CACHE = lookup
        except Exception:
            cls._PLAYERS_POSITION_CACHE = {}
        return cls._PLAYERS_POSITION_CACHE

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

        # Redzone plays (yardline_100 <= 20)
        if "yardline_100" in plays.columns:
            plays["is_redzone_play"] = (plays["yardline_100"].fillna(99) <= 20).astype(int)
            td_col = plays.get("touchdown", pd.Series(0, index=plays.index))
            if isinstance(td_col, pd.Series):
                td_col = td_col.fillna(0)
            plays["is_redzone_score"] = (
                (plays["is_redzone_play"] == 1) & (td_col == 1)
            ).astype(int)
            agg["is_redzone_play"] = "sum"
            agg["is_redzone_score"] = "sum"

        # Sacks
        if "sack" in plays.columns:
            agg["sack"] = "sum"

        team_week = plays.groupby(["season", "week", "posteam"]).agg(agg).reset_index()
        team_week = team_week.rename(columns={
            "posteam": "team",
            "neutral_pass_play": "neutral_pass_plays",
            "neutral_run_play": "neutral_run_plays",
            "is_pass": "pass_attempts",
            "is_run": "rush_attempts",
        })
        if "is_redzone_play" in team_week.columns:
            team_week = team_week.rename(columns={
                "is_redzone_play": "redzone_attempts",
                "is_redzone_score": "redzone_scores",
            })
        if "sack" in team_week.columns:
            team_week = team_week.rename(columns={"sack": "sacks_allowed"})

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
            result_col = "drive_result" if "drive_result" in drive_df.columns else (
                "fixed_drive_result" if "fixed_drive_result" in drive_df.columns else None
            )
            if result_col:
                drive_df["drive_points"] = drive_df[result_col].map(_drive_result_to_points).fillna(0)
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

        # Third-down conversion rate
        if {"third_down_converted", "third_down_failed"}.issubset(plays.columns):
            td = plays.groupby(["season", "week", "posteam"]).agg(
                _td_conv=("third_down_converted", "sum"),
                _td_fail=("third_down_failed", "sum"),
            ).reset_index().rename(columns={"posteam": "team"})
            td_denom = td["_td_conv"] + td["_td_fail"]
            td["third_down_conv"] = np.where(td_denom > 0, td["_td_conv"] / td_denom, np.nan)
            team_week = team_week.merge(
                td[["season", "week", "team", "third_down_conv"]], on=["season", "week", "team"], how="left"
            )
        else:
            team_week["third_down_conv"] = np.nan

        # Time of possession (minutes), summed from per-drive MM:SS
        if drive_col and "drive_time_of_possession" in plays.columns:
            top_df = plays[plays[drive_col].notna()][
                ["season", "week", "posteam", drive_col, "drive_time_of_possession"]
            ].dropna(subset=["drive_time_of_possession"])
            top_df = top_df.drop_duplicates(subset=["season", "week", "posteam", drive_col])
            top_df["_top_seconds"] = top_df["drive_time_of_possession"].map(_mmss_to_seconds)
            top_team = top_df.groupby(["season", "week", "posteam"])["_top_seconds"].sum().reset_index()
            top_team["time_of_possession"] = top_team["_top_seconds"] / 60.0
            top_team = top_team.rename(columns={"posteam": "team"})
            team_week = team_week.merge(
                top_team[["season", "week", "team", "time_of_possession"]], on=["season", "week", "team"], how="left"
            )
        else:
            team_week["time_of_possession"] = np.nan

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
            "pace_sec_per_play", "third_down_conv", "time_of_possession",
        ]:
            if col not in team_week.columns:
                team_week[col] = 0.0

        return team_week


def _mmss_to_seconds(value) -> float:
    """Parse a PBP 'MM:SS' time-of-possession string into seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    try:
        minutes, seconds = str(value).split(":")
        return int(minutes) * 60 + int(seconds)
    except (ValueError, AttributeError):
        return 0.0


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
            if col == "fumbles_lost":
                # Not a neutral default: fumbles_lost feeds fantasy_points at
                # -2 apiece, so a silent 0 overstates the target for everyone
                # who fumbled. This is exactly how 2025 shipped with 0 fumbles
                # league-wide against ~240/season everywhere else -- it reached
                # the DB, the target, and every result read off it, without
                # ever failing anything. Fix with
                # scripts/backfill_fumbles_lost.py after ingest.
                import warnings
                warnings.warn(
                    "PBP aggregation produced no fumbles_lost; defaulting to 0. "
                    "fantasy_points will be overstated by 2 per lost fumble until "
                    "scripts/backfill_fumbles_lost.py is run for these seasons.",
                    RuntimeWarning, stacklevel=2,
                )
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
    # Snap columns (set by merge_with_snaps). Absent means the snap merge
    # never ran for this frame -- unknown, not zero. Defaulting these to 0
    # is what made "no snap data" indistinguishable from "took no snaps".
    for col in ["snap_count", "team_snaps"]:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "snap_share" not in df.columns:
        df["snap_share"] = np.nan
    # Fumbles and two-point conversions
    if "fumbles" not in df.columns:
        df["fumbles"] = df.get("fumbles_lost", pd.Series(0, index=df.index)).fillna(0)
    if "two_point_conversions" not in df.columns:
        df["two_point_conversions"] = 0
    return df


def _atomic_parquet_write(df: pd.DataFrame, path: Path) -> None:
    """Write a parquet file atomically (write to temp, then rename)."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=path.parent)
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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

    For the current in-progress season, the cache is invalidated when the
    cached data has fewer weeks than the current NFL week so new game data
    is picked up automatically.
    """
    if include_advanced is None:
        include_advanced = PBP_ADVANCED_FEATURES_ENABLED

    player_cache, team_cache = _pbp_cache_paths(season)
    player_df = None
    team_df = None

    # Detect whether cached data is stale for the current in-progress season
    cache_is_stale = False
    if use_cache and player_cache.exists():
        try:
            player_df = pd.read_parquet(player_cache)
        except Exception:
            print(f"  Warning: corrupt PBP cache for season {season}, rebuilding")
            player_df = None
        # For the current season, check if cache is missing recent weeks
        if player_df is not None and "week" in player_df.columns:
            try:
                from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week
                current_season = get_current_nfl_season()
                if season == current_season:
                    week_info = get_current_nfl_week()
                    current_week = int(week_info.get("week_num", 0) or 0)
                    cached_max_week = int(player_df["week"].max())
                    if current_week > 0 and cached_max_week < current_week:
                        print(f"  PBP cache for {season} is stale (cached week {cached_max_week}, current week {current_week}); refreshing")
                        cache_is_stale = True
                        player_df = None
            except Exception:
                pass
    if include_team and use_cache and team_cache.exists() and not cache_is_stale:
        try:
            team_df = pd.read_parquet(team_cache)
        except Exception:
            print(f"  Warning: corrupt team PBP cache for season {season}, rebuilding")
            team_df = None
    elif cache_is_stale:
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
                _atomic_parquet_write(player_df, player_cache)
            except Exception as e:
                print(f"  Warning: failed to write PBP player cache for {season}: {e}")
        if include_team and team_df is not None and not team_df.empty:
            try:
                _atomic_parquet_write(team_df, team_cache)
            except Exception as e:
                print(f"  Warning: failed to write PBP team cache for {season}: {e}")

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


def get_personnel_groupings_from_pbp(season: int, use_cache: bool = True) -> pd.DataFrame:
    """Team-week offensive personnel grouping usage (GAPS.md §3.3/§11.1.E).

    Parses PBP's ``offense_personnel`` string column (e.g. "1 RB, 2 TE, 2
    WR") on pass/run plays only -- that column is 100% non-null for real
    offensive snaps (verified on 2016-2025; it's populated starting 2016,
    not ~2013 as originally guessed in GAPS.md, and pre-2016 seasons don't
    carry the column at all). "Backfield count" sums RB+FB per standard
    personnel notation (11 personnel = 1 RB/FB + 1 TE + 3 WR).

    Returns one row per (team, season, week) with pct_11/12/21/13/other
    and n_plays (the pass/run play count the percentages are computed
    over). Empty DataFrame for seasons before the column exists.
    """
    import re

    cache_path = RAW_DATA_DIR / f"pbp_personnel_{season}.parquet"
    if use_cache and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            print(f"  Warning: corrupt personnel-grouping cache for season {season}, rebuilding")

    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    if pbp.empty or "offense_personnel" not in pbp.columns:
        return pd.DataFrame()

    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays = plays[plays["offense_personnel"].notna()]
    if plays.empty:
        return pd.DataFrame()

    def _grouping(s: str) -> str:
        rb = re.search(r"(\d+)\s*RB", s)
        fb = re.search(r"(\d+)\s*FB", s)
        te = re.search(r"(\d+)\s*TE", s)
        rb_n = (int(rb.group(1)) if rb else 0) + (int(fb.group(1)) if fb else 0)
        te_n = int(te.group(1)) if te else 0
        code = f"{rb_n}{te_n}"
        return code if code in ("11", "12", "21", "13") else "other"

    plays["personnel_group"] = plays["offense_personnel"].apply(_grouping)
    plays = plays.rename(columns={"posteam": "team"})
    plays = plays[plays["team"].notna() & (plays["team"] != "")]

    counts = (
        plays.groupby(["team", "season", "week", "personnel_group"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("11", "12", "21", "13", "other"):
        if col not in counts.columns:
            counts[col] = 0
    counts["n_plays"] = counts[["11", "12", "21", "13", "other"]].sum(axis=1)
    for col in ("11", "12", "21", "13", "other"):
        counts[f"pct_{col}"] = counts[col] / counts["n_plays"]

    result = counts.reset_index()[
        ["team", "season", "week", "pct_11", "pct_12", "pct_21", "pct_13", "pct_other", "n_plays"]
    ]
    result["team"] = result["team"].apply(resolver.normalize_team_code)

    if use_cache:
        try:
            _atomic_parquet_write(result, cache_path)
        except Exception as e:
            print(f"  Warning: failed to write personnel-grouping cache for {season}: {e}")

    return result


def get_pass_play_participation_from_pbp(season: int, use_cache: bool = True) -> pd.DataFrame:
    """Per-player pass-play participation rate (GAPS.md §11.1.C/D follow-up).

    Parses PBP's ``offense_players`` column (semicolon-delimited GSIS IDs,
    one row per play) on real dropback pass plays (``play_type ==
    'pass'``) only. Column is 100% non-null from 2016 on (verified
    directly against nfl_data_py; absent before 2016, same coverage start
    as ``offense_personnel`` used for personnel grouping above).

    This is NOT true route participation -- it counts every play a player
    was on the field for during a pass play, whether they ran a route or
    stayed in to block. No accessible data source (NGS, FTN charting, or
    PBP) distinguishes those two cases for non-targeted players; PBP's own
    ``route`` column only describes the *targeted* receiver's route on
    plays with a target. Investigated and confirmed genuinely unavailable
    -- see GAPS.md. Still real signal: distinct from raw snap_share_pct
    (excludes run plays and non-pass snaps).

    GSIS-format player IDs in ``offense_players`` already match this
    project's ``player_id`` format directly -- no ID crosswalk needed.

    Returns one row per (player_id, team, season, week) with
    team_pass_plays, player_pass_plays, and pass_play_participation_pct.
    Empty DataFrame for seasons before the column exists.
    """
    cache_path = RAW_DATA_DIR / f"pbp_participation_{season}.parquet"
    if use_cache and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            print(f"  Warning: corrupt pass-participation cache for season {season}, rebuilding")

    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    if pbp.empty or "offense_players" not in pbp.columns:
        return pd.DataFrame()

    plays = pbp[pbp["play_type"] == "pass"].copy()
    plays = plays[plays["offense_players"].notna() & (plays["offense_players"] != "")]
    plays = plays.rename(columns={"posteam": "team"})
    plays = plays[plays["team"].notna() & (plays["team"] != "")]
    if plays.empty:
        return pd.DataFrame()

    team_pass_plays = (
        plays.groupby(["team", "season", "week"]).size().rename("team_pass_plays").reset_index()
    )

    plays["_players"] = plays["offense_players"].str.split(";")
    exploded = plays[["team", "season", "week", "_players"]].explode("_players")
    exploded = exploded.rename(columns={"_players": "player_id"})
    exploded = exploded[exploded["player_id"].notna() & (exploded["player_id"] != "")]

    player_pass_plays = (
        exploded.groupby(["player_id", "team", "season", "week"])
        .size()
        .rename("player_pass_plays")
        .reset_index()
    )

    result = player_pass_plays.merge(team_pass_plays, on=["team", "season", "week"], how="left")
    result["pass_play_participation_pct"] = (
        result["player_pass_plays"] / result["team_pass_plays"]
    ).clip(0, 1)
    result["team"] = result["team"].apply(resolver.normalize_team_code)

    if use_cache:
        try:
            _atomic_parquet_write(result, cache_path)
        except Exception as e:
            print(f"  Warning: failed to write pass-participation cache for {season}: {e}")

    return result


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

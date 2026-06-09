"""
NFL Data Loader using nfl-data-py package

This module provides a clean interface to load NFL data from the nflverse
data repository via the nfl-data-py package.

Data sources:
- Weekly player stats (fantasy-relevant)
- Seasonal aggregates
- Snap counts
- Schedules
- Rosters

Advantages over scraping:
- Reliable, maintained data source
- Fast (parquet files)
- Comprehensive (all positions, all stats)
- Historical data back to 1999
"""

import os
import ssl
import uuid
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Fix SSL certificate issue before importing nfl_data_py
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    certifi = None


def _get_nfl():
    """Lazy import of nfl_data_py to allow test collection without the package."""
    import nfl_data_py as nfl
    return nfl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.retry import retry_with_backoff

from config.settings import (
    POSITIONS,
    OFFENSIVE_POSITIONS,
    SCORING,
    PBP_ADVANCED_FEATURES_ENABLED,
    PBP_ADVANCED_SEASONS,
    ELIGIBLE_SEASONS_LOOKBACK,
)
from src.utils.database import DatabaseManager
from src.utils.helpers import calculate_fantasy_points
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week, current_season_has_weeks_played
from src.data.schema_validator import validate_weekly_data, validate_schedule_data
from src.data.lineage import persist_dataframe_artifact, set_artifact_id
from src.data.entity_resolver import resolver


# ---------------------------------------------------------------------------
# PFR → GSIS player ID mapping (snap counts use PFR IDs; weekly stats use GSIS)
# ---------------------------------------------------------------------------
_PFR_TO_GSIS_CACHE = None


def get_pfr_to_gsis_map() -> dict:
    """Build PFR player ID → GSIS player ID mapping from nfl-data-py."""
    global _PFR_TO_GSIS_CACHE
    if _PFR_TO_GSIS_CACHE is not None:
        return _PFR_TO_GSIS_CACHE
    try:
        ids_df = _get_nfl().import_ids()
        valid = ids_df[ids_df['pfr_id'].notna() & ids_df['gsis_id'].notna()]
        _PFR_TO_GSIS_CACHE = dict(zip(valid['pfr_id'], valid['gsis_id']))
        print(f"  PFR→GSIS mapping: {len(_PFR_TO_GSIS_CACHE)} players")
    except Exception as e:
        print(f"  WARNING: Could not load PFR→GSIS mapping: {e}")
        _PFR_TO_GSIS_CACHE = {}
    return _PFR_TO_GSIS_CACHE


# ---------------------------------------------------------------------------
# Retry-wrapped nfl-data-py functions (Directive V7 Section 19)
# ---------------------------------------------------------------------------
@retry_with_backoff(max_retries=3, base_delay=2.0)
def _fetch_weekly_data(seasons):
    return _get_nfl().import_weekly_data(seasons)


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _fetch_snap_counts(seasons):
    return _get_nfl().import_snap_counts(seasons)


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _fetch_schedules(seasons):
    return _get_nfl().import_schedules(seasons)


@retry_with_backoff(max_retries=3, base_delay=2.0)
def _fetch_rosters(seasons):
    return _get_nfl().import_seasonal_rosters(seasons)


def _to_scalar_int(x, default: int = 0) -> int:
    """Coerce to int; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return int(x)


def _to_scalar_float(x, default: float = 0.0) -> float:
    """Coerce to float; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return float(x)


def _to_scalar_str(x, default: str = "") -> str:
    """Coerce to str; handle Series from duplicate columns (take first value)."""
    if hasattr(x, "iloc"):
        x = x.iloc[0] if len(x) else default
    if pd.isna(x):
        return default
    return str(x)


def _enrich_team_stats_from_schedule(team_df: pd.DataFrame, sched_df: pd.DataFrame) -> pd.DataFrame:
    """Merge schedule-derived points_scored, points_allowed, home_away into PBP team stats."""
    if sched_df.empty or 'home_team' not in sched_df.columns:
        return team_df

    rows = []
    for _, g in sched_df.iterrows():
        home = g.get('home_team', '')
        away = g.get('away_team', '')
        season = g.get('season', 0)
        week = g.get('week', 0)
        home_score = g.get('home_score', None)
        away_score = g.get('away_score', None)
        if pd.notna(home) and home:
            rows.append({'team': home, 'season': int(season), 'week': int(week),
                        'points_scored': home_score, 'points_allowed': away_score, 'home_away': 'home'})
        if pd.notna(away) and away:
            rows.append({'team': away, 'season': int(season), 'week': int(week),
                        'points_scored': away_score, 'points_allowed': home_score, 'home_away': 'away'})

    if not rows:
        return team_df
    sched_team = pd.DataFrame(rows)

    merge_cols = ['points_scored', 'points_allowed', 'home_away']
    rename_map = {}
    for col in merge_cols:
        if col in team_df.columns:
            rename_map[col] = f'{col}_sched'
    if rename_map:
        sched_team = sched_team.rename(columns=rename_map)

    team_df = team_df.merge(sched_team, on=['team', 'season', 'week'], how='left')

    for col in merge_cols:
        sched_col = rename_map.get(col, col)
        if sched_col in team_df.columns:
            if col in team_df.columns and col != sched_col:
                team_df[col] = team_df[col].fillna(team_df[sched_col])
            else:
                team_df[col] = team_df[sched_col]
            if sched_col != col:
                team_df = team_df.drop(columns=[sched_col])

    return team_df


class NFLDataLoader:
    """
    Load NFL data from nfl-data-py package and store in database.
    
    This replaces the Pro Football Reference data source with a more
    reliable data source.
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        self.lineage_run_id = f"load_{uuid.uuid4().hex[:8]}"
    
    def load_weekly_data(self, seasons: List[int], 
                         store_in_db: bool = True,
                         use_pbp_fallback: bool = True) -> pd.DataFrame:
        """
        Load weekly player statistics. Falls back to PBP aggregation when weekly is empty.
        
        Args:
            seasons: List of seasons to load (e.g., [2020, 2021, 2022])
            store_in_db: Whether to store data in the database
            use_pbp_fallback: If True, use play-by-play aggregation when weekly data is missing
            
        Returns:
            DataFrame with weekly player stats
        """
        print(f"Loading weekly data for seasons: {seasons}")
        
        current_season = get_current_nfl_season()
        in_season_current = current_season_has_weeks_played()
        
        all_dfs = []
        for season in seasons:
            df = pd.DataFrame()
            pbp_adv_df = None
            if use_pbp_fallback and PBP_ADVANCED_FEATURES_ENABLED and season in PBP_ADVANCED_SEASONS:
                try:
                    from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                    pbp_adv_df = get_weekly_stats_from_pbp(
                        season, include_advanced=True, use_cache=True
                    )
                    if pbp_adv_df is not None and not pbp_adv_df.empty:
                        pbp_adv_df = self._standardize_pbp_columns(pbp_adv_df.copy())
                except Exception as e:
                    pbp_adv_df = None
                    print(f"  PBP advanced for {season}: {e}")
            # Current season with at least one week played: try PBP first (weekly often empty for in-progress season)
            if use_pbp_fallback and season == current_season and in_season_current:
                try:
                    pbp_df = pbp_adv_df
                    if pbp_df is None or pbp_df.empty:
                        from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                        pbp_df = get_weekly_stats_from_pbp(season)
                        if not pbp_df.empty:
                            pbp_df = self._standardize_pbp_columns(pbp_df.copy())
                    if pbp_df is not None and not pbp_df.empty:
                        df = pbp_df.copy()
                        print(f"  Current season {season}: loaded from PBP ({len(df)} records)")
                        # Optionally merge with weekly if available (prefer weekly for same player/week)
                        try:
                            weekly_df = _fetch_weekly_data([season])
                            if not weekly_df.empty and len(weekly_df) >= 10:
                                weekly_df = self._standardize_weekly_columns(weekly_df)
                                key_cols = ["player_id", "season", "week"]
                                if all(c in df.columns and c in weekly_df.columns for c in key_cols):
                                    combined = pd.concat([weekly_df, df], ignore_index=True)
                                    df = combined.drop_duplicates(subset=key_cols, keep="first")
                                    print(f"  Merged weekly data for {season}: {len(df)} records")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"  PBP for current season {season}: {e}")
            
            # If we don't have current-season PBP data yet, try weekly
            if df.empty:
                try:
                    df = _fetch_weekly_data([season])
                except Exception as e:
                    print(f"  Weekly data for {season}: {e}")
                    df = pd.DataFrame()
            
            # PBP fallback when weekly is empty or (current season) has fewer weeks than current NFL week
            need_pbp = df.empty or len(df) < 50
            if not need_pbp and use_pbp_fallback and "week" in df.columns:
                cur = get_current_nfl_week()
                if season == cur.get("season") and cur.get("week_num", 0) > 0:
                    max_week = int(df["week"].max()) if len(df) else 0
                    if max_week < cur["week_num"]:
                        need_pbp = True
            if need_pbp and use_pbp_fallback:
                try:
                    pbp_df = pbp_adv_df
                    if pbp_df is None or pbp_df.empty:
                        from src.data.pbp_stats_aggregator import get_weekly_stats_from_pbp
                        pbp_df = get_weekly_stats_from_pbp(season)
                        if not pbp_df.empty:
                            pbp_df = self._standardize_pbp_columns(pbp_df)
                    if pbp_df is not None and not pbp_df.empty:
                        if not df.empty:
                            # Merge: prefer weekly when both have same player_id, season, week
                            key_cols = ["player_id", "season", "week"]
                            if all(c in df.columns and c in pbp_df.columns for c in key_cols):
                                combined = pd.concat([df, pbp_df], ignore_index=True)
                                df = combined.drop_duplicates(subset=key_cols, keep="first")
                            else:
                                df = pbp_df
                        else:
                            df = pbp_df
                        print(f"  Using PBP for {season}: {len(df)} records")
                except Exception as e2:
                    print(f"  PBP fallback for {season}: {e2}")
            if df.empty:
                continue

            bronze_meta = persist_dataframe_artifact(
                df.copy(),
                layer="bronze",
                table="weekly_player_stats",
                run_id=self.lineage_run_id,
                metadata={
                    "source": "nflverse_weekly_stats",
                    "seasons": [int(season)],
                    "week_window": {
                        "start": int(df["week"].min()) if "week" in df.columns and len(df) else None,
                        "end": int(df["week"].max()) if "week" in df.columns and len(df) else None,
                    },
                    "pulled_at": pd.Timestamp.now("UTC").isoformat(),
                },
            )
            
            if not df.empty:
                # Merge advanced PBP-derived columns into weekly data (if available)
                if (
                    pbp_adv_df is not None and not pbp_adv_df.empty
                    and df is not pbp_adv_df
                ):
                    df = self._merge_advanced_pbp_features(df, pbp_adv_df)
                df = self._standardize_weekly_columns(df)
                # Merge snap counts when available (PBP path handles this internally via merge_with_snaps)
                if 'snap_count' not in df.columns or (df['snap_count'] == 0).all():
                    try:
                        snap_df = _fetch_snap_counts([season])
                        if snap_df is not None and not snap_df.empty:
                            from src.data.pbp_stats_aggregator import PBPStatsAggregator
                            snap_agg = PBPStatsAggregator()
                            snap_agg.snap_data = snap_df
                            pre_merge_zeros = (df['snap_count'] == 0).sum() if 'snap_count' in df.columns else len(df)
                            df = snap_agg.merge_with_snaps(df)
                            post_merge_zeros = (df['snap_count'] == 0).sum() if 'snap_count' in df.columns else len(df)
                            if post_merge_zeros == pre_merge_zeros and pre_merge_zeros > 0:
                                print(f"  WARNING: Snap merge for {season} matched 0 rows — snap_count still all zeros. Check name/team key alignment.")
                            else:
                                filled = pre_merge_zeros - post_merge_zeros
                                print(f"  Snap merge for {season}: filled {filled}/{pre_merge_zeros} rows")
                        else:
                            print(f"  WARNING: No snap count data available for season {season}")
                    except Exception as e:
                        print(f"  WARNING: Snap merge failed for {season}: {e}")
                # Backfill opponent and home_away from schedule when missing
                has_empty_opp = 'opponent' not in df.columns or df['opponent'].eq('').all() or df['opponent'].isna().all()
                has_unknown_ha = 'home_away' not in df.columns or df['home_away'].eq('unknown').all() or df['home_away'].isna().all()
                if has_empty_opp or has_unknown_ha:
                    try:
                        sched_df = _fetch_schedules([season])
                        if sched_df is not None and not sched_df.empty:
                            sched_rows = []
                            for _, g in sched_df.iterrows():
                                home = g.get('home_team', '')
                                away = g.get('away_team', '')
                                wk = g.get('week', 0)
                                if pd.notna(home) and home:
                                    sched_rows.append({'team': home, 'week': int(wk), 'sched_opponent': away, 'sched_home_away': 'home'})
                                if pd.notna(away) and away:
                                    sched_rows.append({'team': away, 'week': int(wk), 'sched_opponent': home, 'sched_home_away': 'away'})
                            if sched_rows:
                                sched_lookup = pd.DataFrame(sched_rows)
                                df = df.merge(sched_lookup, on=['team', 'week'], how='left')
                                if 'sched_opponent' in df.columns:
                                    if 'opponent' not in df.columns:
                                        df['opponent'] = ''
                                    empty_opp = df['opponent'].eq('') | df['opponent'].isna()
                                    df.loc[empty_opp, 'opponent'] = df.loc[empty_opp, 'sched_opponent'].fillna('')
                                    df = df.drop(columns=['sched_opponent'])
                                if 'sched_home_away' in df.columns:
                                    if 'home_away' not in df.columns:
                                        df['home_away'] = 'unknown'
                                    unknown_ha = df['home_away'].eq('unknown') | df['home_away'].isna()
                                    df.loc[unknown_ha, 'home_away'] = df.loc[unknown_ha, 'sched_home_away'].fillna('unknown')
                                    df = df.drop(columns=['sched_home_away'])
                    except Exception as e:
                        print(f"  Schedule backfill for {season}: {e}")
                validate_weekly_data(df, strict=True)
                # nfl_data_py weekly data only has offensive positions
                df = df[df['position'].isin(OFFENSIVE_POSITIONS)]
                df['fantasy_points'] = df.apply(
                    lambda row: self._calculate_fantasy_points(row), axis=1
                )
                silver_meta = persist_dataframe_artifact(
                    df.copy(),
                    layer="silver",
                    table="weekly_player_stats_canonical",
                    run_id=self.lineage_run_id,
                    metadata={
                        "source": "nflverse_weekly_stats",
                        "seasons": [int(season)],
                        "normalization": "standardize_weekly_columns + offensive position filter + fantasy points derivation",
                    },
                    parent_artifact_ids=[bronze_meta["artifact_id"]],
                )
                df = set_artifact_id(df, silver_meta["artifact_id"])
                all_dfs.append(df)

                # Persist team-level advanced PBP stats when enabled
                if (
                    store_in_db and use_pbp_fallback and PBP_ADVANCED_FEATURES_ENABLED
                    and season in PBP_ADVANCED_SEASONS
                ):
                    try:
                        from src.data.pbp_stats_aggregator import get_team_stats_from_pbp
                        team_df = get_team_stats_from_pbp(season, use_cache=True)
                        if team_df is not None and not team_df.empty:
                            try:
                                sched_df = _fetch_schedules([season])
                                if sched_df is not None and not sched_df.empty:
                                    team_df = _enrich_team_stats_from_schedule(team_df, sched_df)
                            except Exception as e_sched:
                                print(f"  Schedule enrichment for team stats {season}: {e_sched}")
                            self._store_team_stats_dataframe(team_df)
                    except Exception as e:
                        print(f"  Team PBP stats for {season}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()

        # Schema validation before processing (Directive V7, Section 19)
        for i, chunk_df in enumerate(all_dfs):
            issues = validate_weekly_data(chunk_df, strict=True)
            for issue in issues:
                print(f"  Schema validation [{i}]: {issue}")

        # Ensure unique column names so concat does not raise InvalidIndexError
        def _unique_columns(d: pd.DataFrame) -> pd.DataFrame:
            if d.columns.duplicated().any():
                return d.loc[:, ~d.columns.duplicated()]
            return d
        all_dfs = [_unique_columns(d) for d in all_dfs]
        df = pd.concat(all_dfs, ignore_index=True)
        before = len(df)
        df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")
        if len(df) < before:
            print(f"  Dropped {before - len(df)} duplicate (player_id, season, week) rows")
        print(f"  Loaded {len(df)} offensive records total")
        
        print(f"  Total: {len(df)} records for {POSITIONS}")
        
        if store_in_db:
            self._store_weekly_data(df)
        
        return df
    
    def _standardize_pbp_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align PBP-aggregated columns with our schema (same names as weekly)."""
        if 'completions' in df.columns and 'passing_completions' not in df.columns:
            df = df.rename(columns={'completions': 'passing_completions'})
        for col in ['opponent', 'home_away', 'fumbles_lost']:
            if col not in df.columns:
                df[col] = "" if col == "opponent" else ("unknown" if col == "home_away" else 0)
        if 'pass_plays' not in df.columns and 'passing_attempts' in df.columns:
            df['pass_plays'] = df['passing_attempts']
        if 'rush_plays' not in df.columns and 'rushing_attempts' in df.columns:
            df['rush_plays'] = df['rushing_attempts']
        if 'recv_targets' not in df.columns and 'targets' in df.columns:
            df['recv_targets'] = df['targets']
        normalized = resolver.build_keys(df, source="pbp_weekly", name_col="name")
        df = normalized.dataframe
        if 'canonical_player_id' in df.columns:
            fill_mask = df['player_id'].isna() | (df['player_id'].astype(str).str.strip() == '')
            df.loc[fill_mask, 'player_id'] = df.loc[fill_mask, 'canonical_player_id']
        if 'team_norm' in df.columns:
            df['team'] = df['team_norm'].where(df['team_norm'] != '', df.get('team', ''))
        if 'opponent_norm' in df.columns and 'opponent' in df.columns:
            df['opponent'] = df['opponent_norm'].where(df['opponent_norm'] != '', df['opponent'])
        return df
    
    def _standardize_weekly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match our database schema."""
        column_mapping = {
            'player_id': 'player_id',
            'player_name': 'name',
            'player_display_name': 'display_name',
            'position': 'position',
            'recent_team': 'team',
            'season': 'season',
            'week': 'week',
            'opponent_team': 'opponent',
            'completions': 'passing_completions',
            'attempts': 'passing_attempts',
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'interceptions': 'interceptions',
            'carries': 'rushing_attempts',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'targets': 'targets',
            'receptions': 'receptions',
            'receiving_yards': 'receiving_yards',
            'receiving_tds': 'receiving_tds',
            'sack_fumbles_lost': 'fumbles_lost',
            'rushing_fumbles_lost': 'rushing_fumbles_lost',
            'receiving_fumbles_lost': 'receiving_fumbles_lost',
        }
        
        # Rename columns that exist (skip if target name already present to avoid duplicate columns)
        rename_dict = {
            k: v for k, v in column_mapping.items()
            if k in df.columns and (k == v or v not in df.columns)
        }
        df = df.rename(columns=rename_dict)
        
        # Use display_name if name is missing
        if 'name' not in df.columns and 'display_name' in df.columns:
            df['name'] = df['display_name']
        
        # Combine fumbles lost
        fumble_cols = ['fumbles_lost', 'rushing_fumbles_lost', 'receiving_fumbles_lost']
        existing_fumble_cols = [c for c in fumble_cols if c in df.columns]
        if existing_fumble_cols:
            df['fumbles_lost'] = df[existing_fumble_cols].fillna(0).sum(axis=1)
        else:
            df['fumbles_lost'] = 0

        # Combine total fumbles (not just lost)
        total_fumble_cols = ['sack_fumbles', 'rushing_fumbles', 'receiving_fumbles']
        existing_total_fumble_cols = [c for c in total_fumble_cols if c in df.columns]
        if existing_total_fumble_cols:
            df['fumbles'] = df[existing_total_fumble_cols].fillna(0).sum(axis=1)
        else:
            df['fumbles'] = df['fumbles_lost']  # fallback: count lost as total

        # Combine two-point conversions
        two_pt_cols = ['passing_2pt_conversions', 'rushing_2pt_conversions', 'receiving_2pt_conversions']
        existing_2pt_cols = [c for c in two_pt_cols if c in df.columns]
        if existing_2pt_cols:
            df['two_point_conversions'] = df[existing_2pt_cols].fillna(0).sum(axis=1).astype(int)
        else:
            df['two_point_conversions'] = 0

        # Derive sacks column for QBs (used by qb_features.py for sack_rate)
        if 'sacks' not in df.columns:
            if 'times_sacked' in df.columns:
                df['sacks'] = df['times_sacked'].fillna(0)
            elif 'sack_fumbles' in df.columns:
                df['sacks'] = df['sack_fumbles'].fillna(0)

        # Fill missing numeric columns with 0
        numeric_cols = [
            'passing_attempts', 'passing_completions', 'passing_yards', 'passing_tds',
            'interceptions', 'rushing_attempts', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds',
            'fumbles', 'fumbles_lost', 'two_point_conversions', 'sacks',
            'pass_plays', 'rush_plays', 'recv_targets',
            'pass_epa', 'rush_epa', 'recv_epa',
            'pass_wpa', 'rush_wpa', 'recv_wpa',
            'pass_success_rate', 'rush_success_rate', 'recv_success_rate',
            'neutral_targets', 'neutral_rushes', 'third_down_targets', 'short_yardage_rushes',
            'redzone_targets', 'goal_line_touches', 'two_minute_targets', 'high_leverage_touches'
        ]
        if 'pass_plays' not in df.columns and 'passing_attempts' in df.columns:
            df['pass_plays'] = df['passing_attempts']
        if 'rush_plays' not in df.columns and 'rushing_attempts' in df.columns:
            df['rush_plays'] = df['rushing_attempts']
        if 'recv_targets' not in df.columns and 'targets' in df.columns:
            df['recv_targets'] = df['targets']
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)
        
        # Determine home/away
        if 'home_team' in df.columns:
            df['home_away'] = np.where(df['team'] == df['home_team'], 'home', 'away')
        else:
            df['home_away'] = 'unknown'

        normalized = resolver.build_keys(df, source="weekly", name_col="name")
        df = normalized.dataframe
        fill_mask = df['player_id'].isna() | (df['player_id'].astype(str).str.strip() == '')
        df.loc[fill_mask, 'player_id'] = df.loc[fill_mask, 'canonical_player_id']
        df['team'] = df['team_norm'].where(df['team_norm'] != '', df['team'])
        if 'opponent' in df.columns:
            df['opponent'] = df['opponent_norm'].where(df['opponent_norm'] != '', df['opponent'])

        return df

    def _merge_advanced_pbp_features(self, df: pd.DataFrame, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Merge advanced PBP-derived columns into weekly data (keyed by player_id/season/week)."""
        key_cols = ["player_id", "season", "week"]
        if not all(c in df.columns for c in key_cols) or not all(c in pbp_df.columns for c in key_cols):
            return df
        adv_cols = [
            "pass_plays", "rush_plays", "recv_targets",
            "pass_epa", "rush_epa", "recv_epa",
            "pass_wpa", "rush_wpa", "recv_wpa",
            "pass_success_rate", "rush_success_rate", "recv_success_rate",
            "neutral_targets", "neutral_rushes", "third_down_targets", "short_yardage_rushes",
            "redzone_targets", "goal_line_touches", "two_minute_targets", "high_leverage_touches",
            "rush_inside_10", "rush_inside_5", "targets_15_plus", "air_yards",
        ]
        pbp_subset_cols = [c for c in adv_cols if c in pbp_df.columns]
        if not pbp_subset_cols:
            return df
        pbp_subset = pbp_df[key_cols + pbp_subset_cols].drop_duplicates(subset=key_cols)
        merged = df.merge(pbp_subset, on=key_cols, how="left", suffixes=("", "_pbp"))
        for col in pbp_subset_cols:
            pbp_col = f"{col}_pbp"
            if pbp_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(merged[pbp_col])
                    merged = merged.drop(columns=[pbp_col])
                else:
                    merged = merged.rename(columns={pbp_col: col})
        return merged
    
    def _calculate_fantasy_points(self, row: pd.Series) -> float:
        """Calculate PPR fantasy points for a player."""
        points = 0.0
        
        # Passing
        points += row.get('passing_yards', 0) * SCORING.get('passing_yards', 0.04)
        points += row.get('passing_tds', 0) * SCORING.get('passing_tds', 4)
        points += row.get('interceptions', 0) * SCORING.get('interceptions', -2)
        
        # Rushing
        points += row.get('rushing_yards', 0) * SCORING.get('rushing_yards', 0.1)
        points += row.get('rushing_tds', 0) * SCORING.get('rushing_tds', 6)
        
        # Receiving (PPR)
        points += row.get('receptions', 0) * SCORING.get('receptions', 1)
        points += row.get('receiving_yards', 0) * SCORING.get('receiving_yards', 0.1)
        points += row.get('receiving_tds', 0) * SCORING.get('receiving_tds', 6)
        
        # Fumbles
        points += row.get('fumbles_lost', 0) * SCORING.get('fumbles_lost', -2)
        
        return round(points, 1)
    
    def _store_weekly_data(self, df: pd.DataFrame):
        """Store weekly data in the database."""
        print("  Storing in database...")
        validate_weekly_data(df, strict=True)
        # Avoid duplicate column names (e.g. from merge/concat) so row values are scalars
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        players_inserted = 0
        stats_inserted = 0

        for _, row in df.iterrows():
            # Insert player (coerce to scalar in case of duplicate columns)
            player_data = {
                'player_id': _to_scalar_str(row['player_id'], ''),
                'name': _to_scalar_str(row['name'], ''),
                'position': _to_scalar_str(row['position'], ''),
            }
            self.db.insert_player(player_data)
            players_inserted += 1

            # Insert weekly stats (coerce to scalar; duplicate columns can make row values Series)
            stats_data = {
                'player_id': _to_scalar_str(row['player_id'], ''),
                'season': _to_scalar_int(row['season'], 0),
                'week': _to_scalar_int(row['week'], 0),
                'team': _to_scalar_str(row['team'], ''),
                'opponent': _to_scalar_str(row.get('opponent', ''), ''),
                'home_away': _to_scalar_str(row.get('home_away', 'unknown'), 'unknown'),
                'games_played': 1,
                'passing_attempts': _to_scalar_int(row.get('passing_attempts', 0), 0),
                'passing_completions': _to_scalar_int(row.get('passing_completions', 0), 0),
                'passing_yards': _to_scalar_int(row.get('passing_yards', 0), 0),
                'passing_tds': _to_scalar_int(row.get('passing_tds', 0), 0),
                'interceptions': _to_scalar_int(row.get('interceptions', 0), 0),
                'rushing_attempts': _to_scalar_int(row.get('rushing_attempts', 0), 0),
                'rush_inside_10': _to_scalar_int(row.get('rush_inside_10', 0), 0),
                'rush_inside_5': _to_scalar_int(row.get('rush_inside_5', 0), 0),
                'rushing_yards': _to_scalar_int(row.get('rushing_yards', 0), 0),
                'rushing_tds': _to_scalar_int(row.get('rushing_tds', 0), 0),
                'targets': _to_scalar_int(row.get('targets', 0), 0),
                'targets_15_plus': _to_scalar_int(row.get('targets_15_plus', 0), 0),
                'receptions': _to_scalar_int(row.get('receptions', 0), 0),
                'air_yards': _to_scalar_float(row.get('air_yards', 0.0), 0.0),
                'receiving_yards': _to_scalar_int(row.get('receiving_yards', 0), 0),
                'receiving_tds': _to_scalar_int(row.get('receiving_tds', 0), 0),
                'fumbles_lost': _to_scalar_int(row.get('fumbles_lost', 0), 0),
                'fumbles': _to_scalar_int(row.get('fumbles', 0), 0),
                'two_point_conversions': _to_scalar_int(row.get('two_point_conversions', 0), 0),
                'snap_count': _to_scalar_int(row.get('snap_count', 0), 0),
                'snap_share': _to_scalar_float(row.get('snap_share', 0.0), 0.0),
                'team_snaps': _to_scalar_int(row.get('team_snaps', 0), 0),
                'pass_plays': _to_scalar_int(row.get('pass_plays', row.get('passing_attempts', 0)), 0),
                'rush_plays': _to_scalar_int(row.get('rush_plays', row.get('rushing_attempts', 0)), 0),
                'recv_targets': _to_scalar_int(row.get('recv_targets', row.get('targets', 0)), 0),
                'pass_epa': _to_scalar_float(row.get('pass_epa', 0.0), 0.0),
                'rush_epa': _to_scalar_float(row.get('rush_epa', 0.0), 0.0),
                'recv_epa': _to_scalar_float(row.get('recv_epa', 0.0), 0.0),
                'pass_wpa': _to_scalar_float(row.get('pass_wpa', 0.0), 0.0),
                'rush_wpa': _to_scalar_float(row.get('rush_wpa', 0.0), 0.0),
                'recv_wpa': _to_scalar_float(row.get('recv_wpa', 0.0), 0.0),
                'pass_success_rate': _to_scalar_float(row.get('pass_success_rate', 0.0), 0.0),
                'rush_success_rate': _to_scalar_float(row.get('rush_success_rate', 0.0), 0.0),
                'recv_success_rate': _to_scalar_float(row.get('recv_success_rate', 0.0), 0.0),
                'neutral_targets': _to_scalar_int(row.get('neutral_targets', 0), 0),
                'neutral_rushes': _to_scalar_int(row.get('neutral_rushes', 0), 0),
                'third_down_targets': _to_scalar_int(row.get('third_down_targets', 0), 0),
                'short_yardage_rushes': _to_scalar_int(row.get('short_yardage_rushes', 0), 0),
                'redzone_targets': _to_scalar_int(row.get('redzone_targets', 0), 0),
                'goal_line_touches': _to_scalar_int(row.get('goal_line_touches', 0), 0),
                'two_minute_targets': _to_scalar_int(row.get('two_minute_targets', 0), 0),
                'high_leverage_touches': _to_scalar_int(row.get('high_leverage_touches', 0), 0),
                'fantasy_points': _to_scalar_float(row.get('fantasy_points', 0), 0.0),
            }
            self.db.insert_player_weekly_stats(stats_data)
            stats_inserted += 1
        
        print(f"  Stored {players_inserted} players, {stats_inserted} weekly records")

    def _store_team_stats_dataframe(self, df: pd.DataFrame):
        """Store team-level stats in the database."""
        if df.empty:
            return
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")]

        def _opt_int(row, col):
            if col not in row or pd.isna(row.get(col)):
                return None
            return _to_scalar_int(row.get(col), 0)

        def _opt_float(row, col):
            if col not in row or pd.isna(row.get(col)):
                return None
            return _to_scalar_float(row.get(col), 0.0)

        for _, row in df.iterrows():
            team_data = {
                "team": _to_scalar_str(row.get("team", ""), ""),
                "season": _to_scalar_int(row.get("season", 0), 0),
                "week": _to_scalar_int(row.get("week", 0), 0),
                "opponent": _to_scalar_str(row.get("opponent", ""), "") if "opponent" in row else None,
                "home_away": _to_scalar_str(row.get("home_away", ""), "") if "home_away" in row else None,
                "points_scored": _opt_int(row, "points_scored"),
                "points_allowed": _opt_int(row, "points_allowed"),
                "total_yards": _opt_int(row, "total_yards"),
                "passing_yards": _opt_int(row, "passing_yards"),
                "rushing_yards": _opt_int(row, "rushing_yards"),
                "turnovers": _opt_int(row, "turnovers"),
                "time_of_possession": _opt_float(row, "time_of_possession"),
                "total_plays": _opt_int(row, "total_plays"),
                "pass_attempts": _opt_int(row, "pass_attempts"),
                "rush_attempts": _opt_int(row, "rush_attempts"),
                "redzone_attempts": _opt_int(row, "redzone_attempts"),
                "redzone_scores": _opt_int(row, "redzone_scores"),
                "third_down_conv": _opt_float(row, "third_down_conv"),
                "sacks_allowed": _opt_int(row, "sacks_allowed"),
                "neutral_pass_plays": _opt_int(row, "neutral_pass_plays"),
                "neutral_run_plays": _opt_int(row, "neutral_run_plays"),
                "neutral_pass_rate": _opt_float(row, "neutral_pass_rate"),
                "neutral_pass_rate_lg": _opt_float(row, "neutral_pass_rate_lg"),
                "neutral_pass_rate_oe": _opt_float(row, "neutral_pass_rate_oe"),
                "drive_count": _opt_int(row, "drive_count"),
                "drive_success_rate": _opt_float(row, "drive_success_rate"),
                "avg_drive_epa": _opt_float(row, "avg_drive_epa"),
                "points_per_drive": _opt_float(row, "points_per_drive"),
                "pace_sec_per_play": _opt_float(row, "pace_sec_per_play"),
            }
            self.db.insert_team_stats(team_data)

    def store_weekly_dataframe(self, df: pd.DataFrame) -> None:
        """
        Standardize, filter to POSITIONS, compute fantasy_points if missing, then store in DB.

        Shared path for PBP-only or auto_refresh: ingest a weekly DataFrame into player_weekly_stats
        without duplicating insert logic. Accepts both weekly-style and PBP-style column names.
        """
        if df.empty:
            return
        if not all(c in df.columns for c in ["player_id", "name", "position", "team", "season", "week"]):
            raise ValueError("DataFrame must have player_id, name, position, team, season, week")
        df = df.copy()
        df = self._standardize_pbp_columns(df)
        df = self._standardize_weekly_columns(df)
        df = df[df["position"].isin(POSITIONS)]
        if "fantasy_points" not in df.columns or df["fantasy_points"].isna().any():
            df["fantasy_points"] = df.apply(self._calculate_fantasy_points, axis=1)
        validate_weekly_data(df, strict=True)
        self._store_weekly_data(df)

    def load_snap_counts(self, seasons: List[int], 
                         store_in_db: bool = True) -> pd.DataFrame:
        """Load snap count data."""
        print(f"Loading snap counts for seasons: {seasons}")
        
        try:
            df = _fetch_snap_counts(seasons)
        except Exception as e:
            print(f"Error loading snap counts: {e}")
            return pd.DataFrame()
        
        print(f"  Loaded {len(df)} records")
        
        # Note: When using PBP path (get_weekly_stats_from_pbp), snap_count/team_snaps come from pbp_stats_aggregator.merge_with_snaps.
        
        return df
    
    def load_schedules(self, seasons: List[int],
                       store_in_db: bool = True) -> pd.DataFrame:
        """
        Load NFL schedules.
        
        Args:
            seasons: List of seasons to load
            store_in_db: Whether to store in database
            
        Returns:
            DataFrame with schedule data
        """
        print(f"Loading schedules for seasons: {seasons}")
        
        try:
            df = _fetch_schedules(seasons)
        except Exception as e:
            print(f"Error loading schedules: {e}")
            return pd.DataFrame()
        
        print(f"  Loaded {len(df)} games")
        
        if store_in_db:
            self._store_schedules(df)
        
        return df
    
    def _store_schedules(self, df: pd.DataFrame):
        """Store schedule data in database. Maps game_type to week 19-22 for playoffs (SB=22)."""
        # Schema validation (Directive V7, Section 19)
        issues = validate_schedule_data(df, strict=True)
        for issue in issues:
            print(f"  Schedule schema: {issue}")
        print("  Storing schedules in database...")
        # nflverse game_type: REG, WC, DIV, CON, SB -> align with nfl_calendar week_num 19-22
        _GAME_TYPE_WEEK = {"WC": 19, "DIV": 20, "CON": 21, "SB": 22}
        count = 0
        for _, row in df.iterrows():
            week = _to_scalar_int(row.get("week"), 0)
            if "game_type" in row.index and row.get("game_type") is not None:
                gt = str(row.get("game_type", "")).strip().upper()
                if gt in _GAME_TYPE_WEEK:
                    week = _GAME_TYPE_WEEK[gt]
            schedule_data = {
                "season": _to_scalar_int(row["season"], 0),
                "week": week,
                "home_team": _to_scalar_str(row["home_team"], ""),
                "away_team": _to_scalar_str(row["away_team"], ""),
                "game_id": _to_scalar_str(row.get("game_id", ""), ""),
                "game_time": str(_to_scalar_str(row.get("gameday", ""), "")),
                "venue": _to_scalar_str(row.get("stadium", ""), ""),
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
            }
            self.db.insert_schedule(schedule_data)
            count += 1
        print(f"  Stored {count} games")
    
    def get_available_seasons(self) -> List[int]:
        """Get list of seasons to consider (from config AVAILABLE_SEASONS_START_YEAR through current + 1)."""
        from config.settings import AVAILABLE_SEASONS_START_YEAR
        current = get_current_nfl_season()
        return list(range(AVAILABLE_SEASONS_START_YEAR, current + 2))
    
    def check_schedule_availability(self, season: int) -> bool:
        """Check if schedule is available for a given season."""
        try:
            df = _fetch_schedules([season])
            return len(df) > 0
        except Exception:
            return False
    
    def get_latest_available_season(self) -> int:
        """Get the most recent season with weekly or PBP data available."""
        current = get_current_nfl_season()
        for year in range(current + 1, 2015, -1):
            try:
                df = _fetch_weekly_data([year])
                if len(df) > 0:
                    return year
            except Exception:
                pass
        return current  # Fallback to current NFL season
    
    def load_upcoming_schedule(self, season: int = None) -> pd.DataFrame:
        """
        Load upcoming season schedule.
        
        For seasons not yet in nfl-data-py, this will return empty.
        Defaults to current NFL season when not specified.
        """
        if season is None:
            season = get_current_nfl_season()
        print(f"Loading {season} schedule...")
        
        try:
            df = _fetch_schedules([season])
            if len(df) > 0:
                print(f"  Found {len(df)} games for {season}")
                return df
            else:
                print(f"  No schedule available for {season} yet")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Schedule not available: {e}")
            return pd.DataFrame()
    
    def load_rosters(self, seasons: List[int]) -> pd.DataFrame:
        """
        Load NFL rosters with current team assignments.
        
        This is critical for tracking:
        - Offseason trades and signings
        - Current team assignments
        - Depth chart positions
        - Contract status
        
        Args:
            seasons: List of seasons to load rosters for
            
        Returns:
            DataFrame with roster data including player-team assignments
        """
        print(f"Loading rosters for seasons: {seasons}")
        
        try:
            df = _fetch_rosters(seasons)
            if len(df) > 0:
                print(f"  Loaded {len(df)} roster entries")
                return df
            else:
                print(f"  No roster data available")
                return pd.DataFrame()
        except Exception as e:
            print(f"  Error loading rosters: {e}")
            return pd.DataFrame()
    
    def sync_player_teams(self, season: int = None) -> pd.DataFrame:
        """
        Sync player-team assignments for the current/upcoming season.
        
        This should be run before each season to capture:
        - Free agent signings
        - Trades
        - Roster cuts
        
        Args:
            season: Season to sync (defaults to current year)
            
        Returns:
            DataFrame with updated player-team mappings
        """
        from datetime import datetime
        season = season or datetime.now().year
        
        print(f"Syncing player-team assignments for {season}...")
        
        roster_df = self.load_rosters([season])
        
        if roster_df.empty:
            print("  No roster data available for sync")
            return pd.DataFrame()
        
        # Extract key fields for team assignment
        team_assignments = roster_df[['player_id', 'player_name', 'position', 'team', 
                                       'depth_chart_position', 'status', 'jersey_number']].copy()
        team_assignments = team_assignments.rename(columns={'player_name': 'name'})
        
        # Filter to fantasy-relevant positions
        team_assignments = team_assignments[team_assignments['position'].isin(POSITIONS)]
        
        print(f"  Synced {len(team_assignments)} player-team assignments")
        
        return team_assignments
    
    def get_roster_changes(self, current_season: int = None, 
                           previous_season: int = None) -> pd.DataFrame:
        """
        Identify players who changed teams between seasons.
        
        Args:
            current_season: Current season year
            previous_season: Previous season year
            
        Returns:
            DataFrame with roster changes (trades, signings, etc.)
        """
        from datetime import datetime
        current_season = current_season or datetime.now().year
        previous_season = previous_season or current_season - 1
        
        print(f"Finding roster changes from {previous_season} to {current_season}...")
        
        # Load rosters for both seasons
        current_roster = self.load_rosters([current_season])
        previous_roster = self.load_rosters([previous_season])
        
        if current_roster.empty or previous_roster.empty:
            print("  Cannot compare rosters - missing data")
            return pd.DataFrame()
        
        # Get last team for each player in previous season
        prev_teams = previous_roster.groupby('player_id').agg({
            'player_name': 'first',
            'team': 'last',
            'position': 'first'
        }).reset_index()
        prev_teams = prev_teams.rename(columns={'team': 'previous_team', 'player_name': 'name'})
        
        # Get current team for each player
        curr_teams = current_roster.groupby('player_id').agg({
            'player_name': 'first',
            'team': 'last',
            'position': 'first',
            'depth_chart_position': 'first'
        }).reset_index()
        curr_teams = curr_teams.rename(columns={'team': 'current_team', 'player_name': 'name'})
        
        # Merge to find changes
        merged = prev_teams.merge(curr_teams[['player_id', 'current_team', 'depth_chart_position']], 
                                   on='player_id', how='outer')
        
        # Identify changes
        changes = merged[merged['previous_team'] != merged['current_team']].copy()
        changes = changes[changes['position'].isin(POSITIONS)]
        
        # Categorize change type
        def categorize_change(row):
            if pd.isna(row['previous_team']):
                return 'New Player'
            elif pd.isna(row['current_team']):
                return 'Left League'
            else:
                return 'Team Change'
        
        changes['change_type'] = changes.apply(categorize_change, axis=1)
        changes = changes[changes['change_type'] == 'Team Change']  # Focus on team changes
        
        print(f"  Found {len(changes)} roster changes")
        
        return changes[['name', 'position', 'previous_team', 'current_team', 'depth_chart_position', 'change_type']]
    
    def load_depth_charts(self, season: int = None) -> pd.DataFrame:
        """
        Load depth chart data for utilization modeling.
        
        Depth chart position is critical for predicting:
        - Snap share expectations
        - Target share expectations
        - Red zone opportunity share
        
        Args:
            season: Season to load depth charts for
            
        Returns:
            DataFrame with depth chart positions
        """
        from datetime import datetime
        season = season or datetime.now().year
        
        print(f"Loading depth charts for {season}...")
        
        roster_df = self.load_rosters([season])
        
        if roster_df.empty:
            return pd.DataFrame()
        
        # Extract depth chart info
        depth_df = roster_df[['player_id', 'player_name', 'position', 'team', 
                              'depth_chart_position', 'status']].copy()
        depth_df = depth_df.rename(columns={'player_name': 'name'})
        
        # Filter to fantasy positions
        depth_df = depth_df[depth_df['position'].isin(POSITIONS)]
        
        # Calculate depth chart rank within position/team
        depth_df['depth_rank'] = depth_df.groupby(['team', 'position']).cumcount() + 1
        
        # Create utilization expectation based on depth
        def expected_utilization(row):
            pos = row['position']
            rank = row['depth_rank']
            
            if pos == 'QB':
                return 1.0 if rank == 1 else 0.05
            elif pos == 'RB':
                if rank == 1: return 0.55
                elif rank == 2: return 0.30
                elif rank == 3: return 0.10
                else: return 0.05
            elif pos == 'WR':
                if rank == 1: return 0.30
                elif rank == 2: return 0.25
                elif rank == 3: return 0.20
                elif rank == 4: return 0.15
                else: return 0.05
            elif pos == 'TE':
                if rank == 1: return 0.70
                elif rank == 2: return 0.20
                else: return 0.10
            return 0.0
        
        depth_df['expected_utilization'] = depth_df.apply(expected_utilization, axis=1)
        
        print(f"  Loaded depth charts for {len(depth_df)} players")
        
        return depth_df


def get_eligible_seasons(lookback: int = None) -> List[int]:
    """Return the list of recent seasons used to determine player eligibility.

    A player must have game data in at least one of these seasons to be
    considered active (not retired). Uses the most recent *lookback* seasons
    that actually have data in the database.
    """
    if lookback is None:
        lookback = ELIGIBLE_SEASONS_LOOKBACK
    db = DatabaseManager()
    all_seasons = db.get_seasons_with_data()
    if not all_seasons:
        return []
    return sorted(all_seasons)[-lookback:]


def filter_to_eligible_players(df: pd.DataFrame, eligible_player_ids: List[str] = None) -> pd.DataFrame:
    """Filter a player DataFrame to only include eligible (active) players.

    If *eligible_player_ids* is not provided it is computed from the database
    using the configured lookback window.
    """
    if df.empty or "player_id" not in df.columns:
        return df
    if eligible_player_ids is None:
        db = DatabaseManager()
        eligible_seasons = get_eligible_seasons()
        eligible_player_ids = db.get_eligible_player_ids(eligible_seasons)
    if not eligible_player_ids:
        return df  # Safety: don't filter everything out if DB has no data
    eligible_set = set(eligible_player_ids)
    filtered = df[df["player_id"].isin(eligible_set)]
    n_removed = len(df) - len(filtered)
    if n_removed > 0:
        n_unique_removed = df[~df["player_id"].isin(eligible_set)]["player_id"].nunique()
        print(f"  Filtered out {n_unique_removed} ineligible (retired/inactive) players ({n_removed} rows)")
    return filtered


def load_all_historical_data(seasons: List[int] = None):
    """
    Load all historical data from nfl-data-py.
    
    This is the main entry point for populating the database.
    Loads: weekly stats, schedules, snap counts, rosters, and
    coordinates with external_data / advanced_rookie_injury for
    injuries, combine, and draft data.
    """
    loader = NFLDataLoader()
    
    if seasons is None:
        from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
        seasons = list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))
    
    print("="*60)
    print("Loading NFL Data from nfl-data-py")
    print("="*60)
    
    # Load weekly player data
    weekly_df = loader.load_weekly_data(seasons)
    
    # Load schedules
    schedule_df = loader.load_schedules(seasons)
    
    # Load rosters
    print("\nLoading rosters...")
    try:
        roster_df = loader.load_rosters(seasons)
        print(f"  Loaded rosters: {len(roster_df) if roster_df is not None else 0} records")
    except Exception as e:
        print(f"  Warning: Could not load rosters: {e}")
    
    # Load snap counts
    print("\nLoading snap counts...")
    try:
        snap_df = loader.load_snap_counts(seasons)
        print(f"  Loaded snap counts: {len(snap_df) if snap_df is not None else 0} records")
    except Exception as e:
        print(f"  Warning: Could not load snap counts: {e}")
    
    # Check what we have
    db = DatabaseManager()
    db_seasons = db.get_seasons_with_data()
    
    print("\n" + "="*60)
    print("Data Loading Complete")
    print("="*60)
    print(f"Seasons in database: {db_seasons}")
    
    for s in db_seasons:
        latest_week = db.get_latest_week_for_season(s)
        print(f"  {s}: through week {latest_week}")

    # Populate utilization_scores table
    print("\nComputing utilization scores...")
    try:
        from src.features.utilization_score import UtilizationScoreCalculator
        calc = UtilizationScoreCalculator()
        player_df = db.get_all_players_for_training(min_games=1)
        team_df_util = pd.DataFrame()
        try:
            team_df_util = db.get_team_stats()
        except Exception:
            pass
        if not player_df.empty:
            scored = calc.calculate_all_scores(player_df, team_df_util)
            if 'utilization_score' in scored.columns:
                count = 0
                for _, row in scored[scored['utilization_score'] > 0].iterrows():
                    try:
                        db.insert_utilization_score({
                            'player_id': str(row['player_id']),
                            'season': int(row['season']),
                            'week': int(row['week']),
                            'utilization_score': float(row['utilization_score']),
                            'snap_share': float(row.get('snap_share_pct', 0)),
                            'target_share': float(row.get('target_share_pct', 0)),
                            'rush_share': float(row.get('rush_share_pct', 0)),
                            'redzone_share': float(row.get('redzone_share_pct', 0)),
                            'air_yards_share': float(row.get('air_yards_share_pct', 0)),
                        })
                        count += 1
                    except Exception:
                        pass
                print(f"  Stored {count} utilization scores")
    except Exception as e:
        print(f"  Utilization scores: {e}")

    return weekly_df


if __name__ == "__main__":
    import argparse
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    parser = argparse.ArgumentParser(description='Load NFL data from nfl-data-py (PBP fallback when weekly missing)')
    parser.add_argument('--seasons', type=str, default=None,
                        help=f'Seasons to load (e.g. "{MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON}" or "2022,2023,2024"). Default: {MIN_HISTORICAL_YEAR} through current NFL season ({CURRENT_NFL_SEASON}).')
    
    args = parser.parse_args()
    
    if args.seasons is None:
        seasons = None
    elif '-' in args.seasons:
        start, end = args.seasons.split('-')
        seasons = list(range(int(start), int(end) + 1))
    else:
        seasons = [int(s.strip()) for s in args.seasons.split(',')]
    
    load_all_historical_data(seasons)

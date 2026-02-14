"""
Utilization Score Calculator for NFL Fantasy Football.

The Utilization Score ranges from 0-100 and measures player opportunity/usage.
Higher scores correlate with better fantasy production.

Components (per requirements): snap share, target/touch share, red zone involvement,
and high-value touch rate (rushes inside 10-yard line, targets 15+ air yards) when
PBP-derived data is available. The high_value_touch component is weighted in
UTILIZATION_WEIGHTS and computed by _add_high_value_touch_rate().

Position-specific benchmarks (PPR scoring):
- RB 60-69: ~12.2 PPG, 70%+ finish as RB2/RB3
- RB 70-79: ~15.1 PPG, strong RB2 upside
- RB 80+: Elite usage, RB1 potential

The methodology weights different opportunity metrics by position.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import UTILIZATION_WEIGHTS
from src.utils.helpers import safe_divide


def _bounds_key_to_str(key: Tuple[str, str]) -> str:
    """Serialize (position, component) for JSON."""
    return f"{key[0]}|{key[1]}"


def _bounds_str_to_key(s: str) -> Tuple[str, str]:
    """Deserialize JSON key to (position, component)."""
    a, b = s.split("|", 1)
    return (a, b)


def save_percentile_bounds(position_percentiles: Dict[Tuple[str, str], Tuple[float, float]], path: Path) -> None:
    """Persist percentile bounds (train-only) for use at test/serve. Keys (position, col) -> (lo, hi)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {_bounds_key_to_str(k): list(v) for k, v in position_percentiles.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_percentile_bounds(path: Path) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Load percentile bounds from file. Returns dict (position, col) -> (lo, hi)."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {_bounds_str_to_key(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


class UtilizationScoreCalculator:
    """Calculate Utilization Scores for NFL players by position."""
    
    def __init__(self, weights: Optional[Dict] = None, position_percentiles: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None):
        self.weights = weights if weights is not None else UTILIZATION_WEIGHTS
        self.position_percentiles = dict(position_percentiles) if position_percentiles is not None else {}
    
    def calculate_all_scores(self, player_df: pd.DataFrame, 
                             team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate utilization scores for all players.
        
        Args:
            player_df: DataFrame with player weekly stats
            team_df: DataFrame with team weekly stats
            
        Returns:
            DataFrame with utilization scores added
        """
        # Handle empty DataFrame
        if player_df.empty or "position" not in player_df.columns:
            return player_df
        
        # Merge player and team data
        merged = self._merge_player_team_data(player_df, team_df)

        # High-value touch rate (optional): requires goal-line rushes and/or deep targets.
        # This is computed once here so RB/WR/TE can consume it if weights include it.
        merged = self._add_high_value_touch_rate(merged)
        
        # Calculate position-specific utilization scores
        result_dfs = []
        
        for position in ["QB", "RB", "WR", "TE"]:
            pos_df = merged[merged["position"] == position].copy()
            if len(pos_df) == 0:
                continue
            
            if position == "RB":
                pos_df = self._calculate_rb_utilization(pos_df)
            elif position == "WR":
                pos_df = self._calculate_wr_utilization(pos_df)
            elif position == "TE":
                pos_df = self._calculate_te_utilization(pos_df)
            elif position == "QB":
                pos_df = self._calculate_qb_utilization(pos_df)
            
            result_dfs.append(pos_df)
        
        if not result_dfs:
            return player_df
        
        # Filter out empty DataFrames and ensure consistent dtypes before concat
        result_dfs = [df for df in result_dfs if not df.empty and len(df) > 0]
        
        if not result_dfs:
            return player_df
        
        # Concatenate with explicit handling to avoid FutureWarning
        result = pd.concat(result_dfs, ignore_index=True, sort=False)
        
        # Fill any remaining NaN values in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)
        
        return result

    def _add_high_value_touch_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add high_value_touch_rate (0-100) when source columns are available."""
        if df.empty:
            return df
        df = df.copy()

        rush_inside_10 = df.get("rush_inside_10", pd.Series(0, index=df.index)).fillna(0)
        targets_15_plus = df.get("targets_15_plus", pd.Series(0, index=df.index)).fillna(0)
        rush_att = df.get("rushing_attempts", pd.Series(0, index=df.index)).fillna(0)
        targets = df.get("targets", pd.Series(0, index=df.index)).fillna(0)

        denom = (rush_att + targets).replace(0, np.nan)
        rate = safe_divide((rush_inside_10 + targets_15_plus), denom) * 100
        df["high_value_touch_rate"] = rate.replace([np.inf, -np.inf], 0).fillna(0).clip(0, 100)
        return df
    
    def _merge_player_team_data(self, player_df: pd.DataFrame, 
                                 team_df: pd.DataFrame) -> pd.DataFrame:
        """Merge player stats with team totals for share calculations."""
        # Ensure we have the necessary team columns
        team_cols = ["team", "season", "week", "pass_attempts", "rush_attempts", 
                     "total_plays", "redzone_attempts"]
        
        available_cols = [c for c in team_cols if c in team_df.columns]
        
        if len(available_cols) < 3:
            # If team data is minimal, calculate from player data
            return self._calculate_team_totals_from_players(player_df)
        
        merged = player_df.merge(
            team_df[available_cols],
            on=["team", "season", "week"],
            how="left",
            suffixes=("", "_team")
        )
        
        return merged
    
    def _calculate_team_totals_from_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team totals from aggregated player data."""
        # Check which columns exist for grouping and aggregation
        group_cols = []
        for col in ["team", "season", "week"]:
            if col in df.columns:
                group_cols.append(col)
        
        if not group_cols:
            # No grouping possible -- set team totals to NaN so that
            # downstream share calculations produce NaN rather than
            # the misleading 100% share that results from team == individual.
            import warnings
            warnings.warn(
                "Cannot compute team totals: missing team/season/week columns. "
                "Share-based utilization components will be NaN."
            )
            df = df.copy()
            df["team_targets"] = np.nan
            df["team_rush_attempts"] = np.nan
            df["team_snaps"] = np.nan
            return df
        
        # Build aggregation dict based on available columns
        agg_dict = {}
        col_mapping = {
            "targets": "team_targets",
            "receptions": "team_receptions", 
            "rushing_attempts": "team_rush_attempts",
            "snap_count": "team_snaps",
        }
        
        for col, new_name in col_mapping.items():
            if col in df.columns:
                agg_dict[col] = "sum"
        
        if not agg_dict:
            # No columns to aggregate
            return df
        
        team_totals = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename columns
        rename_dict = {col: col_mapping[col] for col in agg_dict.keys()}
        team_totals = team_totals.rename(columns=rename_dict)
        
        merged = df.merge(team_totals, on=group_cols, how="left")
        
        return merged
    
    def _calculate_rb_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RB Utilization Score.
        
        Components (Fantasy Life aligned: snap share, targets, touches):
        - Snap share, rush share, target share, red zone share, touch share (carries + receptions) / team touches.
        """
        weights = self.weights["RB"]
        
        snap_count = df.get("snap_count", pd.Series(0, index=df.index))
        rushing_attempts = df.get("rushing_attempts", pd.Series(0, index=df.index))
        targets = df.get("targets", pd.Series(0, index=df.index))
        receptions = df.get("receptions", pd.Series(0, index=df.index))
        rushing_tds = df.get("rushing_tds", pd.Series(0, index=df.index))
        receiving_tds = df.get("receiving_tds", pd.Series(0, index=df.index))
        
        team_snaps = df.get("team_snaps", snap_count)
        team_rush = df.get("team_rush_attempts", rushing_attempts)
        team_targets = df.get("team_targets", targets)
        team_receptions = df.get("team_receptions", receptions)
        
        df["snap_share_pct"] = safe_divide(snap_count, team_snaps) * 100
        df["rush_share_pct"] = safe_divide(rushing_attempts, team_rush) * 100
        df["target_share_pct"] = safe_divide(targets, team_targets) * 100
        
        if "redzone_attempts" in df.columns:
            df["redzone_share_pct"] = safe_divide(
                df.get("redzone_touches", rushing_tds + receiving_tds),
                df["redzone_attempts"]
            ) * 100
        else:
            df["redzone_share_pct"] = ((rushing_tds + receiving_tds) * 10).clip(0, 100)
        
        # Touch share (Fantasy Life): (carries + receptions) / team touches
        player_touches = rushing_attempts + receptions
        team_touches = team_rush + team_receptions
        df["touch_share_pct"] = safe_divide(player_touches, team_touches) * 100
        
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="RB", component_key="snap_share_pct")
        df["rush_share_norm"] = self._percentile_normalize(df["rush_share_pct"], position="RB", component_key="rush_share_pct")
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="RB", component_key="target_share_pct")
        df["redzone_share_norm"] = self._percentile_normalize(df["redzone_share_pct"], position="RB", component_key="redzone_share_pct")
        df["touch_share_norm"] = self._percentile_normalize(df["touch_share_pct"], position="RB", component_key="touch_share_pct")
        
        w = weights
        score = (
            df["snap_share_norm"] * w.get("snap_share", 0.20) +
            df["rush_share_norm"] * w.get("rush_share", 0.25) +
            df["target_share_norm"] * w.get("target_share", 0.20) +
            df["redzone_share_norm"] * w.get("redzone_share", 0.20) +
            df["touch_share_norm"] * w.get("touch_share", 0.15)
        )
        # Optional: high-value touch (rushes inside 10, targets 15+ air yards) when data and weight available
        if w.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="RB", component_key="high_value_touch_rate")
            score = score + hv_norm * w["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        df["util_snap_share"] = df["snap_share_pct"]
        df["util_rush_share"] = df["rush_share_pct"]
        df["util_target_share"] = df["target_share_pct"]
        df["util_redzone_share"] = df["redzone_share_pct"]
        
        return df
    
    def _calculate_wr_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WR Utilization Score.
        
        Components:
        - Target share (30%): % of team targets
        - Air yards share (25%): % of team air yards
        - Snap share (15%): % of team offensive snaps
        - Red zone targets (20%): Red zone target involvement
        - Route participation (10%): Routes run / team pass plays
        """
        weights = self.weights["WR"]
        
        # Calculate component metrics
        df["target_share_pct"] = safe_divide(
            df["targets"], df.get("team_targets", df["targets"])
        ) * 100
        
        # Air yards share (estimate from receiving yards if not available)
        if "air_yards" in df.columns:
            df["air_yards_share_pct"] = safe_divide(
                df["air_yards"], df.get("team_air_yards", df["air_yards"])
            ) * 100
        else:
            # Estimate from yards per target
            yards_per_target = safe_divide(df["receiving_yards"], df["targets"])
            df["air_yards_share_pct"] = df["target_share_pct"] * (yards_per_target / 10)
        
        df["snap_share_pct"] = safe_divide(
            df["snap_count"], df.get("team_snaps", df["snap_count"])
        ) * 100
        
        # Red zone targets: use PBP-derived data when available, else TD-based proxy
        if "redzone_targets" in df.columns and "team_redzone_targets" in df.columns:
            df["redzone_targets_pct"] = safe_divide(
                df["redzone_targets"], df["team_redzone_targets"]
            ).clip(0, 1) * 100
        elif "redzone_targets" in df.columns:
            # Normalize against a reasonable max (e.g., 3 RZ targets/game is high for WR)
            df["redzone_targets_pct"] = (df["redzone_targets"] / 3.0 * 100).clip(0, 100)
        else:
            # Fallback: TD-based proxy (documented limitation)
            df["redzone_targets_pct"] = (df["receiving_tds"] * 15).clip(0, 100)
        
        # Route participation: use actual routes when available (Fantasy Life), else proxy from snap share
        if "routes_run" in df.columns and "team_routes" in df.columns:
            df["route_participation_pct"] = safe_divide(df["routes_run"], df["team_routes"]) * 100
        elif "routes_run" in df.columns and "team_snaps" in df.columns:
            df["route_participation_pct"] = safe_divide(df["routes_run"], df["team_snaps"]) * 100
        else:
            df["route_participation_pct"] = (df["snap_share_pct"] * 0.8).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="WR", component_key="target_share_pct")
        df["air_yards_norm"] = self._percentile_normalize(df["air_yards_share_pct"], position="WR", component_key="air_yards_share_pct")
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="WR", component_key="snap_share_pct")
        df["redzone_targets_norm"] = self._percentile_normalize(df["redzone_targets_pct"], position="WR", component_key="redzone_targets_pct")
        df["route_part_norm"] = self._percentile_normalize(df["route_participation_pct"], position="WR", component_key="route_participation_pct")
        
        # Final utilization score; optional high_value_touch (targets 15+ air yards) when weight > 0
        score = (
            df["target_share_norm"] * weights["target_share"] +
            df["air_yards_norm"] * weights["air_yards_share"] +
            df["snap_share_norm"] * weights["snap_share"] +
            df["redzone_targets_norm"] * weights["redzone_targets"] +
            df["route_part_norm"] * weights["route_participation"]
        )
        if weights.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="WR", component_key="high_value_touch_rate")
            score = score + hv_norm * weights["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        # Store component values
        df["util_target_share"] = df["target_share_pct"]
        df["util_air_yards_share"] = df["air_yards_share_pct"]
        df["util_snap_share"] = df["snap_share_pct"]
        
        return df
    
    def _calculate_te_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate TE Utilization Score.
        
        Components:
        - Target share (30%): % of team targets
        - Snap share (20%): % of team offensive snaps
        - Red zone targets (25%): Red zone involvement
        - Air yards share (15%): % of team air yards
        - Inline rate (10%): Usage as inline blocker vs slot
        """
        weights = self.weights["TE"]
        
        # Calculate component metrics
        df["target_share_pct"] = safe_divide(
            df["targets"], df.get("team_targets", df["targets"])
        ) * 100
        
        df["snap_share_pct"] = safe_divide(
            df["snap_count"], df.get("team_snaps", df["snap_count"])
        ) * 100
        
        # Red zone targets: use PBP-derived data when available, else TD-based proxy
        if "redzone_targets" in df.columns and "team_redzone_targets" in df.columns:
            df["redzone_targets_pct"] = safe_divide(
                df["redzone_targets"], df["team_redzone_targets"]
            ).clip(0, 1) * 100
        elif "redzone_targets" in df.columns:
            # TEs typically see 1-2 RZ targets/game at most
            df["redzone_targets_pct"] = (df["redzone_targets"] / 2.0 * 100).clip(0, 100)
        else:
            # Fallback: TD-based proxy (documented limitation; TEs are valuable in RZ)
            df["redzone_targets_pct"] = (df["receiving_tds"] * 20).clip(0, 100)
        
        # Air yards share
        if "air_yards" in df.columns:
            df["air_yards_share_pct"] = safe_divide(
                df["air_yards"], df.get("team_air_yards", df["air_yards"])
            ) * 100
        else:
            yards_per_target = safe_divide(df["receiving_yards"], df["targets"])
            df["air_yards_share_pct"] = df["target_share_pct"] * (yards_per_target / 8)
        
        # Inline rate (estimate - higher snap share with lower target share = more blocking)
        snap_to_target_ratio = safe_divide(df["snap_share_pct"], df["target_share_pct"] + 1)
        df["inline_rate_pct"] = (100 - snap_to_target_ratio * 10).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["target_share_norm"] = self._percentile_normalize(df["target_share_pct"], position="TE", component_key="target_share_pct")
        df["snap_share_norm"] = self._percentile_normalize(df["snap_share_pct"], position="TE", component_key="snap_share_pct")
        df["redzone_targets_norm"] = self._percentile_normalize(df["redzone_targets_pct"], position="TE", component_key="redzone_targets_pct")
        df["air_yards_norm"] = self._percentile_normalize(df["air_yards_share_pct"], position="TE", component_key="air_yards_share_pct")
        df["inline_rate_norm"] = self._percentile_normalize(df["inline_rate_pct"], position="TE", component_key="inline_rate_pct")
        
        # Final utilization score; optional high_value_touch when weight > 0
        score = (
            df["target_share_norm"] * weights["target_share"] +
            df["snap_share_norm"] * weights["snap_share"] +
            df["redzone_targets_norm"] * weights["redzone_targets"] +
            df["air_yards_norm"] * weights["air_yards_share"] +
            df["inline_rate_norm"] * weights["inline_rate"]
        )
        if weights.get("high_value_touch", 0) > 0 and "high_value_touch_rate" in df.columns:
            hv_norm = self._percentile_normalize(df["high_value_touch_rate"], position="TE", component_key="high_value_touch_rate")
            score = score + hv_norm * weights["high_value_touch"]
        df["utilization_score"] = score.clip(0, 100)
        
        # Store component values
        df["util_target_share"] = df["target_share_pct"]
        df["util_snap_share"] = df["snap_share_pct"]
        
        return df
    
    def _calculate_qb_utilization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate QB Utilization Score.
        
        Components:
        - Dropback rate (25%): Pass attempts relative to team plays
        - Rush attempt share (20%): Designed runs and scrambles
        - Red zone opportunity (25%): Red zone play involvement
        - Play volume (30%): Total plays (pass + rush)
        """
        weights = self.weights["QB"]
        
        # Calculate component metrics
        total_plays = df["passing_attempts"] + df["rushing_attempts"]
        team_plays = df.get("team_plays", total_plays)
        
        df["dropback_rate_pct"] = safe_divide(
            df["passing_attempts"], team_plays
        ) * 100
        
        df["rush_share_pct"] = safe_divide(
            df["rushing_attempts"], df["rushing_attempts"] + 5  # Normalize for QBs
        ) * 100
        
        # Red zone opportunity (from TDs)
        total_tds = df["passing_tds"] + df["rushing_tds"]
        df["redzone_opp_pct"] = (total_tds * 12).clip(0, 100)
        
        # Play volume (total plays normalized)
        df["play_volume_pct"] = (total_plays / 50 * 100).clip(0, 100)
        
        # Normalize components (use fitted bounds when set to avoid leakage)
        df["dropback_norm"] = self._percentile_normalize(df["dropback_rate_pct"], position="QB", component_key="dropback_rate_pct")
        df["rush_share_norm"] = self._percentile_normalize(df["rush_share_pct"], position="QB", component_key="rush_share_pct")
        df["redzone_opp_norm"] = self._percentile_normalize(df["redzone_opp_pct"], position="QB", component_key="redzone_opp_pct")
        df["play_volume_norm"] = self._percentile_normalize(df["play_volume_pct"], position="QB", component_key="play_volume_pct")
        
        # Calculate final utilization score
        df["utilization_score"] = (
            df["dropback_norm"] * weights["dropback_rate"] +
            df["rush_share_norm"] * weights["rush_attempt_share"] +
            df["redzone_opp_norm"] * weights["redzone_opportunity"] +
            df["play_volume_norm"] * weights["play_volume"]
        )
        
        # Store component values
        df["util_dropback_rate"] = df["dropback_rate_pct"]
        df["util_rush_share"] = df["rush_share_pct"]
        
        return df
    
    # NOTE: safe_divide is imported from src.utils.helpers
    
    def _percentile_normalize(self, series: pd.Series, position: str = None, component_key: str = None) -> pd.Series:
        """
        Normalize a series to 0-100. If bounds were fitted (fit_percentile_bounds), use them to avoid leakage.
        Otherwise use rank-based percentile within current data (legacy).
        """
        if series.isna().all() or len(series) == 0:
            return series
        # Auto-load persisted bounds if none in memory
        self._ensure_bounds_loaded()
        key = (position, component_key) if (position and component_key) else None
        bounds = self.position_percentiles.get(key) if key else None
        if bounds is not None and isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lo, hi = bounds
            if hi > lo:
                return ((series - lo) / (hi - lo) * 100).clip(0, 100)
            return series.clip(0, 100)
        return series.rank(pct=True, na_option="bottom") * 100

    _BOUNDS_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "utilization_percentile_bounds.json"

    def fit_percentile_bounds(self, train_df: pd.DataFrame, position: str, component_columns: list,
                               persist: bool = True) -> None:
        """
        Fit min/max (or 1st/99th percentile) per component on train data for consistent apply at serve.
        Store in self.position_percentiles keyed by (position, col).
        
        When persist=True (default), auto-saves bounds to disk so that the
        prediction pipeline can load them without retraining.
        """
        pos_df = train_df[train_df["position"] == position]
        if pos_df.empty:
            return
        for col in component_columns:
            if col not in pos_df.columns:
                continue
            s = pos_df[col].dropna()
            if len(s) < 10:
                continue
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            self.position_percentiles[(position, col)] = (float(lo), float(hi))
        
        if persist:
            save_percentile_bounds(self.position_percentiles, self._BOUNDS_DEFAULT_PATH)

    def _ensure_bounds_loaded(self) -> None:
        """Auto-load persisted percentile bounds if none are in memory."""
        if not self.position_percentiles and self._BOUNDS_DEFAULT_PATH.exists():
            self.position_percentiles = load_percentile_bounds(self._BOUNDS_DEFAULT_PATH)
    
    def get_utilization_tier(self, score: float, position: str) -> str:
        """
        Get the utilization tier description for a score.
        
        Returns tier like "Elite", "Strong", "Average", "Below Average", "Low"
        """
        if score >= 80:
            return "Elite"
        elif score >= 70:
            return "Strong"
        elif score >= 60:
            return "Average"
        elif score >= 50:
            return "Below Average"
        else:
            return "Low"
    
    def get_expected_ppg_range(self, score: float, position: str) -> Dict[str, float]:
        """
        Get expected PPG range based on utilization score and position.
        
        Based on historical data:
        - RB 60-69: ~12.2 PPG
        - RB 70-79: ~15.1 PPG
        - RB 80+: ~18+ PPG
        """
        ppg_ranges = {
            "RB": {
                (0, 50): {"min": 3.0, "avg": 6.5, "max": 10.0},
                (50, 60): {"min": 6.0, "avg": 9.5, "max": 13.0},
                (60, 70): {"min": 9.0, "avg": 12.2, "max": 16.0},
                (70, 80): {"min": 12.0, "avg": 15.1, "max": 20.0},
                (80, 100): {"min": 15.0, "avg": 18.5, "max": 28.0},
            },
            "WR": {
                (0, 50): {"min": 2.0, "avg": 5.0, "max": 9.0},
                (50, 60): {"min": 5.0, "avg": 8.0, "max": 12.0},
                (60, 70): {"min": 8.0, "avg": 11.0, "max": 15.0},
                (70, 80): {"min": 11.0, "avg": 14.5, "max": 19.0},
                (80, 100): {"min": 14.0, "avg": 18.0, "max": 26.0},
            },
            "TE": {
                (0, 50): {"min": 1.5, "avg": 4.0, "max": 7.0},
                (50, 60): {"min": 4.0, "avg": 6.5, "max": 10.0},
                (60, 70): {"min": 6.0, "avg": 9.0, "max": 13.0},
                (70, 80): {"min": 9.0, "avg": 12.0, "max": 17.0},
                (80, 100): {"min": 12.0, "avg": 16.0, "max": 24.0},
            },
            "QB": {
                (0, 50): {"min": 8.0, "avg": 12.0, "max": 16.0},
                (50, 60): {"min": 12.0, "avg": 15.0, "max": 19.0},
                (60, 70): {"min": 15.0, "avg": 18.0, "max": 23.0},
                (70, 80): {"min": 18.0, "avg": 21.0, "max": 27.0},
                (80, 100): {"min": 21.0, "avg": 25.0, "max": 35.0},
            },
        }
        
        position_ranges = ppg_ranges.get(position, ppg_ranges["RB"])
        
        for (low, high), ppg in position_ranges.items():
            if low <= score < high:
                return ppg
        
        return position_ranges[(80, 100)]  # Default to highest tier


def calculate_utilization_scores(player_df: pd.DataFrame, 
                                  team_df: pd.DataFrame = None,
                                  weights: Optional[Dict] = None,
                                  percentile_bounds: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None) -> pd.DataFrame:
    """
    Convenience function to calculate utilization scores.
    
    Args:
        player_df: DataFrame with player weekly stats
        team_df: Optional DataFrame with team stats
        weights: Optional position -> component -> weight dict (from utilization_weight_optimizer)
        percentile_bounds: Optional (position, component_col) -> (lo, hi) from train (avoids leakage at test/serve)
        
    Returns:
        DataFrame with utilization_score column added
    """
    calculator = UtilizationScoreCalculator(weights=weights, position_percentiles=percentile_bounds)
    
    if team_df is None:
        team_df = pd.DataFrame()
    
    return calculator.calculate_all_scores(player_df, team_df)


def recalculate_utilization_with_weights(df: pd.DataFrame, 
                                         weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Recompute utilization_score from existing _norm columns using new weights.
    Use when components exist but weights were optimized from data.
    """
    from config.settings import UTILIZATION_WEIGHTS
    
    result = df.copy()
    norm_to_key = {
        "RB": {"snap_share_norm": "snap_share", "rush_share_norm": "rush_share",
               "target_share_norm": "target_share", "redzone_share_norm": "redzone_share",
               "touch_share_norm": "touch_share"},
        "WR": {"target_share_norm": "target_share", "air_yards_norm": "air_yards_share",
               "snap_share_norm": "snap_share", "redzone_targets_norm": "redzone_targets",
               "route_part_norm": "route_participation"},
        "TE": {"target_share_norm": "target_share", "snap_share_norm": "snap_share",
               "redzone_targets_norm": "redzone_targets", "air_yards_norm": "air_yards_share",
               "inline_rate_norm": "inline_rate"},
        "QB": {"dropback_norm": "dropback_rate", "rush_share_norm": "rush_attempt_share",
               "redzone_opp_norm": "redzone_opportunity", "play_volume_norm": "play_volume"},
    }
    for position in ["QB", "RB", "WR", "TE"]:
        mask = result["position"] == position
        if not mask.any():
            continue
        pos_weights = weights.get(position, UTILIZATION_WEIGHTS.get(position, {}))
        mapping = norm_to_key.get(position, {})
        score = pd.Series(0.0, index=result.index)
        for norm_col, key in mapping.items():
            if norm_col in result.columns and key in pos_weights:
                score = score + result[norm_col].fillna(0) * pos_weights[key]
        result.loc[mask, "utilization_score"] = score[mask].clip(0, 100)
    return result

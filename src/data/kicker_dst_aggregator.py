"""
Kicker and DST (Defense/Special Teams) data aggregation from play-by-play data.

Aggregates weekly kicker stats (FG/XP) and team-level DST stats (sacks, INTs,
fumble recoveries, TDs, points allowed) from nfl-data-py PBP data.

PBP data is available from 1999 onward, so this enables K/DST support
even though nfl_data_py.import_weekly_data does not include kickers.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import SCORING_KICKER, SCORING_DST


def _load_pbp(seasons: List[int]) -> pd.DataFrame:
    """Load PBP data for given seasons, with error handling per season."""
    import nfl_data_py as nfl
    all_dfs = []
    for season in seasons:
        try:
            df = nfl.import_pbp_data([season])
            if not df.empty:
                all_dfs.append(df)
                print(f"  PBP {season}: {len(df)} plays")
        except Exception as e:
            print(f"  PBP {season}: skipped ({e})")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def aggregate_kicker_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly kicker stats from PBP data.

    Returns DataFrame with columns:
        player_id, name, position, team, season, week, opponent, home_away,
        fg_made, fg_missed, fg_att, fg_0_39, fg_40_49, fg_50_plus,
        xp_made, xp_missed, xp_att, fantasy_points
    """
    if pbp.empty or "play_type" not in pbp.columns:
        return pd.DataFrame()

    rows = []

    # --- Field goals ---
    fg = pbp[pbp["play_type"] == "field_goal"].copy()
    if not fg.empty and "kicker_player_id" in fg.columns:
        fg["fg_made"] = (fg["field_goal_result"] == "made").astype(int)
        fg["fg_missed"] = (fg["field_goal_result"] != "made").astype(int)
        dist = fg["kick_distance"].fillna(0)
        fg["fg_0_39"] = ((dist <= 39) & (fg["fg_made"] == 1)).astype(int)
        fg["fg_40_49"] = ((dist >= 40) & (dist <= 49) & (fg["fg_made"] == 1)).astype(int)
        fg["fg_50_plus"] = ((dist >= 50) & (fg["fg_made"] == 1)).astype(int)

        fg_agg = fg.groupby(
            ["kicker_player_id", "kicker_player_name", "posteam", "season", "week"]
        ).agg(
            fg_made=("fg_made", "sum"),
            fg_missed=("fg_missed", "sum"),
            fg_0_39=("fg_0_39", "sum"),
            fg_40_49=("fg_40_49", "sum"),
            fg_50_plus=("fg_50_plus", "sum"),
        ).reset_index()
        fg_agg = fg_agg.rename(columns={
            "kicker_player_id": "player_id",
            "kicker_player_name": "name",
            "posteam": "team",
        })
        fg_agg["fg_att"] = fg_agg["fg_made"] + fg_agg["fg_missed"]
        rows.append(fg_agg)

    # --- Extra points ---
    xp = pbp[pbp["play_type"] == "extra_point"].copy()
    if not xp.empty and "kicker_player_id" in xp.columns:
        xp["xp_made"] = (xp["extra_point_result"] == "good").astype(int)
        xp["xp_missed"] = (xp["extra_point_result"] != "good").astype(int)

        xp_agg = xp.groupby(
            ["kicker_player_id", "kicker_player_name", "posteam", "season", "week"]
        ).agg(
            xp_made=("xp_made", "sum"),
            xp_missed=("xp_missed", "sum"),
        ).reset_index()
        xp_agg = xp_agg.rename(columns={
            "kicker_player_id": "player_id",
            "kicker_player_name": "name",
            "posteam": "team",
        })
        xp_agg["xp_att"] = xp_agg["xp_made"] + xp_agg["xp_missed"]
        rows.append(xp_agg)

    if not rows:
        return pd.DataFrame()

    # Merge FG and XP stats
    if len(rows) == 2:
        kicker = rows[0].merge(
            rows[1], on=["player_id", "name", "team", "season", "week"], how="outer"
        ).fillna(0)
    else:
        kicker = rows[0].copy()
        for col in ["xp_made", "xp_missed", "xp_att"]:
            if col not in kicker.columns:
                kicker[col] = 0

    # Ensure numeric columns are int
    int_cols = ["fg_made", "fg_missed", "fg_att", "fg_0_39", "fg_40_49", "fg_50_plus",
                "xp_made", "xp_missed", "xp_att"]
    for c in int_cols:
        if c in kicker.columns:
            kicker[c] = kicker[c].fillna(0).astype(int)

    # Calculate fantasy points
    kicker["fantasy_points"] = (
        kicker["fg_0_39"] * SCORING_KICKER["fg_0_39"]
        + kicker["fg_40_49"] * SCORING_KICKER["fg_40_49"]
        + kicker["fg_50_plus"] * SCORING_KICKER["fg_50_plus"]
        + kicker["xp_made"] * SCORING_KICKER["xp_made"]
        + kicker["fg_missed"] * SCORING_KICKER["fg_missed"]
        + kicker["xp_missed"] * SCORING_KICKER["xp_missed"]
    ).round(1)

    kicker["position"] = "K"

    # Add opponent and home_away from PBP game data
    if "defteam" in pbp.columns and "home_team" in pbp.columns:
        game_info = pbp.dropna(subset=["posteam", "defteam"]).groupby(
            ["posteam", "season", "week"]
        ).agg(
            opponent=("defteam", "first"),
            home_team=("home_team", "first"),
        ).reset_index()
        kicker = kicker.merge(
            game_info.rename(columns={"posteam": "team"}),
            on=["team", "season", "week"],
            how="left",
        )
        kicker["home_away"] = np.where(kicker["team"] == kicker["home_team"], "home", "away")
        kicker = kicker.drop(columns=["home_team"], errors="ignore")
    else:
        kicker["opponent"] = ""
        kicker["home_away"] = "unknown"

    return kicker


def _points_allowed_score(pa: float) -> float:
    """Calculate DST fantasy points from points allowed bracket."""
    if pa <= 0:
        return SCORING_DST["pa_0"]
    elif pa <= 6:
        return SCORING_DST["pa_1_6"]
    elif pa <= 13:
        return SCORING_DST["pa_7_13"]
    elif pa <= 20:
        return SCORING_DST["pa_14_20"]
    elif pa <= 27:
        return SCORING_DST["pa_21_27"]
    elif pa <= 34:
        return SCORING_DST["pa_28_34"]
    else:
        return SCORING_DST["pa_35_plus"]


def aggregate_dst_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly DST stats from PBP data.

    Returns DataFrame with columns:
        player_id (team code), name (team name), position, team, season, week,
        opponent, home_away, sacks, interceptions, fumble_recoveries,
        defensive_td, special_teams_td, safety, blocked_kick, points_allowed,
        fantasy_points
    """
    if pbp.empty or "defteam" not in pbp.columns:
        return pd.DataFrame()

    plays = pbp[pbp["defteam"].notna()].copy()

    # Aggregate defensive stats per team per week
    dst = plays.groupby(["defteam", "season", "week"]).agg(
        sacks=("sack", "sum"),
        interceptions=("interception", "sum"),
        fumble_recoveries=("fumble_lost", "sum"),
        safety=("safety", "sum"),
    ).reset_index()

    # Defensive TDs: touchdowns scored by the defense
    if "td_team" in plays.columns and "touchdown" in plays.columns:
        def_tds = plays[
            (plays["touchdown"] == 1) & (plays["td_team"] == plays["defteam"])
        ].groupby(["defteam", "season", "week"]).size().reset_index(name="defensive_td")
        dst = dst.merge(def_tds, on=["defteam", "season", "week"], how="left")
    else:
        dst["defensive_td"] = 0

    # Special teams TDs (kickoff/punt return TDs)
    if "return_touchdown" in plays.columns:
        st_tds = plays[plays["return_touchdown"] == 1].groupby(
            ["posteam", "season", "week"]
        ).size().reset_index(name="special_teams_td")
        st_tds = st_tds.rename(columns={"posteam": "defteam"})
        dst = dst.merge(st_tds, on=["defteam", "season", "week"], how="left")
    else:
        dst["special_teams_td"] = 0

    # Blocked kicks
    if "field_goal_result" in plays.columns:
        blocked = plays[plays["field_goal_result"] == "blocked"].groupby(
            ["defteam", "season", "week"]
        ).size().reset_index(name="blocked_kick")
        dst = dst.merge(blocked, on=["defteam", "season", "week"], how="left")
    else:
        dst["blocked_kick"] = 0

    # Points allowed: sum of opponent scoring
    if "posteam_score_post" in plays.columns:
        # Get max score by the offense (posteam) in each game for the defense
        pa = plays.groupby(["defteam", "season", "week"]).agg(
            points_allowed=("posteam_score_post", "max"),
        ).reset_index()
        dst = dst.merge(pa, on=["defteam", "season", "week"], how="left")
    else:
        dst["points_allowed"] = 21  # default

    dst = dst.fillna(0)

    # Ensure integer types
    int_cols = ["sacks", "interceptions", "fumble_recoveries", "defensive_td",
                "special_teams_td", "safety", "blocked_kick", "points_allowed"]
    for c in int_cols:
        if c in dst.columns:
            dst[c] = dst[c].astype(int)

    # Calculate fantasy points
    dst["fantasy_points"] = (
        dst["sacks"] * SCORING_DST["sack"]
        + dst["interceptions"] * SCORING_DST["interception"]
        + dst["fumble_recoveries"] * SCORING_DST["fumble_recovery"]
        + dst["safety"] * SCORING_DST["safety"]
        + dst["defensive_td"] * SCORING_DST["defensive_td"]
        + dst["special_teams_td"] * SCORING_DST["special_teams_td"]
        + dst["blocked_kick"] * SCORING_DST["blocked_kick"]
        + dst["points_allowed"].apply(_points_allowed_score)
    ).round(1)

    # DST uses team code as player_id and name
    dst = dst.rename(columns={"defteam": "team"})
    dst["player_id"] = dst["team"] + "_DST"
    dst["name"] = dst["team"] + " Defense"
    dst["position"] = "DST"

    # Add opponent and home_away
    if "posteam" in pbp.columns and "home_team" in pbp.columns:
        game_info = pbp.dropna(subset=["defteam", "posteam"]).groupby(
            ["defteam", "season", "week"]
        ).agg(
            opponent=("posteam", "first"),
            home_team=("home_team", "first"),
        ).reset_index().rename(columns={"defteam": "team"})
        dst = dst.merge(game_info, on=["team", "season", "week"], how="left")
        dst["home_away"] = np.where(dst["team"] == dst["home_team"], "home", "away")
        dst = dst.drop(columns=["home_team"], errors="ignore")
    else:
        dst["opponent"] = ""
        dst["home_away"] = "unknown"

    return dst


def load_kicker_dst_data(seasons: List[int]) -> pd.DataFrame:
    """
    Load and aggregate kicker + DST weekly data from PBP for given seasons.

    Returns a combined DataFrame with both K and DST rows, compatible with
    the player_weekly_stats schema.
    """
    print(f"Loading K/DST data from PBP for seasons: {seasons}")
    pbp = _load_pbp(seasons)
    if pbp.empty:
        print("  No PBP data available")
        return pd.DataFrame()

    kicker = aggregate_kicker_stats(pbp)
    dst = aggregate_dst_stats(pbp)

    print(f"  Kicker: {len(kicker)} player-weeks, {kicker['player_id'].nunique() if not kicker.empty else 0} kickers")
    print(f"  DST: {len(dst)} team-weeks, {dst['player_id'].nunique() if not dst.empty else 0} defenses")

    dfs = []
    if not kicker.empty:
        dfs.append(kicker)
    if not dst.empty:
        dfs.append(dst)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure all expected columns exist
    for col in ["player_id", "name", "position", "team", "season", "week",
                "opponent", "home_away", "fantasy_points"]:
        if col not in combined.columns:
            combined[col] = "" if col in ("player_id", "name", "position", "team", "opponent", "home_away") else 0

    return combined

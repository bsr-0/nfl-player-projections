"""
Multi-season, team-aware feature assembly for a preseason projection
candidate (GAPS.md: "multi-season, team-aware preseason model" follow-up
to the PreseasonProjector gap audit).

PreseasonProjector's existing `_build_season_pairs` uses only the single
immediately-prior season's per-game stat rates, and has zero team
context anywhere (verified by grep -- no `team`/`dest_team`/
`team_changed` reference in the whole file). This module builds a richer
season-pair dataset for a real, honest comparison against that baseline:

- Real multi-year history: each of the player's own prior 1/2/3 seasons
  (not just season N) as separate columns, plus explicit trend features
  (year-over-year PPG delta, a recency-weighted multi-year average,
  games-played stability).
- Destination-team context: the team the player is actually on entering
  the target season, and that team's own 3-season rolling historical
  positional usage (targets/game, carries/game) -- the same real,
  leakage-safe logic already used by the weekly model's
  `FeatureEngineer._add_dest_team_pos_profiles`, reimplemented here
  directly against full player_weekly_stats history (that function
  needs weekly rows with a `week` column; season-pair rows don't have
  one, so the aggregation is done once up front over the full history
  instead of reusing the method call directly).

Also fixes a real, live bug found while building this: PreseasonProjector's
`_build_season_pairs` LEFT JOINs against the `rosters` table for
`years_exp`, but that table is completely empty (0 rows) in this DB --
`weekly_rosters_v2` has real data but only covers 2024-2025 (too sparse
for the full 2006+ training history). Real experience is computed here
instead as "number of distinct prior seasons this player_id appears in
player_weekly_stats" -- derivable from the same full-history table
already being read, no join needed, and correct for the entire history
this project actually has.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

MIN_GAMES = 6
# Undrafted encoding, matching advanced_rookie_injury so the two feature
# sources cannot disagree about what "undrafted" looks like.
UNDRAFTED_ROUND = 8
UNDRAFTED_PICK = 400

LOOKBACK_YEARS = 3


def _load_full_history(db) -> pd.DataFrame:
    """All player-weeks for offensive skill positions, full history."""
    with db._get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT pws.player_id, p.name AS player_name, p.position,
                   p.birth_date, pws.team, pws.season, pws.week,
                   pws.fantasy_points, pws.passing_yards, pws.passing_tds,
                   pws.interceptions, pws.rushing_yards, pws.rushing_attempts,
                   pws.targets, pws.receptions, pws.receiving_yards,
                   pws.air_yards, pws.passing_attempts, pws.passing_completions,
                   COALESCE(us.snap_share, 0) AS snap_share,
                   COALESCE(us.target_share, 0) AS target_share,
                   COALESCE(us.rush_share, 0) AS rush_share
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            LEFT JOIN utilization_scores us
              ON pws.player_id = us.player_id
             AND pws.season = us.season AND pws.week = us.week
            WHERE p.position IN ('QB', 'RB', 'WR', 'TE') AND pws.week <= 18
            """,
            conn,
        )


def _season_aggregates(history: pd.DataFrame) -> pd.DataFrame:
    """Per-player-season aggregate rates, same shape as
    PreseasonProjector._build_season_pairs's per-season SELECT, but
    computed once for every season (not just "prior") so it can be
    reused across multiple lookback years.

    dropna=False is required here: birth_date is NULL for a large share
    of rows (100% of 2025, 87% of 2024 in this DB -- recent players lag
    on that backfill), and pandas groupby silently DROPS every row where
    any key is NaN by default. PreseasonProjector's original SQL
    GROUP BY doesn't have this problem (SQL groups NULLs together rather
    than dropping them), so this bug was introduced translating the same
    logic to pandas, not present in the original. Confirmed directly:
    without dropna=False, season 2025 vanished entirely and 2024 was
    undercounted to 5 QB rows instead of the real count.
    """
    g = history.groupby(["player_id", "player_name", "position", "birth_date", "season"], dropna=False)
    agg = g.agg(
        games_played=("fantasy_points", "size"),
        ppg=("fantasy_points", "mean"),
        passing_yards_pg=("passing_yards", "mean"),
        passing_tds_pg=("passing_tds", "mean"),
        interceptions_pg=("interceptions", "mean"),
        rushing_yards_pg=("rushing_yards", "mean"),
        carries_pg=("rushing_attempts", "mean"),
        targets_pg=("targets", "mean"),
        receptions_pg=("receptions", "mean"),
        receiving_yards_pg=("receiving_yards", "mean"),
        air_yards_pg=("air_yards", "mean"),
        snap_share=("snap_share", "mean"),
        target_share=("target_share", "mean"),
        rush_share=("rush_share", "mean"),
        pass_att_sum=("passing_attempts", "sum"),
        pass_cmp_sum=("passing_completions", "sum"),
    ).reset_index()
    agg["completion_pct"] = np.where(
        agg["pass_att_sum"] > 0, 100.0 * agg["pass_cmp_sum"] / agg["pass_att_sum"], 0.0
    )
    agg = agg.drop(columns=["pass_att_sum", "pass_cmp_sum"])
    agg = agg[agg["games_played"] >= MIN_GAMES]
    return agg


def _team_for_season(history: pd.DataFrame) -> pd.DataFrame:
    """Each player's team for each season they played -- their most
    common team that season (handles in-season trades by majority week
    count, same convention as the rest of this codebase's team-change
    features, which also derive team purely from player_weekly_stats
    rather than a roster snapshot)."""
    return (
        history.groupby(["player_id", "season", "team"])
        .size()
        .reset_index(name="n_weeks")
        .sort_values("n_weeks", ascending=False)
        .drop_duplicates(["player_id", "season"])
        [["player_id", "season", "team"]]
    )


def _destination_team_profiles(history: pd.DataFrame) -> pd.DataFrame:
    """Team x position x season positional usage, 3-season rolling mean,
    shift(1) leakage-safe -- same logic as
    FeatureEngineer._add_dest_team_pos_profiles, reimplemented over
    season-level (not week-level) data since season-pair rows have no
    `week` column to group by."""
    agg = (
        history.groupby(["team", "position", "season"])
        .agg(targets_sum=("targets", "sum"), carries_sum=("rushing_attempts", "sum"),
             games=("week", "nunique"))
        .reset_index()
    )
    agg["tgt_pg"] = agg["targets_sum"] / agg["games"].clip(lower=1)
    agg["carry_pg"] = agg["carries_sum"] / agg["games"].clip(lower=1)
    agg = agg.sort_values(["team", "position", "season"])
    for col in ["tgt_pg", "carry_pg"]:
        agg[f"dest_hist_{col}"] = (
            agg.groupby(["team", "position"])[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
    return agg[["team", "position", "season", "dest_hist_tgt_pg", "dest_hist_carry_pg"]]


def career_static_by_player(db) -> pd.DataFrame:
    """Per-player attributes fixed before any NFL season: draft capital and
    college pedigree.

    Populated for EVERY player, not just rookies. They exist for everyone,
    their predictive weight simply decays with experience, and a tree finds
    that itself given `years_of_history`. Restricting them to rookies would
    also make them unusable for the only arms that need them most.

    Undrafted is carried as its own flag rather than a sentinel pick: ~39% of
    players who reach the league are undrafted, and `draft_pick = 400` would
    be read as an ordinal position rather than a distinct state.
    """
    from src.features.college_conference import conference_for, is_power5

    with db._get_connection() as conn:
        picks = pd.read_sql_query(
            """SELECT player_id, draft_season, draft_round, college,
                      MIN(draft_pick) AS draft_pick
               FROM draft_picks_v2
               WHERE player_id IS NOT NULL AND player_id != ''
               GROUP BY player_id""",
            conn,
        )
        if picks.empty:
            return pd.DataFrame(columns=["player_id"])
        values = pd.read_sql_query(
            "SELECT pick, otc AS draft_pick_value FROM draft_values", conn,
        )
    picks["_clamped"] = picks["draft_pick"].clip(upper=262)
    picks = picks.merge(values, left_on="_clamped", right_on="pick", how="left")
    picks["is_power5"] = [
        is_power5(conference_for(c, s))
        for c, s in zip(picks["college"], picks["draft_season"])
    ]
    return picks[["player_id", "draft_season", "draft_round", "draft_pick",
                  "draft_pick_value", "is_power5"]]


def _cold_start_rows(curr_totals: pd.DataFrame, season_agg: pd.DataFrame,
                     history: pd.DataFrame, target: int) -> pd.DataFrame:
    """Rows for players whose FIRST NFL season is `target`.

    They have no prior-season aggregate to anchor on, so the normal path drops
    them entirely -- measured, 0 of 38 true 2025 WR rookies reached the
    feature matrix. Their `_y1/_y2/_y3` lags are left NaN rather than zeroed:
    a zero would assert the player posted 0 PPG last season, when in fact
    there was no last season, and the Ridge arms' in-fold imputer plus
    indicator columns can then represent that honestly.
    """
    # Prior existence is checked against RAW history, not season_agg. season_agg
    # drops player-seasons under MIN_GAMES, so a player with 3 games last year
    # has no aggregate row and would be misread as a rookie -- that alone
    # inflated the 2025 WR cold-start population from 38 to 82. A thin prior
    # season is not the same as no NFL career.
    #
    # Those sub-threshold players stay dropped, exactly as before this change.
    # That is a real pre-existing population gap, but widening it here would
    # silently mix two different fixes into one measurement.
    prior_ids = set(history.loc[history["season"] < target, "player_id"])
    rookie_ids = sorted(set(curr_totals["player_id"]) - prior_ids)
    if not rookie_ids:
        return pd.DataFrame()

    meta = (history[(history["season"] == target)
                    & (history["player_id"].isin(rookie_ids))]
            [["player_id", "player_name", "position"]].drop_duplicates("player_id"))
    out = meta.copy()
    out["season"] = target - 1          # matches the anchor convention
    out["target_season"] = target
    out["years_exp"] = 0
    out["is_cold_start"] = 1
    return out


def build_multiyear_season_pairs(db, seasons: List[int],
                                  lookback_years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    """Build (multi-year features, next-season target) pairs with real
    destination-team context. One row per (player, target season)."""
    history = _load_full_history(db)
    if history.empty:
        return pd.DataFrame()

    season_agg = _season_aggregates(history)
    team_by_season = _team_for_season(history)
    dest_profiles = _destination_team_profiles(history)

    # Real experience: count of distinct prior seasons this player
    # appears in, computed once over full history (fixes the
    # always-empty `rosters` table join in PreseasonProjector).
    season_agg = season_agg.sort_values(["player_id", "season"])
    season_agg["years_exp"] = season_agg.groupby("player_id").cumcount()

    season_list = sorted(seasons)
    frames = []
    for i in range(len(season_list) - 1):
        target = season_list[i + 1]
        curr_totals = history[history["season"] == target].groupby("player_id")["fantasy_points"].sum()
        curr_totals = curr_totals.rename("season_total").reset_index()
        if curr_totals.empty:
            continue

        # Season N (immediately prior to target) is the anchor row --
        # keeps player_name/position/birth_date/years_exp from there.
        anchor = season_agg[season_agg["season"] == target - 1].copy()
        if anchor.empty:
            continue

        row = anchor.rename(columns={
            c: f"{c}_y1" for c in anchor.columns
            if c not in ("player_id", "player_name", "position", "birth_date", "season", "years_exp")
        })
        row = row.rename(columns={"games_played_y1": "games_played_y1"})
        row["target_season"] = target

        # Prior seasons 2 and 3 (season N-1, N-2), suffixed _y2/_y3.
        for lag in range(2, lookback_years + 1):
            prior_season = target - lag
            lag_df = season_agg[season_agg["season"] == prior_season][
                ["player_id"] + [c for c in season_agg.columns
                                  if c not in ("player_id", "player_name", "position", "birth_date", "season", "years_exp")]
            ].copy()
            lag_df = lag_df.rename(columns={
                c: f"{c}_y{lag}" for c in lag_df.columns if c != "player_id"
            })
            row = row.merge(lag_df, on="player_id", how="left")

        # Trend features: year-over-year PPG delta, recency-weighted
        # multi-year average, durability across available years.
        ppg_cols = [c for c in [f"ppg_y{y}" for y in range(1, lookback_years + 1)] if c in row.columns]
        games_cols = [c for c in [f"games_played_y{y}" for y in range(1, lookback_years + 1)] if c in row.columns]
        if "ppg_y2" in row.columns:
            row["ppg_trend_y1_y2"] = (row["ppg_y1"] - row["ppg_y2"]).fillna(0.0)
        else:
            row["ppg_trend_y1_y2"] = 0.0
        weights = np.array([0.6, 0.25, 0.15])[: len(ppg_cols)]
        weights = weights / weights.sum()
        ppg_matrix = row[ppg_cols].to_numpy(dtype=float)
        have = np.isfinite(ppg_matrix)
        w_matrix = np.where(have, weights, 0.0)
        w_sum = w_matrix.sum(axis=1)
        w_sum[w_sum == 0] = 1.0
        row["ppg_recency_weighted"] = np.nan_to_num(
            (np.nan_to_num(ppg_matrix) * w_matrix).sum(axis=1) / w_sum
        )
        row["years_of_history"] = row[games_cols].notna().sum(axis=1) if games_cols else 0
        row["games_played_std"] = row[games_cols].std(axis=1, ddof=0).fillna(0.0) if len(games_cols) > 1 else 0.0

        # Destination team: the team this player is with entering the
        # TARGET season (real information a preseason projection has
        # access to -- known once free agency/depth charts settle,
        # before Week 1 -- not leakage of the target itself).
        dest_team = team_by_season[team_by_season["season"] == target][["player_id", "team"]]
        dest_team = dest_team.rename(columns={"team": "dest_team"})
        row = row.merge(dest_team, on="player_id", how="left")

        prior_team = team_by_season[team_by_season["season"] == target - 1][["player_id", "team"]]
        prior_team = prior_team.rename(columns={"team": "prior_team"})
        row = row.merge(prior_team, on="player_id", how="left")
        row["team_changed"] = (
            row["dest_team"].notna() & row["prior_team"].notna()
            & (row["dest_team"] != row["prior_team"])
        ).astype(int)

        dest_prof = dest_profiles[dest_profiles["season"] == target].rename(
            columns={"team": "dest_team"}
        )
        row = row.merge(dest_prof, on=["dest_team", "position"], how="left")
        row["dest_hist_tgt_pg"] = row["dest_hist_tgt_pg"].fillna(0.0) * row["team_changed"]
        row["dest_hist_carry_pg"] = row["dest_hist_carry_pg"].fillna(0.0) * row["team_changed"]

        row["is_cold_start"] = 0

        # Rookies, appended AFTER the veteran path so every derived column
        # above already exists on `row` and concat aligns them as NaN rather
        # than silently inventing values.
        cold = _cold_start_rows(curr_totals, season_agg, history, target)
        if not cold.empty:
            cold = cold.merge(dest_team, on="player_id", how="left")
            cold["prior_team"] = pd.NA
            cold["team_changed"] = 0      # never had a team to change from
            cold = cold.merge(dest_prof, on=["dest_team", "position"], how="left")
            # dest_hist_* are multiplied by team_changed on the veteran path,
            # which would zero them here. For a rookie the destination team's
            # positional usage is the whole point, so it is kept as-is.
            cold["dest_hist_tgt_pg"] = cold["dest_hist_tgt_pg"].fillna(0.0)
            cold["dest_hist_carry_pg"] = cold["dest_hist_carry_pg"].fillna(0.0)
            cold["years_of_history"] = 0
            row = pd.concat([row, cold], ignore_index=True)

        row = row.merge(curr_totals, on="player_id", how="inner")
        frames.append(row)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    # Career-static attributes for EVERY row, veteran and rookie alike.
    static = career_static_by_player(db)
    if not static.empty:
        out = out.merge(static, on="player_id", how="left")
        out["is_undrafted"] = out["draft_season"].isna().astype(int)
        # Resolved here, where the NaN still exists to be detected: the
        # downstream Ridge imputer would otherwise fill draft_pick and make an
        # undrafted player indistinguishable from a real late-round selection.
        out["draft_pick"] = out["draft_pick"].fillna(UNDRAFTED_PICK)
        out["draft_round"] = out["draft_round"].fillna(UNDRAFTED_ROUND)
        out["draft_pick_value"] = out["draft_pick_value"].fillna(0.0)
        out["is_power5"] = out["is_power5"].fillna(0).astype(int)
    return out

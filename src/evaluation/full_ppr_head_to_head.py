"""Checked full-PPR truth and cohorts for a served-model versus Plan A test.

Plan A's frozen prediction covers eight components. Its declared full-PPR
extension predicts zero fumbles lost and zero two-point conversions. This
module changes only the evaluation target; it never uses observed extra
components to adjust either arm's prediction.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import SCORING
from src.evaluation.paired_ppr_comparison import KEY, check_keys, finite_columns
from src.evaluation.ppr_truth import TARGETS, checked_truth
from src.utils.helpers import calculate_fantasy_points_df

EXTRA_COMPONENTS = ("fumbles_lost", "two_point_conversions")
FULL_COMPONENTS = (*TARGETS, *EXTRA_COMPONENTS)
FULL_SCORING_DEFINITION = "raw_ten_component_full_ppr_plan_a_zero_extra_forecast"


def scoring_weights() -> dict[str, float]:
    return {target: SCORING[target] for target in FULL_COMPONENTS}


def load_context(db_path: Path, seasons: list[int]) -> tuple[pd.DataFrame, ...]:
    """Read historical comparison context without mutating SQLite."""
    if not seasons or seasons != sorted(set(seasons)):
        raise ValueError("seasons must be nonempty, unique, and increasing")
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        low, high = min(seasons), max(seasons)
        stats = pd.read_sql_query(
            "SELECT player_id,season,week,team,fumbles_lost,two_point_conversions,"
            "fantasy_points FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
            conn, params=(low, high))
        canonical = pd.read_sql_query(
            "SELECT player_id,season,week,team,position,has_stats_row,has_snap_row,"
            "offense_snaps,fantasy_points FROM canonical_player_weeks "
            "WHERE season BETWEEN ? AND ?", conn, params=(low, high))
        history = pd.read_sql_query(
            "SELECT player_id,MIN(season) AS first_canonical_season "
            "FROM canonical_player_weeks GROUP BY player_id", conn)
        draft = pd.read_sql_query(
            "SELECT player_id,MIN(draft_season) AS draft_season FROM draft_picks_v2 "
            "WHERE player_id IS NOT NULL AND player_id != '' GROUP BY player_id", conn)
    finally:
        conn.close()
    return stats, canonical, history, draft


def checked_full_truth(raw: pd.DataFrame, stats: pd.DataFrame,
                       canonical: pd.DataFrame, history: pd.DataFrame,
                       draft: pd.DataFrame) -> pd.DataFrame:
    """Score the ten raw components and attach auditable evaluation cohorts."""
    check_keys(raw, "Plan A raw truth")
    for name, frame, key in (
        ("weekly stats", stats, ["player_id", "season", "week"]),
        ("canonical weeks", canonical, ["player_id", "season", "week"]),
        ("canonical history", history, ["player_id"]),
        ("draft history", draft, ["player_id"]),
    ):
        if set(key) - set(frame) or frame.duplicated(key).any():
            raise ValueError(f"{name}: missing or duplicate identity")
    if "fold" not in raw or raw.fold.isna().any():
        raise ValueError("Plan A raw truth needs a nonnull fold")
    eight = checked_truth(raw, raw[KEY + ["fold"]])
    if "actual_ppr" in raw:
        recorded = finite_columns(raw, ["actual_ppr"], "Plan A raw truth")[:, 0]
        raw_keyed = raw[KEY + ["actual_ppr"]].merge(
            eight[KEY + ["actual_ppr"]], on=KEY, validate="one_to_one", suffixes=("_saved", "_checked"))
        if len(raw_keyed) != len(raw) or not np.allclose(
                raw_keyed.actual_ppr_saved, raw_keyed.actual_ppr_checked, rtol=0, atol=1e-9):
            raise ValueError("Plan A saved eight-component label differs from raw scoring")
        if not np.isfinite(recorded).all():
            raise ValueError("nonfinite Plan A eight-component label")
    identity = KEY
    context_cols = identity + ["has_stats_row", "has_snap_row", "offense_snaps", "fantasy_points"]
    frame = eight.merge(canonical[context_cols], on=identity, how="left",
                        validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Plan A player-week missing from canonical panel")
    frame = frame.drop(columns="_merge")
    frame = frame.merge(stats[["player_id", "season", "week", "team", *EXTRA_COMPONENTS,
                               "fantasy_points"]],
                        on=["player_id", "season", "week", "team"], how="left",
                        validate="one_to_one", indicator=True,
                        suffixes=("_canonical", "_weekly"))
    has_stats = frame.has_stats_row.eq(1)
    if not frame._merge.eq("both").equals(has_stats):
        raise ValueError("canonical stats-row flag disagrees with exact weekly stats identity")
    no_stats = ~has_stats
    if frame.loc[no_stats, TARGETS].ne(0).any().any():
        raise ValueError("missing weekly stats row has nonzero raw PPR component")
    if frame.loc[has_stats, list(EXTRA_COMPONENTS)].isna().any().any():
        raise ValueError("recorded extra PPR component is null")
    for component in EXTRA_COMPONENTS:
        frame.loc[no_stats, component] = 0.0
        values = finite_columns(frame, [component], "full PPR truth")[:, 0]
        if ((values < 0) | (values != np.floor(values))).any():
            raise ValueError(f"invalid nonnegative integer {component}")
    frame["actual_full_ppr"] = calculate_fantasy_points_df(frame[list(FULL_COMPONENTS)])
    # The stored weekly/canonical fantasy_points uses source rounding; the
    # component scorer is authoritative, but a larger gap means bad lineage.
    for label in ("fantasy_points_canonical", "fantasy_points_weekly"):
        values = pd.to_numeric(frame.loc[has_stats, label], errors="coerce").to_numpy(float)
        actual = frame.loc[has_stats, "actual_full_ppr"].to_numpy(float)
        if not np.isfinite(values).all() or (np.abs(values - actual) > 0.05).any():
            raise ValueError(f"{label} differs from raw ten-component PPR beyond rounding")
    missing_weekly_canonical = pd.to_numeric(
        frame.loc[no_stats, "fantasy_points_canonical"], errors="coerce")
    # The canonical panel records these as NULL, while the checked share
    # panel's eight raw components explicitly record zero. A finite nonzero
    # canonical label would contradict that convention.
    if missing_weekly_canonical.dropna().ne(0).any():
        raise ValueError("missing weekly stats row has nonzero canonical PPR")
    frame = frame.merge(history, on="player_id", how="left", validate="many_to_one")
    frame = frame.merge(draft, on="player_id", how="left", validate="many_to_one")
    if frame.first_canonical_season.isna().any():
        raise ValueError("canonical debut season missing")
    frame["draft_rookie"] = frame.draft_season.eq(frame.season)
    frame["returning_player"] = frame.first_canonical_season.lt(frame.season)
    frame["first_observed_season"] = frame.first_canonical_season.eq(frame.season)
    if (frame.draft_rookie & frame.returning_player).any():
        raise ValueError("draft rookie has an earlier canonical season")
    snap = pd.to_numeric(frame.offense_snaps, errors="coerce")
    known_snap = frame.has_snap_row.eq(1)
    if snap[known_snap].isna().any() or (snap[known_snap] < 0).any():
        raise ValueError("confirmed snap row has invalid offense snaps")
    frame["snap_segment"] = np.where(known_snap, np.where(snap.gt(0), "nonzero", "zero"), "unknown")
    frame["has_weekly_stats_row"] = has_stats
    frame["extra_component_points"] = (frame.actual_full_ppr - frame.actual_ppr).round(2)
    return frame[KEY + ["fold", *FULL_COMPONENTS, "actual_ppr", "actual_full_ppr",
                        "extra_component_points", "negative_yards", "off_position_stat",
                        "draft_rookie", "returning_player", "first_observed_season",
                        "snap_segment", "has_weekly_stats_row"]].copy()


def origin_to_target_map(raw_test: pd.DataFrame) -> pd.DataFrame:
    """Map each forecast origin to its next observed game within the season."""
    required = set(KEY + ["fantasy_points", *FULL_COMPONENTS])
    if required - set(raw_test):
        raise ValueError(f"raw held-out fold missing {sorted(required - set(raw_test))}")
    check_keys(raw_test, "production forecast origins")
    ordered = raw_test.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    group = ordered.groupby(["player_id", "season"], sort=False)
    out = ordered[KEY].rename(columns={
        "season": "origin_season", "week": "origin_week", "team": "origin_team",
        "position": "origin_position"}).copy()
    for column in ("season", "week", "team", "position", "fantasy_points", *FULL_COMPONENTS):
        out[f"target_{column}"] = group[column].shift(-1)
    valid = out.target_week.notna()
    if (out.loc[valid, "target_season"] != out.loc[valid, "origin_season"]).any():
        raise ValueError("target crosses a season boundary")
    if (out.loc[valid, "target_week"] <= out.loc[valid, "origin_week"]).any():
        raise ValueError("target week must follow forecast origin")
    return out


def checked_production_fold(captured: pd.DataFrame, target_map: pd.DataFrame) -> pd.DataFrame:
    """Attach target-game keys and independently score a held-out OOF fold."""
    check_keys(captured, "production OOF origins")
    required = {"predicted_points", "actual_points", "train_seasons"}
    if required - set(captured):
        raise ValueError(f"production OOF capture missing {sorted(required - set(captured))}")
    finite_columns(captured, ["predicted_points", "actual_points"], "production OOF capture")
    if captured.train_seasons.isna().any():
        raise ValueError("production OOF training seasons missing")
    if target_map.duplicated(["player_id", "origin_season", "origin_week"]).any():
        raise ValueError("duplicate forecast origin in target mapping")
    origin_key = ["player_id", "season", "week", "team", "position"]
    renamed = target_map.rename(columns={"origin_season": "season", "origin_week": "week",
                                         "origin_team": "team", "origin_position": "position"})
    # Newer captures carry their own target-game columns; keep them apart so the
    # merge cannot suffix-collide, then require them to agree with the mapping.
    own = {c: f"captured_{c}" for c in captured.columns if c in renamed.columns and c.startswith("target_")}
    joined = captured.rename(columns=own).merge(
        renamed, on=origin_key, how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or joined.target_week.isna().any():
        raise ValueError("captured production origin has no next observed target game")
    for column, kept in own.items():
        mine = joined[kept]
        known = mine.notna()
        if not (mine[known].astype(str) == joined.loc[known, column].astype(str)).all() and not (
                pd.to_numeric(mine[known], errors="coerce") == pd.to_numeric(
                    joined.loc[known, column], errors="coerce")).all():
            raise ValueError(f"captured {column} disagrees with the target-game mapping")
    if not joined.target_season.eq(joined.season).all() or not joined.target_week.gt(joined.week).all():
        raise ValueError("production target does not follow its origin in the same season")
    for season, part in joined.groupby("season"):
        for training in part.train_seasons.unique():
            try:
                years = [int(value) for value in str(training).split(",") if value]
            except ValueError as exc:
                raise ValueError("invalid production training-season declaration") from exc
            if not years or max(years) >= int(season):
                raise ValueError("production fold trained on its target season")
    target_cols = [f"target_{name}" for name in FULL_COMPONENTS]
    target_values = finite_columns(joined, target_cols + ["target_fantasy_points"],
                                   "production target game")
    source_actual = target_values[:, -1]
    # Exact: _prepare_training_data leaves test targets raw (2026-10-07), so a
    # captured actual that differs is a mapping or lineage error, not clipping.
    if (np.abs(joined.actual_points.to_numpy(float) - source_actual) > 0.05).any():
        raise ValueError("captured shifted actual does not match mapped target game")
    raw_components = joined[target_cols].rename(
        columns={f"target_{name}": name for name in FULL_COMPONENTS})
    actual = calculate_fantasy_points_df(raw_components).to_numpy(float)
    if (np.abs(actual - source_actual) > 0.05).any():
        raise ValueError("target source PPR differs from ten raw components")
    result = pd.DataFrame({
        "player_id": joined.player_id, "season": joined.target_season.astype(int),
        "week": joined.target_week.astype(int), "team": joined.target_team,
        "position": joined.target_position,
        "origin_season": joined.season.astype(int), "origin_week": joined.week.astype(int),
        "predicted_ppr": joined.predicted_points.to_numpy(float), "actual_ppr": actual,
    })
    check_keys(result, "production target games")
    if not np.isfinite(result[["predicted_ppr", "actual_ppr"]].to_numpy(float)).all():
        raise ValueError("nonfinite production prediction or full-PPR label")
    return result

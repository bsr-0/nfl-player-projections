"""Checked Plan B join and feature contract.

Join share labels and lagged features to roster slots on exact
(player_id, season, week, team) keys. The slot caps omit some otherwise
eligible rows; return_coverage exposes those exclusions explicitly. A share
still refers to the full team total, even when the represented roster is a
subset. Accuracy comparisons must use identical represented rows.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH
from src.models.team_allocation.features import VOLUME_COLS, filter_population, load_share_rows
from src.models.team_allocation.features import feature_columns as allocation_feature_columns
from src.utils.leakage import audit_feature_availability, is_leakage_feature

SLOTS_TABLE_NAME = "team_week_roster_slots"
# NOTE: "slot" is deliberately NOT in ID_COLS -- it's the key fixed-effect
# feature for the mixed-effects model (src/models/team_hierarchical/models.py),
# not an identifier. team_week_id is reserved for roster-level audits;
# this baseline's random-effect grouping key is player_id.
ID_COLS = ["player_id", "season", "week", "team", "position", "team_week_id"]


def _connect(con: Optional[sqlite3.Connection]) -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH)) if con is None else con


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_slot_share_rows(
    target: str,
    seasons: Optional[list] = None,
    con: Optional[sqlite3.Connection] = None,
    slots_frame: Optional[pd.DataFrame] = None,
    return_coverage: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """One row per (player, season, week) that has BOTH a share label
    (Plan A's team_week_player_shares) and a roster slot (Plan B's
    team_week_roster_slots) -- inner join, see module docstring for the
    population consequence. `target` is one of team_allocation.features'
    VOLUME_COLS; population filtering (e.g. QB excluded for targets/
    receiving_yards) is applied the same way Plan A applies it, so the two
    plans' populations agree wherever both have a row at all.
    """
    owns_con = con is None
    conn = _connect(con)
    try:
        if slots_frame is None and not _table_exists(conn, SLOTS_TABLE_NAME):
            raise ValueError(
                f"{SLOTS_TABLE_NAME} table not found -- run "
                "scripts/build_team_week_roster_slots.py --dry-run --csv PATH "
                "and pass that snapshot as slots_frame (CLI: --slots-csv PATH)"
            )
        season_filter = ""
        params: list = []
        if seasons is not None:
            seasons_int = [int(s) for s in seasons]
            season_filter = f"WHERE season IN ({','.join('?' * len(seasons_int))})"
            params = seasons_int
        slots = slots_frame.copy() if slots_frame is not None else pd.read_sql(
            f"SELECT team, season, week, slot, slot_rank, player_id, depth_chart_rank, "
            f"snap_share_s2d FROM {SLOTS_TABLE_NAME} {season_filter}",
            conn, params=params,
        )
        shares = load_share_rows(seasons=seasons, con=conn)
    finally:
        if owns_con:
            conn.close()

    if seasons is not None:
        slots = slots[slots.season.isin([int(s) for s in seasons])].copy()
    merged, coverage = join_slot_share_rows(shares, slots, target)
    return (merged, coverage) if return_coverage else merged


def join_slot_share_rows(shares: pd.DataFrame, slots: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict]:
    """Join exact team/player-week keys and audit every unrepresented row."""
    if target not in VOLUME_COLS:
        raise ValueError(f"Plan B currently supports volume targets {VOLUME_COLS}, got {target!r}")
    key = ["player_id", "season", "week", "team"]
    slot_cols = key + ["slot", "slot_rank", "depth_chart_rank", "snap_share_s2d"]
    for name, frame, required in (("shares", shares, key + ["position"]), ("slots", slots, slot_cols)):
        if set(required) - set(frame.columns):
            raise ValueError(f"{name}: missing required columns {sorted(set(required) - set(frame.columns))}")
        if frame.columns.duplicated().any():
            raise ValueError(f"{name}: duplicate column names")
        if frame[key].isna().any().any() or frame.duplicated(["player_id", "season", "week"]).any():
            raise ValueError(f"{name}: null or duplicate player-week identity")
        for col in ("player_id", "team"):
            if not frame[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{name}: invalid {col}")
        for col in ("season", "week"):
            values = pd.to_numeric(frame[col], errors="coerce")
            if (~np.isfinite(values) | (values < 1) | (values != values.round())).any():
                raise ValueError(f"{name}: invalid {col}")
    if not shares.position.isin(["QB", "RB", "WR", "TE"]).all():
        raise ValueError("shares: invalid position")
    if slots.slot.isna().any() or slots.duplicated(["team", "season", "week", "slot"]).any():
        raise ValueError("slots: null or duplicate roster slot")
    ranks = pd.to_numeric(slots.slot_rank, errors="coerce")
    if (~np.isfinite(ranks) | (ranks < 1) | (ranks != ranks.round())).any():
        raise ValueError("slots: invalid positive integer slot_rank")
    identity = shares[key + ["position"]].merge(slots[slot_cols], on=key, how="outer", indicator=True, validate="one_to_one")
    if identity._merge.eq("right_only").any():
        raise ValueError("slot rows lack an exact share-panel team/player-week match; rebuild stale roster slots")
    matched = identity[identity._merge.eq("both")]
    expected_slot = matched.position + matched.slot_rank.astype(int).astype(str)
    if not matched.slot.eq(expected_slot).all():
        raise ValueError("slot label disagrees with the share-panel position or slot rank")
    # The share panel now also has snap_share_s2d. Keep the roster builder's
    # source named instead of accepting pandas' silent _x/_y suffixes.
    slot_features = slots[slot_cols].rename(columns={"snap_share_s2d": "roster_snap_share_s2d"})
    collisions = (set(slot_features) & set(shares)) - set(key)
    if collisions:
        raise ValueError(f"share panel already contains roster feature columns: {sorted(collisions)}")
    population = filter_population(shares, target)
    merged = population.merge(slot_features, on=key, how="inner", validate="one_to_one")
    excluded = population[key + ["position"]].merge(slots[key], on=key, how="left", indicator=True)
    excluded = excluded[excluded._merge.eq("left_only")].drop(columns="_merge")
    if merged.empty:
        raise ValueError("no player-weeks remain after the roster-slot join")
    merged["team_week_id"] = (
        merged["team"].astype(str) + "_" + merged["season"].astype(str) + "_" + merged["week"].astype(str)
    )
    coverage = {"eligible_rows": len(population), "represented_rows": len(merged), "excluded_rows": len(excluded),
                "excluded_keys": excluded.to_dict("records"),
                "by_season": {str(s): {"eligible": int((population.season == s).sum()),
                                        "represented": int((merged.season == s).sum()),
                                        "excluded": int((excluded.season == s).sum())}
                              for s in sorted(population.season.unique())}}
    return merged, coverage


def feature_columns(df: pd.DataFrame) -> list:
    """Audit the expanded share schema and expose only lagged/roster features.

    Reuse Plan A's raw-stat exclusions, then reject known leakage columns as
    well as unknown columns. player_id groups the random intercepts; neither
    it nor the team_week_id audit identifier is a fixed-effect feature.
    """
    # Share the complete raw-stat exclusion contract, including same-week
    # opportunity counts added to the panel after Plan B's first cut.
    try:
        feature_cols = allocation_feature_columns(df.drop(columns=["team_week_id"], errors="ignore"))
    except ValueError as exc:
        raise ValueError(f"Unclassified team-hierarchical feature columns: {exc}") from exc
    forbidden = [c for c in feature_cols if is_leakage_feature(c)]
    if forbidden:
        raise ValueError(f"Leakage columns in Plan B feature input: {sorted(forbidden)}")

    unclassified = audit_feature_availability(feature_cols)
    if unclassified:
        raise ValueError(
            "Unclassified team-hierarchical feature columns (add to "
            f"src/utils/leakage.py's FEATURE_AVAILABILITY): {sorted(unclassified)}"
        )
    return feature_cols

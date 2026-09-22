"""Feature contract for Plan B's mixed-effects share model.

Joins Plan A's `team_week_player_shares` (labels + lagged features) with
`team_week_roster_slots` (the fixed-slot structure Plan B needs -- see
scripts/build_team_week_roster_slots.py) on (player_id, season, week).

POPULATION NOTE: this join is INNER on slot, not a superset of Plan A's
population -- a player who exceeded their position's
`MAX_SLOTS_PER_POSITION` cap when roster slots were built has no slot row
at all and is silently absent here (not zero-filled, not imputed -- just
outside this model's representable population by construction). This
means Plan B's population is a strict SUBSET of Plan A's whenever the cap
binds. Any Plan A vs. Plan B accuracy comparison must restrict to the same
row set or note this explicitly -- comparing Plan B's MAE against Plan A's
MAE on their respective full populations would silently compare different
things if the cap drops meaningfully many rows (see the coverage report in
scripts/build_team_week_roster_slots.py for how often that happens).
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from config.settings import DB_PATH
from src.models.team_allocation.features import SHARE_COLS, VOLUME_COLS, filter_population, load_share_rows
from src.utils.leakage import audit_feature_availability

SLOTS_TABLE_NAME = "team_week_roster_slots"
# NOTE: "slot" is deliberately NOT in ID_COLS -- it's the key fixed-effect
# feature for the mixed-effects model (src/models/team_hierarchical/models.py),
# not an identifier. "team_week_id" IS excluded here: it's the MixedLM
# random-effect GROUPING key (see that module), not a fixed-effect feature.
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
) -> pd.DataFrame:
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
        if not _table_exists(conn, SLOTS_TABLE_NAME):
            raise ValueError(
                f"{SLOTS_TABLE_NAME} table not found -- run "
                "scripts/build_team_week_roster_slots.py --write first"
            )
        season_filter = ""
        params: list = []
        if seasons is not None:
            seasons_int = [int(s) for s in seasons]
            season_filter = f"WHERE season IN ({','.join('?' * len(seasons_int))})"
            params = seasons_int
        slots = pd.read_sql(
            f"SELECT team, season, week, slot, slot_rank, player_id, depth_chart_rank, "
            f"snap_share_s2d FROM {SLOTS_TABLE_NAME} {season_filter}",
            conn, params=params,
        )
        shares = load_share_rows(seasons=seasons, con=conn)
    finally:
        if owns_con:
            conn.close()

    shares = filter_population(shares, target)
    merged = shares.merge(
        slots.drop(columns=["team"]), on=["player_id", "season", "week"], how="inner",
    )
    merged["team_week_id"] = (
        merged["team"].astype(str) + "_" + merged["season"].astype(str) + "_" + merged["week"].astype(str)
    )
    return merged


def feature_columns(df: pd.DataFrame) -> list:
    """Plan A's own feature set (lagged shares, is_cold_start, lagged team
    totals) plus the roster-slot columns this join adds: `slot`
    (categorical -- the fixed effect a MixedLM formula keys off),
    `slot_rank`, `depth_chart_rank`, `snap_share_s2d` (all pregame-known,
    see scripts/build_team_week_roster_slots.py). `team_week_id` is the
    MixedLM GROUPING key, not a fixed-effect feature -- excluded here the
    same way `team`/`player_id` are id columns, not features.

    Opt-out, not opt-in: every column is a candidate feature EXCEPT the id
    columns and Plan A's own known same-week raw/label columns (excluded by
    exact name, matching how `team_allocation.features.feature_columns()`
    excludes its own raw columns) -- then `audit_feature_availability` is
    the real last line of defense, same pattern used throughout this repo.
    An opt-in allowlist here would make that audit call dead code (nothing
    new could ever reach it to be caught), which is exactly the kind of
    safety-shaped-but-inert code CLAUDE.md warns against.
    """
    always_excluded = set(ID_COLS) | set(VOLUME_COLS) | set(SHARE_COLS) | {f"team_{c}" for c in VOLUME_COLS}
    feature_cols = [c for c in df.columns if c not in always_excluded]

    unclassified = audit_feature_availability(feature_cols)
    if unclassified:
        raise ValueError(
            "Unclassified team-hierarchical feature columns (add to "
            f"src/utils/leakage.py's FEATURE_AVAILABILITY): {sorted(unclassified)}"
        )
    return feature_cols

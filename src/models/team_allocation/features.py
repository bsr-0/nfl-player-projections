"""Feature contract for Plan A's team-allocation share models.

Reads scripts/build_team_week_player_shares.py's output table
(`team_week_player_shares`). `VOLUME_COLS`/`ROLL_WINDOW` here are the single
source of truth for that builder too (it imports them from this module)
rather than each keeping its own copy -- the exact drift risk CLAUDE.md
calls out ("parallel implementations that must remain consistent").

`feature_columns()` is the one place that enforces the label/feature
boundary for every consumer of this table (the backtester,
scripts/train_team_share_model.py): current-week raw volume counts,
current-week team totals, and current-week share_of_team_* values are all
LABELS (or label-adjacent, same-week-derived quantities) and must never be
fed into a model that predicts a same-week share -- only the *_s2d/*_roll3
lagged share history and is_cold_start survive as features. This mirrors
src/models/game_outcome/features.py's `feature_columns()` excluding
home_win/home_margin/game_total by name rather than relying solely on the
FEATURE_AVAILABILITY audit to keep labels out.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

import pandas as pd

from config.settings import DB_PATH
from src.utils.leakage import audit_feature_availability

TABLE_NAME = "team_week_player_shares"
ROLL_WINDOW = 3

# Single source of truth -- scripts/build_team_week_player_shares.py imports
# these rather than keeping its own copy.
VOLUME_COLS = ["targets", "rushing_attempts", "receiving_yards", "rushing_yards"]
FULL_PPR_VOLUME_COLS = [
    "receptions", "receiving_tds", "rushing_tds",
    "passing_yards", "passing_tds", "interceptions",
]
# Opportunity/role measures are deliberately separate from the predicted
# volume labels.  The builder stores their same-week audit values, but only
# their shifted history is admitted as a feature.  These are especially
# important for sparse touchdown allocation, where ordinary volume shares are
# zero-inflated and do not identify goal-line/red-zone roles.
OPPORTUNITY_COLS = [
    "rush_inside_10", "rush_inside_5", "targets_15_plus", "air_yards",
    "pass_plays", "rush_plays", "recv_targets", "neutral_targets",
    "neutral_rushes", "third_down_targets", "short_yardage_rushes",
    "redzone_targets", "goal_line_touches", "two_minute_targets",
    "high_leverage_touches", "snap_count", "snap_share",
]
ALL_VOLUME_COLS = VOLUME_COLS + FULL_PPR_VOLUME_COLS
SHARE_COLS = [f"share_of_team_{c}" for c in VOLUME_COLS]
FULL_PPR_SHARE_COLS = [f"share_of_team_{c}" for c in FULL_PPR_VOLUME_COLS]
ALL_SHARE_COLS = SHARE_COLS + FULL_PPR_SHARE_COLS

ID_COLS = ["player_id", "season", "week", "team", "position"]

# Population each share target is legitimately defined over -- a QB's
# share_of_team_targets is always exactly 0 (QBs don't get targeted), so
# including QB rows would just be a constant the model has to learn is
# meaningless rather than a real population member. See
# docs/TEAM_LEVEL_ALLOCATION_MODELS.md's Plan A population table.
TARGET_POPULATIONS: dict[str, Optional[str]] = {
    "targets": "exclude_qb",
    "receiving_yards": "exclude_qb",
    "rushing_attempts": None,
    "rushing_yards": None,
    "receptions": "exclude_qb",
    "receiving_tds": "exclude_qb",
    "rushing_tds": None,
    "passing_yards": "qb_only",
    "passing_tds": "qb_only",
    "interceptions": "qb_only",
}


def filter_population(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rule = TARGET_POPULATIONS.get(target)
    if rule == "exclude_qb":
        return df[df["position"] != "QB"].copy()
    if rule == "qb_only":
        return df[df["position"] == "QB"].copy()
    return df


def _connect(con: Optional[sqlite3.Connection]) -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH)) if con is None else con


def load_share_rows(
    seasons: Optional[list] = None,
    con: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """One row per (player, season, week) from team_week_player_shares.

    Raises if the table doesn't exist -- run
    scripts/build_team_week_player_shares.py --write first, same contract
    as build_canonical_player_weeks.py's dependents use.
    """
    owns_con = con is None
    conn = _connect(con)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,)
        ).fetchone()
        if not exists:
            raise ValueError(
                f"{TABLE_NAME} table not found -- run "
                "scripts/build_team_week_player_shares.py --write first"
            )
        season_filter = ""
        params: list = []
        if seasons is not None:
            seasons = [int(s) for s in seasons]
            season_filter = f"WHERE season IN ({','.join('?' * len(seasons))})"
            params = seasons
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} {season_filter}", conn, params=params)
    finally:
        if owns_con:
            conn.close()
    return df


def feature_columns(df: pd.DataFrame, *, include_full_ppr: bool = False) -> list:
    excluded = (
        set(ID_COLS) | set(ALL_VOLUME_COLS) | set(ALL_SHARE_COLS)
        | set(OPPORTUNITY_COLS)
        | {f"team_{c}" for c in ALL_VOLUME_COLS}
        | {f"team_{c}" for c in OPPORTUNITY_COLS}
        | {f"share_of_team_{c}" for c in OPPORTUNITY_COLS}
    )
    feature_cols = [c for c in df.columns if c not in excluded]
    if not include_full_ppr:
        full_history = set(FULL_PPR_SHARE_COLS)
        full_history |= {f"{c}_s2d" for c in FULL_PPR_SHARE_COLS}
        full_history |= {f"{c}_roll{ROLL_WINDOW}" for c in FULL_PPR_SHARE_COLS}
        full_history |= {f"team_{c}_s2d" for c in FULL_PPR_VOLUME_COLS}
        full_history |= {f"team_{c}_roll{ROLL_WINDOW}" for c in FULL_PPR_VOLUME_COLS}
        feature_cols = [c for c in feature_cols if c not in full_history]

    unclassified = audit_feature_availability(feature_cols)
    if unclassified:
        raise ValueError(
            "Unclassified team-allocation feature columns (add to "
            f"src/utils/leakage.py's FEATURE_AVAILABILITY): {sorted(unclassified)}"
        )
    return feature_cols

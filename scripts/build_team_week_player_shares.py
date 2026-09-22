#!/usr/bin/env python3
"""Build the team_week_player_shares table -- Plan A's shared prerequisite.

See docs/TEAM_LEVEL_ALLOCATION_MODELS.md for the full plan. This script
produces, for each QB/RB/WR/TE player-week in the audited
`canonical_player_weeks` panel, that player's SAME-WEEK share of team volume
(targets, rushing attempts, receiving yards, rushing yards) plus LAGGED
share-history features (season-to-date and rolling-3 mean of the player's own
share in PRIOR weeks).

Same-week share columns are LABELS, not features -- exactly like
`home_win` in src/models/game_outcome/labels.py, they must never be fed into
a model that predicts them. Only the `_s2d`/`_roll3` lagged columns are
features. See src/models/game_outcome/features.py's `_lagged_form` for the
prior art this mirrors (shift(1) before .expanding()/.rolling(), prior-season
fallback for cold starts) -- kept as an independent copy here (player-keyed,
not team-keyed) rather than a shared import, same rationale used elsewhere in
this repo for the game-outcome module's own copies.

Team-total denominator note: team_X is the SUM of player_X across every
player_id in this table's own (team, season, week) group, using 0 for any
player with no player_weekly_stats row that week. This is NOT the same kind
of decision as leaving `fantasy_points` NULL in build_canonical_player_weeks.py
(where an unknown outcome must not be silently zero-filled) -- here, team_X
and player_X are computed from literally the same source rows, so "no stats
row this week" and "recorded 0 targets this week" are indistinguishable and
both correctly contribute 0 to the team total. There is no unknown/zero
ambiguity to preserve, unlike fantasy_points.

Usage:
    python scripts/build_team_week_player_shares.py --dry-run
    python scripts/build_team_week_player_shares.py --write
    python scripts/build_team_week_player_shares.py --write --seasons 2013 2026
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, POSITIONS
from src.models.team_allocation.features import ROLL_WINDOW, TABLE_NAME, VOLUME_COLS

MIN_PRIOR_GAMES_FOR_FORM = 3

# VOLUME_COLS/ROLL_WINDOW/TABLE_NAME are imported from
# src/models/team_allocation/features.py (single source of truth -- that
# module's feature_columns() must exclude exactly these same names, so it
# owns the definitions rather than each module keeping its own copy). Each
# share is player_X / team_X where team_X is this table's own group-sum of
# the same column -- see module docstring for why 0-fill on a missing stats
# row is correct here specifically (unlike fantasy_points elsewhere in this
# repo).


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_population(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    """QB/RB/WR/TE player-weeks from the audited canonical panel -- this is
    the population every share is computed over, not a raw stats pull."""
    if not _table_exists(conn, "canonical_player_weeks"):
        raise ValueError(
            "canonical_player_weeks table not found -- run "
            "scripts/build_canonical_player_weeks.py --write first"
        )
    df = pd.read_sql(
        """
        SELECT player_id, season, week, team, position
        FROM canonical_player_weeks
        WHERE season BETWEEN ? AND ? AND position IN ({})
        """.format(",".join("?" * len(POSITIONS))),
        conn,
        params=[lo, hi, *POSITIONS],
    )
    return df


def load_volumes(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    cols = pd.read_sql("PRAGMA table_info(player_weekly_stats)", conn)["name"].tolist()
    keep = ["player_id", "season", "week"] + [c for c in VOLUME_COLS if c in cols]
    df = pd.read_sql(
        f"SELECT {','.join(keep)} FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
        conn, params=(lo, hi),
    )
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", *VOLUME_COLS])
    df = df.drop_duplicates(["player_id", "season", "week"], keep="last")
    for c in VOLUME_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[["player_id", "season", "week", *VOLUME_COLS]]


def _safe_share(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return (numer / denom.replace(0, pd.NA)).fillna(0.0)


def build_shares(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    pop = load_population(conn, lo, hi)
    if pop.empty:
        return pop
    vol = load_volumes(conn, lo, hi)

    panel = pop.merge(vol, on=["player_id", "season", "week"], how="left")
    for c in VOLUME_COLS:
        panel[c] = panel[c].fillna(0.0)

    team_totals = (
        panel.groupby(["team", "season", "week"])[VOLUME_COLS]
        .transform("sum")
        .rename(columns={c: f"team_{c}" for c in VOLUME_COLS})
    )
    panel = pd.concat([panel, team_totals], axis=1)

    share_cols = []
    for c in VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        panel[share_col] = _safe_share(panel[c], panel[f"team_{c}"])
        share_cols.append(share_col)

    panel = panel.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grp = panel.groupby(["player_id", "season"], group_keys=False)
    panel["n_prior_games"] = grp.cumcount()

    for col in share_cols:
        panel[f"{col}_s2d"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
        panel[f"{col}_roll{ROLL_WINDOW}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )

    prior_season = panel.groupby(["player_id", "season"])[share_cols].mean().reset_index()
    prior_season["season"] = prior_season["season"] + 1
    prior_season = prior_season.rename(columns={c: f"{c}_prior_season" for c in share_cols})
    panel = panel.merge(prior_season, on=["player_id", "season"], how="left")

    panel["is_cold_start"] = (panel["n_prior_games"] < MIN_PRIOR_GAMES_FOR_FORM).astype(int)
    cold = panel["is_cold_start"].astype(bool)
    for col in share_cols:
        prior_col = f"{col}_prior_season"
        panel.loc[cold, f"{col}_s2d"] = panel.loc[cold, f"{col}_s2d"].where(
            panel.loc[cold, prior_col].isna(), panel.loc[cold, prior_col]
        )
        panel.loc[cold, f"{col}_roll{ROLL_WINDOW}"] = panel.loc[cold, f"{col}_roll{ROLL_WINDOW}"].where(
            panel.loc[cold, prior_col].isna(), panel.loc[cold, prior_col]
        )

    drop_cols = [f"{c}_prior_season" for c in share_cols] + ["n_prior_games"]
    panel = panel.drop(columns=drop_cols)
    return panel.reset_index(drop=True)


def validate_shares(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("team_week_player_shares panel is empty")
    if panel.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("duplicate player_id/season/week rows")
    for c in VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        if (panel[share_col] < -1e-9).any() or (panel[share_col] > 1 + 1e-9).any():
            raise ValueError(f"{share_col} outside [0, 1]")
    # Same-week shares within a team must sum to <= 1 (+ float tolerance) --
    # a share > the whole team's total is a join or double-count bug.
    for c in VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        totals = panel.groupby(["team", "season", "week"])[share_col].sum()
        bad = totals[totals > 1 + 1e-6]
        if not bad.empty:
            raise ValueError(f"{share_col} sums to > 1 for {len(bad)} team-week(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO", "HI"), default=[2013, 2026])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    lo, hi = sorted(args.seasons)

    conn = sqlite3.connect(str(args.db))
    panel = build_shares(conn, lo, hi)
    validate_shares(panel)

    print(f"team_week_player_shares: {len(panel):,} player-weeks, seasons {lo}-{hi}")
    print(f"cold-start rows: {int(panel['is_cold_start'].sum()):,}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.csv, index=False)
        print(f"wrote CSV: {args.csv}")

    if args.write and not args.dry_run:
        panel.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE_NAME}_key "
            f"ON {TABLE_NAME}(player_id, season, week)"
        )
        conn.commit()
        print(f"wrote SQLite table: {TABLE_NAME}")
    else:
        print("dry-run: database unchanged")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

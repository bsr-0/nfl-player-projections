#!/usr/bin/env python3
"""Align player_weekly_stats' snap columns with the authoritative snap_counts.

`snap_counts` is the source of truth for participation; the snap columns on
`player_weekly_stats` are a derived convenience copy. They had drifted badly
(GAPS.md 2026-08-19 audit): 38,178 rows asserted zero snaps for players the
source shows on the field, and 2006-2017 read 100% zero because this
propagation had never run below 2018.

Two defects in the previous version, both fixed here:

1. It matched on normalised NAME. 6,435 pre-2018 rows have a blank
   `players.name`, and `_normalize_name` read the last token as the surname
   ("Odell Beckham Jr." -> "O.Jr."), so the join failed silently and the row
   kept its placeholder 0. Now matched on player_id via the PFR->GSIS map,
   which lifts the pre-2018 match rate from 74.5% to 92.8%.
2. It built its lookup with `WHERE offense_snaps > 0`, excluding zero-snap
   rows, so a failed match and a real zero were indistinguishable. Zero rows
   are now loaded, which is what makes the three-way rule below possible.

THE THREE-WAY RULE -- the point of this script:

    source has a value        -> overwrite (it is authoritative)
    source silent, pws is 0   -> write NULL (0 was never a measurement)
    source silent, pws is > 0 -> LEAVE ALONE

That last case matters: 1,831 rows (all 2018+) hold a positive snap count
with no id-match, because the PFR->GSIS map covers ~81% of ids. Blindly
NULLing "absent" would destroy real data this script cannot regenerate.

Usage:
    python scripts/backfill_snap_counts_to_pws.py --dry-run
    python scripts/backfill_snap_counts_to_pws.py
    python scripts/backfill_snap_counts_to_pws.py --seasons 2013 2025
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}


def _team_alias(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


def load_authoritative(conn, lo: int, hi: int):
    """(per-player snaps, per-team play counts) from snap_counts.

    Regular season only: player_weekly_stats is a REG-season table, and a
    postseason row would otherwise collide on (player_id, season, week).
    """
    from src.data.nfl_data_loader import get_pfr_to_gsis_map
    from src.data.pbp_stats_aggregator import compute_team_snaps

    snaps = pd.read_sql(
        "SELECT season, week, team, pfr_player_id, offense_snaps, offense_pct "
        "FROM snap_counts WHERE game_type = 'REG' AND season BETWEEN ? AND ?",
        conn, params=(lo, hi),
    )
    snaps["team"] = _team_alias(snaps["team"])

    # Team play counts come from the FULL roster (linemen included), so they
    # are computed before narrowing to skill positions -- via the same
    # offense_pct-implied median the PBP ingest path uses, so both writers
    # agree on what team_snaps means.
    team_snaps = compute_team_snaps(snaps).rename(
        columns={"team_snaps": "auth_team_snaps"})

    skill = snaps[snaps["pfr_player_id"].notna()].copy()
    skill["player_id"] = skill["pfr_player_id"].map(get_pfr_to_gsis_map())
    skill = skill.dropna(subset=["player_id"])
    per_player = (
        skill.groupby(["player_id", "season", "week"], as_index=False)["offense_snaps"]
        .max()
        .rename(columns={"offense_snaps": "auth_snaps"})
    )
    return per_player, team_snaps


def resolve(current: pd.Series, authoritative: pd.Series) -> pd.Series:
    """The three-way rule. Returns a nullable Int64 column."""
    current = pd.to_numeric(current, errors="coerce").astype("Int64")
    authoritative = pd.to_numeric(authoritative, errors="coerce").astype("Int64")

    resolved = authoritative.copy()
    keep_existing = authoritative.isna() & current.notna() & (current > 0)
    resolved[keep_existing] = current[keep_existing]
    # Source silent and current is 0 (or already NULL) -> stays NA, i.e. the
    # honest "we don't know" rather than a fabricated zero.
    return resolved


def main() -> int:
    from config.settings import CURRENT_NFL_SEASON, DB_PATH

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO", "HI"),
                    default=[2013, CURRENT_NFL_SEASON])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    lo, hi = sorted(args.seasons)

    conn = sqlite3.connect(str(DB_PATH))

    per_player, team_snaps = load_authoritative(conn, lo, hi)
    print(f"authoritative: {len(per_player):,} player-weeks, "
          f"{len(team_snaps):,} team-weeks")

    pws = pd.read_sql(
        "SELECT rowid AS _rowid, player_id, season, week, team, snap_count, "
        "team_snaps FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
        conn, params=(lo, hi),
    )
    pws["team"] = _team_alias(pws["team"])
    print(f"player_weekly_stats rows in range: {len(pws):,}")

    merged = (
        pws.merge(per_player, on=["player_id", "season", "week"], how="left")
           .merge(team_snaps, on=["season", "week", "team"], how="left")
    )
    new_snaps = resolve(merged["snap_count"], merged["auth_snaps"])
    new_team = resolve(merged["team_snaps"], merged["auth_team_snaps"])
    # A share is only meaningful when both sides are known.
    new_share = (new_snaps / new_team).where(new_team > 0)

    old = pd.to_numeric(merged["snap_count"], errors="coerce").astype("Int64")
    changed = new_snaps.fillna(-1) != old.fillna(-1)
    print(f"\nsnap_count changes: {int(changed.sum()):,}")
    print(f"  0 -> real value : {int((changed & (old == 0) & new_snaps.notna()).sum()):,}")
    print(f"  0 -> NULL       : {int((changed & (old == 0) & new_snaps.isna()).sum()):,}")
    print(f"  corrected >0    : {int((changed & (old > 0) & new_snaps.notna()).sum()):,}")
    print(f"  preserved >0 (no source match): "
          f"{int(((old > 0) & merged['auth_snaps'].isna()).sum()):,}")
    print(f"team_snaps unknown after: {int(new_team.isna().sum()):,}")

    if args.dry_run:
        print("\n--dry-run: no writes.")
        return 0

    payload = [
        (None if pd.isna(s) else int(s),
         None if pd.isna(t) else int(t),
         None if pd.isna(sh) else float(sh),
         int(r))
        for s, t, sh, r in zip(new_snaps, new_team, new_share, merged["_rowid"])
    ]
    conn.executemany(
        "UPDATE player_weekly_stats SET snap_count = ?, team_snaps = ?, "
        "snap_share = ? WHERE rowid = ?", payload)
    conn.commit()
    print(f"\nwrote {len(payload):,} rows")

    print(pd.read_sql(
        """SELECT season, COUNT(*) rows,
                  SUM(snap_count IS NULL) unknown,
                  SUM(snap_count = 0) confirmed_zero,
                  SUM(snap_count > 0) positive
           FROM player_weekly_stats WHERE season BETWEEN ? AND ?
           GROUP BY season ORDER BY season""",
        conn, params=(lo, hi)).to_string(index=False))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

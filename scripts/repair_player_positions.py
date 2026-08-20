#!/usr/bin/env python3
"""Repair and audit `players.position` against authoritative roster data.

Root cause (GAPS.md 2026-08-20): `PBPStatsAggregator.aggregate_passing_stats`
stamps `position='QB'` on every row with a pass attempt, including a single
trick-play pass thrown by a real RB/WR. That row reached `players` through
`insert_player`'s INSERT OR REPLACE, and `_infer_position` then read the
value back out of `players` on the next ingest -- so the corruption
sustained itself. Christian McCaffrey, Derrick Henry, Cooper Kupp, DJ
Moore, Devin Singletary and Courtland Sutton were all labeled QB, and
11.4% of the QB training population were not quarterbacks.

The code paths are fixed; this repairs the data they already wrote, and
doubles as the recurring audit (`--audit-only`).

Authority order, most trusted first:
  1. weekly roster snapshots (`get_authoritative_player_positions`) --
     reported per week by nflverse, not derived from stat lines
  2. `nfl_data_py.import_players()` -- the league-wide player index
Both are reported rather than inferred. Where they disagree the roster wins,
because it is season/week-specific and the player index is not.

Usage:
    python scripts/repair_player_positions.py                # dry-run
    python scripts/repair_player_positions.py --write        # apply
    python scripts/repair_player_positions.py --audit-only   # report, never write
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, POSITIONS

AUDIT_PATH = PROJECT_ROOT / "data" / "experiments" / "position_repair_audit.csv"


# The ingest pipeline already folds fullbacks and halfbacks into RB
# (`pbp_stats_aggregator`), so the authority maps must too -- otherwise a
# correctly-labeled RB whose roster row says FB looks like a defect.
POSITION_ALIASES = {"FB": "RB", "HB": "RB"}


def _normalize(pos):
    return POSITION_ALIASES.get(pos, pos)


def authoritative_positions(use_feed: bool = True) -> tuple[dict, dict]:
    from src.utils.database import DatabaseManager
    roster = {k: _normalize(v)
              for k, v in DatabaseManager().get_authoritative_player_positions().items()}

    feed: dict = {}
    if use_feed:
        try:
            import nfl_data_py as nfl
            p = nfl.import_players()[["gsis_id", "position"]].dropna()
            feed = {g: _normalize(pos) for g, pos in zip(p.gsis_id, p.position) if g}
        except Exception as e:  # offline is not fatal; rosters still work
            print(f"  (player index unavailable, rosters only: {e})")
    return roster, feed


def build_report(conn: sqlite3.Connection, roster: dict, feed: dict) -> pd.DataFrame:
    players = pd.read_sql("SELECT player_id, name, position FROM players", conn)
    counts = pd.read_sql(
        "SELECT player_id, COUNT(*) AS n_rows FROM player_weekly_stats GROUP BY player_id", conn
    )
    df = players.merge(counts, on="player_id", how="left").fillna({"n_rows": 0})
    df["roster_position"] = df.player_id.map(roster)
    df["feed_position"] = df.player_id.map(feed)
    # Roster wins: it is week-specific, the player index is a single snapshot.
    df["correct_position"] = df.roster_position.fillna(df.feed_position)
    df["needs_repair"] = (
        df.correct_position.notna()
        & (df.correct_position != df.position)
        & df.correct_position.isin(POSITIONS)
    )
    df["sources_disagree"] = (
        df.roster_position.notna() & df.feed_position.notna()
        & (df.roster_position != df.feed_position)
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--no-feed", action="store_true", help="rosters only, skip the network call")
    args = ap.parse_args()

    roster, feed = authoritative_positions(use_feed=not args.no_feed)
    print(f"Authority: {len(roster)} roster positions, {len(feed)} player-index positions")

    conn = sqlite3.connect(str(DB_PATH))
    df = build_report(conn, roster, feed)
    bad = df[df.needs_repair]

    print(f"\nplayers table: {len(df)} rows")
    print(f"  needing repair: {len(bad)} players, {int(bad.n_rows.sum())} player-weeks")
    if len(bad):
        print("\n  by (stored -> correct):")
        print(bad.groupby(["position", "correct_position"]).size()
              .sort_values(ascending=False).head(15).to_string())
        print("\n  worst by affected player-weeks:")
        print(bad.nlargest(10, "n_rows")[
            ["player_id", "name", "position", "correct_position", "n_rows"]
        ].to_string(index=False))

    dis = df[df.sources_disagree]
    if len(dis):
        print(f"\n  sources disagree on {len(dis)} players (roster wins):")
        print(dis[["name", "roster_position", "feed_position"]].head(10).to_string(index=False))

    unknown = df[df.correct_position.isna() & (df.n_rows > 0)]
    print(f"\n  no authoritative source, but have stats rows: {len(unknown)} players "
          f"({int(unknown.n_rows.sum())} player-weeks) -- left as-is")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(AUDIT_PATH, index=False)
    print(f"\nFull audit written to {AUDIT_PATH}")

    if args.audit_only or not args.write:
        print("\nDRY RUN -- nothing written." if not args.audit_only else "\nAudit only.")
        conn.close()
        sys.exit(1 if (len(bad) and args.audit_only) else 0)

    if not len(bad):
        print("\nNothing to repair.")
        conn.close()
        return

    backup = DB_PATH.parent / f"nfl_data.db.bak-position-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup)
    print(f"\nBacked up database to {backup}")
    conn.executemany(
        "UPDATE players SET position = ?, updated_at = CURRENT_TIMESTAMP WHERE player_id = ?",
        [(r.correct_position, r.player_id) for r in bad.itertuples(index=False)],
    )
    conn.commit()
    print(f"Repaired {len(bad)} players ({int(bad.n_rows.sum())} player-weeks).")
    conn.close()


if __name__ == "__main__":
    main()

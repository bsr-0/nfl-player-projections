#!/usr/bin/env python3
"""Build the team_week_roster_slots table -- Plan B's data-shape prerequisite.

See docs/TEAM_LEVEL_ALLOCATION_MODELS.md's Plan B section. This script
assigns each QB/RB/WR/TE player-week in the audited `canonical_player_weeks`
panel to a fixed SLOT (e.g. "RB1", "WR3") within their team-week, so a
mixed-effects or set-model architecture has a stable, fixed-width structure
to key off of instead of a variable-length roster.

IMPORTANT: this reuses the codebase's EXISTING, already-audited pregame-safe
depth-chart lookup (`_load_depth_chart_asof_table` in
src/features/feature_engineering.py -- the same table
`_add_depth_chart_rank`/`_lookup_depth_chart_rank_asof` already use for the
production per-position models) rather than building a new depth-chart join
from scratch. That table's `depth_chart_rank` is an "as of this row's own
week" snapshot (week <= target week is legitimate pre-game info, same
category as Vegas lines -- see that module's docstring), with a 1-season
staleness bound and a 2013 coverage start already enforced there.

WHAT'S GENUINELY NEW HERE (not already solved by the reused lookup): that
rank is not unique. Multiple players commonly share the same
`depth_chart_rank` (ties in the raw listing) or all fall back to the same
"3" default when stale/missing -- neither is a problem for using rank as a
FEATURE (which the production models already do), but it IS a problem for
assigning a UNIQUE slot, which a fixed-width architecture requires. This
script's actual contribution is a deterministic, documented, tested
tie-break: (depth_chart_rank asc, lagged season-to-date snap_share desc,
player_id asc) -- the first two are real, pregame-known signals of role;
player_id is the final tie-break purely to guarantee determinism when both
real signals still tie.

Usage:
    python scripts/build_team_week_roster_slots.py --dry-run
    python scripts/build_team_week_roster_slots.py --write
    python scripts/build_team_week_roster_slots.py --write --seasons 2013 2026
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

TABLE_NAME = "team_week_roster_slots"

# Starting assumption, not a measured constant -- how many active roster
# slots per position to represent in the fixed-width shape. A team-week
# with more rostered players at a position than its cap simply cannot be
# represented (those extra players are dropped from this table entirely,
# flagged in the coverage report rather than silently lost). Revisit once
# the coverage report below is run against real data.
MAX_SLOTS_PER_POSITION = {"QB": 2, "RB": 4, "WR": 6, "TE": 3}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_population(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    if not _table_exists(conn, "canonical_player_weeks"):
        raise ValueError(
            "canonical_player_weeks table not found -- run "
            "scripts/build_canonical_player_weeks.py --write first"
        )
    return pd.read_sql(
        """
        SELECT player_id, season, week, team, position
        FROM canonical_player_weeks
        WHERE season BETWEEN ? AND ? AND position IN ({})
        """.format(",".join("?" * len(POSITIONS))),
        conn, params=[lo, hi, *POSITIONS],
    )


def load_depth_chart_rank_asof(db_path: Path, population: pd.DataFrame) -> pd.Series:
    """Delegates to the existing, already-audited as-of lookup rather than
    re-querying `depth_charts` directly -- see module docstring. Uses the
    REAL-ROW convention (week <= target week, `_add_depth_chart_rank`'s
    cutoff), matching this script's population (completed games only, same
    as Plan A's training builders) -- not the stricter synthetic/future-row
    convention (`_lookup_depth_chart_rank_asof`), which is a separate,
    intentionally different cutoff for not-yet-played weeks, and out of
    scope for this first data-shape cut.

    GOTCHA found while wiring this up (2026-09, worth knowing if this
    reused function is ever touched again): `_load_depth_chart_asof_table`
    does NOT take a connection -- it always opens its own connection to
    `config.settings.DB_PATH` and CACHES the result at module/process
    scope (`_depth_chart_asof_cache`). This script's own `--db` argument
    would therefore silently be ignored for the depth-chart lookup only
    (while still being respected for canonical_player_weeks/
    player_weekly_stats), UNLESS `config.settings.DB_PATH` is pointed at
    the same file first -- which is what this function does, plus clearing
    the cache so a stale table from a previously-used DB_PATH in the same
    process is never silently reused. Production single-DB runs never hit
    this; multi-DB test runs would, silently, without this fix.
    """
    import config.settings as settings
    from src.features import feature_engineering
    from src.features.feature_engineering import FeatureEngineer

    settings.DB_PATH = db_path
    feature_engineering._depth_chart_asof_cache.clear()

    engineer = FeatureEngineer.__new__(FeatureEngineer)  # avoid full __init__ (DB connections, etc.)
    df = engineer._add_depth_chart_rank(population.copy())
    return df["depth_chart_rank"]


def load_snap_share_tiebreak(conn: sqlite3.Connection, lo: int, hi: int, population: pd.DataFrame) -> pd.DataFrame:
    """Lagged (shift(1) season-to-date mean) snap_share per (player, season,
    week) -- a real, pregame-known signal of established role, used only as
    the tie-break's second key (see module docstring). Same lag discipline
    as every other rolling feature in this repo: the value attached to a
    given week uses only that player's games strictly before it.

    `population` (player_id/season/week, from `canonical_player_weeks`) is
    used as the base sequence, NOT whatever rows happen to exist in
    `player_weekly_stats` -- a player-week can legitimately be in the
    canonical population with no stats row at all (Phase 1's "unknown"
    participation state), and computing the lag only over weeks THAT HAVE a
    stats row would silently skip that week's slot in the shift(1)/
    expanding() sequence, distorting the lag attached to the FOLLOWING
    week too. Same failure mode `_append_placeholder_rows` in
    src/models/game_outcome/features.py already exists to prevent -- found
    here by a failing test, not assumed away.
    """
    stats = pd.read_sql(
        "SELECT player_id, season, week, snap_share FROM player_weekly_stats "
        "WHERE season BETWEEN ? AND ?",
        conn, params=(lo, hi),
    )
    stats = stats.drop_duplicates(["player_id", "season", "week"], keep="last")

    base = population[["player_id", "season", "week"]].drop_duplicates()
    df = base.merge(stats, on=["player_id", "season", "week"], how="left")
    df = df.sort_values(["player_id", "season", "week"])
    grp = df.groupby(["player_id", "season"], group_keys=False)
    df["snap_share_s2d"] = grp["snap_share"].transform(lambda s: s.shift(1).expanding().mean())
    return df[["player_id", "season", "week", "snap_share_s2d"]]


def build_roster_slots(conn: sqlite3.Connection, lo: int, hi: int, db_path: Path) -> pd.DataFrame:
    pop = load_population(conn, lo, hi)
    if pop.empty:
        return pop

    pop["depth_chart_rank"] = load_depth_chart_rank_asof(db_path, pop).values
    tiebreak = load_snap_share_tiebreak(conn, lo, hi, pop)
    pop = pop.merge(tiebreak, on=["player_id", "season", "week"], how="left")

    # Sort key: depth_chart_rank ascending (1=starter first), then lagged
    # snap_share descending (a player with no history -- NaN -- sorts LAST,
    # i.e. treated as the least-established, never invented as "average" or
    # "best"), then player_id ascending as the final deterministic
    # tie-break when both real signals still tie.
    pop["_snap_share_sort"] = pop["snap_share_s2d"].fillna(-1.0)
    pop = pop.sort_values(
        ["team", "season", "week", "position", "depth_chart_rank", "_snap_share_sort", "player_id"],
        ascending=[True, True, True, True, True, False, True],
    )

    grp = pop.groupby(["team", "season", "week", "position"], group_keys=False)
    pop["slot_rank"] = grp.cumcount() + 1

    pop["max_slot"] = pop["position"].map(MAX_SLOTS_PER_POSITION)
    dropped = pop[pop["slot_rank"] > pop["max_slot"]].copy()
    kept = pop[pop["slot_rank"] <= pop["max_slot"]].copy()
    kept["slot"] = kept["position"] + kept["slot_rank"].astype(str)

    if len(dropped):
        print(
            f"  {len(dropped):,} player-week(s) exceeded their position's "
            f"MAX_SLOTS_PER_POSITION cap and were dropped (see coverage report)"
        )

    out = kept[[
        "team", "season", "week", "position", "slot", "slot_rank", "player_id",
        "depth_chart_rank", "snap_share_s2d",
    ]].reset_index(drop=True)
    return out


def validate_roster_slots(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("team_week_roster_slots panel is empty")
    # The one invariant this whole data shape depends on: at most one
    # player per (team, season, week, slot). If this ever fails, the
    # fixed-width assumption downstream (mixed-effects / set-model input)
    # is broken, not just this table.
    if panel.duplicated(["team", "season", "week", "slot"]).any():
        raise ValueError("duplicate (team, season, week, slot) rows -- slot uniqueness violated")
    if panel.duplicated(["team", "season", "week", "player_id"]).any():
        raise ValueError("a player was assigned to more than one slot in the same team-week")
    for pos, cap in MAX_SLOTS_PER_POSITION.items():
        bad = panel[(panel["position"] == pos) & (panel["slot_rank"] > cap)]
        if not bad.empty:
            raise ValueError(f"{pos} slot_rank exceeds its cap of {cap}")


def audit_slot_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-position summary: how often each slot is filled, and how often
    the depth-chart default (rank 3, i.e. missing/stale data -- see
    feature_engineering.py's `_add_depth_chart_rank`) drove the ranking
    rather than a real listed rank. Read before trusting any Plan B result
    built on this table -- a position/season where slots are rarely filled
    or the default dominates means the slot assignment itself is mostly
    noise for that population, same discipline as Plan A's coverage report.
    """
    total_team_weeks = panel.drop_duplicates(["team", "season", "week"]).groupby(
        ["season"]
    ).size().rename("team_weeks")

    rows = []
    for pos, cap in MAX_SLOTS_PER_POSITION.items():
        pos_df = panel[panel["position"] == pos]
        for slot_rank in range(1, cap + 1):
            slot_df = pos_df[pos_df["slot_rank"] == slot_rank]
            n_filled = len(slot_df)
            n_default_rank = int((slot_df["depth_chart_rank"] >= 3).sum())
            rows.append({
                "position": pos, "slot": f"{pos}{slot_rank}",
                "n_filled": n_filled,
                "pct_default_depth_chart_rank": round(n_default_rank / n_filled, 3) if n_filled else float("nan"),
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO", "HI"), default=[2013, 2026])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--audit-csv", type=Path,
        default=Path("data") / "experiments" / "team_week_roster_slots_audit.csv",
    )
    args = ap.parse_args()
    lo, hi = sorted(args.seasons)

    conn = sqlite3.connect(str(args.db))
    panel = build_roster_slots(conn, lo, hi, args.db)
    validate_roster_slots(panel)

    print(f"team_week_roster_slots: {len(panel):,} slot-weeks, seasons {lo}-{hi}")

    coverage = audit_slot_coverage(panel)
    print("\nslot coverage (n_filled out of total team-weeks; pct_default_depth_chart_rank "
          "is the share of that slot's assignments driven by the missing/stale-data default "
          "rather than a real listed rank):")
    print(coverage.to_string(index=False))
    if args.audit_csv:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.audit_csv, index=False)
        print(f"wrote coverage audit CSV: {args.audit_csv}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.csv, index=False)
        print(f"wrote CSV: {args.csv}")

    if args.write and not args.dry_run:
        panel.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE_NAME}_key "
            f"ON {TABLE_NAME}(team, season, week, slot)"
        )
        conn.commit()
        print(f"wrote SQLite table: {TABLE_NAME}")
    else:
        print("dry-run: database unchanged")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

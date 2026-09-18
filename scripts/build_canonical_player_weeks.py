#!/usr/bin/env python3
"""Build a canonical player-week participation panel without mutating PPR targets.

Phase 1 goal:
- one row per fantasy-relevant player / regular-season week where the player's
  team actually played and we have roster, snap, or stats evidence;
- preserve confirmed zero snaps vs unknown snaps;
- retain the original stats target only when it truly exists;
- expose evidence flags for later availability/opportunity modeling.

This script NEVER fabricates fantasy_points and NEVER writes to
player_weekly_stats. Output is a standalone SQLite table plus optional CSV.

Evidence precedence:
1. player_weekly_stats: confirms a real stat row exists.
2. snap_counts: authoritative participation measurement when mapped.
3. weekly_rosters_v2, then weekly_rosters: roster/status context.

A missing stat row is not assigned zero PPR here. That decision belongs to a
later modeling/labeling phase after participation evidence is audited.

Usage:
    python scripts/build_canonical_player_weeks.py --dry-run
    python scripts/build_canonical_player_weeks.py --write
    python scripts/build_canonical_player_weeks.py --write --seasons 2013 2026
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, POSITIONS, regular_season_max_week
from src.data.nfl_data_loader import get_pfr_to_gsis_map

TABLE_NAME = "canonical_player_weeks"
TEAM_ALIASES = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "OAK": "LV", "SD": "LAC", "SL": "LA", "STL": "LA",
    "LAR": "LA", "JAC": "JAX",
}


def _norm_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_ALIASES)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_schedule_team_weeks(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    sched = pd.read_sql(
        """SELECT season, week, home_team, away_team
           FROM schedule WHERE season BETWEEN ? AND ?""",
        conn, params=(lo, hi),
    )
    if sched.empty:
        return pd.DataFrame(columns=["season", "week", "team", "opponent", "home_away"])
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")
    sched = sched.dropna(subset=["week"]).copy()
    sched["week"] = sched["week"].astype(int)
    sched = sched[sched["week"] <= sched["season"].map(regular_season_max_week)].copy()

    home = sched.rename(columns={"home_team":"team","away_team":"opponent"})
    home = home[["season","week","team","opponent"]]
    home["home_away"] = "home"
    away = sched.rename(columns={"away_team":"team","home_team":"opponent"})
    away = away[["season","week","team","opponent"]]
    away["home_away"] = "away"
    out = pd.concat([home, away], ignore_index=True)
    out["team"] = _norm_team(out["team"])
    out["opponent"] = _norm_team(out["opponent"])
    return out.drop_duplicates(["season","week","team"])


def load_stats(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    cols = pd.read_sql("PRAGMA table_info(player_weekly_stats)", conn)["name"].tolist()
    keep = ["player_id","season","week","team","position","fantasy_points","snap_count","snap_share"]
    keep = [c for c in keep if c in cols]
    q = f"SELECT {','.join(keep)} FROM player_weekly_stats WHERE season BETWEEN ? AND ?"
    df = pd.read_sql(q, conn, params=(lo, hi))
    if df.empty:
        return pd.DataFrame(columns=keep + ["has_stats_row"])
    df["team"] = _norm_team(df["team"])
    df["has_stats_row"] = 1
    return df.drop_duplicates(["player_id","season","week"], keep="last")


def load_snaps(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    if not _table_exists(conn, "snap_counts"):
        return pd.DataFrame(columns=[
            "player_id","season","week","team","position","offense_snaps","offense_pct",
            "has_snap_row"
        ])
    snaps = pd.read_sql(
        """SELECT season, week, team, position, pfr_player_id,
                  offense_snaps, offense_pct
           FROM snap_counts
           WHERE game_type='REG' AND season BETWEEN ? AND ?""",
        conn, params=(lo, hi),
    )
    if snaps.empty:
        return snaps.assign(player_id=[], has_snap_row=[])
    snaps["team"] = _norm_team(snaps["team"])
    snaps["player_id"] = snaps["pfr_player_id"].map(get_pfr_to_gsis_map())
    snaps = snaps.dropna(subset=["player_id"]).copy()
    snaps = snaps[snaps["position"].isin(POSITIONS)].copy()
    snaps["has_snap_row"] = 1
    return (
        snaps.sort_values("offense_snaps")
        .drop_duplicates(["player_id","season","week"], keep="last")
        [["player_id","season","week","team","position","offense_snaps","offense_pct","has_snap_row"]]
    )


def load_rosters(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    frames = []
    specs = [
        ("weekly_rosters_v2", "player_name"),
        ("weekly_rosters", "full_name"),
    ]
    for priority, (table, name_col) in enumerate(specs):
        if not _table_exists(conn, table):
            continue
        cols = pd.read_sql(f"PRAGMA table_info({table})", conn)["name"].tolist()
        if not {"player_id","season","week","team","position"}.issubset(cols):
            continue
        select = ["player_id","season","week","team","position"]
        for optional in (name_col, "status", "game_type"):
            if optional in cols:
                select.append(optional)
        df = pd.read_sql(
            f"SELECT {','.join(select)} FROM {table} WHERE season BETWEEN ? AND ?",
            conn, params=(lo, hi),
        )
        if df.empty:
            continue
        if "game_type" in df.columns:
            df = df[df["game_type"].fillna("REG").eq("REG")]
        df = df[df["position"].isin(POSITIONS)].copy()
        df["team"] = _norm_team(df["team"])
        df["player_name"] = df[name_col] if name_col in df.columns else None
        if "status" not in df.columns:
            df["status"] = None
        df["roster_source"] = table
        df["_priority"] = priority
        frames.append(df[[
            "player_id","season","week","team","position","player_name",
            "status","roster_source","_priority"
        ]])
    if not frames:
        return pd.DataFrame(columns=[
            "player_id","season","week","team","position","player_name",
            "status","roster_source"
        ])
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("_priority").drop_duplicates(
        ["player_id","season","week"], keep="first"
    )
    return out.drop(columns="_priority")


def load_player_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Stable player-level position fallback for evidence rows lacking weekly position."""
    if not _table_exists(conn, "players"):
        return pd.DataFrame(columns=["player_id", "player_position"])
    cols = pd.read_sql("PRAGMA table_info(players)", conn)["name"].tolist()
    if not {"player_id", "position"}.issubset(cols):
        return pd.DataFrame(columns=["player_id", "player_position"])
    out = pd.read_sql("SELECT player_id, position AS player_position FROM players", conn)
    out = out[out["player_position"].isin(POSITIONS)].drop_duplicates("player_id")
    return out


def build_panel(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    sched = load_schedule_team_weeks(conn, lo, hi)
    stats = load_stats(conn, lo, hi)
    snaps = load_snaps(conn, lo, hi)
    rosters = load_rosters(conn, lo, hi)
    players = load_player_positions(conn)

    keys = ["player_id","season","week"]
    evidence = []
    for df, tag in ((stats, "stats"), (snaps, "snaps"), (rosters, "roster")):
        if not df.empty:
            x = df[keys].drop_duplicates().copy()
            x[f"evidence_{tag}"] = 1
            evidence.append(x)
    if not evidence:
        return pd.DataFrame()

    universe = evidence[0]
    for x in evidence[1:]:
        universe = universe.merge(x, on=keys, how="outer")
    for c in ["evidence_stats","evidence_snaps","evidence_roster"]:
        if c not in universe:
            universe[c] = 0
        universe[c] = universe[c].fillna(0).astype(int)

    roster_cols = keys + ["team","position","player_name","status","roster_source"]
    snap_cols = keys + ["team","position","offense_snaps","offense_pct","has_snap_row"]
    stat_cols = keys + [c for c in stats.columns if c not in keys and c not in ("team","position")]
    roster = rosters[roster_cols] if not rosters.empty else pd.DataFrame(columns=roster_cols)
    snap = snaps[snap_cols] if not snaps.empty else pd.DataFrame(columns=snap_cols)
    stat = stats[stat_cols] if not stats.empty else pd.DataFrame(columns=stat_cols)

    roster = roster.rename(columns={"team":"roster_team","position":"roster_position"})
    snap = snap.rename(columns={"team":"snap_team","position":"snap_position"})
    panel = universe.merge(roster, on=keys, how="left").merge(snap, on=keys, how="left")
    panel = panel.merge(stat, on=keys, how="left")

    # Resolve team/position from evidence; roster first for identity context,
    # snaps second, stats team is recovered below if neither exists.
    panel["team"] = panel["roster_team"].combine_first(panel["snap_team"])
    panel["position"] = panel["roster_position"].combine_first(panel["snap_position"])

    stats_team = stats[keys + ["team"]].rename(columns={"team":"stats_team"}) if not stats.empty else pd.DataFrame(columns=keys+["stats_team"])
    panel = panel.merge(stats_team, on=keys, how="left")
    panel["team"] = panel["team"].combine_first(panel["stats_team"])

    if "position" in stats.columns:
        stats_pos = stats[keys + ["position"]].rename(columns={"position":"stats_position"})
        panel = panel.merge(stats_pos, on=keys, how="left")
        panel["position"] = panel["position"].combine_first(panel["stats_position"])
    else:
        panel["stats_position"] = None

    if not players.empty:
        panel = panel.merge(players, on="player_id", how="left")
        panel["position"] = panel["position"].combine_first(panel["player_position"])
    else:
        panel["player_position"] = None

    # Only include player-weeks where that team actually had a REG game.
    panel = panel.merge(sched, on=["season","week","team"], how="inner")

    panel["has_stats_row"] = panel.get("has_stats_row", 0)
    panel["has_stats_row"] = panel["has_stats_row"].fillna(0).astype(int)
    panel["has_snap_row"] = panel.get("has_snap_row", 0)
    panel["has_snap_row"] = panel["has_snap_row"].fillna(0).astype(int)

    # Three-state participation semantics:
    # confirmed_played: measured >0 offensive snaps
    # confirmed_zero_snaps: measured exactly 0
    # unknown: no mapped authoritative snap measurement
    panel["participation_state"] = "unknown"
    panel.loc[panel["has_snap_row"].eq(1) & panel["offense_snaps"].eq(0), "participation_state"] = "confirmed_zero_snaps"
    panel.loc[panel["has_snap_row"].eq(1) & panel["offense_snaps"].gt(0), "participation_state"] = "confirmed_played"

    panel["missing_stats_row"] = (panel["has_stats_row"] == 0).astype(int)
    panel["played_without_stats_row"] = (
        panel["participation_state"].eq("confirmed_played") & panel["missing_stats_row"].eq(1)
    ).astype(int)
    panel["zero_snaps_without_stats_row"] = (
        panel["participation_state"].eq("confirmed_zero_snaps") & panel["missing_stats_row"].eq(1)
    ).astype(int)

    # Do not fabricate PPR. fantasy_points remains NULL when no real stats row exists.
    if "fantasy_points" in panel.columns:
        panel.loc[panel["has_stats_row"].eq(0), "fantasy_points"] = pd.NA

    keep = [
        "player_id","season","week","team","opponent","home_away","position",
        "player_name","status","roster_source",
        "has_stats_row","has_snap_row","participation_state",
        "offense_snaps","offense_pct","fantasy_points",
        "missing_stats_row","played_without_stats_row","zero_snaps_without_stats_row",
        "evidence_stats","evidence_snaps","evidence_roster",
    ]
    keep = [c for c in keep if c in panel.columns]
    panel = panel[keep].drop_duplicates(["player_id","season","week"])
    return panel.sort_values(["season","week","team","position","player_id"]).reset_index(drop=True)


def validate_panel(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("canonical player-week panel is empty")
    if panel.duplicated(["player_id","season","week"]).any():
        raise ValueError("duplicate player_id/season/week rows")
    if panel["team"].isna().any() or panel["opponent"].isna().any():
        raise ValueError("canonical row missing team/opponent schedule identity")
    if panel["position"].isna().any():
        raise ValueError("canonical row missing fantasy position")
    invalid_positions = sorted(set(panel["position"].dropna()) - set(POSITIONS))
    if invalid_positions:
        raise ValueError(f"canonical row has invalid positions: {invalid_positions}")
    bad = panel["participation_state"].eq("confirmed_played") & panel["offense_snaps"].le(0)
    if bad.any():
        raise ValueError("confirmed_played row has non-positive offense_snaps")
    bad = panel["participation_state"].eq("confirmed_zero_snaps") & panel["offense_snaps"].ne(0)
    if bad.any():
        raise ValueError("confirmed_zero_snaps row is not exactly zero")
    if "fantasy_points" in panel:
        bad = panel["has_stats_row"].eq(0) & panel["fantasy_points"].notna()
        if bad.any():
            raise ValueError("fabricated fantasy_points found on missing-stat rows")


def audit_panel(conn: sqlite3.Connection, panel: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """Run Phase 1 acceptance checks and return season/position population summary."""
    validate_panel(panel)

    # Every canonical stats row must correspond to a real source row and preserve
    # its fantasy-points target exactly. This is the central non-mutation gate.
    raw = load_stats(conn, lo, hi)
    source = raw[["player_id","season","week","fantasy_points"]].drop_duplicates(
        ["player_id","season","week"], keep="last"
    )
    observed = panel.loc[
        panel["has_stats_row"].eq(1),
        ["player_id","season","week","fantasy_points"],
    ]
    chk = observed.merge(
        source, on=["player_id","season","week"], how="left",
        suffixes=("_panel","_source"), indicator=True,
    )
    if not chk["_merge"].eq("both").all():
        raise ValueError("canonical has_stats_row=1 without matching player_weekly_stats source")

    a = pd.to_numeric(chk["fantasy_points_panel"], errors="coerce")
    b = pd.to_numeric(chk["fantasy_points_source"], errors="coerce")
    same = (a.isna() & b.isna()) | (a.eq(b))
    if not same.all():
        raise ValueError("canonical panel altered an existing fantasy_points target")

    # A measured snap row must never be classified unknown.
    bad = panel["has_snap_row"].eq(1) & panel["participation_state"].eq("unknown")
    if bad.any():
        raise ValueError("mapped snap measurement classified as unknown")

    summary = (
        panel.groupby(["season","position"], dropna=False)
        .agg(
            player_weeks=("player_id","size"),
            players=("player_id","nunique"),
            confirmed_played=("participation_state", lambda s: int((s == "confirmed_played").sum())),
            confirmed_zero_snaps=("participation_state", lambda s: int((s == "confirmed_zero_snaps").sum())),
            unknown=("participation_state", lambda s: int((s == "unknown").sum())),
            missing_stats_rows=("missing_stats_row","sum"),
            played_without_stats_rows=("played_without_stats_row","sum"),
            zero_snaps_without_stats_rows=("zero_snaps_without_stats_row","sum"),
        )
        .reset_index()
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO","HI"), default=[2013, 2026])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--audit-csv", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "canonical_player_weeks_audit.csv")
    args = ap.parse_args()
    lo, hi = sorted(args.seasons)

    conn = sqlite3.connect(str(DB_PATH))
    panel = build_panel(conn, lo, hi)
    summary = audit_panel(conn, panel, lo, hi)

    print(f"canonical player-weeks: {len(panel):,}")
    print(panel["participation_state"].value_counts(dropna=False).to_string())
    print(f"missing stats rows: {int(panel['missing_stats_row'].sum()):,}")
    print(f"played but no stats row: {int(panel['played_without_stats_row'].sum()):,}")
    print(f"zero snaps but no stats row: {int(panel['zero_snaps_without_stats_row'].sum()):,}")
    print("\nseason/position audit:")
    print(summary.to_string(index=False))
    if args.audit_csv:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.audit_csv, index=False)
        print(f"wrote audit CSV: {args.audit_csv}")

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
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_season_week "
            f"ON {TABLE_NAME}(season, week)"
        )
        conn.commit()
        print(f"wrote SQLite table: {TABLE_NAME}")
    else:
        print("dry-run: database unchanged")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build the complete player-game panel: fixes training-target selection bias
in `player_weekly_stats`, which only contains rows for players who recorded
a real stat line (confirmed: `nfl_data_py.import_weekly_data()` is itself
box-score-triggered, not an ingestion-code filter). This means "played but
produced zero fantasy points" is silently indistinguishable from "didn't
play at all" -- a real problem for any model, and specifically breaks a
two-stage/hurdle architecture's ability to learn P(PPR > 0).

Two-tier eligibility rule (see GAPS.md for full reasoning):
  - 2018-2025: snap_counts (PFR->GSIS mapped) confirms offense_snaps > 0.
    Precise "took the field" signal. Tag: inferred_snap_verified_zero.
  - 2006-2017: weekly_rosters status='ACT', tightened to players with >=1
    REAL player_weekly_stats row elsewhere that season (bridges "rostered,
    in-rotation player's off week" without injecting rows for players who
    never factored into any game that season). Weaker signal, explicitly
    tagged low-confidence. Tag: inferred_roster_zero_low_confidence.

Never touches bye weeks, non-rostered players, or players with no
reasonable evidence of participation. Position scope: QB/RB/WR/TE only.

Usage:
    python scripts/build_complete_player_game_panel.py               # dry-run (default)
    python scripts/build_complete_player_game_panel.py --write       # real insert, backs up DB first
    python scripts/build_complete_player_game_panel.py --write --seasons 2018 2025
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

SNAP_COUNT_MIN_SEASON = 2018
REGULAR_SEASON_MAX_WEEK = 18

# weekly_rosters uses era-specific team abbreviations; `schedule` normalizes
# to current codes retroactively (confirmed: schedule has no 'OAK'/'SD'/'SL'
# rows even for 2006-2017 seasons, only 'LV'/'LAC'/'LA'). Same class of bug
# as the already-documented LAR/JAC mismatch (GAPS.md Sec 9.2), different
# table/codes. Without this, ~29% of the 2006-2017 tier's candidate rows
# silently lose their schedule-derived opponent and get dropped.
TEAM_CODE_NORMALIZATION = {
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "OAK": "LV", "SD": "LAC", "SL": "LA", "LAR": "LA", "JAC": "JAX",
}
AUDIT_CSV_PATH = PROJECT_ROOT / "data" / "experiments" / "complete_panel_new_players_audit.csv"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def _team_opponent_home_away_map(conn: sqlite3.Connection) -> pd.DataFrame:
    """(season, week, team) -> (opponent, home_away) from schedule, regular season only."""
    sched = pd.read_sql(
        "SELECT season, week, home_team, away_team FROM schedule WHERE week <= ?",
        conn, params=[REGULAR_SEASON_MAX_WEEK],
    )
    home = sched.rename(columns={"home_team": "team", "away_team": "opponent"})[["season", "week", "team", "opponent"]]
    home["home_away"] = "home"
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent"})[["season", "week", "team", "opponent"]]
    away["home_away"] = "away"
    return pd.concat([home, away], ignore_index=True).drop_duplicates(subset=["season", "week", "team"])


def load_snap_verified_candidates(conn: sqlite3.Connection, seasons: tuple) -> pd.DataFrame:
    """2018+: players with offense_snaps > 0 in snap_counts, PFR->GSIS mapped, no existing stats row."""
    from src.data.nfl_data_loader import get_pfr_to_gsis_map

    lo, hi = max(seasons[0], SNAP_COUNT_MIN_SEASON), seasons[1]
    if lo > hi:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "position", "player", "data_source"])

    placeholders = ",".join("?" * len(POSITIONS))
    snaps = pd.read_sql(
        f"""SELECT season, week, team, position, pfr_player_id, player, offense_snaps
            FROM snap_counts
            WHERE game_type = 'REG' AND offense_snaps > 0 AND season BETWEEN ? AND ?
              AND position IN ({placeholders})""",
        conn, params=[lo, hi] + list(POSITIONS),
    )
    if snaps.empty:
        return snaps.assign(player_id=[], data_source=[])

    pfr_to_gsis = get_pfr_to_gsis_map()
    snaps["player_id"] = snaps["pfr_player_id"].map(pfr_to_gsis)
    unmapped = snaps["player_id"].isna().sum()
    if unmapped:
        print(f"  snap_counts: {unmapped}/{len(snaps)} rows have no PFR->GSIS mapping, dropped")
    snaps = snaps.dropna(subset=["player_id"]).copy()
    snaps["data_source"] = "inferred_snap_verified_zero"
    return snaps[["player_id", "season", "week", "team", "position", "player", "data_source"]]


def load_roster_candidates(conn: sqlite3.Connection, seasons: tuple) -> pd.DataFrame:
    """2006-2017: roster ACT status, tightened to players with >=1 real stat row that season."""
    lo, hi = seasons[0], min(seasons[1], SNAP_COUNT_MIN_SEASON - 1)
    if lo > hi:
        return pd.DataFrame(columns=["player_id", "season", "week", "team", "position", "player", "data_source"])

    placeholders = ",".join("?" * len(POSITIONS))
    rosters = pd.read_sql(
        f"""SELECT player_id, season, week, team, position, full_name AS player
            FROM weekly_rosters
            WHERE game_type = 'REG' AND status = 'ACT' AND season BETWEEN ? AND ?
              AND position IN ({placeholders})""",
        conn, params=[lo, hi] + list(POSITIONS),
    )
    if rosters.empty:
        return rosters.assign(data_source=[])

    real_seasons = pd.read_sql(
        "SELECT DISTINCT player_id, season FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
        conn, params=[lo, hi],
    )
    real_seasons["has_real_row_this_season"] = True
    merged = rosters.merge(real_seasons, on=["player_id", "season"], how="left")
    merged = merged[merged["has_real_row_this_season"] == True].drop(columns=["has_real_row_this_season"])
    merged["data_source"] = "inferred_roster_zero_low_confidence"
    return merged


def build_candidate_rows(conn: sqlite3.Connection, seasons: tuple) -> pd.DataFrame:
    snap_candidates = load_snap_verified_candidates(conn, seasons)
    roster_candidates = load_roster_candidates(conn, seasons)
    candidates = pd.concat([snap_candidates, roster_candidates], ignore_index=True)
    if candidates.empty:
        return candidates

    # Normalize era-specific team codes to what `schedule` and
    # `player_weekly_stats` both already use (confirmed identical modern
    # codes in both) -- see TEAM_CODE_NORMALIZATION comment.
    candidates["team"] = candidates["team"].replace(TEAM_CODE_NORMALIZATION)

    existing = pd.read_sql(
        "SELECT player_id, season, week FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
        conn, params=list(seasons),
    )
    existing_keys = set(zip(existing.player_id, existing.season, existing.week))
    candidates = candidates[
        ~candidates.apply(lambda r: (r.player_id, r.season, r.week) in existing_keys, axis=1)
    ].drop_duplicates(subset=["player_id", "season", "week"])

    ctx = _team_opponent_home_away_map(conn)
    candidates = candidates.merge(ctx, on=["season", "week", "team"], how="left")
    return candidates


def resolve_player_identities(conn: sqlite3.Connection, candidates: pd.DataFrame) -> tuple:
    """Exact gsis_id lookup against `players` -- not fuzzy matching. Returns
    (candidates_to_insert, new_players_df, ambiguous_df). Ambiguous
    (existing player_id with a DIFFERENT position on file) rows are
    excluded from insertion, not silently resolved either way.
    """
    existing_players = pd.read_sql("SELECT player_id, name, position FROM players", conn)
    existing_map = existing_players.set_index("player_id")["position"].to_dict()

    known_ids = set(existing_map.keys())
    candidate_ids = set(candidates["player_id"].unique())
    new_ids = candidate_ids - known_ids

    def _is_ambiguous(row):
        known_pos = existing_map.get(row["player_id"])
        return known_pos is not None and known_pos != row["position"]

    ambiguous_mask = candidates.apply(_is_ambiguous, axis=1)
    ambiguous = candidates[ambiguous_mask]
    clean = candidates[~ambiguous_mask]

    new_players = (
        clean[clean["player_id"].isin(new_ids)][["player_id", "player", "position", "season"]]
        .sort_values("season")
        .drop_duplicates(subset=["player_id"], keep="first")
        .rename(columns={"player": "name", "season": "first_season_added"})
    )
    new_players["source"] = "complete_player_game_panel"

    return clean, new_players, ambiguous


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="Perform real inserts (default: dry-run only)")
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO", "HI"), default=[2006, 2025])
    args = ap.parse_args()
    seasons = tuple(sorted(args.seasons))

    conn = _connect()
    print(f"Building candidate rows for seasons {seasons[0]}-{seasons[1]} ...")
    candidates = build_candidate_rows(conn, seasons)
    print(f"\n{len(candidates)} candidate rows before identity resolution.")

    if candidates.empty:
        print("Nothing to do.")
        return 0

    clean, new_players, ambiguous = resolve_player_identities(conn, candidates)

    print(f"\n{'=' * 70}\nZero-row eligibility audit\n{'=' * 70}")
    audit = (
        clean.assign(is_snap_verified=clean["data_source"] == "inferred_snap_verified_zero")
        .groupby(["season", "position", "data_source"])
        .size()
        .unstack(fill_value=0)
    )
    print(audit.to_string())
    print(f"\nTotal new rows: {len(clean)}")
    print(f"  inferred_snap_verified_zero: {(clean.data_source == 'inferred_snap_verified_zero').sum()}")
    print(f"  inferred_roster_zero_low_confidence: {(clean.data_source == 'inferred_roster_zero_low_confidence').sum()}")
    print(f"Players affected: {clean['player_id'].nunique()}")
    print(f"New players (no prior `players` row): {len(new_players)}")
    print(f"Ambiguous (position conflict, EXCLUDED from insert): {len(ambiguous)}")
    if not ambiguous.empty:
        print(ambiguous[["player_id", "player", "position", "season", "week"]].head(20).to_string())

    snap_rows = clean[clean.data_source == "inferred_snap_verified_zero"]
    if not snap_rows.empty:
        # By construction this is 100% -- load_snap_verified_candidates only
        # ever selects snap_counts rows with offense_snaps > 0 to begin with.
        # Stated explicitly (not just implied) since the plan called for this
        # check to be visible in the audit output, not just true in code.
        print(f"\nsnap-verified rows: {len(snap_rows)} "
              f"(100% have offense_snaps > 0 by construction -- that's the selection criterion)")

    missing_opponent = clean["opponent"].isna().sum()
    if missing_opponent:
        print(f"\nWARNING: {missing_opponent} candidate rows have no schedule-derived opponent (dropped before insert)")
        clean = clean.dropna(subset=["opponent"])

    if not args.write:
        print("\n--dry-run (default): no writes. Re-run with --write to apply.")
        return 0

    backup_path = DB_PATH.parent / f"nfl_data.db.bak-panel-build-{datetime.now():%Y%m%d%H%M%S}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"\nBacked up database to {backup_path}")

    cur = conn.cursor()
    if not new_players.empty:
        for _, r in new_players.iterrows():
            cur.execute(
                "INSERT OR IGNORE INTO players (player_id, name, position) VALUES (?, ?, ?)",
                (r["player_id"], r["name"], r["position"]),
            )
        conn.commit()
        AUDIT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_players.to_csv(AUDIT_CSV_PATH, index=False)
        print(f"Inserted {len(new_players)} new `players` rows, logged to {AUDIT_CSV_PATH}")

    inserted = 0
    for _, r in clean.iterrows():
        cur.execute(
            """INSERT OR IGNORE INTO player_weekly_stats
               (player_id, season, week, team, opponent, home_away, fantasy_points, data_source)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (r["player_id"], int(r["season"]), int(r["week"]), r["team"], r["opponent"],
             r["home_away"], r["data_source"]),
        )
        inserted += cur.rowcount
    conn.commit()
    print(f"Inserted {inserted} new player_weekly_stats rows.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

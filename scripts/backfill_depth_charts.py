#!/usr/bin/env python
"""Load new-schema depth charts, bridging nflverse's ESPN-style feed.

`feature_engineering.py` recorded that 2025 "is NOT backfillable
(import_depth_charts([2025]) fails upstream)". That is wrong: the call
succeeds, but returns a completely different feed. The old weekly file
(2013-2024) had one row per team/week/slot with season/week/depth_team;
the new one is a DAILY league-wide snapshot keyed on `dt`, ranking players
within a position slot via `pos_rank`:

    old: season club_code week depth_team position depth_position gsis_id
    new: dt team pos_grp pos_abb pos_rank pos_slot gsis_id espn_id

554,215 rows across 221 daily snapshots (2025-08-03 .. 2026-03-14). Those
are exactly the "season IS NULL junk rows" that `_load_depth_chart_asof_table`
filters out -- not junk, just unmapped.

Week assignment is deliberately conservative: a week's snapshot is the last
one taken STRICTLY BEFORE the calendar date of that week's first kickoff.
`schedule.game_time` is date-only, so an intra-day comparison isn't
available; taking the prior day costs a few hours of freshness and makes
leakage structurally impossible, matching the stricter-cutoff precedent in
season_projection.py::_lookup_depth_chart_rank_asof.

Usage:
    python scripts/backfill_depth_charts.py                 # current season
    python scripts/backfill_depth_charts.py --season 2025
    python scripts/backfill_depth_charts.py --dry-run
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# The season is a parameter, not a constant: the new schema started in 2025
# and every season since ships the same daily feed, so hardcoding one meant
# each new season silently had no depth charts at all.

# Target schema, in `depth_charts` column order.
COLUMNS = ["season", "club_code", "week", "depth_team", "last_name",
           "first_name", "football_name", "position", "jersey_number",
           "gsis_id", "depth_position", "full_name"]

# The old feed's `depth_team` only ever takes 1/2/3 -- across 2013-2024 there
# is no 4. `pos_rank` runs to 8+, so it is clipped rather than stored raw:
# an unclipped 7 would be a value the models have never seen, and 3 is
# already the neutral fallback in _add_depth_chart_rank. This deliberately
# discards "how deep beyond third string", which the old data never carried.
MAX_DEPTH_TEAM = 3

# `pos_abb` -> the old feed's normalized `position` vocabulary. The new feed
# only supplies the slot label, so left/right variants collapse and the
# nickel back lands in the generic DB bucket the old data also used.
POS_ABB_TO_POSITION = {
    "QB": "QB", "RB": "RB", "FB": "FB", "WR": "WR", "TE": "TE",
    "LT": "T", "RT": "T", "LG": "G", "RG": "G", "C": "C",
    "LDE": "DE", "RDE": "DE", "LDT": "DT", "RDT": "DT", "NT": "NT",
    "MLB": "MLB", "SLB": "OLB", "WLB": "OLB", "LILB": "ILB", "RILB": "ILB",
    "LCB": "CB", "RCB": "CB", "NB": "DB", "FS": "FS", "SS": "SS",
    "PK": "K", "P": "P", "LS": "LS", "H": "P",
}

# Returner slots carry no position of their own; resolve them from the same
# player's offensive/defensive listing in that snapshot (below).
RETURNER_SLOTS = {"KR", "PR"}


FIRST_POSTSEASON_WEEK = 19


def _week_boundaries(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """First kickoff date per (season, week). game_time is date-only."""
    sched = pd.read_sql(
        "SELECT week, MIN(game_time) AS first_game FROM schedule "
        "WHERE season = ? AND game_time IS NOT NULL "
        "GROUP BY week ORDER BY week",
        conn, params=(int(season),),
    )
    sched["first_game"] = pd.to_datetime(sched["first_game"], errors="coerce")
    return sched.dropna(subset=["first_game"])


def _teams_by_week(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """(week, team) for every club actually playing that week."""
    return pd.read_sql(
        "SELECT week, home_team AS team FROM schedule WHERE season = ? "
        "UNION SELECT week, away_team AS team FROM schedule WHERE season = ?",
        conn, params=(int(season), int(season)),
    )


def _drop_eliminated_teams(out: pd.DataFrame, playing: pd.DataFrame) -> pd.DataFrame:
    """Keep postseason rows only for clubs still playing.

    The old weekly feed narrowed naturally -- 2023 carries 14 teams in week
    20, 8 in 21, 2 in 22 -- because it was published per game. The new feed
    is a league-wide daily snapshot, so an eliminated team keeps emitting a
    depth chart into February. Left alone that is harmless downstream (those
    players have no game rows to merge onto) but it would make 2025 the only
    season whose postseason lists all 32 clubs, so it is filtered here rather
    than left as a silent shape difference.

    Regular-season weeks are untouched: the old data keeps bye-week teams,
    all 32 every week.
    """
    post = out["week"] >= FIRST_POSTSEASON_WEEK
    valid = set(zip(playing["week"], playing["team"]))
    in_game = pd.Series(list(zip(out["week"], out["club_code"])),
                        index=out.index).isin(valid)
    return out[~post | in_game]


def _assign_snapshots(raw: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Pick one daily snapshot per week: the latest strictly before kickoff."""
    dt = pd.to_datetime(raw["dt"], errors="coerce", utc=True).dt.tz_localize(None)
    raw = raw.assign(_dt=dt).dropna(subset=["_dt"])
    snapshot_days = pd.Series(sorted(raw["_dt"].unique()))

    chosen = {}
    for _, row in weeks.iterrows():
        prior = snapshot_days[snapshot_days < row["first_game"]]
        if prior.empty:
            print(f"  week {int(row['week']):>2}: no snapshot before "
                  f"{row['first_game'].date()} — skipped")
            continue
        chosen[prior.iloc[-1]] = int(row["week"])

    picked = raw[raw["_dt"].isin(chosen)].copy()
    picked["week"] = picked["_dt"].map(chosen)
    return picked


def _resolve_returner_positions(df: pd.DataFrame) -> pd.Series:
    """KR/PR rows get the position the same player holds elsewhere that week."""
    position = df["pos_abb"].map(POS_ABB_TO_POSITION)
    known = df.loc[position.notna(), ["gsis_id", "week"]].assign(
        _pos=position[position.notna()])
    # A player can hold several slots; take the most common real position.
    lookup = (known.groupby(["gsis_id", "week"])["_pos"]
              .agg(lambda s: s.value_counts().idxmax()))
    fallback = pd.MultiIndex.from_arrays([df["gsis_id"], df["week"]]).map(lookup)
    return position.fillna(pd.Series(fallback, index=df.index))


def _to_target_schema(picked: pd.DataFrame, season: int) -> pd.DataFrame:
    out = pd.DataFrame(index=picked.index)
    out["season"] = int(season)
    out["club_code"] = picked["team"]
    out["week"] = picked["week"].astype(int)
    out["depth_team"] = (
        pd.to_numeric(picked["pos_rank"], errors="coerce")
        .clip(upper=MAX_DEPTH_TEAM).astype("Int64").astype(str)
    )
    names = picked["player_name"].fillna("").str.strip()
    out["first_name"] = names.str.split(" ").str[0]
    out["last_name"] = names.str.split(" ").str[1:].str.join(" ")
    # Not carried by the new feed; the old one had them.
    out["football_name"] = None
    out["jersey_number"] = None
    out["position"] = _resolve_returner_positions(picked)
    out["gsis_id"] = picked["gsis_id"]
    out["depth_position"] = picked["pos_abb"]
    out["full_name"] = names
    return out[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--season", type=int, default=None,
                    help="season to load (default: the current NFL season)")
    ap.add_argument("--cache", default=None,
                    help="reuse a saved raw pull instead of refetching ~500K rows "
                         "(default data/raw/depth_charts_<season>_raw.parquet)")
    args = ap.parse_args()

    from config.settings import DB_PATH
    from src.utils.nfl_calendar import get_current_nfl_season
    season = args.season or get_current_nfl_season()
    print(f"season {season}")
    conn = sqlite3.connect(str(DB_PATH))

    cache = Path(args.cache or f"data/raw/depth_charts_{season}_raw.parquet")
    if cache.exists():
        raw = pd.read_parquet(cache)
        print(f"raw: {len(raw):,} rows from {cache}")
    else:
        import nfl_data_py as nfl
        raw = nfl.import_depth_charts([season])
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache)
        print(f"raw: {len(raw):,} rows pulled and cached to {cache}")

    if "pos_rank" not in raw.columns:
        print("upstream returned the OLD schema — use the generic backfill path")
        return

    raw = raw.dropna(subset=["gsis_id"])
    print(f"after dropping null gsis_id: {len(raw):,} rows")

    weeks = _week_boundaries(conn, season)
    print(f"schedule weeks: {len(weeks)} ({int(weeks.week.min())}-{int(weeks.week.max())})")

    picked = _assign_snapshots(raw, weeks)
    out = _to_target_schema(picked, season)

    before = len(out)
    out = _drop_eliminated_teams(out, _teams_by_week(conn, season))
    print(f"dropped {before - len(out):,} postseason rows for eliminated clubs")

    print(f"\nmapped {len(out):,} rows across {out.week.nunique()} weeks")
    print(out.groupby("week").agg(rows=("gsis_id", "size"),
                                  teams=("club_code", "nunique")).to_string())
    print("\ndepth_team:", out.depth_team.value_counts().to_dict())
    skill = out[out.position.isin(["QB", "RB", "WR", "TE"])]
    print(f"skill rows: {len(skill):,} ({skill.gsis_id.nunique()} players)")
    print("unmapped position rows:", int(out.position.isna().sum()))

    existing = pd.read_sql(
        "SELECT COUNT(*) n FROM depth_charts WHERE season = ?", conn,
        params=(int(season),)).n.iloc[0]
    if existing:
        print(f"\n{season} already has {existing:,} rows — deleting before reload")

    if args.dry_run:
        print(f"\nDRY RUN — would write {len(out):,} rows")
        return

    if existing:
        conn.execute("DELETE FROM depth_charts WHERE season = ?", (int(season),))
        conn.commit()
    out.to_sql("depth_charts", conn, if_exists="append", index=False)

    print(pd.read_sql(
        "SELECT season, COUNT(*) rows, COUNT(DISTINCT week) weeks, "
        "COUNT(DISTINCT gsis_id) players FROM depth_charts "
        "GROUP BY season ORDER BY season", conn).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()

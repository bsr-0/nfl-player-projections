#!/usr/bin/env python
"""Can snap data classify rostered player-weeks that have no stat line?

For target/population construction, not availability weighting. A player-week
where the player was on an active roster and his team played, but no row
exists in player_weekly_stats, is currently either absent from training or
manufactured as a synthetic week. Snap data should let us tell three cases
apart:

    snap row, offense_snaps > 0   -> he PLAYED; a legitimate observation we
                                     are missing, and inventing a synthetic
                                     row for it is wrong
    snap row, offense_snaps == 0  -> he was there and did not take an
                                     offensive snap; a legitimate zero, not
                                     something to synthesise
    no snap row                   -> genuinely unknown

The question is not whether historical snaps exist -- they do, 2013+ -- but
whether the zero/nonzero signal is dense and reliable enough for the
player-weeks we actually care about. Reported per position because the
synthetic-row contamination was far worse for RB/WR/TE than QB.

Usage:
    python scripts/audit_rostered_no_stats.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=list(range(2013, 2026)))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/rostered_no_stats_audit.csv"))
    args = ap.parse_args()

    from config.settings import DB_PATH
    from src.data.nfl_data_loader import get_pfr_to_gsis_map
    conn = sqlite3.connect(str(DB_PATH))
    lo, hi = min(args.seasons), max(args.seasons)

    # Weeks each team actually played (excludes byes).
    from config.settings import regular_season_max_week

    sched = pd.read_sql(
        f"""SELECT season, week, home_team AS team FROM schedule
            WHERE season BETWEEN {lo} AND {hi}
            UNION SELECT season, week, away_team FROM schedule
            WHERE season BETWEEN {lo} AND {hi}""", conn)
    sched["week"] = pd.to_numeric(sched.week, errors="coerce")
    sched = sched.dropna(subset=["week"])
    sched["week"] = sched.week.astype(int)
    # Era-aware: 17 through 2020, 18 from 2021. A flat 18 marked the
    # pre-2021 wild-card round as "team played this week", so playoff
    # participants looked rostered-but-statless for a game that was
    # never part of the regular season.
    sched = sched[sched.week <= sched.season.map(regular_season_max_week)]
    sched["played_game"] = True

    # Universe: active-roster skill-position player-weeks whose team played.
    rosters = pd.read_sql(
        f"""SELECT player_id, season, week, team, position FROM weekly_rosters
            WHERE game_type='REG' AND status='ACT'
              AND season BETWEEN {lo} AND {hi}
              AND position IN ({','.join(repr(p) for p in POSITIONS)})""", conn)
    rosters["week"] = rosters.week.astype(int)
    rosters = rosters.drop_duplicates(["player_id", "season", "week"])
    universe = rosters.merge(sched, on=["season", "week", "team"], how="inner")

    stats = pd.read_sql(
        f"""SELECT DISTINCT player_id, season, week, 1 AS has_stats
            FROM player_weekly_stats WHERE season BETWEEN {lo} AND {hi}""", conn)
    stats["week"] = stats.week.astype(int)
    u = universe.merge(stats, on=["player_id", "season", "week"], how="left")
    u["has_stats"] = u.has_stats.fillna(0).astype(int)

    snaps = pd.read_sql(
        f"""SELECT season, week, pfr_player_id, offense_snaps FROM snap_counts
            WHERE game_type='REG' AND season BETWEEN {lo} AND {hi}""", conn)
    snaps["week"] = snaps.week.astype(int)
    snaps["player_id"] = snaps.pfr_player_id.map(get_pfr_to_gsis_map())
    snaps = snaps.dropna(subset=["player_id"])
    snaps = (snaps.groupby(["player_id", "season", "week"], as_index=False)
             ["offense_snaps"].max())
    u = u.merge(snaps, on=["player_id", "season", "week"], how="left")

    def classify(r):
        if pd.isna(r.offense_snaps):
            return "no snap row"
        return "snap row, played" if r.offense_snaps > 0 else "snap row, zero snaps"

    gap = u[u.has_stats == 0].copy()
    gap["classification"] = gap.apply(classify, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gap.to_csv(args.output, index=False)

    print("\n" + "=" * 84)
    print("ROSTERED, TEAM PLAYED, NO STAT ROW — can snaps classify it?")
    print("=" * 84)
    print(f"\nuniverse (ACT roster + team played): {len(u):,} player-weeks")
    print(f"  with a stat row    : {int((u.has_stats==1).sum()):,}")
    print(f"  WITHOUT a stat row : {len(gap):,}  <- the population in question\n")

    tab = pd.crosstab(gap.position, gap.classification)
    pct = (tab.div(tab.sum(axis=1), axis=0) * 100).round(1)
    print("counts by position:")
    print(tab.reindex(POSITIONS).to_string())
    print("\npercent by position:")
    print(pct.reindex(POSITIONS).to_string())

    print("\n\nBY ERA (is the older data as good?)")
    gap["era"] = pd.cut(gap.season, [2012, 2017, 2025], labels=["2013-2017", "2018-2025"])
    era = pd.crosstab(gap.era, gap.classification, normalize="index") * 100
    print(era.round(1).to_string())

    resolvable = gap.classification.ne("no snap row").mean() * 100
    print(f"\n  resolvable by snaps overall: {resolvable:.1f}% of the gap population")
    conn.close()


if __name__ == "__main__":
    main()

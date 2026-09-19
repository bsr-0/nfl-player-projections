#!/usr/bin/env python3
"""
One-time backfill of `team_stats.points_scored`/`points_allowed`/`turnovers`/
`third_down_conv` for historical seasons.

Root cause (see src/data/nfl_data_loader.py's `_local_schedule_df` and the
comment at its call site, added 2026-09-18): `turnovers`/`third_down_conv`
were already computed correctly by the PBP aggregator and cached to disk
(`data/raw/pbp_team_advanced_{season}.parquet`), but `team_stats` itself
held stale pre-fix zeros that were never re-written. `points_scored`/
`points_allowed` come from a separate enrichment step that used to fetch
schedule data over the network (`_fetch_schedules`) -- a flaky remote host
(the same one `backfill_vegas_lines.py` documents avoiding) -- and silently
no-op'd on failure, leaving those two columns at their schema default of 0
for years while everything else in the same row wrote fine. The loader now
sources schedule data locally instead; this script re-runs that (already
fixed) enrichment against historical seasons so the correction actually
lands in the DB.

Idempotent: `insert_team_stats`'s upsert COALESCEs new values over old, so
re-running this is always safe and never regresses a column with a
genuinely-missing (NULL) new value back over a real existing one.

Usage:
    python scripts/backfill_team_stats_points_turnovers.py                # all seasons 2006-current
    python scripts/backfill_team_stats_points_turnovers.py -s 2006 2022   # explicit range
    python scripts/backfill_team_stats_points_turnovers.py --dry-run      # preview only, no writes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.settings import CURRENT_NFL_SEASON, MIN_HISTORICAL_YEAR
from src.data.nfl_data_loader import NFLDataLoader, _enrich_team_stats_from_schedule
from src.data.pbp_stats_aggregator import get_team_stats_from_pbp


def backfill_season(loader: NFLDataLoader, season: int, dry_run: bool) -> dict:
    team_df = get_team_stats_from_pbp(season, use_cache=True)
    if team_df is None or team_df.empty:
        return {"season": season, "status": "no_pbp_team_data", "n_rows": 0}

    before_turnovers = team_df["turnovers"].mean() if "turnovers" in team_df.columns else None

    sched_df = loader._local_schedule_df(season)
    if sched_df.empty:
        return {"season": season, "status": "no_local_schedule", "n_rows": len(team_df)}

    enriched = _enrich_team_stats_from_schedule(team_df, sched_df)
    points_mean = enriched["points_scored"].mean() if "points_scored" in enriched.columns else None

    if not dry_run:
        loader._store_team_stats_dataframe(enriched)

    return {
        "season": season,
        "status": "written" if not dry_run else "dry_run",
        "n_rows": len(enriched),
        "turnovers_mean": round(float(before_turnovers), 3) if before_turnovers is not None else None,
        "points_scored_mean": round(float(points_mean), 3) if points_mean is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--seasons", nargs=2, type=int, metavar=("START", "END"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start, end = args.seasons if args.seasons else (MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON)
    seasons = list(range(start, end + 1))

    loader = NFLDataLoader()
    results = []
    for season in seasons:
        result = backfill_season(loader, season, args.dry_run)
        results.append(result)
        print(f"  {season}: {result}")

    df = pd.DataFrame(results)
    print(f"\n{'DRY RUN -- ' if args.dry_run else ''}Backfilled {len(df)} seasons.")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

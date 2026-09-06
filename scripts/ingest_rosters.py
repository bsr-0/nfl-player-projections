"""Load nflverse seasonal rosters into the `rosters` table.

WHY THIS EXISTS. `preseason_features._team_for_season` derives a player's team
from player_weekly_stats, which has no rows for a season nobody has played, so
before this table was populated the 2026 inference frame arrived with
`dest_team` NaN, `team_changed` 0 and both destination profiles zero-filled --
four features the model trains on, served as zeros (GAPS.md 2026-09-06). The
board also reads this table to point each player at the team he is on now
rather than the one he played for last year, which is what puts the right
week-1 opponent on the page.

So the projections depend on this data, the database is gitignored, and
nothing else declares the dependency. Run it after a fresh clone, and again
whenever rosters move.

REPLACES THE WHOLE TABLE. `bulk_insert_rosters` writes with if_exists=replace,
so a run must fetch every season worth keeping. Seasons already in the table
are added to the requested range rather than dropped -- loading only 2026
would otherwise silently delete the history the backtests use.

Usage:
    python scripts/ingest_rosters.py
    python scripts/ingest_rosters.py --seasons 2019 2026
    python scripts/ingest_rosters.py --dry-run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH

# The backtests refit step 8 from 2019 forward; earlier rosters have no
# consumer today.
DEFAULT_FIRST = 2019


def existing_seasons() -> list:
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = pd.read_sql("SELECT DISTINCT season FROM rosters", conn)
    except Exception:                                # noqa: BLE001
        return []
    finally:
        conn.close()
    return sorted(int(s) for s in rows["season"].dropna())


def seasons_to_fetch(requested, existing) -> list:
    """Everything asked for, plus everything already stored.

    The insert replaces the table, so a season left out of the fetch is a
    season deleted. Keeping the union makes a narrow run additive instead.
    """
    return sorted(set(requested) | set(existing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=None,
                    help=f"season range to load (default {DEFAULT_FIRST} "
                         "through the upcoming season)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch and report without writing")
    args = ap.parse_args()

    from src.utils.nfl_calendar import get_projection_season
    if args.seasons:
        requested = range(args.seasons[0], args.seasons[1] + 1)
    else:
        requested = range(DEFAULT_FIRST, get_projection_season() + 1)

    have = existing_seasons()
    seasons = seasons_to_fetch(requested, have)
    added = sorted(set(seasons) - set(have))
    print(f"stored: {have[0] if have else '-'}"
          f"{'-' + str(have[-1]) if have else ''}  "
          f"fetching {seasons[0]}-{seasons[-1]}"
          + (f"  (new: {added})" if added else ""))

    import nfl_data_py as nfl
    df = nfl.import_seasonal_rosters(list(seasons))
    skill = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    print(f"  {len(df):,} rows, {len(skill):,} at skill positions")
    counts = skill.groupby("season").size()
    print("  skill rows by season: "
          + ", ".join(f"{int(s)}:{n}" for s, n in counts.items()))

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    from src.utils.database import DatabaseManager
    written = DatabaseManager().bulk_insert_rosters(df)
    print(f"\nWrote {written:,} rows to `rosters`.")
    print("Regenerate the board to pick them up: "
          "python scripts/generate_draft_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

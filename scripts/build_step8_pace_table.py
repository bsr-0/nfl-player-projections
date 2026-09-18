"""Build the Step 8 season-pace table the weekly ensemble reads as a feature.

One row per (player, season): the Step 8 season-total projection / 17,
refit STRICTLY on seasons before the one projected (same training window,
2019+, as scripts/generate_draft_data.py uses for the live board), so every
value is out-of-sample and the training frame sees exactly the kind of
number serving sees. Cold-start rookies are covered the same way production
covers them; players with no Step 8 row (no prior season and not in the
draft class) stay absent -> NaN in the frame, on both paths.

Seasons before FIRST_SEASON have no value: the training window would be
under two seasons. The weekly model learns the feature on 2021+ rows and
sees NaN for the older era, which trees route like any other structural
missingness. The upcoming season is included so serving has its value.

Output: data/step8_pace_by_season.csv (config.settings.STEP8_PACE_TABLE).

Usage:
    python scripts/build_step8_pace_table.py
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from config.settings import STEP8_PACE_TABLE  # noqa: E402

FIRST_SEASON = 2021


def main() -> int:
    from run_week1_coldstart_experiment import step8_pace
    from src.utils.data_manager import DataManager

    from src.utils.database import DatabaseManager
    latest = max(DataManager().get_available_seasons_from_db())
    seasons = list(range(FIRST_SEASON, latest + 1))
    # The next season only once its schedule exists (i.e. the offseason
    # before it). With the 3-season anchor window, pairs can be built for
    # any future season, so without this cap the table grew a "2027" that
    # nothing serves and that a reader could mistake for a projection.
    if DatabaseManager().has_schedule_for_season(latest + 1):
        seasons.append(latest + 1)
    frames = []
    for season in seasons:
        df = step8_pace(season)
        if df.empty:
            print(f"  {season}: no rows (no training pairs?)")
            continue
        df["season"] = season
        frames.append(df[["player_id", "season", "position", "cold_start", "step8_pace"]])
        print(f"  {season}: {len(df)} players, {int(df.cold_start.sum())} cold-start, "
              f"median pace {df.step8_pace.median():.2f}")
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(["player_id", "season"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate (player, season) rows -- refusing to write")
    out["built_at"] = datetime.now().isoformat(timespec="seconds")
    STEP8_PACE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STEP8_PACE_TABLE, index=False)
    print(f"wrote {STEP8_PACE_TABLE} ({len(out)} rows, seasons {out.season.min()}-{out.season.max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

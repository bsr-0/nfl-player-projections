#!/usr/bin/env python
"""Pre-flight leakage/validity gate for the 2015-2025 walk-forward
architecture comparison (GAPS.md, "information-matched preseason test").

Run this BEFORE trusting any head-to-head produced from these artifacts.
The comparison is intended to inform a production decision, so the
invariants it depends on are asserted mechanically rather than eyeballed.

Checks, in order of how badly a violation would mislead:

  1. Every Phase 7 preseason artifact has ZERO known-played weeks. A
     silently-inert `--preseason-mode` flag produced exactly this failure
     once already (GAPS.md, "A bug worth recording"): the run reported a
     plausible in-season MAE under a preseason label, and only the
     synthetic-week share revealed it.
  2. Each artifact covers all four positions and exactly its own season.
  3. Phase 7's training window ends at test_season - 1 and never contains
     the test season -- the requirement that training always uses the most
     recent PAST data, and only past data.

Usage:
    python scripts/verify_loyo_no_leakage.py
    python scripts/verify_loyo_no_leakage.py --dir data/experiments/phase7_preseason_loyo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.single_week_ppr.final_config import FINAL_CONFIG
from src.models.single_week_ppr.windows import window_to_season_list

POSITIONS = ("QB", "RB", "WR", "TE")
REAL_WEEK_COLS = (
    "weeks_real_stats",
    "weeks_inferred_snap_verified",
    "weeks_inferred_pbp_confirmed",
)


def _first_nfl_season() -> dict:
    import sqlite3
    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT player_id, MIN(season) FROM player_weekly_stats GROUP BY player_id"
        ).fetchall()
    finally:
        conn.close()
    return {r[0]: r[1] for r in rows}


def check_artifacts(paths: list[Path], require_rookies: bool = False) -> list[str]:
    failures = []
    debut = _first_nfl_season() if require_rookies else {}
    print(f"{'season':>8}{'rows':>7}{'known-played':>14}{'synth share':>13}{'rookies':>9}  positions")
    for path in sorted(paths):
        df = pd.read_csv(path)
        seasons = sorted(df["season"].unique())
        known = int(sum(df[c].sum() for c in REAL_WEEK_COLS if c in df.columns))
        share = df["weeks_synthetic"].sum() / df["weeks_predicted"].sum()
        present = tuple(sorted(df["position"].unique()))
        label = seasons[0] if len(seasons) == 1 else seasons
        n_rookie = (sum(1 for pid in df["player"] if debut.get(pid) == seasons[0])
                    if require_rookies and len(seasons) == 1 else -1)
        rk = str(n_rookie) if n_rookie >= 0 else "-"
        print(f"{str(label):>8}{len(df):>7}{known:>14}{share:>13.4f}{rk:>9}  {','.join(present)}")

        # A cold-start artifact that produced NO rookies is the inert-flag
        # failure mode: every other invariant still passes, the file is the
        # right shape, and the whole point of the run is silently absent.
        if require_rookies and n_rookie == 0:
            failures.append(
                f"{path.name}: ZERO rookies. --cold-start produced no "
                f"first-NFL-season players, so the run is preseason-valid but "
                f"pointless -- the population it exists to add is missing.")

        if known != 0:
            failures.append(
                f"{path.name}: {known} known-played weeks -- preseason_mode was "
                f"not in effect, so these are in-season numbers under a "
                f"preseason label"
            )
        if abs(share - 1.0) > 1e-9:
            failures.append(f"{path.name}: synthetic-week share {share:.4f}, expected exactly 1.0")
        if len(seasons) != 1:
            failures.append(f"{path.name}: expected one season, found {seasons}")
        missing = set(POSITIONS) - set(present)
        if missing:
            failures.append(f"{path.name}: missing positions {sorted(missing)} (partial/truncated run?)")
    return failures


def check_training_windows(test_seasons: list[int], available: list[int]) -> list[str]:
    failures = []
    print(f"\n{'test':>6}  Phase 7 training span per position (must end at test-1)")
    for ts in test_seasons:
        parts = []
        for pos in POSITIONS:
            train = window_to_season_list(FINAL_CONFIG[pos]["window"], ts, available)
            if not train:
                failures.append(f"test {ts} / {pos}: empty training window")
                continue
            parts.append(f"{pos}={min(train)}-{max(train)}")
            if max(train) != ts - 1:
                failures.append(
                    f"test {ts} / {pos}: training ends at {max(train)}, not {ts - 1} "
                    f"-- the most recent past season is being discarded")
            if ts in train:
                failures.append(f"test {ts} / {pos}: TEST SEASON PRESENT IN TRAINING")
        print(f"{ts:>6}  " + "  ".join(parts))
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data/experiments/phase7_preseason_loyo"))
    ap.add_argument("--extra", type=Path, nargs="*",
                    default=[Path("data/experiments/phase7_season_projection_2025_preseason.csv")],
                    help="Additional preseason artifacts to include (2025 was generated separately)")
    ap.add_argument("--require-rookies", action="store_true",
                    help="Also assert each artifact contains players whose FIRST "
                         "NFL season is that artifact's season (cold-start runs).")
    ap.add_argument("--min-season", type=int, default=2015)
    ap.add_argument("--max-season", type=int, default=2025)
    args = ap.parse_args()

    paths = sorted(args.dir.glob("*.csv")) + [p for p in args.extra if p.exists()]
    if not paths:
        print(f"No artifacts found in {args.dir}")
        return 1

    failures = check_artifacts(paths, require_rookies=args.require_rookies)
    failures += check_training_windows(
        list(range(args.min_season, args.max_season + 1)), list(range(2006, args.max_season + 1)),
    )

    found = set()
    for p in paths:
        found.update(int(s) for s in pd.read_csv(p, usecols=["season"])["season"].unique())
    expected = set(range(args.min_season, args.max_season + 1))
    if expected - found:
        failures.append(f"missing seasons: {sorted(expected - found)}")

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED -- {len(failures)} invariant violation(s). DO NOT use these artifacts:")
        for f in failures:
            print(f"  * {f}")
        return 1
    print(f"PASSED -- {len(paths)} artifacts, seasons {min(found)}-{max(found)}, "
          f"zero known-played weeks, training always ends at test-1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

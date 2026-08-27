#!/usr/bin/env python
"""CLI entrypoint for Phase 7 (next_focus.md) 18-week season projection.

Usage:
    python scripts/run_phase7_season_projection.py --positions WR --seasons 2024
    python scripts/run_phase7_season_projection.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.season_projection import run_season_projection
from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def main():
    parser = argparse.ArgumentParser(description="Phase 7 season projection")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/phase7_season_projection.csv"),
                         help="Output CSV path (appended incrementally)")
    parser.add_argument("--exclude-pbp-confirmed-zeros", action="store_true",
                         help="Sensitivity toggle: treat the weaker 2006-2017 "
                              "inferred_pbp_confirmed_zero tier as absent, falling back "
                              "to synthetic estimation for those weeks instead (default: off)")
    parser.add_argument(
        "--feature-rows-output", type=Path, default=None,
        help="Also dump the EXACT feature rows the model scored, to parquet. In "
             "cold-start preseason mode every scored row is synthetic and built "
             "per week, so a stratum derived from real historical rows describes "
             "a different population -- this is how to compare arms honestly.")
    parser.add_argument(
        "--preseason-mode", action="store_true",
        help="Score as a TRUE preseason (August) forecast, information-matched "
             "to the three arms in walk_forward_preseason.py: no week is treated "
             "as known-played, carry-forward is restricted to prior seasons, and "
             "opponents are graded on last season's defense. See "
             "run_season_projection's docstring for the full list of closed leaks.")
    parser.add_argument(
        "--cold-start", action="store_true",
        help="Also project players with NO prior NFL history (true rookies), "
             "who are otherwise dropped at row-construction time -- 0 of 95 in "
             "2025. Career-static features (draft capital, combine, college, "
             "age, destination-team context) populate; NFL-history features are "
             "left NaN for LightGBM's missing-aware splits. Requires "
             "--preseason-mode.")
    args = parser.parse_args()
    if args.cold_start and not args.preseason_mode:
        parser.error("--cold-start requires --preseason-mode: in-season runs "
                     "already project rookies off their real weekly rows.")

    result = run_season_projection(
        positions=args.positions, seasons=args.seasons, output_path=args.output,
        exclude_pbp_confirmed_zeros=args.exclude_pbp_confirmed_zeros,
        feature_row_output_path=args.feature_rows_output,
        preseason_mode=args.preseason_mode, cold_start=args.cold_start,
    )
    if result.empty:
        print("No results produced.")
        return

    result["abs_error"] = (result["predicted_season_total"] - result["actual_season_total"]).abs()
    result["bias"] = result["predicted_season_total"] - result["actual_season_total"]
    result["full_season"] = result["games_actually_played"] >= result["possible_weeks"]

    print("\n" + "=" * 70)
    print("Season-total MAE / bias by position")
    print("=" * 70)
    summary = result.groupby("position")[["abs_error", "bias"]].mean()
    summary["n"] = result.groupby("position").size()
    print(summary.round(2).to_string())

    print("\n" + "=" * 70)
    print("Full-season players vs. players who missed time")
    print("=" * 70)
    split = result.groupby(["position", "full_season"])["abs_error"].agg(["mean", "count"])
    print(split.round(2).to_string())

    print("\n" + "=" * 70)
    print("Reliance on synthetic (unknown-week) estimation vs. real/inferred data")
    print("=" * 70)
    result["synthetic_share"] = result["weeks_synthetic"] / result["possible_weeks"].replace(0, np.nan)
    result["any_synthetic"] = result["weeks_synthetic"] > 0
    reliance = result.groupby(["position", "any_synthetic"])["abs_error"].agg(["mean", "count"])
    print(reliance.round(2).to_string())
    print(f"\n  Mean synthetic-week share of possible_weeks: {result['synthetic_share'].mean():.3f}")


if __name__ == "__main__":
    main()

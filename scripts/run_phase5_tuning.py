#!/usr/bin/env python
"""CLI entrypoint for Phase 5 (next_focus.md) hyperparameter tuning.

Runs nested-CV hyperparameter tuning for each position's FINAL_CONFIG
architecture, then compares tuned vs. default (Phase 4) MAE/bias.

Usage:
    python scripts/run_phase5_tuning.py --positions WR --seasons 2024 --trials 20
    python scripts/run_phase5_tuning.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models.single_week_ppr.evaluate import (
    run_tuned_validation, compute_metrics, DEFAULT_VALIDATION_SEASONS,
    TUNED_OUTPUT_PATH, ROW_LEVEL_OUTPUT_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 hyperparameter tuning")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to tune (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--trials", type=int, default=100,
                         help="Optuna trials per (position, season) fold (default: 100)")
    parser.add_argument("--output", type=Path, default=TUNED_OUTPUT_PATH,
                         help="Tuned row-level output CSV path")
    args = parser.parse_args()

    result = run_tuned_validation(
        positions=args.positions, seasons=args.seasons, n_trials=args.trials, output_path=args.output,
    )
    if result.empty:
        print("No results produced.")
        return

    print("\n" + "=" * 70)
    print("Tuned MAE/bias by position")
    print("=" * 70)
    tuned_summary = result.groupby("position").apply(
        lambda g: pd.Series(compute_metrics(g["actual_ppr"], g["prediction"]))
    )
    print(tuned_summary[["mae", "bias", "n"]].round(3).to_string())

    if ROW_LEVEL_OUTPUT_PATH.exists():
        print("\n" + "=" * 70)
        print("Tuned vs. default (Phase 4) MAE comparison")
        print("=" * 70)
        default_df = pd.read_csv(ROW_LEVEL_OUTPUT_PATH)
        for position in result["position"].unique():
            tuned_mae = tuned_summary.loc[position, "mae"]
            arch = result[result["position"] == position]["model"].iloc[0].replace("_tuned", "")
            default_sub = default_df[(default_df["position"] == position) & (default_df["model"] == arch)
                                      & (default_df["season"].isin(args.seasons))]
            if default_sub.empty:
                continue
            default_metrics = compute_metrics(default_sub["actual_ppr"], default_sub["prediction"])
            delta = tuned_mae - default_metrics["mae"]
            print(f"  {position}: default MAE={default_metrics['mae']:.3f} -> tuned MAE={tuned_mae:.3f} "
                  f"({'improved' if delta < 0 else 'worse'} by {abs(delta):.3f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI entrypoint for Phase 4 (next_focus.md) single-week validation.

Runs each position's FINAL_CONFIG (chosen architecture/window/weighting
from Phases 2-3), saves row-level predictions, then reports breakdowns by
position, season, predicted-score bucket, player tier, and quantile
calibration.

Usage:
    python scripts/run_phase4_validation.py --positions WR --seasons 2024
    python scripts/run_phase4_validation.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.evaluate import run_final_validation, DEFAULT_VALIDATION_SEASONS, ROW_LEVEL_OUTPUT_PATH
from src.models.single_week_ppr.analysis import (
    summarize, compute_bucket_metrics, compute_tier_metrics, compute_calibration_report,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 single-week validation")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--output", type=Path, default=ROW_LEVEL_OUTPUT_PATH,
                         help="Row-level output CSV path (default: "
                              "data/experiments/phase4_row_level_predictions.csv). Appended incrementally.")
    args = parser.parse_args()

    result = run_final_validation(positions=args.positions, seasons=args.seasons, output_path=args.output)
    if result.empty:
        print("No results produced.")
        return

    import pandas as pd
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("\n" + "=" * 70)
    print("Overall MAE by model")
    print("=" * 70)
    report = summarize(result)
    print(report["overall"][["model", "mae", "rmse", "bias", "n"]].round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("MAE by position")
    print("=" * 70)
    print(report["by_position"].pivot(index="position", columns="model", values="mae").round(2).to_string())

    print("\n" + "=" * 70)
    print("MAE by predicted-score bucket")
    print("=" * 70)
    bucket = compute_bucket_metrics(result)
    print(bucket.pivot_table(index=["position", "bucket"], columns="model", values="mae").round(2).to_string())

    print("\n" + "=" * 70)
    print("MAE by player tier")
    print("=" * 70)
    tier = compute_tier_metrics(result)
    print(tier.pivot_table(index=["position", "tier"], columns="model", values="mae").round(2).to_string())

    print("\n" + "=" * 70)
    print("Quantile calibration (E_quantile_gbm): coverage should track nominal (p25->0.25, etc.)")
    print("=" * 70)
    calib = compute_calibration_report(result)
    print(calib.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

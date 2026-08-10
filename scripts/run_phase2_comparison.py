#!/usr/bin/env python
"""CLI entrypoint for Phase 2 (next_focus.md) single-week PPR architecture comparison.

Usage:
    python scripts/run_phase2_comparison.py --positions WR --seasons 2024
    python scripts/run_phase2_comparison.py --positions QB RB WR TE --seasons 2023 2024 2025
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.evaluate import run_comparison, DEFAULT_VALIDATION_SEASONS, COMPARISON_OUTPUT_PATH


def main():
    parser = argparse.ArgumentParser(description="Phase 2 single-week PPR architecture comparison")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--tune", action="store_true",
                         help="Enable hyperparameter tuning for the existing-methodology comparison model "
                              "(slower; default off, matching next_focus.md 'reasonable defaults' for Phase 2)")
    parser.add_argument("--output", type=Path, default=COMPARISON_OUTPUT_PATH,
                         help="Output CSV path (default: data/experiments/phase2_single_week_comparison.csv). "
                              "Useful for running positions in separate processes to bound runtime, then "
                              "merging the resulting CSVs.")
    args = parser.parse_args()

    result = run_comparison(
        positions=args.positions,
        seasons=args.seasons,
        tune_hyperparameters=args.tune,
        output_path=args.output,
    )

    if result.empty:
        print("No results produced.")
        return

    print("\n" + "=" * 70)
    print("MAE by position / model (lower is better)")
    print("=" * 70)
    summary = result.groupby(["position", "model"])["mae"].mean().unstack("model")
    print(summary.round(2).to_string())

    print("\n" + "=" * 70)
    print("Best architecture per position vs. best naive baseline")
    print("=" * 70)
    for position in result["position"].unique():
        pos_summary = summary.loc[position]
        baseline_cols = [c for c in pos_summary.index if c.startswith("baseline_")]
        model_cols = [c for c in pos_summary.index if not c.startswith("baseline_")]
        if not baseline_cols or not model_cols:
            continue
        best_baseline = pos_summary[baseline_cols].idxmin()
        best_model = pos_summary[model_cols].idxmin()
        print(f"  {position}: best baseline = {best_baseline} ({pos_summary[best_baseline]:.2f} MAE), "
              f"best model = {best_model} ({pos_summary[best_model]:.2f} MAE)")


if __name__ == "__main__":
    main()

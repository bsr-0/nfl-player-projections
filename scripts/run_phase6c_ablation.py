#!/usr/bin/env python
"""CLI entrypoint for Phase 6c (next_focus.md follow-up) feature-count ablation.

Usage:
    python scripts/run_phase6c_ablation.py --positions WR --seasons 2024
    python scripts/run_phase6c_ablation.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models.single_week_ppr.ablation import run_ablation, FEATURE_COUNT_CANDIDATES
from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def main():
    parser = argparse.ArgumentParser(description="Phase 6c feature-count ablation")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--feature-counts", nargs="+", default=list(FEATURE_COUNT_CANDIDATES),
                         help="Feature-count candidates, e.g. 10 20 30 all (default: 10 20 30 all)")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/phase6c_feature_ablation.csv"),
                         help="Output CSV path (appended incrementally)")
    args = parser.parse_args()

    feature_counts = [int(c) if c != "all" else "all" for c in args.feature_counts]

    result = run_ablation(
        positions=args.positions, seasons=args.seasons,
        feature_counts=feature_counts, output_path=args.output,
    )
    if result.empty:
        print("No results produced.")
        return

    print("\n" + "=" * 70)
    print("MAE by position / feature count (averaged across seasons)")
    print("=" * 70)
    summary = result.groupby(["position", "n_features_requested"])["mae"].mean().unstack("n_features_requested")
    # Order columns numerically then "all" last.
    cols = sorted([c for c in summary.columns if c != "all"], key=int) + (["all"] if "all" in summary.columns else [])
    print(summary[cols].round(3).to_string())


if __name__ == "__main__":
    main()

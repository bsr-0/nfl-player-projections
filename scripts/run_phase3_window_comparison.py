#!/usr/bin/env python
"""CLI entrypoint for Phase 3 (next_focus.md) training-window / recency-weighting comparison.

Chunk runs by position/window/season to stay under background-run timeouts —
each (position, window) pair reloads data from scratch, which is the
expensive axis (see Phase 3 plan). Results are appended to --output
incrementally as each fold/weighting/architecture completes, so a
killed/timed-out run keeps whatever finished.

Usage:
    python scripts/run_phase3_window_comparison.py --positions WR --windows 5y --seasons 2024
    python scripts/run_phase3_window_comparison.py --positions QB RB WR TE
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.evaluate import run_window_comparison, DEFAULT_VALIDATION_SEASONS, WINDOW_OUTPUT_PATH
from src.models.single_week_ppr.windows import WINDOW_CANDIDATES, WEIGHTING_SCHEMES


def main():
    parser = argparse.ArgumentParser(description="Phase 3 training-window / recency-weighting comparison")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--windows", nargs="+", default=list(WINDOW_CANDIDATES),
                         choices=list(WINDOW_CANDIDATES),
                         help="Training windows to test (default: every candidate)")
    parser.add_argument("--weightings", nargs="+", default=list(WEIGHTING_SCHEMES),
                         choices=list(WEIGHTING_SCHEMES),
                         help="Recency-weighting schemes to test (default: all 3)")
    parser.add_argument("--output", type=Path, default=WINDOW_OUTPUT_PATH,
                         help="Output CSV path (default: data/experiments/phase3_training_window_comparison.csv). "
                              "Appended to incrementally, not overwritten.")
    args = parser.parse_args()

    result = run_window_comparison(
        positions=args.positions,
        seasons=args.seasons,
        windows=args.windows,
        weightings=args.weightings,
        output_path=args.output,
    )

    if result.empty:
        print("No results produced.")
        return

    print("\n" + "=" * 70)
    print("MAE by position / window / weighting (architecture-averaged, lower is better)")
    print("=" * 70)
    model_rows = result[result["model"] != "existing_methodology"]
    summary = model_rows.groupby(["position", "window", "weighting"])["mae"].mean().unstack("weighting")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

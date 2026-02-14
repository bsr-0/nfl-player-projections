#!/usr/bin/env python3
"""
CLI runner for the leakage-free time-series backtester.

Usage:
    python scripts/run_ts_backtest.py                     # Backtest latest season, Ridge
    python scripts/run_ts_backtest.py --season 2024       # Backtest specific season
    python scripts/run_ts_backtest.py --model gbm         # Use GBM instead of Ridge
    python scripts/run_ts_backtest.py --positions QB RB    # Only backtest QB and RB
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ts_backtester import run_ts_backtest


def main():
    parser = argparse.ArgumentParser(
        description="Run leakage-free expanding-window time-series backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--season", "-s",
        type=int,
        default=None,
        help="Season to backtest (default: latest available)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=["ridge", "gbm"],
        default="ridge",
        help="Model type: ridge (fast) or gbm (higher fidelity)",
    )
    parser.add_argument(
        "--positions", "-p",
        nargs="+",
        default=None,
        help="Positions to backtest (default: QB RB WR TE)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NFL Time-Series Backtester (Expanding Window)")
    print("=" * 60)
    print(f"  Season: {args.season or 'auto (latest)'}")
    print(f"  Model: {args.model}")
    print(f"  Positions: {args.positions or 'all'}")
    print()

    pred_df, results = run_ts_backtest(
        season=args.season,
        model_type=args.model,
        positions=args.positions,
        verbose=not args.quiet,
    )

    print(f"\nDone. {len(pred_df)} predictions generated.")
    if results.get("metrics"):
        m = results["metrics"]
        print(f"  MAE:  {m.get('mae', 'N/A')}")
        print(f"  RMSE: {m.get('rmse', 'N/A')}")
        print(f"  R²:   {m.get('r2', 'N/A')}")


if __name__ == "__main__":
    main()

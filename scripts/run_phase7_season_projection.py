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
    args = parser.parse_args()

    result = run_season_projection(positions=args.positions, seasons=args.seasons, output_path=args.output)
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
    print("Naive baseline: assume full health (availability_rate=1.0) instead")
    print("=" * 70)
    naive_pred = (result["predicted_season_total"] / result["availability_rate"].replace(0, np.nan))
    naive_mae = (naive_pred - result["actual_season_total"]).abs().mean()
    print(f"  Naive (no P(plays) weighting) MAE: {naive_mae:.2f}")
    print(f"  With P(plays) weighting MAE:       {result['abs_error'].mean():.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI entrypoint for Phase 9 (next_focus.md) Monte Carlo season-total
quantiles (P25/P50/P75/P90).

Usage:
    python scripts/run_phase9_season_simulation.py --positions WR --seasons 2024
    python scripts/run_phase9_season_simulation.py --positions QB RB WR TE --n-sims 2000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.season_simulation import run_season_simulation, DEFAULT_N_SIMULATIONS
from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Monte Carlo season simulation")
    parser.add_argument("--positions", nargs="+", default=None,
                         help="Positions to evaluate (default: QB RB WR TE)")
    parser.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS),
                         help="Validation seasons (default: 2023 2024 2025)")
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMULATIONS,
                         help=f"Simulations per player (default: {DEFAULT_N_SIMULATIONS})")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/phase9_season_simulation.csv"),
                         help="Output CSV path (appended incrementally)")
    parser.add_argument(
        "--residual-csv", nargs="+", type=Path, default=None,
        help="Phase-4-style row-level CSVs to draw block-bootstrap residuals from "
             "(default: the corrected 2023-2025 file plus the 2020-2022 extension)",
    )
    args = parser.parse_args()

    result = run_season_simulation(
        positions=args.positions, seasons=args.seasons, n_sims=args.n_sims,
        residual_csv_paths=args.residual_csv, output_path=args.output,
    )
    if result.empty:
        print("No results produced.")
        return

    result["in_p25_p75"] = (
        (result["actual_season_total"] >= result["p25"])
        & (result["actual_season_total"] <= result["p75"])
    )
    result["p50_abs_error"] = (result["p50"] - result["actual_season_total"]).abs()

    print("\n" + "=" * 70)
    print("P50 MAE by position (compare against Phase 7's point-estimate MAE)")
    print("=" * 70)
    summary = result.groupby("position")["p50_abs_error"].mean()
    n = result.groupby("position").size()
    for pos in summary.index:
        print(f"  {pos}: P50 MAE={summary[pos]:.2f} (n={n[pos]})")

    print("\n" + "=" * 70)
    print("Calibration: P25-P75 coverage (nominal 50%)")
    print("=" * 70)
    coverage = result.groupby("position")["in_p25_p75"].mean()
    for pos in coverage.index:
        print(f"  {pos}: {coverage[pos]:.3f} (nominal 0.50)")

    print("\n" + "=" * 70)
    print("Interval width (P90 - P25) by position -- wider = more uncertain")
    print("=" * 70)
    result["interval_width"] = result["p90"] - result["p25"]
    width = result.groupby("position")["interval_width"].mean()
    for pos in width.index:
        print(f"  {pos}: {width[pos]:.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Walk-forward evaluate Plan A's PARTIAL fantasy-points reconstruction.

Standalone from scripts/train_team_share_model.py -- that script evaluates
share-prediction accuracy directly (acceptance criterion 1); this one wires
those predictions through renormalize_shares/reconstruct_volume/
reconstruct_partial_fantasy_points (src/models/team_allocation/reconstruct.py)
to evaluate the RECONSTRUCTED yardage-only partial fantasy-points MAE (the
revised acceptance criterion 2 -- see docs/TEAM_LEVEL_ALLOCATION_MODELS.md).
"YARDAGE-ONLY PARTIAL" because this covers rushing_yards/receiving_yards
points only, not touchdowns/receptions/passing -- see that module's
docstring for why.

Usage:
    python scripts/evaluate_team_share_reconstruction.py
    python scripts/evaluate_team_share_reconstruction.py --seasons 2013 2025
    python scripts/evaluate_team_share_reconstruction.py --n-test-seasons 3

Requires scripts/build_team_week_player_shares.py --write to have been run
first. This script does not fit/save any models of its own -- it re-fits
fresh models per walk-forward fold (same as scripts/train_team_share_model.py),
so it doesn't need to be run after that script, only after the shares table
exists.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODELS_DIR
from src.evaluation.team_reconstruction_backtester import run_reconstruction_backtest


def _print_report(report: dict) -> None:
    print(
        f"\nReconstructing partial fantasy points from: {report['targets_reconstructed']} "
        f"({report['n_rows_total']:,} player-weeks)\n"
    )
    print("Pooled across all walk-forward folds:")
    for arm, m in report["pooled"].items():
        print(f"    {arm:10s} n={m['n']:5d}  mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  r2={m['r2']:.3f}")

    print("\nBy position (pooled):")
    for arm, m in report["pooled"].items():
        for pos, seg in m.get("by_position", {}).items():
            print(f"    {arm:10s} {pos:4s} n={seg['n']:5d}  mae={seg['mae']:.3f}")

    baseline_mae = report["pooled"].get("rolling3", {}).get("mae")
    print("\nvs. rolling-3 reconstruction baseline (paired bootstrap CI on the MAE delta; "
          "negative = candidate has lower error -- see the revised acceptance criterion 2, "
          "this compares reconstructed-vs-reconstructed, not against the production model's "
          "full-PPR MAE):")
    for arm in ("ridge", "xgboost"):
        m = report["pooled"].get(arm, {})
        mae = m.get("mae")
        boot = m.get("vs_rolling3_bootstrap", {})
        if mae is None or baseline_mae is None or not boot:
            continue
        sig = "PASSES revised criterion 2 (95% CI excludes 0)" if boot.get("significant_improvement") else \
              "does NOT reliably beat the reconstruction baseline (CI includes 0 or is positive)"
        print(
            f"    {arm}: mae={mae:.3f} vs rolling3={baseline_mae:.3f}  "
            f"delta CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]  n_bootstrap={boot['n_bootstrap']}  "
            f"-> {sig}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"), default=None)
    ap.add_argument("--n-test-seasons", type=int, default=None)
    args = ap.parse_args()

    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None

    print("Running walk-forward reconstruction backtest...")
    report = run_reconstruction_backtest(seasons=seasons, n_test_seasons=args.n_test_seasons)
    _print_report(report)

    metadata = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seasons_requested": seasons,
        "reconstruction": report,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = MODELS_DIR / "team_share_reconstruction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nWrote reconstruction evaluation metadata to {metadata_path}")


if __name__ == "__main__":
    main()

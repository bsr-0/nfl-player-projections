#!/usr/bin/env python3
"""Evaluate player-level allocation shares for new full-PPR components."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.full_ppr_allocation_backtester import run_allocation_backtest
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", choices=PPR_RECONSTRUCTION_TARGETS, default=PPR_RECONSTRUCTION_TARGETS)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument(
        "--sparse-feature-families", nargs="+",
        choices=["player_event_role", "team_event_context"],
        help="Use a named lagged sparse-event family configuration (for ablation only).",
    )
    ap.add_argument("--output-dir", type=Path, default=Path("data/experiments/full_ppr_allocations"))
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target in args.targets:
        result = run_allocation_backtest(
            target, seasons, args.n_test_seasons,
            sparse_feature_families=args.sparse_feature_families,
        )
        result["predictions"].to_csv(args.output_dir / f"{target}_predictions.csv", index=False)
        result.pop("predictions")
        (args.output_dir / f"{target}_metrics.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        summary[target] = result["metrics"]
        print(target, json.dumps(result["metrics"], indent=2, default=float))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()

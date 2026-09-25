#!/usr/bin/env python3
"""Run leakage-safe, incremental sparse-event feature-family ablations.

The paired configurations intentionally differ by exactly one lagged family:
player event role alone, then player event role plus team event context.  It
does not blend a broad, untraceable feature bundle into Plan A.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.full_ppr_allocation_backtester import run_allocation_backtest
from src.models.team_allocation.hierarchical import SPARSE_EVENT_TARGETS


CONFIGURATIONS = {
    "player_event_role": {"player_event_role"},
    "player_plus_team_event_context": {"player_event_role", "team_event_context"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", choices=sorted(SPARSE_EVENT_TARGETS), default=sorted(SPARSE_EVENT_TARGETS))
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for target in args.targets:
        report[target] = {}
        for name, families in CONFIGURATIONS.items():
            result = run_allocation_backtest(
                target, seasons, args.n_test_seasons, sparse_feature_families=families
            )
            predictions = result.pop("predictions")
            arm = f"hierarchical_conditional_{'+'.join(sorted(families))}_event_mass"
            report[target][name] = {
                "families": sorted(families),
                "arm": arm,
                "metrics": result["metrics"].get(arm),
            }
            predictions.to_csv(args.output_dir / f"{target}_{name}_predictions.csv", index=False)
            (args.output_dir / f"{target}_{name}_metrics.json").write_text(
                json.dumps(result, indent=2, default=float) + "\n"
            )
            print(target, name, json.dumps(report[target][name]["metrics"], default=float))
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()

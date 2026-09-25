#!/usr/bin/env python3
"""Produce fold-local component-attribution for a joint Plan A report."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.joint_ppr_contributions import component_contribution_report
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS
from src.evaluation.team_total_backtester import run_team_total_backtest
from src.evaluation.ppr_truth import validate_inputs
from src.models.team_allocation.features import load_share_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation-dir", type=Path, required=True)
    ap.add_argument("--selector-report", type=Path, required=True)
    ap.add_argument(
        "--team-total-dir", type=Path,
        help="Read the exact team-total prediction CSVs used by the selector.",
    )
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.selector_report.read_text())
    allocation = {}
    for target in PPR_RECONSTRUCTION_TARGETS:
        path = args.allocation_dir / f"{target}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing allocation predictions: {path}")
        allocation[target] = {"predictions": pd.read_csv(path, float_precision="round_trip")}
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    if args.team_total_dir:
        totals = {
            target: {"predictions": pd.read_csv(args.team_total_dir / f"{target}_predictions.csv", float_precision="round_trip")}
            for target in PPR_RECONSTRUCTION_TARGETS
        }
    else:
        totals = {
            target: run_team_total_backtest(target, seasons, args.n_test_seasons)
            for target in PPR_RECONSTRUCTION_TARGETS
        }
    expected_hashes = report.get("input_sha256", {})
    if expected_hashes:
        if not args.team_total_dir:
            raise ValueError("selector used frozen team-total CSVs; provide --team-total-dir")
        for directory in (args.allocation_dir, args.team_total_dir):
            for target in PPR_RECONSTRUCTION_TARGETS:
                path = directory / f"{target}_predictions.csv"
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes.get(str(path)):
                    raise ValueError(f"prediction file changed since selection: {path}")
    truth, _ = validate_inputs(allocation, totals, load_share_rows(seasons=seasons))
    ordered_truth = truth.sort_values(["season", "week", "team", "player_id"])
    truth_hash = hashlib.sha256(
        pd.util.hash_pandas_object(ordered_truth[PPR_RECONSTRUCTION_TARGETS + ["player_id", "season", "week", "team", "position"]],
                                   index=False).to_numpy().tobytes()
    ).hexdigest()
    if truth_hash != report.get("raw_truth_sha256"):
        raise ValueError("raw PPR truth changed since selection")
    result = component_contribution_report(allocation, totals, report, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result["pooled"], indent=2, default=float))


if __name__ == "__main__":
    main()

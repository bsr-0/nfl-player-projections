#!/usr/bin/env python3
"""Run walk-forward arm selection on reconstructed full-PPR MAE."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.full_ppr_allocation_backtester import run_allocation_backtest
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS, select_walk_forward_arms, score_selector_report
from src.evaluation.team_total_backtester import run_team_total_backtest
from src.evaluation.ppr_truth import validate_inputs
from src.models.team_allocation.features import load_share_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument(
        "--allocation-dir", type=Path,
        help="Read already-generated allocation prediction CSVs instead of rerunning them.",
    )
    ap.add_argument(
        "--team-total-dir", type=Path,
        help="Write team-total prediction CSVs used by this selector for later attribution.",
    )
    ap.add_argument("--team-total-input-dir", type=Path,
                    help="Read saved team-total predictions without retraining.")
    ap.add_argument("--output", type=Path, default=Path("data/experiments/full_ppr_shrinkage_residual_count/joint_ppr_selection.json"))
    args = ap.parse_args()
    if args.team_total_input_dir and args.team_total_dir:
        ap.error("--team-total-input-dir and --team-total-dir are mutually exclusive")
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    if args.allocation_dir:
        allocation = {
            target: {"predictions": pd.read_csv(args.allocation_dir / f"{target}_predictions.csv", float_precision="round_trip")}
            for target in PPR_RECONSTRUCTION_TARGETS
        }
    else:
        allocation = {
            target: run_allocation_backtest(target, seasons, args.n_test_seasons)
            for target in PPR_RECONSTRUCTION_TARGETS
        }
    if args.team_total_input_dir:
        totals = {
            target: {"predictions": pd.read_csv(args.team_total_input_dir / f"{target}_predictions.csv", float_precision="round_trip")}
            for target in PPR_RECONSTRUCTION_TARGETS
        }
    else:
        totals = {target: run_team_total_backtest(target, seasons, args.n_test_seasons) for target in PPR_RECONSTRUCTION_TARGETS}
    if args.team_total_dir:
        args.team_total_dir.mkdir(parents=True, exist_ok=True)
        for target, result in totals.items():
            result["predictions"].to_csv(args.team_total_dir / f"{target}_predictions.csv", index=False)
    raw = load_share_rows(seasons=seasons)
    truth, audit = validate_inputs(allocation, totals, raw)
    report = select_walk_forward_arms(allocation, totals, truth=truth)
    scored_rows, comparison = score_selector_report(allocation, totals, report, truth)
    report["comparison"] = comparison
    report["scoring_definition"] = "raw_eight_component_ppr_excludes_fumbles_and_two_point_conversions"
    report["input_audit"] = audit
    ordered_truth = truth.sort_values(["season", "week", "team", "player_id"])
    report["raw_truth_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(ordered_truth[PPR_RECONSTRUCTION_TARGETS + ["player_id", "season", "week", "team", "position"]],
                                   index=False).to_numpy().tobytes()
    ).hexdigest()
    if args.allocation_dir and args.team_total_input_dir:
        report["input_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (args.allocation_dir, args.team_total_input_dir)
            for path in sorted(directory.glob("*_predictions.csv"))
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored_rows.to_csv(args.output.with_name("ppr_oof_rows.csv"), index=False)
    args.output.with_name("truth_audit.json").write_text(json.dumps({
        "scoring_definition": report["scoring_definition"],
        "raw_truth_sha256": report["raw_truth_sha256"],
        "input_sha256": report.get("input_sha256", {}),
        "input_audit": audit,
    }, indent=2, default=float) + "\n")
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({"comparison": comparison, "input_audit": audit}, indent=2, default=float))


if __name__ == "__main__":
    main()

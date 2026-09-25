#!/usr/bin/env python3
"""Evaluate full-PPR reconstruction from strict component/team-total folds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.full_ppr_allocation_backtester import run_allocation_backtest
from src.evaluation.full_ppr_reconstruction import evaluate_reconstruction
from src.evaluation.team_total_backtester import run_team_total_backtest
from src.models.team_allocation.features import load_share_rows
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS


SPARSE = {"receiving_tds", "rushing_tds", "passing_tds", "interceptions"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("data/experiments/full_ppr_hierarchical_allocations/reconstruction.json"))
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    allocation = {target: run_allocation_backtest(target, seasons, args.n_test_seasons) for target in PPR_RECONSTRUCTION_TARGETS}
    totals = {target: run_team_total_backtest(target, seasons, args.n_test_seasons) for target in PPR_RECONSTRUCTION_TARGETS}
    arms = {
        target: "hierarchical_conditional_event_mass" if target in SPARSE else "xgb_blend_renorm"
        for target in PPR_RECONSTRUCTION_TARGETS
    }
    report = evaluate_reconstruction(allocation, totals, arms=arms, raw_rows=load_share_rows(seasons=seasons))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()

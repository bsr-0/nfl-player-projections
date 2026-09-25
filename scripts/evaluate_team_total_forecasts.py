#!/usr/bin/env python3
"""Evaluate full-PPR component team-total forecasts without leakage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.team_total_backtester import run_team_total_backtest
from src.models.team_allocation.features import FULL_PPR_VOLUME_COLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", choices=FULL_PPR_VOLUME_COLS, default=FULL_PPR_VOLUME_COLS)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2006, 2025])
    ap.add_argument("--n-test-seasons", type=int, default=3)
    ap.add_argument("--output", type=Path, default=Path("data/experiments/full_ppr_team_total_backtests.json"))
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))
    reports = {target: run_team_total_backtest(target, seasons, args.n_test_seasons) for target in args.targets}
    # Row-level predictions are useful to reconstruction evaluators but are
    # intentionally kept out of this compact JSON summary.
    payload_reports = {
        target: {k: v for k, v in report.items() if k != "predictions"}
        for target, report in reports.items()
    }
    payload = {"seasons": seasons, "targets": args.targets, "reports": payload_reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    for target, report in reports.items():
        print(target, json.dumps(report["pooled"], indent=2, default=float))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

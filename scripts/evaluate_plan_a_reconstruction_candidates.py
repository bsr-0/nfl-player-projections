#!/usr/bin/env python3
"""Evaluate reconstruction-aware Plan A candidates against rolling-3.

Unlike the original evaluator, this includes volume-weighted regressors and
blend alphas selected on lagged-team-total-weighted yardage error.  The
walk-forward split remains strict and all arms are reconstructed on the same
rows and the same lagged team totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.team_reconstruction_candidates import run_reconstruction_candidate_backtest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("START", "END"))
    ap.add_argument("--n-test-seasons", type=int, default=None)
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="Exponent for lagged team-total training weights (default: 1.0)")
    ap.add_argument("--output", type=Path, default=Path("data/experiments/plan_a_reconstruction_candidates.json"))
    args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1] + 1)) if args.seasons else None
    report = run_reconstruction_candidate_backtest(seasons, args.n_test_seasons, args.weight_power)
    payload = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "seasons": seasons, "reconstruction": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(json.dumps(report["pooled"], indent=2, default=float))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

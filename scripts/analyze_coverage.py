#!/usr/bin/env python3
"""
Phase 4A — coverage analysis for conformal-interval CSVs.

Reads the output of ``scripts/add_conformal_intervals.py`` and
reports per-position + overall empirical coverage at 80 % and
95 %.  Also reports interval width statistics so we can see how
"useful" the intervals are (a 100-wide interval is technically
covering but actionably useless).

Usage:
    python scripts/analyze_coverage.py \\
        data/backtest_results/ts_backtest_2024_<ts>_predictions_conformal.csv \\
        [more.csv ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def _coverage_row(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0}
    in80 = ((sub["actual"] >= sub["lo80"]) & (sub["actual"] <= sub["hi80"])).mean()
    in95 = ((sub["actual"] >= sub["lo95"]) & (sub["actual"] <= sub["hi95"])).mean()
    width80 = (sub["hi80"] - sub["lo80"]).mean()
    width95 = (sub["hi95"] - sub["lo95"]).mean()
    return {
        "n": n,
        "cov80": float(in80),
        "cov95": float(in95),
        "mean_width80": float(width80),
        "mean_width95": float(width95),
        "median_actual": float(sub["actual"].median()),
    }


def analyze(path: Path) -> dict:
    df = pd.read_csv(path)
    df = df[(df.get("is_active", 1) == 1) & df["actual"].notna()].copy()
    summary = {"file": str(path), "season": int(df["season"].iloc[0])}
    summary["overall"] = _coverage_row(df)
    summary["per_position"] = {pos: _coverage_row(df[df["position"] == pos]) for pos in POSITIONS}
    return summary


def _print(summary: dict) -> None:
    print(f"=== {summary['file']} (season {summary['season']}) ===")
    o = summary["overall"]
    print(f"  overall: n={o['n']:>5} cov80={o['cov80']:.1%} (gate 76-84%) "
          f"cov95={o['cov95']:.1%} (gate 91-99%) "
          f"width80={o['mean_width80']:.2f} width95={o['mean_width95']:.2f}")
    for pos in POSITIONS:
        r = summary["per_position"][pos]
        if r["n"] == 0:
            continue
        print(f"  {pos}:      n={r['n']:>5} cov80={r['cov80']:.1%} cov95={r['cov95']:.1%} "
              f"width80={r['mean_width80']:.2f} width95={r['mean_width95']:.2f} "
              f"medFP={r['median_actual']:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of pretty-print")
    args = ap.parse_args()

    summaries = [analyze(p) for p in args.paths]
    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for s in summaries:
            _print(s)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

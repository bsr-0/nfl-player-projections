#!/usr/bin/env python3
"""
Analyze prediction std_ratio vs Pearson r per position from a walk-forward
backtest predictions CSV.

For an unbiased linear estimator with correlation r between predicted and
actual values, std(pred) / std(actual) = r exactly.  When std_ratio - r is
near zero, "variance compression" in the predictions is just correlation
arithmetic and not a fixable shrinkage problem.  When std_ratio - r is
materially positive, there is extra shrinkage on top of the correlation
floor that is potentially fixable (e.g. by tuning Ridge alpha).

The April 14 council verdict (council-transcript-20260414-034617.md, Step
3) requires running this analysis at multiple Ridge alpha values to
determine whether the QB/TE 7-11 point gap reported in
CRITICAL_LIMITATION.md is reproduced across folds with a tight standard
error (= authorize per-position alpha tuning) or dissolves (= the gap was
an artifact of a single alpha and no fix is warranted).

Usage:
    python scripts/analyze_std_ratio.py <predictions.csv>
    python scripts/analyze_std_ratio.py --json <predictions.csv>

Stdlib-only.  Does not require pandas/numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Numerics (stdlib only)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    """Population standard deviation (ddof=0), matching numpy default."""
    if len(xs) < 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation; returns 0.0 if either side has zero variance."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _summarize(pred: List[float], actual: List[float]) -> Dict[str, float]:
    """Return n, mean_pred, mean_actual, std_pred, std_actual, r, std_ratio,
    gap.  Gap = std_ratio - r; near zero means the variance compression is
    correlation arithmetic, not shrinkage."""
    n = len(pred)
    mp, ma = _mean(pred), _mean(actual)
    sp, sa = _std(pred), _std(actual)
    r = _pearson(pred, actual)
    ratio = sp / sa if sa > 0 else 0.0
    gap = ratio - r
    bias_pct = (mp - ma) / ma * 100 if ma > 0 else 0.0
    return {
        "n": n,
        "mean_pred": mp,
        "mean_actual": ma,
        "std_pred": sp,
        "std_actual": sa,
        "pearson_r": r,
        "std_ratio": ratio,
        "gap": gap,
        "bias_pct": bias_pct,
    }


def _se(xs: List[float]) -> float:
    """Standard error of the mean for a list of per-fold values."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    sample_var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(sample_var / n)


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------

def _load_predictions(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _group_by_position(rows: List[Dict[str, str]]) -> Dict[str, Tuple[List[float], List[float]]]:
    by_pos: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for row in rows:
        try:
            p = float(row["predicted"])
            a = float(row["actual"])
        except (ValueError, KeyError):
            continue
        pos = row.get("position", "?")
        by_pos[pos][0].append(p)
        by_pos[pos][1].append(a)
    return by_pos


def _group_by_position_and_week(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, Tuple[List[float], List[float]]]]:
    by_pos_week: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: ([], []))
    )
    for row in rows:
        try:
            p = float(row["predicted"])
            a = float(row["actual"])
        except (ValueError, KeyError):
            continue
        pos = row.get("position", "?")
        week = row.get("week", "?")
        by_pos_week[pos][week][0].append(p)
        by_pos_week[pos][week][1].append(a)
    return by_pos_week


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_table(report: Dict, *, source: Path) -> str:
    lines = []
    lines.append(f"std_ratio vs Pearson r analysis — {source.name}")
    lines.append("=" * 78)
    lines.append("")

    lines.append("Season-aggregate (single value per position):")
    lines.append(
        f"  {'pos':<4} {'n':>5} {'mean_p':>7} {'mean_a':>7} "
        f"{'std_p':>6} {'std_a':>6} {'r':>7} {'ratio':>7} {'gap':>7}"
    )
    for pos in sorted(report["per_position"]):
        s = report["per_position"][pos]
        lines.append(
            f"  {pos:<4} {s['n']:>5d} {s['mean_pred']:>7.3f} {s['mean_actual']:>7.3f} "
            f"{s['std_pred']:>6.3f} {s['std_actual']:>6.3f} {s['pearson_r']:>+7.3f} "
            f"{s['std_ratio']:>7.3f} {s['gap']:>+7.3f}"
        )
    lines.append("")
    lines.append("Per-week breakdown (gap = std_ratio - r per fold; SE across folds):")
    lines.append(
        f"  {'pos':<4} {'folds':>5} {'gap_mean':>9} {'gap_se':>8} "
        f"{'gap_min':>8} {'gap_max':>8} {'r_mean':>7} {'ratio_mean':>10}"
    )
    for pos in sorted(report["per_position_per_week"]):
        wstats = report["per_position_per_week"][pos]
        gaps = [w["gap"] for w in wstats.values()]
        rs = [w["pearson_r"] for w in wstats.values()]
        ratios = [w["std_ratio"] for w in wstats.values()]
        lines.append(
            f"  {pos:<4} {len(gaps):>5d} {_mean(gaps):>+9.3f} {_se(gaps):>8.3f} "
            f"{min(gaps):>+8.3f} {max(gaps):>+8.3f} {_mean(rs):>+7.3f} "
            f"{_mean(ratios):>10.3f}"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  gap ≈ 0  -> variance compression is correlation arithmetic; not fixable by less shrinkage.")
    lines.append("  gap > 0  -> extra shrinkage on top of the correlation floor; potentially fixable (e.g. lower Ridge alpha).")
    lines.append("  |gap_mean| > 2*gap_se  -> stable across folds; supports authorizing a per-position fix.")
    return "\n".join(lines)


def _build_report(rows: List[Dict[str, str]]) -> Dict:
    by_pos = _group_by_position(rows)
    per_position = {pos: _summarize(p, a) for pos, (p, a) in by_pos.items()}

    by_pos_week = _group_by_position_and_week(rows)
    per_position_per_week = {
        pos: {week: _summarize(p, a) for week, (p, a) in weeks.items()}
        for pos, weeks in by_pos_week.items()
    }
    return {
        "per_position": per_position,
        "per_position_per_week": per_position_per_week,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-position std_ratio vs Pearson r from a walk-forward predictions CSV.",
    )
    parser.add_argument("csv", type=Path, help="Path to a *_predictions.csv file")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted table.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"error: {args.csv} not found", file=sys.stderr)
        return 1

    rows = _load_predictions(args.csv)
    report = _build_report(rows)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_table(report, source=args.csv))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Bootstrap the 43-week cross-season H2H record (Step 1 of the 2026-04-22
re-council verdict — ``council-transcript-20260422-032550.md``).

Re-council kill criterion: if the 5th-percentile bootstrap win rate is
**below 52.4 %** (the -110 cash H2H breakeven for a 1.8× payout), the 43-week
sample is promising but not yet a deployable edge — pause before Phase 4
ensemble work and extend the validation window.

Reads the two production-config walk-forward runs (α=10 000, post-Vegas,
post-injury; opp-defense s2d reverted) and resamples their ``won_vs_hindsight``
flags with replacement 10 000 times.  Stdlib-only on purpose: this script
must be runnable in any environment that has the repo checked out, even
one without pandas/numpy.

Usage:
    python scripts/bootstrap_h2h_record.py
    python scripts/bootstrap_h2h_record.py --trials 50000 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The production config as of 2026-04-22: α=10 000 Ridge, Vegas+injury
# merges firing, opp-defense s2d reverted.  These two JSONs carry the
# 30-13 cross-season record the re-council was based on.
DEFAULT_SOURCES: Tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2024_20260422_025527.json",
    PROJECT_ROOT / "data" / "backtest_results" / "ts_backtest_2025_20260422_024024.json",
)

# -110 American odds = risk 110 to win 100.  Break-even = 110/(110+100) ≈ 0.5238.
BREAKEVEN_WIN_RATE = 110.0 / 210.0


def load_weekly_flags(paths: List[Path]) -> List[int]:
    """Extract the 0/1 ``won_vs_hindsight`` flag per week from each run."""
    flags: List[int] = []
    for p in paths:
        with p.open() as f:
            data = json.load(f)
        dq = data.get("decision_quality") or {}
        weekly = dq.get("weekly_results") or []
        if not weekly:
            raise RuntimeError(f"No decision_quality.weekly_results in {p}")
        for w in weekly:
            flags.append(1 if w.get("won_vs_hindsight") else 0)
    return flags


def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion — doesn't collapse at
    extremes the way the normal-approx does and is the advisor's reference
    number (54.9 % – 81.4 % for 30/43)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bootstrap(flags: List[int], trials: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    n = len(flags)
    out: List[float] = []
    for _ in range(trials):
        resample = [flags[rng.randrange(n)] for _ in range(n)]
        out.append(sum(resample) / n)
    return out


def percentile(values: List[float], q: float) -> float:
    """Linear-interp percentile on a list of floats; avoids statistics.quantiles
    Python-3.8+ API variance."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10_000, help="Number of bootstrap resamples (default 10000).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    ap.add_argument("--sources", nargs="+", type=Path, default=list(DEFAULT_SOURCES),
                    help="JSON run files to pool (default: 2024+2025 α=10000 post-Vegas+injury).")
    ap.add_argument("--breakeven", type=float, default=BREAKEVEN_WIN_RATE,
                    help=f"Break-even win rate to test against (default {BREAKEVEN_WIN_RATE:.4f} = -110).")
    args = ap.parse_args()

    flags = load_weekly_flags(args.sources)
    wins = sum(flags)
    n = len(flags)
    observed = wins / n

    print(f"Sources:")
    for p in args.sources:
        print(f"  {p.relative_to(PROJECT_ROOT)}")
    print()
    print(f"Observed record:          {wins}-{n - wins}  ({observed * 100:.2f}%)")

    lo, hi = wilson_interval(wins, n, 0.95)
    print(f"Wilson 95% CI:            [{lo * 100:.2f}%, {hi * 100:.2f}%]")
    print(f"Break-even (H2H -110):    {args.breakeven * 100:.2f}%")
    print()

    draws = bootstrap(flags, args.trials, args.seed)
    p5 = percentile(draws, 0.05)
    p50 = percentile(draws, 0.50)
    p95 = percentile(draws, 0.95)
    mean = sum(draws) / len(draws)
    frac_above_be = sum(1 for x in draws if x > args.breakeven) / len(draws)

    print(f"Bootstrap ({args.trials:,} resamples, seed={args.seed}):")
    print(f"  mean:                   {mean * 100:.2f}%")
    print(f"  5th percentile:         {p5 * 100:.2f}%  <-- kill criterion gate")
    print(f"  median:                 {p50 * 100:.2f}%")
    print(f"  95th percentile:        {p95 * 100:.2f}%")
    print(f"  P(resample > {args.breakeven * 100:.1f}%): {frac_above_be * 100:.2f}%")
    print()

    verdict = "PASS — sample clears -110 break-even" if p5 >= args.breakeven else \
              "FAIL — 43-week sample is not yet a deployable edge"
    print(f"Verdict: {verdict}")
    return 0 if p5 >= args.breakeven else 1


if __name__ == "__main__":
    sys.exit(main())

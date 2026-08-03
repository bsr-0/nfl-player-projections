#!/usr/bin/env python3
"""
Continuous-margin significance audit — Phase #1 from the
"how do we tighten the CI" plan (2026-04-25).

The wide_symmetric_replay reports a binary per-week WR (74.7% =
62-21 across 4 seasons).  Binary outcomes throw away ~80% of the
per-week signal — the per-week point margin (model lineup actual
minus opponent lineup actual) is a continuous statistic with much
more information per observation.

This script:
1. Reuses the wide_symmetric_replay logic to derive per-week
   (model_sym, opp_sym) lineup totals on the same 4-season run-set.
2. Computes margin = model_sym - opp_sym per week.
3. Reports mean margin, SD, SE, t-based 95% CI on the mean, and a
   block-bootstrap CI on the mean.  Compares CI widths to the
   binary-WR Wilson CI for the same data.
4. Translates "mean margin > 0" into a significance test (paired
   t-test + sign test as a robustness check).

Usage:
    python scripts/analyze_continuous_margin.py --runs <json> [<json> ...]
    python scripts/analyze_continuous_margin.py \\
        --runs data/backtest_results/ts_backtest_2022_..._032705.json \\
               data/backtest_results/ts_backtest_2023_..._032744.json \\
               data/backtest_results/ts_backtest_2024_..._153723.json \\
               data/backtest_results/ts_backtest_2025_..._153712.json \\
        --apply-active-filter
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the symmetric-replay primitives.
from scripts.wide_symmetric_replay import (
    DB_PATH,
    load_wide_predictions,
    symmetric_model_actual,
    build_prospective_opponent_pool,
    opponent_prospective_actual,
    load_active_roster_ids,
    wilson_interval,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--apply-active-filter", action="store_true", default=True)
    ap.add_argument("--bootstrap-trials", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    margins: List[float] = []
    weeks_meta: List[Dict] = []

    for run_path in args.runs:
        season, rows = load_wide_predictions(run_path)
        weeks = sorted({r["week"] for r in rows})
        for week in weeks:
            active_ids = None
            if args.apply_active_filter:
                active_ids = load_active_roster_ids(conn, season, week)
                if not active_ids:
                    continue
            model_sym, _ = symmetric_model_actual(rows, week, active_ids=active_ids)
            opp_pool = build_prospective_opponent_pool(conn, season, week)
            opp_sym, _ = opponent_prospective_actual(opp_pool, active_ids=active_ids)
            if not opp_pool or (model_sym == 0.0):
                continue
            margin = model_sym - opp_sym
            margins.append(margin)
            weeks_meta.append({
                "season": season, "week": week,
                "model": round(model_sym, 2),
                "opp": round(opp_sym, 2),
                "margin": round(margin, 2),
            })

    n = len(margins)
    if n == 0:
        print("No weeks scored.", file=sys.stderr)
        return 1

    mean = sum(margins) / n
    var = sum((m - mean) ** 2 for m in margins) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    # t critical for 95% with df = n-1.  For n=83 → df=82 → t ≈ 1.989.
    # Use 1.96 (large-n normal approximation) — safe for n>=30.
    t_crit = 1.96
    ci_lo = mean - t_crit * se
    ci_hi = mean + t_crit * se

    # Block bootstrap: resample seasons (4 unique seasons, but we
    # also do a per-week i.i.d. bootstrap and a 4-week-block
    # bootstrap for comparison).
    random.seed(args.seed)

    def boot_iid(trials: int) -> List[float]:
        ms = []
        for _ in range(trials):
            sample = random.choices(margins, k=n)
            ms.append(sum(sample) / n)
        return ms

    def boot_season_block(trials: int) -> List[float]:
        season_groups: Dict[int, List[float]] = {}
        for m, meta in zip(margins, weeks_meta):
            season_groups.setdefault(meta["season"], []).append(m)
        seasons = list(season_groups.keys())
        ms = []
        for _ in range(trials):
            chosen = random.choices(seasons, k=len(seasons))
            sample: List[float] = []
            for s in chosen:
                sample.extend(season_groups[s])
            ms.append(sum(sample) / len(sample))
        return ms

    def boot_4week_block(trials: int) -> List[float]:
        blocks: List[List[float]] = []
        # Group margins into 4-week blocks within each season,
        # preserving (season, week) order.
        by_season: Dict[int, List[Tuple[int, float]]] = {}
        for m, meta in zip(margins, weeks_meta):
            by_season.setdefault(meta["season"], []).append((meta["week"], m))
        for season, lst in by_season.items():
            lst.sort()
            for i in range(0, len(lst), 4):
                blocks.append([m for _, m in lst[i:i + 4]])
        n_blocks = len(blocks)
        ms = []
        for _ in range(trials):
            chosen = random.choices(blocks, k=n_blocks)
            sample = [m for blk in chosen for m in blk]
            ms.append(sum(sample) / len(sample))
        return ms

    iid_ms = boot_iid(args.bootstrap_trials)
    season_ms = boot_season_block(args.bootstrap_trials)
    block4_ms = boot_4week_block(args.bootstrap_trials)

    def percentile(vals: List[float], q: float) -> float:
        s = sorted(vals)
        idx = int(q / 100.0 * (len(s) - 1))
        return s[idx]

    # Binary WR comparison
    wins = sum(1 for m in margins if m > 0)
    wr = wins / n
    wilson_lo, wilson_hi = wilson_interval(wins, n)

    # Sign test (binomial vs 0.5)
    from math import comb
    p_sign = sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)

    # Paired t-test on margin > 0 (one-sided)
    if sd > 0:
        t_stat = mean / se
        # df = n-1, but for large n we can use normal approx.
        # P(T > t) ≈ 1 - Φ(t)
        from math import erf
        p_t = 0.5 * (1 - erf(t_stat / math.sqrt(2)))
    else:
        t_stat = float("inf"); p_t = 0.0

    # ROI (assumes paying -110 vig per matchup; payout 1.8x as
    # in DECISION_QUALITY).
    payout = 1.8
    roi_per_week = [payout - 1.0 if m > 0 else -1.0 for m in margins]
    roi_mean = sum(roi_per_week) / n
    roi_sd = math.sqrt(sum((r - roi_mean) ** 2 for r in roi_per_week) / (n - 1))
    roi_se = roi_sd / math.sqrt(n)

    print(f"Continuous-margin audit ({len(args.runs)} seasons, n={n} weeks)\n")
    print(f"Per-week margin (model_lineup_actual - opponent_lineup_actual):")
    print(f"  mean:          {mean:+8.2f} fantasy points / week")
    print(f"  median:        {sorted(margins)[n // 2]:+8.2f}")
    print(f"  SD:            {sd:8.2f}")
    print(f"  SE of mean:    {se:8.2f}")
    print(f"  95% CI (norm): [{ci_lo:+.2f}, {ci_hi:+.2f}]  width={ci_hi - ci_lo:.2f}")
    print()
    print(f"Bootstrap CIs on mean margin (n={args.bootstrap_trials} resamples):")
    for label, ms in (("IID per-week", iid_ms),
                      ("4-week blocks", block4_ms),
                      ("Whole-season blocks", season_ms)):
        lo = percentile(ms, 2.5); hi = percentile(ms, 97.5)
        p5 = percentile(ms, 5); p95 = percentile(ms, 95)
        print(f"  {label:<22} 95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"5th-95th [{p5:+.2f}, {p95:+.2f}]  width₉₅={hi - lo:.2f}")
    print()
    print(f"Significance vs zero margin (model better than opponent on average):")
    print(f"  Paired t-test:   t={t_stat:.3f}, one-sided p ≈ {p_t:.2e}")
    print(f"  Sign test:       p ≈ {p_sign:.2e}")
    print()
    print(f"Per-week binary win rate (the OLD reporting):")
    print(f"  Record: {wins}-{n - wins} = {wr*100:.2f}%")
    print(f"  Wilson 95% CI: [{wilson_lo*100:.2f}%, {wilson_hi*100:.2f}%]  width={ (wilson_hi - wilson_lo)*100:.2f}pp")
    print()
    print(f"ROI at -110 vig (payout 1.8x):")
    print(f"  mean ROI:   {roi_mean*100:+6.2f}%  per matchup")
    print(f"  SE:         {roi_se*100:6.2f}%")
    print(f"  95% CI:     [{(roi_mean - 1.96*roi_se)*100:+6.2f}%, {(roi_mean + 1.96*roi_se)*100:+6.2f}%]")
    print()

    # Width comparison: how much more precise is the continuous
    # estimator?  Express both in standard "WR-equivalent" terms by
    # noting the fraction of CI half-width relative to the mean.
    rel_width_continuous = (ci_hi - ci_lo) / max(1e-9, abs(mean))
    rel_width_wr = (wilson_hi - wilson_lo) / max(1e-9, abs(wr - 0.5))
    print(f"Relative precision (CI width / |effect|):")
    print(f"  continuous margin:  {rel_width_continuous:.3f}")
    print(f"  binary WR vs 0.5:   {rel_width_wr:.3f}")
    print(f"  precision gain:     {rel_width_wr / rel_width_continuous:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())

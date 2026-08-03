#!/usr/bin/env python3
"""
Phase 4A — add conformal prediction intervals to a walk-forward
predictions CSV.

Conformal idea (sequential / online split-conformal): at the time
we'd be predicting week W, we have access to the model's residuals
for weeks 1..W-1 of the current season (and optionally the prior
season's tail).  Take the last K residuals per position, compute
empirical quantiles, and form the interval as
``prediction ± quantile(|residual|, level)``.  The intervals are
distribution-free; no Gaussian assumption.

Crucially this is causal — week W's interval is computed from only
strictly-prior weeks' residuals.  A naive global quantile would
peek at the future.

Usage:
    python scripts/add_conformal_intervals.py \\
        --predictions data/backtest_results/ts_backtest_2024_<ts>_predictions.csv

    # With prior-season CSV to seed the W1/W2 pool:
    python scripts/add_conformal_intervals.py \\
        --predictions data/backtest_results/ts_backtest_2025_<ts>_predictions.csv \\
        --prior-season data/backtest_results/ts_backtest_2024_<ts>_predictions.csv

Output: writes ``<input_basename>_conformal.csv`` next to the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Window of prior weeks used to build the residual pool per position.
DEFAULT_WINDOW_WEEKS = 8
LEVELS = [0.80, 0.95]
POSITIONS = ["QB", "RB", "WR", "TE"]

# Fallback half-widths in PPR fantasy points if we have zero residuals
# yet (truly week 1 with no prior-season pool).  Picked to match
# rough Phase 4 ceiling-audit per-position MAE (~5-6 PPG); the 80 %
# half-width should be larger than MAE since |residual| is heavier-
# tailed than its mean.
FALLBACK_HALFWIDTHS: Dict[str, Dict[float, float]] = {
    "QB": {0.80: 8.0, 0.95: 14.0},
    "RB": {0.80: 7.0, 0.95: 12.0},
    "WR": {0.80: 6.5, 0.95: 11.0},
    "TE": {0.80: 5.5, 0.95: 9.5},
}


def _quantile_or_fallback(
    residuals: np.ndarray, level: float, fallback: float
) -> float:
    """Return the level-th empirical quantile of |residuals|; if pool
    is empty or too small, fall back to the configured constant."""
    if residuals.size == 0:
        return fallback
    if residuals.size < 4:
        # Too few obs — empirical quantile is unstable; blend toward fallback.
        emp = float(np.quantile(np.abs(residuals), level))
        return 0.5 * emp + 0.5 * fallback
    return float(np.quantile(np.abs(residuals), level))


def add_intervals(
    predictions: pd.DataFrame,
    window_weeks: int = DEFAULT_WINDOW_WEEKS,
    seed_residuals: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Walk through the predictions in temporal order; for each
    (season, week) emit per-row interval bounds based on the prior
    ``window_weeks`` of in-season residuals (per position).  Optional
    ``seed_residuals`` provides a per-position numpy array carried in
    from the prior season's tail."""
    df = predictions.sort_values(["season", "week", "player_id"]).reset_index(drop=True).copy()
    # Pool seeded with prior-season tail if provided.
    pools: Dict[str, List[float]] = {p: [] for p in POSITIONS}
    if seed_residuals is not None:
        for pos, vals in seed_residuals.items():
            if pos in pools:
                pools[pos].extend(list(vals))

    # Buffer rows of the most recent window_weeks for each position
    # so we can prune the pool by week.
    week_residuals: List[Dict[str, List[float]]] = []  # one dict per processed week
    out_lo80, out_hi80, out_lo95, out_hi95 = [], [], [], []

    for (season, week), group in df.groupby(["season", "week"], sort=True):
        # Compute current pool quantiles BEFORE incorporating this week's residuals.
        for pos in POSITIONS:
            arr = np.array(pools[pos], dtype=float) if pools[pos] else np.empty(0)
            fb = FALLBACK_HALFWIDTHS.get(pos, FALLBACK_HALFWIDTHS["WR"])
            for lvl in LEVELS:
                _ = _quantile_or_fallback(arr, lvl, fb[lvl])  # warm cache (no-op now)

        for _, row in group.iterrows():
            pos = row["position"]
            pred = row["predicted"]
            arr = np.array(pools.get(pos, []), dtype=float) if pools.get(pos) else np.empty(0)
            fb = FALLBACK_HALFWIDTHS.get(pos, FALLBACK_HALFWIDTHS["WR"])
            hw80 = _quantile_or_fallback(arr, 0.80, fb[0.80])
            hw95 = _quantile_or_fallback(arr, 0.95, fb[0.95])
            out_lo80.append(pred - hw80)
            out_hi80.append(pred + hw80)
            out_lo95.append(pred - hw95)
            out_hi95.append(pred + hw95)

        # After emitting THIS week's intervals, fold this week's
        # residuals into the pool — only for active rows with a
        # finite actual.
        active = group[(group.get("is_active", 1) == 1) & group["actual"].notna()]
        this_week_residuals: Dict[str, List[float]] = {p: [] for p in POSITIONS}
        for _, row in active.iterrows():
            pos = row["position"]
            if pos not in pools:
                continue
            res = float(row["actual"] - row["predicted"])
            pools[pos].append(res)
            this_week_residuals[pos].append(res)
        week_residuals.append(this_week_residuals)
        # Prune the pool to the last ``window_weeks`` of residuals
        # per position.  We track per-week additions so the prune is
        # exact rather than approximate.
        if len(week_residuals) > window_weeks:
            old = week_residuals.pop(0)
            for pos, vals in old.items():
                if pools[pos] and vals:
                    # Drop the same number of leading entries.  Order
                    # is preserved because we appended in week order.
                    pools[pos] = pools[pos][len(vals):]

    df["lo80"] = out_lo80
    df["hi80"] = out_hi80
    df["lo95"] = out_lo95
    df["hi95"] = out_hi95
    return df


def _seed_from_prior(prior_csv: Path, window_weeks: int) -> Dict[str, np.ndarray]:
    """Return per-position residual arrays from the LAST window_weeks
    of a prior-season predictions CSV."""
    df = pd.read_csv(prior_csv)
    df = df[(df.get("is_active", 1) == 1) & df["actual"].notna()].copy()
    if df.empty:
        return {}
    last_weeks = sorted(df["week"].unique())[-window_weeks:]
    df = df[df["week"].isin(last_weeks)]
    df["residual"] = df["actual"] - df["predicted"]
    return {
        pos: df[df["position"] == pos]["residual"].to_numpy()
        for pos in POSITIONS
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--prior-season", type=Path, default=None,
                    help="Predictions CSV from the prior season; its last "
                         f"{DEFAULT_WINDOW_WEEKS} weeks of residuals seed "
                         "the W1/W2 pool of the target season.")
    ap.add_argument("--window-weeks", type=int, default=DEFAULT_WINDOW_WEEKS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    print(f"Read {len(df):,} predictions from {args.predictions}.")

    seed = None
    if args.prior_season:
        seed = _seed_from_prior(args.prior_season, args.window_weeks)
        sizes = {k: len(v) for k, v in seed.items()}
        print(f"Seeding pool from {args.prior_season} last {args.window_weeks} weeks: {sizes}")

    out = add_intervals(df, window_weeks=args.window_weeks, seed_residuals=seed)
    out_path = args.out or args.predictions.with_name(
        args.predictions.stem + "_conformal.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    # Quick coverage check
    active = out[(out.get("is_active", 1) == 1) & out["actual"].notna()]
    if len(active) > 0:
        c80 = ((active["actual"] >= active["lo80"]) & (active["actual"] <= active["hi80"])).mean()
        c95 = ((active["actual"] >= active["lo95"]) & (active["actual"] <= active["hi95"])).mean()
        print(f"Quick coverage: 80% interval = {c80:.1%}, 95% interval = {c95:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

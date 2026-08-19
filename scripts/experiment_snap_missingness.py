#!/usr/bin/env python
"""Pre-registered A/B: how should Ridge represent unknown historical snaps?

    A (control)  snap_share_pct unknown -> 0, no missingness indicator.
                 Exactly today's production path, untouched.
    B (treatment) snap_share_pct unknown -> NaN, then the rolling feature is
                 imputed with the position x era median computed INSIDE each
                 training fold, and snap_share_pct_roll3_known joins the
                 feature list.

Only A and B. Global-median C is deliberately not tested unless B materially
beats A -- if representing missingness doesn't help, optimising the
imputation constant won't either.

SEASON CHOICE IS THE POINT. Genuine partial-missingness (history exists but
part of it is unknown) by season:

    2014-2017   371-383 rows/season   ~7.0%
    2018        276                    5.2%
    2019+        12-73                 0.2-1.3%
    2024         17                    0.29%   <- no power at all

A 2024-only backtest cannot answer the question. Test seasons are therefore
2016/2017 (pre-2018 regime, ~750 affected rows) and 2018/2024 (the 2018+
regression check).

Leakage: every imputation statistic is fitted on the training fold alone --
see apply_snap_imputation() and tests/test_snap_imputation_leakage.py, whose
checks fail if test rows are allowed to influence the fill value.

Usage:
    python scripts/experiment_snap_missingness.py            # A and B
    python scripts/experiment_snap_missingness.py --variants A
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

SEASONS = [2016, 2017, 2018, 2024]
POSITIONS = ["RB", "WR", "TE"]
OUT_DIR = Path("data/experiments/snap_missingness")


def configure(variant: str) -> None:
    from src.features import utilization_score
    from src.evaluation import ts_backtester

    if variant == "A":
        utilization_score.set_snap_missingness_mode("zero")
        ts_backtester.SNAP_IMPUTATION_MODE = "zero"
    else:
        utilization_score.set_snap_missingness_mode("preserve")
        ts_backtester.SNAP_IMPUTATION_MODE = "median"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from src.evaluation.ts_backtester import run_ts_backtest

    for variant in args.variants:
        for season in args.seasons:
            tag = f"{variant}_{season}"
            out = OUT_DIR / f"preds_{tag}.csv"
            if out.exists():
                print(f"[{tag}] already done — skipping", flush=True)
                continue

            configure(variant)
            print(f"\n=== variant {variant} | season {season} ===", flush=True)
            t0 = time.time()
            pred, metrics = run_ts_backtest(
                season=season, positions=POSITIONS,
                target_mode="component", verbose=False,
            )
            pred["variant"] = variant
            pred.to_csv(out, index=False)
            (OUT_DIR / f"metrics_{tag}.json").write_text(
                json.dumps(metrics.get("metrics", {}), indent=2, default=str))
            print(f"[{tag}] {len(pred):,} preds in {time.time()-t0:.0f}s -> {out}",
                  flush=True)

    print("\nAll runs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

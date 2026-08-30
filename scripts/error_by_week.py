"""Where does weekly-model error actually sit, by week of season?

Six retrains over the 2026-08-29/30 data-integrity audit moved aggregate 1-week
MAE by <=0.03. That is the wrong instrument for the fixes that were made: the
prior-week zero-fill corrupted weeks 1-4 specifically (~25% of rows), the
rookie prior touches week-1 rookies (0.80% of rows), and the pre-window
context repaired the first training season. An all-weeks average dilutes each
of those toward nothing.

This holds out a season, trains on the seasons before it, and reports MAE by
week bucket -- the measurement that distinguishes "the fixes did nothing" from
"the fixes helped where the corruption was, and the average hid it".

CAVEAT, stated because it limits the conclusion: run_fold calls
_prepare_training_data WITHOUT context_data, so this harness does NOT exercise
the pre-window context fix. The label, team_stats, PFR and prior-week fixes do
flow through, since they live in the data and feature layers.

Usage:
    python scripts/error_by_week.py --season 2025
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.evaluate import run_fold

BUCKETS = [(1, 1, "wk 1"), (2, 4, "wk 2-4"), (5, 8, "wk 5-8"),
           (9, 13, "wk 9-13"), (14, 18, "wk 14-18")]


def bucket_of(week: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= week <= hi:
            return name
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--positions", nargs="*", default=["QB", "RB", "WR", "TE"])
    args = ap.parse_args()

    rows = []
    for pos in args.positions:
        print(f"\n=== {pos}: holding out {args.season} ===", flush=True)
        try:
            _, test_df, pred, train_seasons = run_fold(pos, args.season)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        t = test_df[test_df["position"] == pos].copy()
        t["pred"] = pd.to_numeric(pred.reindex(t.index), errors="coerce")
        # target_1w is next week's actual points, which is what a 1-week model
        # is scored on. Rows without it have no future to be judged against.
        t["actual"] = pd.to_numeric(t.get("target_1w"), errors="coerce")
        t = t[t["actual"].notna() & t["pred"].notna()]
        if t.empty:
            print("  no scorable rows")
            continue
        t["abs_err"] = (t["pred"] - t["actual"]).abs()
        t["bucket"] = t["week"].map(bucket_of)
        print(f"  trained on {train_seasons}, scorable rows {len(t)}")

        for _, _, name in BUCKETS:
            sub = t[t["bucket"] == name]
            if sub.empty:
                continue
            rows.append(dict(position=pos, bucket=name, n=len(sub),
                             mae=sub["abs_err"].mean(),
                             mean_actual=sub["actual"].mean()))

    if not rows:
        print("\nnothing scorable")
        return 1

    df = pd.DataFrame(rows)
    print("\n\nMAE BY WEEK BUCKET")
    piv = df.pivot(index="bucket", columns="position", values="mae")
    order = [n for _, _, n in BUCKETS if n in piv.index]
    print(piv.reindex(order).round(3).to_string())
    print("\nrow counts")
    print(df.pivot(index="bucket", columns="position",
                   values="n").reindex(order).to_string())
    print("\nmean actual points (context for the MAE above)")
    print(df.pivot(index="bucket", columns="position",
                   values="mean_actual").reindex(order).round(2).to_string())

    out = Path("data/experiments") / f"error_by_week_{args.season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

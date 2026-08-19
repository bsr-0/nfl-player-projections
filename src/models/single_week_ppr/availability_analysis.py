"""Scoring for the availability-estimator comparison.

Deliberately does NOT rank by MAE. The failure mode being corrected is a
*gradient*: season-total bias rising with the share of the season that had
to be synthesized. An estimator that lowers average MAE while leaving that
slope intact has not fixed the mechanism -- it has just moved the average.

Decision hierarchy (in order):
  1. synthetic-share bias gradient  -- does the slope disappear?
  2. bias on real (non-synthetic) rows -- no new bias introduced elsewhere
  3. overall season MAE
  4. stability across folds/seasons
  5. complexity / number of assumptions
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = [-.01, .001, .25, .5, .75, 1.0]
BUCKET_LABELS = ["none", "0-25%", "25-50%", "50-75%", "75-100%"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bias"] = out.predicted_season_total - out.actual_season_total
    out["ae"] = out.bias.abs()
    out["synth_share"] = out.weeks_synthetic / out.possible_weeks
    out["bucket"] = pd.cut(out.synth_share, BUCKETS, labels=BUCKET_LABELS)
    return out


def gradient_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-estimator quantification of the failure mode.

    slope/r2   : OLS of bias ~ synthetic_share. slope->0 means availability
                 no longer drives season-total error. r2 says how much of
                 the bias variation the share still explains.
    max_bucket_delta / mean_abs_bucket_bias : distribution-free companions,
                 robust to the slope being dragged by a sparse extreme bucket.
    real_row_bias : bias on players needing NO synthetic weeks -- guards
                 against fixing the gradient by biasing everyone else.
    """
    rows = []
    for est, g in df.groupby("estimator"):
        x = g.synth_share.to_numpy(dtype=float)
        y = g.bias.to_numpy(dtype=float)
        if len(g) > 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            slope, r2 = np.nan, np.nan

        bucket_bias = g.groupby("bucket", observed=True).bias.mean()
        real_rows = g[g.weeks_synthetic == 0]
        rows.append({
            "estimator": est,
            "slope": slope,
            "r2": r2,
            "max_bucket_delta": (float(bucket_bias.max() - bucket_bias.min())
                                 if len(bucket_bias) > 1 else np.nan),
            "mean_abs_bucket_bias": float(bucket_bias.abs().mean()) if len(bucket_bias) else np.nan,
            "real_row_bias": float(real_rows.bias.mean()) if len(real_rows) else np.nan,
            "mae": float(g.ae.mean()),
            "bias": float(g.bias.mean()),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows).set_index("estimator")


def stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season slope, and its spread. An estimator that fixes the
    gradient in one season and inverts it in another is not a fix."""
    recs = []
    for (est, season), g in df.groupby(["estimator", "season"]):
        x = g.synth_share.to_numpy(dtype=float)
        y = g.bias.to_numpy(dtype=float)
        slope = np.polyfit(x, y, 1)[0] if len(g) > 2 and np.ptp(x) > 0 else np.nan
        recs.append({"estimator": est, "season": season, "slope": slope,
                     "bias": float(g.bias.mean())})
    per = pd.DataFrame(recs)
    piv = per.pivot(index="estimator", columns="season", values="slope")
    piv["slope_std"] = piv.std(axis=1)
    return piv


def availability_sensitivity(df: pd.DataFrame, rates=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Counterfactual curve: season MAE if EVERY synthetic week used a
    constant availability r. Exact, not simulated --
    season_total(r) = known_sum + r * synth_pred_sum. Locates the
    error-minimizing constant rate, which is the yardstick the adaptive
    estimators should be judged against.
    """
    base = df.drop_duplicates(subset=["player", "season"])[
        ["player", "season", "known_sum", "synth_pred_sum", "actual_season_total",
         "weeks_synthetic", "possible_weeks"]]
    out = []
    for r in rates:
        total = base.known_sum + r * base.synth_pred_sum
        bias = total - base.actual_season_total
        share = base.weeks_synthetic / base.possible_weeks
        slope = (np.polyfit(share.to_numpy(float), bias.to_numpy(float), 1)[0]
                 if len(base) > 2 and np.ptp(share) > 0 else np.nan)
        out.append({"constant_rate": r, "mae": float(bias.abs().mean()),
                    "bias": float(bias.mean()), "slope": slope})
    return pd.DataFrame(out)


def report(df: pd.DataFrame) -> None:
    d = prepare(df)
    pd.set_option("display.width", 160)

    print("\n" + "=" * 88)
    print("1. GRADIENT — the primary diagnostic (slope->0 means the mechanism is fixed)")
    print("=" * 88)
    gm = gradient_metrics(d).sort_values("slope", key=abs)
    print(gm.round(3).to_string())

    print("\n" + "=" * 88)
    print("2. BIAS BY SYNTHETIC-WEEK SHARE (want flat, not necessarily zero)")
    print("=" * 88)
    print(d.pivot_table(index="estimator", columns="bucket", values="bias",
                        aggfunc="mean", observed=False).round(1).to_string())

    print("\n" + "=" * 88)
    print("4. STABILITY — per-season slope (an estimator that flips sign is not a fix)")
    print("=" * 88)
    print(stability(d).round(2).to_string())

    print("\n" + "=" * 88)
    print("COUNTERFACTUAL — season MAE under a CONSTANT availability rate")
    print("(the adaptive estimators should beat the best constant)")
    print("=" * 88)
    print(availability_sensitivity(d).round(2).to_string(index=False))

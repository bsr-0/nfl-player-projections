#!/usr/bin/env python3
"""
Calibrate floor/ceiling spread for the draft board (GAPS.md §11.2.C
follow-up: "asymmetric floor/ceiling from real skew, not a full
mixture-density network").

Rebuilds the real prediction-vs-actual dataset behind
generate_draft_data.py's _floor_ceiling_spread() (real PreseasonProjector
predictions vs. real season totals, 2019-2025) and fits two one-sided
quantile regressions (floor / ceiling) instead of one symmetric one,
since real backtest residuals are confirmed right-skewed (not
classically bimodal — see GAPS.md for the skew/kurtosis/bimodality-
coefficient check this is based on).

Compares against the CURRENT symmetric formula
(FLOOR_CEILING_REL_SPREAD_* constants in generate_draft_data.py) on:
  - overall coverage (target ~86.6%, must not regress)
  - coverage by fp_std quintile (must stay tight, not reintroduce the
    tier-miscalibration problem the symmetric fix solved)
  - floor-side vs. ceiling-side coverage measured SEPARATELY (the new
    thing this script checks that the symmetric formula can't get right
    by construction)

Ships new coefficients only if the asymmetric fit is a real, measured
improvement on the per-side metric; otherwise reports the honest null
result and leaves the existing symmetric formula in place.

Usage:
    python scripts/calibrate_floor_ceiling.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from src.models.preseason_projector import PreseasonProjector

FIT_SEASONS = list(range(2019, 2023))    # projection_season 2019-2022
TEST_SEASONS = list(range(2023, 2026))   # projection_season 2023-2025
ALL_SEASONS = list(range(2018, 2026))    # needs prior season too -> train from 2018

# Two-sided coverage the bands aim for. 0.866 matches the original
# z=1.5-implied target; --target-coverage overrides it.
#
# The 86.6% bands are correctly calibrated (89.1% measured on holdout) and
# simultaneously uninformative: measured on the live 2026 board their median
# width is 206% of the projection (p90 326%), e.g. proj 330 -> floor 123,
# ceiling 490. That width is not a defect in the interval, it is an honest
# statement about a season-total model whose MAE is ~43 on a mean actual near
# 100. A narrower target trades coverage for decision-usefulness; it does NOT
# make the projection more certain.
DEFAULT_TARGET_COVERAGE = 0.866
TARGET_COVERAGE = DEFAULT_TARGET_COVERAGE
TAIL = (1 - TARGET_COVERAGE) / 2  # ~0.067

POSITIONS = ["QB", "RB", "WR", "TE"]

# Current symmetric formula (generate_draft_data.py), reproduced here for
# a real side-by-side comparison rather than trusting the docstring.
SYM_INTERCEPT = 2.620
SYM_LOG_PRED_COEF = -0.335
SYM_CONF_COEF = 0.022
SYM_POSITION_COEF = {"QB": 0.0, "RB": -0.121, "WR": -0.172, "TE": -0.295}
SYM_MIN, SYM_MAX = 0.15, 3.0


def symmetric_rel_spread(log_total, conf, position):
    pos_coef = SYM_POSITION_COEF.get(position, 0.0)
    rel = SYM_INTERCEPT + SYM_LOG_PRED_COEF * log_total + SYM_CONF_COEF * conf + pos_coef
    return np.clip(rel, SYM_MIN, SYM_MAX)


def build_dataset() -> pd.DataFrame:
    """Real PreseasonProjector predictions vs. real season totals,
    same methodology as the existing symmetric formula's calibration
    (see generate_draft_data.py:34-88)."""
    proj, pairs = PreseasonProjector.train(seasons=ALL_SEASONS)
    prepared = proj._prepare_feature_frame(pairs)

    rows = []
    for pos in POSITIONS:
        pos_df = prepared[prepared["position"] == pos].copy()
        if pos_df.empty or pos not in proj.feature_names:
            continue
        details = proj.predict_with_details(pos_df, pos)
        pos_df = pos_df.reset_index(drop=True)
        details = details.reset_index(drop=True)
        pos_df["pred_total"] = details["pred"].to_numpy()
        pos_df["confidence_score"] = details["confidence_score"].to_numpy()
        rows.append(pos_df)

    df = pd.concat(rows, ignore_index=True)
    df = df[df["pred_total"] > 0]
    df["rel_error"] = (df["season_total"] - df["pred_total"]) / df["pred_total"]
    df["log_pred"] = np.log(df["pred_total"].clip(lower=1.0))
    df["confidence_score"] = df["confidence_score"].fillna(0.7)
    return df


def fit_quantile(df: pd.DataFrame, q: float):
    """Quantile regression of signed relative error on log(pred) +
    confidence + position dummies (QB as reference level, matching the
    existing symmetric formula's parameterization)."""
    X = pd.DataFrame({
        "log_pred": df["log_pred"],
        "confidence_score": df["confidence_score"],
    })
    for pos in ["RB", "WR", "TE"]:
        X[f"pos_{pos}"] = (df["position"] == pos).astype(float)
    X = sm.add_constant(X)
    model = QuantReg(df["rel_error"], X)
    result = model.fit(q=q)
    return result


def predict_quantile(result, log_pred, conf, position):
    x = {"const": 1.0, "log_pred": log_pred, "confidence_score": conf,
         "pos_RB": 0.0, "pos_WR": 0.0, "pos_TE": 0.0}
    if position in ("RB", "WR", "TE"):
        x[f"pos_{position}"] = 1.0
    ordered = [x[c] for c in result.params.index]
    return float(np.dot(result.params.values, ordered))


def coverage_report(df: pd.DataFrame, floor_col: str, ceiling_col: str, label: str):
    within = (df["season_total"] >= df[floor_col]) & (df["season_total"] <= df[ceiling_col])
    below = df["season_total"] < df[floor_col]
    above = df["season_total"] > df[ceiling_col]
    print(f"\n{label}")
    print(f"  Overall coverage: {within.mean():.1%}  (target {TARGET_COVERAGE:.1%})")
    print(f"  Floor breached (actual < floor):   {below.mean():.1%}  (target ~{TAIL:.1%})")
    print(f"  Ceiling breached (actual > ceil):  {above.mean():.1%}  (target ~{TAIL:.1%})")

    fp_std = df.groupby("player_id")["season_total"].transform("std") if "player_id" in df.columns else None
    if "fp_std_quintile" in df.columns:
        q_cov = df.groupby("fp_std_quintile").apply(
            lambda g: ((g["season_total"] >= g[floor_col]) & (g["season_total"] <= g[ceiling_col])).mean()
        )
        print(f"  Coverage by fp_std quintile: {q_cov.round(3).to_dict()}")
    return within.mean(), below.mean(), above.mean()


def _configure_coverage(target: float) -> None:
    """Set the module-level coverage target and derived tail."""
    global TARGET_COVERAGE, TAIL
    if not 0.0 < target < 1.0:
        raise SystemExit(f"--target-coverage must be in (0,1), got {target}")
    TARGET_COVERAGE = float(target)
    TAIL = (1 - TARGET_COVERAGE) / 2


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE,
                    help="Two-sided coverage the bands aim for (default %(default)s). "
                         "0.50 gives a much narrower, more decision-useful band at "
                         "the cost of containing the outcome half the time.")
    args = ap.parse_args()
    _configure_coverage(args.target_coverage)
    print(f"Target coverage: {TARGET_COVERAGE:.1%}  (floor q={TAIL:.3f}, ceiling q={1-TAIL:.3f})")

    print("Building real prediction-vs-actual dataset (PreseasonProjector, 2018-2025)...")
    df = build_dataset()
    print(f"  {len(df)} player-seasons across {df['position'].nunique()} positions")

    fit_df = df[df["curr_season"].isin(FIT_SEASONS)].copy()
    test_df = df[df["curr_season"].isin(TEST_SEASONS)].copy()
    print(f"  Fit: {len(fit_df)} rows ({FIT_SEASONS}) | Test: {len(test_df)} rows ({TEST_SEASONS})")

    print(f"\nFitting floor quantile regression (q={TAIL:.3f})...")
    floor_result = fit_quantile(fit_df, TAIL)
    print(floor_result.summary())

    print(f"\nFitting ceiling quantile regression (q={1 - TAIL:.3f})...")
    ceiling_result = fit_quantile(fit_df, 1 - TAIL)
    print(ceiling_result.summary())

    # Apply both formulas to the test set.
    test_df = test_df.copy()
    test_df["sym_spread"] = test_df.apply(
        lambda r: symmetric_rel_spread(r["log_pred"], r["confidence_score"], r["position"]) * r["pred_total"],
        axis=1,
    )
    test_df["sym_floor"] = test_df["pred_total"] - test_df["sym_spread"]
    test_df["sym_ceiling"] = test_df["pred_total"] + test_df["sym_spread"]

    test_df["asym_floor_rel"] = test_df.apply(
        lambda r: predict_quantile(floor_result, r["log_pred"], r["confidence_score"], r["position"]), axis=1
    )
    test_df["asym_ceiling_rel"] = test_df.apply(
        lambda r: predict_quantile(ceiling_result, r["log_pred"], r["confidence_score"], r["position"]), axis=1
    )
    test_df["asym_floor"] = test_df["pred_total"] * (1 + test_df["asym_floor_rel"])
    test_df["asym_ceiling"] = test_df["pred_total"] * (1 + test_df["asym_ceiling_rel"])
    # Guard: floor must not exceed the point estimate, ceiling must not be below it.
    test_df["asym_floor"] = np.minimum(test_df["asym_floor"], test_df["pred_total"])
    test_df["asym_ceiling"] = np.maximum(test_df["asym_ceiling"], test_df["pred_total"])

    # fp_std quintile for the per-tier check (within-season week-to-week volatility
    # isn't in this season-total dataset; use season-total itself as the closest
    # available size/variance proxy, consistent with what's actually checkable here).
    test_df["fp_std_quintile"] = pd.qcut(test_df["pred_total"], 5, labels=False, duplicates="drop")

    sym_cov, sym_below, sym_above = coverage_report(test_df, "sym_floor", "sym_ceiling", "CURRENT symmetric formula")
    asym_cov, asym_below, asym_above = coverage_report(test_df, "asym_floor", "asym_ceiling", "NEW asymmetric formula")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    sym_side_gap = abs(sym_below - TAIL) + abs(sym_above - TAIL)
    asym_side_gap = abs(asym_below - TAIL) + abs(asym_above - TAIL)
    print(f"  Symmetric per-side miscalibration (sum of |actual-target|): {sym_side_gap:.3f}")
    print(f"  Asymmetric per-side miscalibration (sum of |actual-target|): {asym_side_gap:.3f}")
    if asym_side_gap < sym_side_gap and abs(asym_cov - TARGET_COVERAGE) <= abs(sym_cov - TARGET_COVERAGE) + 0.02:
        print("  -> Asymmetric formula measurably improves per-side calibration. SHIP IT.")
        print("\n  Fitted floor coefficients:")
        print(floor_result.params)
        print("\n  Fitted ceiling coefficients:")
        print(ceiling_result.params)
    else:
        print("  -> No measurable improvement over the symmetric formula. NOT shipping.")


if __name__ == "__main__":
    main()

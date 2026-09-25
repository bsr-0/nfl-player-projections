#!/usr/bin/env python
"""Paired comparison of two walk-forward runs from their logs.

`train.py --walk-forward` prints only `mean +/- std` per position and
discards the per-fold metrics (`wf_metrics` is a local that never leaves the
function). That output cannot answer "did this change help": comparing two
independent mean+/-std bands at n=4 folds throws away the pairing, which is
the only thing that makes 4 folds informative. Two arms trained on identical
folds differ only by the change under test, so the per-fold *delta* has far
less variance than either arm's spread across seasons.

This recovers the per-fold, per-position numbers from the logs and does the
paired analysis instead.

    python scripts/compare_walkforward_runs.py \
        --baseline /tmp/ab_postfix.log --baseline-label post-fix \
        --variant  /tmp/ab_prefix.log  --variant-label  pre-fix

Reads logs only; runs nothing and writes nothing.

Statistical caveats, stated rather than buried:
  * n=4 folds. Sign consistency across folds is more informative here than
    any p-value, and no p-value is reported for that reason.
  * Walk-forward folds share training data (nested expanding windows), so
    they are not independent draws. The bootstrap CI is descriptive, not a
    calibrated frequentist interval.
  * The most recent fold may test a partial (in-progress) season with far
    fewer player-weeks. It is flagged and reported both in and out of the
    pooled result, because pooling it equally overweights the noisiest fold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

POSITIONS = ("QB", "RB", "WR", "TE")

_FOLD_START = re.compile(r"Loading data for QB")
_TEST_SEASON = re.compile(r"Test season: (\d{4})")
_METRIC_HEAD = re.compile(r"^\s*(QB|RB|WR|TE) Training Metrics:")
_METRIC_BODY = re.compile(
    r"1w: RMSE=([0-9.]+), MAE=([0-9.]+), R.=(-?[0-9.]+)")
_SUMMARY = re.compile(
    r"^\s*(QB|RB|WR|TE): RMSE ([0-9.]+) \+/- ([0-9.]+)\s+MAE ([0-9.]+) \+/- ([0-9.]+)")

Metrics = Dict[str, float]


def parse_log(path: Path) -> Tuple[List[Tuple[Optional[int], Dict[str, Metrics]]], Dict[str, Metrics]]:
    """Return (folds, reported_summary).

    folds is [(test_season, {position: {rmse, mae, r2}})] in run order. The
    pre-walk-forward data load also emits a "Loading data for QB" line; it is
    dropped because no metric block follows it before the next fold starts.
    """
    lines = path.read_text(errors="replace").splitlines()

    starts = [i for i, l in enumerate(lines) if _FOLD_START.search(l)]
    bounds = list(zip(starts, starts[1:] + [len(lines)]))

    folds: List[Tuple[Optional[int], Dict[str, Metrics]]] = []
    for start, end in bounds:
        season = None
        for back in range(start, max(start - 8, 0) - 1, -1):
            found = _TEST_SEASON.search(lines[back])
            if found:
                season = int(found.group(1))
                break

        by_position: Dict[str, Metrics] = {}
        for i in range(start, end):
            head = _METRIC_HEAD.match(lines[i])
            if not head:
                continue
            for j in range(i + 1, min(i + 4, end)):
                body = _METRIC_BODY.search(lines[j])
                if body:
                    by_position[head.group(1)] = {
                        "rmse": float(body.group(1)),
                        "mae": float(body.group(2)),
                        "r2": float(body.group(3)),
                    }
                    break
        if by_position:
            folds.append((season, by_position))

    summary: Dict[str, Metrics] = {}
    for line in lines:
        found = _SUMMARY.match(line)
        if found:
            summary[found.group(1)] = {
                "rmse": float(found.group(2)), "rmse_std": float(found.group(3)),
                "mae": float(found.group(4)), "mae_std": float(found.group(5)),
            }
    return folds, summary


def parse_fold_metrics_json(path: Path) -> Tuple[List[Tuple[Optional[int], Dict[str, Metrics]]], Dict[str, Metrics]]:
    """Read walk_forward_fold_metrics.json -- full precision, no scraping.

    Preferred over parse_log(): the log rounds to 2dp, which cannot resolve
    effects below ~0.005 MAE.
    """
    payload = json.loads(path.read_text())
    folds = [(int(f["test_season"]), f["by_position"]) for f in payload["folds"]]
    return folds, {}


def _load(path: Path):
    if path.suffix == ".json":
        return parse_fold_metrics_json(path)
    return parse_log(path)


def _bootstrap_ci(deltas: np.ndarray, n: int = 20000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(n, deltas.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--variant-label", default="variant")
    ap.add_argument("--metric", default="mae", choices=("mae", "rmse", "r2"))
    ap.add_argument("--partial-season", type=int, default=None,
                    help="fold season to flag as in-progress and report separately")
    args = ap.parse_args()

    base_folds, base_summary = _load(args.baseline)
    var_folds, var_summary = _load(args.variant)

    if not base_folds or not var_folds:
        print("Could not recover per-fold metrics from one or both logs.")
        print(f"  {args.baseline.name}: {len(base_folds)} folds")
        print(f"  {args.variant.name}: {len(var_folds)} folds")
        return 1

    base_by_season = {s: m for s, m in base_folds if s is not None}
    var_by_season = {s: m for s, m in var_folds if s is not None}
    shared = sorted(set(base_by_season) & set(var_by_season))

    print("=" * 72)
    print(f"Paired walk-forward comparison — metric: {args.metric.upper()}")
    print(f"  baseline: {args.baseline_label}  ({args.baseline.name})")
    print(f"  variant : {args.variant_label}  ({args.variant.name})")
    print("=" * 72)

    if len(base_folds) != len(var_folds):
        print(f"\n!! fold-count mismatch: {len(base_folds)} vs {len(var_folds)}. "
              "The arms are not comparable fold-for-fold; investigate before "
              "reading anything below.")
    if not shared:
        print("\n!! no folds share a test season; cannot pair.")
        return 1

    lower_is_better = args.metric != "r2"
    print(f"\nPer-fold deltas (variant - baseline; "
          f"{'negative' if lower_is_better else 'positive'} = variant better)\n")
    header = f"  {'pos':<4}" + "".join(f"{s:>12}" for s in shared) + f"{'mean Δ':>11}{'folds won':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    overall: Dict[str, np.ndarray] = {}
    for pos in POSITIONS:
        deltas, cells = [], []
        for season in shared:
            b = base_by_season[season].get(pos)
            v = var_by_season[season].get(pos)
            if b is None or v is None:
                cells.append(f"{'--':>12}")
                continue
            d = v[args.metric] - b[args.metric]
            deltas.append(d)
            flag = "*" if args.partial_season == season else ""
            cells.append(f"{d:>+11.3f}{flag:<1}")
        if not deltas:
            continue
        arr = np.array(deltas)
        overall[pos] = arr
        wins = int((arr < 0).sum() if lower_is_better else (arr > 0).sum())
        print(f"  {pos:<4}" + "".join(cells) + f"{arr.mean():>+11.3f}{f'{wins}/{arr.size}':>11}")

    print("\nPer-position paired summary")
    for pos, arr in overall.items():
        lo, hi = _bootstrap_ci(arr)
        straddles = lo <= 0 <= hi
        verdict = ("no consistent effect" if straddles else
                   ("variant better" if (arr.mean() < 0) == lower_is_better else "variant worse"))
        print(f"  {pos}: mean Δ {arr.mean():+.3f}  paired sd {arr.std(ddof=1):.3f}  "
              f"bootstrap 95% [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    if args.partial_season and args.partial_season in shared:
        print(f"\nExcluding the in-progress {args.partial_season} fold (*):")
        keep = [i for i, s in enumerate(shared) if s != args.partial_season]
        for pos, arr in overall.items():
            if arr.size != len(shared):
                continue
            sub = arr[keep]
            print(f"  {pos}: mean Δ {sub.mean():+.3f} over {sub.size} folds "
                  f"(pooled was {arr.mean():+.3f})")

    print("\nFor contrast — what the run's own summary reports:")
    for pos in POSITIONS:
        b, v = base_summary.get(pos), var_summary.get(pos)
        if not b or not v:
            continue
        key, std = args.metric, f"{args.metric}_std"
        print(f"  {pos}: {args.baseline_label} {b[key]:.2f}+/-{b.get(std, float('nan')):.2f}   "
              f"{args.variant_label} {v[key]:.2f}+/-{v.get(std, float('nan')):.2f}")
    print("\n  Those bands overlap almost by construction: each spans four "
          "SEASONS, whose difficulty varies far more than the change under "
          "test. The paired deltas above are the signal; the bands are not.")

    from_json = args.baseline.suffix == ".json" and args.variant.suffix == ".json"
    if from_json:
        print("\nFull-precision inputs (walk_forward_fold_metrics.json): no "
              "rounding limit applies to the deltas above.")
        print("\nRead this with the caveats in the module docstring: n=4, folds "
              "share training data, and sign consistency matters more than the CI.")
        return 0

    resolution = 0.005  # logs print RMSE/MAE at 2 decimal places
    biggest = max((np.abs(a).max() for a in overall.values()), default=0.0)
    print(f"\nRESOLUTION LIMIT: the logs round to 2dp, so recovered deltas are "
          f"quantised at +/-{resolution:.3f}.")
    if biggest <= 4 * resolution:
        print(f"  The largest delta observed ({biggest:.3f}) is within a few "
              "quantisation steps of zero. A delta of '+0.000' here means "
              "'below log precision', NOT 'provably identical' -- do not read "
              "these as exact nulls.")
        print("  What this CAN support: an upper bound. If every paired delta "
              f"is <={biggest:.3f} MAE, the effect is smaller than that, which "
              "is already below the per-fold noise band this repo has "
              "established elsewhere (~0.046 MAE, see GAPS.md).")
        print("  What it CANNOT support: a point estimate of the effect, or a "
              "claim that the effect is exactly zero.")
    print("  Fix for future runs: train.py --walk-forward now writes "
          "walk_forward_fold_metrics.json at full precision; prefer "
          "--from-json over log parsing when that file exists.")

    print("\nRead this with the caveats in the module docstring: n=4, folds "
          "share training data, and sign consistency matters more than the CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

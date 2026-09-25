#!/usr/bin/env python
"""Row-level, segment-aware A/B comparison of two OOF panels.

`scripts/compare_walkforward_runs.py` compares two runs' aggregate fold
metrics -- four numbers per position per fold. It cannot answer "does this
help returning players and hurt cold-start players equally" because it never
sees a row. This is the missing half: it inner-joins two
`walk_forward_oof_predictions.parquet`-shaped panels (use the per-run
directories from `write_run_panel`, NEVER the flat "latest" pointer -- that
file is overwritten by whichever run wrote last and cannot represent two
runs at once) on (player_id, season, week), computes the per-row paired
error delta, and reports cluster-bootstrapped segment comparisons.

    python scripts/compare_oof_panels.py \
        --baseline data/experiments/oof_panels/<run>/panel.parquet --baseline-label post-fix \
        --variant  data/experiments/oof_panels/<run>/panel.parquet --variant-label  pre-fix \
        --by position,is_cold_start,week_bucket

Refuses to report anything if the two panels' row overlap is too small to
mean anything (default: fewer than 30 paired rows, or under 50% of either
panel's rows) -- a comparison built on 4 shared rows out of 900 offered is
not a comparison, it is noise with a p-value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.oof_capture import cluster_bootstrap_ci  # noqa: E402

JOIN_KEYS = ("player_id", "season", "week")
MIN_OVERLAP_ROWS = 30
MIN_OVERLAP_FRACTION = 0.5


def _load(path: Path) -> pd.DataFrame:
    if path.name == "walk_forward_oof_predictions.parquet":
        print(f"WARNING: {path} is the flat 'latest' pointer, overwritten by "
              "whichever run wrote last. If you are comparing two arms, point "
              "this at their per-run directories instead "
              "(data/experiments/oof_panels/<run_id>/panel.parquet), or the "
              "'baseline' and 'variant' you load may be the SAME run.",
              file=sys.stderr)
    return pd.read_parquet(path)


def paired_frame(baseline: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on identity, keeping only rows both panels scored.

    `position` / `is_cold_start` / `week_bucket` are context, not values
    being compared -- they should agree between the two panels for the same
    player-week (position doesn't change; week_bucket depends only on week;
    is_cold_start depends on panel history, which should match if both arms
    cover the same seasons). Merging them naively would suffix BOTH copies
    (`position_base`/`position_var`), silently breaking any `--by position`
    grouping downstream with a KeyError on the now-absent plain name. Keep
    one copy (baseline's) and only suffix genuine value columns.
    """
    context_cols = [c for c in ("position", "is_cold_start", "week_bucket") if c in baseline.columns]
    value_cols = [c for c in ("actual_points", "predicted_points", "residual")
                 if c in baseline.columns and c in variant.columns]

    base = baseline[[*JOIN_KEYS, *context_cols, *value_cols]]
    var = variant[[*JOIN_KEYS, *value_cols]]
    merged = base.merge(var, on=list(JOIN_KEYS), suffixes=("_base", "_var"))

    if "position" in variant.columns:
        # Cheap, worth catching: if this ever fires, the two panels disagree
        # on what position a player-week is, which would silently mis-group
        # every segment comparison below it.
        check = baseline[[*JOIN_KEYS, "position"]].merge(
            variant[[*JOIN_KEYS, "position"]], on=list(JOIN_KEYS), suffixes=("_a", "_b"))
        mismatched = check["position_a"] != check["position_b"]
        if mismatched.any():
            print(f"WARNING: {int(mismatched.sum())} shared player-weeks have a "
                  "different `position` between baseline and variant; using "
                  "baseline's value for grouping.", file=sys.stderr)

    merged["abs_err_base"] = merged["residual_base"].abs()
    merged["abs_err_var"] = merged["residual_var"].abs()
    merged["paired_delta"] = merged["abs_err_var"] - merged["abs_err_base"]  # variant - baseline
    return merged


def segment_comparison(paired: pd.DataFrame, *, by, cluster="player_id", n_boot=1000, seed=0):
    rows = []
    for key, group in paired.groupby(list(by), dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        entry = dict(zip(by, key))
        entry["n"] = len(group)
        entry["mean_paired_delta"] = float(group["paired_delta"].mean())
        lo, hi = cluster_bootstrap_ci(
            group["paired_delta"].to_numpy(), group[cluster].to_numpy(),
            statistic=np.mean, n_boot=n_boot, seed=seed)
        entry["ci_lo"], entry["ci_hi"] = lo, hi
        entry["straddles_zero"] = bool(lo <= 0 <= hi) if not (np.isnan(lo) or np.isnan(hi)) else None
        rows.append(entry)
    return pd.DataFrame(rows).sort_values(list(by)).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--variant-label", default="variant")
    ap.add_argument("--by", default="position,is_cold_start,week_bucket",
                    help="comma-separated grouping columns")
    ap.add_argument("--cluster", default="player_id")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-overlap-rows", type=int, default=MIN_OVERLAP_ROWS)
    ap.add_argument("--min-overlap-fraction", type=float, default=MIN_OVERLAP_FRACTION)
    args = ap.parse_args()

    baseline, variant = _load(args.baseline), _load(args.variant)
    if baseline.empty or variant.empty:
        print("One or both panels are empty.")
        return 1

    paired = paired_frame(baseline, variant)
    smaller = min(len(baseline), len(variant))
    overlap_fraction = len(paired) / smaller if smaller else 0.0

    print("=" * 72)
    print("Row-level OOF panel comparison")
    print(f"  baseline: {args.baseline_label}  ({args.baseline}, {len(baseline):,} rows)")
    print(f"  variant : {args.variant_label}  ({args.variant}, {len(variant):,} rows)")
    print(f"  paired (inner-joined on player/season/week): {len(paired):,} rows "
          f"({overlap_fraction:.0%} of the smaller panel)")
    print("=" * 72)

    if len(paired) < args.min_overlap_rows or overlap_fraction < args.min_overlap_fraction:
        print(f"\nREFUSING to report segment deltas: overlap is below the "
              f"stated threshold (>= {args.min_overlap_rows} rows AND "
              f">= {args.min_overlap_fraction:.0%} of the smaller panel).")
        print("This usually means the two panels come from different fold "
              "structures (different test seasons) or one is the flat "
              "'latest' pointer -- see the warning above if so.")
        return 1

    by = tuple(args.by.split(","))
    report = segment_comparison(paired, by=by, cluster=args.cluster, n_boot=args.n_boot)

    print(f"\nPaired per-row |error| delta ({args.variant_label} - {args.baseline_label}; "
          f"negative = variant more accurate), clustered by {args.cluster}:\n")
    print(report.to_string(index=False))

    print("\nRead the CI, not just the sign of mean_paired_delta: "
          "'straddles_zero'=True means this cell cannot distinguish the "
          "variant from the baseline at the given cluster-bootstrap "
          "resolution -- it is not evidence of 'no effect', it is an "
          "underpowered cell. No multiple-comparison correction is applied "
          "across the printed cells; treat any single cell picked out after "
          "seeing this table as exploratory, not confirmatory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

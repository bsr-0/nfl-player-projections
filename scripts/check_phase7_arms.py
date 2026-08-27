#!/usr/bin/env python3
"""Mid-run validator for a two-arm Phase 7 comparison.

Exists because the 2026-08-26 history-NaN run burned 12 hours before its
falsification check ran, and that check then failed for a reason that was
detectable after the FIRST season: the strata described real historical rows
while cold-start scores synthetic ones, so "zero dose" rows were not actually
zero dose.

Every check here is runnable against PARTIAL output and is safe to call
repeatedly. Exit code is non-zero if any hard check fails, so a driver loop
can abort the run instead of spending the remaining hours.

    python scripts/check_phase7_arms.py --dir data/experiments/<run>
    python scripts/check_phase7_arms.py --dir <run> --season 2015
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["player", "position", "season", "week"]
SEASON_KEY = ["player", "position", "season"]


def _seasons_ready(d: Path) -> list[int]:
    """Seasons with BOTH arms' season CSV and feature parquet on disk."""
    out = []
    for f in sorted(glob.glob(str(d / "arm_off" / "phase7_*.csv"))):
        s = int(Path(f).stem.split("_")[-1])
        need = [d / "arm_on" / f"phase7_{s}.csv",
                d / "arm_off" / f"features_{s}.parquet",
                d / "arm_on" / f"features_{s}.parquet"]
        if all(p.exists() for p in need):
            out.append(s)
    return out


def _dose(d: Path, season: int) -> pd.DataFrame:
    """Per player-season dose, computed from the SCORED synthetic feature rows.

    Dose is derived from FEATURES, never from the prediction difference --
    deriving the stratum from the outcome is what pre-registration exists to
    prevent.
    """
    a = pd.read_parquet(d / "arm_off" / f"features_{season}.parquet")
    b = pd.read_parquet(d / "arm_on" / f"features_{season}.parquet")
    a = a.sort_values(KEY).reset_index(drop=True)
    b = b.sort_values(KEY).reset_index(drop=True)
    if len(a) != len(b) or not (a[KEY].values == b[KEY].values).all():
        raise SystemExit(f"FAIL {season}: captured feature rows do not align between arms "
                         f"({len(a)} vs {len(b)})")
    feats = [c for c in a.columns if c not in KEY and
             pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    va, vb = a[feats].to_numpy(dtype="float64"), b[feats].to_numpy(dtype="float64")
    diff = ~((va == vb) | (np.isnan(va) & np.isnan(vb)))
    t = a[KEY].copy()
    t["week_changed"] = diff.any(axis=1)
    g = t.groupby(SEASON_KEY)["week_changed"].agg(["size", "sum"]).reset_index()
    g.columns = SEASON_KEY + ["weeks", "weeks_changed"]
    g["dose"] = g.weeks_changed / g.weeks
    return g


def check_season(d: Path, season: int, verbose: bool = True,
                 placebo: bool = False) -> list[str]:
    fails: list[str] = []
    off = pd.read_csv(d / "arm_off" / f"phase7_{season}.csv")
    on = pd.read_csv(d / "arm_on" / f"phase7_{season}.csv")

    if len(off) != len(on):
        fails.append(f"{season}: row counts differ ({len(off)} vs {len(on)})")
    j = off.merge(on, on=SEASON_KEY, suffixes=("_off", "_on"))
    if len(j) == 0:
        fails.append(f"{season}: no joinable rows")
        return fails

    # Ground truth cannot depend on the arm.
    if not np.allclose(j.actual_season_total_off, j.actual_season_total_on):
        fails.append(f"{season}: actual_season_total differs between arms")

    if j.predicted_season_total_off.isna().any() or j.predicted_season_total_on.isna().any():
        fails.append(f"{season}: NaN predictions present")

    # Preseason invariant: every scored week must be synthetic.
    for arm, df in (("off", off), ("on", on)):
        if not (df.weeks_synthetic == df.weeks_predicted).all():
            fails.append(f"{season}/{arm}: preseason invariant violated "
                         f"(a week was treated as known-played)")

    # Era-aware week cap: 16 game-weeks through 2020, 17 from 2021.
    expected = 17 if season >= 2021 else 16
    got = sorted(off.possible_weeks.unique().tolist())
    if any(g > expected for g in got):
        fails.append(f"{season}: possible_weeks {got} exceeds {expected} "
                     f"(wild-card round leaking into the season)")

    # THE check that the 2026-08-26 run needed on day one.
    dose = _dose(d, season)
    j = j.merge(dose, on=SEASON_KEY, how="left")
    if j.dose.isna().any():
        fails.append(f"{season}: {int(j.dose.isna().sum())} rows have no dose "
                     f"(feature capture does not cover the scored population)")
    # Paired primary outcome, reported per season (not gated -- this is the
    # thing being measured, not an integrity condition).
    d = ((j.predicted_season_total_on - j.actual_season_total_on).abs()
         - (j.predicted_season_total_off - j.actual_season_total_off).abs())
    ident_pred = int(np.isclose(j.predicted_season_total_off,
                                j.predicted_season_total_on, atol=1e-9).sum())

    # Mechanism note: how much of the effect could possibly be row-level.
    # NOT a gate. A near-100% identical-feature share with ~0% identical
    # predictions is the expected signature of a GLOBAL (model-level)
    # treatment -- the flag changes the training matrix, so both arms fit
    # different models. There is deliberately no zero-dose gate here: v1
    # gated on one and it failed on every run because no null stratum can
    # exist in this design. See PRE_REGISTRATION.md.
    zero_dose = int((j.dose == 0).sum())

    if placebo:
        # Placebo pair: two identical arms. This is the falsification check,
        # and unlike a null stratum it CAN pass.
        if ident_pred != len(j):
            fails.append(f"{season}: PLACEBO mismatch -- {len(j) - ident_pred} of "
                         f"{len(j)} predictions differ between two identical arms. "
                         f"max|diff|={d.abs().max():.6g}. Pipeline is nondeterministic; "
                         f"no ON-OFF difference is interpretable.")

    if verbose:
        print(f"  {season}: n={len(j):4d}  paired dMAE={d.mean():+.4f}  "
              f"identical_preds={ident_pred}/{len(j)}  "
              f"zero_dose_rows={zero_dose}  "
              f"identical_feature_rows={100 * (1 - j.dose.gt(0).mean()):.1f}%  "
              f"{'OK' if not fails else 'FAIL'}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--placebo", action="store_true",
                    help="both arms are the SAME configuration; require exact agreement")
    ap.add_argument("--season", type=int, default=None,
                    help="check one season (default: every season with both arms ready)")
    args = ap.parse_args()

    seasons = [args.season] if args.season else _seasons_ready(args.dir)
    if not seasons:
        print("no season has both arms complete yet — nothing to check")
        return

    print(f"checking {len(seasons)} season(s) with both arms present: {seasons}")
    fails: list[str] = []
    for s in seasons:
        fails += check_season(args.dir, s, placebo=args.placebo)

    if fails:
        print("\n" + "!" * 70)
        for f in fails:
            print("!! " + f)
        print("!" * 70)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()

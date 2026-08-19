#!/usr/bin/env python
"""Phase 7C: does a mean-oriented weekly objective produce better season totals?

Pre-registered, single-lever test. Phase 7A established that for players who
play a full season the season bias is inherited almost entirely from the
weekly estimator (aggregation contributes 2-7%, with inconsistent sign), and
that the weekly model under-predicts because MAE/Huber objectives target the
conditional MEDIAN of a right-skewed target. Summing ~16 medians
under-estimates the mean of the sum.

    ARM A (baseline)   FINAL_CONFIG as deployed
        QB F_yeojohnson_huber | RB C_gbm_mae | WR C_gbm_mae | TE B_gbm_huber

    ARM B (treatment)  identical in every respect EXCEPT the weekly objective
        QB G_yeojohnson_mse   | RB A_gbm_mse  | WR A_gbm_mse  | TE A_gbm_mse

Same features, same training windows, same weighting, same folds, same
population, no hyperparameter retuning. QB keeps its Yeo-Johnson transform so
its arm changes one thing; note that minimising MSE in transformed space is
not the conditional mean in PPR space, so QB is the near-control arm (its
skew is 0.29 and its mean-median gap 0.11, the smallest of the four).

Emits per-week rows alongside season rows from the SAME fit, so weekly and
season metrics cannot drift apart across runs.

DECISION RULE, fixed before running:
  weekly MAE same/slightly worse + season MAE materially better  -> strong candidate
  weekly improves + season improves                              -> adopt
  weekly modestly worse + season modestly better                 -> discuss
  weekly materially worse + season modestly better               -> reject
  season better only for zero-synthetic, synthetic worse         -> reject/investigate
  no meaningful season improvement                               -> reject

Season bias is NOT required to reach zero; the objective is a better season
estimator, not perfect calibration.

Usage:
    python scripts/run_phase7c_objective_test.py --arm baseline
    python scripts/run_phase7c_objective_test.py --arm treatment
    python scripts/run_phase7c_objective_test.py --arm treatment --positions TE --seasons 2024
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TREATMENT_ARCHITECTURES = {
    "QB": "G_yeojohnson_mse",
    "RB": "A_gbm_mse",
    "WR": "A_gbm_mse",
    "TE": "A_gbm_mse",
}

OUT_DIR = Path("data/experiments/phase7c")


def main() -> int:
    from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS
    from src.models.single_week_ppr.season_projection import run_season_projection

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["baseline", "treatment"], required=True)
    ap.add_argument("--positions", nargs="+", default=None)
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=list(DEFAULT_VALIDATION_SEASONS))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    season_path = OUT_DIR / f"season_{args.arm}{suffix}.csv"
    week_path = OUT_DIR / f"weeks_{args.arm}{suffix}.csv"
    for p in (season_path, week_path):
        if p.exists():
            p.unlink()

    override = TREATMENT_ARCHITECTURES if args.arm == "treatment" else None
    print(f"=== PHASE 7C arm={args.arm} override={override} ===", flush=True)

    result = run_season_projection(
        positions=args.positions, seasons=args.seasons,
        output_path=season_path, week_output_path=week_path,
        architecture_override=override,
    )
    print(f"\narm={args.arm}: {len(result)} season rows -> {season_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

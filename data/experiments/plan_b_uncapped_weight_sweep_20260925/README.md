# Plan B joint-MAE `other_weight` sweep on the uncapped population — 2026-09-25

Exploratory, dev-only sweep on the frozen 2020-2022 folds
(`data/experiments/plan_b_uncapped_preflight_20260925/`), all six targets
Plan B's joint architecture now covers (the original 4 plus `receptions`/
`passing_yards`, added the same day the roster-slot cap was removed --
see `docs/PLAN_B_FUTURE_RUN.md`). Not a confirmation result; the 2023-2025
real confirmation lives in `data/experiments/plan_b_uncapped_joint_mae_confirm_20260925/`.

## Grid

`other_weight` in `{0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0}`,
all six targets, 2020-2022 test seasons, default `epsilon=0.001`,
`penalty=0.00001`, `smooth_width=0.002`.

## Finding: uncapping essentially erased the row-MAE/mass-calibration trade-off

The earlier capped-population sweep found a genuine, smooth trade-off
frontier across the whole weight range -- lower weight always improved row
MAE and always worsened mass calibration, with real cost either way. On this
uncapped population, mass MAE still jumps sharply between `other_weight=0`
(~0.15-0.31, still diagnostic-failure territory -- an unconstrained model
still badly misjudges the now-much-smaller omitted bucket) and
`other_weight=0.01` (~0.002-0.08), but **from 0.01 to 1.0 it is essentially
flat, often continuing to improve slightly, while row MAE is also
essentially flat across that same range** (typically a 4th-5th-decimal
difference between `other_weight=0.01` and `other_weight=1.0` -- see any
target's full `report.json` in this directory for the exact numbers per
weight). Uncapping removed most of the population that made the omitted
bucket large and consequential in the first place, so there is very little
left to trade off once the model is calibrated at all.

Per-target selected weight via the same nested procedure as the
capped-population investigation (select on 2020-2021 only, check on
untouched 2022) is documented in
`data/experiments/plan_b_uncapped_joint_mae_confirm_20260925/README.md`,
which also flags that `targets` and `rushing_attempts` did not cleanly pass
that nested check at any calibrated weight, unlike the other four targets.

## Caveats

- This is a hyperparameter sweep on dev folds, read alongside its own nested
  check before treating any weight as validated -- see the confirmation
  README for the honest version of that comparison.
- `receptions`/`passing_yards` have no equal-weighting/unconstrained-
  diagnostic precedent to compare against; this is their first sweep.

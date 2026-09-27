# Plan B joint-MAE `other_weight` sweep — 2026-09-25

Exploratory, dev-only sweep. Not a predeclared confirmation result. Frozen
2020–2022 dev folds only (`data/experiments/plan_b_preflight_20260925/<target>/input_panel.csv`,
same panel used by the earlier `plan_b_joint_*_dev_20260925` runs); the
2023–2025 confirmation population is untouched.

## Why

The earlier dev comparison found a hard trade-off: the unconstrained
smooth-MAE joint model (`other_weight` implicitly 0, no omitted-bucket term)
beat rolling-3 on row MAE but badly mis-stated represented-team mass; adding
an *equally weighted* omitted-bucket MAE term (`other_weight=1.0`, the
existing default) fixed mass calibration but erased the row-MAE gain. Both
terms are already means (over player-rows and over team-weeks respectively),
so equal weight is not automatically the right trade-off — this sweep tests
whether an intermediate weight keeps most of the mass-calibration fix while
keeping some of the row-MAE gain.

`other_weight` was added as a constructor parameter to `JointOtherMAEModel`
(`src/models/team_hierarchical/joint_other_mae.py`) and threaded through
`run_backtest`/`evaluate_plan_b_joint_mae.py --other-weight`. `other_weight=1.0`
was verified to reproduce the original `plan_b_joint_mae_mass_dev_20260925`
run bit-for-bit (identical pooled MAE and mass diagnostics) before running
this sweep, confirming the refactor is a no-op at the old default.

## Grid

`other_weight` in `{0.0, 0.1, 0.25, 0.5, 1.0}`, all four volume targets,
2020-2022 test seasons, default `epsilon=0.001`, `penalty=0.00001`,
`smooth_width=0.002` (unchanged from the prior dev runs).

## Result

| Target | w | Joint MAE | Rolling-3 MAE | Delta | 95% paired CI | Mean represented mass (actual / candidate) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| targets | 0.0 | 0.04389 | 0.04554 | -0.00165 | [-0.00200, -0.00132] | 0.934 / 0.820 |
| targets | 0.1 | 0.04506 | 0.04554 | -0.00049 | [-0.00083, -0.00017] | 0.934 / 0.957 |
| targets | 1.0 | 0.04678 | 0.04554 | +0.00123 | [+0.00066, +0.00176] | 0.934 / 0.966 |
| rushing_attempts | 0.0 | 0.03639 | 0.03814 | -0.00175 | [-0.00211, -0.00143] | 0.926 / 0.813 |
| rushing_attempts | 0.1 | 0.03781 | 0.03814 | -0.00033 | [-0.00070, +0.00000] | 0.926 / 0.950 |
| rushing_attempts | 1.0 | 0.03933 | 0.03814 | +0.00119 | [+0.00058, +0.00182] | 0.926 / 0.963 |
| receiving_yards | 0.0 | 0.05306 | 0.05762 | -0.00456 | [-0.00506, -0.00404] | 0.937 / 0.676 |
| receiving_yards | 0.1 | 0.05665 | 0.05762 | -0.00097 | [-0.00137, -0.00058] | 0.937 / 0.968 |
| receiving_yards | 1.0 | 0.05801 | 0.05762 | +0.00039 | [-0.00021, +0.00098] | 0.937 / 0.978 |
| rushing_yards | 0.0 | 0.04139 | 0.04516 | -0.00377 | [-0.00420, -0.00336] | 0.928 / 0.721 |
| rushing_yards | 0.1 | 0.04417 | 0.04516 | -0.00099 | [-0.00142, -0.00055] | 0.928 / 0.956 |
| rushing_yards | 1.0 | 0.04558 | 0.04516 | +0.00042 | [-0.00021, +0.00108] | 0.928 / 0.969 |

(`w=0.25`/`w=0.5` rows are in each target's own `report.json`; omitted here
for brevity — mass calibration at those weights is close to `w=1.0`, row MAE
is between the `w=0.1` and `w=1.0` rows, monotonically.)

Full per-weight, per-target `report.json`/`predictions.csv`/`manifest.json`
are under `<target>_w<weight>/`, each independently saved-row-MAE-verified by
`evaluate_plan_b_joint_mae.py` the same way as every other Plan B run.

**Mass calibration jumps almost entirely between `w=0.0` and `w=0.1`, then is
roughly flat from `w=0.1` through `w=1.0`.** At `w=0.1`, mean represented mass
is already within ~2-3 points of actual for every target (vs. 11-26 points
off at `w=0.0`), essentially as calibrated as the fully-weighted `w=1.0` case.
Meanwhile row MAE at `w=0.1` is still better than rolling-3, with a paired
95% CI entirely negative (significant improvement) for **targets,
receiving_yards, and rushing_yards**; the **rushing_attempts** interval at
`w=0.1` touches zero (`[-0.00070, +0.00000]`), so that target is inconclusive
under this doc's own "CI must exclude zero" bar, not a clean win.

## Caveats — read before treating this as a Plan B win

- **This is a hyperparameter sweep on the dev folds, not a predeclared
  confirmation.** The mass-calibration tolerance used to pick "`w=0.1` looks
  like the right trade-off" was not numerically fixed before running the
  grid — I looked at the table and judged it, the same kind of after-the-fact
  read this repo's own methodology warns against. Selecting `other_weight`
  by eye across a 5-point grid, across 4 targets, is itself a form of
  multiple-comparison selection; the reported CIs do not correct for it.
- **No confirmation-season run has been done.** 2023-2025 remains untouched.
  Before this counts as an actual Plan B improvement, `other_weight` must be
  fixed at a single predeclared value (recommend 0.1, or a narrower re-sweep
  over e.g. {0.05, 0.1, 0.15} on the *same* dev folds only, never touching
  2023-2025) and then run exactly once against 2023-2025 as the confirmation
  population, per `docs/PLAN_B_FUTURE_RUN.md`'s existing reproduction
  contract.
- This remains a capped-roster share-MAE result, not a PPR result, and still
  does not evaluate the full joint team architecture.

## Next step (not yet run)

Freeze `other_weight=0.1` (or a value chosen from a narrower dev-only
re-sweep), then run `evaluate_plan_b_joint_mae.py --test-seasons 2023 2024 2025`
against the same frozen `plan_b_preflight_20260925` input panels as the
one real confirmation attempt for this configuration.

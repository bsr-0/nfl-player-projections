# Plan B joint-MAE weighted confirmation — 2026-09-25

**First result in the entire Plan B investigation where a candidate model beats
rolling-3 with a paired 95% CI excluding zero, on the real 2023-2025
confirmation seasons, on all four volume targets simultaneously.** Read the
caveats below before treating this as settled -- it changes the record but
does not close it.

## Result

| Target | other_weight | Held-out rows | Rolling-3 MAE | Joint MAE | Delta, paired 95% CI | Mean mass (actual → candidate) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Targets | 0.01 | 21,211 | 0.04457 | 0.04327 | -0.00130 [-0.00168, -0.00094] | 0.9466 → 0.8995 |
| Rushing attempts | 0.05 | 24,475 | 0.03555 | 0.03488 | -0.00067 [-0.00112, -0.00027] | 0.9511 → 0.9605 |
| Receiving yards | 0.03 | 21,211 | 0.05705 | 0.05502 | -0.00203 [-0.00253, -0.00153] | 0.9502 → 0.9263 |
| Rushing yards | 0.10 | 24,475 | 0.04330 | 0.04213 | -0.00117 [-0.00166, -0.00065] | 0.9514 → 0.9725 |

Every interval excludes zero on the improvement side. Mass calibration is in
the same ballpark this repo previously accepted as "corrected" (~0.05-0.07
mean absolute team-week mass error); no target is anywhere near the ~0.13-0.27
diagnostic-failure level the fully unconstrained (`other_weight=0`) objective
produced. `evaluate_plan_b_joint_mae.py`'s built-in checks passed for all four
runs: saved-row MAE recomputation matched the report, and the frozen-panel/
prior-run-predictions hash checks confirmed identical held-out keys, labels,
and rolling-3 baseline against the earlier mixed-effects confirmation run
(`data/experiments/plan_b_run_20260925/`).

## Methodology (why these specific weights)

`other_weight` (new parameter, `src/models/team_hierarchical/joint_other_mae.py`)
scales the omitted-player bucket's contribution to the training loss relative
to the player-row term. A sweep on the frozen 2020-2022 dev folds
(`plan_b_joint_weight_sweep_20260925`, `_resweep_`, `_finesweep_`) showed this
is a smooth trade-off frontier, not a corner solution: lower weight always
improves row MAE and always worsens mass calibration, with no universal
elbow. Two of four targets (`targets`, `receiving_yards`) show diminishing
mass-calibration returns above `w≈0.03-0.05`; the other two do not plateau
before `w=1`.

Because there is no natural elbow, weight selection requires a predeclared
tolerance. A round tolerance (mean team-week mass MAE ≤ 0.08 -- between the
diagnostic-failure level at `w=0` and the ~0.05-0.07 level judged acceptable
by the original equal-weighted fix) was applied per target, but naively
picking the smallest passing weight from the full pooled 2020-2022 dev metric
conflates hyperparameter selection with the very data used to report
significance. To avoid that, the weight was instead selected using **only**
the 2020-2021 dev folds, then checked against the untouched 2022 fold before
being trusted:

| Target | Selected w (2020-21 only) | Holds up on untouched 2022? |
| --- | ---: | --- |
| targets | 0.01 | yes -- CI [-0.00195, -0.00087] |
| receiving_yards | 0.03 | yes -- CI [-0.00289, -0.00157] |
| rushing_yards | 0.10 | yes -- CI [-0.00169, -0.00030] |
| rushing_attempts | 0.05 | no -- CI [-0.00095, +0.00008], crosses zero |

Only after that nested check were these four weights carried into the one
real 2023-2025 confirmation run above. `rushing_attempts` was carried forward
anyway (for completeness, flagged as the shakiest case) and it, too, came out
significant on 2023-2025 -- but see caveats.

## Caveats -- read before treating this as a Plan B win

- **The 0.08 tolerance was chosen after seeing the pooled 2020-2022 sweep**,
  which includes the 2022 fold later used as the nested "check." The nested
  selection-then-check step (2020-2021 select, 2022 check) is a real
  improvement over naively picking from the full pooled dev metric, but the
  tolerance value itself is not fully independent of the season later used to
  validate it. The true 2023-2025 confirmation above is the only fully clean
  test in this chain, since it was never used for any selection.
- **`rushing_attempts` was flagged as unlikely to replicate by the more
  conservative nested check, then did replicate on the real confirmation.**
  This could mean the nested check (using only 2 seasons for selection and 1
  for checking) was underpowered, or it could mean this specific 2023-2025
  result is itself the kind of pattern that would not survive a second
  independent confirmation population. With only one confirmation run
  available (this is the whole point of not re-running against 2023-2025
  repeatedly), this cannot be distinguished further from data alone.
- **Four separate hyperparameters were searched (one other_weight per
  target)**, which is more researcher degrees of freedom than a single shared
  weight, even with the nested dev/check split. The reported CIs are
  standard per-target paired bootstraps; they are not adjusted for having
  searched a weight per target.
- **This remains a capped-roster share-MAE result**, not a PPR result, and
  does not evaluate the full joint team allocation architecture (still just
  the softmax-with-omitted-bucket architecture, first cut). Roster caps
  still exclude the same populations documented in the earlier Plan B runs.
- No serving integration or artifact promotion follows from this result.

## Next step (not yet done)

If this holds up under scrutiny, the natural next steps are: (1) an
independent sanity check of the pipeline itself (rerun with a different
bootstrap seed, spot-check a handful of rows by hand) before treating the
significance as load-bearing; (2) decide whether `rushing_attempts`'s
confirmation result should be trusted given the nested check's warning; (3)
only then consider whether this justifies further Plan B investment (e.g.
extending to the PPR-level reconstruction Plan A already has).

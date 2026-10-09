# Plan A + Plan B arm on leak-free Plan A — 2026-10-09

Re-run of `data/experiments/full_ppr_with_plan_b_20260925/` after the Plan A
team-total leak was fixed (`2f602233`; GAPS.md 2026-10-08). That earlier run's
2.346 used team totals that read the predicted game's own team stats and is
void (docs/PRODUCTION_SELECTION_RULE.md, A1.3).

This is development evidence on the 2023–2025 folds. It is not a gate result
and does not change the production decision: the arm is not a candidate under
Amendment 1 and has no serving path.

## Inputs

- Allocations: `full_ppr_with_plan_b_20260925/allocations/`. Plan A's arms
  there are row-for-row identical to `full_ppr_safe_blend_event_mass_20260924/`,
  the allocations the corrected Plan A run used; the only addition is Plan B's
  `joint_other_mae` arm for rushing yards, receiving yards, receptions and
  passing yards.
- Team totals: `full_ppr_selector_no_sameweek_team_20261008/team_totals/`
  (leak-free).
- Plan B's share model uses lagged shares, lagged snap share and the as-of
  depth chart; the current week's team total is read only as a training label.

Command:

    python scripts/evaluate_joint_ppr_selector.py \
      --allocation-dir data/experiments/full_ppr_with_plan_b_20260925/allocations \
      --team-total-input-dir data/experiments/full_ppr_selector_no_sameweek_team_20261008/team_totals \
      --output data/experiments/full_ppr_with_plan_b_no_sameweek_team_20261009/joint_selector.json

## Result

Same 40,559 player-weeks and actual points as the corrected Plan A run.
Paired difference = Plan A + B minus corrected Plan A; 95% interval from a
two-way week × player bootstrap (B = 2,000, seed 42), in `paired_vs_plan_a.json`.

| Slice | Corrected Plan A | Plan A + B | Difference [95% CI] |
| --- | ---: | ---: | --- |
| Pooled 2023–2025 | 2.4262 | 2.4028 | −0.0234 [−0.0433, −0.0067] |
| 2024 fold | 2.4206 | 2.4014 | −0.0192 [−0.0641, +0.0164] |
| 2025 fold | 2.3624 | 2.3119 | −0.0505 [−0.0849, −0.0191] |
| QB (2024–25) | 2.9664 | 2.7350 | −0.2314 [−0.4096, −0.0754] |
| RB (2024–25) | 2.4644 | 2.4610 | −0.0034 [−0.0247, +0.0160] |
| WR (2024–25) | 2.4200 | 2.4213 | +0.0013 [−0.0103, +0.0145] |
| TE (2024–25) | 1.8844 | 1.8744 | −0.0100 [−0.0229, +0.0027] |

Against rolling-3 (2.4913): −0.0885 (selector's 500-draw paired bootstrap
[−0.1002, −0.0771]), versus −0.0651 for corrected Plan A and −0.1452 in the
void 09-25 run. The 2023 fold has no prior-fold evidence, so every target
falls back to rolling-3 there in both runs.

Selected allocation arms: `joint_other_mae` for rushing yards, receiving
yards and passing yards; `xgb_blend_renorm` for receptions; rolling-3 for the
TD and interception targets. Corrected Plan A used rolling-3 for passing
yards, which is where nearly all of the gain is (QB).

## Reading

- The gain is real but small, about a sixth of the void 09-25 figure, and
  concentrated in QB passing yards. RB, WR and TE are flat.
- These MAEs are on Plan A's scale, not the 2025 selection-set scale (~4.27).
  Corrected Plan A tied the live blend on 2025 (4.268 vs 4.266); a ~0.05 gain
  on the 2025 fold is unlikely to clear the rule's 0.10 minimum effect.
- Recommendation (2026-10-09, not yet decided): no general Plan B serving path for now.
  A QB passing-yards allocation is the narrower option; it would need its own
  serving path, population and leakage check, and a rule amendment before it
  could enter the forward test.

## Files

- `joint_selector.json` (sha256 `b7939c58…31d435f`), `truth_audit.json`,
  `paired_vs_plan_a.json`: tracked.
- `ppr_oof_rows.csv`, `run.log`: local only (gitignored); hashes of the row
  files are in `paired_vs_plan_a.json`.

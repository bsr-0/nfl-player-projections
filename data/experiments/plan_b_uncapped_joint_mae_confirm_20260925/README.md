# Plan B joint-MAE confirmation on the uncapped, full population — 2026-09-25

**First confirmation of Plan B's joint-softmax share model against the full
(uncapped) player population, and the first-ever confirmation of the two new
full-PPR-adjacent targets (`receptions`, `passing_yards`).** All six targets
beat rolling-3 with a paired 95% interval excluding zero on the real,
never-before-touched 2023-2025 seasons.

## Result

| Target | Held-out rows | Rolling-3 MAE | Joint MAE | Delta, paired 95% CI | Mass MAE |
| --- | ---: | ---: | ---: | --- | ---: |
| Targets | 34,932 | 0.02985 | 0.02950 | -0.00035 [-0.00066, -0.00005] | 0.00158 |
| Rushing attempts | 40,559 | 0.02394 | 0.02342 | -0.00052 [-0.00085, -0.00021] | 0.00009 |
| Receiving yards | 34,932 | 0.03751 | 0.03668 | -0.00083 [-0.00125, -0.00043] | 0.00124 |
| Rushing yards | 40,559 | 0.02881 | 0.02772 | -0.00109 [-0.00146, -0.00073] | 0.00014 |
| Receptions | 34,932 | 0.03337 | 0.03301 | -0.00036 [-0.00072, -0.00003] | 0.00171 |
| Passing yards | 5,627 | 0.12028 | 0.08924 | -0.03104 [-0.04616, -0.01680] | 0.00111 |

`other_weight=0.1` for every target (see "Weight selection" below). Every
interval excludes zero on the improvement side. All 18 seasonal fits (6
targets x 3 seasons) converged; zero rows needed the unseen-slot fallback in
any fold (the training history by 2023 already covers essentially every real
slot). `evaluate_plan_b_joint_mae.py`'s own checks passed for all six runs:
saved-row MAE recomputation matched the report, and (since no prior run of
this uncapped population exists) `--first-confirmation-run` verified the
preflight's own `coverage.json` shows `excluded_rows == 0` in place of the
usual matched-row provenance check.

## The mass-calibration trade-off has essentially disappeared

The capped-population investigation (`data/experiments/plan_b_joint_mae_confirm_20260925/`)
found a real trade-off between row-level accuracy and omitted-bucket mass
calibration, because ~38% of the population was structurally excluded by the
roster-slot cap, creating substantial mass for the model to get right. With
the cap removed (`scripts/build_team_week_roster_slots.py`, 2026-09-25 --
every rostered player gets a slot, no ceiling), there is almost nothing left
in the omitted bucket to miscalibrate. A fresh dev-only sweep on the same
frozen 2020-2022 folds
(`data/experiments/plan_b_uncapped_weight_sweep_20260925/`) confirmed this
directly: mass MAE jumps from ~0.15-0.31 at `other_weight=0` (still
diagnostic-failure territory) down to ~0.002-0.08 at `other_weight=0.01`,
then stays essentially flat (often *improving* further) all the way to
`other_weight=1` -- and unlike the capped case, **row-level MAE is now also
essentially flat across that same range** (differences in the 4th-5th
decimal place). There is no meaningful trade-off left to tune: `0.1` was
picked as one clean, comfortably-calibrated value for all six targets, not a
per-target-optimized minimum -- picking a smaller boundary-hugging weight
would have bought negligible additional row MAE at the cost of looking like
precision-fitting to sweep noise.

## Weight selection and a genuine discrepancy worth flagging

Weights were selected the same way as the capped-population investigation:
using only the 2020-2021 dev folds, then checked against the untouched 2022
fold before being trusted. That nested check flagged two targets as
questionable at any calibrated weight:

| Target | Nested check (2020-21 select, 2022 check) at a calibrated weight | Real 2023-2025 confirmation |
| --- | --- | --- |
| `targets` | Did **not** hold up -- CI crossed zero at every weight from 0.01 to 1.0; only the miscalibrated `other_weight=0` passed the check | Significant, CI [-0.00066, -0.00005] |
| `rushing_attempts` | Borderline/inconsistent from `other_weight=0.02` onward -- CI hugged zero on both sides, noise-level | Significant, CI [-0.00085, -0.00021] |

Both ended up replicating on the real holdout anyway -- the same pattern seen
in the capped-population investigation, where `rushing_attempts` was flagged
by the nested check and then replicated on real data too. With only one
confirmation run by design, there is no way to tell from this data alone
whether the nested check is simply underpowered (2 dev seasons is a small
sample for a bootstrap CI) or whether these two results are the kind of
pattern that would not survive a second independent confirmation. Flagging
this explicitly rather than treating it as a clean, uniformly-strong result.

## Population and rolling-3 baseline both shifted, not just the model

Uncapping added back ~38% of rows the earlier confirmation excluded. This
changed the **rolling-3 baseline's own MAE**, not just the candidate's:
e.g. `targets`' rolling-3 MAE dropped from 0.04457 (capped) to 0.02985
(uncapped) -- consistent with the newly-added rows being disproportionately
low-share, low-variance deep-bench players who are easier for a naive
trailing-share baseline to predict. Absolute MAE deltas versus rolling-3 are
correspondingly smaller in absolute terms for `targets`/`receiving_yards`
than in the capped run (though still significant); this is an expected
consequence of the population change, not a weakening of the effect itself
-- relative improvement should be read with this in mind rather than
comparing raw delta magnitudes across the two populations directly.

## Caveats

- **Passing_yards's large relative improvement (-26%, 0.1203 -> 0.0892) comes
  from a much smaller population (5,627 rows, QB-only)** than the other five
  targets (~35-41k rows) -- a real result, but with correspondingly wider
  bootstrap uncertainty (CI half-width ~0.015, vs. ~0.0002-0.0004 for the
  larger-population targets). Treat the point estimate with more caution
  than the others.
- `receptions`/`passing_yards` have no equal-weighting or unconstrained-
  diagnostic precedent from the capped-population era to sanity-check
  against -- this is their first-ever Plan B evaluation of any kind.
- Still a share-MAE result, not a PPR result, and does not yet evaluate the
  full joint team allocation architecture at the fantasy-points level. No
  serving integration or artifact promotion follows from this result.
- One shared `other_weight=0.1` was used for all six targets rather than a
  per-target-optimized value; given the flat frontier (see above) this is a
  deliberate choice, not an oversight -- but it means no target's number here
  reflects an exhaustively-tuned optimum.

## Next step (not yet done)

Wire this confirmed allocation into Plan A's existing arm-agnostic full-PPR
selector (`src/evaluation/joint_ppr_selector.py`) as a new candidate arm for
the four points-eligible targets it now covers (`rushing_yards`,
`receiving_yards`, `receptions`, `passing_yards`), and measure whether the
share-level win here transfers to a full-PPR MAE improvement over the
existing 2.36984 baseline.

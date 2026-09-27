# Plan B: first-cut share evaluation and reproduction

Updated 2026-09-25. The user reopened Plan B for future evaluation. This
supersedes the older document's instruction to stop all Plan B evaluator work.

## Joint allocation development update

A second research arm groups each team's represented players in a softmax and
adds an explicit bucket for omitted players. It uses only lagged player features
and trains on past seasons. The frozen 2020–2022 development folds, with
20,774 receiving rows and 23,970 rushing rows, were evaluated against rolling-3
on identical keys. The fractional cross-entropy fit was worse on all four
targets. A smooth player MAE objective improved row MAE, but assigned too much
team share to the omitted-player bucket (receiving-yards represented mass
0.676 predicted versus 0.937 recorded). That run is a diagnostic failure, not
a usable candidate.

Adding an equally weighted mean absolute error for the omitted-player mass
corrected that failure mode but removed the row-level gain. The final
mass-aware development comparison is (superseded by the weighted follow-up
below; kept for the record):

| Share target | Rolling-3 MAE | Joint MAE | Joint minus rolling-3, 95% paired week interval |
| --- | ---: | ---: | ---: |
| Targets | 0.045542 | 0.046776 | +0.001234 [+0.000661, +0.001764] |
| Rushing attempts | 0.038140 | 0.039331 | +0.001191 [+0.000585, +0.001825] |
| Receiving yards | 0.057619 | 0.058013 | +0.000395 [−0.000211, +0.000980] |
| Rushing yards | 0.045159 | 0.045582 | +0.000423 [−0.000206, +0.001076] |

The paired interval excludes zero on the harmful side for targets and rushing
attempts; the other two comparisons are inconclusive. Saved row-level MAEs
were independently rechecked by the evaluator. The exact inputs are the four
preflight panels below, with hash checks. Development outputs are in
`data/experiments/plan_b_joint_other_dev_20260925/`,
`data/experiments/plan_b_joint_mae_dev_20260925/` (unconstrained diagnostic),
and `data/experiments/plan_b_joint_mae_mass_dev_20260925/` (mass-aware, equal
weighting). The frozen input panels remain local; the smaller row-level
prediction CSVs, manifests, and reports are tracked. At the time this equal
weighting was tried, the 2023–2025 confirmation was not run because the
development gate failed, and no Plan B model was selected, promoted, or used
in serving. These remain share metrics on capped rosters, not PPR metrics or a
comparison to the full Plan A population.

## Joint allocation weighted development and confirmation update — 2026-09-25 (later)

The equal weighting above (mean player-row absolute error plus an equally
weighted mean omitted-bucket absolute error) is one point on a continuous
trade-off, not the only option: `JointOtherMAEModel` (`src/models/team_hierarchical/joint_other_mae.py`)
now takes an `other_weight` parameter that scales the omitted-bucket term
relative to the player-row term (`other_weight=1.0` reproduces the equal
weighting above bit-for-bit; `other_weight=0.0` reproduces the unconstrained
diagnostic-failure objective). A sweep on the same frozen 2020–2022 dev folds
(`data/experiments/plan_b_joint_weight_sweep_20260925/`, `_resweep_`,
`_finesweep_`, weights from 0.0 to 1.0) showed this is a smooth frontier with
no universal elbow: lower weight always improves row MAE and always worsens
mass calibration. Two of four targets (`targets`, `receiving_yards`) show
diminishing mass-calibration returns above `other_weight≈0.03–0.05`; the
other two (`rushing_attempts`, `rushing_yards`) do not plateau before
`other_weight=1`.

Because there is no natural elbow, a weight must be chosen against a
predeclared tolerance. Picking the smallest weight meeting a mean team-week
mass-MAE tolerance of 0.08 directly from the pooled 2020–2022 dev metric would
conflate hyperparameter selection with the data used to report significance.
Instead, weights were selected using only the 2020–2021 dev folds, then
checked against the untouched 2022 fold before being trusted:

| Target | Selected `other_weight` (2020–21 only) | Holds up on untouched 2022? |
| --- | ---: | --- |
| targets | 0.01 | yes — CI [-0.00195, -0.00087] |
| receiving_yards | 0.03 | yes — CI [-0.00289, -0.00157] |
| rushing_yards | 0.10 | yes — CI [-0.00169, -0.00030] |
| rushing_attempts | 0.05 | no — CI [-0.00095, +0.00008], crosses zero |

Only after that nested check were these four weights carried into one real,
first-ever 2023–2025 confirmation run
(`data/experiments/plan_b_joint_mae_confirm_20260925/`, see its `README.md`
for full methodology and caveats). **Result: all four targets beat rolling-3
with a paired 95% interval excluding zero on the real confirmation seasons —
the first such result in the entire Plan B investigation:**

| Target | `other_weight` | Held-out rows | Rolling-3 MAE | Joint MAE | Delta, paired 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| Targets | 0.01 | 21,211 | 0.04457 | 0.04327 | -0.00130 [-0.00168, -0.00094] |
| Rushing attempts | 0.05 | 24,475 | 0.03555 | 0.03488 | -0.00067 [-0.00112, -0.00027] |
| Receiving yards | 0.03 | 21,211 | 0.05705 | 0.05502 | -0.00203 [-0.00253, -0.00153] |
| Rushing yards | 0.10 | 24,475 | 0.04330 | 0.04213 | -0.00117 [-0.00166, -0.00065] |

Mass calibration on the confirmation seasons (mean actual vs. candidate
represented mass, mean absolute team-week mass error) is in the same range
this repo previously treated as corrected: Targets 0.9466→0.8995 (0.0704),
Rushing attempts 0.9511→0.9605 (0.0528), Receiving yards 0.9502→0.9263
(0.0623), Rushing yards 0.9514→0.9725 (0.0512) — no target is near the
~0.13–0.27 diagnostic-failure level the unconstrained objective produced.
`evaluate_plan_b_joint_mae.py`'s own checks passed for all four runs: saved-row
MAE recomputation matched the report, and `--prior-run-predictions` confirmed
identical held-out keys, labels, and rolling-3 baseline against the earlier
mixed-effects confirmation run.

**Read before treating this as a resolved Plan B win, not just a result:**

- The 0.08 tolerance was chosen after seeing the pooled 2020–2022 sweep,
  which includes the 2022 fold later used as the nested check — that check
  is a real improvement over selecting straight from the pooled dev metric,
  but the tolerance itself is not fully independent of the season later used
  to validate it. The 2023–2025 run is the only fully clean test in this
  chain, since it was never used for any selection.
- `rushing_attempts` was flagged as unlikely to replicate by the more
  conservative nested check (2 dev seasons to select, 1 to check), then did
  replicate on the real confirmation. This could mean the nested check was
  underpowered, or that this particular result would not survive a second
  independent confirmation population — undecidable from this data alone,
  and there is intentionally no second confirmation run to check further.
- Four separate hyperparameters were searched (one `other_weight` per
  target), more researcher degrees of freedom than a single shared weight;
  reported CIs are standard per-target paired bootstraps, not adjusted for
  a per-target search.
- Still a capped-roster share-MAE result on the same softmax-with-omitted-
  bucket architecture as the equal-weighting run above — not a PPR result,
  and not an evaluation of the full joint team allocation architecture. No
  serving integration or artifact promotion follows from this result. A
  future attempt should independently sanity-check this pipeline (different
  bootstrap seed, hand-checked rows) and decide whether `rushing_attempts`'s
  confirmation result is trustworthy given the nested check's warning, before
  relying on it further.

## Uncapped-population confirmation and two new targets — 2026-09-25 (still later)

The capped-roster caveat throughout this document ("materially smaller
populations than Plan A's full PPR population") is now addressed for Plan
B's joint architecture. `scripts/build_team_week_roster_slots.py`'s
`MAX_SLOTS_PER_POSITION` cap (`{QB:2, RB:4, WR:6, TE:3}`, excluding ~38% of
otherwise-eligible rows) has been removed entirely — every rostered player
at a position now gets a slot, with no ceiling (e.g. `WR1`..`WR23` in the
real data). Rebuilding the real roster-slot snapshot against 2013–2025
confirmed zero rows are now excluded for any of the four original targets
(`data/experiments/plan_b_uncapped_inputs_20260925/`,
`data/experiments/plan_b_uncapped_preflight_20260925/`).

Two real defects in the joint-softmax model were found and fixed while
preparing for this (not deferred, per this repo's standing rule): (1)
`joint_other.py`'s `_prepare()` had no unseen-slot fallback — a slot absent
from the training vocabulary silently got an all-zero one-hot row,
equivalent to whichever slot is lexicographically first overall (e.g. a
never-before-seen `WR9` row would have silently scored identically to a
`QB1` row), with no diagnostic. Fixed to fall back explicitly to the row's
own position's rank-1 slot (mirroring the mixed-effects model's existing
pattern), counted in a new `prediction_diagnostics_["fallback_slot_rows"]`.
Uncapping made this defect far more consequential (rare tail slots routinely
absent from shorter training windows), so it could not be deferred. (2)
`team_hierarchical/features.py::feature_columns()` was missing
`include_full_ppr=True`, which would have silently stripped the two new
targets' lagged history columns and hard-failed the feature audit the
moment they were used.

The joint architecture now also covers `receptions` (QB-excluded) and
`passing_yards` (QB-only) — both continuous, non-sparse shares,
architecturally identical in shape to the original 4. The 4 sparse
zero-inflated TD/INT targets remain explicitly out of scope: Plan A's own
specialized architecture for those (`src/models/team_allocation/hierarchical.py`)
still mostly loses to rolling-3, and extending a continuous-softmax model to
zero-inflated counts is a different modeling problem, not a scaling step.

A fresh dev-only `other_weight` sweep on the same frozen 2020–2022 folds
(`data/experiments/plan_b_uncapped_weight_sweep_20260925/`) found that
uncapping essentially erased the row-MAE/mass-calibration trade-off that
drove the weighting work above: mass MAE still jumps sharply between
`other_weight=0` and `0.01`, but from `0.01` to `1.0` both mass calibration
and row MAE are essentially flat for every target — there is very little
omitted-bucket mass left to trade off once ~38% of the population is no
longer structurally excluded. `other_weight=0.1` was used for all six
targets as one clean, comfortably-calibrated value, not a per-target-tuned
minimum.

**Result: the real 2023–2025 confirmation
(`data/experiments/plan_b_uncapped_joint_mae_confirm_20260925/`, see its
`README.md`) beat rolling-3 with a paired 95% interval excluding zero on all
six targets**, including the two confirmed for the first time ever:

| Target | Held-out rows | Rolling-3 MAE | Joint MAE | Delta, paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| Targets | 34,932 | 0.02985 | 0.02950 | −0.00035 [−0.00066, −0.00005] |
| Rushing attempts | 40,559 | 0.02394 | 0.02342 | −0.00052 [−0.00085, −0.00021] |
| Receiving yards | 34,932 | 0.03751 | 0.03668 | −0.00083 [−0.00125, −0.00043] |
| Rushing yards | 40,559 | 0.02881 | 0.02772 | −0.00109 [−0.00146, −0.00073] |
| Receptions | 34,932 | 0.03337 | 0.03301 | −0.00036 [−0.00072, −0.00003] |
| Passing yards | 5,627 | 0.12028 | 0.08924 | −0.03104 [−0.04616, −0.01680] |

A genuine discrepancy worth flagging rather than smoothing over: the same
nested selection-then-check discipline used above (select on 2020–2021,
verify on untouched 2022) found that `targets` did **not** hold up at any
calibrated weight, and `rushing_attempts` was borderline/noise-level from
`other_weight=0.02` onward — yet both replicated on the real confirmation
anyway, the same pattern `rushing_attempts` showed in the capped-population
investigation. With only one confirmation run by design, whether the nested
check is underpowered or these are results that would not survive a second
independent population is undecidable from this data alone.

Population change also shifted the baseline itself, not just the candidate:
e.g. `targets`' rolling-3 MAE dropped from 0.04457 (capped) to 0.02985
(uncapped), since the newly-restored rows are disproportionately low-share,
easier-to-predict deep-bench players — read absolute deltas across the two
populations with that in mind, not as a direct before/after comparison of
the same quantity. `passing_yards`'s large relative improvement (−26%) comes
from a much smaller QB-only population (5,627 rows) with correspondingly
wider bootstrap uncertainty than the other five targets.

Still a share-MAE result on the capped-vs-uncapped softmax-with-omitted-
bucket architecture, not yet a PPR result — the natural next step is wiring
this confirmed allocation into Plan A's existing arm-agnostic full-PPR
selector (`src/evaluation/joint_ppr_selector.py`) for the four points-
eligible targets it now covers, to see whether this transfers to a full-PPR
MAE improvement. No serving integration or artifact promotion follows from
this result.

## Wired into Plan A's full-PPR selector — 2026-09-25 (final, this pass)

**The uncapped confirmation above transfers to the fantasy-points level:
added as a new candidate arm to Plan A's existing gated full-PPR selector,
it further reduces full-PPR MAE from 2.36984 to 2.34608** on the same
40,559-row, three-fold (2023/2024/2025) real holdout Plan A's own guarded
validation used — paired bootstrap point estimate −0.14522, 95% CI
[−0.15977, −0.12922] versus the all-rolling3 baseline, `significant_improvement:
true`. Full result, methodology, and verification steps in
`data/experiments/full_ppr_with_plan_b_20260925/README.md`.

This required no changes to already-verified Plan A code
(`joint_ppr_selector.py`, `ppr_truth.py`, `full_ppr_allocation_backtester.py`
are all unmodified): a new `src/evaluation/plan_b_arm_adapter.py` reshapes
Plan B's wide predictions into Plan A's long arm schema — deriving fold
numbers by literally instantiating `SeasonAwareTimeSeriesSplit` with Plan
A's real config rather than hand-derived arithmetic, confirmed to produce
exactly `{2023: 0, 2024: 1, 2025: 2}` — and validates the population and
labels match before concatenating. `scripts/build_full_ppr_allocations_with_plan_b.py`
writes a merged 8-target allocation directory (Plan B's arm attached for
its 4 covered targets, the 4 sparse targets copied through unchanged),
which is then fed to the existing, unmodified selector script.

The selector chose Plan B's arm for **3 of its 4 covered targets**
(`rushing_yards`, `receiving_yards`, `passing_yards`); `receptions` still
preferred Plan A's existing `xgb_blend_renorm` — Plan B's confirmed
share-level win for that target did not translate into the best full-PPR
choice, likely because the selector optimizes reconstructed PPR jointly
with the chosen team-total arm, not share MAE in isolation. Independently
recomputed pooled MAE from the saved row-level CSV, and a fresh
`checked_truth()` hash check, both matched the reported numbers exactly;
`input_audit` confirmed the new arm was genuinely validated (not silently
skipped) for all 4 targets. All of the caveats from the uncapped-population
confirmation above carry forward unchanged (population/baseline shift
versus the capped run, `targets`/`rushing_attempts`'s nested-check
discrepancy, `passing_yards`'s small QB-only sample) — `targets`/
`rushing_attempts` are not points-eligible, so don't appear in this
full-PPR step at all. No serving integration or artifact promotion follows
from this result.

## Status and scope

The mixed-effects baseline was evaluated on 2023–2025 held-out share rows using
the audited 2013–2025 roster snapshot at
`data/experiments/plan_b_inputs_20260925/slots.csv`. All four preflights and
all 12 seasonal fits completed. **Mixed effects scored worse than rolling-3
and fixed ridge on every target; no improvement is established.** The
SQLite roster table remains absent; the evaluator reads the snapshot CSV. The earlier failed
preflight without roster input is preserved at
`data/experiments/plan_b_readiness_20260925/failure.json`.

Full results, row-level predictions, fold diagnostics, and independent saved-row
verification are under `data/experiments/plan_b_run_20260925/`; start with its
`README.md` and `summary.json`. This first cut remains a player random-intercept
model, not the proposed full joint team architecture or a PPR model.

The snapshot has 101,138 rows from 163,358 canonical player-weeks. Roster caps
exclude 62,220 rows (38.1%). `targets` and `receiving_yards` exclude QB rows
by design, leaving 87,510 represented rows each. The held-out 2023/2024/2025
rows are 8,155/8,160/8,160 for the two rushing targets and
7,067/7,072/7,072 for the two receiving targets. These populations are
materially smaller than Plan A's full PPR population. See
`data/experiments/plan_b_inputs_20260925/manifest.json` for source-key and cap
checks, and `data/experiments/plan_b_preflight_20260925/summary.json` for all
four preflight results and input hashes.
The database file changed when the separate A/B jobs finished. A read-only
rebuild of slots and the complete represented rushing-yards input panel from
the current database reproduced their saved SHA-256 hashes exactly; see
`data/experiments/plan_b_inputs_20260925/reverification.json`.

This first experiment covers `targets`, `rushing_attempts`, `receiving_yards`,
and `rushing_yards`. Its metric is **player-week share MAE**, not fantasy-point
MAE. Each fit has roster-slot fixed effects, lagged numeric features, and a
player random intercept. It does not yet implement the proposed joint roster
architecture or enforce a team sum constraint. It cannot establish that Plan B
beats Plan A's full eight-component PPR result.

## Correctness changes

- Join on exact player/season/week/team identity. Reject duplicate player-week
  rows, duplicate team slots, invalid ranks, stale unmatched slot identities,
  and position/rank disagreements. Preserve both the panel and roster source
  of season-to-date snap share under distinct names.
- Reuse the expanded share panel's raw-stat exclusions and explicitly reject
  known leakage columns. Audit the entire schema before choosing features.
- Retain training rows with missing lagged features. Fill numeric values with
  training medians (zero only for entirely missing training columns); reuse
  those medians for held-out predictions. Reject infinite features and
  missing/nonfinite labels.
- Remove fixed-effect columns that are redundant with slot indicators or
  other numeric columns using training-only rank-revealing QR. In particular,
  numeric slot rank cannot add an independent effect alongside slot categories.
- Require a usable converged mixed-effects fit and finite random effects.
  Record optimizer warnings and retries. Try L-BFGS, then Powell only if the
  first fit is unusable; held-out accuracy never selects the optimizer.
  Failed folds stop the research run, with no pooled success report. The model
  still offers an explicitly requested mean fallback for other callers; this
  evaluator never enables it.
- Apply known players' fitted random intercepts explicitly. Unseen players use
  fixed effects. Unseen slots use the same-position rank-1 slot if available;
  otherwise the fixed component uses the training mean. These rows are counted.

The implementation follows statsmodels' documented
[fixed-effect prediction semantics](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.predict.html)
and [optimizer interface](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html).

## Predeclared comparison

For each target, evaluate three arms on exactly the same represented rows:

| Arm | Fit and prediction |
| --- | --- |
| Rolling-3 | Existing lagged share; missing values become zero and are counted |
| Fixed ridge | Slot one-hot encoding plus the numeric features below; training median imputation and scaling; alpha 1 |
| Mixed effects | Slot fixed effects plus the same numeric features; player random intercept; REML |

Features are fixed before the run: slot, depth-chart rank, roster season-to-date
snap share, cold-start flag, and the target's rolling-3 and season-to-date share
and team total. No hyperparameter sweep or arm selection uses evaluation rows.
Ridge and mixed-effects predictions are clipped to [0, 1]; unbounded predictions
and clipping counts are saved. No arm is renormalized across the represented
roster. Labels retain their original full-team denominator, including when
roster caps exclude teammates.

Use expanding **past-season-only** training: for test season S, fit only seasons
less than S. Within a held-out season, each week's lagged observed features are
available; models are not refit using that season's labels. The default tests
are 2023, 2024, and 2025, with training history starting in 2013. These are reused
retrospective seasons, not a fresh final holdout.

Report pooled, seasonal, position, and unseen-player share MAE; exact roster
exclusions; team-level represented share mass and mass errors; and paired 95%
intervals for mixed-effects minus each control. Bootstrap whole calendar weeks
within each season with identical paired draws, weighted by player-week counts.
These intervals do not address dependence across weeks or multiple-target
selection. Do not select the best target and describe its interval as a
prespecified overall Plan B result.

## Reproduction

The roster CSV, four preflights, and four evaluations below were completed on
2026-09-25. Use a new experiment directory for any repeat; the evaluator
refuses to overwrite an existing directory.

### 1. Export and inspect the roster snapshot

The snapshot was built with the existing builder's read-only CSV mode:

```bash
python scripts/build_team_week_roster_slots.py \
  --seasons 2013 2025 --dry-run \
  --csv data/experiments/plan_b_inputs_20260925/slots.csv \
  --audit-csv data/experiments/plan_b_inputs_20260925/slot_coverage.csv \
  --audit-by-season-csv data/experiments/plan_b_inputs_20260925/slot_coverage_by_season.csv
```

The saved slot CSV SHA-256 is
`24ab8997d0b1ebd5ab716d3d752e93af7592fbe495f38914e307b66cce1f07d8`.
The independent input audit checked that the canonical and share keys match;
saved slots have unique player-week and team-slot keys, exact source identities,
and contiguous ranks through each position cap. The by-season coverage file
shows notably higher default-depth rates for 2025 WR and TE rows (29.3% and
27.9% respectively). Inspect these and the large exclusion count before
interpreting model results.
The loader verifies identities, not the historical provenance of arbitrary
caller-supplied slot values; use this audited builder against the same database
snapshot. Its source inputs must already follow pregame availability rules.
Unexpected team/identity mismatches require investigating the input lineage.

### 2. Preflight without fitting

```bash
python scripts/evaluate_plan_b_shares.py \
  --target rushing_yards \
  --slots-csv data/experiments/plan_b_inputs_20260925/slots.csv \
  --output-dir data/experiments/plan_b_preflight_next/rushing_yards
```

This preflight passed for all four targets under
`data/experiments/plan_b_preflight_20260925/`. The command above illustrates
a repeat in a new output directory. A successful preflight means the data
contract passes; it does not guarantee numerical convergence. Missing roster
input is a prerequisite failure, never permission to score a different
population.

### 3. Run the comparison explicitly

```bash
python scripts/evaluate_plan_b_shares.py \
  --target rushing_yards \
  --slots-csv data/experiments/plan_b_inputs_20260925/slots.csv \
  --output-dir data/experiments/plan_b_run_next/rushing_yards \
  --preflight-manifest data/experiments/plan_b_preflight_20260925/rushing_yards/manifest.json \
  --run
```

Run targets sequentially. Every target uses its own new output directory. A
failed run writes `failure.json`; inspect the fit diagnostics before changing
settings. Parameter changes define a new experiment and should retain the
failed attempt. No silent dropping of failed seasons is allowed.
The run requires the matching preflight manifest and rejects any changed slot
CSV or joined player panel before fitting.

## Outputs and acceptance gate

The CLI opens SQLite in a read-only transaction and writes only its new output
directory. Outputs include the exact joined input CSV, source/feature-code
hashes, supplied slot CSV hash, dependency versions, fold populations and
cutoffs, coverage and excluded keys, model diagnostics, row predictions, and
metrics. It independently recomputes pooled MAEs from saved row predictions
before publishing `report.json`. The database file itself is not hashed while
other jobs may write it; the queried joined snapshot is saved and hashed.

Before deciding the next architecture:

1. Confirm all declared folds and all eligible represented test keys survived.
   Review excluded keys separately; a capped roster score is not a full-panel
   score. All three arms must use the identical keys.
2. Review convergence, warnings, rank reduction, missing-feature counts, unseen
   slots/players, clipping, and implausible team sums. Finite output alone is not
   sufficient evidence of a useful fit.
3. Require a negative candidate-minus-control delta with an interval excluding
   zero to claim measured improvement for that declared comparison. Report
   negative or inconclusive results and all four targets. Check whether any
   gain holds across seasons and is useful for unseen players.
4. If player random effects add no value over ridge, do not attribute any gain
   over rolling-3 to hierarchical modeling. If useful signal is present, next
   investigate a genuinely joint allocation with explicit omitted-player mass.
   Full PPR evaluation then requires the remaining component models, team-total
   forecast integration, common raw truth, and identical Plan A comparison rows.

No serving integration or artifact promotion is part of this run.

## Verification

```bash
python -m pytest tests/test_team_hierarchical_features.py \
  tests/test_team_hierarchical_models.py \
  tests/test_team_hierarchical_backtester.py -q
```

Tests include exact team/slot joins, explicit capped-row coverage, raw-stat
exclusion, missing-value handling without row drops, redundant columns, fit
failure visibility, known/unseen player behavior, future-label independence,
identical arm populations, saved-row metric recomputation, input immutability,
and the preflight-only default. Synthetic tests establish contracts; they are
not evidence of real-data model quality.

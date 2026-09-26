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
mass-aware development comparison is:

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
and `data/experiments/plan_b_joint_mae_mass_dev_20260925/` (mass-aware).
The large frozen panels and prediction CSVs remain local; their manifests and
reports identify the inputs and code. The 2023–2025 confirmation was not run
because the development gate failed. No Plan B model was selected, promoted,
or used in serving. A future attempt should first address omitted-player mass
calibration and sparse zero rows on development seasons, then use a separate
predeclared confirmation population. These are share metrics on capped rosters,
not PPR metrics or a comparison to the full Plan A population.

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

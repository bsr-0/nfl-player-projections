# Pre-registration v2 — Phase 7 history-NaN arm, 2026-08-27

Replaces v1 (`superseded/PRE_REGISTRATION_v1_dose_stratification.md`), whose
primary analysis rested on a stratum that cannot exist. Written and committed
before the run it governs.

## Why v1 was withdrawn

v1 stratified player-seasons by **dose** — how much of a season's features the
flag changed — and used the zero-dose stratum as a falsification check: rows
the flag does not touch must not move.

**No such stratum exists.** `PRESERVE_HISTORY_MISSINGNESS` is applied inside
`add_advanced_features`, which `_prepare_training_data` runs on the TRAINING
matrix as well as the scored rows. Each arm therefore fits a *different model*,
and every prediction differs whether or not that row's own features changed.

Measured on 2015, with dose computed correctly from the scored synthetic rows:

- 8,368 of 8,608 scored feature rows (97.2%) are byte-identical between arms
- 523 of 538 player-seasons are zero-dose by those features
- **0 of 538 players received the same prediction**

So the row-level channel is small and the model-level channel dominates. A
per-row stratification cannot separate them; both v1 failures (+0.371 on 202
rows, −0.919 on 523) were this same fact, not two different bugs.

## Hypothesis

Preserving genuine debut-week history NaN improves season-total projection
accuracy relative to filling it. The treatment is **global**: it changes the
training data and therefore the model, not merely individual scored rows.

## Arms

| arm | flag |
|-----|------|
| OFF (control) | `NFL_PRESERVE_HISTORY_MISSINGNESS=0` |
| ON (treatment) | `NFL_PRESERVE_HISTORY_MISSINGNESS=1` |

Identical otherwise: same commit, same FINAL_CONFIG, `--cold-start
--preseason-mode`, folds 2015–2025 (11), interleaved so each season's pair
completes together.

## Primary analysis — paired, per fold

The same player-seasons appear in both arms, so the comparison is paired.
For each player-season *i* in fold *f*:

    d_i = |pred_ON_i − actual_i| − |pred_OFF_i − actual_i|

Report mean `d` per fold, then across the 11 folds report the mean, the
paired standard error, and the number of folds favouring ON.

**Primary outcome**: mean paired ΔMAE across folds. Negative favours ON.

## Decision rule — fixed in advance

Adopt `PRESERVE_HISTORY_MISSINGNESS=1` as default only if BOTH:

1. ON favoured in **≥ 8 of 11 folds** (sign consistency), and
2. mean paired ΔMAE ≤ −0.25 points, i.e. beyond ~2 paired standard errors
   given the fold-level spread seen in comparable runs.

A result that meets neither is a null. A result meeting one but not the other
is reported as inconclusive and the flag stays OFF. Per-player variance is
large (2015: sd 10.8, range −35.6 to +44.1), so a small mean difference on 11
folds is not evidence of anything.

## Falsification — placebo pair, replacing the null stratum

Since no null stratum exists, the noise floor is established by a **placebo**:
run the OFF arm twice on the same season and require the two runs to agree
**exactly**.

- Already measured once: 463/463 rows bit-identical, max |diff| 0.000000000.
- Re-run as the first step of this experiment and gated on.

Requirement: placebo ΔMAE **exactly 0**. This is falsifiable — nondeterminism,
uncontrolled seeds, or ordering effects would all break it — and if it fails,
every ON−OFF difference is uninterpretable and the run is void.

This is a weaker guarantee than a null stratum would have been. A placebo
bounds run-to-run noise; it cannot isolate the flag's row-level channel from
its model-level channel. **That separation is not available in this design**
and is not claimed.

## Secondary analyses

1. **Rookie vs veteran**, descriptive only. Reported as paired ΔMAE within
   each group. NOT a dose proxy: rookies are not "high dose", they are a
   subgroup of a globally-treated population.
2. **Per-position**, exploratory. 11 folds leaves too few rookie seasons per
   position (QB: 109 total) for anything confirmatory.
3. **Mechanism note**: share of scored rows whose features are identical
   between arms, reported per fold. This quantifies how much of the effect can
   possibly be row-level. On 2015 it was 97.2% identical, implying the effect
   is almost entirely model-level.

## Known limitations, stated in advance

- The design cannot attribute the effect to debut-week NaN specifically, only
  to the flag as a whole. The flag also changes `prev_season_ppg` for
  non-debut players and two team-context rolling features.
- 11 folds, paired. Fold-level MAE sd in the comparable Phase 3 run was
  0.05–0.23 per position.
- `existing_methodology` is not part of this comparison.
- Both arms use FINAL_CONFIG as re-validated 2026-08-26 and left unchanged.

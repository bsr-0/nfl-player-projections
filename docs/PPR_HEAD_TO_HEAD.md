# Plan A and served-model comparison

The comparison uses one raw eight-component PPR label and the same target-game
population for both predictions. Its output is retrospective research evidence;
passing this comparison alone does not promote an artifact.

## Input contract

Each completed arm has a CSV or Parquet file and a JSON manifest. Identity columns are
`player_id, season, week, team, position`. They identify the **target game**, not
the row from which a forecast was made. There must be exactly one row per player,
season, week; differing teams or positions cannot disguise duplicate games.
Keys, predictions, and recorded actuals cannot be missing or nonfinite. Player
IDs are strings. The manifest names the prediction and actual columns.

The truth CSV contains those identities and all eight raw recorded components:
rushing yards, receiving yards, receptions, receiving TDs, rushing TDs, passing
yards, passing TDs, and interceptions. It comes from the checked canonical
`team_week_player_shares` panel. The scorer reuses `ppr_truth.checked_truth` and
the existing scoring helper. Negative yards and off-position stats are retained.
The panel's zero for a missing weekly stats row is already represented upstream;
the comparator never fills missing truth or predictions with zero.

Every arm's recorded actual must agree with the independently calculated raw
truth. The manifest's scoring definition **and all eight weights** must agree
with the evaluator. Fumbles lost and two-point conversions are excluded.

### Production export requirements

The current `train.py` backtest selects `target_1w` as its actual value.
`feature_preparation._create_horizon_targets` constructs this by shifting the
next observed player-season row. Consequently the existing `week` is a forecast
origin, and `week + 1` is not a safe target key across byes or missing rows.
Preserve the next observed row's target identity **when constructing the
target**, before filtering evaluation rows; use its team and position too.
Do not infer target identity from the scored subset or by matching point values.
Optional `origin_season, origin_week` columns are checked to precede the target.

Also inspect the actual prediction target. A direct prediction of full PPR cannot
be renamed to eight-component PPR. It needs a verified prediction for these eight
components, or a separately specified comparison using a common full-PPR target.
Subtracting observed fumbles or two-point points from a prediction would use the
outcome and is invalid. Do not change the manifest simply to make an export pass.

The existing `backtest_2025_20260924.json` contains aggregate metrics, no complete
row-level predictions. The running pre/post-fix walk-forward processes were
started before row-export support existed; their completion logs or per-fold
metrics alone cannot supply this comparison. New full-precision fold metrics
remain useful for that A/B, but cannot recover player-level predictions.

The newly added `src/models/oof_capture.py` writes
`data/experiments/walk_forward_oof_predictions.parquet` on future runs. At the
time of this review it captures the original row's week/team beside the shifted
`actual_for_backtest`; that export still requires verified target-game mapping
and a scoring-compatible prediction before it can enter this comparison.

The producer must emit a completed row file, then a manifest with its SHA-256. A
minimal manifest (replace placeholders with actual producer evidence) is:

```json
{
  "schema_version": 1,
  "label": "served_model",
  "rows_file": "production_rows.csv",
  "rows_sha256": "SHA256_OF_COMPLETED_CSV",
  "prediction_column": "predicted_ppr",
  "actual_column": "actual_ppr",
  "key_semantics": "target_game",
  "scoring_definition": "raw_eight_component_ppr_excludes_fumbles_and_two_point_conversions",
  "scoring_weights": {
    "rushing_yards": 0.1, "receiving_yards": 0.1, "receptions": 1,
    "receiving_tds": 6, "rushing_tds": 6, "passing_yards": 0.04,
    "passing_tds": 4, "interceptions": -2
  },
  "folds": [
    {"test_season": 2025, "trained_through_season": 2024, "selection_through_season": 2024}
  ],
  "provenance": {
    "code_commit": "ACTUAL_PRODUCER_COMMIT",
    "artifact_hashes": "ACTUAL_MODEL_HASHES",
    "target_identity_source": "DESCRIBE_VERIFIED_TARGET_MAPPING",
    "prediction_target": "DESCRIBE_VERIFIED_EIGHT_COMPONENT_OUTPUT"
  }
}
```

Fold cutoffs are upper bounds on seasons used to fit/tune or select an arm.
Both must precede the held-out season; `selection_through_season` is null when
no labels were used to choose the arm. Manifests are producer declarations, not
proof of training lineage. The report preserves them for review; it does not
deserialize model files or certify the declarations automatically.

## Population and statistical procedure

The default requires exact candidate/baseline key equality and fails on any
missing or extra key. A different declared evaluation population can be provided
with `--population-keys`; both exports must contain every requested key. Choose
and freeze this population using eligibility rules before examining errors.
There is no automatic intersection. All excluded prediction keys are saved, with
counts by season and position. Raw truth may contain additional games, whose
count is also recorded. Inputs are validated before restricting the population.

The primary result is pooled player-week MAE and candidate-minus-baseline MAE.
The 95% percentile interval uses 2,000 paired bootstrap draws, seed 42, resampling
whole calendar weeks **within each season**. Each draw sums absolute-error
differences and row counts, preserving the pooled row-weighted estimand despite
unequal weekly populations. A season with fewer than two weeks produces no
interval. A negative upper endpoint supports improvement on these rows.

This preserves within-week dependence but does not solve serial dependence
across weeks, repeated-player effects across weeks, or shared training data.
Per-season, position, season/position, negative-yardage and off-position-stat
intervals are exploratory, with no multiple-comparison correction. The original
Plan A report used an individual-row paired bootstrap; its interval is retained
in that historical report. The new week-block interval is explicitly a different
uncertainty calculation. Reusing 2023–2025 after research is not a new untouched
holdout, even when each model's training/selection is season-forward.

## Commands

Prepare Plan A inputs by verifying all 16 saved source hashes, raw truth lineage,
fold populations, and saved row-level MAEs. The SQLite connection is read-only:

```sh
python scripts/prepare_plan_a_comparison.py \
  --output-dir data/experiments/ppr_head_to_head_20260924/inputs
```

Verify the comparator against Plan A versus rolling-3:

```sh
python scripts/compare_ppr_predictions.py \
  --baseline-manifest data/experiments/ppr_head_to_head_20260924/inputs/rolling3.manifest.json \
  --candidate-manifest data/experiments/ppr_head_to_head_20260924/inputs/plan_a.manifest.json \
  --truth data/experiments/ppr_head_to_head_20260924/inputs/raw_truth.csv \
  --output-dir data/experiments/ppr_head_to_head_20260924/rolling3_verification
```

After a compatible production export exists, use its manifest as baseline and
Plan A as candidate. Supply a frozen `--population-keys` file for any narrower
population (for example 2025 only). The production run's eligible games need not
equal the Plan A canonical panel, so publish exclusion counts with the result.

Both commands refuse existing output directories. Source files and the database
are never written. Reports include input hashes, counts, fold declarations, and
the hash of saved paired rows. MAE is independently recomputed from those saved
rows before `report.json` is published as the completion marker. Failed validation
does not publish a comparison. No training, serving changes, or promotion occur.

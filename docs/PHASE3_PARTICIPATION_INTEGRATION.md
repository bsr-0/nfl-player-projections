# Phase 3: Does Phase 2 improve weekly PPR?

Phase 3 is the only phase allowed to test Phase 2 against weekly PPR. It does
not alter production configuration or serving predictions.

## Input contract

The input is Phase 2's `oof_predictions.csv`, selecting its `hist_gbm` rows.
Every row must include player/week identity plus `phase2_test_season` and
`phase2_train_max_season`. The harness rejects a row unless:

- its Phase 2 model trained before its player-week season;
- its recorded test season equals that row's season;
- probability and expected snap share are finite values in `[0, 1]`;
- selected-model player-week keys are unique.

## Matched comparison

Phase 2 cannot produce OOF predictions before its snap-observed era. Therefore
this is not a full-PPR-history comparison. For each Phase 7 position/fold, the
harness inner-joins PPR rows to valid Phase 2 OOF rows and uses that exact
matched train and test population in every arm. It never imputes missing OOF
predictions or compares a full-data baseline to a restricted treatment.

| Arm | Added feature(s) |
|---|---|
| `baseline` | none |
| `p_only` | predicted meaningful-participation probability |
| `p_plus_expected` | probability + predicted expected snap share |

Each arm refits the existing Phase 7 `FINAL_CONFIG` architecture/window and
recency weighting. Phase 2 outputs are features, never labels and never
recomputed in-sample.

## Run

```bash
python scripts/run_phase3_participation_integration.py \
  --phase2-oof data/experiments/phase2/oof_predictions.csv \
  --phase2-model hist_gbm
```

Outputs are `row_predictions.csv`, `fold_metrics.csv`, and `report.json` under
`data/experiments/phase3_participation/`.

## Decision gate

No production change is made by this script. The Phase 2 model label is
explicit rather than hardcoded, and should be chosen only after the canonical
Phase 2 comparison. A candidate requires all requested folds to complete, a
negative player-clustered paired-bootstrap 95% CI for aggregate MAE delta, and
position-level review. Phase 2's canonical-panel acceptance must also be
complete; a result from the raw-snap smoke file is not eligible for adoption.

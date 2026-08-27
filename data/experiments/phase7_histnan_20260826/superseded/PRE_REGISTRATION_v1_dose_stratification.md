# Pre-registration — Phase 7 history-NaN arm, 2026-08-26

Written and committed BEFORE the run. The analysis below is fixed; anything
not specified here is exploratory and must be labelled as such.

## Hypothesis

Preserving genuine debut-week history NaN (`PRESERVE_HISTORY_MISSINGNESS=1`)
improves season-total projections for players whose history features are
actually affected, relative to filling that NaN.

## What changed since this arm was first designed

The arm was pre-registered as a **rookie** intervention. It is not one, and
the run must not be read as one. Measured on the frozen strata below:

- The flag changes ~10% of player-WEEKS, but 96.8% of player-SEASONS contain
  at least one changed week. A binary changed/unchanged split is therefore
  **degenerate at the season level**, which is Phase 7's unit of analysis.
  This plan uses a DOSE instead.
- Only 19% of affected rows belong to rookie seasons. `prev_season_ppg`
  dominates the changed cells (2,088 of 18,355 for WR) and goes NaN for any
  player with no prior season on record — veterans returning from absence and
  players left-censored at the sample boundary, not just debuts. Two of the
  top changed features (`team_plays_roll3_mean`,
  `team_pace_sec_per_play_roll3_mean`) are team context with no rookie
  relationship at all.
- Mean dose is 0.279 for rookies against 0.209 for veterans — a 1.3x tilt,
  not the 4x a rookie-framed reading would assume.

## Arms

| arm | flag |
|-----|------|
| OFF (control) | `NFL_PRESERVE_HISTORY_MISSINGNESS=0` |
| ON (treatment) | `NFL_PRESERVE_HISTORY_MISSINGNESS=1` |

Identical in all other respects, same commit, same fold set. Both arms are
re-run; no prior artifact is reused, because every existing Phase 7 file
predates the training-data corrections and the component-mode fix.

Command per arm, per season:

```
python scripts/run_phase7_season_projection.py --cold-start --preseason-mode \
  --seasons <S> --output <arm_dir>/phase7_<S>.csv
```

Fold set: 2015–2025 (11 seasons).

## Strata — frozen in `strata.csv`, computed before the run

`dose` = fraction of a player-season's weeks with ≥1 feature cell differing
between arms.

| stratum | dose | n | rookies |
|---------|------|---|---------|
| none | 0 | 204 | 0 |
| low | 0–0.25 | 4,832 | 893 |
| mid | 0.25–0.75 | 641 | 124 |
| high | 0.75–1.0 | 621 | 171 |

`strata.csv` is joined on `(player_id, season)`. It is fixed; it will not be
recomputed after seeing results.

## Primary analysis

Mean absolute error and mean bias on `predicted_season_total` vs
`actual_season_total`, ON minus OFF, **within each dose stratum**, pooled
across positions and folds.

Directional prediction: if the mechanism is real, the ON−OFF improvement
should increase monotonically with dose, and should be ~0 in the `none`
stratum. **A `none`-stratum difference materially different from zero
falsifies the setup** — those rows have identical features in both arms, so
any difference there is run-to-run nondeterminism, not effect.

## Secondary analysis

1. Rookie vs veteran **within** each dose stratum. Reported separately, never
   pooled across strata — pooling is what made the previous result
   ("tie on veterans, split sharply on rookies") uninterpretable, since dose
   and rookie status are correlated.
2. Per-position breakdown, exploratory only. 11 folds × 4 positions leaves too
   few rookie seasons per cell (QB: 109 across all 11 folds) for anything
   confirmatory.

## Decision rule

Adopt `PRESERVE_HISTORY_MISSINGNESS=1` as default only if BOTH hold:

1. `none`-stratum |ΔMAE| < 0.05 points (sanity: no effect where no change), and
2. ΔMAE in the `high` stratum favours ON, and the sign is consistent across
   the `mid` and `high` strata.

If ON wins only in the pooled aggregate but not by dose, that is a
composition artifact and the flag stays OFF.

## Known limitations, stated in advance

- Strata are computed from a full-history feature build, independent of the
  per-fold train/test split. The flag's effect on a row is essentially
  fold-independent (it restores per-player debut history), but rolling windows
  do depend on train span, so a small number of rows may be misclassified by
  at most one stratum boundary.
- 11 folds. Fold-level MAE sd in the comparable Phase 3 run was 0.05–0.23
  points per position; differences below that are not interpretable.
- `existing_methodology` is not part of this comparison; both arms use the
  same FINAL_CONFIG, which was re-validated on 2026-08-26 and left unchanged.

# Phase 3 window/weighting re-selection — 2026-08-26

Re-run of the architecture/window/weighting comparison after a day of
training-data corrections. The previous selection (`FINAL_CONFIG`) predated
all of them.

## Conclusion: change nothing

| pos | current config | best in grid | delta MAE | in fold sd |
|-----|----------------|--------------|-----------|------------|
| QB | `all` / F_yeojohnson_huber / none — 5.9013 | *same* | +0.0000 | 0.00 |
| RB | `all` / C_gbm_mae / linear — 4.3157 | `since2013` / C_gbm_mae / none — 4.3104 | −0.0053 | 0.05 |
| WR | `3y` / C_gbm_mae / none — 4.1160 | `all` / C_gbm_mae / none — 4.0496 | −0.0664 | 0.69 |
| TE | `10y` / C_gbm_mae / none — 2.7652 | `all` / C_gbm_mae / linear — 2.7619 | −0.0033 | 0.06 |

QB reproduces its current config exactly. RB and TE differ by 0.05–0.06 of a
fold standard deviation — noise on 3 folds. WR is the only non-trivial delta
(0.69 sd) but `all` ranks FIFTH of six on WR's window mean and only wins in
one architecture/weighting cell out of 36, which is what selection noise
looks like. `FINAL_CONFIG` left unchanged.

**On the `since2013` question**: the candidate was added specifically to test
whether the pre-2013 measurement era should be cut. It ranks 2nd–3rd for every
position and wins nothing outright. Measured answer: no floor, and it barely
matters — those seasons neither help nor hurt materially now that they carry
honest NaN instead of fabricated zeros.

## Files

| file | commit | contents | valid |
|------|--------|----------|-------|
| `phase3_main_a776b38.csv` | a776b38 | full 72-fold run | QB/RB/WR **architecture rows only** |
| `phase3_repair_TE_80bd866.csv` | 80bd866 | TE, all windows × 3 seasons | all |
| `phase3_repair_2023_80bd866.csv` | 80bd866 | QB/RB/WR, 2023 only | all |
| `phase3_MERGED.csv` | mixed | the three above, deduplicated | all — carries a `_src` column |

`phase3_MERGED.csv` is the file to read. Provenance per row is in `_src`;
QB/RB/WR 2024–25 come from the main run, QB/RB/WR 2023 and all TE from the
repairs.

## Why two of the four files are partly invalid

The main run completed 72/72 folds with no crashes and looked clean. Auditing
it rather than reading it found two defects:

1. **Component mode fell back to fp mode in 24 of 72 folds** — every
   `test_season=2023` fold, all four positions, all six windows.
   `team_motion_rate` and `team_play_action_rate` are 100% NaN in those
   training sets (FTN charting is too recent), so their median was NaN,
   imputing NaN with NaN left NaN, and
   `np.isfinite(X_arr).all(axis=1)` then rejected every row. That made
   `existing_methodology` component-mode for 2024/25 and fp-mode for 2023 —
   not comparable across seasons.

2. **TE was swept on the wrong architecture.**
   `PHASE3_WINNER_ARCHITECTURE["TE"]` was still `B_gbm_huber` while
   `FINAL_CONFIG` had moved to `C_gbm_mae` on 2026-08-21. TE's entire window
   ranking was measured against a model production does not use, and TE
   produced 12 rows per window instead of 21.

Both fixed in `80bd866`, which also makes `_architectures_for_position` raise
if those two constants ever diverge again.

The architecture rows were never affected — `_build_feature_matrices` passes
NaN straight to LightGBM and touches neither bug. **Verified, not assumed**:
all 36 WR/2023 architecture rows that appear in both the main run and the
2023 repair are bit-identical (max |delta| = 0). That is what made salvaging
the main run legitimate rather than hopeful.

## Reproduce

```
# main (superseded for TE and for all baselines)
python scripts/run_phase3_window_comparison.py --seasons 2023 2024 2025

# repairs, at commit 80bd866
python scripts/run_phase3_window_comparison.py --positions TE --seasons 2023 2024 2025 \
  --output data/experiments/phase3_repair_TE_80bd866.csv
python scripts/run_phase3_window_comparison.py --positions QB RB WR --seasons 2023 \
  --output data/experiments/phase3_repair_2023_80bd866.csv
```

## Caveats

- **3 folds per position.** Every delta above is within or near noise. Nothing
  here should be treated as establishing a config; it establishes that the
  current config is not obviously wrong on corrected data.
- Timing: the main run took 7.2h (6.4 min/fold, cold). The repairs ran at
  ~40 s/fold because caches were warm — a 10× difference that is caching, not
  skipped work, as the bit-identical cross-check proves.

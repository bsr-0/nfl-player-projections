# Plan B wired into Plan A's full-PPR selector — 2026-09-25

> **Void (2026-10-08).** The team totals used here read the predicted game's
> own team stats (GAPS.md 2026-10-08). The leak-free re-run is
> `data/experiments/full_ppr_with_plan_b_no_sameweek_team_20261009/`:
> 2.403 pooled, −0.023 vs corrected Plan A.

**Plan B's confirmed uncapped joint-softmax allocation, added as a new
candidate arm to Plan A's existing gated full-PPR selector, further reduces
full-PPR MAE from 2.36984 to 2.34608** on the same 40,559-row, three-fold
(2023/2024/2025) real holdout Plan A's own guarded validation used. This is
the first time Plan B has been evaluated at the actual fantasy-points level,
not just raw share MAE.

## Result

| | Pooled MAE | vs. all-rolling3 baseline (2.49130) |
| --- | ---: | --- |
| Plan A only (existing arms: ridge/xgb blends, rolling3) | 2.36984 | −0.12146 |
| **Plan A + Plan B's `joint_other_mae` arm** | **2.34608** | **−0.14522** |

Paired bootstrap on the full run: point estimate −0.14522, 95% CI
[−0.15977, −0.12922], `significant_improvement: true` (500 bootstraps, same
methodology as every other selector run). Per-fold: fold 0 (2023) has no
completed prior-fold evidence to gate on, so — identically to the pre-Plan-B
baseline run — every target defaults to `rolling3` there (delta exactly
0.0, matching the earlier run's fold-0 numbers exactly). All of the
additional lift is in folds 1 (2024) and 2 (2025), where the gate has prior
evidence to admit Plan B's arm.

**Selected allocation arm per target after adding Plan B's arm as a
candidate:**

| Target | Selected arm | Plan B's arm available? |
| --- | --- | --- |
| rushing_yards | **joint_other_mae** | yes |
| receiving_yards | **joint_other_mae** | yes |
| receptions | xgb_blend_renorm | yes (not selected) |
| passing_yards | **joint_other_mae** | yes |
| receiving_tds | rolling3 | no (sparse, out of scope) |
| rushing_tds | rolling3 | no (sparse, out of scope) |
| passing_tds | rolling3 | no (sparse, out of scope) |
| interceptions | rolling3 | no (sparse, out of scope) |

Plan B's arm was gate-eligible and won the coordinate-descent step for 3 of
its 4 covered targets. `receptions` still preferred Plan A's existing
`xgb_blend_renorm` — Plan B's share-level win for that target (confirmed
separately, see `data/experiments/plan_b_uncapped_joint_mae_confirm_20260925/`)
did not translate into the best full-PPR choice, most likely because the
selector optimizes reconstructed PPR jointly with the chosen team-total arm,
not share MAE in isolation.

## How this was built (no changes to already-verified Plan A code)

1. `src/evaluation/plan_b_arm_adapter.py` (new) reshapes Plan B's wide
   `predictions.csv` into Plan A's long arm schema
   (`player_id, season, week, team, position, is_cold_start, fold, arm,
   actual_share, predicted_share`), deriving fold numbers by literally
   instantiating `SeasonAwareTimeSeriesSplit` with Plan A's real config
   (`--seasons 2006 2025 --n-test-seasons 3`, `cv_gap_seasons=0` from
   `config.settings.TEAM_ALLOCATION_MODEL_CONFIG`) rather than hand-derived
   arithmetic — confirmed to produce exactly `{2023: 0, 2024: 1, 2025: 2}`,
   matching the real existing allocation CSVs' own fold/season columns
   exactly.
2. `attach_plan_b_arm` validates Plan B's rows exactly match the truth
   population for the given folds and that `actual_share` agrees with the
   existing arms' labels, before concatenating — fails loud on any
   population or label mismatch rather than relying solely on
   `ppr_truth.validate_inputs` downstream.
3. `scripts/build_full_ppr_allocations_with_plan_b.py` (new) uses the
   adapter to write a full set of 8 target CSVs: the 4 targets Plan B
   covers get its arm attached to Plan A's existing arms; the 4 sparse
   targets are copied through unchanged. Output:
   `data/experiments/full_ppr_with_plan_b_20260925/allocations/`.
4. `scripts/evaluate_joint_ppr_selector.py` (unmodified) was then run
   exactly as usual against that merged directory
   (`--allocation-dir .../allocations --team-total-input-dir
   .../full_ppr_safe_blend_event_mass_20260924/team_totals`) — Plan A's
   team-total arms are reused completely unchanged; only the allocation
   side gained one new candidate.

## Verification performed before trusting this number

- Independently recomputed pooled selected/baseline MAE directly from
  `ppr_oof_rows.csv` (not just the JSON report) — matches to floating-point
  precision.
- Recomputed `raw_truth_sha256` from a fresh, independent `checked_truth()`
  call — matches the reported hash exactly, confirming the truth used was
  not stale or silently altered.
- `input_audit.targets` in the report shows `allocation_arms: 5` for each
  of the 4 Plan-B-covered targets (up from 4 in the pre-Plan-B run) with
  `allocation_rows_per_arm` exactly matching the full population size for
  every arm — confirms `validate_inputs`'s fail-closed population check
  actually ran against the new arm, not a silent no-op.
- Fold-0 delta of exactly 0.0 (every target forced to `rolling3` with no
  prior-fold evidence) matches the pre-existing baseline run's fold-0
  numbers exactly (same `n`, same `baseline_mae`) — confirms this is
  expected selector behavior, not an artifact of the merge.
- Source allocation/team-total CSVs' SHA-256 hashes were confirmed
  unchanged from the original 2026-09-24 run before use (see
  `data/experiments/full_ppr_with_plan_b_20260925/allocations/manifest.json`).

## Caveats

- This reuses Plan A's existing team-total arms unchanged; Plan B does not
  forecast team totals itself. Any residual error from those forecasts is
  shared with the pre-Plan-B baseline and is not something this result
  speaks to.
- The sparse zero-inflated TD/INT targets are entirely untouched — this
  result says nothing about whether Plan B-style architectures could help
  there (a different modeling problem, out of scope per the uncapping
  work).
- The one confirmed uncapped Plan B run (`other_weight=0.1` for all four
  targets, see its own README) is the same run wired in here; none of that
  run's caveats (population/baseline shift versus the earlier capped run,
  the `targets`/`rushing_attempts` nested-check discrepancy, `passing_yards`'s
  smaller QB-only sample) are resolved by this step — they carry forward
  unchanged, and `targets`/`rushing_attempts` are not points-eligible so
  don't appear in this specific result at all.
- No serving integration or artifact promotion follows from this result.

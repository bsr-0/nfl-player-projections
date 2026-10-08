# 2025 Plan A versus served architecture: full PPR

This research comparison uses ten raw recorded full-PPR components for the
actual label. Frozen Plan A still predicts its original eight components; its
fumbles-lost and two-point-conversion forecasts are declared as zero. The
served architecture directly predicts full-PPR points. Neither arm receives
observed extra-component points in its prediction.

The held-out served fold trains on seasons 2006–2024 and predicts 2025. Its
normal `target_1w` is the *next observed player game*, so the export maps each
forecast origin to that target game's exact player/season/week/team/position.
There is no `week + 1` assumption. Fit weights are written to an isolated
experiment directory; the live production weights are checked for changes.
Before fitting, two training-path defects were corrected: duplicate pregame
injury reports could expand training rows, and rookie comparable features
could use outcomes from the row's own or later seasons. The feature path now
preserves the player-row count and restricts comparable outcomes to seasons
strictly before each row's season. The historical saved production backtests
predate these corrections and are not used as the served arm here.

## Frozen population

The population was selected before the served fold completed, from target-key
eligibility alone. The 2025 production fold has 6,680 raw player-game rows,
of which 6,093 have a next observed game. Exactly 5,826 targets match frozen
Plan A's 2025 keys. The other 267 comprise 251 postseason targets outside the
Plan A regular-season panel and 16 regular-season position disagreements
between the two source panels. The full Plan A 2025 panel has 13,689 rows.
Both producers must supply every one of the 5,826 declared keys, or comparison
stops. Saved row-level actuals must match the separately frozen raw truth.

The paired sample contains 809 confirmed draft rookies, 4,896 returning
players, and 930 players in their first observed canonical season. These are
different concepts: the first-observed group includes undrafted players and
players whose earlier history is absent. Observed offensive snaps are nonzero
for 5,817 rows and unknown for 9; there are no zero-offensive-snap rows in
this matched sample. Snap status is a retrospective evaluation slice and is
never supplied as a pregame predictor.

This matched group is much harder for Plan A than its full 2025 population:
Plan A's full-PPR MAE is 4.08303 on the 5,826 matched games, versus 2.26559
on all 13,689 canonical 2025 games. The full-population number is descriptive
context; it cannot be paired with the served fold, which does not forecast all
those canonical rows.

The comparison reports pooled and position-specific MAE, paired absolute-error
differences, and a 95% interval from paired calendar-week block resampling.
It also reports cohorts for confirmed draft rookies, returning players,
first-observed players, nonzero/zero/unknown offensive snaps, and position
intersections. Segment intervals are exploratory and have no multiple-testing
adjustment. A negative delta favors the served architecture.

## Reproduce

The 2026-09-25 attempt is under `data/experiments/full_ppr_head_to_head_20260925/`.
The current run is under `data/experiments/full_ppr_head_to_head_20260930/`
(preflight/population v4, fold v5). Earlier 09-30 folds stopped on
winsorized test targets (GAPS.md 2026-10-07). Before any long fit, run the
~4-minute smoke check:

```sh
python scripts/smoke_served_fold.py \
  --preflight-dir <a prepared preflight> \
  --output-dir data/experiments/smoke_served_fold/<stamp>
```

If `run` fails after training, fix the check and re-run
`export_served_fold_full_ppr.py finalize --input-dir <preflight> --output-dir <fold>`;
the fit's capture, coverage and model hashes are persisted before validation.

Run these in order with new output directories if repeating:

```sh
python scripts/prepare_full_ppr_head_to_head.py \
  --plan-a-manifest data/experiments/ppr_head_to_head_20260924/inputs/plan_a.manifest.json \
  --raw-truth data/experiments/ppr_head_to_head_20260924/inputs/raw_truth.csv \
  --db data/nfl_data.db \
  --output-dir data/experiments/full_ppr_head_to_head_20260925/inputs

python scripts/export_served_fold_full_ppr.py prepare \
  --test-season 2025 \
  --output-dir data/experiments/full_ppr_head_to_head_20260925/served_2025_preflight_v6

python scripts/compare_full_ppr_head_to_head.py predeclare \
  --plan-a-manifest data/experiments/full_ppr_head_to_head_20260925/inputs/plan_a_full.manifest.json \
  --preflight-manifest data/experiments/full_ppr_head_to_head_20260925/served_2025_preflight_v6/manifest.json \
  --output-dir data/experiments/full_ppr_head_to_head_20260925/population_2025_v4

python scripts/export_served_fold_full_ppr.py run \
  --input-dir data/experiments/full_ppr_head_to_head_20260925/served_2025_preflight_v6 \
  --output-dir data/experiments/full_ppr_head_to_head_20260925/served_2025_fold_v5

python scripts/compare_full_ppr_head_to_head.py compare \
  --population-manifest data/experiments/full_ppr_head_to_head_20260925/population_2025_v4/manifest.json \
  --served-manifest data/experiments/full_ppr_head_to_head_20260925/served_2025_fold_v5/manifest.json \
  --output-dir data/experiments/full_ppr_head_to_head_20260925/comparison_2025
```

The preparation and comparator reject changed input hashes, duplicate or
missing target keys, nonfinite predictions or raw labels, mismatched scoring
definitions, and same-season training. Saved paired rows are rescored before
the final report is written. This is one held-out-season comparison of the
served *architecture*, not an evaluation of the live 2026 fitted weights or a
promotion decision. It does not measure game simulation or correlation layers.

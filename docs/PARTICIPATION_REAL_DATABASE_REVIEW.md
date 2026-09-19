# Participation system: repository database review, 2026-09-18

**Phase 2 remains incomplete under the stated acceptance gate.** Histogram
GBM is the selected experimental classifier, but its calibration is worse
than the position-rate baseline within positions. No serving promotion is
justified by these results.

## Reproduction and provenance

- Updated `main` from `origin/main`, preserving the local commit and merge.
- Database: `data/nfl_data.db`, not the raw-snap smoke archive.
- Consistent SQLite backup, verified with `PRAGMA quick_check`:
  `data/backups/nfl_data_before_participation_20260918_101958.db`.
- Runtime: pyenv Python 3.11.15, pandas 2.1.4, scikit-learn 1.4.0.
  The installed Conda `nflproj` environment lacked pandas.
- Command: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python
  scripts/run_participation_pipeline.py --phase2-model hist_gbm`.
- Outputs: `data/experiments/participation_system/`, including the requested
  Phase 1 audit, Phase 2 evaluation, OOF CSV and manifest. Generated run data
  is ignored by Git; this review records the decision.

## Coverage and label audit

The final canonical population contains **164,298** identified player-weeks:
81,555 confirmed played, 5,489 measured zero snaps, and 77,254 unknown.
Unknown targets remain null and are excluded from training and OOF scoring.
Source `player_weekly_stats` rows were compared in both directions against
the backup and were unchanged.

Coverage changes are traceable to source populations:

- 2013–2015: approximately 8,100–8,200 canonical rows per season and
  6,400–6,500 measured snap rows.
- 2016: 10,063 canonical rows, with broader ACT/RES roster coverage;
  measured snap rows remain 6,397.
- 2017–2019: approximately 15,700–16,100 canonical rows. CUT roster rows
  expand to 3,851–4,518 per season, and DEV rows to 1,835–2,019. The
  measured snap population remains approximately 6,450 per season.
- 2020: CUT rows fall from 3,851 to 277, explaining much of the population
  reduction; observed labels increase to 6,669.
- 2021–2025: the 17-game schedule raises observed rows to approximately
  7,000–7,100; roster coverage is broadly stable by position.
- 2026: only 940 identified Week 1 player-weeks, all unknown. There are no
  2026 snap rows in this database. No 2026 outcome evaluation is claimed.

The snap ID audit (`snap_id_coverage.csv`) records 9–68 unmapped source rows
per season, at most about 1.04% of the fantasy-position snap population;
there is no collapse in mapped-label coverage. The ID source returned 7,813
PFR-to-GSIS mappings. These missing mappings and changing roster populations
remain limitations: evaluated performance is conditional on observed labels,
not evidence of accuracy on the full roster-only population.

Phase 1 originally failed on 1,044 nonfantasy stats rows (kickers, defenses,
linemen and defenders). Position resolution now excludes known out-of-scope
positions while retaining the failure for genuinely missing positions.
The roster loader reports 84 source rows without player IDs and excludes
them before joins/deduplication. Previously they collapsed to 61 anonymous
canonical rows and then disappeared from feature generation. Null identities
now fail validation rather than disappearing silently.

## Phase 2 results

Expanding-season evaluation uses 2013–2015 as the first training window and
holds out 2016 through 2025. Each of the two classifiers has 67,628 OOF rows
(135,256 total); every row has training max season strictly before its test
season. Metrics below are equal-weight averages of the ten seasonal folds.

| Primary 10% target | Brier | Log loss | 10-bin ECE |
|---|---:|---:|---:|
| Histogram GBM | 0.082056 | 0.263840 | 0.009407 |
| Logistic | 0.084810 | 0.274446 | 0.011521 |
| Position rate | 0.142910 | 0.457141 | 0.011523 |
| Previous game | 0.122172 | 1.371977 | 0.112948 |
| Rolling 3 | 0.106035 | 0.943982 | 0.083603 |

GBM beats logistic on both loss metrics in all ten seasons, and beats every
simple baseline in aggregate and by position. Baselines now use the history
of the threshold being scored; the previous implementation incorrectly used
any-snap history for the 10% and 25% targets.

Conditional opportunity MAE is **0.116648** versus **0.133336** for rolling-3,
with improvements at every position. The 1,332 classification cold-start
rows are reported explicitly: GBM Brier 0.234815 versus baseline 0.311669;
log loss 0.662315 versus 0.864627. Conditional cold-start MAE is 0.188690
versus 0.286890. Cold-start performance is materially weaker than performance
with history, but its improvement is not hidden by the aggregate.

### Calibration gate: not passed

| Position | GBM ECE | Position-rate ECE |
|---|---:|---:|
| QB | 0.021566 | 0.009330 |
| RB | 0.023059 | 0.017346 |
| TE | 0.018954 | 0.018485 |
| WR | 0.013404 | 0.005768 |

Aggregate ECE improves, but each position's mean ECE is worse than the
position-rate baseline. No noninferiority tolerance was preregistered, so
these results cannot establish “without worse calibration.” ECE is a binned
diagnostic, not a significance test; a future calibration experiment should
fit any calibrator only within past-season training data and evaluate on
untouched future seasons. The acceptance gate was not relaxed after seeing
these results, and logistic does not solve the position calibration issue.

## Validation and integration

The focused regression suite passes (37 tests). The real-data perturbation
check changes outcomes at and after 2024 Week 9 and verifies that all declared
features through that week remain unchanged (143,075 rows compared).
OOF population, uniqueness, finite prediction and temporal provenance checks
pass. Detailed results are in `real_data_checks.json` and
`acceptance_review.json`.

The full test-suite and Phase 3 outcomes are recorded below when execution
finishes. Phase 7 serving features and FINAL_CONFIG remain unchanged.

# Pre-registration: season-to-date volume features for the WR model (2025 development test)

Status: **pre-registered 2026-10-14, binding from the commit that adds this file.** No candidate had been
fit or scored when it was written. It is a development test on a season that has been viewed many times
(see "What this is not"). It changes nothing that is served, pinned or part of
`docs/PRODUCTION_SELECTION_RULE.md` or `docs/PREREG_NO_PRIOR_2026.md`.

## Question

Does adding season-to-date volume to the WR model's inputs reduce its under-projection of high-volume
receivers, without hurting accuracy elsewhere, when scored week by week on 2025?

## Why (exploratory, not evidence here)

On 2025 WRs, the served model's misses correlate with season-to-date receptions, targets, high-leverage
touches, receiving yards and neutral-situation targets, holding its projection fixed (Holm p < 0.005,
+1.0 to +1.7 points per SD). The model has only 3-game averages of these, plus preseason anchors
(prev_season_ppg, age_curve, bayesian_prior_ppg, draft features).

## Design

- **Model:** the served WR 1-week model, refit on seasons through 2024 only (`raw_train.parquet` of the
  frozen 2025 preflight), with the served fold's tuned hyperparameters held fixed and no retuning. Features
  for each 2025 row are built as of that row's week (`prepared_test_asof.parquet`).
- **Population:** the 2,385 WR rows of the 2025 fold (256 games), keyed by target game, actual = the fold's
  recorded actual points.
- **Candidates**, features computed from `player_weekly_stats` as the expanding mean over the player's games
  of that season up to and including the origin game; NaN filled with the training-set median in both train
  and test:
  - **C0 baseline:** the 70 current features, refit under the identical procedure.
  - **C1:** C0 + `targets_s2d`, `receptions_s2d`, `receiving_yards_s2d`.
  - **C2:** C1 + `neutral_targets_s2d`, `high_leverage_touches_s2d`.
  Both extra features are populated for every season 2013-2025, so C2 is eligible.
- **Windows:** development = 2025 target weeks 1-5; confirmation = target weeks 6-18. The confirmation
  window is not scored until the development decision below is written to `report_dev.json` and committed.

## Metrics (paired vs C0, same player-weeks)

- Pooled MAE difference d = |err_candidate| - |err_C0|, two-way (week x player) pigeonhole bootstrap,
  B = 10,000, seed 42 (`selection_gates._weight_batches`). Negative favours the candidate.
- RMSE; mean lean (pred - actual) overall and by C0 projection tier (1-12, 13-36, 37+ within week);
  mean within-week Spearman; pair-ordering accuracy within week.
- Form-lag slope: OLS slope of (actual - projected) on (season-to-date PPG before the target week minus the
  projection), rows with at least 3 prior games. Closer to 0 is better.
- Per-week table and cumulative curve; the four players (Waddle, Evans, Flowers, Collins) are shown for
  illustration, not scored.

## Decision rules (fixed here)

- **Development (weeks 1-5, about 700 rows):** a candidate advances if its point d < 0, its RMSE is not
  higher than C0's, and its |lean| is not larger. Intervals are reported but wide; this is a screen.
- **Confirmation (weeks 6-18):** a candidate passes if
  1. point d < 0 and the one-sided 95% upper bound of d is below +0.03;
  2. RMSE and mean Spearman are not significantly worse (one-sided 95% bound on worsening not above 0);
  3. the form-lag slope is closer to 0 than C0's.
- If both C1 and C2 pass, the simpler (C1) is preferred unless C2 improves d by at least 0.05 more.
- A pass leads only to a forward pre-registration for 2026 weeks after the weeks 6-13 window. Nothing is
  adopted, served or retrained on the strength of this test.

## Procedure and limits

- Script `scripts/wr_feature_candidates_2025.py` (sha256 recorded below before the first fit), tests in
  `tests/test_wr_feature_candidates.py` on synthetic rows. Outputs under
  `data/experiments/wr_features_2025/`.
- Refits run inside `redirect_models_dir` on a temporary directory; live model files are hashed before and
  after and must be identical.
- C0 is reported against the saved tuned fold's WR MAE (4.265). A large gap stops the run for a report.
- Features added after preprocessing are not bounded-scaled or winsorized like the existing ones (trees do
  not care; the Ridge learner standardises internally).
- Only the 1-week WR model is refit; the live blend, the 4- and 18-week models, QB/RB/TE are not.

## What this is not

2025 is not blind: it chose the candidate features, the blend constant and this design. A pass here is
development evidence only. The confirmation is 2026, scored under its own pre-registration.

## Amendment 1 (2026-10-14), before any candidate was fit

Reading the trainer showed `ModelTrainer.train_all_positions` runs its own feature selection (correlation
filter, stability selection, VIF pruning) before fitting. Season-to-date volume is highly correlated with
the existing 3-game averages, so selection could drop a candidate and the test would measure the selector,
not the information. The harness therefore fits `MultiWeekModel("WR")` directly on an explicit column list
for every arm (C0 = the fold model's 70 `feature_names`; C1 and C2 add their columns), with the trainer's
other steps unchanged: median fill on the training prefix, rows without a 1-week target dropped, recency
weights, season labels for CV. New-feature NaN in the test frame is filled with the training median, not
the predict-time zero-fill. Design, candidates, metrics and decision rules are as above.

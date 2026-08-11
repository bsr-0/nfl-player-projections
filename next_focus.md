NFL Fantasy PPR — Final Modeling Plan

⚠ 2026-08-10: Complete Player-Game Panel prerequisite landed (GAPS.md,
search "SUPERSEDED — 2026-08-10"). `player_weekly_stats` was missing
"played but scored zero" rows entirely (silently indistinguishable from
"didn't play") — confirmed empirically, ~5x understatement of true
zero-production-game frequency (6.3% -> 29.7% in the corrected panel).
Phases 2-7 below were ALL built and validated on the pre-fix data and
should be treated as PROVISIONAL, not final, until re-run on the corrected
panel — that re-run has not happened yet. This is most consequential for
architecture D (two-stage/hurdle), which specifically needs true zero
observations to learn P(PPR > 0) and could not have learned it correctly
before this fix.

Goal

Build two related but distinct products:

1. Single-week projection: predict a player's PPR points for a specific upcoming game.
2. 18-week projection: predict a player's total PPR production over the upcoming season.

Build and optimize the single-week model first, then use it to construct the 18-week projection.

⸻

Phase 1 — Data & Leakage Framework [COMPLETE — 2026-08-10]

Audit performed; two live leakage bugs found and fixed (same-week own-team
stats in get_all_players_for_training; injury reports modified after
kickoff), formal FEATURE_AVAILABILITY registry added to src/utils/leakage.py.
Full writeup: GAPS.md §7.6. Tests: tests/test_leakage_guards.py (30 passing).

1. Define prediction timestamps

For every prediction, establish:

prediction_timestamp
target_period

Every feature must represent information available at or before the prediction timestamp.

This applies to both models.

Explicitly audit:

* projection_1w
* projection_4w
* projection_18w
* predicted_points
* rolling features
* season-to-date statistics
* injury/status information
* team/opponent statistics

Missingness ≠ leakage.

Future information = leakage.

Where possible, programmatically enforce:

available_timestamp <= prediction_timestamp

⸻

Phase 2 — Single-Week Model [COMPLETE — 2026-08-10]

Target/baselines/architectures A-F implemented in src/models/single_week_ppr/
and evaluated via walk-forward folds (train up to season N-1, test season N)
for 2023/2024/2025, all 4 positions. Raw fantasy_points target, no
winsorization. All 6 architectures beat all 4 naive baselines and the
existing production methodology at every position. Best per position:
QB/RB/WR = C (GBM+MAE) or F (Yeo-Johnson+Huber); TE = C. B (Huber, the
plan's stated primary candidate) was never the single best but was always
within ~0.3 MAE of the winner. Full results: data/experiments/
phase2_single_week_comparison.csv (144 rows).

Findings to carry into Phase 4 (best-MAE != best-calibrated):
- Systematic bias: C, F, and E's median (p50) all under-predict by ~0.7-1.1
  fantasy points on average, across every position. A (GBM+MSE) and D
  (hurdle) stay near-zero bias. MAE alone hides this — worth weighing before
  picking a "winner" per position.
- Quantile coverage (architecture E): P25/P50/P75 track nominal coverage
  reasonably well (e.g. P50 actual ~49-58% vs. target 50%). P90 is
  consistently UNDER-covered (~84-89% actual vs. 90% target) at every
  position — the ceiling estimate runs a bit conservative. Full pinball
  loss + coverage per quantile/position/season is in the CSV
  (p25/p50/p75/p90_pinball, p25/p50/p75/p90_coverage columns).

Incidents + fixes during this phase (see GAPS.md §7.7-7.8 for full detail):
- §7.7: settings.MODELS_DIR redirection alone doesn't protect production
  model artifacts during experimental runs (some modules bind MODELS_DIR at
  import time) — same risk likely present in train.py's own --walk-forward
  flag, flagged but not fixed there.
- §7.8: the first fix for §7.7 (a denylist over all of data/) deleted the
  production SQLite database (data/nfl_data.db, gitignored, legitimately
  written during normal operation) because it looked like untracked
  "pollution." Recovered from an existing 2026-08-04 backup + auto-refresh
  catch-up; no data permanently lost. Re-fixed with a narrow allowlist of
  the three specific paths actually observed leaking, which can never
  include live data assets like the database.

Tests: tests/test_single_week_ppr_architectures.py (11 passing).

2. Target

For each player/game:

Y = PPR points in the upcoming game

Build separate models:

* QB
* RB
* WR
* TE

Use raw PPR as the primary target.

Do not remove or winsorize high-scoring games.

⸻

3. Establish naive baselines

Before ML, evaluate:

* Recent-game average
* Rolling 3-game PPR average
* Rolling 5-game PPR average
* Position/role baseline

ML models must beat these out-of-time.

⸻

4. Test model architectures

For each position:

A. MSE

Gradient Boosting
Raw PPR
MSE loss

B. Huber

Gradient Boosting
Raw PPR
Huber loss

Primary candidate.

C. MAE

Gradient Boosting
Raw PPR
MAE loss

D. Two-stage / hurdle

Particularly important for RB/WR/TE:

Stage 1:
P(PPR > 0)
Stage 2:
E[PPR | PPR > 0]
Final prediction:
P(PPR > 0) × E[PPR | PPR > 0]

Also test PPR > 5 if useful.

E. Quantile models

Train:

P25
P50
P75
P90

These provide floor/median/upside/ceiling estimates.

F. Target transformation challenger

Test:

Yeo-Johnson(PPR) + Huber

Do not prioritize log transformation because PPR contains negative values.

⸻

Phase 3 — Optimize Historical Training Data [COMPLETE — 2026-08-10]

Full grid run: 4 positions x 5 windows (3y/5y/7y/10y/all) x 3 weightings
(none/linear/exponential) x 2 architectures per position (B-Huber + that
position's Phase 2 MAE winner) x 3 validation seasons = 420 rows in
data/experiments/phase3_training_window_comparison.csv.

Deliberately bypassed the codebase's existing hard 2018+ training floor
(TRAINING_START_YEAR_DEFAULT, src/utils/data_manager.py) for this
experiment only, reaching back to MIN_HISTORICAL_YEAR (2006) via
src/models/single_week_ppr/windows.py, so "10-year"/"all history" would
actually differ from "5-year" (under the floor they'd be identical for our
2023-2025 test seasons, since only 5-7 post-2018 seasons ever exist).

Findings:
- More history helps for QB/RB/TE — MAE improves monotonically from 3y to
  all-history (QB: 6.58->6.37; RB: 4.79->4.71; TE: 3.70->3.65). Pre-2018 data
  does NOT hurt despite structural missingness in NGS/EPA/modern-snap-share
  features there (handled via LightGBM's native NaN splits, not fillna(0) —
  see below). This pushes back on the codebase's stated rationale for the
  hard 2018+ floor, at least for these 3 positions.
- WR is the exception: flat/mixed after 5y (4.76-4.79 MAE across 5y-all,
  within noise) — extra history doesn't help WR the way it helps the others.
- Recency weighting: uniform ("none") won 13/20 position x window combos,
  "linear" 6, "exponential" (the ONLY scheme currently live in production)
  won just 1. Production may be over-weighting recency relative to what the
  data supports, at least for the single-week horizon.
- Best single config per position: QB/WR -> all-history/none/C(GBM-MAE);
  RB -> all-history/none/F(Yeo-Johnson); TE -> 10-year/none/C(GBM-MAE).

Code changes: sample_weight threaded through all 4 architecture wrapper
classes; run_fold() gained train_seasons_override to bypass load_training_
data's hard floor without monkeypatching (2 prior incidents this session
established that monkeypatching module-level constants is unreliable — see
GAPS.md §7.7-7.8); feature matrices switched from unconditional fillna(0)
to LightGBM-native NaN handling (fillna(0) only for the sklearn fallback
path) since pre-2018 rows have structurally-missing feature families, not
randomly-missing values. Results append incrementally to CSV per fold
(Phase 2 lesson: a killed/timed-out run shouldn't lose completed work).
Tests: tests/test_phase3_window_weighting.py (18 passing).

After identifying promising model architectures, optimize training history and recency weighting.

Test training windows:

* All history
* 10 years
* 7 years
* 5 years
* 3 years

Test:

* No weighting
* Linear recency weighting
* Exponential recency weighting

At minimum compare:

5-year + no weighting
10-year + exponential weighting
All history + exponential weighting

Optimize:

Position × architecture × training window × recency weighting

using only rolling validation.

⸻

Phase 4 — Single-Week Validation [COMPLETE — 2026-08-10]

Ran each position's FINAL_CONFIG (chosen architecture/window/weighting from
Phases 2-3) with row-level predictions saved: 114,989 rows across 4
positions x 3 seasons in data/experiments/phase4_row_level_predictions.csv
(src/models/single_week_ppr/final_config.py, tiers.py, analysis.py). Column
shape matches Phase 10's spec, so this artifact is reusable there.

Overturns the Phase 2/3 "winner" framing — MAE alone was misleading:
- The winning architectures (C/F/quantile-p50) all carry NEGATIVE bias of
  -1.1 to -1.3 points on average, vs. existing_methodology's near-zero bias
  (-0.04). They win on MAE but systematically underpredict.
- Broken out by player tier: the new architectures' advantage all but
  disappears for the ELITE tier specifically (QB 6.34 vs 6.42, RB tied
  6.76 vs 6.76, TE 5.12 vs 5.17, WR tied 6.46 vs 6.46) — the improvement
  Phase 2/3 found is concentrated in depth/starter/waiver/rookie tiers, not
  the players most likely to matter for lineup decisions.
- Predicted-score-bucket breakdown is inconsistent across positions: WR's
  "20+" bucket sees the new architecture dramatically outperform existing
  methodology (4.40 vs 7.86 MAE), but QB/RB's "20+" buckets are roughly a
  wash. Not a uniform pattern to generalize from.
- Quantile calibration (averaged across seasons, all 4 positions): p25
  coverage ~0.26-0.30 (slightly over nominal 0.25), p50 ~0.50-0.53 (on
  target), p75 ~0.73-0.75 (on target), but p90 is UNDER nominal 0.90 at
  every single position (0.855-0.887) — the ceiling estimate is
  systematically too conservative across the board, not just noise in one
  position (confirms/strengthens the Phase 2 finding).

Net: before treating Phase 2/3's per-position "winner" as final, the bias
and elite-tier findings above should inform Phase 12 (final model
selection) — a lower-bias, existing-methodology-like option may be
preferable for elite-player decisions even if its overall MAE is slightly
higher. Full writeup: GAPS.md §7.10. Tests: tests/test_phase4_analysis.py
(22 passing).

Use expanding/rolling time-series validation.

Example:

Train: 2006–2022 → Validate: 2023
Train: 2006–2023 → Validate: 2024
Train: 2006–2024 → Validate: 2025

Never random K-fold.

Primary metric

MAE

Secondary

* RMSE
* Median absolute error
* R²
* Spearman correlation
* Mean bias

Evaluate overall and by:

* Position
* Season
* Predicted-score bucket
* Player tier

For quantile models, measure calibration/coverage.

⸻

Phase 5 — Hyperparameter Tuning [COMPLETE — 2026-08-10]

Nested-CV Optuna tuning (100 trials, inner walk-forward split within each
fold's training seasons only — outer 2023/2024/2025 test seasons never
touched by the search, per the user-confirmed design) for each position's
FINAL_CONFIG architecture. src/models/single_week_ppr/tuning.py,
run_tuned_validation() in evaluate.py. Results: data/experiments/
phase5_tuned_predictions.csv (row-level, 16,427 rows) + phase5_tuned_
hyperparameters.csv (winning params per position/season).

Finding: tuning barely moved the needle. QB got marginally WORSE
(+0.024 MAE), RB/WR/TE improved marginally (-0.008 to -0.019 MAE) — noise-
level changes, not a meaningful lift. Bias was essentially unchanged in
every position (still -1.0 to -1.4 for RB/WR/TE, -0.32 for QB). This
reinforces the Phase 4 finding rather than fixing it: the bias is a
loss-function/target-skew artifact (median-seeking losses vs. a
right-skewed target), not an undertuned-model artifact — 100 trials of
real hyperparameter search around Phase 2's "reasonable defaults" couldn't
touch it, which is itself informative. Full writeup: GAPS.md §7.11.
Tests: tests/test_phase5_tuning.py (8 passing).

Only after selecting:

1. Model architecture
2. Training window
3. Recency weighting

Then tune model hyperparameters.

This prevents wasting compute optimizing an inferior architecture.

⸻

Phase 6 — Feature Engineering [COMPLETE — 2026-08-10]

Fresh audit found CAUSAL_FEATURES (config/settings.py, then v30) already
covered nearly everything in this phase's Opportunity/Efficiency/Context
priority list, built up over 30 prior feature-version bumps — GAPS.md
§8.1's "missing features" table was stale (10 of its items were actually
already resolved v24-v30; corrected in GAPS.md). Two real gaps remained:
route participation (confirmed genuinely infeasible — no route/routes_run
data in any ingested source, nfl_data_py has no participation import) and
rolling catch rate (real gap — raw catch_rate existed but was never rolled
into a leakage-safe feature). Added catch_rate_roll3_mean to
CAUSAL_FEATURES[RB/WR/TE] via the existing generic rolling engine
(src/features/feature_engineering.py:_create_causal_rolling_features) —
same shift(1) mechanism as every neighboring feature. FEATURE_VERSION
bumped to 31. Verified via real production smoke train (WR, --fast
--no-tune): 65 features (up from 64), feature_version.txt=31, new
column 0% NaN / 0-100 range / mean ~64% catch rate (sane). Full writeup:
GAPS.md §8.1.

Phase 6b follow-up (same day): user directly asked whether team run/pass
tendency and hybrid usage (pass-catching RBs, rushing WRs) were captured.
Two more real gaps found and fixed: team_neutral_pass_rate_oe_roll3_mean
(all 4 positions) and rush_share_pct_roll3_mean (WR only — RB already had
it). FEATURE_VERSION bumped to 32, verified the same way (67 features,
Deebo-Samuel-topped rush-share sanity check). Feature-count/ablation
testing (never done in Phases 2-5) scoped but not started — needs explicit
go-ahead. Full writeup: GAPS.md §8.4.

Phase 6c (same day, follow-up executed): feature-count ablation, run for
real. 48 rows (4 positions x 3 seasons x 4 feature-count candidates:
10/20/30/all), src/models/single_week_ppr/ablation.py. Finding: MAE
improves monotonically from 10->all features at every position (no
degradation at higher counts) — current ~57-67-feature CAUSAL_FEATURES
set is not past diminishing returns for the tree-based FINAL_CONFIG
architectures. Specific to tree models (robust to multicollinearity);
says nothing about the actually-deployed Ridge-based production path
(src/models/component_predictor.py), which is a separate, unresolved
question. Full writeup: GAPS.md §8.5.

After the modeling strategy is established, improve the feature set.

Prioritize:

Opportunity

* Targets
* Carries
* Snap share
* Routes
* Red-zone opportunities
* Goal-line touches

Efficiency

* Yards/target
* Yards/carry
* Catch rate
* Position-specific efficiency

Context

* QB/team metrics
* Opponent metrics
* Game environment
* Lagged statistics
* Rolling statistics
* Season-to-date statistics

Maintain strict temporal availability.

⸻

Phase 7 — Build the 18-Week Projection [MECHANISM BUILT, REAL RUN PENDING — 2026-08-10]

src/models/single_week_ppr/season_projection.py: option C design (per-week
formula on retrospective seasons — synthetic feature rows for missed weeks,
reusing/extending Predictor.predict()'s carry-forward mechanism, plus a new
non-circular prior-seasons-only P(plays) estimator). 8 unit tests passing.
The real end-to-end run against 2023-2025 was intentionally NOT executed —
mid-smoke-test, the Complete Player-Game Panel issue above was raised and
took priority, since Phase 7's "missed week" concept is now understood
differently (a synthetic-row estimate) than the corrected panel's explicit
inferred-zero rows (a real row with data_source provenance). Re-evaluate
Phase 7's design against the corrected panel before running it for real —
it may need to change (e.g. use real inferred-zero rows directly where
available, reserving the synthetic-row mechanism only for weeks with no
row of any kind) rather than just re-running as-is.

Do not simply apply the single-week model's validation framework to an 18-week target.

Treat it as a separate prediction problem.

Preferred approach: aggregate weekly predictions

For each future week:

Week 1 expected PPR
Week 2 expected PPR
...
Week 18 expected PPR

Then:

18-week projection =
Σ expected weekly PPR

The weekly model should incorporate:

* Opponent
* Schedule
* Expected role
* Team/QB context
* Current-season performance
* Injury/status
* Other information available at that point

⸻

8. Account for games played

The 18-week projection must account for the possibility that a player:

* Misses games
* Loses their role
* Changes teams
* Experiences an injury
* Becomes a starter
* Becomes a backup

Therefore the seasonal projection should ultimately represent:

Expected PPR across the entire season, including probability of playing each week.

A useful formulation is:

Expected weekly PPR =
P(player plays) × Expected PPR conditional on playing

Then sum across future weeks.

⸻

Phase 8 — 18-Week Direct Model Benchmark

Build a separate direct seasonal model:

Features available at season projection date
        ↓
Direct 18-week PPR model
        ↓
Season PPR

Compare it against:

Optimized weekly model
        ↓
18 weekly predictions
        ↓
Sum
        ↓
18-week projection

This answers an important question:

Is season-level prediction more accurate when modeled directly, or when constructed from weekly predictions?

⸻

Phase 9 — 18-Week Quantiles / Simulation

Once the weekly model is reliable, use simulation to generate seasonal uncertainty.

Produce:

P25 season PPR
P50 season PPR
P75 season PPR
P90 season PPR

Simulation should account for uncertainty in:

* Weekly performance
* Games played
* Player role
* Other meaningful weekly variables

This is preferable to simply summing P25/P50/P75/P90 independently because season-level distributions are not equivalent to the sum of corresponding weekly quantiles.

⸻

Phase 10 — Save Out-of-Sample Predictions

For every single-week validation observation save:

player
position
season
week
actual_ppr
prediction
model
fold
training_window
weighting_strategy

For quantiles:

p25
p50
p75
p90

For 18-week predictions save:

player
season
projection_date
actual_season_ppr
predicted_season_ppr
model_type

This makes model comparison reproducible without retraining.

⸻

Phase 11 — Final Holdout

Reserve the most recent season as an untouched final holdout, if the data permits.

Before touching it, freeze:

* Features
* Architecture
* Training window
* Recency weighting
* Hyperparameters
* Model-selection decisions

Then evaluate once.

⸻

Final Architecture

                         Player Data
                              │
                    Leakage-Safe Features
                              │
               ┌──────────────┴──────────────┐
               │                             │
        SINGLE-WEEK MODEL              18-WEEK MODEL
               │                             │
     QB / RB / WR / TE                Weekly aggregation
               │                             │
     MSE / Huber / MAE               + games-played model
     Two-stage / Quantiles                    │
               │                             │
      Optimize recency/window         Direct 18-week model
               │                             │
               └──────────────┬──────────────┘
                              │
                    Out-of-Time Evaluation
                              │
                    Final Untouched Season

Recommended development order

1. Leakage audit
2. Time-series evaluation framework
3. Naive baselines
4. Single-week model architecture comparison
5. Training-window / recency optimization
6. Hyperparameter tuning
7. Feature engineering
8. Lock optimized weekly model
9. Build weekly → 18-week aggregation
10. Build direct 18-week benchmark
11. Add seasonal simulation/quantiles
12. Final untouched-season evaluation

Most important principle

The single-week model is the foundation. Optimize it first. Then use it to construct the 18-week projection, while maintaining a separate direct 18-week model as a benchmark.

This gives you a clean answer to both questions:

What is the most accurate next-game PPR model?

and

What is the most accurate way to project full-season PPR?

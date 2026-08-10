NFL Fantasy PPR — Final Modeling Plan

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

Phase 3 — Optimize Historical Training Data

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

Phase 4 — Single-Week Validation

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

Phase 5 — Hyperparameter Tuning

Only after selecting:

1. Model architecture
2. Training window
3. Recency weighting

Then tune model hyperparameters.

This prevents wasting compute optimizing an inferior architecture.

⸻

Phase 6 — Feature Engineering

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

Phase 7 — Build the 18-Week Projection

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

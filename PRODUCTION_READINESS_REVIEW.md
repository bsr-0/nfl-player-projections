# Production-Readiness Review: NFL Player Prediction System

**Review Date:** 2026-02-25
**Reviewer Posture:** Skeptical Senior ML Engineer
**Scope:** Full codebase, configuration, data schemas, evaluation results, test suite
**Verdict:** **NOT PRODUCTION-READY** — 5 critical issues, 8 high-severity issues, multiple moderate findings

---

## Executive Summary

This is a well-architected NFL fantasy football prediction system with genuine thought put into temporal integrity, position-specific modeling, and multi-horizon predictions. The feature engineering pipeline correctly applies `.shift(1)` to rolling windows, the train/test split is strictly season-based, and there is a dedicated leakage detection module. A comprehensive 31-file test suite exists, including a 7-phase ML audit.

**However, the system has contradictory evaluation artifacts that make it impossible to assess true model performance.** One result file claims R²=0.959 while another claims R²=0.672 for what appears to be the same system. The confidence intervals are severely miscalibrated (73% coverage for a nominal 90% interval). Adversarial validation reveals the train and test distributions are nearly perfectly separable (AUC=0.957). No real expert-projection baselines exist for comparison. CI only exercises 3 of 31 test files.

A senior ML engineer at a quantitative shop would not sign off on deploying this system until the critical issues below are resolved.

---

## Table of Contents

1. [Objective and Problem Framing](#1-objective-and-problem-framing)
2. [Data Pipeline and Feature Integrity](#2-data-pipeline-and-feature-integrity)
3. [Splits, Cross-Validation, and Baselines](#3-splits-cross-validation-and-baselines)
4. [Metrics and Error Analysis](#4-metrics-and-error-analysis)
5. [Overfitting Control and Model Complexity](#5-overfitting-control-and-model-complexity)
6. [Data Quality, Label Noise, and Robustness](#6-data-quality-label-noise-and-robustness)
7. [Code Quality, Reproducibility, and Experiment Tracking](#7-code-quality-reproducibility-and-experiment-tracking)
8. [Production Readiness and Ongoing Monitoring](#8-production-readiness-and-ongoing-monitoring)
9. [Human-in-the-Loop and Domain Expertise](#9-human-in-the-loop-and-domain-expertise)
10. [Summary Assessment](#10-summary-assessment)

---

## 1. Objective and Problem Framing

### 1.1 Restated Prediction Task

| Dimension | Value | Source |
|-----------|-------|--------|
| **Primary target** | Utilization Score (0-100 continuous scale) | `src/features/utilization_score.py` |
| **Secondary target** | PPR Fantasy Points (direct, QB dual-train) | `src/models/train.py:235-245` |
| **Horizon** | 1-week, 4-week, 18-week (rest-of-season) | `config/settings.py` MODEL_CONFIG |
| **Granularity** | Player-game (one row per player per week) | `src/utils/database.py:864-879` |
| **Output type** | Regression (continuous points) with uncertainty bands | `src/models/ensemble.py` |
| **Positions** | QB, RB, WR, TE | `config/settings.py:75` |
| **Scoring** | PPR primary; Half-PPR, Standard supported | `config/settings.py:80-110` |

The system predicts a player's **utilization score** (a weighted composite of snap share, target share, rush share, red-zone share, and touch share) for future weeks, then converts that utilization to expected fantasy points via a position-specific Ridge regression (`src/models/utilization_to_fp.py`). For QBs, the system dual-trains on both utilization and direct fantasy points, selecting whichever produces higher R² on held-out data (`src/models/train.py:235-245`).

### 1.2 Downstream Use Alignment

**Finding: AMBIGUOUS** | Severity: **HIGH**

The system serves multiple downstream uses but does not commit to a primary task:

- **Weekly start/sit decisions** (1-week horizon): Requires week-level error distribution and calibration
- **Trade deadline analysis** (4-week horizon): Requires multi-week aggregation accuracy
- **Draft rankings** (18-week horizon): Requires season-long rank stability and total-points error

The `LIMITATIONS.md` document, frontend tabs (Dashboard, Rankings, Draft Assistant, Player Lookup), and evaluation metrics all suggest the system tries to serve all three. However, **the evaluation metrics in `data/ml_evaluation_results.json` appear to be computed on a single combined evaluation** — not stratified by use case. Week-level calibration (critical for start/sit) and rank stability (critical for draft) are not separately reported.

**Remediation:** Define a single canonical "primary task" (e.g., weekly start/sit). Treat other horizons as secondary diagnostics. Report metrics stratified by horizon and use case.

### 1.3 Edge Hypothesis

**Finding: NOT STATED** | Severity: **HIGH**

Nowhere in the codebase, documentation, or configuration is there an explicit, falsifiable hypothesis for why this model should outperform alternatives. The `approach_comparison_results.json:30-35` contains "key_findings" that state:

```json
"Rolling averages (fp_rolling_3) are the most important feature (44.3%)",
"NO data leakage - all features are known before game time"
```

But "rolling averages are important" is not an edge hypothesis. An edge hypothesis would be something like: "By combining utilization trends with opponent-adjusted strength of schedule, this model captures opportunity shifts that expert projections systematically lag by 1-2 weeks." Without such a hypothesis, there is no principled way to evaluate whether the model is working *for the right reasons*.

**Remediation:** Document a clear, falsifiable edge hypothesis. Test it by ablation: does removing the hypothesized signal source reduce performance to baseline levels?

---

## 2. Data Pipeline and Feature Integrity

### 2.1 Data Sources

| Source | Type | Location |
|--------|------|----------|
| `nfl-data-py` weekly data | Player stats, snap counts | `src/data/nfl_data_loader.py` |
| `nfl-data-py` PBP data | EPA, WPA, success rate | `src/data/pbp_stats_aggregator.py` |
| Team weekly stats | Offense/defense aggregates | `data/raw/team_weekly_stats_*.csv` |
| Schedule data | Opponents, game metadata | `src/scrapers/schedule_scraper.py` |
| SQLite database | Joined tables | `data/nfl_data.db` |

Historical data goes back to 2006 (`config/settings.py:46`), with weekly data considered reliable from 2016+ (`config/settings.py:48`).

### 2.2 Temporal Leakage Verification

#### Features Verified as Safe

| Feature | Mechanism | Verified At |
|---------|-----------|-------------|
| Rolling means (3/4/5/8/12 week) | `x.shift(1).rolling(window, min_periods=1).mean()` | `feature_engineering.py:249-251` |
| Rolling std | `x.shift(1).rolling(window, min_periods=2).std()` | `feature_engineering.py:255-257` |
| EWM averages | `x.shift(1).ewm(span=s, adjust=False).mean()` | `feature_engineering.py:266-268` |
| Lag features (1-4 weeks) | `df.groupby("player_id")[col].shift(lag)` | `feature_engineering.py:326` |
| Season-to-date average | `expanding().mean().shift(1)` | `utilization.py:369-371` |
| Position expanding mean | `x.shift(1).expanding(min_periods=1).mean()` | `feature_engineering.py:276-278` |
| Opponent defense stats | SQL: `tds.week = pws.week - 1` | `database.py:870-871` |
| Season position rank | `prev_season_fp_mean` via `.shift(1)` on season-level | `season_long_features.py:749` |
| Position rank (weekly) | Based on `fp_rolling_3` (already shifted) | `season_long_features.py:730-738` |
| Multi-week horizon targets | `shift(-1)` with proper min_periods | `train.py:268-297` |

The leakage test suite (`tests/test_data_leakage.py`) includes concrete numerical verification: for week 5 with fantasy_points = 50, rolling mean should be (10+20+30+40)/4 = 25, not include the current value. This is a good practice.

#### Critical Issue C1: Leakage Guard Inconsistency

**Severity: CRITICAL**

The centralized leakage guard (`src/utils/leakage.py:36-41`) defines these substrings as forward-looking:

```python
_FORWARD_SUBSTRINGS = ("_future", "_next", "_forward")
```

This would correctly flag `sos_next_1`, `favorable_matchups_next_1`, `expected_games_next_1`, etc. However, the enforcement in `train.py:153-160` is:

```python
leaked = [c for c in find_leakage_columns(combined.columns, ban_utilization_score=False)
          if c.startswith(("predicted_", "projection_"))]
```

This **silently ignores** any leakage-flagged column that does not start with `predicted_` or `projection_`. The `sos_next_*` features pass through because they start with `sos_`, not `predicted_`. This creates a **false sense of security**: the leakage module flags them, but the training pipeline ignores the flags.

In this specific case, the `sos_next_*` features are actually **safe during training** because `multiweek_features.py:130-137` uses backward-looking rolling windows when `is_training=True`. But the naming is misleading (`sos_next_5` during training is actually `sos_past_5`), and the guard/enforcement mismatch means a future developer could introduce a genuinely leaky `*_next_*` feature that would slip through undetected.

**Remediation:**
1. Rename training-mode features to reflect their actual semantics (e.g., `sos_backward_5` during training)
2. Either: (a) tighten `train.py` enforcement to block ALL flagged columns, with an explicit allowlist for `sos_*` features, or (b) remove `_next` from `_FORWARD_SUBSTRINGS` and instead check for specific known-leaky patterns
3. Add a test that calls `assert_no_leakage_columns` on the actual feature set used during training and verify it passes

#### Critical Issue C2: Contradictory Result Files

**Severity: CRITICAL**

Two evaluation result files exist with irreconcilable numbers:

| Metric | `ml_evaluation_results.json` | `approach_comparison_results.json` |
|--------|-----|-----|
| **R²** | 0.959 (XGBoost) | 0.672 |
| **RMSE** | 1.67 (XGBoost) | 4.74 |
| **n_features** | 59 (lasso) | 18 |
| **position_rank in features?** | Yes | Noted as removed for leakage |

The `approach_comparison_results.json:37` includes the note: *"position_rank was removed as it used current-game fantasy points (data leakage). Now using only legitimate pre-game features."* Yet `ml_evaluation_results.json` still lists `position_rank` and `season_position_rank` in its feature sets.

It is **impossible to determine the system's true performance** without knowing which result file reflects the current codebase state. An R² of 0.959 on weekly fantasy points would be unrealistically high (weekly fantasy is extremely noisy); even on utilization scores (which are more stable), it warrants scrutiny.

**Remediation:**
1. Delete or clearly deprecate the stale result file
2. Re-run evaluation with the current codebase and save results with metadata (commit hash, date, exact feature list, target variable, evaluation methodology)
3. Document which target variable each result file evaluates (utilization_score vs. fantasy_points)

#### Critical Issue C3: `id` Column as a Feature

**Severity: CRITICAL**

`ml_evaluation_results.json:8` lists `"id"` as a feature. If this is a row identifier or auto-increment integer, the model can memorize training examples by their ID. Even if it's a player_id hash, it allows the model to learn player-specific intercepts without explicit regularization, and these intercepts will fail for unseen players.

Feature importance analysis in `approach_comparison_results.json` does not list `id`, suggesting the two result files used different feature sets — reinforcing the contradiction in C2.

**Remediation:** Explicitly exclude `id`, `player_id`, and any row-index columns from feature sets. Add these to the leakage guard's blocklist. Add a test asserting no identifier columns appear in the feature set.

### 2.3 Feature Engineering Correctness

#### Moderate Issue M1: `sos_next_*` Naming Mismatch

**Severity: MODERATE**

During training (`multiweek_features.py:130-137`), `sos_next_5` is computed as a **backward-looking** rolling mean of past opponents' defense strength — not a forward-looking schedule metric. The variable name is misleading and could cause confusion or errors during model interpretation and maintenance.

During inference (`multiweek_features.py:146-151`), the same feature name switches to truly forward-looking behavior. This dual-semantics pattern is fragile.

**Remediation:** Use distinct feature names for training vs. inference, or document the semantic difference prominently in the feature engineering code.

#### Moderate Issue M2: Schedule Strength Rolling Window Lacks shift(1)

**Severity: MODERATE**

In `multiweek_features.py:133-137`, the backward-looking SOS during training:

```python
result[f'sos_next_{n_weeks}'] = result.groupby(
    ['player_id', 'season']
)['opp_defense_strength'].transform(
    lambda x: x.rolling(n_weeks, min_periods=1).mean()
)
```

This rolling window does **not** use `.shift(1)` — it includes the current week's opponent defense strength. Since the current week's opponent is known before the game, this is arguably not leakage. But it's inconsistent with all other rolling features in the codebase, which uniformly use `.shift(1)`. The inconsistency creates confusion about the leakage protection methodology.

**Remediation:** Add `.shift(1)` for consistency, or add a comment explaining why it's intentionally omitted here.

---

## 3. Splits, Cross-Validation, and Baselines

### 3.1 Temporal Validation

**Train/test split: CORRECT.** The latest available season is held out as the test set (`train.py:124-128, 162-167`). An assertion enforces that the test season does not appear in the training set (`train.py:163`). During in-season operation, the current season is forced to be the test set (`train.py:170-183`).

**Cross-validation: WELL-DESIGNED.** `RobustTimeSeriesCV` (`robust_validation.py:42-67`) provides:
- Season-aware splits (test always after training)
- Scaler fit only on training data
- Optional purge gap (`gap_seasons` parameter)
- Multiple folds using different test seasons

**Minimum data requirements are enforced:**
- 1-week model: >= 3 training seasons
- 4-week model: >= 5 training seasons
- 18-week model: >= 8 training seasons

#### High Issue H1: No Rolling-Origin Cross-Validation

**Severity: HIGH**

The system uses static season-level splits for CV (e.g., train on 2018-2023, test on 2024). There is no rolling-origin or expanding-window CV within seasons. The internal audit (`docs/ML_AUDIT_REPORT.md:34`) acknowledges this: *"Primary backtest uses static feature snapshot, not weekly-refit expanding window."*

A `TimeSeriesBacktester` exists in `src/evaluation/ts_backtester.py`, but the primary evaluation pipeline (`train.py`) does not use it. The reported metrics in the result files come from the static-snapshot approach, which is optimistic because features are computed once over the entire dataset before splitting.

**Remediation:**
1. Make the weekly-refit expanding-window backtester the primary evaluation method
2. Report fold-wise metrics (mean and standard deviation across weeks) to assess stability
3. Deprecate the static-snapshot evaluation results

### 3.2 Baselines

#### Critical Issue C4: No Real Expert-Projection Baselines

**Severity: CRITICAL**

The evaluation metrics module (`src/evaluation/metrics.py`) defines comparison targets against expert projections:

- QB/WR: 8-12% RMSE improvement
- RB: 10-15% RMSE improvement
- TE: 12-18% RMSE improvement

But these are **aspirational targets, not measured comparisons**. There is no code that loads, joins, or evaluates against actual ESPN, Yahoo, FantasyPros, or consensus projections. The `approach_comparison_results.json` contains only one approach ("All Features (Corrected)") — there is no baseline to compare against.

The only baseline implemented is a naive season-average predictor. This is a weak baseline that most simple models will beat. Without comparison to expert projections or market-based ADP expectations, **there is no evidence the model provides value beyond freely available information**.

**Remediation:**
1. Scrape or load consensus expert projections for at least 2 test seasons
2. Implement an ADP-based baseline (draft position -> expected fantasy points from historical data)
3. Implement a "last 3 games average" baseline (simple, commonly used)
4. Evaluate all baselines alongside the model on the same test set and report deltas
5. If the model does not beat at least one strong baseline, mark the system as "no demonstrable edge"

### 3.3 Multi-Season Evaluation

**Finding:** The system appears to evaluate on only one held-out season (the latest). Multi-season backtesting exists in `backtester.py` (`run_multi_season_backtest`), but the primary training pipeline `train.py` uses a single test season.

**Remediation:** Run the multi-season backtest as part of the standard evaluation pipeline. Report metrics for at least 3 test seasons to assess generalization stability.

---

## 4. Metrics and Error Analysis

### 4.1 Metric Suite Assessment

The metrics suite is comprehensive and well-designed:

| Metric | Implementation | Target | Source |
|--------|---------------|--------|--------|
| RMSE | Standard | Position/horizon specific | `metrics.py` |
| MAE | Standard | — | `metrics.py` |
| R² | Standard | 1w: 0.50, 4w: 0.40, 18w: 0.30 | `config/settings.py` |
| Spearman correlation | Custom (scipy-free) | >= 0.65 for top-50 | `metrics.py:18-47` |
| Tier accuracy | Elite/Strong/Flex/Waiver | >= 75% | `metrics.py:50-69` |
| Boom/bust F1 | 20+ pts boom, <5 pts bust | Track precision/recall | `metrics.py:72-101` |
| VOR accuracy | Position-aware replacement levels | Spearman on VOR | `metrics.py:104-120` |
| Within-N accuracy | 7 pts: >= 70%, 10 pts: >= 80% | Predefined targets | `config/settings.py` |
| Calibration | 50/80/90/95% coverage | Max error < 10pp | Defined in evaluation |

This is a strong metric suite. The inclusion of rank-based metrics (Spearman), decision-oriented metrics (tier accuracy, boom/bust), and calibration checks is appropriate for fantasy football.

### 4.2 Critical Evaluation Issues

#### Critical Issue C5: Confidence Interval Miscalibration

**Severity: CRITICAL**

`ml_evaluation_results.json:328-330`:
```json
"uncertainty": {
    "coverage_90": 0.7311770943796394
}
```

A 90% prediction interval covers only **73.1%** of actual outcomes. This is a 17-percentage-point gap — far beyond the stated target of < 10pp calibration error. Users relying on floor/ceiling estimates for start/sit decisions will be systematically overconfident.

**Remediation:**
1. Recalibrate prediction intervals using conformal prediction or post-hoc calibration on a validation set
2. Report coverage at multiple nominal levels (50%, 80%, 90%, 95%) to assess calibration curve shape
3. If intervals cannot be calibrated within 5pp, widen them conservatively and document the limitation
4. Consider quantile regression as an alternative to point-estimate-plus-interval

#### High Issue H2: Adversarial AUC = 0.957

**Severity: HIGH**

`ml_evaluation_results.json:2`:
```json
"adversarial_auc": 0.9566537162568622
```

An adversarial validation score of 0.957 means a classifier can distinguish train from test samples with near-perfect accuracy. This indicates **massive distribution shift** between training and test sets. Possible causes:

1. **Temporal drift**: Feature distributions change across seasons (rule changes, new teams, scoring trends)
2. **Feature leakage**: Some features encode which season the data comes from
3. **Scale differences**: Utilization scores computed with different percentile bounds across seasons

A model trained on one distribution and tested on a very different one may perform well in-sample but poorly on genuine future data.

**Remediation:**
1. Investigate which features are most discriminative between train and test (train a classifier and inspect feature importance)
2. Apply recency weighting more aggressively to reduce distribution shift
3. Consider domain adaptation techniques or detrending features that shift systematically across seasons
4. Re-run adversarial validation after fixes; target AUC < 0.60

#### High Issue H3: Contradictory R² Values (0.959 vs 0.672)

**Severity: HIGH**

As detailed in C2, the two result files show drastically different performance:

- R² = 0.959 would mean the model explains 95.9% of variance — extraordinary for weekly fantasy predictions
- R² = 0.672 would mean the model explains 67.2% of variance — reasonable for utilization scores

The realistic range for weekly fantasy point RMSE is 6-9 points (per the system's own benchmarks in `config/settings.py`). An RMSE of 1.67 on a fantasy-point scale would imply almost-perfect prediction, which is not achievable given the inherent noise in NFL outcomes.

**Most likely explanation:** `ml_evaluation_results.json` reports metrics on **utilization score** (0-100 scale, relatively stable week-to-week) while `approach_comparison_results.json` reports on **fantasy points** (much noisier). But neither file explicitly states its target variable.

**Remediation:** Every evaluation result file must include: target variable name, exact feature list used, commit hash, evaluation date, train/test seasons, and whether the evaluation was static-snapshot or weekly-refit.

### 4.3 Missing Stratified Analysis

**Finding: NO STRATIFIED ERROR ANALYSIS IN RESULTS** | Severity: **HIGH**

The system defines position-specific evaluation benchmarks but does not report:

- Error by **player volume tier** (workhorse vs. committee RB, WR1 vs. WR3)
- Error by **player archetype** (mobile QB vs. pocket passer, slot WR vs. deep threat)
- Error by **game context** (high vs. low Vegas total, dome vs. outdoor, primetime vs. early)
- Error for **rookies vs. veterans** (the system has rookie features but no rookie-specific evaluation)
- Error for **returning-from-injury** players

The `approach_comparison_results.json` feature importance analysis shows `snap_share: 0.0` and `target_share: 0.0` — these are supposedly central to the utilization prediction yet contribute zero importance. This could indicate collinearity issues or that the model is relying on proxy features instead.

**Remediation:**
1. Add stratified evaluation cuts to the backtester output
2. Identify systematic biases (e.g., over-projecting TE1s, under-projecting committee RBs)
3. Convert each discovered bias into a targeted feature or regularization improvement
4. Track bias magnitudes across evaluation runs as regression tests

---

## 5. Overfitting Control and Model Complexity

### 5.1 Model Architecture Assessment

The ensemble architecture is appropriate for this data regime:

| Model | Role | Appropriateness |
|-------|------|-----------------|
| XGBoost | Primary (40% weight) | Excellent for tabular data |
| LightGBM | Secondary | Complementary to XGBoost |
| Ridge Regression | Linear baseline (30% weight) | Good diversity in ensemble |
| Random Forest | Bagging complement (30% weight) | Appropriate |
| 4-week LSTM+ARIMA | Optional hybrid | Questionable — see below |
| 18-week Deep Residual | Optional deep model | Questionable — see below |

The tree-based ensemble is well-suited to the moderate-sized tabular data. Ridge provides valuable regularization and diversity. The stacked ensemble (RMSE=1.60) outperforms weighted averaging (RMSE=1.72).

#### High Issue H4: Complex Models Without Justification

**Severity: HIGH**

The 4-week LSTM (256/128/64 units, 3 layers) and 18-week Deep Residual Network (8 effective layers) are complex architectures for a problem with moderate sample sizes. The `approach_comparison_results.json` reports only one approach with 18 features — not a comparison between simple and complex models.

Without evidence that these deep models significantly outperform the simpler tree ensemble:
- The LSTM adds training complexity, GPU requirements, and debugging difficulty
- The deep residual net has many parameters relative to available 18-week-horizon samples
- PyTorch/CUDA dependency increases deployment complexity

**Remediation:**
1. Run a formal comparison: tree ensemble vs. LSTM hybrid vs. deep residual on the same test set
2. If deep models do not beat the tree ensemble by at least 5% RMSE, prefer the simpler model for production
3. Document the comparison and justification for whichever architecture is chosen

### 5.2 Hyperparameter Tuning

**Finding: WELL-DESIGNED**

Optuna tuning with 100 trials using TPE sampler and TimeSeriesSplit with 5 folds is appropriate. Early stopping at 25 rounds prevents overfitting during boosting. The `FAST_MODEL_CONFIG` provides a reduced search (6.7x fewer trials) for development iteration.

### 5.3 Regularization

| Technique | Present? | Source |
|-----------|----------|--------|
| Ridge L2 | Yes | Ensemble component |
| XGBoost depth/min_child_weight | Yes (Optuna-tuned) | `position_models.py` |
| Early stopping | Yes (25 rounds) | `config/settings.py` MODEL_CONFIG |
| Recency weighting | Yes (halflife: 1w=2, 4w=3, 18w=4 seasons) | `config/settings.py` |
| Feature count capping | Yes (50 per position, adaptive by sqrt(n)) | `config/settings.py` |
| Correlation filtering | Yes (threshold 0.92) | `config/settings.py` |
| VIF filtering | Yes (threshold 10) | `config/settings.py` |
| Target transformation | Yes (Log1p for skewed targets) | `position_models.py` TargetTransformer |

### 5.4 Stability Concerns

#### Moderate Issue M3: Feature Importance Concentration

**Severity: MODERATE**

`approach_comparison_results.json:11-28` shows extreme feature importance concentration:

- `season_position_rank`: 44.32%
- `utilization_score`: 31.76%
- `consistency_score`: 10.10%
- All other features combined: ~14%

The model is overwhelmingly dominated by two features that both encode **historical performance level**. `season_position_rank` is the prior season's rank; `utilization_score` reflects recent usage. This means the model is primarily saying "players who were good last year and are getting lots of snaps will be good this year" — which is already well-known and captured by expert consensus projections and ADP.

If the model's "edge" is entirely in these two features, there may be no incremental value beyond a simple lookup table. The remaining features (opponent strength, weather, Vegas lines, injury indicators, age curves) that should provide actual edge have near-zero importance.

**Remediation:**
1. Evaluate model performance WITH and WITHOUT the top-2 features
2. If removing them degrades performance to baseline levels, the model has no real edge
3. Focus development on features that provide genuine incremental signal (opponent adjustments, injury return patterns, game script)

#### Moderate Issue M4: No Fold-Wise Stability Reporting

**Severity: MODERATE**

The system computes per-fold metrics in `RobustTimeSeriesCV` but does not report fold-wise variability in any result file. Large swings across folds would indicate regime sensitivity or overfitting to specific seasons.

**Remediation:** Report mean +/- std of all metrics across CV folds. If std > 30% of mean, investigate and address the instability.

---

## 6. Data Quality, Label Noise, and Robustness

### 6.1 Data Cleanliness

The system handles missing data systematically (`feature_engineering.py:1-16`):
- Features exceeding 5% missing are flagged (logged but not blocked)
- Imputation: column median first, then 0 for any remaining NaN
- Outliers: flagged at 3 sigma, not removed

This is reasonable but could be improved. Using column median globally (across all seasons) may not reflect the distribution in a specific season. Imputing with 0 for utilization-related features could create misleading signals (0 utilization != missing utilization).

### 6.2 Label Noise

**Single-week fantasy outcomes are extremely noisy.** A running back can score 30 points one week and 5 the next due to game script, touchdowns (high variance), and injury. The system does not explicitly model this noise.

#### High Issue H5: No Distributional Predictions

**Severity: HIGH**

The system produces point estimates plus simple uncertainty bands, but the confidence intervals are miscalibrated (C5). It does not model the actual distribution of outcomes (e.g., quantile regression, mixture density networks, or conformal prediction).

For fantasy football decisions, **the distribution matters as much as the mean**. A player with E[points]=12 and high variance (boom/bust) is strategically different from one with E[points]=12 and low variance (consistent). The system computes boom/bust metrics post-hoc but does not produce boom/bust probabilities as predictions.

**Remediation:**
1. Train quantile regression models (or use XGBoost quantile objective) for 10th, 25th, 50th, 75th, 90th percentiles
2. Expose boom probability (P(pts > 20)) and bust probability (P(pts < 5)) as prediction outputs
3. Calibrate intervals using conformal prediction on a held-out calibration set

### 6.3 Robustness

**Finding:** No systematic robustness checks exist. There is no evaluation of:
- Performance in bad-weather weeks vs. dome games
- Performance in blowouts (garbage time distortion) vs. close games
- Performance on short rest (Thursday games) vs. standard rest
- Sensitivity to removing extreme outlier weeks

The `test_ml_audit.py` Phase 4 tests for "distribution shift & drift" and "season shift resilience" using synthetic data, but does not evaluate on actual weather/blowout subsets.

**Remediation:**
1. Tag historical games with context labels (weather, game script, rest days)
2. Report model performance stratified by these contexts
3. If performance degrades significantly in specific contexts, add context-specific features or separate models

---

## 7. Code Quality, Reproducibility, and Experiment Tracking

### 7.1 Repository Structure

The repo has a clear separation of concerns:

```
src/
  data/       - Data loading, processing, aggregation
  features/   - Feature engineering, utilization scores
  models/     - Training, prediction, ensembling
  evaluation/ - Backtesting, metrics, explainability
  utils/      - Leakage guards, database, calendar, helpers
config/       - Centralized settings
tests/        - 31 test files
api/          - FastAPI serving layer
frontend/     - React SPA
data/         - Database, raw CSVs, model artifacts, results
docs/         - Technical documentation
```

This is well-organized and follows reasonable ML project conventions.

### 7.2 Reproducibility

| Requirement | Status | Details |
|-------------|--------|---------|
| Environment specification | Partial | `requirements.txt` with version pins, but no lockfile |
| Docker | Yes | `Dockerfile` with multi-stage build |
| One-command training | Yes | `python run_app.py --refresh --with-predictions` |
| Random seed | Yes | `random_state: 42` in MODEL_CONFIG |
| Deterministic data loading | Partial | Depends on `nfl-data-py` external API stability |

#### Moderate Issue M5: No Lockfile

**Severity: MODERATE**

`requirements.txt` pins major versions (e.g., `xgboost==2.0.3`) but there is no `requirements.lock`, `poetry.lock`, or `pip-compile` output. Transitive dependencies are not pinned, which can cause reproducibility failures when sub-dependencies update.

**Remediation:** Generate a lockfile using `pip-compile` or switch to Poetry/PDM for deterministic dependency resolution.

### 7.3 Experiment Tracking

#### High Issue H6: No Experiment Tracking System

**Severity: HIGH**

The system uses ad-hoc JSON files for result persistence:
- `data/ml_evaluation_results.json`
- `data/approach_comparison_results.json`
- `data/training_years_optimization.csv`
- `data/feature_engineering_results.json`

These files lack:
- Commit hash or code version
- Exact training/test season specification
- Timestamp of when the evaluation was run
- Which target variable was used
- Hyperparameter configuration that produced the results

This is the root cause of Critical Issue C2 (contradictory results) — without metadata, it's impossible to determine which result file reflects the current state of the system.

**Remediation:**
1. Adopt a lightweight experiment tracking system (even a simple JSON log with mandatory fields: commit_hash, timestamp, target_var, train_seasons, test_season, feature_count, metrics)
2. Every training run must produce a timestamped entry in this log
3. Results files should be named with timestamps or commit hashes, not overwritten

### 7.4 Testing and CI

The test suite is extensive:

| Category | Files | Coverage |
|----------|-------|----------|
| Leakage detection | `test_data_leakage.py`, `test_leakage_guards.py`, `test_target_and_history_causality.py` | Good |
| ML audit | `test_ml_audit.py` (42KB, 7 phases), `test_ml_robustness_15_steps.py` | Excellent |
| Feature engineering | `test_feature_engineering.py`, `test_utilization_score.py` | Good |
| Models & training | `test_models.py`, `test_training_pipeline.py` | Good |
| Integration | `test_integration.py`, `test_api_predictions.py` | Good |
| Domain-specific | `test_rookie_projections.py`, `test_injury_modeling.py`, `test_matchup_aware_prediction.py` | Good |
| Backtesting | `test_backtester.py`, `test_ts_backtester.py` | Good |

#### High Issue H7: CI Only Runs 3 of 31 Test Files

**Severity: HIGH**

`.github/workflows/rubric-compliance.yml:43-47` only runs:

```yaml
pytest -q \
  tests/test_rubric_compliance_checker.py \
  tests/test_metrics_evaluator.py \
  tests/test_production_retrain.py
```

28 test files are never run in CI, including the critical leakage tests (`test_data_leakage.py`), the ML audit suite (`test_ml_audit.py`), and the feature engineering tests. Any code change that re-introduces leakage would not be caught by the CI pipeline.

**Remediation:**
1. Run the full test suite (or at least leakage + audit tests) in CI
2. If full suite is too slow, create a "critical" test marker and run those in CI
3. Add the leakage tests to CI as a **blocking gate** — PRs should not merge if leakage is detected

### 7.5 Code Quality Issues

| Issue | Count | Severity | Files |
|-------|-------|----------|-------|
| Broad `except Exception` handlers | 167 | Moderate | Throughout `src/` |
| `sys.path.insert(0, ...)` hacks | 46 | Low | All `src/*.py` |
| Blanket `warnings.filterwarnings('ignore')` | Multiple | Moderate | `robust_validation.py:21`, `feature_engineering.py:38` |
| Print-based logging | Common | Low | Training pipeline, scrapers |

The broad exception handlers are the most concerning — they allow the system to silently degrade without any alerting. The internal audit report (`docs/ML_AUDIT_REPORT.md:47-49`) identifies this as a hidden failure mode: *"When external data sources fail, features silently fall back to hardcoded defaults. No alert is raised."*

**Remediation:**
1. Replace `except Exception: pass` with specific exception types and logging at WARNING level
2. Install the project as a package to eliminate `sys.path.insert` hacks
3. Replace blanket warning filters with specific patterns

---

## 8. Production Readiness and Ongoing Monitoring

### 8.1 Serving Infrastructure

The system has a functional serving layer:

- **FastAPI backend** (`api/main.py`): REST endpoints for predictions, model insights, backtest results
- **React frontend** (`frontend/`): Dashboard, Rankings, Draft Assistant, Player Lookup, Model Insights
- **Docker deployment** (`Dockerfile`): Multi-stage build, environment-variable configuration
- **Health check**: `GET /api/health`
- **Weekly auto-refresh**: GitHub Actions cron job (`refresh-static-api.yml`, Mondays at 9 AM)

### 8.2 Missing Production Components

#### High Issue H8: No Online Monitoring or Alerting

**Severity: HIGH**

There is no mechanism to:
- Track real-time model performance as new weeks are played
- Compare live predictions against actuals
- Alert when model performance falls below baseline or historical band
- Detect when input data quality degrades (missing features, stale data)

The A/B testing module (`src/evaluation/ab_testing.py`) exists but is evaluation-only — it is not connected to a production monitoring pipeline.

**Remediation:**
1. Implement a weekly "actuals vs. predictions" comparison that runs after each NFL week
2. Track rolling RMSE, Spearman correlation, and tier accuracy against baseline
3. Set alert thresholds (e.g., if 4-week rolling RMSE exceeds 2x historical median, alert)
4. Log all prediction requests and results for post-hoc analysis

#### Moderate Issue M6: No Kill Switch or Fallback

**Severity: MODERATE**

If the model begins producing unreasonable predictions (e.g., after a data pipeline failure), there is no mechanism to:
- Automatically fall back to a simple baseline (e.g., rolling average)
- Disable model predictions and show a "data unavailable" message
- Override specific player predictions with expert-provided values

**Remediation:**
1. Implement a baseline prediction service that can replace model predictions
2. Add a configuration flag to switch between model and baseline serving
3. Add bounds checking on predictions (flag any prediction outside [0, 60] for individual players)

#### Moderate Issue M7: No Retraining Cadence Documentation

**Severity: MODERATE**

The weekly refresh (`refresh-static-api.yml`) refreshes data and regenerates static predictions, but it is unclear whether this involves model retraining or just re-scoring with the existing model on new data. There is no documented:
- Retraining schedule (weekly during season? yearly in offseason?)
- Safe deployment process (shadow mode, A/B comparison before switching)
- Rollback procedure if a new model underperforms

A `RETRAINING_CONFIG` exists in `config/settings.py` but its usage is not documented.

**Remediation:** Create a runbook documenting: when to retrain, how to validate a new model before deployment, how to roll back, and what to do during data outages (bye weeks, lockouts, etc.).

---

## 9. Human-in-the-Loop and Domain Expertise

### 9.1 Explainability

The system includes meaningful explainability:

- **SHAP values** (`src/evaluation/explainability.py`): Feature importance per prediction
- **Top-10 features per position**: Exported as JSON for frontend display
- **Partial dependence plots**: Available for key features
- **Frontend Model Insights tab**: Visualizes feature importance and model behavior

### 9.2 Missing Human Feedback Mechanisms

#### Moderate Issue M8: No Expert Override or Feedback Collection

**Severity: MODERATE**

- No mechanism for domain experts to override specific predictions
- No way to flag "bad" predictions for review
- No feedback collection from end users
- No integration of expert knowledge as prior information

**Remediation:**
1. Add an expert override API endpoint that allows manual adjustments to specific player predictions
2. Implement a feedback mechanism (thumbs up/down, "this prediction was bad") in the frontend
3. Collect and store feedback for future analysis
4. Consider using expert feedback to define "prior" predictions that the model can learn to adjust from

---

## 10. Summary Assessment

### Scorecard

| Section | Rating | Key Finding |
|---------|--------|-------------|
| 1. Objective & Framing | **FAIL** | No edge hypothesis; ambiguous primary use case |
| 2. Data Pipeline & Features | **PASS with issues** | Temporal integrity verified; leakage guard inconsistency |
| 3. Splits, CV & Baselines | **FAIL** | No real expert baselines; no rolling-origin CV in primary pipeline |
| 4. Metrics & Error Analysis | **FAIL** | Contradictory results; miscalibrated intervals; high adversarial AUC |
| 5. Overfitting & Complexity | **PASS with issues** | Good regularization; feature importance concentration |
| 6. Data Quality & Robustness | **PASS with issues** | Reasonable imputation; no distributional predictions |
| 7. Code Quality & Repro | **PASS with issues** | Good test suite; poor CI coverage; no experiment tracking |
| 8. Production Readiness | **FAIL** | No monitoring, no alerting, no kill switch |
| 9. Human-in-the-Loop | **PASS with issues** | SHAP exists; no feedback collection |

### Critical Issues (Must Fix Before Production)

| ID | Issue | Section |
|----|-------|---------|
| **C1** | Leakage guard inconsistency — enforcement only checks `predicted_`/`projection_` prefixes | 2.2 |
| **C2** | Contradictory evaluation results (R²=0.959 vs R²=0.672) — true performance unknown | 2.2 |
| **C3** | `id` column used as a feature — memorization risk | 2.2 |
| **C4** | No real expert-projection baselines — no evidence of edge | 3.2 |
| **C5** | 90% confidence interval covers only 73.1% — severely miscalibrated | 4.2 |

### High Issues (Should Fix Before Production)

| ID | Issue | Section |
|----|-------|---------|
| **H1** | No rolling-origin cross-validation in primary pipeline | 3.1 |
| **H2** | Adversarial AUC = 0.957 — massive train/test distribution shift | 4.2 |
| **H3** | Contradictory R² values without target variable labels | 4.2 |
| **H4** | Complex deep models without comparison to simpler alternatives | 5.1 |
| **H5** | No distributional predictions (boom/bust probabilities) | 6.2 |
| **H6** | No experiment tracking system — results lack metadata | 7.3 |
| **H7** | CI only runs 3 of 31 test files — leakage tests not in CI | 7.4 |
| **H8** | No online monitoring or alerting for deployed predictions | 8.2 |

### Moderate Issues

| ID | Issue | Section |
|----|-------|---------|
| **M1** | `sos_next_*` naming mismatch (backward-looking in training) | 2.3 |
| **M2** | Schedule strength rolling window lacks `shift(1)` unlike other features | 2.3 |
| **M3** | Feature importance extremely concentrated in 2 features (75%) | 5.4 |
| **M4** | No fold-wise stability reporting | 5.4 |
| **M5** | No dependency lockfile | 7.2 |
| **M6** | No kill switch or fallback to baselines | 8.2 |
| **M7** | No retraining cadence documentation | 8.2 |
| **M8** | No expert override or feedback collection | 9.2 |

### Verdict

**This system is NOT production-ready.**

The core temporal integrity of the feature pipeline is sound — the `.shift(1)` methodology is correctly applied to rolling features, lag features, and season-level aggregates. The test suite is impressive in scope. The model architecture (tree-based ensemble) is appropriate.

However, the system fails on the most fundamental question: **Does it work?** The contradictory evaluation results make it impossible to assess true performance. The absence of real expert baselines means there is no evidence of edge. The confidence intervals are unreliable. The train/test distributions are so different that the adversarial validation model can distinguish them with 96% accuracy.

### Recommended Priority Actions

1. **Reconcile evaluation results** — Re-run evaluation on the current codebase with full metadata. Determine true R² and RMSE on fantasy points (not utilization).
2. **Add expert baselines** — Obtain consensus projections for test seasons. If the model does not beat them, acknowledge this honestly.
3. **Fix CI** — Add leakage and audit tests to the CI pipeline.
4. **Calibrate intervals** — Use conformal prediction or post-hoc calibration to fix the 73% coverage problem.
5. **Investigate adversarial AUC** — Determine which features drive the 0.957 score and address the distribution shift.
6. **Add online monitoring** — Track weekly performance against actuals and baselines.

### What Works Well

Despite the critical issues, this system has genuine strengths:

- **Temporal architecture** is sound and well-tested
- **Utilization-based prediction** is a smart intermediate target that reduces noise
- **Position-specific modeling** with appropriate feature sets
- **Multi-horizon support** with appropriate minimum data requirements
- **Comprehensive test suite** (even if CI underutilizes it)
- **SHAP explainability** for prediction transparency
- **Defensive coding** for missing data and edge cases (bye weeks, rookies, injuries)

With the critical issues resolved, this system has the potential to be a valuable fantasy football decision-support tool — positioned as an augmented tool with honest uncertainty bounds, not an oracle.

---

*Review conducted on 2026-02-25 against the current state of the repository.*

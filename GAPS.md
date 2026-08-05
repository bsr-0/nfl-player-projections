# NFL Player Projections — Gap Analysis & Audit Report

**Audit date:** 2026-08-03
**Scope:** Full codebase audit for prediction accuracy gaps, market bias, team context, and missing signals. Focused on full-season (draft) projections with notes on weekly horizons.
**Perspective:** What would a senior sports statistician flag as critical issues in this ML system?

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [PROJECT_NOTES.md Corrections](#2-project_notesmd-corrections)
3. [Market/ADP Calibration Bias](#3-marketadp-calibration-bias)
4. [Team Context & Offensive Scheme Gaps](#4-team-context--offensive-scheme-gaps)
5. [Player-Team Assignment & Roster Dynamics](#5-player-team-assignment--roster-dynamics)
6. [Utilization Score Pipeline Gaps](#6-utilization-score-pipeline-gaps)
7. [Model Architecture & Validation Weaknesses](#7-model-architecture--validation-weaknesses)
8. [Feature Engineering Gaps](#8-feature-engineering-gaps)
9. [Data Pipeline Gaps](#9-data-pipeline-gaps)
10. [Prioritized Remediation Roadmap](#10-prioritized-remediation-roadmap)
11. [External Techniques & Variables This Project Should Adopt](#11-external-techniques--variables-this-project-should-adopt)

---

## 1. Executive Summary

The system has strong fundamentals: leakage-safe temporal CV, position-specific utilization scores, component prediction (stat lines → fantasy points), and a well-guarded feature pipeline. However, several structural gaps limit prediction accuracy:

**Critical gaps:**
- ADP/ECR used as a raw feature without bias correction — model absorbs consensus rather than learning to beat it
- Team offensive scheme captured at a coarse level (pass rate, momentum) but missing coaching staff, personnel grouping, formation tendencies, and play-calling evolution
- Utilization-to-FP conversion trained on OOF predictions is circular when ensemble is weak
- Rookies not integrated into the main ensemble; early-season projections are effectively zero-filled
- No separate full-season projection model — just sums weekly 1-week predictions, inheriting per-week noise
- Players predicted independently — no modeling of teammate/opponent interactions (GNN, correlation matrices)
- Weather data ingested but not wired into ML features; red zone allocation buried inside utilization composite
- Raw opponent FPA used without DVOA-style schedule strength adjustment

**Current production metrics (v9, 2025 holdout):**

| Position | R² | Spearman ρ | Notes |
|----------|-----|-----------|-------|
| QB | 0.085 | 0.334 | Worst position; early-season R²=0.166 |
| RB | 0.224 | 0.488 | Best R² |
| WR | 0.243 | 0.464 | Best R² |
| TE | 0.118 | 0.405 | Small sample, limited signal |

These were generated at v9 (9–14 features). Current model is v22 (33–37 features) but has not been re-benchmarked on a holdout. The predictive ceiling documented in PROJECT_NOTES.md (R²≈0.27, loses to trailing-average heuristic at R²=0.279) still stands as the core constraint.

---

## 2. PROJECT_NOTES.md Corrections

Issues found in `PROJECT_NOTES.md` that need updating:

### Factual corrections needed

1. **Line 13**: States "FEATURE_VERSION = 21" — current setting is `FEATURE_VERSION = "v22"` (bumped for dest_team features). Update to v22.

2. **Line 13**: States "Ridge α=10,000 as walk-forward production default" — but `position_target_type` is now `"component"` for all positions (QB, RB, WR, TE). The production path is component prediction (individual stat Ridge models), not a single Ridge on utilization. The α=10,000 Ridge is still the default for non-component mode, but production mode bypasses it.

3. **Line 30**: Utilization Score weights table is stale. The weights are now data-driven via `fit_utilization_weights()` in `utilization_weight_optimizer.py` — the hardcoded weights in `config/settings.py` are only defaults; the optimizer replaces them during training. Document this.

4. **Lines 62–71**: Production metrics table is from v9 models. These are stale — v22 models have not been benchmarked. Mark this table as "v9, stale" and add a note that current production metrics are unknown.

5. **Line 86**: "variance compression" insight is correct but the documented fix ("lifting correlation via better features") has not produced measurable R² lift in 3 months. Consider whether the theoretical ceiling has been reached with available data.

6. **Line 126–129**: "Pure ML pre-draft ranking shelved — loses to ADP" is the key finding but the *reason* is not fully documented. The reason: ADP embeds offseason signals (camp reports, depth chart battles, coaching changes, free agency context) that the model has zero access to. This is a data gap, not a model gap. The correct fix is ingesting those signals, not blending with ADP.

7. **Line 193–196**: Data gaps section lists "Real air yards: estimated" and "Red zone data: estimated from TDs" — but `ngs_avg_intended_air_yards` is available from NGS (2018+) and is used in features. Clarify which air yards metric is estimated vs. actual.

### Structural issues

8. **Missing section**: No documentation of the component prediction architecture (predict stat lines, assemble FP). This is the actual production path and is only mentioned once in the System Overview.

9. **Missing section**: No documentation of the utilization weight optimization process. The weights are no longer static; they're fitted from data each training run.

10. **Missing section**: No documentation of the draft prep mode (June–August) training pipeline changes made today.

---

## 3. Market/ADP Calibration Bias

### Current state

ADP/ECR data is integrated as a **raw feature** (`preseason_ecr`) in the ML pipeline. The model trains with ECR as one of 33–37 features and learns whatever correlation exists between consensus rank and actual performance.

Additionally:
- `market_anchor` in `preseason_projector.py` fits an ECR → PPG curve from prior-season calibration for cold-start projections
- `compute_market_projections.py` aggregates player prop lines across bookmakers and converts to fantasy points
- `adp_value_score`, `adjusted_adp_value` derived from position ranking (not actual ADP data)

### The bias problem — TWO contamination points

**The model is structurally biased toward market consensus at both training AND inference time.** There are two distinct contamination points that must both be removed:

#### Contamination Point 1: Pre-training (feature-level)

`preseason_ecr` is included as a raw input feature in the 33–37 causal features. When ECR is predictive (consensus rank correlates ~0.5-0.6 with actual PPG), the model learns to lean on it. Predictions regress toward ADP. The model cannot beat ADP if ADP is a dominant feature.

#### Contamination Point 2: Post-training (inference-level) — `preseason_projector.py`

After the ML model produces raw predictions, a **market anchor system** applies post-processing calibration that blends predictions back toward ADP consensus. This is a full pipeline, not just a cold-start fallback:

1. **`_fit_market_anchor_curve()`** — fits a position-specific ECR → PPG regression curve from prior-season data (e.g., "ECR rank 5 → ~18 PPG for WR")
2. **`_attach_market_anchor()`** — adds a `market_anchor` column to every player's prediction row, converting their ECR rank into an expected PPG via the fitted curve
3. **`market_gap = market_anchor - raw_pred`** — computes how far the ML prediction diverges from ADP consensus for each player
4. **Blending at inference** — the market anchor values are stored in model artifacts (`market_anchor_curves` dict), serialized/deserialized with the model, and applied at prediction time to anchor outputs toward ADP
5. **Scope**: This applies to ALL preseason predictions, not just cold-start players. Every player with a valid `preseason_ecr` gets market-anchored

**Net effect**: Even if you remove `preseason_ecr` from training features (Point 1), the post-processing market anchor (Point 2) will still pull predictions back toward ADP. Both must be removed for pure ML.

#### Additional bias vectors

3. **No de-biasing**: There is no mechanism to identify when ADP is systematically wrong (overvalued players, undervalued players) and adjust. The model absorbs market consensus without contrarian signal.

4. **April 2026 kill gate**: The council process documented that "pure ML pre-draft ranking loses to ADP on both 2024 and 2025." The prescribed fix was blending with ADP — but this guarantees the system can never beat ADP, only match it. A contrarian model that identifies *where* ADP is wrong would be more valuable.

### All post-processing layers that touch predictions (full inventory)

After the base ML model produces `raw_pred`, predictions pass through multiple post-processing layers before becoming final output. Each must be evaluated for market bias:

#### Layer 1: UpstreamCalibrator (`preseason_projector.py:371-439`) — REMOVE (market-biased)

A Ridge regression that takes `raw_pred` + calibration features and produces an adjusted prediction. **The calibration features include `market_anchor` and `market_gap`** (lines 245-282 — in `CALIBRATION_FEATURES_BY_POSITION` for all four positions). This means the calibrator is explicitly trained to pull predictions toward ADP.

Additionally, after the Ridge adjustment, there is a **second explicit market blending step** (lines 426-437):
```python
pred[market_valid] = pred[market_valid] + market_weight[market_valid] * (
    market_values[market_valid] - pred[market_valid]
)
```
This directly interpolates the prediction toward the market anchor value. The weight is controlled by `CALIBRATOR_MARKET_WEIGHT_CAP` — up to **36% for RB, 34% for WR, 28% for TE, 22% for QB**. These are enormous market bias weights.

**Action**: Remove `market_anchor`, `market_gap`, and `market_gap_x_low_information` from `CALIBRATION_FEATURES_BY_POSITION`. Remove the explicit market blending step. The calibrator can remain if it uses only non-market features (confidence, support class, volume stats).

#### Layer 2: MarketAnchorCurve (`preseason_projector.py:302-340`) — REMOVE (market-biased)

Fits a log-linear regression `ECR → PPG` per position from prior-season data. Used to generate the `market_anchor` column that feeds Layer 1. Without this, Layer 1's market features have no values.

**Action**: Remove the `MarketAnchorCurve` class, `_fit_market_anchor_curve()`, and `_attach_market_anchor()`. Remove `market_anchor_curves` from model serialization.

#### Layer 3: Conformal calibration (`position_models.py:384-503`) — KEEP (legitimate)

Computes residual distributions from OOF predictions to produce calibrated confidence intervals. This is standard uncertainty quantification — it adjusts interval *width*, not point predictions. No market data involved.

**Action**: Keep as-is. This is good statistical practice.

#### Layer 4: IsotonicRegression calibrator (`position_models.py:341-342`) — EVALUATE

Isotonic regression fitted on OOF predictions vs. actuals to correct systematic prediction biases (e.g., the model consistently underpredicts high scorers). This is a legitimate post-processing step IF the training data doesn't contain market features. However, if the upstream model was trained with `preseason_ecr`, the isotonic calibrator learns to correct predictions that already embed market bias — removing ECR could change the calibration surface.

**Action**: Keep, but re-fit after removing market features from training. Verify it doesn't introduce new systematic bias.

#### Layer 5: VeteranEliteCalibration / FragileRoleCalibration (`preseason_projector.py:461-462`) — EVALUATE

Legacy calibration objects stored on the projector. These apply cohort-specific adjustments (e.g., elite veterans get boosted, fragile role players get discounted).

**Action**: Evaluate whether these encode market-derived signals. If they were fit on data that included market anchors, they may have absorbed market bias indirectly. Re-fit after market feature removal.

#### Layer 6: `adp_value_score` / `adjusted_adp_value` (`season_long_features.py:803-808`) — REMOVE (market-biased)

Derived ADP features injected during feature engineering. These translate `preseason_ecr` into a "value score" and "adjusted value" that enter the model as input features.

**Action**: Remove from feature engineering pipeline along with `preseason_ecr`.

### Recommended approach: pure ML with ADP as validation, not feature

**For 2026 draft projections**, the pipeline should:

1. **Remove all market-biased layers** (Layers 1-market-features, 2, and 6 above):
   - `preseason_ecr` from CAUSAL_FEATURES
   - `market_anchor`, `market_gap`, `market_gap_x_low_information` from CALIBRATION_FEATURES_BY_POSITION
   - `MarketAnchorCurve` class and all `_fit_market_anchor_curve()` / `_attach_market_anchor()` calls
   - The explicit market blending step in `UpstreamCalibrator.calibrate()`
   - `adp_value_score` / `adjusted_adp_value` from season_long_features.py

2. **Re-fit legitimate layers** (Layers 3-5) after market feature removal to ensure they don't carry residual market bias from prior training.

3. **Use 2025 ADP vs. 2025 actuals as validation baseline** — after training the model without ADP, compare model rankings to 2025 ADP rankings and 2025 actual PPG. Measure:
   - Where the model agreed with ADP and was right (consensus correct)
   - Where the model agreed with ADP and was wrong (consensus wrong — model needs better features)
   - Where the model disagreed with ADP and was right (model edge — the value-add)
   - Where the model disagreed with ADP and was wrong (model weakness — investigate features)

4. **Create an "ADP divergence" report** — compute `model_rank - adp_rank` per player as a post-hoc analysis tool (NOT as a feature or blending signal). Present this to the user as actionable insight showing where the pure ML model disagrees with market.

### Validation plan using 2025 data

With the new draft prep mode (2026 as test, 2018–2025 as training):
- Train model A: **with** `preseason_ecr` in features (current behavior)
- Train model B: **without** `preseason_ecr` (pure ML)
- Backtest both on 2025 holdout (most recent completed season with ADP data)
- Compare: which model has better Spearman ρ for full-season PPG rankings?
- If model B (pure ML) has worse ρ, the gap tells you exactly how much "ADP signal" the model is borrowing and what features need to replace it.

---

## 4. Team Context & Offensive Scheme Gaps

### What IS captured

| Feature | Source | Granularity |
|---------|--------|-------------|
| `team_a_pass_rate` | PBP aggregation | Per-team per-week |
| `neutral_pass_rate` | PBP (script-adjusted) | Per-team per-week |
| `team_momentum` | 60-30-10 weighted composite | Per-team per-week |
| `team_prior_season_wins` | Schedule table | Per-team per-season (lagged) |
| `team_qb_pass_epa_per_att` | PBP QB stats | Per-team per-season (lagged) |
| `current_qb_epa_per_att` | Roster-sourced QB1 identity | 3-season career avg |
| `team_motion_rate` | FTN charting (if available) | Per-team per-season |
| `team_play_action_rate` | FTN charting (if available) | Per-team per-season |
| Points scored, rushing/passing yards | Team stats | Per-team per-week |

### What IS NOT captured (critical gaps)

**A. Position-specific target/carry allocation within team**

The system knows a team's pass rate but NOT how targets are distributed among positions. A 65% pass rate means different things depending on whether the team runs 3-WR sets (spread) or 2-TE sets (heavy). Missing features:

- `team_rb_target_share` — % of team targets going to RBs
- `team_wr_target_share` — % to WRs (further split: WR1/WR2/WR3 concentration)
- `team_te_target_share` — % to TEs
- `team_target_concentration` — Herfindahl index of target distribution (concentrated vs. spread)
- `team_rb_rush_share` — lead back vs. committee indicator

These are computable from existing `player_weekly_stats` data — just need team-level aggregation.

**B. Coaching staff and scheme identity**

No data source for:
- Head coach, OC, DC identity or tenure
- Scheme philosophy (air raid, West Coast, zone run, power run)
- OC changes mid-season (e.g., coordinator fired week 10)
- First-year coach adjustment factors

Impact: When a new OC arrives (e.g., Ben Johnson to Bears in 2025), the target distribution changes dramatically. The model sees stale historical team tendencies that no longer apply. This is a first-order signal for draft projections.

**C. Personnel grouping and formation data**

No tracking of:
- 11, 12, 13, 21 personnel snap percentages
- Shotgun vs. under-center rate per team
- Empty backfield frequency
- Motion usage frequency

Available from FTN charting data (`nfl.import_ftn_data()`, ~48K plays/season) but not ingested into the main pipeline.

**D. Tempo and play volume**

- `pace_sec_per_play` exists in `team_stats` but is NOT used as a feature
- Plays per game not computed as a standalone feature
- Fast-tempo teams generate more fantasy opportunities (more plays = more chances for points)
- Missing: `team_plays_per_game`, `team_pace_rank`

**E. Red zone scheme tendencies**

- Red zone attempts/scores exist in `team_stats` at the team level
- But no breakdown of red zone play calling: pass vs. rush inside the 20, inside the 10
- No team-level red zone target allocation (does this team throw to the TE in the red zone, or hand off to the RB?)
- This directly affects TD projection accuracy for all positions

### Impact on full-season projections

For draft projections, these team context gaps are especially damaging because:
1. Free agents are assigned to new teams — the model needs to know the *new team's scheme*, not just their pass rate
2. Coaching changes reset team tendencies — the model's historical team data becomes stale
3. Depth chart uncertainty is highest pre-season — the model can't distinguish WR1 from WR3 without scheme context

---

## 5. Player-Team Assignment & Roster Dynamics

### What IS handled

The feature engineering pipeline (`feature_engineering.py:1978-2069`) has a solid team-change detection system:

- `team_changed` binary flag (1 if team differs from prior season)
- `dest_team_pos_tgt_pg` — destination team's historical targets/game for position (3-season rolling)
- `dest_team_pos_carry_pg` — destination team's RB carry volume
- `scheme_fit_score` — positional preference vs. destination team pass rate
- `weeks_on_team` — cumulative weeks since joining current team

### What IS NOT handled (critical for 2026 draft projections)

**A. New team's 2026 coaching staff context**

When a player signs with a new team, `dest_team_pos_tgt_pg` uses the *historical* target volume of that position on the destination team. But if the team also hired a new OC, the historical target distribution is irrelevant. Example: A WR signs with a team that historically targeted TEs heavily — but the new OC runs a spread offense. The model would underproject this WR.

**B. Depth chart competition**

The system detects that a player is on a new team but doesn't model:
- How many competitors exist at the same position on the new team
- Whether the player is the clear starter or in a timeshare
- The `depth_charts` table (591K rows) is in the DB but NOT used in the main pipeline

**C. Snap count projection for new-team players**

`dest_team_pos_tgt_pg` approximates opportunity but doesn't project snap share. A WR signing as the WR2 on a pass-heavy team needs different snap projections than the same WR signing as WR1 on a run-heavy team.

**D. Mid-season roster changes**

- Trades detected but rolling features carry over from the previous team
- No reset of rolling averages at the trade date
- No "injury replacement usage bump" signal (when the WR1 goes down, WR2 gets a snap share spike — the model can't predict this proactively)

---

## 6. Utilization Score Pipeline Gaps

### Current architecture

Utilization scores (0–100, percentile-normalized) combine position-specific opportunity metrics:
- RB: snap share (20%), rush share (25%), target share (20%), red zone share (20%), touch share (10%), high-value touches (5%)
- WR: target share (30%), air yards share (25%), snap share (15%), red zone targets (20%), route participation (5%), high-value touches (5%)
- TE: target share (35%), snap share (25%), air yards (20%), red zone targets (20%)
- QB: dropback share (35%), rush attempts (20%), red zone attempts (25%), pressure-adjusted (20%)

Weights are now data-driven (optimized via `fit_utilization_weights()` each training run), not the hardcoded defaults above.

### Gap: utilization → FP conversion is circular

The `UtilizationToFPConverter` is a two-stage tree ensemble (RF + XGBoost blend) trained on **OOF-predicted** utilization scores, not actual utilization. The rationale is correct (reduces train/serve distribution mismatch), but:

1. If the ensemble that generates OOF utilization predictions is weak (and it is — R²≈0.27), the converter learns from noisy signal
2. No mechanism to detect when OOF predictions are garbage and reject them
3. The converter calibration is fit on train/val split, not true OOF — weaker than cross-validated estimates

### Gap: utilization doesn't capture efficiency

Utilization scores measure *opportunity* (targets, snaps, carries) but not *efficiency* (yards per carry, catch rate, TD rate). Two players with identical utilization scores can have vastly different fantasy output based on:
- Yards per carry (5.0 vs. 3.5 YPC for RBs)
- Catch rate (70% vs. 55% for WRs)
- TD rate (0.08 vs. 0.03 TDs per target)
- Yards after catch (breakaway speed)

The converter attempts to account for this by including efficiency features, but they're secondary to the utilization input. A better approach: predict efficiency and opportunity separately, then multiply.

### Gap: utilization scores don't account for team context

A RB with 20% target share on a team that throws 40 passes/game gets 8 targets. The same 20% on a team that throws 25 passes/game gets 5 targets. The utilization score treats these identically because it's a *share*, not a *volume*. The system needs:
- `utilization_volume` = share × team opportunity (e.g., target_share × team_pass_attempts)
- This converts relative shares to absolute opportunity counts

---

## 7. Model Architecture & Validation Weaknesses

### 7.1 No dedicated full-season model

The system has three horizon models (1-week, 4-week, 18-week) but no dedicated full-season projection model. For draft projections, the system either:
- Sums 18 weekly 1-week predictions (inheriting per-week noise × 18)
- Uses the 18-week deep model (disabled in component mode)

A dedicated full-season model would:
- Predict total season PPG directly (not sum of weekly predictions)
- Use season-level features (prior-season totals, career trajectory, age curve, team context)
- Avoid compounding weekly prediction errors

### 7.2 Rookie projections not integrated

The `advanced_rookie_injury.py` module has a sophisticated rookie projector with draft capital weighting, archetype classification, and week-weighted blending. But it is NOT wired into the main ensemble pipeline. Rookies in the main pipeline:
- Are excluded from training data if they have <5 games of history
- At prediction time, missing lag/rolling features are zero-filled
- No draft capital, college production, or comparable player matching

For draft projections, rookies represent 25-35% of interesting draft picks. This is a critical gap.

### 7.3 Component predictor not reconciled with ensemble

The component predictor (predict individual stats → assemble FP) and the main ensemble (predict utilization/FP directly) are independent pipelines. No explicit check that their outputs are calibrated to the same scale. When `position_target_type = "component"` (current production), the ensemble pipeline is bypassed entirely.

### 7.4 Validation methodology concerns

| Issue | Severity | Detail |
|-------|----------|--------|
| QB target selection on val, not test | Medium | Trains both util and FP models, picks winner on validation split, then refits on all data — winner may not generalize |
| Hyperparameter tuning overfits to recent data | Medium | Optuna subsamples last 60% (recent) + 40% from older — hyperparams may not generalize to next season |
| Conformal calibration not externally verified | Medium | 90% CI coverage of 73% documented (17pp gap) — calibration computed on OOF, not true holdout |
| No learning curves | Low | No train-size-vs-error analysis — can't tell if more data helps or model is saturated |
| Uncertainty blending weights arbitrary | Medium | 0.5 heteroscedastic + 0.3 conformal + 0.2 ensemble — no optimization or sensitivity analysis |
| Multi-week CI scaling inconsistent | Low | n_weeks^0.4 in multi-week model vs. sqrt(n_weeks) in single-week — different assumptions, same positions |

### 7.5 Overfitting signals

- Train/OOF RMSE ratio threshold is 1.3 — detected but not acted on (just warned)
- XGBoost/LightGBM with 500+ rounds and early stopping — the early stopping is on a temporal split, which is good
- Ridge α=10,000 provides strong regularization but also compresses prediction variance (documented as "variance compression")
- Component predictors use Ridge with α fitted from CV — moderate overfitting risk for per-stat models with small sample sizes

---

## 8. Feature Engineering Gaps

### 8.1 Missing features by impact tier

**Tier 1: High impact, data available now**

| Feature | Source | Why it matters | Current status |
|---------|--------|---------------|----------------|
| **Depth chart position** | `depth_charts` table (591K rows in DB) | Distinguishes WR1 from WR3; critical pre-season | In DB, NOT used |
| **Contracts / contract year** | `contracts` table (51K rows in DB) | Contract year players historically outperform | In DB, NOT used |
| **Weekly PFR advanced stats** | `nfl.import_weekly_pfr()` | Drops, pressures, bad throws — QB predictive signal | Not ingested |
| **Position-specific target allocation** | Computable from `player_weekly_stats` | Team targets RBs X%, WRs Y%, TEs Z% | Not computed |
| **Team plays per game / tempo** | Computable from `team_stats.pace_sec_per_play` | More plays = more fantasy opportunity | In DB, NOT used |

**Tier 2: Medium impact, requires new ingestion**

| Feature | Source | Why it matters |
|---------|--------|---------------|
| **Coaching staff identity** | NFL.com, ESPN rosters | New OC = scheme change = target distribution shift |
| **Personnel grouping %** | FTN charting data (`nfl.import_ftn_data()`) | 11 vs. 12 personnel snap % predicts TE/WR opportunity |
| **Weather (outdoor games)** | Weather API | Cold/wind depresses passing; rain increases fumbles |
| **Seasonal PFR (prior-season summary)** | `nfl.import_seasonal_pfr()` | Pre-season drop rate, bad throw % for cold-start |
| **Vegas preseason win totals** | `nfl.import_win_totals()` | Team quality proxy for full-season game script |

**Tier 3: Lower impact or high implementation cost**

| Feature | Source | Why it matters |
|---------|--------|---------------|
| College production (rookie projection) | cfbd.com API | Target share, YAC rate predict NFL translation |
| Combine measurables | `combine_data_v2` table (in DB, unused) | Speed score for RBs; burst metrics for WRs |
| Referee tendencies | `nfl.import_officials()` | Small effect; some refs = more penalties = more drives |
| Training camp reports | RSS/NLP pipeline | What ADP experts have that the model doesn't |

### 8.2 Rolling window limitation

Causal mode uses only 3-week rolling windows (`CAUSAL_ROLLING_WINDOW = 3`). No lag features in causal mode. Full mode supports 3/4/5/8/12-week windows and 1-4 week lags, but full mode is not the production path.

For full-season projections, 3-week rolling windows are irrelevant — the model needs:
- Prior-season totals (16-17 game sample)
- Career trajectory (multi-season trend)
- Age-adjusted decline curves (already computed but underweight in causal mode)

### 8.3 Opponent defense features are coarse

Current: `opp_fpts_allowed` — a single number per position per opponent per week (4-week rolling, shifted 1 week).

Missing:
- Position-specific defensive weakness: "DEF allows 8.5 rec/game to slot WRs"
- Coverage scheme tendencies: man vs. zone frequency (determines WR archetype matchup value)
- Defensive personnel injuries: "CB1 out → 25% more yards allowed to WRs"
- Red zone defensive efficiency by position

---

## 9. Data Pipeline Gaps

### 9.1 Data in DB but not used

| Table | Rows | Status | Impact if wired in |
|-------|------|--------|-------------------|
| `depth_charts` | 591K | Unused | Starter/backup distinction pre-season |
| `contracts` | 51K | Unused | Contract year flag, guaranteed money as motivation proxy |
| `combine_data_v2` | 8,968 | Unused | Rookie speed score, athletic measurables |
| `team_scheme_tendencies` | — | Defaults to 0.5 | Offensive formation/play-calling scheme |

### 9.2 Known data quality issues

1. **LAR/JAC team code mismatch**: `player_weekly_stats` uses `LAR`/`JAC`, `schedule` uses `LA`/`JAX`. Opponent features for Rams and Jaguars players in 2025 are still dead. Documented in PROJECT_NOTES but not fixed.

2. **Snap counts pre-2018**: Zero-filled. No nflverse source available. Affects ~12 of 20 available seasons (2006–2017).

3. **Injury severity not tracked**: Only injury status (Out, Doubtful, Questionable, Probable) — no tissue type (ACL vs. hamstring), no recovery timeline precedent.

4. **Depth charts are preseason-only**: The `depth_charts` table has weekly granularity but the pipeline only uses week 1 (preseason). In-season depth chart changes are invisible.

5. **ADP data source**: Uses dynastyprocess/data mirror of FantasyPros ECR. Not the raw ADP from drafts — it's Expert Consensus Rank. Different metric, different bias profile.

### 9.3 No data pipeline resilience

Documented in Agent Directive V7 audit: no DAG, no idempotency, no schema validation on ingestion. Data loads can partially fail silently. The `except Exception: pass` pattern (documented as "recurring bug #1") has caused silent feature death for Vegas, injuries, and opponent defense.

---

## 10. Prioritized Remediation Roadmap

### Phase 1: Immediate wins (data exists, just needs wiring)

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1.1 | Wire `depth_charts` table into features (starter/backup/WR1/WR2/WR3 rank) | High — critical for draft projections | Low |
| 1.2 | Wire `contracts` table (contract year flag, APY) | Medium — empirical contract year bump | Low |
| 1.3 | Compute team-level position target allocation from `player_weekly_stats` | High — team targets to RB/WR/TE % | Low |
| 1.4 | Use `pace_sec_per_play` and compute `team_plays_per_game` | Medium — tempo = opportunity | Low |
| 1.5 | Fix LAR/JAC team code normalization | Low–Medium — currently dead opponent features | Low |
| 1.6 | Convert utilization shares to volume (`share × team_attempts`) | Medium — absolute opportunity, not relative | Low |

### Phase 2: ADP de-biasing and pure ML validation

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 2.1 | Train model without `preseason_ecr` feature; compare to ADP-inclusive model on 2025 holdout | Diagnostic — quantifies market dependence | Medium |
| 2.2 | Compute "ADP divergence" signal (`model_rank - adp_rank`) as user-facing output | High user value — identifies where model disagrees with market | Low |
| 2.3 | Time-decay `preseason_ecr` weight (full week 0 → zero by week 4) if feature retained | Medium — prevents stale market anchoring | Low |
| 2.4 | Validate 2025 ADP vs. 2025 actuals as calibration baseline | Diagnostic — establishes how good ADP itself is | Low |

### Phase 3: Team context enrichment

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 3.1 | Ingest coaching staff DB (HC, OC, DC identity per team per season) | High for draft projections — scheme change signal | Medium |
| 3.2 | Compute red zone target/carry allocation per team per position | Medium — improves TD projection | Low |
| 3.3 | Ingest FTN charting data for personnel grouping (11/12/13 % per team) | Medium — WR/TE opportunity split | Medium |
| 3.4 | Ingest weekly PFR advanced stats (drops, pressures, bad throws) | High for QB accuracy | Medium |

### Phase 4: Model architecture improvements

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 4.1 | Integrate rookie projector (`advanced_rookie_injury.py`) into main ensemble | High for draft — 25-35% of interesting picks are rookies | Medium |
| 4.2 | Build dedicated full-season projection model (predict season PPG directly) | High — avoids compounding weekly noise × 18 | High |
| 4.3 | Add efficiency features alongside utilization (YPC, catch rate, TD rate as separate inputs) | Medium — utilization ≠ production | Medium |
| 4.4 | Reconcile component predictor calibration with ensemble scale | Medium — consistency across pipelines | Medium |
| 4.5 | Externally validate conformal CI coverage on true holdout (fix 73%/90% gap) | Medium — trust in uncertainty estimates | Low |

### Phase 5: Data pipeline hardening

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 5.1 | Add schema validation on all data ingestion (column types, ranges, non-null) | Medium — prevents silent data corruption | Medium |
| 5.2 | Replace all `except Exception: pass` with structured logging | Medium — prevents silent feature death | Low |
| 5.3 | Add in-season depth chart tracking (use weekly `depth_charts` snapshots) | Medium for in-season accuracy | Low |
| 5.4 | Implement injury severity tiers (tissue type → recovery timeline) | Low–Medium | Medium |

---

## 11. External Techniques & Variables This Project Should Adopt

Cross-referenced against open-source projects, academic papers, and documented methodologies from top commercial projection systems (PFF, 4for4, SaberSim, numberFire, FantasyPoints). Organized by category with notes on what this project already has vs. what's missing.

### 11.1 Missing Features & Data Signals

#### A. Red Zone / Goal-Line Per-Player Allocation (HIGH IMPACT)

**What exists:** PBP aggregator computes team-level `red_zone_opportunity_rate` and `goal_line_opportunity_rate`. Utilization weights include `redzone_share` and `redzone_targets`.

**What's missing:** Per-player rolling red zone target share, red zone carry share, and goal-line carry share as explicit features. The utilization score aggregates these into a single number, but the model never sees the raw per-player allocation. A back with 4 goal-line carries/week vs. 0 is the biggest single predictor of TD variance, and that signal is buried inside the utilization composite.

**Features to add:**
- `rz_target_share_roll3` — player's share of team red zone targets (3-week rolling)
- `rz_carry_share_roll3` — player's share of team red zone carries
- `goalline_carry_share_roll3` — inside-5 carries per game
- `rz_td_rate_roll3` — red zone conversion rate (TDs per RZ opportunity)

**Source:** Already computable from existing PBP data. No new ingestion needed.

#### B. WOPR — Weighted Opportunity Rating (MEDIUM IMPACT)

**What exists:** `target_share_pct` and `air_yards_share_pct` as separate features for WR/TE.

**What's missing:** The composite `WOPR = 1.5 × target_share + 0.7 × air_yards_share`. WOPR is stickier week-to-week than either component alone and is the standard opportunity metric used by player-prop models and sharp DFS projections. Its stability makes it a better feature than either raw share.

**Source:** Trivially derived from existing features.

#### C. Yards Per Route Run (YPRR) (MEDIUM IMPACT)

**What exists:** `yards_per_target` for WR/TE. `snap_share_pct` as a proxy for route participation.

**What's missing:** YPRR = receiving yards / routes run. Widely considered the single best efficiency metric for receivers — better than yards per target because it accounts for route volume, not just target volume. YPRR with high route participation is the most stable predictor of WR breakout seasons. `yards_per_target` is contaminated by target volume effects (a 2-target game with 80 yards looks identical to a 10-target game with 400 yards on a per-target basis).

**Source:** Route participation data available via NGS receiving tables or derivable from PBP snap data. May need `routes_run` column from nflverse — check `ngs_receiving` table for `avg_intended_air_yards` as proxy.

#### D. Route Participation Rate (MEDIUM IMPACT)

**What exists:** `snap_share_pct` — but snap share counts run-blocking snaps where the player isn't running a route.

**What's missing:** Route participation = routes run / team pass plays. Rising route participation before target share rises is a documented leading indicator of breakout. A WR can have 85% snap share but only 60% route participation (heavy blocking TE, for example). The gap between route participation and target share = untapped opportunity.

**Source:** NGS or derivable from PBP (if route data available). nflverse `participation` data may have this.

#### E. Personnel Grouping Tendencies (MEDIUM IMPACT)

**What exists:** `team_motion_rate`, `team_play_action_rate` — partial scheme capture.

**What's missing:** Team-level personnel grouping rates: 11 personnel % (3WR), 12 personnel % (2TE), 13 personnel % (3TE), 21 personnel % (2RB), 22 personnel % (2RB+2TE). NFL trend toward 12/13 personnel is shrinking the WR3 pool and expanding TE opportunity. A team running 45% 12 personnel has a WR3 ceiling that's fundamentally lower than a team running 75% 11 personnel.

**Source:** PBP data has personnel grouping info. FTN charting data (if available) is more reliable.

#### F. DVOA-Adjusted Opponent Strength (HIGH IMPACT)

**What exists:** `opp_fpts_allowed` — raw fantasy points allowed by opponent defense per position.

**What's missing:** Raw FPA is contaminated by schedule difficulty. A defense that "allows" 25 PPG to RBs looks bad, but if they faced Derrick Henry, Saquon Barkley, and Josh Jacobs, it's expected. FTN's DVOA-adjusted points-against accounts for opponent quality and recency-weights recent performance. This is the single most impactful matchup feature upgrade.

**Approach:** Compute opponent-adjusted FPA using the existing PBP EPA data: `actual_fpa - expected_fpa_given_opponents_faced`. This is a two-pass computation (first compute player strength, then adjust opponent FPA for player strength faced).

#### G. Coaching Staff Identity & Coordinator Change Flags (HIGH IMPACT for draft)

**What exists:** `coaching_change` binary flag, `oc_change`, `scheme_type`, `scheme_pass_rate_delta` in `advanced_analytics.py`.

**What's missing:**
- **Coordinator coaching tree** — Shanahan tree OCs (McVay, LaFleur, McDaniel) run outside zone + play action at predictable rates. Knowing the coaching tree predicts scheme before a single snap is played. This is the most underpriced edge for draft projections.
- **Prior coordinator's historical tendencies** — new OC's career pass rate, pace (plays/game), RB usage patterns from previous stops
- **Coordinator continuity score** — how many years has this OC been with this team? Year 2+ coordinators have different variance profiles than year 1.
- **21 of 32 NFL teams changed OC in 2026** — this is an enormous signal that the binary `oc_change` flag doesn't fully capture.

**Source:** Manual data collection or scraping of coaching staff from team websites / PFR coaching pages. Could be maintained as a simple CSV.

#### H. Offensive Line Quality Metrics (MEDIUM IMPACT)

**What exists:** `rb_ybc_avg` (yards before contact, captures run blocking quality). `qb_pressure_pct_roll3_mean` (captures pass protection indirectly through QB stats).

**What's missing:**
- **Team-level pass-block grade or sack rate allowed** — distinct from QB pressure rate (which blends OL quality with QB pocket behavior). A team allowing 8% sack rate vs. 3% directly affects QB time to throw and WR route development time.
- **Run-block grade** — YBC is good but is also affected by scheme and RB vision. A composite OL run-blocking grade would be more pure.

**Source:** PFF grades are paywalled. Alternative: derive from PBP (team sack rate, team QB hits allowed, team pressure rate allowed as team-level features distinct from QB-level).

#### I. Weather Features (MEDIUM IMPACT)

**What exists:** `game_weather` table with temperature, wind speed, precipitation per stadium per game. Stadium dome flags.

**What's missing:** These weather features appear to be ingested and stored in the DB but **not wired into the ML feature pipeline**. The weather scraper exists (`src/scrapers/weather_scraper.py`), the DB table exists, but the causal feature set has no weather columns.

**Features to add:**
- `wind_speed_mph` — 20+ mph kills passing output, suppresses QB/WR and boosts RB floor
- `is_dome` — dome games have tighter scoring variance
- `precipitation_flag` — rain/snow suppresses passing
- `temperature_bucket` — extreme cold (<20°F) affects grip, passing, and kicking

**Source:** Already in DB. Just needs feature engineering wiring.

#### J. News Sentiment as ML Feature (LOW-MEDIUM IMPACT)

**What exists:** `NewsSentimentAnalyzer` in `advanced_analytics.py` produces `news_sentiment`, `news_volume`, etc. `docs/data/news.json` exists.

**What's missing:** These sentiment features appear to be computed but it's unclear if they're in the causal feature set used for training. Transformer-based sentiment analysis on beat reporter tweets and injury reports is a documented edge in FPL research (2024 ResearchGate paper). Camp reports, depth chart battles, and coaching quotes contain signal that no box-score metric captures.

**Evaluation needed:** Confirm whether sentiment features are in `CAUSAL_FEATURES` or just computed but unused.

### 11.2 Missing Modeling Techniques

#### A. Player Interaction Graph Networks (HIGH IMPACT, HIGH EFFORT)

**What this project does:** Predicts each player independently — no modeling of how teammates or opponents affect each other's output.

**What state-of-the-art does:** The GATv2-TCN paper ("Who You Play Affects How You Play", arXiv 2303.16741) constructs dynamic player interaction graphs where Graph Attention heads learn which opponent players most affect a given player's output. Key finding: **models that ignore player-to-player interactions systematically underperform** on RMSE, MAE, and correlation. A WR facing Sauce Gardner produces different output than facing a CB2 — this is not captured by team-level `opp_fpts_allowed`.

**Implementation path:** Build a GNN layer that takes the weekly matchup graph (player → opponent position group) and produces interaction-adjusted predictions. Could be layered on top of the existing ensemble as a residual correction.

#### B. Player Embeddings / Historical Twin Matching (MEDIUM IMPACT, MEDIUM EFFORT)

**What this project does:** `bayesian_prior_ppg` provides shrinkage toward position mean. Rookie projector uses combine score + draft capital similarity.

**What state-of-the-art does:**
- **Player2Vec** (Football2Vec, GitHub): Treats match events as "sentences" and uses Word2Vec to embed player actions in 32-dimensional space. Player embeddings enable similarity search — find the 5 most similar historical players and use their trajectories as projection inputs.
- **Baller2Vec** (arXiv 2102.03291): Multi-entity Transformer that learns "idiosyncratic qualities of players" in embeddings. Outperforms graph RNNs.
- **Similarity scores** (Forbes/4for4 methodology): Match current players to historical analogs based on stats, age, role, team style, coaching context. Weighted average of analogs' subsequent performance becomes a projection input.

**Implementation path:** Train a player embedding model on historical season stat vectors. For each player entering 2026, find top-5 historical twins and use their next-season performance as an additional feature or ensemble member. Especially valuable for rookies and team-changers where individual history is limited.

#### C. Mixture Density / Bimodal Output Modeling (MEDIUM IMPACT, HIGH EFFORT)

**What this project does:** Predicts a single point estimate per player, with conformal intervals for uncertainty. Boom/bust probability is computed as a feature but not modeled as an output distribution.

**What state-of-the-art does:** The 2PM-Transformer (2026 ICAART paper) models fantasy output as a two-component Poisson mixture, explicitly capturing the bimodal nature of fantasy scores: many low-scoring outcomes + occasional explosions. Achieved MSE 9.31 vs. 9.89 for vanilla Transformer. Fantasy scores are NOT normally distributed — they cluster near 0-5 (injured/inactive/limited) and then spread across 5-35 (active). A Gaussian assumption systematically misestimates floor and ceiling.

**Implementation path:** Replace the single-point prediction head with a mixture density network (MDN) that outputs mixture weights, means, and variances for 2-3 components. This directly produces floor/ceiling estimates instead of deriving them from conformal residuals.

#### D. Monte Carlo Game Simulation (HIGH IMPACT for DFS/Lineup, MEDIUM for season)

**What this project does:** Produces point estimates. No simulation of game outcomes or correlated player performances.

**What state-of-the-art does:** SaberSim (rated #1 DFS optimizer 2026) runs thousands of play-by-play simulations to model:
- **Correlated player outcomes** — QB-WR1 correlation, RB game script correlation
- **Full outcome distributions** — not just E[points] but P(points > 20), P(points > 30)
- **Game script cascades** — team falls behind → pass rate increases → WR1 volume spikes → RB volume drops

**Implementation path:** Build a lightweight game-level simulator that takes Vegas lines (spread, total) and team tendencies, simulates play distribution, allocates touches/targets based on player utilization shares, and produces 1000+ outcome samples per player per game. This replaces point estimates with full distributions.

#### E. QB-WR/RB Correlation Matrices (HIGH IMPACT for lineup optimizer)

**What this project does:** Players are projected independently. The lineup optimizer (`docs/lineup.html`) presumably selects players without modeling correlations.

**What state-of-the-art does:** Every competitive DFS optimizer uses player correlation matrices. QB-WR1 correlation is typically 0.3-0.5. Stacking (QB + pass-catcher from same team) exploits positive correlation. Bring-back (opposing pass-catcher) exploits game total correlation.

**Implementation path:** Compute empirical correlation matrices from historical weekly data: for each team, correlate QB FP with WR1/WR2/TE/RB FP. Feed these into the lineup optimizer as constraints.

#### F. Bayesian Hierarchical Matchup Model (MEDIUM IMPACT, MEDIUM EFFORT)

**What this project does:** `opp_fpts_allowed` as a flat matchup feature. Bayesian shrinkage for player means only.

**What state-of-the-art does:** srome's Bayesian hierarchical model (2015, well-documented) uses partial pooling to model team-vs-position matchup effects. The model simultaneously estimates team offensive strength, team defensive strength vs. each position, and player-within-team effects. Key advantage: **inconsistent teams get wider posteriors** — the model knows when it doesn't know, rather than treating all matchups as equally predictable.

**Implementation path:** Layer a Bayesian hierarchical matchup adjustment on top of the existing ensemble. The existing `BayesianPlayerModel` class could be extended to include defense random effects.

### 11.3 Missing Analytical Frameworks

#### A. Opportunity Share → Volume Conversion (partially exists)

**What exists:** Utilization score is share-based (e.g., 25% target share).

**What's missing (flagged in §6):** Converting shares to absolute volume: `target_share × team_pass_attempts_per_game = expected_targets`. A 25% target share on a team throwing 40 times/game (10 targets) is fundamentally different from 25% on a team throwing 25 times/game (6.25 targets). This conversion requires `team_plays_per_game` and `team_pass_rate` — both computable from existing data.

#### B. Strength of Schedule Adjustment for Projections

**What exists:** `team_sos` as a feature. `opp_fpts_allowed` as raw matchup.

**What's missing:** A full-season SOS-adjusted projection that accounts for the entire upcoming schedule. For draft projections (full season), a player facing 6 top-10 defenses vs. 2 top-10 defenses should have different projections even with identical talent. The schedule is known pre-draft — this is free information.

**Implementation path:** Compute per-player SOS multiplier from the 2026 schedule: for each game, look up opponent defense quality (DVOA-adjusted or raw FPA), average across 17 games, and apply as a scaling factor to the base projection.

#### C. Regression-to-Mean for Efficiency Metrics

**What exists:** `bayesian_prior_ppg` shrinks toward position mean.

**What's missing:** Explicit regression-to-mean for volatile efficiency stats. TD rate, yards per carry, and yards per target are known to regress strongly toward position means year-over-year. A RB with 6% TD rate in 2025 should be projected closer to 4% (position mean) for 2026, not 6%. The Bayesian prior handles this implicitly for PPG but not for individual component stats in the component predictor.

**Implementation path:** For each component target (rushing_tds, receiving_tds, etc.), compute the player's prior rate, the position mean rate, and blend based on sample size. Apply this before the component predictor trains.

#### D. Aging Curves with Workload Interaction

**What exists:** `age_curve` — position-specific quadratic decline from peak age.

**What's missing:** Age × cumulative workload interaction. A 27-year-old RB with 1,500 career carries ages differently than a 27-year-old with 600 career carries. The football research literature (Football Outsiders, PFF) documents that cumulative touch count is a better predictor of decline than age alone for RBs. For WRs, age alone is more predictive.

**Implementation path:** Add `career_touches` (cumulative carries + receptions) as a feature and/or interaction term with age for RB projections.

### 11.4 Summary: Highest-Impact Adoptions (Ranked)

| # | Technique/Variable | Category | What It Replaces/Adds | Expected Impact | Effort |
|---|-------------------|----------|----------------------|-----------------|--------|
| 1 | DVOA-adjusted opponent FPA | Feature | Raw `opp_fpts_allowed` | High — removes schedule contamination from matchup signal | Medium |
| 2 | Per-player RZ/GL carry & target share | Feature | Buried inside utilization composite | High — drives TD variance, biggest single-game swing factor | Low |
| 3 | Wire existing weather data into features | Feature | Currently ingested but unused | Medium — wind 20+ mph is a concrete game-level signal | Low |
| 4 | Coordinator coaching tree + change flags | Feature | Binary `oc_change` flag | High for draft — underpriced scheme change signal | Low-Med |
| 5 | WOPR composite metric | Feature | Separate target share + air yards share | Medium — stickier than components, trivial to compute | Trivial |
| 6 | Personnel grouping rates (11/12/13%) | Feature | Nothing — currently unmodeled | Medium — determines WR/TE opportunity ceiling | Medium |
| 7 | Player correlation matrices | Analytics | Independent player predictions | High for lineup optimizer — enables proper stacking | Medium |
| 8 | Monte Carlo game simulation | Architecture | Point estimates only | High — enables floor/ceiling/upside probability | High |
| 9 | Player embeddings / historical twins | Architecture | Bayesian prior only | Medium — especially valuable for rookies and team-changers | Medium |
| 10 | GATv2 player interaction graphs | Architecture | No player-to-player modeling | High theoretical impact — but highest implementation effort | High |
| 11 | Route participation rate | Feature | Snap share as proxy | Medium — leading breakout indicator for WR | Medium |
| 12 | YPRR (yards per route run) | Feature | Yards per target | Medium — better WR efficiency metric | Medium |
| 13 | Mixture density output modeling | Architecture | Single point estimate + conformal CI | Medium — captures bimodal fantasy score distribution | High |
| 14 | Bayesian hierarchical matchup model | Architecture | Flat `opp_fpts_allowed` feature | Medium — proper uncertainty for inconsistent defenses | Medium |
| 15 | Full-season SOS-adjusted projections | Analytics | No schedule-aware scaling | Medium for draft — schedule is known, free information | Low |
| 16 | OL quality metrics (team sack rate, etc.) | Feature | QB pressure rate only | Medium — affects QB time to throw and WR development | Low |
| 17 | Aging curves × cumulative workload | Feature | Age-only quadratic curve | Medium for RB — career carries predict decline better than age | Low |
| 18 | Regression-to-mean for component stats | Analytics | Bayesian prior on PPG only | Medium — prevents projecting unsustainable TD rates | Low |
| 19 | Sentiment features wired into training | Feature | Computed but possibly unused | Low-Medium — camp reports contain pre-season signal | Low |
| 20 | QB-WR bring-back correlations for DFS | Analytics | Independent projections | Medium for DFS product — standard in competitive optimizers | Medium |

### 11.5 Key Sources

- GATv2-TCN: "Who You Play Affects How You Play" (arXiv 2303.16741) — player interaction graphs
- Football2Vec (GitHub: ofirmg/football2vec) — player embeddings via Word2Vec on match events
- Baller2Vec (arXiv 2102.03291) — multi-entity Transformer for player embeddings
- 2PM-Transformer (ICAART 2026) — bimodal Poisson mixture for fantasy score modeling
- srome's Bayesian hierarchical model — partial pooling for matchup adjustments
- SaberSim — Monte Carlo play-by-play simulation for correlated outcomes
- FTN Fantasy — DVOA-adjusted fantasy points against methodology
- PFF projection methodology — personnel grouping trends, hidden QB metrics
- Forbes (Malloy, 2025) — similarity scores and human-in-the-loop projection methodology
- SMU Data Science Review — comprehensive survey of fantasy football ML approaches
- NFL NGS — route recognition model, pressure probability, occlusion-aware separation
- 4for4 — WOPR, YPRR, target quality metrics
- Fantasy Football Analytics Textbook (Petersen) — Monte Carlo simulation for draft strategy

---

## Appendix: What This System Does Well

To be clear, the codebase has significant strengths that should be preserved:

1. **Leakage defense**: `src/utils/leakage.py` + season-aware CV + lag-shifted features — sophisticated and well-tested
2. **Utilization score framework**: Position-specific, percentile-normalized, data-driven weights — conceptually sound
3. **Component prediction**: Predicting individual stat lines (pass yards, rush TDs, receptions) then assembling FP is more interpretable and auditable than black-box FP prediction
4. **Temporal integrity**: Walk-forward expanding-window validation, train-only scaler fitting, purge gaps between folds
5. **OOF stacking**: Cross-validated out-of-fold predictions for meta-learner training — state-of-the-art
6. **Heteroscedastic uncertainty**: Player-specific error bars via GBM on OOF residuals — accounts for player volatility
7. **Team-change detection**: `dest_team_pos_tgt_pg`, `scheme_fit_score` — proactive handling of free agency impact
8. **PBP-derived features**: EPA, WPA, success rate, neutral pass rate — advanced metrics from play-by-play data
9. **Experiment tracking**: Append-only JSONL with git commit hashes for reproducibility
10. **Production monitoring**: Feature drift (KS test), prediction drift, RMSE degradation alerts


## ORDER OF IMPLEMENTATION
1. §3 — Market/ADP bias removal — Do this first because every model trained with market contamination produces tainted results. No point training anything else until the ADP features and post-processing market anchor are stripped out.        
2. §11 — External techniques (low-effort items only) — Wire weather data, add RZ/GL per-player allocation, WOPR, DVOA-adjusted opponent FPA. These are features that already exist in your data but aren't connected. Do these before retraining
so the first clean model has better inputs.                                                                                                                                                                                                       
3. §4 — Team context & offensive scheme — Coaching tree, personnel groupings, position-specific target allocation. Feeds directly into draft projections.
4. §6 — Utilization pipeline gaps — Share-to-volume conversion, utilization weight optimization on actual data.                                                                                                                                   
5. §8 — Feature engineering gaps — Fills in remaining missing signals.                                                                                                                                                                            
6. §5 — Player-team assignment & roster dynamics — Depth charts, free agency handling.                                                                                                                                                            
7. §7 — Model architecture & validation — Retrain and validate after features are solid.                                                                                                                                                          
8. §11 — External techniques (high-effort items) — GNN, Monte Carlo, player embeddings. Build on top of a clean, feature-rich model.                                                                                                              
9. §9 — Data pipeline hardening — Schema validation, error handling.                                                                                                                                                                              
10. §4.4/§10 — Remaining roadmap items — Mop up.                                                                                                                                                                                                  
11. §2 — PROJECT_NOTES corrections — Update docs last, after the code reflects reality.

---

## STANDING INSTRUCTION FOR ANY AGENT WORKING FROM THIS DOC

Every session so far has turned up real bugs that weren't already listed
here — stale team codes, a `.gitignore` rule silently untracking a whole
directory, a dead detector class with a row-ordering bug, a mislabeled
fallback path, features declared but never actually computed, etc. — none
of these were things anyone set out to look for; they surfaced while
verifying something else.

**When that happens: fix it immediately if it's small and safe (matches
the size/risk of fixes already made throughout this doc), or document it
here in GAPS.md right away if it's bigger — in either case, before moving
on to the next task.** Don't let a found-but-unfixed, found-but-undocumented
bug slip past silently just because it wasn't what you were originally
asked to look at. This doc has repeatedly been the only reason a
next-session agent didn't have to rediscover the same thing from scratch —
keep that going.

## STATUS BRIEF — 2026-08-04

### §3 (Market/ADP bias removal) — CLOSED, plus fallout cleanup

Both contamination points described above were **already removed from the source**
before this session started (no `MarketAnchorCurve`, no `market_anchor`/`market_gap`
in `CALIBRATION_FEATURES_BY_POSITION`, no market-blending step in
`UpstreamCalibrator.calibrate()` — `preseason_projector.py`'s own docstring says
"no market/ADP data — pure ML"). The prior session's handoff notes describing
Contamination Point 2 as "still live" were stale.

What *was* actually broken, found by verifying the handoff notes against the real
source instead of trusting them:
- `tests/test_preseason_projector.py` still imported `MarketAnchorCurve` and
  referenced `market_anchor`/`market_gap` — the file failed to import at all.
  Rewritten to match current source; 8/8 pass.
- `scripts/audit_preseason_projector.py` read a `market_anchor` column from
  `predict_with_details()` output that no longer exists — would `KeyError` on
  next run. Fixed.
- **Real bug**, not just stale tests: `_market_objective_score()` in
  `preseason_projector.py` read three fields (`pred_market_mae`,
  `pred_large_divergence_share`, `pred_rb_wr_gap_excess`) that nothing populated
  after the market-anchor removal, so it silently returned the same constant for
  every model variant — model-variant selection had stopped discriminating
  between candidates whenever the draft-sim gate didn't pick a winner (always true
  for direct `.fit()` calls, e.g. the test suite). Replaced with
  `_outcome_objective_score()` = `pred_mae + abs(pred_bias)`, using metrics that
  are actually computed. Renamed the `market_objective_score` selection-report key
  to `outcome_objective_score` (nothing downstream reads the old key name).

**Lesson for future sessions**: this doc and prior handoff notes can describe code
that's already changed. Verify against the actual file (grep the class/function
names, read the real column lists) before planning around a GAPS.md claim.

### §11 low-effort items — weather / WOPR / RZ-GL — DONE, retrain complete and verified

Verified against `CAUSAL_FEATURES` in `config/settings.py` (the actual training
feature list) rather than GAPS.md's descriptions. Findings differed from the
GAPS.md text above in ways worth recording:

- **WOPR**: was already computed (`utilization.py:_add_wopr`) but by a pipeline
  (`calculate_utilization_scores`) that `create_causal_features()` — the function
  that actually builds training data — never calls. Adding `"wopr"` straight to
  `CAUSAL_FEATURES` would have been a silent no-op (same failure mode as the
  `_market_objective_score` bug above). Instead added `wopr_roll3`, computed
  in `_create_causal_rolling_features` directly from the already-rolled
  `target_share_pct_roll3_mean` / `air_yards_share_pct_roll3_mean` (leakage-safe).
- **RZ/GL per-player allocation**: real raw counts exist in `player_weekly_stats`
  (`redzone_targets`, `rush_inside_5` — NOT `goal_line_touches`, which is
  rush+target combined per `pbp_stats_aggregator.py:506` and would have
  mislabeled a "carry share" feature). No distinct red-zone (inside-20) rushing
  column exists, only goal-line (inside-5) — so this shipped as 2 features
  (`redzone_target_share_pct_roll3_mean`, `goal_line_carry_share_pct_roll3_mean`),
  not GAPS.md's original 4. Computed in `_create_base_features` (per-team-week
  share, same pattern as `target_share_pct`) and rolled in
  `_create_causal_rolling_features`.
- **Weather**: confirmed gap as described — `game_weather` table has 5,431 fully
  populated rows, zero wiring into any feature pipeline. Added
  `_add_weather_features()` (new method), merged unrolled (like Vegas
  spread/total — pre-game info, not a lagged stat). Dome games (1,849 rows) have
  NULL temp/wind/precip in the source table; these get "no weather effect"
  defaults (wind=0, precip=0, temperature_bucket=mild) rather than inheriting the
  outdoor-game median.
- **DVOA-adjusted opponent FPA**: NOT done. Confirmed as a real gap (only raw
  `opp_fpts_allowed` exists), and unlike the other three this is genuine new
  two-pass computation, not wiring. Left for a future phase per the original
  scoping decision.

**Wired into `CAUSAL_FEATURES`**: RB gets redzone-target-share + goal-line-carry-
share + weather; WR/TE get redzone-target-share + `wopr_roll3` + weather; QB gets
weather only. Files: `src/features/feature_engineering.py`, `config/settings.py`.

**Verification performed**: called `create_causal_features()` directly on real
`player_weekly_stats` data (2022+, 22,118 rows) and confirmed every new column is
100% non-null with real variance (not constant-filled) — e.g.
`redzone_target_share_pct_roll3_mean` mean=2.06 std=6.96 nunique=332,
`wind_speed_mph` mean=5.36 std=5.46 nunique=194. `wopr_roll3` has a small negative
tail (min -0.09) from rare negative `air_yards_share_pct` values (real signal
noise, not a bug).

**Retrain — DONE (2026-08-04, same day, later in this session).** Unblocking it
surfaced three more pre-existing, unrelated bugs, each fixed:

1. `data/cached_features.parquet` didn't exist. Regenerated via
   `python scripts/generate_app_data.py` (uses the already-trained model that
   existed in `data/models/` to build it — no circularity, since that model
   predates this session).
2. `validate_training_cache_integrity()` in `src/data/quality_gates.py` then
   failed on "742 ghost rows" and "4 duplicate rows" — both were the single
   not-yet-played 2026-week-1 "prediction target" stub row block that
   `generate_app_data.py` appends for app display (confirmed 100% null for
   that exact week, not partial — a real corruption signature would be
   partial). Fixed by excluding that one specific stub week from the
   ghost-row/duplicate checks (only when it's 100% null), while leaving the
   checks fully strict for any other week — narrow, doesn't weaken the gate's
   actual corruption-detection purpose. Only caller of this function is
   `train.py`, so no other blast radius.
3. `src/models/data_loading.py`'s in-season guard then raised
   `"pipeline requires the current season as test"` — it was missing the
   draft-prep-window carve-out that `DataManager.get_train_test_seasons()`
   already has correctly (`is_draft_prep_window()` was imported but never
   used). In August, `current_season_has_weeks_played()` is misleadingly
   `True` (the *prior* season had weeks played — it just ended months ago),
   so the guard needs the same June–Aug exemption. Fixed by mirroring the
   existing correct logic.
4. `feature_preparation.py:_apply_bounded_scaling()` then crashed with
   `KeyError` on ~60 column names (including 4 of this phase's new ones) —
   line 227 indexed `test_df[cols]` unconditionally, one line below an
   existing `if not test_df.empty` guard for the same dataframe. In draft-prep
   mode `test_df` is genuinely empty (2026 hasn't happened yet). Fixed by
   extending the existing guard to cover the indexing too — this was a
   pre-existing bug in the empty-test-set path, unrelated to which features
   are declared, that would have blocked *any* draft-prep training run.

Also bumped `FEATURE_VERSION` from `"22"` to `"23"` in `config/settings.py` —
had labeled the new features "v23" in comments but forgot the actual version
bump the project convention expects.

**Verified against the real trained artifacts**, not just that training
exited 0: loaded `data/models/component_{rb,wr,te,qb}.json` after training and
confirmed all 6 new features appear in `feature_names` for every position that
declares them (RB 41/41, WR 41/42, TE 40/41, QB 39/39 — the one WR/TE column
short in each case was `recv_epa_per_target_roll3_mean`).

**`recv_epa_per_target_roll3_mean` gap — also fixed, same session.** Root
cause: `recv_epa_per_target` (this-week value) is computed in
`_create_base_features`, but only ever rolled into `_roll3_mean` inside
`_create_rolling_features` — the **full-mode** pipeline. `create_causal_features()`
(what actually builds training data) calls `_create_causal_rolling_features`
instead, whose own `roll_cols` list never included it, even though
`CAUSAL_FEATURES["WR"]`/`["TE"]` have declared it all along. Identical failure
pattern to the WOPR gap above (declared in the causal feature list, computed
only in a pipeline the causal path doesn't call) — pre-existing, predates this
session, unrelated to anything added here. Fixed by adding
`"recv_epa_per_target"` to `_create_causal_rolling_features`'s `roll_cols`.
Verified on real data (100% non-null, 4,225 unique values) and confirmed in
the retrained WR (42/42) and TE (41/41) model artifacts.

Feature-importance / accuracy-lift comparison (old model vs. new) was not done
this session — the retrain succeeded and features are confirmed live in the
model, but nobody has yet measured whether they improve `pred_mae`. Worth
doing before leaning on these features for real draft decisions.

**Test coverage note**: `tests/test_quality_gates.py` and
`tests/test_target_and_history_causality.py` pass (7/7) against all of this
session's fixes. `tests/test_ml_robustness_15_steps.py` and
`tests/test_integration.py` were NOT run to completion — they train full,
untuned models (`test_full_pipeline_all_positions`, `test_cross_validation`,
etc.) and took over an hour without finishing; killed rather than let it keep
running. A partial run showed some `F`s in the dot output before being killed,
never diagnosed — unknown whether pre-existing or related to this session's
changes. Worth running standalone (outside an interactive session) before
trusting this suite's coverage.

### UNCOMMITTED — read this first if picking up in a fresh session

Everything above (§3 fallout cleanup + §11 low-effort wiring + the 4
pre-existing bugs it surfaced) is sitting **uncommitted** in the working tree.
`git status` as of the end of this session:

```
 M GAPS.md
 M config/settings.py
 M scripts/audit_preseason_projector.py
 M src/features/feature_engineering.py
 M src/models/data_loading.py
 M src/models/feature_preparation.py
 M src/models/preseason_projector.py
 M tests/test_preseason_projector.py
```

No commit was requested, so none was made — do not assume any of this has
landed. If you're a fresh agent/session: check `git status`/`git diff` yourself
before trusting this doc's "DONE" claims, the same way this session learned
not to trust the previous handoff notes at face value.

### Next up per the ORDER OF IMPLEMENTATION list

§4 (team context & offensive scheme) is next. DVOA-adjusted opponent FPA (the
one §11 item deferred above, genuine new two-pass computation rather than
wiring) is still open too — do it whenever convenient relative to §4, no hard
ordering between them.

**Worth doing first, cheaply**: this session found the same bug twice
(WOPR, then `recv_epa_per_target_roll3_mean`) — a feature declared in
`CAUSAL_FEATURES` for a position but never actually computed in
`_create_causal_rolling_features`/`_create_base_features`, silently dropped
rather than erroring. A quick systematic check — diff each position's
`CAUSAL_FEATURES` list against the actual output columns of
`create_causal_features()` on real data — would catch any other dead-declared
features before spending effort on §4's new ones.

### Systematic CAUSAL_FEATURES audit — DONE, clean (2026-08-04, new session)

Ran the check above using the real training-data loader
(`DatabaseManager.get_all_players_for_training()`, which performs the
`team_defense_stats`/`team_stats`/`utilization_scores` LEFT JOINs that
`create_causal_features()` depends on) on 2022+ data (22,118 rows), then
`FeatureEngineer().create_causal_features()` on the result.

First pass without the real loader (raw `player_weekly_stats` table only)
falsely flagged `opp_fpts_allowed` as missing for all four positions — that
was a test-harness gap, not a bug: `opp_fpts_allowed` is built from
`fantasy_points_allowed_{qb,rb,wr,te}` columns that only exist after the
`team_defense_stats` LEFT JOIN in `get_all_players_for_training` (see
`src/utils/database.py:1592-1629`), which raw-table loading skips.

With the real loader: **all `CAUSAL_FEATURES` columns present for
RB/WR/TE/QB, all ≥95% non-null, all with real variance (nunique > 1).** No
dead-declared features found — the WOPR/`recv_epa_per_target_roll3_mean`
failure mode does not currently recur elsewhere. Clear to proceed to §4.

### §4 / §9.2 (LAR/JAC team-code fix) — DONE, root cause fixed + DB migrated (2026-08-04)

Started on the "low effort" §4/Phase-1 items (position target allocation,
LAR/JAC fix, tempo). The LAR/JAC item turned out to be a live, worse-than-
documented bug, not the stale one-off GAPS.md originally described — found
by checking the actual DB instead of trusting the doc (same lesson as the
§3 section above).

**What was actually wrong:** starting with the 2025 season, `team_stats`,
`team_defense_stats`, and `player_weekly_stats` contained BOTH the old codes
(`JAX`, `LA`) and new codes (`JAC`, `LAR`) for Jacksonville/Rams
simultaneously — not a static mismatch between two tables as GAPS.md §9.2
described, but two codes coexisting in the same table across different
weeks/seasons. `schedule` stayed on `JAX`/`LA` consistently 2006–2026.

**Root cause:** `entity_resolver.py`'s `TEAM_CODE_ALIASES` mapped
`JAX→JAC` and `LA→LAR` (backwards relative to schedule's convention). This
gets applied via `resolver.build_keys()` inside `nfl_data_loader.py`'s
`_standardize_weekly_columns`/`_standardize_pbp_columns` — i.e. every
ingestion/refresh run. Historical seasons (2006–2024) were loaded before
this alias existed and kept `JAX`/`LA`; any row touched by a refresh after
the alias was added (2025 season, still in progress) picked up `JAC`/`LAR`
instead — splitting the same team's data across two codes mid-table.
`nfl_data_loader.py` also had a second, independent copy of the same
backwards mapping (`_SCHED_TO_PWS`) in its schedule-based opponent-backfill
path, and `scripts/backfill_snap_counts_to_pws.py` had one stale line
(`"JAX": "JAC"`) in its join-key normalizer.

**Fix (4 files):**
- `src/data/entity_resolver.py` — flipped `TEAM_CODE_ALIASES` to
  `JAC→JAX`, `LAR→LA` (canonical = schedule's convention). `STL→LA` updated
  to match (was `STL→LAR`); `SD→LAC`, `OAK→LV` unchanged (no observed
  duplication issue there).
- `src/data/nfl_data_loader.py` — removed the now-redundant/backwards
  `_SCHED_TO_PWS` local remap in the opponent-backfill path; schedule and
  pws use the same codes now, no translation needed.
- `scripts/backfill_snap_counts_to_pws.py` — fixed the stale `"JAX": "JAC"`
  alias line to `"JAC": "JAX"`.
- `scripts/backfill_opponent_lar_jac_2025.py` — updated docstring/removed
  its own backwards remap (same reasoning).

**New migration script**: `scripts/normalize_lar_jac_team_codes.py` —
one-time cleanup of rows already written under the old alias direction.
`player_weekly_stats` and `team_defense_stats` got a plain rename
(verified first: zero duplicate player-weeks in pws; `team_defense_stats`
2025 only existed under the new code, no old-code row to collide with).
`team_stats` has a `UNIQUE(team, season, week)` constraint and, for 2025,
had a genuine duplicate pair under both codes for every (team, week) —
verified 100% consistent (39/39 old-code rows fully populated on
`points_scored`/`pace_sec_per_play`, 38/38 new-code rows NULL on both) before
writing the script, so those sparse new-code duplicates are deleted rather
than merged, not renamed.

**Before running the migration**: backed up the live DB
(`data/nfl_data.db.bak-lar-jac-20260804154651`, gitignored, not deleted —
safe to remove once this fix has been trusted for a while).

**Verified after migration:**
- Zero `JAC`/`LAR` rows remain in any of the three tables (script's own
  post-run check).
- `create_causal_features()` on real 2025 data: `opp_fpts_allowed` is 100%
  non-null for both `JAX` (186 rows) and `LA` (200 rows) players in 2025 —
  previously dead for exactly these two teams, per the original GAPS.md
  finding, just for a different underlying reason than documented.
- `tests/test_preseason_projector.py`, `tests/test_quality_gates.py`,
  `tests/test_target_and_history_causality.py`: 15/15 pass.

### §4 low-effort batch continued: position target/carry allocation + tempo — DONE (2026-08-04)

Added to `src/features/feature_engineering.py` (`_create_base_features` +
`_create_causal_rolling_features`) and `config/settings.py` (`CAUSAL_FEATURES`,
`FEATURE_VERSION` bumped to `"24"`):

- `team_rb_target_share` / `team_wr_target_share` / `team_te_target_share` —
  each position's share of the team's total targets that week (GAPS.md §4.A).
- `team_target_concentration` — Herfindahl index of individual player target
  shares within the team-week (higher = concentrated on fewer players).
- `team_rb_lead_share` — the lead RB's share of the team's RB rushing
  attempts that week (bell-cow vs. committee indicator).
- `team_plays_roll3_mean`, `team_pace_sec_per_play_roll3_mean` — tempo
  (GAPS.md §4.D). `team_plays`/`team_pace_sec_per_play` were already joined
  onto the raw training frame from `team_stats` (via
  `get_all_players_for_training`) but never rolled/used.

All five follow the codebase's established leakage-safe pattern: computed as
a this-week raw value (broadcast to every row of that team-week), then
consumed only via the `_roll3_mean` variant (`shift(1).rolling(3).mean()`
per player, in `_create_causal_rolling_features`) — same pattern as
`target_share_pct`/WOPR/redzone-share from the §11 phase.

Wired into `CAUSAL_FEATURES`: RB gets `team_rb_target_share_roll3_mean` +
`team_rb_lead_share_roll3_mean` + tempo; WR/TE get their own target-share
variant + `team_target_concentration_roll3_mean` + tempo; QB gets tempo only.

**Real data bug found and fixed while verifying this (not caused by this
session's feature code):** `team_plays_roll3_mean`/`team_pace_sec_per_play_
roll3_mean` were initially ~70-93% zero-filled for large chunks of the
training window, because the underlying `team_stats.total_plays` /
`pace_sec_per_play` columns were themselves zero for most rows 2020-2024
(and `pace_sec_per_play` specifically was 100% zero for 2018-2019 too) —
found by checking distribution/variance before trusting the feature, not by
assuming the join worked. Root cause: those two columns are populated by a
PBP-based aggregator (`pbp_stats_aggregator.get_team_stats_from_pbp`) that
had only ever been run for 2025 (current-season pipeline); everything else
in `team_stats` for those seasons was already fully populated. New script
`scripts/backfill_team_pace.py` re-derives both columns from PBP for
2018-2024 and UPDATEs the existing rows (idempotent — only touches rows
currently zero on either column). Ran against the DB (already backed up
from the LAR/JAC fix); post-backfill only week-0 preseason placeholder rows
remain zero (verified legitimate, not a gap).

**Verified**: all 5 new `_roll3_mean` features 100% non-null with realistic
values on real 2018+ data (target shares sum to ~100% across RB/WR/TE as
expected: 18.6% + 59.3% + 21.9%; `team_plays_roll3_mean` mean=62.5,
std=5.3; `team_pace_sec_per_play_roll3_mean` mean=31.2s, std=1.7 — both
match real NFL norms post-backfill, versus mean=15.8/std=25.6 and highly
degenerate before it). Full `CAUSAL_FEATURES` dead-declaration check
(same method as the earlier systematic audit) re-run clean for all four
positions.

**Also checked, turned out already done** (GAPS.md §10 Phase-1 table listed
these as open, but they're not): `depth_chart_rank`, `is_contract_year`,
`contract_apy_rank` are already in `CAUSAL_FEATURES` for RB/WR/TE/QB and
verified non-degenerate on real data (depth_chart_rank: 3 real buckets;
is_contract_year: 3.8% flagged, plausible; contract_apy_rank: continuous,
807 unique values). §10's Phase-1 items 1.1/1.2 (depth_charts/contracts
wiring) can be crossed off — they predate this session. Another instance of
the doc-vs-source drift this session keeps finding; verify against
`CAUSAL_FEATURES` + a real-data audit before trusting any GAPS.md "not yet
wired" claim.

**Retrain — DONE.** `python -m src.models.train --fast --no-tune` completed
(exit 0), `feature_version.txt` = 24, previous model version archived (5
versions available for rollback). Verified directly against
`data/models/component_{qb,rb,wr,te}.json` `feature_names`: all 7 new
`_roll3_mean` columns present exactly where `CAUSAL_FEATURES` declares them
(QB: tempo only; RB: `team_rb_target_share`+`team_rb_lead_share`+tempo;
WR/TE: their own target-share variant + `team_target_concentration` +
tempo). `tests/test_preseason_projector.py`, `tests/test_quality_gates.py`,
`tests/test_target_and_history_causality.py`: 15/15 pass post-retrain.

Feature-importance/accuracy-lift comparison against the pre-v24 model was
NOT done — same caveat as the v23 retrain: features are confirmed live,
nobody has measured whether they move `pred_mae`. Worth doing before
leaning on these for real draft decisions, ideally batched with the v23
features' still-outstanding lift measurement.

**Still open / not attempted this phase**: `1.6` (utilization share→volume
conversion — a design change to the utilization-score pipeline, not a
wiring task, out of scope for this batch), `3.2`/`3.3`/`3.4` (red-zone
carry allocation beyond what §11 already shipped, FTN personnel grouping,
PFR weekly advanced stats).

### §3.1 / §11.1.G — Coaching staff DB — DONE (2026-08-04)

`src/features/advanced_analytics.py` already had a complete, well-built
`CoachingChangeDetector` class (HC/OC/DC change detection, exponential
adaptation decay, position-weighted impact, tenure/stability, scheme
classification) — but it had **never been fed real coaching identity
data**, anywhere in the codebase, so it always silently fell back to a
`scheme_pass_rate_delta` proxy instead of true coach-identity detection.

**Data source found**: `nfl_data_py.import_schedules()` — already used
elsewhere in this codebase for schedule ingestion — returns `home_coach`/
`away_coach` per game (from nflverse's `games.csv`), 0% null 2006-2026,
using the same canonical team codes as `player_weekly_stats` (post the
LAR/JAC fix). No new external dependency needed. No comparably reliable
structured source exists for OC/DC identity across 20 years, so this phase
covers HC only — the detector's OC/DC paths degrade safely to their
existing defaults (`oc_change=0`, etc.) when those columns aren't present,
which was already correct dead-code behavior, just now actually exercised
instead of untested.

**Real bug found and fixed before wiring in** (would have corrupted the
signal if shipped as-is): `_detect_hc_changes` and `_compute_coaching_
tenure` both did `df.groupby("team")[...].shift(1)`/`.cumsum()` directly
on the per-player training frame, which is sorted by `player_id` first
(see `add_coaching_change_features`'s own `sort_values(["player_id",
"season","week"])`). Since pandas groupby preserves row order within each
group, shifting on a player-sorted frame compares whichever two rows
happen to be adjacent *for that team* in player-ID order — i.e. two
different players' unrelated season/weeks — not consecutive team-weeks.
Verified with a synthetic 2-player test: a player's first row after
another player's last row was spuriously flagged `coaching_change=1`
purely from index adjacency, not an actual coaching change. Fixed both
methods to compute on a `(team, season, week)`-deduplicated, properly
sorted team-week table, then merge the results back onto every player row
for that team-week — confirmed correct on the same synthetic test after
the fix (both players now correctly show the change only at the true
2020→2021 HC transition, with matching tenure counts).

**New infra**:
- `team_coaching_staff` table (`src/utils/database.py` `_init_database`) —
  `(team, season, week, head_coach)`, `UNIQUE(team, season, week)`.
- `DatabaseManager.ensure_team_coaching_staff()` — fetches via
  `nfl_data_py.import_schedules()`, only missing seasons unless
  `force_refresh=True`; degrades to a no-op on fetch failure (doesn't
  raise), matching the try/except pattern used elsewhere in
  `_prepare_training_data`.
- `scripts/backfill_coaching_staff.py` — one-time/re-runnable full backfill
  entry point.
- `get_all_players_for_training()` — added `LEFT JOIN team_coaching_staff`
  (same-week, since HC identity is pre-game-known info, not an outcome
  stat — no leakage).
- `FeatureEngineer._add_coaching_change_features()` (`feature_engineering.py`,
  called from `create_causal_features()`) — merges `head_coach` in if not
  already present (self-contained for callers other than the main training
  path), then calls the fixed `CoachingChangeDetector`. Explicitly
  preserves the pre-existing `scheme_fit_score` column around the call —
  `CoachingChangeDetector` emits a column of the same name, but the
  existing one (from `_create_team_change_features`, destination-team-aware
  for free-agent signings) is more specific and was already in production
  use; letting the coaching detector's version silently overwrite it would
  have been a regression.

**Second team-code bug found and fixed while backfilling** (same class as
the LAR/JAC fix, different teams): `nfl_data_py.import_schedules()` uses
the team code that was historically accurate at the time — `OAK` for
2018-2019 Raiders — while `player_weekly_stats.team` is already normalized
to the current franchise code (`LV`) for those seasons. Fixed by applying
`entity_resolver.EntityResolver.normalize_team_code` (the same canonical
alias map used for the LAR/JAC fix, which already had `OAK→LV`/`SD→LAC`/
`STL→LA` correct) to the coaching-staff ingestion path before upsert.
Also cleaned up 570 stale rows left under the old codes from an initial
test run that predated this fix (`OAK`/`SD`/`STL`, deleted directly —
`INSERT OR REPLACE` doesn't clean up rows under a *different* primary key,
only same-key rows, so the pre-fix and post-fix rows had briefly
coexisted).

**`CAUSAL_FEATURES` additions** (all 4 positions — coaching changes affect
every skill position, not just one): `coaching_change` (binary),
`coaching_adaptation_score` (continuous exponential decay, supersedes
`weeks_since_coaching_change`/`new_coaching_staff` as a smoother single
signal), `coaching_stability` (log-scaled tenure), `coaching_change_impact`
(position-weighted expected PPG delta). Deliberately did NOT add
`oc_change`/`dc_change`/`any_coaching_change`/`weeks_since_oc_change`/etc.
— those would be constant/degenerate given no OC/DC data source (same
"don't declare dead features" lesson as the WOPR/`recv_epa_per_target`
bug earlier this session), and `any_coaching_change` reduces to
`coaching_change` anyway with OC/DC always 0. `FEATURE_VERSION` bumped to
`"25"`.

**Verified**: full `CAUSAL_FEATURES` dead-declaration audit clean for all
4 positions on real 2006+ data (22,118+ rows). New columns 100% non-null;
`coaching_change` fires at real, known HC transitions (spot-checked
against `groupby(["team","season"])["coaching_change"].max()` — ARI 2019,
ATL 2020/2021/2024, CAR 2020/2022/2023/2024, CHI 2022, etc., all real HC
changes). `tests/test_preseason_projector.py`,
`tests/test_quality_gates.py`, `tests/test_target_and_history_causality.py`
(15/15) and `tests/test_advanced_analytics.py` (79/79, covers
`CoachingChangeDetector`) all pass post-fix.

**Retrain — DONE.** `feature_version.txt` = 25, previous model archived.
Verified directly against `data/models/component_{qb,rb,wr,te}.json`
`feature_names`: all 4 coaching features present for every position
(QB 45, RB 49, WR 50, TE 49 total features). Full test suite re-run
post-retrain: 94/94 pass (`test_preseason_projector.py`,
`test_quality_gates.py`, `test_target_and_history_causality.py`,
`test_advanced_analytics.py`). Feature-importance/accuracy-lift
measurement against the pre-v25 model still not done — same standing
caveat as v23/v24, now three feature-version bumps deep without a
pred_mae comparison. Worth doing as its own pass before leaning on any of
v23/v24/v25's features for real draft decisions.

**Not attempted**: OC/DC identity (no reliable 20-year structured source
found), coaching tree / scheme-lineage classification (GAPS.md §11.1.G's
"Shanahan tree" idea — would need a hand-maintained coach→tree mapping,
out of scope for this phase), prior-coordinator historical-tendency
features.

### §2 (PROJECT_NOTES.md corrections) — DONE (2026-08-04)

Applied all 10 corrections listed in §2 above, plus documented everything
fixed in this session (LAR/JAC root cause, OAK/LV, coaching staff,
weather, PFR/contract/depth-chart features already being live). Added the
three previously-missing sections: Component Prediction Architecture,
Utilization Weight Optimization, Draft Prep Mode. Extended "Recurring
Bugs/Patterns to Watch" with 3 new entries from this session (declared-
but-never-computed features, historical-vs-current team-code drift, and
row-order-dependent groupby on a differently-sorted frame — the
`CoachingChangeDetector` bug).

### Remaining §9/§11/§10 items — explicitly scoped OUT of this pass

To close out the "gap plan" phase (before moving to validation testing),
scoping these as deliberately deferred rather than half-attempted:

- **DVOA-adjusted opponent FPA** (§11.1.F) — genuine new two-pass
  computation (team offensive strength → opponent-adjusted defensive FPA),
  not wiring. Real, well-defined, bounded work — just not done this
  session given time already spent. Next candidate for a "low-effort
  wiring"-style session.
- **FTN/PBP personnel grouping** (§3.3) — checked feasibility: FTN's
  `import_ftn_data()` doesn't carry a direct personnel-grouping column
  (only motion/play-action/rpo/backfield-count, which `team_scheme_
  tendencies` already captures from *some* source, 2022+ only — matches
  FTN's real coverage start, not a bug). PBP's `offense_personnel` string
  column (e.g. "1 RB, 2 TE, 2 WR") is the actual source for 11/12/13%
  groupings, has wider historical coverage, but isn't available for at
  least early-2010s seasons (`KeyError` on 2010) and needs string-parsing
  + team-week aggregation work. Real gap, genuinely medium effort, not
  attempted.
- **Vegas preseason win totals** (§8.1 Tier 2) — scraper exists
  (`odds_scraper.py fetch_win_totals`/`scrape_win_totals`) but the
  `win_totals` market has never actually been scraped into `game_odds`
  (checked: only h2h/spreads/totals present). Requires running the
  scraper against The Odds API (needs `ODDS_API_KEY`, not confirmed
  available/funded this session) — not attempted.
- **§9 data pipeline hardening** (schema validation on ingestion,
  replacing remaining `except Exception: pass` with structured logging) —
  this session fixed every *specific* silent-failure instance it
  encountered (LAR/JAC, OAK/LV, team_stats pace zero-fill, coaching
  detector ordering bug), each with real verification, but did not do a
  systematic codebase-wide audit for other `except Exception: pass`
  patterns. That's explicitly flagged in PROJECT_NOTES.md's "Recurring
  Bugs" #1 as an ongoing pattern to watch, not resolved wholesale here —
  doing so properly would mean auditing every try/except in the ingestion
  path, which is its own multi-session effort.
- **§11 high-effort architecture items** — GNN player-interaction graphs,
  Monte Carlo game simulation, player embeddings/historical-twin matching,
  mixture-density output modeling, a dedicated full-season projection
  model (§7.1), rookie-projector integration into the main ensemble
  (§7.2). These are each legitimately multi-week research/engineering
  efforts requiring their own design discussion before implementation —
  GAPS.md's own ORDER OF IMPLEMENTATION lists them last, "build on top of
  a clean, feature-rich model." Not attempted, not scoped down — they
  need a dedicated planning pass, not a rushed partial implementation
  bolted onto this session's feature-wiring work.

### Next: validation testing across all shipped features (v22→v25)

Per explicit instruction: before any of this is trusted for production
draft decisions, run a proper accuracy validation — not just the
"non-degenerate, doesn't crash" checks this session has relied on so far.
Using `scripts/run_ts_backtest.py` (existing leakage-free expanding-window
backtester, the same tool behind PROJECT_NOTES.md's documented April 2026
walk-forward baseline: overall R²=0.269, Pearson r=0.520 at Ridge
α=10,000) against the 2025 holdout season, on the pipeline as it stands
now with all v22–v25 features live.

### Results — real, positive lift over the documented baseline

`python scripts/run_ts_backtest.py --season 2025 --model ridge --alpha
10000` (expanding-window, weekly refit, 22 weeks, 5,612 predictions,
~67s/week — full log in `ts_backtest_2025_20260804_182714.json`):

| Metric | This run (v25 features) | Documented baseline (April 2026, pre-v22) |
|---|---|---|
| Overall R² | **0.345** | 0.269 |
| QB R² | 0.223 | 0.092 |
| RB R² | 0.364 | 0.258 |
| WR R² | 0.276 | 0.257 |
| TE R² | 0.263 | 0.152 |
| Overall MAE | 4.79 | — (not recorded in that baseline) |
| Overall RMSE | 6.39 | — |

**This is a real, first-time-measured lift** — every position improved,
not just an aggregate average masking a regression somewhere. Overall R²
(0.345) now **exceeds the trailing-average-heuristic ceiling (R²=0.279)**
documented in PROJECT_NOTES.md's "Predictive Ceiling Summary" — that
section's core claim ("model adds no value over a simple blend until R²
exceeds 0.279") no longer holds as written; updated there directly.

Caveats, so this isn't overstated:
- The documented baseline table mixes two different runs (the α=1.0 "April
  10" table shows QB R²=0.092, but the "production default" is α=10,000 —
  this backtest used α=10,000 to match production, so it's the right
  comparison, but the two numbers weren't generated under identical
  conditions to begin with; treat this as directionally strong, not a
  precise paired A/B).
- This is one backtest on one holdout season (2025) with weekly-refit
  expanding-window Ridge — not a multi-season average, not the
  higher-fidelity `--model ensemble` path (too slow — see PROJECT_NOTES
  Phase 4, ensemble runs took ~12.8h wall clock and were killed by a
  pre-registered kill gate), and it does not isolate *which* of v22–v25's
  changes drove the lift (market-bias removal, weather/WOPR/redzone,
  team-target-allocation/tempo, or coaching-change detection individually)
  — only that the pipeline as a whole, right now, is meaningfully better
  than the last time anyone measured it.
- Decision-quality panel from the same run: Hindsight 63.6% (14-8, p=0.14,
  ROI +14.5%) — weaker than the 75.61%/p5=65.85% H2H figure in
  PROJECT_NOTES, but that number came from a different, broader bootstrap
  validation (10,000 resamples across more weeks), not a single-season
  walk-forward — not a fair apples-to-apples regression signal, just
  noted for completeness.

**Verdict**: clear go-ahead to trust the v22–v25 feature work as a net
positive, not a regression. A proper per-feature-version ablation (train
excluding just v25's coaching features, or v24's team-allocation features,
and compare) would be needed to attribute the lift to a specific change,
but that's a further-refinement question, not a production-readiness
blocker.

### §7.1 (dedicated full-season model) — real gap found and fixed: wired `PreseasonProjector` into the production draft board (2026-08-04)

User asked to pick the single most impactful remaining item, regardless of
effort. Investigated the three strongest candidates via parallel Explore
agents (DVOA-adjusted opponent FPA, §7.1's "no full-season model" claim,
§7.2's "rookies not integrated" claim) before picking — two of the three
turned out to be stale/inaccurate as originally documented, same pattern
as everything else this session:

- **§7.2 verdict: partially stale.** `advanced_rookie_injury.py`'s
  sophisticated rookie/injury features (draft capital, comp-player
  matching, combine scores, archetype tiers, breakout/bust probability) DO
  run in both training and serving (`feature_preparation.py`,
  `predict.py`) — contrary to "NOT wired into the main ensemble pipeline."
  But every single one of those columns is absent from `CAUSAL_FEATURES`
  (config/settings.py has a comment admitting this: "available in full
  mode but not promoted to causal yet") — so none of it reaches the model
  as an actual input. Real gap, just narrower and differently-shaped than
  documented. **Not fixed this session** — user picked a different item.
- **§7.1 verdict: partially stale, and this is the one picked.**
  `src/models/preseason_projector.py`'s `PreseasonProjector` IS a real,
  already-trained season-total regression (predicts `SUM(fantasy_points)`
  for next season directly from prior-season aggregates — ppg, games
  played, per-stat-category rates, target/snap/rush share — with an
  upstream calibration layer). Trained every run, saved to
  `data/models/preseason_projector.json`, tested (8/8,
  `tests/test_preseason_projector.py`). **But it was never called from
  the two scripts that actually build user-facing draft data**
  (`scripts/generate_app_data.py`, `scripts/generate_draft_data.py`) — its
  only callers were training, its own test suite, and
  `scripts/snake_draft_sim.py` (a simulation harness). The actual
  production draft data — `data/players_{QB,RB,WR,TE}.json`, the file
  that sets both the displayed projection AND the sort/rank order — was
  built from `projection_18w`, which `EnsemblePredictor.predict()`
  produces via `fp_pred * n_weeks` (a single week's component prediction
  multiplied by 18) in the component-mode branch. Confirmed by direct code
  read, not just agent report: `src/models/ensemble.py:480-483`.

Also corrected a documentation error while investigating: `CLAUDE.md`'s
UI-freeze list names `docs/index.html`/`docs/data/board.json`/`_site/*` as
locked files — verified none of `docs/`, `_site/`, or `frontend/` exist
anywhere in this checkout (`find`, `git ls-files`, `ls` all empty). The
real current output is `data/players_{POS}.json` and
`data/cached_features.parquet`, neither in the freeze list nor tracked by
git. This fix targeted those, not any `docs/`/`_site/` path.

**Fix**: reused `scripts/snake_draft_sim.py`'s existing
`load_preseason_projections()` (lines ~546-719) — a complete, working,
already-proven implementation of exactly this wiring, whose feature query
deliberately mirrors `PreseasonProjector`'s training-time query (a
same-file code comment warns that a simplified/partial query silently
produces severely under-estimated projections via zero-fill +
`StandardScaler` centering). Reused via import rather than a third copy of
that SQL. Changes:

- `snake_draft_sim.py`: small additive change — `load_preseason_projections`
  now calls `predict_with_details()` instead of `predict()` and carries
  `confidence_score`/`support_class` alongside the existing `pred_total`
  column. No existing behavior changed (`predict()` was already a thin
  wrapper around `predict_with_details()`); 16/16 existing tests still pass.
- `generate_draft_data.py`: new `_load_preseason_projections()` function,
  new `_resolve_projection()` helper implementing a 3-tier fallback chain
  (`preseason_model` → `weekly_18w` → `ppg`-sort), wired into `main()` and
  `output_position_files()`. New `projection_source` field on every player
  record so it's inspectable which tier produced a given number. Floor/
  ceiling reuses the existing `fp_std`-based spread formula, recentered on
  whichever tier is active, scaled by a confidence multiplier
  (`1.0 + (1.0 - confidence_score)`) when available.
- **Deliberate behavior change, flagged explicitly**: `PreseasonProjector`
  consumes no schedule/matchup data at all (verified — no schedule columns
  anywhere in its training query), so unlike `projection_18w` it is NOT
  gated behind `schedule_available`. Players now get real numbers earlier
  in the offseason, before the schedule drops, instead of showing
  "pending."

**A real implementation bug found and fixed during verification, not
anticipated in the plan**: initially called `load_preseason_projections`
with `projection_mode="auto"`, which has its own internal silent fallback
to `ppg × 17` when the model file is missing — and that fallback's output
is indistinguishable from a real ML prediction in the return value, so
every player would have been mislabeled `projection_source:
"preseason_model"` even when the model never loaded. Caught by explicitly
testing the fallback path (moving `preseason_projector.json` aside and
re-running) rather than assuming the try/except made it safe. Fixed by
using `projection_mode="ml"` (raises instead of silently degrading) and
catching that in `_load_preseason_projections()`, so a genuinely missing
model now correctly falls through to the `weekly_18w`/`ppg` tiers with
accurate labeling.

**Verified end-to-end on real data** (2026 draft board, from completed
2025 season, 620 players): 100% of players in every position now sourced
from the preseason model (0 fallback, 0 pending) — notably, `projection_18w`
wasn't even available this run (no cached weekly ML predictions), so
*every* player would have shown "pending" under the old logic. All 160 RBs'
projections changed; 151/160 changed rank order. Zero invariant violations
across all 620 players (`floor <= total <= ceiling`, `floor >= 0`, no
negatives/NaNs). Top-10 RB by new projection: McCaffrey, Bijan Robinson,
Gibbs, Achane, Jonathan Taylor, Cook, Henry, Brown, Barkley, Williams — all
plausible real 2025-season-ending workload leaders. Fallback path
re-verified after the bug fix (model file moved aside → correctly falls to
"pending" when `projection_18w` is also unavailable, no crash). 26/26
relevant tests pass (`test_preseason_projector.py`, `test_generate_draft_data.py`,
`test_snake_draft_sim.py`).

### Unrelated but significant: `.gitignore` bug untracked all of `src/data/` — FIXED, COMMITTED (2026-08-04)

Found while checking `git status` after the LAR/JAC fix: `src/data/entity_
resolver.py` and `src/data/nfl_data_loader.py` — the two files with the
root-cause fix — didn't show as modified despite being edited. Root cause:
`.gitignore`'s `data/` pattern (line 41, meant for the top-level `data/`
DB+model-artifacts directory) is unanchored, so it also matched `src/data/`
and `scripts/data/`. `git log --all -- src/data/` showed **zero commits
ever touched that directory** — 12 files, 548K, core ingestion pipeline
code (`entity_resolver.py`, `nfl_data_loader.py`, `pbp_stats_aggregator.py`,
`external_data.py`, `quality_gates.py`, `injury_validator.py`, etc.) has
had no version history for the life of the repo.

`docs/` (referenced in CLAUDE.md's UI-freeze list — `docs/index.html`,
`docs/data/board.json`, etc.) turned out not to exist on disk at all in
this checkout, so that part of the concern was moot — nothing there to
lose. `scripts/data/` only contained a `cache/` subdirectory (parquet +
metadata, genuinely generated/regenerable) — correctly desired to stay
ignored.

**Fix**: anchored `.gitignore`'s data rule to `/data/` (repo-root only) and
added an explicit `scripts/data/cache/` entry to preserve that one
legitimate exclusion. Verified via `git check-ignore` that `/data/` and
`scripts/data/cache/` are still ignored while `src/data/*.py` is not.
Scanned `src/data/*.py` for credential/secret patterns before staging
(clean). Committed separately from the feature work (`0061e9c`, "Fix
.gitignore scoping bug that untracked all of src/data/") since it's an
unrelated repo-hygiene fix, not part of the §4 feature-wiring changes.

### §11.1.F (DVOA-adjusted opponent FPA) — Phase 1 DONE: fixed a real, currently-live serving-path bug found while planning the feature (2026-08-04)

User asked for the next most impactful remaining item, regardless of
effort. Two other strong candidates — §7.4 (validation/calibration issues:
73%/90% conformal coverage gap, QB target selection methodology,
uncertainty-blend weights, multi-week CI scaling) and §6 (utilization
pipeline circularity) — were investigated via parallel Explore agents and
found to be **mostly stale or moot**:

- §7.4: the 73%/90% coverage gap is already fixed (`position_models.py`
  computes per-level conformal correction from OOF residuals, with a code
  comment quoting that exact finding; `src/evaluation/metrics.py`'s
  `confidence_interval_calibration` + `backtester.py`'s "CI CALIBRATION"
  report evaluate on a genuinely unseen holdout, not OOF as GAPS.md
  claimed). The "QB target selection on val not test" claim is a
  mischaracterization — that's correct ML methodology (the code has an
  explicit comment: "Never use the held-out test set to pick between model
  variants"), not a bug. Two items are real but small: hardcoded
  uncertainty-blend weights (`position_models.py:469,606`, still
  0.5/0.3/0.2 constants) and inconsistent multi-week CI scaling
  (`n_weeks**0.4` vs `sqrt(n_weeks)` in different code paths,
  `ensemble.py:569,641`). Neither is high-impact enough to prioritize over
  DVOA.
- §6: `UtilizationToFPConverter`'s circularity concern is **moot in
  production** — component mode (`position_target_type="component"` for
  all 4 positions) short-circuits before ever reaching that code path
  (`ensemble.py:479-484`). The "utilization doesn't capture efficiency"
  claim is stale — `CAUSAL_FEATURES` already has separate efficiency
  inputs (`yards_per_carry_roll3_mean`, `recv_epa_per_target_roll3_mean`,
  etc.) and `utilization_score` itself is explicitly banned as a feature
  (`src/utils/leakage.py:100`, `ban_utilization_score`). The share→volume
  gap has a kernel of truth (`ComponentPredictor` is Ridge/linear, so it
  can't synthesize `share × team_plays` on its own) but both share and raw
  volume, plus team-pace context (v24, this session), are already present
  as separate inputs — a small optional interaction-term addition, not a
  major missing capability.

DVOA held up as the real, high-impact, still-unbuilt gap — GAPS.md's own
§11.4 ranked table puts it at #1.

**While researching the implementation** (via a Plan agent, before writing
any code), a real, currently-live correctness bug surfaced in the
already-shipped `opp_fpts_allowed` feature — not a DVOA prerequisite, a
bug affecting every live prediction today: `predict.py` overwrites
`opponent` to the upcoming matchup (`predict.py:266` area) and calls
`refresh_matchup_features()`, but that function only recomputed
`team_sos`/`matchup_difficulty`/`opponent_rating`
(`feature_engineering.py:2513-2530` pre-fix) — never `opp_fpts_allowed` or
`opp_fpts_allowed_s2d_lag1`. Those were computed earlier in the pipeline
using the player's **last-played** opponent, before the overwrite. So
every live prediction was scoring the wrong defense's matchup strength.
Per the user's standing instruction above, fixed as its own phase before
touching DVOA — building the new feature on top of the same broken refresh
path would have shipped the identical bug twice.

**Root cause detail, worth remembering:** `opp_fpts_allowed` (unlike
`opp_fpts_allowed_s2d_lag1`, which already queries `team_defense_stats`
directly and was safe to just call from `refresh_matchup_features`) is
computed **inline** inside `_create_opponent_features()` by reading
`fantasy_points_allowed_{qb,rb,wr,te}` columns that were bulk-joined onto
the training frame by `get_all_players_for_training()`'s SQL — i.e. it
depends on pre-joined columns, not a live query. Those columns reflect
whichever opponent the row had *at join time*, so simply calling the
existing logic again post-overwrite wouldn't have helped — the source
columns themselves were stale. Fix: new self-contained
`FeatureEngineer._add_opp_fpts_allowed_from_db()` (`feature_engineering.py`,
after `_add_opp_fpts_allowed_s2d_lag1`) that queries `team_defense_stats`
directly by `(opponent, season, week-1)`, mirroring the SQL join's
leakage-safe semantic. Left the training-path inline logic in
`_create_opponent_features()` completely untouched (still correct and
more efficient for bulk training) — only `refresh_matchup_features()` now
calls the new DB-querying variant.

**A second real bug found and fixed during verification** (not caught by
code review, caught by actually running the fix on real data and
cross-checking against a direct DB query — same lesson as the coaching-
detector and pace-zero-fill bugs earlier this session): the new merge
collided with the stale `fantasy_points_allowed_{pos}` columns already
present on `df` from the original bulk join — pandas silently suffixed
both sides (`_x`/`_y`) instead of erroring, and the code's lookup for the
unsuffixed column name found nothing, so `opp_fpts_allowed` stayed `NaN`
for every row regardless of opponent. Fixed by dropping the stale columns
from `df` before merging in the fresh ones.

Also added: `predict.py`'s `initialize()` now calls
`db.ensure_team_defense_stats()` (previously only the training path did;
serving never refreshed this table, so `team_defense_stats` could be
stale relative to the most recently completed week at prediction time).

**Verified**: built a real player row (2025 season, week 10, RB), swapped
its opponent to a different real team, confirmed `opp_fpts_allowed` and
`opp_fpts_allowed_s2d_lag1` both changed from their pre-swap values and
matched an independent direct SQL query for the new opponent's actual
prior-week defensive stats exactly (26.6 both ways). 94/94 relevant tests
pass (`test_preseason_projector.py`, `test_quality_gates.py`,
`test_target_and_history_causality.py`, `test_advanced_analytics.py`) plus
21/21 in the two files that specifically cover `refresh_matchup_features`
(`test_matchup_aware_prediction.py`, `test_missing_data_and_new_features.py`).
No retrain needed for this phase — it's a serving-time computation fix,
not a training-data or `CAUSAL_FEATURES` change.

### §11.1.F (DVOA-adjusted opponent FPA) — Phase 2 DONE: the actual feature, built and verified (2026-08-04)

**New infra** (`src/utils/database.py`):
- `team_offense_stats` table — sibling of `team_defense_stats`, same
  `(team, season, week)` shape, `fantasy_points_produced_{qb,rb,wr,te}`.
  Deliberately a new table rather than extending `team_stats` — keeps this
  cleanly derived from the single source (`player_weekly_stats`) instead
  of conflating with `team_stats`'s multi-source box-score columns (which
  already had their own zero-fill history this session, for
  `total_plays`/`pace_sec_per_play`).
- `aggregate_team_offense_from_players()` / `ensure_team_offense_stats()`
  — structurally parallel to the existing defense-side functions, just
  grouped by `pws.team` instead of `pws.opponent`.
- Wired into both the training path (`feature_preparation.py`, next to the
  existing `ensure_team_defense_stats()` call) and the serving path
  (`predict.py`'s `initialize()`, next to Phase 1's addition there) —
  same try/except-and-warn pattern as everywhere else.

**New feature** (`src/features/feature_engineering.py`):
`_add_opp_fpts_allowed_dvoa_adjusted_lag1()`, a genuine two-pass
leakage-safe computation:
- Pass 1: each team's own season-to-date offensive output by position
  (`off_s2d_lag1_{pos}`, `shift(1).expanding()` per team-season — identical
  idiom to the existing `opp_fpts_allowed_s2d_lag1`).
- Pass 2: for each defense-week, look up the opponent it actually faced
  (via `team_stats.opponent`) and that opponent's own pre-game expected
  output from Pass 1, compute the residual against what the defense
  actually allowed, then causally expanding-mean that residual per
  defense — i.e. "does this defense over/under-perform its raw
  FPA-allowed number once you account for who it's actually played."
- **Built entirely on team-week-deduplicated, `(team, season, week)`-sorted
  tables, never on the per-player training frame directly** — the exact
  discipline the `CoachingChangeDetector` bug (fixed earlier this session)
  showed is necessary: that bug came from `groupby("team").shift()` on a
  frame sorted by `player_id` first, silently comparing unrelated players'
  rows. Same risk class here, avoided by construction from the start
  rather than found-and-fixed after the fact.
- Called from `_create_opponent_features()` (training) and
  `refresh_matchup_features()` (serving, alongside Phase 1's additions —
  verified separately that swapping a real player's opponent and
  re-running `refresh_matchup_features()` correctly changes the DVOA value
  too, not just the raw `opp_fpts_allowed`/`_s2d_lag1` from Phase 1).

**`CAUSAL_FEATURES` wiring**: added both
`opp_fpts_allowed_dvoa_adjusted_lag1` (new) and `opp_fpts_allowed_s2d_lag1`
(already computed, previously never added despite being noted as a "free
quick win" multiple times in this doc — verified fresh rather than
bundled on the doc's say-so, since Phase 1's bug meant it wasn't actually
trustworthy live until that fix landed) to all four positions, alongside
the existing `opp_fpts_allowed`. `FEATURE_VERSION` bumped to `"26"`.

**Verified**:
- Team-code consistency (lesson from LAR/JAC and OAK/LV): `team_offense_stats`,
  `team_defense_stats`, and `team_stats.opponent` all show exactly the
  same 32 canonical team codes, zero discrepancies.
- Dead-feature check: full `CAUSAL_FEATURES` audit on real 2018+ data
  (43,135 rows) — all four positions clean, no missing declarations.
- Distribution check: `opp_fpts_allowed_dvoa_adjusted_lag1` — 100%
  non-null (post-fillna), nunique=15,153, mean≈0 (expected — it's a
  residual), std=5.98, range [-55.9, 50.7] — real spread, not degenerate.
  A >10%-missing warning fires (11.4%) but 99.7% of those rows are week
  1-2 (the unavoidable cold-start gap — no season-to-date baseline exists
  yet for either the defense or its week-1/2 opponent); confirmed by
  checking the week distribution of the defaulted rows directly rather
  than assuming the warning meant a real bug.
- **A real bug found and fixed during this same verification pass**: the
  new merge initially collided with stale `fantasy_points_allowed_{pos}`
  columns already on `df` from `get_all_players_for_training()`'s bulk
  join — pandas silently suffixed both sides (`_x`/`_y`) instead of
  erroring, so the lookup for the unsuffixed name found nothing and
  `opp_fpts_allowed` came back `NaN` for every row regardless of opponent.
  Caught by cross-checking a live computed value against an independent
  direct SQL query, not by code review — third time this exact "test
  against ground truth, don't trust that the code ran without error"
  lesson has paid off this session (after the WOPR/dead-feature pattern
  and the `total_plays` zero-fill pattern).
- Retrain: `feature_version.txt` = 26, both new features confirmed present
  in `data/models/component_{qb,rb,wr,te}.json` `feature_names` for every
  position (QB 47, RB 51, WR 52, TE 51 total features).
- 115/115 relevant tests pass post-retrain (`test_preseason_projector.py`,
  `test_quality_gates.py`, `test_target_and_history_causality.py`,
  `test_advanced_analytics.py`, `test_matchup_aware_prediction.py`,
  `test_missing_data_and_new_features.py`).

Feature-importance/accuracy-lift measurement against the pre-v26 model
still not done — same standing caveat as every version bump since v23,
now four deep without a `pred_mae` comparison. The one thing this session
did measure (§11.1.F's own entry point) was the aggregate walk-forward
R²=0.345 vs. documented-baseline 0.269 result — that was measured on v25,
before this phase existed, so it doesn't cover v26 either. Worth doing as
its own pass, covering all of v23-v26 at once, before leaning on any of
this for real draft decisions.

### §7.2 (rookie features) — DONE: promoted, but only after finding and fixing a bug that had silently disabled this entire feature module for the project's whole history

Picked as the next item after a periodic-checkpoint discussion (walk-
forward backtests are expensive enough that per-item ablation isn't worth
it, but letting the whole remaining GAPS.md list go unmeasured isn't
either — settled on checking after each natural batch, same cadence as
the v22-v25 checkpoint already done).

Before promoting anything, read the actual computation in
`src/features/advanced_rookie_injury.py` rather than assuming the earlier
session's audit ("genuinely runs in train and serve, just excluded from
CAUSAL_FEATURES") meant it was safe to add wholesale. It wasn't.

**Confirmed real, severe target leakage**: `AdvancedRookieProjector.
calculate_opportunity_score(df, player_id)` sums `fantasy_points` for the
player's team/position across the **entire** `df` passed in, with no
season/week filtering, no shift, no exclusion of future rows. Verified
empirically with a synthetic test: the same rookie's opportunity score
came back **0.4** using only weeks 1-5, vs. **0.986** using the full
season including weeks 6-18 — a massive, unambiguous difference driven
entirely by information that wouldn't exist yet at prediction time. This
matters because `add_advanced_rookie_features()` (line ~898) computes this
**once per rookie** using the full `df` (`profile = self.project_rookie(...,
df=result)`, line ~936) and then broadcasts that single leaked value onto
**every one of that player's rows**, including their week-1 row — so a
rookie's week-1 feature would already "know" how their whole season
played out.

Traced the contamination downstream: `rookie_opportunity_score` itself,
`rookie_breakout_prob` (`calculate_breakout_probability` takes
`opportunity_score` as a direct parameter), and `rookie_ceiling_ppg`/
`rookie_floor_ppg` (both derived from `adjusted_ppg = base_ppg * (0.8 +
opportunity * 0.4)`) are all contaminated. **`rookie_draft_value`**
(`calculate_draft_capital_value(draft_pick)` — pure function of draft
pick, no `df` dependency) and **`rookie_bust_prob`**
(`calculate_bust_probability(draft_pick, position)` — same, no
`opportunity`/`df` dependency) are clean.

**A second, less severe but still real leakage issue found in the same
module**: `AdvancedInjuryPredictor.add_advanced_injury_features()`
computes `weekly_workload = rushing_attempts + targets` **for the row's
own week** (not the prior week), then `season_workload =
groupby(player,season)['weekly_workload'].cumsum()` — which includes that
same current-week value in the cumulative sum. `injury_prob_advanced`,
`injury_workload_risk`, and (via the 0.6/0.4 blend) `injury_prob_combined`
all consume `weekly_workload`/`season_workload` directly, so they use the
current week's own rushing attempts/targets — the outcome of the game
being predicted — as a same-week input feature. Not promoted.

**Confirmed safe and not leaky**: `is_rookie` (derived from `years_exp`/
first-season-in-data, structural, not outcome-based), `rookie_draft_value`,
`rookie_bust_prob` (both pure functions of draft position, established
above), and `combine_score`/`athleticism_grade` (`add_combine_features()`
matches against a static, pre-draft `combine_df` by player name —
verified by reading the join logic, no `df`-outcome dependency at all;
`athleticism_grade` is categorical/redundant with `combine_score` and
would need separate encoding work to use as a Ridge input, so not
promoted — same reasoning as `support_class` earlier this session).

**Status as of this entry: investigation and leakage triage done,
promotion of the safe subset NOT yet completed.** A verification run
(`create_causal_features()` → `add_advanced_rookie_injury_features()` on
real 2024 RB data, 1,408 rows) is still in progress in the background —
it's taking far longer than expected (single-position, single-season
slice still running after several minutes of CPU time, despite this same
step completing within the ~2-3 minute full `--fast --no-tune` training
run in past sessions). Not yet root-caused; worth checking whether
`add_combine_features()`'s row-wise `.iterrows()` + `.str.contains()`
matching against `combine_df` (scanning every row, not just rookies) is
the bottleneck, and whether the full training pipeline benefits from a
warm cache (NGS/combine data already loaded by an earlier step in the
same process) that a standalone script doesn't get.

The two leakage bugs above (`rookie_opportunity_score`/`rookie_breakout_
prob`/`rookie_ceiling_ppg`/`rookie_floor_ppg` via `calculate_opportunity_
score`'s unshifted full-`df` sum; `injury_prob_*`/`injury_workload_risk`
via same-week `weekly_workload`/`season_workload`) remain real and
**dormant** — neither is in `CAUSAL_FEATURES`, so neither is corrupting
any live model. Flagged here so a future session doesn't promote them
without independently rediscovering this.

### A much bigger discovery while verifying the "safe" subset: this entire module has never actually run in production

Running the standard non-degeneracy check (chain the real column-
population order: `create_causal_features()` → `add_season_long_features()`
→ `add_advanced_rookie_injury_features()`, since the rookie/injury step
runs *after* causal features, not inside them) immediately crashed with
`KeyError: 'first_season'`.

**Root cause**: `season_long_features.py`'s `add_rookie_features()` and
`advanced_rookie_injury.py`'s `add_advanced_rookie_features()` both
independently compute an identically-named `first_season` column with the
identical formula (`groupby('player_id')['season'].min()`), and the
first one runs earlier in the real pipeline. When the second one merges
its own freshly-computed `first_season` onto a frame that already has a
column of that name, pandas silently suffixes both sides
(`first_season_x`/`first_season_y`) instead of erroring — so the very
next line's `result['first_season']` lookup throws `KeyError`.

**This is not hypothetical or specific to my test setup — confirmed
against the actual training logs from every feature version shipped this
session.** `grep "skipped" /tmp/train_out_v25.log /tmp/train_out_v26.log`
shows, verbatim, in both: `"Adding advanced rookie features..."` followed
immediately by `"Advanced rookie/injury features skipped: 'first_season'"`.
The crash is caught by a bare `except Exception as e: print(...); return
data` in `feature_preparation.py`'s `add_advanced_features()` — a
textbook instance of Recurring Bug #1 already documented in
PROJECT_NOTES.md ("silent fallback pattern... any new data integration
should log structured warnings instead"), just never previously traced to
this specific module. **Every rookie-draft-capital/comp-player-match/
combine-score/injury-hazard feature this module computes has been dead
code in every training run for the life of this project** — not merely
excluded from `CAUSAL_FEATURES` as the original session-start
investigation concluded (that investigation confirmed the function was
*called*, which was true, but didn't check whether the call actually
*succeeded* — a gap in that verification worth remembering).

**Fix**: `result = result.drop(columns=['first_season'], errors='ignore')`
immediately before `add_advanced_rookie_features()`'s own `first_season`
computation, so the merge always creates a clean, unsuffixed column. One
line, in `advanced_rookie_injury.py`.

### A second bug surfaced by fixing the first one: an uncached, reliably-failing network call per rookie

With the crash fixed, a full-dataset verification run took an
unreasonable 979 seconds for what should have been a fast operation.
Root cause: `AdvancedRookieProjector.project_rookie()` defaults to
`use_comparables=True`, which calls `get_comparable_projection()` →
`_load_historical_rookies()` — a **fresh, uncached `nfl_data_py` network
fetch on every single call** (`import_seasonal_data` + `import_draft_picks`,
no caching, no memoization), called once per unique rookie in the
calling loop. One of the two fetches (`import_draft_picks` for the
current, not-yet-happened season) reliably 404s, so every rookie pays a
real network round-trip that's guaranteed to fail.

Checked what actually consumes the comparable-player blend before
deciding how to fix it: only `rookie_opportunity_score`,
`rookie_ceiling_ppg`, and `rookie_floor_ppg` do — all three already
excluded from promotion due to the leakage bug documented above. **Fix**:
pass `use_comparables=False` in the call site inside
`add_advanced_rookie_features()`. Costs nothing for the four features
actually promoted, and cuts the full-dataset (43,135 rows, 2018+, all
positions) runtime for this step from an untested-but-clearly-multi-
minute number down to **105 seconds** — confirmed reasonable relative to
the ~2-3 minute total `--fast --no-tune` training run.

### Final promotion — verified and shipped

`CAUSAL_FEATURES` gained `is_rookie`, `rookie_draft_value`,
`rookie_bust_prob`, `combine_score` for all four positions.
`FEATURE_VERSION` bumped to `"27"`.

**Verified**:
- Full dead-feature audit (real production column order, 43,135 rows,
  2018+, all positions): `CAUSAL_FEATURES` clean, zero missing
  declarations.
- Distribution check: `is_rookie` nonnull=100%, 26.7% flagged rookie
  (somewhat elevated vs. a true ~10-15% rookie rate — an artifact of the
  2018+ data window used for this check, where players whose real debut
  predates 2018 but who first appear in the *queried slice* at 2018 look
  like rookies; not a new bug, an inherent limitation of first-season
  derivation on any truncated window, noted but not chased further since
  production presumably has deeper history available for this
  computation). `rookie_draft_value`: nunique=244, range [0.1, 1.0], real
  spread by draft position. `rookie_bust_prob`: nunique=20 (tiered by
  draft-pick bucket, as designed). `combine_score`: nunique=210, range
  [10, 92], real spread.
- Retrain (`python -m src.models.train --fast --no-tune`): completed
  clean, **no "skipped" message** — first time this module has ever
  successfully executed inside a real training run for this project.
  `feature_version.txt` = 27.
- Model artifact confirmation: all four features present in
  `data/models/component_{qb,rb,wr,te}.json` `feature_names` (QB 51, RB
  55, WR 56, TE 55 total features).
- 155/155 relevant tests pass (`test_preseason_projector.py`,
  `test_quality_gates.py`, `test_target_and_history_causality.py`,
  `test_advanced_analytics.py`, `test_matchup_aware_prediction.py`,
  `test_missing_data_and_new_features.py`, `test_rookie_projections.py`,
  `test_training_pipeline.py`).

**Not attempted**: fixing the two dormant leakage bugs
(`rookie_opportunity_score` and the injury same-week-workload features) —
real, scoped, but a separate task from "promote the already-safe subset."
Root-causing why the full-dataset run showed a still-somewhat-elevated
rookie rate (worth checking whether real `_prepare_training_data` loads
deeper history than the `season >= 2018` slice used for this session's
verification, which would resolve it without any code change).

### Both dormant leakage bugs from §7.2 — FIXED (2026-08-04, new session)

Both bugs flagged above as "not attempted" are fixed. Neither feature is in
`CAUSAL_FEATURES` yet, so this is a correctness fix to dead code, not a
retrain-triggering change — no model artifacts touched, no `FEATURE_VERSION`
bump. Both are still explicitly *not* promoted this session; that remains a
separate decision.

**Bug 1 — `calculate_opportunity_score` full-season leakage
(`src/features/advanced_rookie_injury.py`).** Previously summed
`fantasy_points` across the entire df with no time filter, computed once per
rookie *player*, then broadcast to every row of that player including week 1.
Fix:
- `calculate_opportunity_score` gained optional `as_of_season`/`as_of_week`
  params; when given, `team_data` is filtered to strictly-prior
  (season, week) rows before computing the position share.
- New `_compute_prior_opportunity_scores(df)` — a vectorized equivalent
  (team-week and team-position-week fantasy-point totals via
  `groupby(...).shift(1).cumsum()`, the same idiom already established in
  this file's DVOA/team-offense work, merged back onto every row) — needed
  because the fix also moves `add_advanced_rookie_features`'s loop from
  once-per-player to once-per-rookie-*row* (each week now gets its own
  opportunity value instead of one value broadcast across the season), and
  re-filtering the full frame per row via the non-vectorized path would have
  been slow at that granularity.
- `project_rookie()` gained an `opportunity_score` passthrough param so the
  per-row caller can hand in the precomputed value directly instead of
  re-deriving it from `df` per call.

**Bug 2 — same-week workload in `add_advanced_injury_features`.**
`weekly_workload` used the row's own `rushing_attempts`/`targets` (the
outcome of the game being predicted) and `season_workload` was an unshifted
`cumsum` that included that same value. Fix: sort by
`(player_id, season, week)`, shift `weekly_workload` by 1 within
player-season before use, and derive `season_workload` as the cumsum of the
*shifted* series (so it only ever sums weeks strictly before the current
row). Falls back to the old unshifted behavior only if `week` isn't present
in `df` (can't safely order without it) — same defensive pattern as the rest
of this module.

**Verified**:
- Synthetic tests for both: a 4-week rookie with weeks 2-4 having much higher
  output than week 1 now gets `rookie_opportunity_score == 0.5` (the no-data
  default) at week 1, rising only as prior weeks accrue — confirmed it no
  longer front-loads future performance. A 3-week workload synthetic
  confirms `weekly_workload`/`season_workload` are both 0 at week 1 and
  correctly reflect only prior weeks thereafter.
- Real-data run (`get_all_players_for_training()` → `create_causal_features()`,
  2022+, 21,889 rows): `add_advanced_rookie_features` now runs in 0.6s
  (8,254 rookie-rows, 843 unique rookies), `add_advanced_injury_features` in
  0.8s — both comfortably inside the existing `--fast --no-tune` training
  budget, confirming the per-row loop (up from per-player) didn't reintroduce
  the earlier performance problem from this module's history.
  `rookie_opportunity_score` has real spread (mean 0.58, std 0.23, range
  [0, 1]) and — checked directly — is *not* constant across a given player's
  weeks anymore, it's cumulative team-history-to-date so it drifts smoothly
  week to week rather than being either a flat broadcast value or a hard
  reset to 0.5 every season (matches the original design intent of "team's
  historical usage," now just without the future leak).
- 149/149 relevant tests pass (`test_preseason_projector.py`,
  `test_quality_gates.py`, `test_target_and_history_causality.py`,
  `test_advanced_analytics.py`, `test_matchup_aware_prediction.py`,
  `test_missing_data_and_new_features.py`, `test_rookie_projections.py`).
- `git status`: only `src/features/advanced_rookie_injury.py` touched.

**Not attempted / still open**: promoting `rookie_opportunity_score`,
`rookie_breakout_prob`, `rookie_ceiling_ppg`, `rookie_floor_ppg` to
`CAUSAL_FEATURES` now that they're leakage-safe — a separate decision (would
need its own retrain + verification pass, same as every other promotion this
project has done). The "still-somewhat-elevated rookie rate on a truncated
window" question from the prior entry also remains unexamined.

### `rookie_opportunity_score`/`rookie_breakout_prob`/`rookie_ceiling_ppg`/`rookie_floor_ppg` — PROMOTED to v28 (2026-08-04, new session)

Closed out the item flagged above as "not attempted" — the four rookie
features excluded from v27 specifically because of the two leakage bugs are
now leakage-safe, so promoted them.

Added to `CAUSAL_FEATURES` for all 4 positions in `config/settings.py`,
`FEATURE_VERSION` bumped to `"28"`.

**A verification wrinkle worth recording**: an initial standalone repro
(`create_causal_features()` → `add_advanced_features()` directly, skipping
`add_season_long_features()`) showed `rookie_floor_ppg` completely
degenerate (nunique=1, always 0.0) for TE and QB specifically. Traced this
to `draft_round`/`draft_pick` not existing in that shortened pipeline at
all — `add_advanced_rookie_features()` silently defaults every rookie to
`draft_round=5, draft_pick=150` (worst tier) when those columns are
missing, and for TE/QB's `round_3_plus` archetype the floor formula
(`max(0, adjusted_ppg - 1.5*std)`) is mathematically negative for every
possible opportunity value in [0,1] — so it's *always* 0 regardless of the
leakage fix, purely a downstream artifact of my simplified repro skipping
`season_long_features.add_season_long_features()` (which is what actually
merges real `draft_round`/`draft_pick` from the `draft_picks_v2` table,
step "0. Merge draft data before rookie features" in that module, and does
run before `add_advanced_features()` in the real training path via
`feature_preparation.py`'s `_prepare_training_data`). Re-ran through the
correct real-pipeline order and all four features showed real spread for
every position (e.g. TE `rookie_floor_ppg`: nunique=535, mean=0.218,
std=0.491; QB: nunique=642, mean=2.779, std=2.995) — not a real bug, but a
reminder (yet again) that a shortened verification script that skips a
real pipeline stage can manufacture a false-positive "degenerate feature"
finding. Worth remembering for the next session that reaches for a quick
repro instead of the full `_prepare_training_data` path.

**Retrain**: `python -m src.models.train --fast --no-tune`, exit 0,
`feature_version.txt` = 28, previous version archived. All 4 new features
confirmed present in `data/models/component_{qb,rb,wr,te}.json`
`feature_names` (QB 55, RB 59, WR 60, TE 59 total). 149/149 relevant tests
pass post-retrain (same suite as the leakage-fix entry above).

### v23–v28 accuracy-lift measurement — DONE, honest result: flat since v25

The standing caveat repeated after every version bump since v23 ("worth
doing before leaning on this for real draft decisions") finally measured,
covering v26/v27/v28 (v23-v25 were already covered by the earlier
mid-session checkpoint). Same tool/methodology as that checkpoint:
`python scripts/run_ts_backtest.py --season 2025 --model ridge --alpha
10000` (2025 holdout, expanding-window, weekly refit, 22 weeks, 5,612
predictions).

| Metric | v28 (this run) | v25 checkpoint | Pre-v22 documented baseline |
|---|---|---|---|
| Overall R² | 0.346 | 0.345 | 0.269 |
| Overall MAE | 4.79 | — | — |
| Overall RMSE | 6.38 | — | — |
| QB R² | 0.226 | 0.223 | 0.092 |
| RB R² | 0.366 | 0.364 | 0.258 |
| WR R² | 0.277 | 0.276 | 0.257 |
| TE R² | 0.264 | 0.263 | 0.152 |

**Honest verdict: v26 (DVOA-adjusted opponent FPA), v27 (rookie
identity/draft-capital/combine), and v28 (rookie opportunity/breakout/
ceiling/floor) show no measurable aggregate lift over v25 on this
backtest** — every number above is within noise of the v25 checkpoint, not
a regression but not a detected improvement either. The big, real jump
(0.269 → 0.345) happened between the documented baseline and v25; nothing
shipped since has moved this particular metric.

This does not mean v26-v28 were wasted work — plausible reasons the signal
doesn't show up here, none of them chased down further this session:
- Ridge α=10,000 is heavy regularization; a handful of new features among
  55-60 total per position get shrunk hard, especially sparse/rookie-only
  columns that are the default value for ~75-85% of non-rookie rows.
- Single holdout season, single model type (Ridge) — not the higher-
  fidelity `--model ensemble` path, which is too slow to run routinely
  (PROJECT_NOTES Phase 4: ~12.8h wall clock, killed by a pre-registered
  kill gate). A real but nonlinear-only signal would wash out here.
  Same caveat as the v25 checkpoint: this isn't a precise paired A/B
  (aggregate R² across all players), so it wouldn't isolate a rookie-
  specific effect even if one exists — a rookie-only-rows slice would be
  the right instrument for that, not attempted here.

**Not attempted**: per-version ablation (training with v26/v27/v28
features individually excluded to isolate which, if any, helped or hurt)
— explicitly scoped out before as a "further-refinement question, not a
production-readiness blocker," and that scoping still holds. This
measurement closes out the repeated "worth doing" caveat honestly rather
than leaving it to roll forward again, but doesn't replace a real
ablation study if one is ever wanted.
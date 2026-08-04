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

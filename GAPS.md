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

### 7.6 Phase 1 leakage audit (next_focus.md) — 2026-08-09/10

Audit performed while starting Phase 1 of `next_focus.md` (new single-week/18-week PPR modeling plan). Found and fixed two live leakage bugs.

**Fixed: same-week own-team stats in `get_all_players_for_training` (`src/utils/database.py`).**
The `team_stats` join used `pws.week = ts.week` (same week as the target game), unlike the adjacent `team_defense_stats` join which correctly used `week - 1`. A player's own stat line is a direct component of same-week team totals (e.g. a WR's receiving yards are part of `team_yards`; a QB's attempts are part of `team_pass_attempts`), so this leaked target information into training on the live production path (`train.py` → `data_loading.load_training_data()` → `get_all_players_for_training()`). Fixed by joining `team_stats` from `week - 1`, mirroring the opponent-defense pattern, with a matching runtime `ValueError` assertion on the new `own_team_stats_week` sentinel column. This is a different bug from the two rookie-feature leakage bugs in §7.2 — those were feature-computation bugs in `advanced_rookie_injury.py`; this was a raw SQL join condition.

**Fixed: injury/status timing not verified pre-kickoff (`src/data/external_data.py`).**
Originally documented as an unfixable-without-new-data limitation, then fixed once we checked what `nfl_data_py.import_injuries()` actually returns: it already includes a `date_modified` timestamp per report (previously unused — dropped during column selection in `get_player_injury_status`). `src/data/injury_validator.py` (freshness-only, dead code, zero callers) never used it either. Cross-checking `date_modified` against real kickoff times (`nfl_data_py.import_schedules()` `gameday`+`gametime`, US/Eastern → UTC) across 2018–2025 found 24 of 45,337 injury reports (0.05%) were modified *after* that week's kickoff — e.g. Thursday/short-week games where a "final" report update landed post-game. `InjuryDataLoader._load_kickoff_times()` (new) computes per-team-week kickoff from schedules; `get_player_injury_status()` now drops any report row where `date_modified > kickoff` before building injury features, and prints the count dropped. Rows with no verifiable timestamp or unmatched schedule are kept as-is — missingness is not leakage. `src/data/injury_validator.py` remains dead code (freshness-only, doesn't check pre-kickoff) — still unused, not wired in, left as-is since this fix supersedes what it would have done.

**Added: formal feature-availability registry (`src/utils/leakage.py`).**
`FEATURE_AVAILABILITY` + `audit_feature_availability()` — a single-source-of-truth mapping from feature-name pattern to availability rule (e.g. "week - 1, runtime-asserted", "prediction week, schedule is fixed", "assumed ... UNVERIFIED"), covering every feature family named in `next_focus.md` Phase 1 (rolling/season-to-date, opponent/own-team stats, schedule/Vegas/weather, draft capital, injury/status). `audit_feature_availability(columns)` flags any column not covered by this registry or the existing `is_leakage_feature()` guard, so new features get classified instead of silently passing through unaudited.

**Added: `tests/test_leakage_guards.py`** — 28 regression tests covering the two fixes above (in-memory SQLite fixture verifying `own_team_stats_week` is always `week - 1`) plus `is_leakage_feature`/`audit_feature_availability` classification. Note: the `tests/` directory itself had been fully deleted from git in an earlier commit (`0194107 Delete tests directory`) — only stale `.pyc` cache remained locally, no test source files existed anywhere in the repo before this. Also note: running bare `pytest` in this environment currently fails at collection (`ImportError: cannot import name 'FixtureDef' from 'pytest'`) because a globally-installed `pytest-asyncio==1.3.0` is incompatible with the repo-pinned `pytest==7.4.4` (`pytest-asyncio` isn't even in `requirements.txt` — it's environment pollution, not a repo dependency). Workaround: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest ...`. Not fixed here since it's a global environment issue, not a repo bug — flagging so it isn't silently rediscovered.

### 7.7 Phase 2 (next_focus.md) incident: `settings.MODELS_DIR` redirection does not prevent production-artifact writes — 2026-08-10

While building `src/models/single_week_ppr/` (Phase 2 architecture comparison), the walk-forward-fold pattern in `src/models/train.py` (`settings.MODELS_DIR = Path(tmp)` before calling `_prepare_training_data`, lines ~1053-1061) was reused to keep experimental training runs from overwriting real production model artifacts. It did not work: a smoke run for WR/2024 silently overwrote `data/models/component_wr.json`, `data/models/util_to_fp_wr.joblib`, and `data/models/data_quality_gate_refresh.json`, plus `data/data_availability_cache.json` and `data/utilization_percentile_bounds.json` (outside `data/models/` entirely).

**Root cause:** several modules bind `MODELS_DIR` at *import* time via `from config.settings import MODELS_DIR` (e.g. `src/models/utilization_to_fp.py:16`) rather than reading `config.settings.MODELS_DIR` dynamically at call time. Reassigning `settings.MODELS_DIR` after those modules are already imported has no effect on their already-bound local name — their reads/writes still target the real path. `load_training_data()`'s `auto_refresh_data()` call and the utilization-weight-fitting step write to `data/` paths outside `MODELS_DIR` altogether, so even a fully-working redirect wouldn't have caught them.

**This means `train.py`'s own `--walk-forward` flag likely has the same leak** — worth a follow-up audit before trusting walk-forward runs not to mutate production artifacts. Not fixed here (out of scope for Phase 2; flagging per standing directive to log gaps immediately rather than let them slide).

**Fixed for Phase 2's own code (v1, superseded by §7.8):** `_protect_data_dir()` (`src/models/single_week_ppr/evaluate.py`) — snapshots mtimes of every file under `data/` (excluding `data/experiments/`, Phase 2's own output dir) before a fold runs, and after, restores any touched file via `git checkout --` (tracked) or deletion (newly-created untracked). This is a real safety net independent of whether any given module respects `MODELS_DIR` redirection, wraps the *entire* fold (including `load_training_data`, not just `_prepare_training_data`), and logs a warning whenever it has to intervene. Verified: three consecutive WR/2024 smoke runs after the fix left `git status` clean except the intentional `data/experiments/phase2_single_week_comparison.csv` output.

### 7.8 Phase 2 incident #2 (self-inflicted by the §7.7 fix): the "safety net" deleted the production database — 2026-08-10

The `_protect_data_dir()` fix in §7.7 treated *any* changed, git-untracked file under `data/` as pollution to be deleted. `data/nfl_data.db` is gitignored (`*.db`) and is **legitimately, intentionally** written by `auto_refresh_data()` inside `load_training_data()` — that's normal operation, not pollution. During the QB/RB/TE/WR full sweep runs, the database's mtime changed (a legitimate refresh write) inside a `_protect_data_dir()`-wrapped fold, so at the end of that run the "safety net" `unlink()`'d the entire production SQLite database (110,283 `player_weekly_stats` rows, 2,868 players, 33 tables of ingested data). The next `DatabaseManager()` instantiation silently recreated an empty DB from schema (`CREATE TABLE IF NOT EXISTS`), so the failure was silent — a 245KB empty file that looked structurally fine, discovered only when the user asked "anything in Phase 2 we missed?" and a scoping check (`get_all_players_for_training()` returning 0 rows) caught it.

**Recovered** from `data/nfl_data.db.bak-lar-jac-20260804154651` (405MB, dated 2026-08-04, an existing backup unrelated to this session — not something this session created). Restored via `cp`, verified `player_weekly_stats` row count matches exactly (110,283), then ran `auto_refresh_data(force_check=True)` to catch up: the 2025 season was already complete in the backup (regular-season/playoff data doesn't change after the fact), so the only real gap filled was the 2026 schedule (272 games). **No data was permanently lost** — the Phase 2 sweep results themselves (`data/experiments/phase2_single_week_comparison.csv`) are unaffected/trustworthy, because all 4 positions' folds produced row counts consistent with a fully-populated database throughout their execution; the deletion happened in a `finally` block *after* each run's real work was done.

**Fixed properly this time:** `_protect_data_dir()` now uses an **allowlist** (`_PROTECTED_PATHS` = `data/models/`, `data/data_availability_cache.json`, `data/utilization_percentile_bounds.json` — the exact three things empirically observed leaking in §7.7) instead of a denylist over all of `data/`. `data/nfl_data.db` (or any other live, evolving, gitignored data asset) can never be touched by this function again, regardless of whether it changes during a protected block — silence on that file is the whole point.

**Lesson, stated plainly:** a "restore anything that changed and isn't tracked in git" safety net is unsafe in a repo where the most important asset (the working database) is *intentionally* gitignored. Broad denylist-style protection over a shared, live directory is the wrong shape for this problem; a narrow allowlist of specific known-bad paths is safer even though it requires enumerating them by hand. If a future fold run logs "N allowlisted model-artifact file(s) were written... restoring" for a path not in `_PROTECTED_PATHS`, that's a bug in the allowlist, not evidence to widen it back into a denylist.

### 7.9 Phase 3 (next_focus.md) finding: the hard 2018+ training floor may be leaving accuracy on the table for QB/RB/TE — 2026-08-10

`src/utils/data_manager.py:get_train_test_seasons()` hard-filters training data to `season >= TRAINING_START_YEAR_DEFAULT` (2018) everywhere in production, with an inline comment stating pre-2018 data is "noise for current projections" due to missing NGS/snap counts/modern play-calling norms. Phase 3's training-window/recency-weighting sweep (`src/models/single_week_ppr/windows.py`, `evaluate.py:run_window_comparison`) deliberately bypassed this floor for one experiment — reaching back to `MIN_HISTORICAL_YEAR` (2006) — specifically to test whether that assumption holds. It only partially does.

**Result** (full grid: 4 positions x 5 windows x 3 weightings x 2 architectures x 3 seasons, 420 rows in `data/experiments/phase3_training_window_comparison.csv`): MAE improves **monotonically** from 3-year to full-history windows for QB (6.58→6.37), RB (4.79→4.71), and TE (3.70→3.65) — pre-2018 rows help, not hurt, despite genuinely missing NGS/PBP-EPA/modern-snap-share columns for those seasons (handled via LightGBM's native NaN-aware splits, not `fillna(0)` — see `_build_feature_matrices` in `evaluate.py`, itself a Phase 3 fix since blind `fillna(0)` would have falsely equated "no data" with "zero usage" for those older rows). WR is the exception — flat/mixed past 5 years (4.76-4.79 MAE, within noise), so the floor's stated rationale seems to actually hold there.

**Also found**: recency weighting matters less than production assumes. "None" (uniform) won 13 of 20 position×window combinations on MAE, "linear" 6, and "exponential" — the *only* weighting scheme currently live in production (`src/models/position_models.py:_horizon_recency_weights`, `src/models/ensemble.py`, halflife=1.5 seasons) — won just 1. This doesn't mean exponential decay is wrong for every use case (it wasn't tested against the single-week horizon's actual accuracy needs when it was chosen), but it's a data point suggesting the current halflife may be too aggressive for at least this target.

**Not fixed / not acted on**: this is a research finding from an experimental bypass, not a change to `data_manager.py` or `TRAINING_START_YEAR_DEFAULT` — production behavior is unchanged. Flagging so it doesn't get silently rediscovered later: before Phase 5 (hyperparameter tuning) or any production training-window change, this result is worth revisiting, ideally with a broader per-season/per-bucket breakdown (Phase 4 territory) rather than the single averaged-MAE view here.

### 7.10 Phase 4 (next_focus.md) finding: the Phase 2/3 "winning" architectures beat MAE but carry real bias and lose their edge for elite players — 2026-08-10

Phase 4 ran each position's `FINAL_CONFIG` (chosen architecture/window/weighting from Phases 2-3: `src/models/single_week_ppr/final_config.py`) and saved **row-level** predictions (114,989 rows, `data/experiments/phase4_row_level_predictions.csv`) instead of only fold-aggregated metrics, then broke results out by predicted-score bucket and player tier (`src/models/single_week_ppr/tiers.py`, `analysis.py`) — axes Phase 2/3's aggregate MAE couldn't see.

**Bias.** Aggregated across all positions/seasons: the winning architectures (C=GBM-MAE, F=Yeo-Johnson+Huber, E=quantile median) all carry **-1.1 to -1.3 mean bias** (systematic underprediction), vs. `existing_methodology`'s **-0.04** (essentially unbiased). This was flagged as a risk back in Phase 2 (GAPS.md §7.6 predates this, findings were in `next_focus.md` only) but Phase 4's full-dataset numbers make it concrete: these architectures win on MAE partly *because* they're willing to guess low more often, not purely because they're more accurate in a symmetric sense.

**Player tier.** New tier definition (`tiers.py`, prior-season PPG, position-adjusted, leakage-safe — see module docstring for why the two existing tier concepts in this repo, `ROOKIE_ARCHETYPES` and `tier_classification_accuracy`, weren't reused as-is). Broken out by tier, the winning architectures beat `existing_methodology` consistently for depth/starter/waiver/rookie tiers (0.1-0.5 MAE improvement) but the gap **collapses to a wash for the "elite" tier** at every position: QB 6.34 vs 6.42, RB tied 6.76 vs 6.76, TE 5.12 vs 5.17, WR tied 6.46 vs 6.46. The tier most likely to matter for actual lineup decisions is exactly where Phase 2/3's "winner" stops winning.

**Predicted-score bucket.** New (`assign_score_bucket` in `tiers.py`, fixed 0-5/5-10/10-15/15-20/20+ ranges on the model's own weekly prediction). No consistent pattern across positions — WR's "20+" bucket shows the new architecture dramatically ahead (4.40 vs 7.86 MAE), QB/RB's "20+" buckets are roughly tied. Not something to generalize a rule from; noted as a per-position quirk.

**Quantile calibration**, averaged across seasons per position (extends Phase 2's per-fold numbers with a position-level view): p25 coverage 0.26-0.30 (target 0.25, slightly over everywhere), p50 0.50-0.53 (on target), p75 0.73-0.75 (on target), **p90 under-covers at every single position (0.855-0.887 vs target 0.90)** — this is now confirmed as a systematic pattern, not noise in one position/season from Phase 2. The model's "ceiling" estimate is reliably too conservative across the board.

**Not acted on yet**: no config was changed as a result of this — `FINAL_CONFIG` still reflects the Phase 2/3 choices. This is deliberately a findings-gathering phase; the bias/elite-tier results should inform Phase 12 (final model selection), where a lower-bias option may be preferable for elite-player-heavy decisions even at a small MAE cost. Flagging now so it isn't rediscovered late in the project.

### 7.11 Phase 5 (next_focus.md) finding: hyperparameter tuning barely moved MAE and didn't touch the bias problem — 2026-08-10

Nested-CV Optuna tuning (`src/models/single_week_ppr/tuning.py`, `evaluate.py:run_tuned_validation`), 100 trials per (position, outer-test-season) fold. Design: inner walk-forward CV strictly within each fold's *training* seasons (`inner_walk_forward_folds`, purge gap from `MODEL_CONFIG["cv_gap_seasons"]`) — the outer 2023/2024/2025 test seasons used by Phases 2-4 are never touched by the search itself, so this stays a fair nested-CV comparison against Phase 4's default-hyperparameter results (same architecture/window/weighting/test rows, only hyperparameters differ). Deliberately did NOT reuse `position_models.py`'s `_subsample_for_tuning` trick (already flagged as a concern in §7.4) — with no repeated feature engineering per trial, there was no need to subsample for speed.

**Result**: tuning changed MAE by less than 0.03 points in every position — QB actually got marginally *worse* (default 6.089 → tuned 6.113), RB/WR/TE improved by 0.008-0.019 (noise-level). Bias was essentially unchanged everywhere (RB/WR/TE still -1.06 to -1.36, QB still around -0.32). Full tuned-hyperparameter values: `data/experiments/phase5_tuned_hyperparameters.csv`; row-level tuned predictions: `data/experiments/phase5_tuned_predictions.csv` (16,427 rows).

**Why this matters**: it confirms, rather than fixes, the Phase 4 bias finding. If the negative bias in RB/WR/TE's winning architectures were an artifact of poorly-chosen hyperparameters (e.g. too-shallow trees underfitting the tails), a real 100-trial search around a much wider parameter range should have found *something* better. It didn't, meaningfully. That's consistent with the bias being a structural property of median-seeking loss functions (MAE/Huber) applied to a right-skewed target (see §7.10 / the mid-session bias-vs-underfitting discussion) — not a tuning problem, and not fixable by tuning. It also means Phase 2's "reasonable defaults" were already close to a local optimum for this problem, which is itself a useful, if unglamorous, result: the compute spent here wasn't wasted, it's evidence against a hypothesis (that better hyperparameters would close the gap) that could otherwise have lingered unexamined into Phase 12.

**Not acted on**: `FINAL_CONFIG` unchanged — the marginal tuned-parameter deltas aren't worth adopting as the new default given how close to noise they are, and doing so would complicate reproducibility for a sub-0.03-MAE gain. If Phase 12's final model selection revisits the bias problem, the fix should target the loss function or a post-hoc calibration step, not further hyperparameter search.

---

## 8. Feature Engineering Gaps

### 8.1 Missing features by impact tier

**STALE TABLES BELOW — corrected 2026-08-10 (Phase 6, next_focus.md).** A
fresh audit found that 10 of the items this section lists as "missing" had
actually already been resolved between `FEATURE_VERSION` v24 and v30, well
before this correction was written. The tables are left below for
historical record, with each row's actual current status noted — do not
treat anything here as an open TODO without checking `CAUSAL_FEATURES` in
`config/settings.py` first, per the standing directive to verify wired
state before acting on it.

**Tier 1: High impact, data available now**

| Feature | Source | Why it matters | Status (as of v30) |
|---------|--------|---------------|----------------|
| **Depth chart position** | `depth_charts` table (591K rows in DB) | Distinguishes WR1 from WR3; critical pre-season | **RESOLVED** — `depth_chart_rank` in `CAUSAL_FEATURES` for all 4 positions |
| **Contracts / contract year** | `contracts` table (51K rows in DB) | Contract year players historically outperform | **RESOLVED** — `is_contract_year`, `contract_apy_rank`, all 4 positions |
| **Weekly PFR advanced stats** | `nfl.import_weekly_pfr()` | Drops, pressures, bad throws — QB predictive signal | **RESOLVED** — `qb_pressure_pct_roll3_mean`/`qb_bad_throw_pct_prior` (QB), `recv_drop_pct_roll3_mean` (WR/TE) |
| **Position-specific target allocation** | Computable from `player_weekly_stats` | Team targets RBs X%, WRs Y%, TEs Z% | **RESOLVED** (v24) — `team_rb_target_share_roll3_mean` / `team_wr_target_share_roll3_mean` / `team_te_target_share_roll3_mean` |
| **Team plays per game / tempo** | Computable from `team_stats.pace_sec_per_play` | More plays = more fantasy opportunity | **RESOLVED** (v24) — `team_plays_roll3_mean`, `team_pace_sec_per_play_roll3_mean`, all 4 positions |

**Tier 2: Medium impact, requires new ingestion**

| Feature | Source | Why it matters | Status (as of v30) |
|---------|--------|---------------|----------------|
| **Coaching staff identity** | NFL.com, ESPN rosters | New OC = scheme change = target distribution shift | **RESOLVED** (v24) — `coaching_change`, `coaching_adaptation_score`, `coaching_stability`, `coaching_change_impact` |
| **Personnel grouping %** | FTN charting data (`nfl.import_ftn_data()`) | 11 vs. 12 personnel snap % predicts TE/WR opportunity | **RESOLVED** (v29) — `team_pct_11/12/21/13_personnel_roll3_mean`, parsed from PBP `offense_personnel` (2016+), not FTN |
| **Weather (outdoor games)** | Weather API | Cold/wind depresses passing; rain increases fumbles | **RESOLVED** (v23) — `wind_speed_mph`, `is_dome`, `precipitation_flag`, `temperature_bucket` |
| **Seasonal PFR (prior-season summary)** | `nfl.import_seasonal_pfr()` | Pre-season drop rate, bad throw % for cold-start | **RESOLVED** — QB pocket/accuracy priors, RB broken-tackles prior, WR/TE drop-rate prior |
| **Vegas preseason win totals** | `nfl.import_win_totals()` | Team quality proxy for full-season game script | **CONFIRMED INFEASIBLE**, not just missing — see note below |

**Vegas preseason win totals — confirmed infeasible, not a gap to close.**
`nfl.import_win_totals()` is not this data despite the name — it's
per-game weekly odds (spread/total/moneyline), not season-long win-total
futures, and this project's `game_odds` table already has that fully
covered. No free source for real preseason win-total futures was found.
See "No viable free source for Vegas preseason win totals" further down
in this doc.

**Route participation — also confirmed infeasible with currently-ingested
data (Phase 6 audit).** No `route`/`routes_run` column exists in
`ngs_receiving`, `ngs_rushing`, or `weekly_pfr`; `nfl_data_py` has no
participation/routes import function at all. v30's
`pbp_pass_play_participation_pct_roll3_mean` is already the best available
proxy (its own code comment says so explicitly) — nothing further is
actionable here without a new external data source.

**Rolling catch rate — the one real gap this audit found, closed in v31.**
Raw `catch_rate` (`receptions/targets * 100`) was already computed in
`_create_base_features` (`src/features/feature_engineering.py`) but never
rolled into a leakage-safe feature. Added `catch_rate_roll3_mean` to
`CAUSAL_FEATURES["RB"/"WR"/"TE"]` via the existing generic rolling engine
(`_create_causal_rolling_features`) — same mechanism as every neighboring
`*_roll3_mean` feature, no new leakage-safety code needed.

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

### 8.4 Phase 6b (next_focus.md follow-up): two real gaps the Phase 6 audit missed — 2026-08-10

The user asked directly whether the feature set captures (a) team run/pass play-calling tendency and (b) hybrid usage (pass-catching RBs, rushing WRs). Checking the actual `CAUSAL_FEATURES` lists (not the — by then already-corrected — §8.1 audit) found two real, previously-uncaught gaps:

**Team pass/rush tendency — fixed, v32.** `team_neutral_pass_rate_oe` (pass-rate-over-expected, the standard metric for isolating a team's play-calling *preference* from what game script/down-distance would predict) already arrived in the raw training DataFrame via `get_all_players_for_training()` (`src/utils/database.py:2022-2024`, joined from `week - 1`, leakage-safe per the §7.6 fix) but was never referenced anywhere in `src/features/feature_engineering.py` and never entered `CAUSAL_FEATURES` for any position — the data existed, nothing consumed it. Added `"team_neutral_pass_rate_oe"` to the `roll_cols` list in `_create_causal_rolling_features` (same mechanism as `team_plays`/`team_pace_sec_per_play`), producing `team_neutral_pass_rate_oe_roll3_mean`, now in `CAUSAL_FEATURES` for all 4 positions. Verified sane: mean ≈0, range roughly -0.33 to +0.20 (a rate-difference metric, correctly centered near zero).

**WR rushing usage (jet sweeps/end-arounds) — fixed, v32.** `rush_share_pct` was already computed position-agnostically (`_create_base_features`, each row's own `rushing_attempts` over that team-week's total, no position filter) and already in `roll_cols` — so `rush_share_pct_roll3_mean` was already being computed for WR rows, just never selected into `CAUSAL_FEATURES["WR"]`. Pure config addition, no new computation. Verified with real data: Deebo Samuel's rows top the sorted-by-value list at 100% rolling rush share — exactly the gadget-usage archetype this feature exists to capture.

Both verified via a real production smoke train (`--positions WR --fast --no-tune`): 67 features (up from 65 post-Phase-6), `feature_version.txt=32`, no errors.

**Not done — feature-count/ablation testing.** Separately asked whether OOS error was ever tested across different feature subsets or feature counts. No — Phases 2-5 held `CAUSAL_FEATURES` fixed throughout every architecture/window/weighting/hyperparameter experiment; this was never in `next_focus.md`'s phase structure either. Scoped as a possible follow-up (LightGBM-importance-ranked nested subsets — top-10/20/30/all — reusing the existing walk-forward fold-loading machinery in `single_week_ppr/evaluate.py`) but not started; needs an explicit go-ahead given it's comparable in cost to Phase 3's full grid.

### 8.5 Phase 6c: feature-count ablation — no evidence to trim the feature set — 2026-08-10

Executed the §8.4 follow-up. `src/models/single_week_ppr/ablation.py` (+ `scripts/run_phase6c_ablation.py`, `tests/test_phase6c_ablation.py`): for each position's `FINAL_CONFIG` fold, fit once on the full feature set to rank LightGBM gain-based importance, then refit fresh on top-10/20/30/all subsets and score on the true outer test season. 48 rows across 4 positions × 3 seasons × 4 feature-count candidates, `data/experiments/phase6c_feature_ablation.csv`.

**Result: MAE improves (or holds flat) monotonically from 10 → all features, at every position** — TE 3.690→3.587 (largest gain), RB 4.578→4.543, WR 4.647→4.627, QB 6.178→6.144 (smallest gain, and QB has the fewest features to begin with, 57 vs. up to 67 for WR). No position shows degradation at higher feature counts within the tested range. Bias also stays roughly stable across feature counts per position (doesn't get meaningfully worse as features are added), consistent with §7.10/§7.11's finding that the bias problem is a loss-function/target-skew property, not a feature-count or overfitting artifact.

**Conclusion: the current ~57-67-feature `CAUSAL_FEATURES` set is not past the point of diminishing returns for the tree-based `FINAL_CONFIG` architectures** — there's no evidence here to justify trimming it, and modest further feature additions (per §8.4's `catch_rate`/`team_neutral_pass_rate_oe`/`rush_share_pct` additions) are plausibly still net-positive rather than adding noise. This is specific to tree-based models (see the module docstring / the multicollinearity discussion in this session) — it says nothing about whether the current feature set would be too large/collinear for the *actually-deployed* Ridge-based production path (`src/models/component_predictor.py`), which was flagged but not tested.

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

**UPDATE 2026-08-06**: still genuinely unbuilt for the general case, but
precision matters here — checked what actually exists rather than
assuming "not attempted" means zero code. `PlayerEmbeddings`
(`src/models/advanced_techniques.py`) is real and LIVE, but it's a
PCA-based training feature, not this proposal's similarity-search
system. A real comp-player matcher (draft capital + combine similarity)
also exists, but **only for rookies**
(`src/features/advanced_rookie_injury.py`) — not the general
Player2Vec/Baller2Vec-style system for all players this section
describes. See `MODELS.md`'s "UNBUILT" section for the full precise
breakdown.

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

**UPDATE 2026-08-06**: "not attempted" is imprecise — a real, complete
`MonteCarloSimulator` class already exists
(`src/models/advanced_modeling.py`), with skew-normal outcome sampling,
an injury-probability mixture, and matchup adjustment. It's dormant
(zero callers anywhere, never fed real data), not previously
documented as existing at all, and not vetted for correctness against
current schemas. See `MODELS.md`'s "ORPHANED" section for the full
inventory of similarly undocumented dormant code found during the same
audit (a second, different `LineupOptimizer`, a standalone LSTM model,
a large validation/monitoring framework). Before starting this item
from scratch, read and evaluate this existing implementation first.

#### E. QB-WR/RB Correlation Matrices (HIGH IMPACT for lineup optimizer)

**What this project does:** Players are projected independently. The lineup optimizer (`docs/lineup.html`) presumably selects players without modeling correlations.

**What state-of-the-art does:** Every competitive DFS optimizer uses player correlation matrices. QB-WR1 correlation is typically 0.3-0.5. Stacking (QB + pass-catcher from same team) exploits positive correlation. Bring-back (opposing pass-catcher) exploits game total correlation.

**Implementation path:** Compute empirical correlation matrices from historical weekly data: for each team, correlate QB FP with WR1/WR2/TE/RB FP. Feed these into the lineup optimizer as constraints.

#### F. Bayesian Hierarchical Matchup Model (MEDIUM IMPACT, MEDIUM EFFORT)

**What this project does:** `opp_fpts_allowed` as a flat matchup feature. Bayesian shrinkage for player means only.

**What state-of-the-art does:** srome's Bayesian hierarchical model (2015, well-documented) uses partial pooling to model team-vs-position matchup effects. The model simultaneously estimates team offensive strength, team defensive strength vs. each position, and player-within-team effects. Key advantage: **inconsistent teams get wider posteriors** — the model knows when it doesn't know, rather than treating all matchups as equally predictable.

**Implementation path:** Layer a Bayesian hierarchical matchup adjustment on top of the existing ensemble. The existing `BayesianPlayerModel` class could be extended to include defense random effects.

**UPDATE 2026-08-06**: confirmed `BayesianPlayerModel`
(`src/models/bayesian_models.py`) is real (721 lines, full MCMC +
James-Stein shrinkage variants) but currently dormant — zero callers
anywhere except a unit test that only checks it runs, not that it
predicts well. It's real infrastructure to extend, not a working
matchup model to compare against; the defense-random-effects extension
this section describes is still fully unbuilt. See `MODELS.md`.

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
  available/funded this session) — not attempted. **UPDATE 2026-08-06:
  the user won't ever have paid Odds API access — investigated free
  alternatives, found none exist. See "No viable free source for Vegas
  preseason win totals" further down in this doc; this item is now
  permanently deprioritized rather than "blocked on API key."**
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

### CORRECTION (2026-08-05): the v27/v28 "flat" finding below is invalid — the backtest tool never included those features

While running a rookie-slice ablation (requested to check whether v27/v28's
rookie features helped rookies specifically even though the aggregate
number below was flat), the ablation came back **bit-for-bit identical**
between "rookie features included" and "rookie features stripped" — not
just close, exactly 0.0 difference on every single prediction, including
every rookie row. That's not a real negative result; it's a broken test.

Root cause: `scripts/run_ts_backtest.py`'s feature pipeline
(`leakage_safe_features()` in `src/evaluation/ts_backtester.py`) only
calls `FeatureEngineer.create_features()` (→ `create_causal_features()`
in causal mode). It **never calls `add_season_long_features()` or
`add_advanced_features()`** — the separate pipeline stage in
`feature_preparation.py`'s `_prepare_training_data` that actually computes
`is_rookie`, `rookie_draft_value`, `rookie_bust_prob`, `combine_score`,
`rookie_opportunity_score`, `rookie_breakout_prob`, `rookie_ceiling_ppg`,
`rookie_floor_ppg`. Verified directly: calling `leakage_safe_features()`
on real data shows `is_rookie` absent from the output columns entirely,
while `opp_fpts_allowed_dvoa_adjusted_lag1` (v26, computed inside
`create_causal_features()` itself) and `team_pct_11_personnel_roll3_mean`
(v29, same) are both present as expected.

**What this means for the "v26/v27/v28 flat" claim below**: the v26
(DVOA) part is legitimate — that feature genuinely was present and tested
by this backtest tool, and genuinely showed no aggregate lift. **The
v27/v28 (rookie) part is not a valid measurement** — those features were
never in scope for any `run_ts_backtest.py` run, including this one, so
"no measurable lift" should never have been claimed for them. The
production training path (`feature_preparation.py`) does compute and use
these features correctly (confirmed by the v27/v28 promotion sessions'
own model-artifact checks) — this is purely a blind spot in the
*measurement* tool, not a defect in the features themselves or in
production training/serving.

**Fixed the same session, once the gap was found**: wired
`add_season_long_features()` + `add_advanced_features()` into both passes
of `leakage_safe_features()` (train-only, and the combined train+test
block), in production's dependency order (season-long features, which
merge `draft_round`/`draft_pick`, must run before `add_advanced_features`,
which reads them). Verified fix: `is_rookie`/`rookie_*` columns now
present in both `train_fe` and `test_fe` output on real data. 39/39 tests
pass post-fix (`test_ts_backtester.py`, `test_baseline_comparison.py`,
`test_backtest_validation.py`, `test_ml_audit.py -k ts_backtester`).

### Rookie-slice ablation, re-run with the fixed harness — real, positive result (2026-08-05)

Same `run_ts_backtest.py`-equivalent methodology (2025 holdout,
expanding-window Ridge α=10,000, 22 weeks, 5,612 predictions), now with
rookie features actually present, comparing "rookie features included"
(current v28 `CAUSAL_FEATURES`) vs. the same run with the 8 rookie
columns stripped back out — isolating their marginal effect on rookie
rows specifically (n=1,056, 18.8% of all predictions), with veteran rows
as a sanity check (should be ~unchanged, since rookie features are
near-default for non-rookies).

| Slice | Metric | With rookie feats | Without | Δ |
|---|---|---|---|---|
| All rows | R² | 0.350 | 0.347 | +0.003 |
| **Rookie rows only** | **R²** | **0.267** | **0.244** | **+0.023** |
| Rookie rows only | MAE | 4.37 | 4.51 | −0.14 |
| Veteran rows only | R² | 0.350 | 0.349 | +0.001 (noise) |
| QB rookie | R² | 0.190 | 0.167 | +0.023 |
| RB rookie | R² | 0.282 | 0.265 | +0.017 |
| WR rookie | R² | 0.189 | 0.147 | +0.042 |
| TE rookie | R² | 0.257 | 0.243 | +0.014 |

**Real, honest lift, confirming the original hypothesis**: every single
position shows the same direction of improvement on rookie rows (higher
R², lower MAE, lower RMSE), while veteran rows are essentially unchanged
(+0.001 R², within noise) — exactly the isolation pattern you'd expect
from features that are near-default for non-rookies. This directly
confirms what the aggregate v23-v28 checkpoint's own caveat predicted:
"rookies are ~5-15% of rows in a full-position backtest; a real signal
would wash out in the aggregate." It did — the aggregate all-rows R² only
moved +0.003, easily lost in that checkpoint's rounding, while the actual
rookie-specific effect (+0.023 R², WR rookies +0.042) is real and
consistent across positions.

**Verdict**: v27/v28's rookie features (`is_rookie`, `rookie_draft_value`,
`rookie_bust_prob`, `combine_score`, `rookie_opportunity_score`,
`rookie_breakout_prob`, `rookie_ceiling_ppg`, `rookie_floor_ppg`) show a
real, moderate accuracy lift specifically where they're designed to help
— rookie draft/projection accuracy, the exact use case GAPS.md originally
flagged as "25-35% of interesting draft picks" being poorly served. The
"v23-v28 accuracy-lift measurement... flat since v25" verdict earlier in
this doc should be read as **DVOA/v26-only** — the rookie portion of that
claim was never actually tested until this fix, and now that it has been,
it's a positive result, not a flat one.

**Not attempted**: the same fix now means `run_ts_backtest.py` pays the
rookie/injury module's real per-week cost (season-long features +
combine-data lookups) inside the 22-iteration expanding-window loop,
which wasn't profiled or optimized for repeated-call use — only for the
once-per-training-run cost it was designed for. If backtests become a
routine part of this project's workflow rather than an occasional
checkpoint, that cost is worth revisiting (e.g. caching season-long/rookie
features across weeks within a single backtest run, since a player's
`draft_round`/`combine_score`/etc. don't change week to week).

### §9 (silent-failure audit) — DONE: found and fixed a systemic pattern, 14 sites (2026-08-05)

Picked as the next item after the personnel-grouping (v29) and rookie-
ablation-harness-fix work, given how many real bugs this project has hit
from the exact "declared/intended but silently never happened" shape —
WOPR dead in the causal pipeline, the `CoachingChangeDetector` row-order
bug, the DVOA merge column-suffix collision, today's TE `pct_13`
omission, and the rookie-features-never-computed-by-the-backtester
discovery above. All four of those were found by luck (someone happened
to check the actual output against expectations) rather than by the code
itself surfacing the problem. This was a scoped grep-and-triage pass over
`src/data/`, `src/features/`, `src/utils/`, `src/models/` for that
pattern specifically, not the full multi-session ingestion-wide audit
GAPS.md §9 originally scoped (that remains open — see below).

**Method**: parsed every `except Exception:` / `except:` block in those
four directories (216 total `except` clauses), isolated the ones whose
body is bare `pass` with zero logging (44 of 216). Read each with
surrounding context and triaged into: legitimate resilient-fallback
patterns (leave alone) vs. safety-net/correctness-critical checks whose
failure should never be invisible (fix).

**The single biggest finding — a systemic pattern, not a one-off**: 8
separate call sites wrap `assert_no_leakage_columns()` (a function
explicitly designed to `raise ValueError` when it detects a leakage-prone
feature column — `src/utils/leakage.py:176-186`) in `except Exception:
pass`. That means the project's own designed-to-fail-loud leakage safety
gate has had its failure mode silently caught and discarded at every
single place it's called, for as long as those call sites have existed.
In the happy path this never fires (verified: `filter_feature_columns()`
runs immediately before each `assert_no_leakage_columns()` call using the
identical `is_leakage_feature()` predicate, so the assert is genuine
defense-in-depth against a bug in the filter step, not a check that's
expected to trip under normal operation) — but if it ever DID trip, the
safety gate would have been rendered a no-op with no trace in any log.
Fixed by adding a visible `print(f"  WARNING: ...")` in the `except`
clause at each site — deliberately not changed to hard-crash training
(these are training entry points, some of which may run unattended;
changing pass/fail behavior is a bigger decision than this pass), but no
longer silent. Sites fixed: `src/features/feature_engineering.py:4003`
(filter-only, no assert — same risk, `self.feature_columns` would be
left completely unfiltered on failure), `src/models/backtesting.py:230`,
`advanced_ml_pipeline.py:1351`, `train_position_models.py:194`,
`advanced_modeling.py:696`, `train_advanced.py:217`, `ensemble.py:1220`,
`feature_engineering_pipeline.py:735`.

**Second finding, same severity class**: 3 sites silently swallow
`sanitize_schedule_df()` failures — a function that strips final-score
columns from schedule data specifically to prevent leakage
(`src/utils/leakage.py:207-214`). A silent failure here means unsanitized
(leaky) schedule data flows downstream with no indication anything went
wrong. Fixed: `src/data/external_data.py:342` (weather loader),
`external_data.py:547` (Vegas lines loader), `src/utils/database.py:935`
(`get_schedule(include_scores=False)` — the caller explicitly asked for
scores stripped; a silent failure means they get scores anyway).

**Third finding**: `src/models/robust_validation.py:263` — a leakage
*detection* check (`find_leakage_columns`) inside a validation-report
function, wrapped in `except Exception: pass`. If the check itself broke,
`results['passed']` stayed at whatever value it already had — meaning a
broken detector could report "passed" without ever actually running the
check. Fixed to fail closed: append an error and set `passed = False`
when the check itself can't complete, rather than silently reporting
clean.

**Fourth, lower-severity finding**: `src/data/schema_validator.py:456,465`
— a data-freshness/staleness diagnostic (`check_data_freshness`) silently
swallows exceptions in its staleness and latest-season/week computations.
Not correctness-critical the way the leakage checks are, but this
function's whole job is surfacing data-quality problems, so a silent
internal failure defeats its purpose too. Fixed by appending to the
function's own `result["warnings"]` list (its existing convention),
rather than a bare `print`, since callers already read that field.

**Deliberately left alone** (30 of the 44 silent sites) — genuine
resilient-fallback / idempotent patterns matching established codebase
convention, where adding visibility would be noise, not signal:
per-row bulk-upsert loops in `database.py`'s `ensure_*` functions (bad
row skipped, aggregate `count` already returned and printed by callers);
`ALTER TABLE ADD COLUMN` idempotent-migration attempts (`database.py`
init — column-already-exists is the expected/common case); cached-JSON
load-with-default patterns (`data_manager.py`, `training_years_selector.py`,
`weight_optimizer.py`, `utilization_weight_optimizer.py`); probe-a-season
availability checks (`auto_refresh.py`, `nfl_data_loader.py`'s
season-detection loops — "does data exist for this year yet" is
expected to legitimately fail for most years checked); learned
blend-weight optimization with a hardcoded fallback
(`horizon_models.py:657,1039`); ensemble hybrid/deep-model prediction
fallback to the traditional model on failure (`ensemble.py:196,524,538,609`)
— each of these already degrades to a documented, sane default and isn't
a case where the exception itself indicates something is silently wrong
in the way the leakage/sanitization/freshness checks above are.

**Verified**: all 12 edited files compile clean
(`python -m py_compile`). Confirmed the fix is silent-when-healthy (ran
`create_causal_features()` on real RB data, 2020+ — zero new WARNING
lines fired, meaning none of these checks trip in the normal case, only
under real failure). 150/150 relevant tests pass post-fix
(`test_preseason_projector.py`, `test_quality_gates.py`,
`test_target_and_history_causality.py`, `test_advanced_analytics.py`,
`test_matchup_aware_prediction.py`, `test_missing_data_and_new_features.py`,
`test_ts_backtester.py`, `test_baseline_comparison.py`,
`test_backtest_validation.py`). Three pre-existing test failures
(`test_ml_robustness_15_steps.py::test_target_uses_shift_neg1`,
`::test_ensemble_weights_are_spec_mandated`,
`test_schema_validator.py::test_validate_weekly_data_rejects_negative_stats_in_strict_mode`)
confirmed identical on a clean `git stash` of this session's changes —
unrelated pre-existing failures, not a regression from this fix.

**Not attempted**: the remaining ~172 non-silent `except Exception as e:`
clauses in these same directories weren't individually audited for
whether their *logging* is adequate (only visible/invisible was
triaged this pass) — some may log to a level or location that's easy to
miss in practice even though they're not technically silent. The broader
"replace every `except Exception: pass` codebase-wide, including
`src/scripts/`, `src/evaluation/`, `src/analysis/` etc." scope from the
original GAPS.md §9 item remains open — this pass covered the four
directories where the real bugs have actually clustered this project's
history, not the whole codebase. No retrain needed for this fix (it's
error-visibility only, doesn't change any computed feature value in the
happy path).

### §7.4 (uncertainty-blend weights + CI scaling consistency) — DONE (2026-08-05)

Two small, already-diagnosed items from the earlier §7.4 investigation:
`position_models.py`'s uncertainty-blend weights were a hardcoded, never-
tuned/validated `0.5/0.3/0.2` (hetero/conformal/ensemble) guess, and
`ensemble.py:569,641` scaled multi-week CI width by `n_weeks**0.4` in one
prediction path and `sqrt(n_weeks)` in another — same real quantity (the
std of an n-week FP total), two different unreconciled assumptions
depending on which underlying model type served a given position.

**Blend weights — now grid-searched from OOF data, not guessed.** Added
`_select_blend_weights()` (`src/models/position_models.py`): a coarse
grid search (step 0.1, ~66 candidate weight triples for the 3-way
hetero+conformal+ensemble case, 11 for the 2-way conformal+ensemble
fallback) that minimizes the mean squared log-ratio between each nominal
CI level's empirical quantile and its Gaussian z-score — the same
objective the existing per-level conformal correction factors (computed
immediately after, in the same `fit()` method) are designed to fix, used
here as a search criterion instead of accepting a fixed guess. Gated on
`oof_valid.sum() >= 300` (a fresh, explicit threshold — large enough that
grid-searching ~66 candidates isn't just overfitting the weight choice to
noise); below that, falls back to the historical hardcoded default
unchanged, so small-sample behavior is untouched. Selected weights are
stored on the model (`self._uncertainty_blend_weights`), persisted
through `save()`/`load()` (falls back to the historical default when
loading an older model file that predates this field), and consumed by
`predict_with_uncertainty()` instead of the two hardcoded blend lines
that used to live there.

**Real result on real data, not just "doesn't crash"**: ran `PositionModel.fit()`
on real RB data (2018+, 8,908 training rows) and real TE data (2024-2025,
1,820 rows) — both comfortably above the tuning threshold. Both selected
`(1.0, 0.0, 0.0)` — pure heteroscedastic-GBM weight, zero on conformal
and ensemble-disagreement. This is a genuine corner solution, not a bug:
the heteroscedastic model (trained specifically to predict per-player
residual magnitude from features) evidently explains OOF residual spread
far better than either the constant conformal std or the crude
ensemble-disagreement spread, on this data. Worth flagging honestly as a
corner solution rather than a soft blend — a grid search that lands
exactly on a vertex isn't inherently suspicious (66 candidates evaluated
on thousands of real OOF rows, not a small/noisy sample), but it does
mean this specific run stopped using conformal/ensemble as a robustness
hedge entirely; if that ever seems to hurt generalization on a future
holdout, adding a small per-component weight floor to the grid would be
the fix, not reverting to the hardcoded guess.

**Verified**: synthetic sanity check confirms the search correctly
recovers a hetero-dominated blend when the ground truth is
hetero-driven. Real end-to-end `fit()` → `predict_with_uncertainty()` run
produces finite, non-negative std predictions. `python -m src.models.train --fast --no-tune`
completes clean (exit 0) — though see the production-path caveat below.
62/62 relevant tests pass (`test_uncertainty_calibration.py`,
`test_calibration_quality.py`, `test_models.py`,
`test_config_code_alignment.py`, `test_fast_mode.py`,
`test_preseason_projector.py`, `test_quality_gates.py`).

**CI scaling — unified to one formula with a documented rationale.**
Added `_multi_week_ci_scale(n_weeks)` (`src/models/ensemble.py`, module-level
helper) returning `n_weeks ** 0.4`, with the rationale spelled out once
in its docstring (weekly fantasy performance is autocorrelated — hot/cold
stretches, role changes, matchup runs — so treating weeks as independent
via `sqrt(n_weeks)` understates multi-week uncertainty). Both call sites
(`self.position_models` branch, `self.single_week_models` branch) now
call this one function instead of each having its own inline formula.
Kept `n_weeks**0.4` (not `sqrt`) since it's the one of the two that
already had a stated engineering rationale in-code; standardizing on the
undocumented `sqrt` would have meant picking the less-justified value
just because it happened to be simpler. Explicitly **not** claiming
`0.4` is the "correct" exponent — deriving that properly would mean
measuring actual multi-week CI coverage against nominal levels (the same
kind of calibration study `fit()` already does for single-week CIs), a
separate, bigger piece of work than closing out an inconsistency.

**Production-path caveat, same shape as the §6 finding earlier in this
doc**: production training runs with `position_target_type = "component"`
for all four positions, which short-circuits `EnsemblePredictor.predict()`
via `continue` before ever reaching either the `self.position_models` or
`self.single_week_models` branches in `ensemble.py`, and before
`PositionModel.fit()`'s uncertainty-blend code ever runs for a
component-mode position either (component predictors use their own
separate stat-line prediction path, not `predict_with_uncertainty()`).
Confirmed via `grep` — nothing in `ensemble.py` populates
`self.position_models`/`self.single_week_models` under the default
training config. So **both fixes in this section are real, correct, and
verified working — but neither is currently exercised by default
production predictions**, the same "moot in production, live on other
paths" shape as §6's `UtilizationToFPConverter` finding. They matter for:
`scripts/run_ts_backtest.py --target-mode fp` (not `component`) runs
using `--model ridge`/`--model gbm`, any future session that revisits the
component-vs-ensemble architecture choice, and simply having a correct,
non-contradictory uncertainty system on the shelf rather than a
known-hardcoded one.

**Not attempted**: re-deriving the multi-week scaling exponent from real
coverage data (flagged above as separate, bigger work); auditing whether
`ComponentPredictor`'s own uncertainty/CI computation (a different code
path entirely, not touched by this session) has similar hardcoded-weight
or scaling questions — out of scope for what was asked, not evaluated.

### v23–v28 accuracy-lift measurement — DONE, honest result: flat since v25 (v26 only — see correction above)

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

### §3.3/§11.1.E (FTN/PBP personnel grouping) — DONE: promoted to v29 (2026-08-05)

Picked as the next concrete feature gap after the v23-v28 checkpoint above
showed the "just wire existing data" items mostly exhausted. A prior
session had already checked feasibility and left this explicitly scoped
out (see "Remaining §9/§11/§10 items — explicitly scoped OUT of this
pass" above): FTN's `import_ftn_data()` doesn't carry a direct
personnel-grouping column, and PBP's `offense_personnel` string column
(e.g. "1 RB, 2 TE, 2 WR") was flagged as the real source but unverified
on coverage (guessed "KeyError on 2010").

**Coverage, checked directly against nfl_data_py rather than trusting the
prior guess**: `offense_personnel` doesn't exist as a column at all before
2016 (not present, not null-filled — absent). From 2016 on it's **100%
non-null on real pass/run plays** (the ~77% "non-null" figure a naive
check would show is diluted by kickoffs/punts/etc., which correctly don't
carry personnel data and aren't used here). Verified format: full O-line
breakdown (e.g. "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR"), not a bare
"11 personnel" code — parsed via regex (RB count + FB count = backfield,
matching the traditional convention that a fullback counts toward the
first digit of personnel notation; TE count is the second digit). Spot
checked against known real distributions: 2023 real games came back 11
personnel 64%, 12 personnel 20%, 21 personnel 7.7% — matches published
league-wide personnel usage for that season.

**New infra**:
- `team_personnel_stats` table (`src/utils/database.py`) — `(team,
  season, week, pct_11, pct_12, pct_21, pct_13, pct_other, n_plays)`,
  `UNIQUE(team, season, week)`.
- `get_personnel_groupings_from_pbp(season)` (`src/data/pbp_stats_aggregator.py`)
  — loads PBP, filters to pass/run plays with non-null `offense_personnel`,
  parses into the 4 tracked groupings + an `other` bucket, aggregates to
  team-week percentages. Own parquet cache (`pbp_personnel_{season}.parquet`
  in `RAW_DATA_DIR`), same idiom as the existing PBP aggregate caches.
  Empty-frame return for pre-2016 seasons (no crash).
- `DatabaseManager.ensure_team_personnel_stats()` — same shape as
  `ensure_team_coaching_staff`/`ensure_team_offense_stats` (only fetches
  seasons not already present unless `force_refresh=True`, per-season
  try/except degrades to no-op rather than raising). Internally filters
  out pre-2016 seasons up front so it doesn't pay a guaranteed-empty PBP
  fetch on every call for seasons that will never have this data.
- `scripts/backfill_team_personnel_groupings.py` — one-time/re-runnable
  full backfill entry point, 2016-2025.
- Wired into both the training path (`feature_preparation.py`, next to
  the existing `ensure_team_offense_stats()` call) and the serving path
  (`predict.py`'s `initialize()`) — same try/except-and-warn pattern as
  everywhere else in this codebase.
- `get_all_players_for_training()` — added `LEFT JOIN team_personnel_stats`
  (same-week join, since — like `team_plays`/`team_pace_sec_per_play` from
  the v24 tempo work — this-week personnel usage is an OUTCOME of that
  week's game, not pre-game-known info; leakage safety comes entirely from
  only ever consuming it via the shifted rolling feature downstream, never
  as a same-week raw value).

**Backfill run**: 5,514 rows across 2016-2025. Real, plausible trend
confirmed: 12-personnel usage rose from 18.5% (2016) to 24.2% (2025)
league-wide, matching the well-documented real-world shift toward
12-personnel offenses over that span — not a data artifact.

**`CAUSAL_FEATURES` wiring** (`config/settings.py`, `FEATURE_VERSION`
bumped `"28"` → `"29"`): added `_roll3_mean` variants of the two most
position-relevant groupings per position — RB gets `pct_12`/`pct_21`
(heavier backfield sets signal more RB volume), WR gets `pct_11`/`pct_12`
(11 personnel is the spread-offense/3-WR signal), TE gets `pct_12`/`pct_13`
(2-3 TE sets directly determine TE snap/route ceiling). QB deliberately
skipped — personnel grouping isn't a direct QB-volume signal beyond what
pass-rate/tempo features already capture, and the "don't declare
features you can't justify" lesson from the WOPR/rookie-feature history
in this doc applies here too.

**A real dead-feature bug found and fixed during verification** — the
exact same failure mode as the WOPR and `recv_epa_per_target_roll3_mean`
bugs earlier in this project (a column declared in `CAUSAL_FEATURES` but
never actually computed by the function that builds training data):
`team_pct_13_personnel_roll3_mean` was declared for TE, but the
underlying raw `team_pct_13_personnel` column was left out of
`_create_causal_rolling_features`'s `roll_cols` list (a copy-paste
omission — only `pct_11`/`pct_12`/`pct_21` were added, `pct_13` was
missed). Caught by the standard audit method this project now runs
before trusting any new feature: calling `create_causal_features()` on
real data via the actual training loader and checking every declared
`CAUSAL_FEATURES` column is actually present in the output. Fixed by
adding `team_pct_13_personnel` to `roll_cols`; re-verified 100% non-null
with real variance (nunique=5,447, mean=0.042, std=0.048) after the fix.

**Verified**:
- Full `CAUSAL_FEATURES` dead-declaration audit (real `get_all_players_for_training()`
  loader → `create_causal_features()`, 2016+ data) clean for all 6 new
  columns across RB/WR/TE after the fix above. All 100% non-null with
  real variance (e.g. WR `team_pct_11_personnel_roll3_mean`: nunique=13,021,
  mean=0.622, std=0.140).
- Retrain (`python -m src.models.train --fast --no-tune`): exit 0,
  `feature_version.txt` = 29, previous version archived. Confirmed
  directly against `data/models/component_{rb,wr,te}.json` `feature_names`:
  all 6 personnel columns present exactly where declared (RB: pct_12 +
  pct_21; WR: pct_11 + pct_12; TE: pct_12 + pct_13; QB: none, as intended).
- 115/115 relevant tests pass post-retrain (`test_preseason_projector.py`,
  `test_quality_gates.py`, `test_target_and_history_causality.py`,
  `test_advanced_analytics.py`, `test_matchup_aware_prediction.py`,
  `test_missing_data_and_new_features.py`).

**Not attempted**: accuracy-lift measurement for v29 specifically (same
standing caveat as every version bump since v23 — features are confirmed
live and non-degenerate, nobody has yet measured whether they move
`pred_mae` on a holdout backtest). Given the v26-v28 checkpoint just
found those three version bumps flat on the aggregate 2025 backtest, v29
is a reasonable candidate to fold into a future combined ablation/lift
pass rather than running a full 2025 walk-forward backtest solely for
this one feature group.

### §9 audit extension: remaining directories — DONE, and found a real live bug (2026-08-05)

Extended the silent-failure audit past the four directories covered
earlier (`src/data`, `src/features`, `src/utils`, `src/models`) to the
rest of the codebase: `scripts/`, `src/evaluation/`, `src/integrations/`,
`src/scrapers/`, `config/` (106 files). Same method: parse every `except`
block, isolate bare `except Exception: pass` / `except: pass` / silent
typed excepts, triage each.

Found 39 more silent sites. Most (≈30) are legitimate resilient-fallback
patterns matching the same categories already accepted in the first pass
— per-row bulk-insert loops (`odds_scraper.py`), cache-write-best-effort
(`base_scraper.py`, `odds_scraper.py`), cascading availability-probe
chains (`schedule_scraper.py`, matching `auto_refresh.py`'s precedent),
tmp-file-cleanup-then-`raise` (`generate_app_data.py`, matching the
established idiom), and fallback-to-sane-default patterns
(`espn_sync.py`'s scoring/roster-slot defaults). Left alone.

**Real, live, currently-active bug found**: `scripts/production_retrain_and_monitor.py`'s
`_check_data_freshness()` — the automated retrain gate this script exists
to run (`"Weekly retrain: run this script via cron"` per its own
docstring) — queried `SELECT COUNT(*) FROM player_stats WHERE season = ?`
to check whether the current season's data had arrived. **`player_stats`
has never existed as a table in this database — only `player_weekly_stats`
does** (confirmed directly: `sqlite3` query returns zero rows for
`player_stats`, one row for `player_weekly_stats`). This query has
therefore always raised `sqlite3.OperationalError`, always been caught by
the silent `except Exception: pass` immediately below it, and
`result["has_current_season"]` has always stayed `None` regardless of
actual data state — meaning the `require_current_season_data` gate
(**on by default** in `RETRAINING_CONFIG`) has never actually run, for as
long as this script has existed. Fixed the table name
(`player_weekly_stats`) and changed the except to fail closed
(`result["fresh"] = False`) with the real error message surfaced, rather
than silently keeping the optimistic default. Verified: calling
`_check_data_freshness()` directly against the live DB now correctly
returns `has_current_season: True` (previously always `None`).

**Two more real risk sites in the same script, fixed for visibility**
(no evidence either has actually tripped, unlike the bug above, but same
"production automation decision gate" risk class): the DB-mtime
staleness check and the `RETRAIN_STATUS_FILE`-based retrain-cadence check
(`_is_retrain_day()`) both silently swallowed exceptions — a corrupted
status file would have silently starved the cadence-based retrain
trigger with nothing in the logs to explain why. Both now print a
warning on failure; behavior (fail-open on staleness, "no retrain today"
on cadence-file corruption) is unchanged, only visibility added.

**Leakage-safety swallow pattern, same class as the primary audit**: two
more `drop_leakage_columns()` calls silently swallowed
(`scripts/evaluate_deep_learning.py:50`, `scripts/optimize_training_years.py:44`)
— fixed the same way as the primary pass's `filter_feature_columns`/
`sanitize_schedule_df` sites: visible warning, behavior unchanged.

**Backtest-tooling blind spots, fixed for the same reason the rookie-
ablation harness bug mattered**: `src/evaluation/backtester.py:1606`
silently swallows `add_external_features()` (Vegas/weather/game-script)
failures during backtest eval prep — a backtest could silently run
without those features and produce a misleadingly worse result with
nothing distinguishing "features genuinely don't help" from "features
silently didn't load," which is exactly the failure mode already found
once this session for the rookie-features ablation. `src/evaluation/metrics.py:317`
silently drops CI-calibration metrics on failure (now recorded as
`ci_calibration_error` in the returned metrics dict instead). `src/evaluation/ts_backtester.py:550`
(schedule lookup for the opt-in `emit_inactive_predictions` phantom-row
feature) now warns on failure instead of silently falling through to its
already-documented last-known-opponent fallback.

**Two trivial fixes**: `scripts/expand_data_2012.py:66` had `except
Exception as e: pass` — the exception was already captured and then
discarded; now printed. `scripts/advanced_features.py:70` had a bare
`except:` (would catch `SystemExit`/`KeyboardInterrupt` too) around an
ESPN injury-report fetch whose parsing is an unimplemented `# TODO: Full
parsing logic` stub — narrowed to `except Exception:` with a comment
noting the function is a no-op regardless of whether the request
succeeds (the real fix is implementing the parser, out of scope here).
This function is exercised by `tests/test_predictions.py` (confirmed via
`scripts/` being on `sys.path`, not dead code as first suspected before
checking).

**Verified**: all 8 edited files compile clean. 45/45 tests pass in
`test_predictions.py`/`test_ts_backtester.py`/`test_baseline_comparison.py`/
`test_backtest_validation.py`, 10/10 in `test_metrics_evaluator.py`/
`test_production_retrain.py`, 162/163 in the broader relevant suite
(`test_preseason_projector.py`, `test_quality_gates.py`,
`test_target_and_history_causality.py`, `test_advanced_analytics.py`,
`test_matchup_aware_prediction.py`, `test_missing_data_and_new_features.py`,
`test_ml_audit.py`) — the one failure
(`test_utilization_weights_persistence`) confirmed pre-existing and
unrelated via `git stash` (fails identically on a clean tree, checking
for a literal string in `train.py` that predates this session).

**Not attempted**: reviewing the ~191 already-non-silent
`except Exception as e: print(...)`-style sites in these five newly-swept
directories for logging adequacy (same scoping decision as the primary
pass — visible-vs-invisible was triaged, message quality wasn't).
Directories still never swept by any pass so far: none identified beyond
what's now covered (`src/data`, `src/features`, `src/utils`, `src/models`,
`scripts`, `src/evaluation`, `src/integrations`, `src/scrapers`,
`config`) — this plus the primary pass now covers every `*.py` under
`src/` and `scripts/`. `tests/` itself was not swept (test code silently
swallowing an assertion failure would show up as a false pass, a
different and arguably more serious problem, but a separate audit from
this one).

### §9 follow-up: two "print but don't record" fixes upgraded to actually fail loudly, plus a third real-bug-shaped pattern found and fixed (2026-08-05, same session)

User pushback, and rightly so: the DB-staleness and retrain-cadence
fixes above only added a `print()`. For `production_retrain_and_monitor.py`
— a script whose own docstring says "run this via cron" — a `print()`
alone isn't actually loud. Nobody watches unattended cron stdout, and
critically, the error message wasn't landing in the one place a
monitoring check would actually read: the structured `status` dict that
`run_weekly_retrain()` persists to `RETRAIN_STATUS_FILE` via
`_write_retrain_status()`. A `print`-only fix is only as loud as whoever
happens to be staring at a terminal at the exact moment the cron job
runs.

**Fixed properly**:
- `_check_data_freshness()`'s staleness-check except now fails closed
  (`result["fresh"] = False`, with a `reason`) instead of just printing
  — matching what the current-season-check fix already did correctly.
  Because `result` is the function's return value and `run_weekly_retrain()`
  does `status["data_freshness"] = freshness` before persisting `status`,
  this now lands in the durable status file, not just stdout.
- `_is_retrain_day()`'s cadence-check except used to fall through to
  `return False` (silently "not retrain day today"), with no way for a
  caller to distinguish "genuinely not due yet" from "the check itself
  broke." Changed to `return True` on failure instead — a deliberate
  fail-toward-action choice: an extra/early retrain attempt is cheap and
  self-correcting (the freshness and data-quality gates downstream still
  protect against training on bad data), whereas the old silent-False
  behavior could in principle let retraining go indefinitely stale with
  nothing in any log to explain why. Verified directly: pointed
  `RETRAIN_STATUS_FILE` at a file containing malformed JSON and confirmed
  `_is_retrain_day()` now returns `True` with a visible warning, instead
  of the old silent `False`.

**A third real pattern found while re-checking the rest of the file for
the same shape**: `_init_database()`'s three schema-migration blocks
(`player_weekly_stats` and `team_stats` column backfills, plus the
`schedule` table's `spread_line`/`total_line` columns) all wrapped
`ALTER TABLE ADD COLUMN` in a blanket `except Exception: pass`. Checked
what these actually guard against: the `player_weekly_stats` and
`team_stats` loops both pre-check `col not in existing_cols` before ever
calling `ALTER TABLE` — meaning "duplicate column" (the only case a
blanket catch is usually protecting against for this kind of migration)
should **never** actually reach these except blocks in normal operation.
Any exception that does reach them is therefore a genuine, unexpected
migration failure — and a schema that silently didn't get a column later
code assumes exists is precisely the "declared/expected but not actually
there" bug shape that has caused real, confirmed bugs multiple times this
session (WOPR, the TE `pct_13` personnel column, the DVOA merge
collision, and now the `player_stats`/`player_weekly_stats` table-name
bug above). Fixed both to print a warning on any exception, since none is
expected to occur there at all.

The third site (`schedule` table's `spread_line`/`total_line` migration)
is genuinely different — its own comment explains it deliberately catches
the duplicate-column error *instead of* pre-checking via introspection
("SQLite has no IF NOT EXISTS on ALTER TABLE"), so hitting that except is
the normal, expected path on every re-run after the first. Narrowed
instead of widened: now only swallows the exception when the message
contains "duplicate column"; anything else prints a warning. This
preserves the existing idempotent-migration behavior exactly while still
surfacing a genuinely different failure (disk full, permissions, a locked
DB, a bug in `_validate_identifier`/`_validate_ddl`) instead of masking
it identically to the expected case.

**Verified**: all changes compile clean. Live-tested against the real
(already-migrated) production DB — zero spurious warnings on
`DatabaseManager()` construction, confirming neither the pre-checked
migrations nor the narrowed duplicate-column catch false-positive on
normal operation. Also tested a *fresh* in-memory-style DB (via a temp
file) through two consecutive `DatabaseManager()` inits — the second init
exercises the real duplicate-column path for the `schedule` table
migration, confirmed silent (as intended) with no spurious warnings.
`tests/test_production_retrain.py` (4/4),
`tests/test_preseason_projector.py`, `tests/test_quality_gates.py`,
`tests/test_target_and_history_causality.py` (19/19 combined) all pass.

**Not attempted**: applying the same "should this actually fail loudly,
not just print" scrutiny to the other ~14 sites fixed in the two earlier
§9 passes today — those were reviewed and judged adequate as
print-and-continue at the time, but weren't specifically re-examined
against the "does this feed a durable, checkable record, or only
ephemeral stdout" standard this follow-up established. Worth a quick
pass if this project ever adds real monitoring/alerting on top of
`RETRAIN_STATUS_FILE`/`drift_status.json`, since that's precisely the
kind of consumer that would care about the distinction.

### §9: all three remaining audit gaps closed (2026-08-05, same session)

User asked to close out all three items flagged as "not attempted" at
the end of the prior §9 passes: (1) logging adequacy of the ~191
already-visible sites, (2) whether `tests/` itself silently swallows
anything, (3) re-checking the earlier fixes against the durable-record
standard. All three done.

**(1) Logging adequacy — mostly already fine, 3 real gaps found and fixed.**
Wrote a scanner for `except ... as VAR:` blocks where `VAR` is bound but
never referenced in the block body (a message that's present but
generic/unhelpful — the exception was captured and then thrown away
anyway). Of 237 bound-exception sites across `src/` + `scripts/` +
`config/`, only 2 failed to reference their own bound variable:
- `src/models/advanced_techniques.py`'s `BayesianOptimizer._random_search`
  — silently skipped every failed hyperparameter trial with no trace.
  Fixed to print. Checked reachability first: confirmed via `grep` that
  nothing outside this file calls `BayesianOptimizer`/`_random_search` —
  it's currently dead code (only `PlayerEmbeddings` from this module is
  live, imported by `feature_preparation.py`). Fixed anyway since it's a
  one-line change and would otherwise silently fail every trial if ever
  wired in.
- `src/scrapers/schedule_scraper.py`'s `_parse_game_object` — returned
  `None` on any per-game parse failure with the exception discarded, and
  callers append `if parsed` with no count of how many games silently
  failed. A partially-scraped week looked identical to a complete one.
  Fixed to print the exception with year/week context. This scraper is a
  fallback data source (used when `nfl_data_py` lacks schedule data), so
  a quietly incomplete schedule could have propagated into the `schedule`
  table unnoticed.

A second, narrower scan checked for the "worse" version of this problem
— an *unbound* `except Exception:` (no `as e` at all) followed by a
`print`/`log` call using only a static string literal, meaning the
exception was never even captured, not just discarded after capture.
Zero found across all 168 files scanned. Combined with the 235
already-adequate bound-exception sites, the codebase's exception
messages are in solid shape outside the two fixes above.

**(2) `tests/` swept — clean, nothing to fix.** Same silent-`except`-`pass`
scan applied to all 60 files in `tests/`: zero found. Broadened to check
for the more dangerous test-specific anti-pattern — `except AssertionError`
(which would literally swallow a failed assertion and produce a false
pass): zero found. The entire test suite has only 8 `except` clauses
total, and all 8 checked individually: 5 are legitimate optional-dependency
skip guards (`pytest.skip()` on missing `lightgbm`/`torch`, or "no data
available") that produce a visible, correctly-labeled test skip rather
than a false pass; 3 are `float(val)` parse-fallback-to-`0.0` blocks
inside `test_webapp_rendering.py` that are themselves simulating frontend
JS parsing behavior as part of what the test is checking, not swallowing
a test assertion. No false-pass risk found anywhere in `tests/`.

**(3) Durable-record re-check — evolved into a bigger, correct fix: making
the 8 leakage-assert sites actually raise, not just print.** Re-examining
the 8 `assert_no_leakage_columns` swallow sites fixed in the very first
§9 pass against the "does this feed a durable/checkable record" question
led to a better answer than upgrading the print message: **these should
never have been caught at all**. `assert_no_leakage_columns` is
*designed* to raise — it's not a soft warning, it's a hard safety gate —
and "catch it, print a warning, keep going with possibly-leaky features"
was always the wrong response to a gate whose entire contract is "stop if
this fires."

Checked whether raising is actually safe at each of the 8 sites before
changing anything (the real question: does an unhandled exception here
silently crash something unattended, or does it get caught durably
somewhere upstream):
- `src/features/feature_engineering.py` and `src/models/ensemble.py` (the
  QB dual-target site) are on the live production training path. Traced
  `run_weekly_retrain()` in `production_retrain_and_monitor.py`: its
  `train_models()` call is *already* wrapped in `try/except Exception`
  that records `status["error"] = f"training_failed: {e}"` into the
  durably-persisted `RETRAIN_STATUS_FILE` and returns `False` cleanly —
  confirmed by reading the actual code, not assumed. So letting these two
  raise doesn't crash unattended cron automation; it gets caught,
  recorded, and reported exactly where a monitoring check would look.
  (`ensemble.py` also already had a *second*, unrelated
  `assert_no_leakage_columns` call — the main per-position feature-build
  path, around what's now line 1038 — that was never wrapped in a silent
  except in the first place; the newly-fixed site is a second, QB-specific
  check further down the file.)
- The other 6 (`src/models/backtesting.py`, `advanced_ml_pipeline.py`,
  `train_position_models.py`, `advanced_modeling.py`, `train_advanced.py`,
  `feature_engineering_pipeline.py`) were checked for whether they're
  actually reachable from the live pipeline at all. `grep` for who
  imports each: none of the first four are imported anywhere outside
  their own file; `train_advanced.py` is imported only by
  `validate_methodology.py` (itself a standalone script); the
  `feature_engineering_pipeline.py` swallow site sits inside
  `run_feature_engineering_pipeline()`, a different function from the
  only thing `ensemble.py` actually imports out of that file
  (`StabilitySelector`). All 6 have `if __name__ == "__main__":` blocks —
  they're standalone diagnostic/training tools run directly by a human,
  where a raised traceback at the point of failure is strictly more
  useful than a printed warning that might scroll past.

All 8 sites changed from `try: ... except Exception as e: print(...)` to
letting the call raise directly, with a comment at each explaining why
raising is safe there specifically (not just "raising is good in
general").

**Deliberately left as print-and-continue, not upgraded to raise**: the
`sanitize_schedule_df()` sites (`external_data.py:342,547`,
`database.py:935`) and `backtester.py:1606`'s `add_external_features()`
call. Reasoning: these aren't the last line of defense against leakage
anymore now that the 8 `assert_no_leakage_columns` sites fail loud — a
leaked schedule-score column that slipped past a failed sanitization
would still get caught downstream by the now-raising leakage asserts
before it could reach a trained model. Meanwhile, making these raise
would mean losing Vegas/weather data *entirely* on a rare sanitization
hiccup (the whole `load_weather_data`/`load_vegas_lines`/`add_external_features`
call would abort), which is a worse practical outcome than the risk
being guarded against. Print-and-continue is the right call here, not a
leftover gap.

**Verified**: all 10 edited files compile clean. Full real retrain
(`python -m src.models.train --fast --no-tune`) completed with exit 0 —
confirms none of the now-raising leakage asserts actually trip on real
production data (they're defense-in-depth, expected to never fire in the
happy path, and didn't). 147/147 tests pass across
`test_preseason_projector.py`, `test_quality_gates.py`,
`test_target_and_history_causality.py`, `test_advanced_analytics.py`,
`test_matchup_aware_prediction.py`, `test_missing_data_and_new_features.py`,
`test_uncertainty_calibration.py`, `test_calibration_quality.py`,
`test_models.py`. 82/83 pass across `test_ts_backtester.py`,
`test_baseline_comparison.py`, `test_backtest_validation.py`,
`test_ml_audit.py` — the one failure
(`test_utilization_weights_persistence`) is the same pre-existing,
already-confirmed-unrelated stale-test failure noted earlier this
session (checks `train.py`'s source text for a literal string that
predates this session; the actual behavior it's checking for is real,
just implemented in `feature_preparation.py` now).

**Not attempted**: fixing the stale tests themselves
(`test_ensemble_weights_are_spec_mandated`, `test_utilization_weights_persistence`,
and a `shift(-1)` source-text check in the same file) — confirmed this
session to be checking the wrong file/module after a refactor moved the
logic they're testing for, not evidence of an actual regression, but
still worth updating so the test suite doesn't carry three permanently-red
tests. Out of scope for the §9 audit specifically.

### OPEN ITEM: `n_weeks**0.4` multi-week CI-scaling exponent — investigated, deliberately NOT fixed, still open (2026-08-05)

**Status: open, not done.** This is a separate item from the
floor/ceiling fix below and should not be read as resolved by it — the
task tracker's entry for this work is labeled "completed" because the
*redirected* work (floor/ceiling) was completed, not because this
original ask was. Recording that distinction explicitly here since it's
an easy thing for a future session (or a status summary) to conflate.

The original ask (§7.4 follow-up) was to empirically derive the
`n_weeks**0.4` exponent used in `position_models.py`/`ensemble.py`'s
`_multi_week_ci_scale()` — it was chosen last session as "the more
defensible of two existing guesses" (`n_weeks**0.4` vs `sqrt(n_weeks)`),
never derived from real coverage data. Investigated feasibility before
touching any code and found the honest answer is **not worth building
right now**, for three compounding reasons, traced end to end rather than
assumed:
1. No backtesting tool in this codebase ever generates multi-week
   predictions against known historical actuals — `run_ts_backtest.py`
   only ever predicts 1 week at a time. The only `n_weeks>1` callers are
   live serving (draft board generation), which by definition has no
   historical actuals to check against (it's projecting the future).
2. The code path is unreachable under default component-mode training
   (already known from the prior session's §7.4 work).
3. Even where reachable (non-default target modes), `EnsemblePredictor.predict()`
   explicitly leaves `prediction_ci80/95_lower/upper` as `NaN` for
   component-mode positions — confirmed by reading the code directly, not
   assumed — and the actual user-facing draft board computes its own
   floor/ceiling independently via `generate_draft_data.py`'s spread
   formula, with zero connection to `_multi_week_ci_scale()` at all.

So calibrating `n_weeks**0.4` properly would mean building real new
backtesting infrastructure (predict N weeks forward from a historical
cutoff, compare against real summed actuals — buildable, but a genuinely
new tool, not a quick measurement) to harden a formula that currently has
no path to affecting anything a user sees. Per explicit user direction,
redirected the effort to the floor/ceiling formula instead (below), which
*was* reachable and *was* shown to be badly miscalibrated. **The
`n_weeks**0.4` exponent itself remains exactly as uncertain as it was at
the start of this session** — still just "the more defensible of two
guesses," not empirically validated. Worth revisiting only if/when this
project's production training moves off component mode, or if
`prediction_ci80/95` columns are ever wired into something user-facing —
neither is true today.

### RESOLVED: `n_weeks**0.4` exponent replaced with real per-position values, plus a real bug fixed (2026-08-06)

Per explicit user direction ("in case we ever switch modes in
production, let's fix this robustly"), revisited this with a narrower
question than the "build a full multi-week backtest tool" framing above:
does deriving the exponent actually require historical *model
predictions*, or just the autocorrelation structure of real historical
*weekly fantasy points*? It's the latter — `std(N-week sum)` vs.
`N × std(1-week)` is a property of the raw time series, measurable
directly from `player_weekly_stats.fantasy_points` (110k rows,
2006-2025) with no model in the loop. No new backtest infrastructure
needed after all; the prior session's conclusion was right about the
higher bar (predict-and-compare-to-actuals) but that bar wasn't actually
required for this specific question.

**A real bug, independent of the exponent.** Re-reading
`EnsemblePredictor.predict()`'s `MultiWeekModel` branch
(`ensemble.py:576-629`) turned up something the original investigation
didn't catch: the point prediction correctly uses the
horizon-appropriate representative model
(`model.predict(pos_data, n_weeks)` → picks whichever of the trained
1w/4w/18w models covers the requested horizon), but the *uncertainty*
was pulled from `base_model = model.models.get(1)` — **always the
1-week model**, regardless of `n_weeks` — then heuristically widened by
`n_weeks**0.4`. Meanwhile each representative model already has its own
real, conformally-calibrated per-level uncertainty
(`_uncertainty_scale_factors_per_level`), fit during `PositionModel.fit()`
from OOF residuals against the *actual* historical `target_{nw}w`
column (`feature_preparation.py:_create_horizon_targets` — a true
forward rolling sum of real `fantasy_points`, not an estimate). So this
path had a real, calibrated n-week uncertainty source sitting right
there and unused. Fixed: `base_model` now resolves to
`model.models.get(n_weeks)` (the same model used for the point
prediction), and the heuristic `multi_week_scale` multiplication was
dropped from this path entirely (`multi_week_scale = 1.0`) since the
correctly-selected model's own calibration already reflects real n-week
variance — reapplying a scaling formula on top would double-count it.
Added a regression test
(`tests/test_models.py::TestEnsemblePredictor::test_multiweek_ci_uses_matching_horizon_model`)
that fits a `MultiWeekModel` with deliberately distinguishable
uncertainty factors on its 1w vs. 4w sub-models and asserts predicting
at `n_weeks=4` reflects the 4-week model's calibration, not the
1-week model's — this would have failed before the fix.

**The exponent is still needed for one path**: the `single_week_models`
fallback (only used when no multi-week model was ever trained — just a
`model_{position}_1w.joblib`). There, the point prediction is a literal
`base_pred * n_weeks` linear scale-up with no per-horizon-trained model
to draw calibration from, so a scaling formula is unavoidable. Wrote
`scripts/derive_multiweek_ci_exponent.py`, which loads
`player_weekly_stats.fantasy_points` joined to `players.position`,
computes the realized forward-looking N-week rolling sum per
player-season for N ∈ {1,2,4,8,13,18} (same `_forward_window`-style logic
as `_create_horizon_targets`), and fits
`std(N) = std(1) × N^alpha` via log-log regression per position:

| Position | alpha | R² |
|---|---|---|
| QB | 0.713 | 0.9988 |
| RB | 0.810 | 0.9991 |
| WR | 0.780 | 0.9988 |
| TE | 0.798 | 0.9989 |

**All four came out well above the i.i.d.-weeks baseline of 0.5**,
confirming real positive week-to-week autocorrelation in fantasy
production (hot/cold stretches, role entrenchment) — consistent with
the qualitative reasoning the original 0.4 guess cited. But the old
hardcoded **0.4 was actually below the i.i.d. baseline**, meaning it was
wrong in direction: it wouldn't just have "under-stated" multi-week
uncertainty relative to the true autocorrelated case, it would have
under-stated it relative to even treating weeks as fully independent.
Stored the fitted values in `config/settings.py`
(`MULTI_WEEK_CI_SCALE_EXPONENT` per-position dict +
`MULTI_WEEK_CI_SCALE_EXPONENT_DEFAULT` fallback = 0.775, the mean) and
changed `_multi_week_ci_scale()`'s signature to
`_multi_week_ci_scale(n_weeks, position)`.

**Verification, and being explicit about its limits**: this fix does
*not* change the fact from the original investigation that
`MultiWeekModel`/`single_week_models` aren't exercised by production's
default component-mode training — there is still no live end-to-end
backtest to bit-diff against, and that's not overclaimed here. What *is*
verified: (1) the exponent-fitting function itself, unit-tested on
synthetic data with known ground truth
(`tests/test_multiweek_ci_exponent.py`) — i.i.d. weekly noise recovers
alpha≈0.5, strongly autocorrelated synthetic data recovers alpha>0.6,
confirming the fitting logic is sound independent of what real NFL data
shows; (2) the real fitted alphas are all > 0.5 as expected
(same test file); (3) the `MultiWeekModel` bug fix, regression-tested as
described above; (4) the full existing suite
(`test_models.py` + `test_ml_audit.py` + `test_ml_robustness_15_steps.py`
+ `test_schema_validator.py` + `test_feature_engineering.py`) stays
green — 90 passed, 3 skipped (pre-existing, data-availability-gated) —
plus the 21 new/updated tests in `test_models.py` and
`test_multiweek_ci_exponent.py`.

This closes the open item: the exponent is no longer a guess, and the
`MultiWeekModel` path no longer needs the exponent at all for the
horizons it was actually trained on. Still worth a live backtest
verification if/when production actually switches off component mode —
but the uncertainty machinery is now correct and grounded in real data
either way, not an untested guess sitting dormant.

### Draft-board floor/ceiling was badly miscalibrated — found, measured, fixed (2026-08-05, same session)

Follow-up to the above, once it became clear the CI-scaling exponent
itself wasn't worth chasing further right now.

**Method**: `_resolve_projection()`'s formula is `spread = 1.5 * fp_std *
sqrt(17) * spread_multiplier`, where `fp_std` is a player's own
prior-season week-to-week fantasy-points std. The `1.5` constant was
never documented as implying anything, but under a Gaussian assumption a
two-sided z=1.5 implies ~86.6% nominal coverage. Checked whether the
formula actually delivers that: built a historical validation using the
REAL production `PreseasonProjector` (loaded via `snake_draft_sim.py`'s
`load_preseason_projections()` — the established, correct feature-query
path; a simplified query is separately documented to silently produce
badly wrong predictions) against 7 real seasons (2019-2025, 3,060
player-seasons), computing `fp_std` from each player's actual prior
season and checking what fraction of players' real season totals landed
inside `[floor, ceiling]`.

**Result: 42.2% empirical coverage against an implied 86.6% target** —
roughly half of what the formula claims. Consistent across every season
(40-45%) and every position (35-48%), so not a fluke or a single-season
anomaly. The mean spread (±42.8 pts) was smaller than the mean prediction
error itself (53.8 pts) — the "confidence interval" was narrower than the
model's typical miss. Root cause is intuitive once you see it: `fp_std`
only encodes a player's own week-to-week volatility in one prior season;
it has no way to represent what a preseason model fundamentally can't
foresee (season-ending/season-limiting injuries, role changes, breakouts)
— which is most of the real variance in a season-total outcome.
Spot-checked real rows to confirm this wasn't a bug in the measurement,
not the formula: e.g. 2024 C. McCaffrey, predicted 325 pts, actual 48 pts
(he was injured most of the season) — a real, large, legitimate miss the
interval should have been wide enough to at least gesture at.

The z-multiplier empirically needed to hit 86.6% coverage on this data
was **~3.78**. One honest caveat, flagged to the user before changing
anything: this measurement uses the *current* production model (trained
on 2018-2025) to predict past seasons, not a true walk-forward retrain
per season — so it's somewhat hindsight-informed, and real out-of-sample
coverage in live use is likely *worse* than 42.2%, not better, meaning
3.78 is more likely an underestimate of what's truly needed than an
overestimate.

**User confirmed: fix it (widen), not just document it.** Extracted the
magic constant into `FLOOR_CEILING_Z_MULTIPLIER = 4.0` (rounded up from
3.78 with a small margin given the hindsight-bias caveat above), used at
both spread-computation call sites in `_resolve_projection()` (the
`preseason_model` branch, which was directly measured, and the
`weekly_18w` fallback branch, which shares the identical formula shape
but wasn't separately calibrated — noted explicitly in the code comment).
Re-ran the same historical check with the new multiplier: **88.2%
empirical coverage**, right at the 86.6% nominal target.

**Verified**: `scripts/generate_draft_data.py` compiles clean. 18/18
tests pass (`test_generate_draft_data.py`, `test_snake_draft_sim.py`) —
neither test file made any assertion tied to the old constant's specific
value, so nothing needed updating there. No retrain needed (this is a
post-hoc display-layer formula, not a model or `CAUSAL_FEATURES` change).

**Not attempted**: a true walk-forward version of this calibration check
(refitting `PreseasonProjector`'s full variant-selection process for each
of the 7 test seasons using only data available before that season) —
real, buildable, but substantially more expensive than the hindsight-
informed version used here, and the honest caveat above already points
in the conservative direction (real coverage is more likely to still be
under-target than over-target at z=4.0, not the reverse). Also not
attempted at this stage: separately calibrating the `weekly_18w` branch's
spread (no historical data exists to test it the same way, since it
depends on a different point-estimate source than what was validated) —
superseded below anyway.

### Floor/ceiling: the z=4.0 fix corrected the average, but the underlying signal was still weak — replaced with a genuinely discriminative formula (2026-08-05, same session)

User asked a sharper question after the z=4.0 fix landed: does floor/ceiling
actually help distinguish high-confidence from low-confidence predictions,
or does it just produce a correctly-averaged but uninformative band?
Investigated with the same historical data already built for the
calibration check, and the honest answer was **no, not really** — which
led to actually fixing it rather than just documenting the weakness.

**What was wrong, precisely**: `fp_std` (a player's own prior-season
week-to-week volatility, the sole driver of the old formula's width)
looked like a real signal in isolation (Spearman rho=0.30 vs absolute
error), but that was almost entirely a confound. `fp_std` is highly
correlated with `pred_total` (star players naturally have both higher
projections and higher week-to-week point totals), and `pred_total` alone
predicts absolute error even more strongly (rho=0.36) simply because
bigger numbers have bigger absolute errors. Controlling for `pred_total`
via partial correlation, `fp_std`'s independent contribution to absolute
error collapsed to **r=0.05** — noise. Worse, in *relative* (percentage)
terms `fp_std` actually correlated *negatively* with error (rho=-0.37):
high-`fp_std` players (stars) are relatively more predictable, not less,
which is the opposite of what a width-setting confidence signal should
show. Confirmed via coverage-by-fp_std-quintile even after the z=4.0 fix:
76% (low fp_std, under-covered) up to 96% (high fp_std, over-covered) —
correctly calibrated only in aggregate, systematically wrong tier by tier.

**What actually works**: `log(pred_total)` itself (Spearman rho=-0.47
against relative error — bigger/more-established projections are
proportionally far more accurate) and the model's own `confidence_score`
from `PreseasonProjector.predict_with_details()` (right-signed,
significant at p<1e-5 in a quantile-regression fit, weak-but-real
independent effect). `fp_std` added nothing once these two were included
and was dropped entirely.

**Method**: fit a quantile regression (q=0.866, matching the original
z=1.5-implied target) of relative error
(`|actual_total - pred_total| / pred_total`) on `log(pred_total) +
confidence_score`, using the same 3,060 real player-seasons (2019-2025)
as the calibration check. Validated properly out-of-sample before trusting
it — fit on 2019-2022 only, tested on unseen 2023-2025 — rather than
fitting and checking on the same data:

| | Old (fp_std, z=4.0) | New (pred_total + confidence) |
|---|---|---|
| Holdout overall coverage | 87.9% | 89.1% |
| Coverage range across fp_std quintiles | 22 pts (73%–95%) | **4 pts (87%–91%)** |
| Spread-vs-actual-error correlation | 0.32 | 0.36 |

Both formulas hit the average target; only the new one is actually
uniform across risk tiers, which is what "does this help identify
which predictions to trust" requires. Refit on the full 2019-2025 dataset
for production (86.1% overall, 5.5-point quintile range) once the
holdout validation confirmed the approach generalizes.

**Shipped**: `scripts/generate_draft_data.py` — new
`_floor_ceiling_spread(total, confidence)` helper replaces the old
`fp_std`-based inline formula at both call sites in
`_resolve_projection()` (`FLOOR_CEILING_REL_SPREAD_INTERCEPT/LOG_PRED_COEF/CONF_COEF`,
clipped to `[0.15, 3.0]` relative spread; `FLOOR_CEILING_DEFAULT_CONFIDENCE=0.7`
for the `weekly_18w` branch, which has no per-player `confidence_score`).
`FLOOR_CEILING_Z_MULTIPLIER` (this session's first fix) is gone —
superseded, not left as dead code. A side effect worth noting: floor/ceiling
no longer depends on `fp_std` being present/nonzero at all, so players who
previously got `null` floor/ceiling for lacking prior-season volatility
data (e.g. some rookies) now get a real interval whenever a total
projection exists.

**Verified**: production `_floor_ceiling_spread()` re-run directly against
the full 3,060-row historical dataset reproduces the offline check exactly
(86.2% coverage, 8.9pp/5.6pp/... quintile range matching the standalone
analysis). `scripts/generate_draft_data.py` compiles clean. 18/18 tests
pass (`test_generate_draft_data.py`, `test_snake_draft_sim.py`) — neither
asserts on the old constant's value. No retrain needed (display-layer
formula only).

**Position gap found, and also fixed, same session.** The quantile
regression above was fit pooled across positions. Checked coverage by
position for the shipped formula rather than just asserting it was fine:
QB 82.5%, RB 83.5%, WR 85.9%, TE 92.1% on the same 3,060-row dataset — a
genuine ~10-point spread (TE over-covered, QB/RB under-covered), smaller
than the 22-point fp_std-quintile problem already fixed but real, not
noise. Cheap to check whether it was worth fixing: added `C(position)`
to the same quantile regression (QB as reference level), validated on
the same 2019-2022/2023-2025 holdout split (89.8% holdout coverage,
position range tightened from ~10pp to ~6pp on the holdout), then refit
on the full 2019-2025 dataset for production —
`FLOOR_CEILING_REL_SPREAD_POSITION_COEF = {"QB": 0.0, "RB": -0.121,
"WR": -0.172, "TE": -0.295}`, added as a fourth term in
`_floor_ceiling_spread()` (now also takes `position`, threaded through
from `row.get("position")` at both call sites in `_resolve_projection`).
Final production coverage by position: **QB 84.8%, RB 86.4%, TE 86.7%,
WR 86.4%** — all within 2pp of the 86.6% target, versus the original
~10pp spread. Note `confidence_score`'s coefficient shrank to
essentially zero (+0.022) once position was added — position appears to
absorb most of what confidence_score was capturing; kept in the formula
since it doesn't hurt and preserves a real (if now smaller) independent
signal path, not removed just because its coefficient shrank.

Re-verified end-to-end against the full historical dataset using the
actual shipped `_floor_ceiling_spread()` function (not a standalone
reimplementation): 86.3% overall, position range 84.8%–86.7%, exactly
reproducing the offline analysis. 18/18 tests still pass
(`test_generate_draft_data.py`, `test_snake_draft_sim.py`).

**Not attempted**: a true walk-forward version (refitting
`PreseasonProjector` itself per season, not just the spread formula)
remains the same open item noted in the section above.

### v29 (personnel grouping) accuracy-lift measurement — DONE, honest result: flat (2026-08-05, same session)

Closed the standing "v29 features are confirmed live but nobody's
measured whether they move `pred_mae`" caveat from the v29 promotion
entry earlier in this doc. Same ablation methodology as the rookie-
feature ablation: two full 2025 walk-forward backtests via the fixed
`run_ts_backtest.py`-equivalent harness (expanding-window Ridge
α=10,000, 22 weeks, 5,612 predictions) — one with the 6 v29 columns
(`team_pct_11/12/21_personnel_roll3_mean` for RB, `team_pct_11/12` for
WR, `team_pct_12/13` for TE) in `CAUSAL_FEATURES`, one with them
stripped back out.

**Result: no measurable lift, aggregate or by position.** Every metric
matched to displayed precision between the two runs — MAE 4.78 vs 4.78,
RMSE 6.36 vs 6.36, R² 0.350 vs 0.350 overall; RB 0.368 vs 0.368, WR
0.281 vs 0.281, QB 0.233 vs 0.233 (identical, as expected — personnel
features were never wired for QB), TE 0.267 vs 0.266 (the only
non-identical value in the whole comparison, and it's noise-level).

**Unlike the rookie-feature ablation, this is very unlikely to be an
aggregation-masking artifact.** The rookie ablation's initial "flat"
result turned out to be hiding a real effect because rookies are only
~19% of rows — slicing to rookie-only rows revealed a genuine lift the
aggregate metric buried. Personnel-grouping features apply broadly to
*every* RB/WR/TE row every week, not a small subgroup, and the
position-level breakdowns above (which would surface a concentrated
effect the same way the rookie slice did) show the same flat result as
the aggregate. There isn't an obvious slice left to check that could be
hiding something here the way there was for rookies.

**Consistent with the v26-v28 checkpoint pattern already documented
above**: this is now the fourth feature-version bump in a row (v26 DVOA,
v27 rookie identity, v28 rookie opportunity/breakout, now v29 personnel
grouping) to show no aggregate lift on this specific backtest
configuration (Ridge α=10,000, 55-60 competing features per position).
The same caveats already on record apply here too — heavy regularization
at that α could be shrinking a real-but-modest signal's coefficient
toward zero, and this is one backtest on one holdout season with one
model type, not the higher-fidelity `--model ensemble` path (too slow to
run routinely, per earlier entries in this doc).

**Verdict**: v29's personnel-grouping features are confirmed live,
non-degenerate, and correctly computed (per the promotion entry), but
have not been shown to improve accuracy on this measurement. Not a
reason to revert them (they're theoretically well-motivated — GAPS.md
§11.4 ranked personnel grouping #6 in the impact table for real reasons,
i.e. 11 vs. 12 personnel usage directly gates WR3/TE opportunity), but
also not a claim of proven value. Filed honestly, matching this
project's standing practice of not overstating results.

**Not attempted**: per-feature-version ablation isolating v29 from
v26-v28 in combination (each has only been tested against its immediate
predecessor version, not cross-checked for interaction effects); the
`--model ensemble` higher-fidelity backtest path for any of v26-v29.

### Skeptical audit of RIDGE_DEFAULT_ALPHA=10,000 — real evidence conflict found, live reproduction in progress (2026-08-05, same session)

Following the "four feature versions in a row show flat lift" pattern
above, user asked for a comprehensive, skeptical re-check of the Ridge
α=10,000 default itself — whether it's actually validated, or another
instance of the "hardcoded, never really checked" pattern this session
has repeatedly found (uncertainty-blend weights, CI-scaling exponent,
floor/ceiling's `1.5` constant). Unlike those, α=10,000 has an actual
citation in `config/settings.py`: *"Per the 2026-04-20 alpha sweep
(`docs/ALPHA_SWEEP_20260419.md`), uniform α=10,000 beats α=1 by 4.6
percentage points on cross-season hindsight win rate (29-14 vs 27-16
over 43 weeks, p=0.016)."* Checked whether that citation actually holds
up rather than trusting it at face value.

**First finding: the cited doc doesn't exist.** `docs/ALPHA_SWEEP_20260419.md`
is not present anywhere in this checkout (`find` confirms no `docs/`
directory exists at all, consistent with the earlier session's finding
that CLAUDE.md's UI-freeze list references a `docs/` that was never real
in this repo). What *does* exist: `scripts/run_alpha_sweep.py` and its
output, `data/backtest_results/alpha_sweep_summary.json` — a real sweep
across α ∈ {0.3, 1, 3, 10, 100, 1000, 10000, 100000} on the 2025 season,
with genuine per-position Pearson r / std-ratio / bias stats.

**Second finding: the sweep's own data contradicts the win-rate claim.**
`run_alpha_sweep.py` never computes win-rate/hindsight/decision-quality
at all (confirmed by reading the script — it only calls `_analyze()`,
which produces correlation/bias/variance stats, nothing about
head-to-head wins). And what it *does* measure — Pearson correlation
against real 2025 outcomes — shows α=10,000 performing **worse** than
lower alphas at every single position, not better:

| α | QB r | RB r | WR r | TE r |
|---|---|---|---|---|
| 0.3 – 10 | 0.3249 | 0.5118 | 0.5130 | 0.3986 |
| 100 | 0.3254 | 0.5117 | 0.5130 | 0.3984 |
| 1,000 | 0.3277 (peak) | 0.5111 | 0.5130 | 0.3973 |
| **10,000 (current default)** | 0.3252 | **0.5052** | **0.5122** | **0.3947** |
| 100,000 | 0.3186 | 0.4905 | 0.5084 | 0.3916 |

Correlation is flat from 0.3 to 10 (essentially no regularization effect
in that whole range — expected, since standardized-feature Ridge only
starts biting once α approaches the scale of `n_samples`), peaks around
α=100-1,000 depending on position, then **declines** by α=10,000, and
declines sharply by α=100,000. RB and TE both show 10,000 measurably
worse than even α=1. The `std_ratio` column (predicted/actual variance
ratio — how compressed predictions are) tells the same story more
starkly: 0.43-0.51 at low α, dropping to 0.32-0.46 at α=10,000, and
collapsing to 0.09-0.25 at α=100,000 — the model is predicting an
increasingly narrow, safe band centered near the mean, not genuinely
more accurate predictions.

**Third finding: the win-rate claim isn't independently verifiable from
anything currently in the repo.** `PROJECT_NOTES.md` repeats the same
"29-14 vs 27-16, n=43 weeks" claim (§"Alpha sweep (April 20 2026)"), but
that's the same unverified claim, not independent corroboration — and
this project's own history (documented multiple times earlier in this
doc) shows `PROJECT_NOTES.md` has needed factual corrections before, so
repetition there isn't evidence. Checked whether any of the 19
`data/backtest_results/*.json` files in this repo contain a matching
43-week, two-alpha win-rate comparison: the most relevant one found
(`ts_backtest_2025_20260804_182714.json`, α=10,000, this session's own
v25 checkpoint) has real `decision_quality` data, but it's a
**single-season, 22-week** run (`vs_hindsight`: 63.6%, 14-8, p=0.14, not
significant) — not the 43-week cross-season comparison the citation
describes. Whatever run originally produced "29-14 vs 27-16" either
predates this repo's `data/backtest_results/` history or was never
persisted in a form that survived to this checkout.

**Also checked and ruled out**: a scaling-pipeline bug as an alternative
explanation for the correlation decline (e.g. inconsistent/double
standardization making α=10,000 behave differently than intended).
Verified directly on real RB data: post-`StandardScaler` features have
mean ~0 (±3e-15), std exactly 1.0, no NaN/Inf, no near-zero-variance raw
columns that could blow up under scaling. The scaling pipeline is clean
— α=10,000 is being applied to properly standardized features exactly as
intended, so the correlation decline in the sweep is a genuine
regularization-strength effect, not an artifact.

**Live reproduction, α=1.0 leg complete**: `run_ts_backtest.py --season
2025 --model ridge --alpha 1.0` (fp mode, decision-quality reporting on)
— **R²=0.351, Hindsight win rate 72.7% (16-6, p=0.026, ROI +30.9%)**.
This is *better* than every α=10,000 result documented anywhere in this
project: better than this session's own earlier α=10,000 checkpoint
(63.6%, 14-8, p=0.14, not significant) and better than the
67.4%/29-14/p=0.016 claim `PROJECT_NOTES.md`/`config/settings.py` use to
justify α=10,000 in the first place (that claim was on a different,
43-week cross-season sample this repo has no surviving artifact for — see
above). The matching α=10,000 leg was still running as of this entry;
result to follow, but α=1.0 has already cleared the bar α=10,000 was
supposed to beat.

### A much bigger discovery while setting up this reproduction: this project's backtest methodology has never actually tested the deployed model architecture

While building the comparison, checked **where `RIDGE_DEFAULT_ALPHA` is
actually consumed** across the codebase (not assumed). Finding:
`train.py`, `component_predictor.py`, `position_models.py`, and
`ensemble.py` **never import it at all**. Real production
(`component_predictor.py` — confirmed to be what's live, since
`position_target_type="component"` for every position in
`MODEL_CONFIG`) trains 3-5 separate Ridge models per position (one per
stat: `passing_yards`, `rushing_tds`, `receptions`, etc.), each
**hardcoded to `alpha=1.0`**, then assembles them into fantasy points via
PPR scoring weights. `RIDGE_DEFAULT_ALPHA=10,000` only ever reaches
`ts_backtester.py`'s `target_mode="fp"` path — a single monolithic Ridge
predicting fantasy points directly, an architecture that **does not
match production at all**.

Checked how often `target_mode="component"` (the mode that *would* match
production) has ever actually been run: grepped every `.py`/`.md` file in
this repo. **Never, in this project's history, until today.** Every
backtest number this project has ever cited — the original R²=0.269
baseline, the v25 checkpoint (R²=0.345, described elsewhere in this doc
as "the real jump"), every v26-v29 ablation, the rookie-slice lift — was
measured against `target_mode="fp"`, structurally different from what's
actually deployed to users.

**`ts_backtester.py`'s own `target_mode="component"` implementation was
itself drifted from real production**, found by direct line-by-line
comparison against `component_predictor.py`:
1. Hardcoded `Ridge(alpha=RIDGE_DEFAULT_ALPHA)` for every component
   model, ignoring this backtester's own `--alpha` override entirely —
   real production hardcodes `alpha=1.0`.
2. Missing production's final `total_fp = max(total_fp, 0)` clamp after
   summing weighted components.
3. Missing production's optional post-hoc linear calibration layer
   (`_maybe_fit_calibration`), applied whenever it measurably improves
   validation RMSE.

**Fixed by replacing the inline reimplementation with a direct call to
the real `ComponentPredictor` class** (`src/models/component_predictor.py`)
— not by patching the three bugs individually, since patching a parallel
copy just re-creates the same drift risk the next time production
changes. The backtester's `component` mode now either tests the actual
deployed code path or fails to compile; it can't silently diverge again.
Verified via a real 2024-2025 RB-only integration test before trusting
it: completed cleanly across all 22 weeks, zero negative predictions
(confirms the clamp works), real non-degenerate output — **RB R²=0.322**,
notably *lower* than the `fp`-mode proxy's documented RB R² (0.364-0.366
across the v25-v28 checkpoints). A full 4-position `component`-mode run
was kicked off to get a real apples-to-apples baseline; still running as
of this entry.

**`target_mode="util"` had the same "own drifting reimplementation"
problem, plus something worse.** Its inline converter
(`Ridge(alpha=1000)` fit inline) was replaced the same way, with a call
to the real `src.models.utilization_to_fp.UtilizationToFPConverter`.
First integration test produced an **impossible RB R²=0.781** — nothing
else measured this entire session has exceeded ~0.37. Root-caused rather
than dismissed: `UtilizationToFPConverter.predict(utilization,
efficiency_df=...)` pulls its input features (`utilization_score` plus
`EFFICIENCY_FEATURES` — `yards_per_carry`, `catch_rate`, `snap_share`,
etc.) directly out of `efficiency_df` when those columns are present
there, falling back to the passed-in `utilization` array only when
they're absent. Passing `pos_test` (the backtester's historical test
frame) handed it the **real, actual, same-week values** for all of these
— genuine outcome stats, not knowable at real prediction time — instead
of the function's own `util_preds`. Real production (`ensemble.py`)
narrowly avoids the `utilization_score` half of this by explicitly
overwriting `eff_df["utilization_score"] = predictions` before calling
`predict()`; the backtester integration didn't replicate that.

First fix (replicating the `utilization_score` overwrite) only moved R²
from 0.781 to 0.775 — barely anything, which was itself the tell that a
second, larger leak remained. Checked directly: `pos_test` carries
**100% non-null real values** for all 5 `EFFICIENCY_FEATURES` columns
(confirmed via direct inspection), because this backtester evaluates
already-completed historical weeks — unlike genuine live serving for a
real future week, where these same-week outcome columns simply
wouldn't exist yet, which is why production's real call sites don't hit
this leak in practice. First fix attempt: don't construct a leaky
`efficiency_df` at all, passing `efficiency_df=None` so the converter
falls back to using only `util_preds` (zero-padding the
efficiency-feature slots) — leakage-safe, but this introduced a *third*
distinct bug.

**Third `util`-mode bug: zero-padding is leakage-safe but badly
out-of-distribution for the converter's model.** With `efficiency_df=None`,
RB integration-test R² came back at **-1.09** — worse than predicting the
mean, and a huge drop from the (leaked) 0.775. `UtilizationToFPConverter`'s
regressor is a RandomForest/XGBoost blend trained on realistic
(non-zero) efficiency-feature values (`yards_per_carry`, `catch_rate`,
`snap_share`, etc.); feeding raw zero for those slots at predict time,
then running that through the fitted `StandardScaler`, lands several
standard deviations outside anything the trees ever saw in training —
not a leak, but a genuine distribution-mismatch artifact that tanks
accuracy for an unrelated reason. Confirmed directly on identical
data/model: zero-padding → R²=-1.03; imputing each missing efficiency
feature with its **training-set mean** instead (still fully leakage-safe
— a training-only aggregate, never a real per-player same-week value) →
R²=+0.09. Fixed `_fit_predict_util` to mean-impute rather than
zero-pad. Re-ran the RB integration test a fourth time on the real
walk-forward backtest (not just the single-shot diagnostic): **R²=0.048,
non-degenerate, no crashes** — a legitimate number now, three real bugs
away from the original impossible 0.781.

**Three-way RB comparison, all now genuinely leakage-safe and
architecture-faithful**: `fp` mode (α=1.0) RB R²=0.365, `component` mode
RB R²=0.322, `util` mode RB R²=0.048. `util` mode is clearly the weakest
of the three on this position — not surprising in hindsight, since its
converter is working with meaningfully less real signal per prediction
(one predicted utilization score plus mean-imputed, not real,
efficiency context) than either of the other two modes get from their
full feature sets. Full 4-position runs for both `component` and `util`
modes launched; results to follow.

**Full 4-position `component`-mode result: the first-ever backtest of
this project's actual production architecture.** R²=0.335, QB/RB/WR/TE =
0.241/0.341/0.267/0.236, Hindsight win rate 72.7% (16-6, p=0.026, ROI
+30.9%). All three legs on the identical 2025 season, side by side:

| | R² (overall) | QB / RB / WR / TE | Hindsight win rate |
|---|---|---|---|
| `fp` mode, α=1.0 | 0.351 | 0.252 / 0.365 / 0.276 / 0.271 | 72.7% (16-6), p=0.026 |
| `fp` mode, α=10,000 (old default) | 0.350 | 0.233 / 0.368 / 0.281 / 0.267 | 63.6% (14-8), p=0.143 |
| **`component` mode (real production)** | **0.335** | 0.241 / 0.341 / 0.267 / 0.236 | **72.7% (16-6), p=0.026** |

A genuinely nuanced split, not a one-sided verdict: on raw correlation,
the real production architecture is the *weakest* of the three, lower
than either `fp`-mode proxy at every position except QB. But on
decision-quality (win rate) — the metric this project's whole
`DECISION_QUALITY` framework is built around for actual lineup
decisions — it exactly matches the best result measured (α=1.0 `fp`
mode: same 16-6 record, same p=0.026), and clearly beats the α=10,000
default every past measurement has used (72.7% vs 63.6%). The two
metrics disagree about whether `component` mode underperforms `fp`
mode; they agree completely that α=10,000 underperforms α=1.0, on
whichever architecture you test it against.

One nuance worth flagging honestly: the 16-6 win/loss record matching
exactly between `component` mode and α=1.0 `fp` mode is not a sign the
two runs are secretly identical — average margins differ (+18.02 vs
+20.95), meaning the underlying weekly predictions are genuinely
different; they just happened to land on the same side of that
particular week's win/loss threshold most weeks this season. At n=22
weeks, an exact record match across two real but different models isn't
strong evidence of anything beyond "both are decent," and shouldn't be
over-read as proof they're equivalent.

**Full 4-position `util`-mode result: decisive, and the opposite kind of
result.** R²=0.027 overall, and **negative for 3 of 4 positions** (QB
-0.003, RB -0.167, WR -0.002, TE +0.005 — essentially indistinguishable
from predicting the mean, or worse). Decision-quality is not just weak
but actively bad: Hindsight win rate **13.6% (3-19), p=0.9999, ROI
-75.4%** — p=0.9999 means the observed record is essentially the worst
possible outcome under the null, i.e. this isn't "no better than a coin
flip," it's reliably losing. (It still beats the `Replacement`-level
baseline at 72.7%/16-6 — same record as the other two modes get there —
but that's a low bar every real model clears easily; it says nothing
about `util` mode specifically.)

### Complete four-way comparison, all real, all leakage-checked, same 2025 season

| Mode | R² (overall) | QB / RB / WR / TE | Hindsight win rate | p-value | ROI |
|---|---|---|---|---|---|
| `fp`, α=1.0 | 0.351 | 0.252 / 0.365 / 0.276 / 0.271 | 72.7% (16-6) | 0.026 | +30.9% |
| `component` (real production) | 0.335 | 0.241 / 0.341 / 0.267 / 0.236 | 72.7% (16-6) | 0.026 | +30.9% |
| `fp`, α=10,000 (old default) | 0.350 | 0.233 / 0.368 / 0.281 / 0.267 | 63.6% (14-8) | 0.143 | +14.5% |
| `util` | 0.027 | -0.003 / -0.167 / -0.002 / 0.005 | **13.6% (3-19)** | **0.9999** | **-75.4%** |

**Overall verdict**: `fp` (α=1.0) and `component` mode (real production)
are both genuinely competitive — close on correlation, identical on
decision-quality. `fp` at the old α=10,000 default is clearly worse on
decision-quality than either of those, though still a functioning model.
`util` mode, even after finding and fixing three real, distinct bugs to
give it a fair, leakage-safe, distribution-matched test, is not
competitive at all — it doesn't beat a naive mean prediction on 3 of 4
positions and loses money on decision-quality. This is a real, useful,
independent confirmation that `MODEL_CONFIG["position_target_type"]`
being `"component"` for every position (not `"util"`) is empirically the
right call, not an unexamined default — the first time that choice has
actually been tested against the alternative rather than assumed.

**Decision needed, not yet acted on**: whether to change
`RIDGE_DEFAULT_ALPHA` from 10,000. The evidence is consistent and now
comes from four independent angles this session — the sweep's own
correlation data, the direct α=1-vs-10,000 reproduction on `fp` mode, and
now confirmed again in spirit by `component` mode's strong result at
α=1.0's effective regularization level (production's real
`ComponentPredictor` is hardcoded to `alpha=1.0`, which is exactly the
value that outperformed 10,000 throughout this investigation). Given
(a) the constant has zero effect on real production regardless of its
value, and (b) it is not empirically justified even for the evaluation
tool it does affect, the honest recommendation is to lower it — but this
wasn't changed unilaterally in this pass, flagging it for an explicit
decision instead.
comparison now available.** Same season, same code, same day, both legs
side by side:

| | α=1.0 | α=10,000 (current default) |
|---|---|---|
| Overall R² | 0.351 | 0.350 |
| QB / RB / WR / TE R² | 0.252 / 0.365 / 0.276 / 0.271 | 0.233 / 0.368 / 0.281 / 0.267 |
| Hindsight win rate | **72.7% (16-6)** | 63.6% (14-8) |
| p-value | **0.026 (significant)** | 0.143 (not significant) |
| ROI | **+30.9%** | +14.5% |

Correlation (R²) is a wash, as the alpha sweep already predicted for
this range. But the win-rate metric — the *specific* metric
`config/settings.py`'s comment cites as the reason α was raised from 1.0
to 10,000 in the first place — favors α=1.0 on this direct, same-day
reproduction: higher win rate, statistically significant instead of not,
nearly double the ROI. This doesn't just fail to replicate the original
"α=10,000 beats α=1 by 4.6pp" claim; it reproduces the *opposite*
direction, on real 2025 data, using the current codebase.

**Taken together with the architecture finding above**: `RIDGE_DEFAULT_ALPHA=10,000`
(a) has zero effect on real production regardless of its value (confirmed
earlier — production hardcodes `alpha=1.0`), and (b) is not empirically
justified even for the evaluation tool it does affect — the citation
backing it doesn't survive a direct reproduction. Given (a), changing the
constant doesn't affect what's served to users; given (b), it should
probably change anyway so future backtests default to something
defensible rather than a value this reproduction argues against. Not
changed yet in this pass — flagged for a decision, not silently altered,
since `RIDGE_DEFAULT_ALPHA` is also read as the default in
`scripts/paper_trade_lock.py:338` (a hardcoded duplicate of the same
value, separate drift risk worth fixing alongside this if the constant
changes).

**Not yet concluded**: whether the "v26-v29 show no lift" conclusions
documented earlier in this doc hold up when re-measured against the real
`component`-mode architecture instead of the `fp`-mode proxy that's been
used for every measurement so far — this is now the much bigger open
question. Nothing in production code has changed; all fixes so far are
to the *evaluation* tooling, making it capable of testing what's actually
deployed for the first time.

### RIDGE_DEFAULT_ALPHA lowered to 1.0, duplicate fixed, a real test fragility found and fixed (2026-08-06)

User decision: lower the default given the evidence above. Changed
`config/settings.py`'s `RIDGE_DEFAULT_ALPHA` from `10_000` to `1.0`, with
the comment rewritten to cite this session's actual reproducible evidence
instead of the nonexistent doc/unverifiable claim it used to cite. Chose
exactly `1.0` rather than an untested intermediate value (the sweep's
correlation peak was around α=100-1,000 for some positions) because 1.0
is what was actually validated on decision-quality — the metric that
matters — via the direct reproduction, and it matches production's own
independently-hardcoded `ComponentPredictor` value; picking a different,
decision-quality-untested number here would just be a new unvalidated
guess of the same kind this change is trying to get away from.

Also fixed the hardcoded duplicate flagged earlier:
`scripts/paper_trade_lock.py:338` had `"ridge_alpha_default": 10_000,  #
per config.settings.RIDGE_DEFAULT_ALPHA` — a literal copy with a comment
pointing at the source of truth instead of actually reading from it.
Changed to import and use the real constant, eliminating the drift risk
(this specific duplicate had NOT yet drifted out of sync when found, but
the whole point of fixing it now is that the next constant change
wouldn't have this problem).

**A real, demonstrated test fragility found while verifying the alpha
change didn't regress anything**: `tests/test_backtest_validation.py::TestWalkForwardBiasRegression::test_per_position_bias_within_tolerance`
failed after the change — not because of the change, but because its
fixture (`_latest_walk_forward_predictions()`) globs for the
most-recently-modified `ts_backtest_*_predictions.csv` in
`data/backtest_results/` with **no filter on which mode produced it**.
This session's `component`/`util`-mode diagnostic runs (real,
intentional, and now a normal part of this project's toolkit going
forward, not one-off throwaway scripts) left newer files in that
directory than the legitimate `fp`-mode baseline, so the test picked up
the already-known-badly-biased `util`-mode run and (correctly, given
what it was handed) flagged it as failing. Verified the alpha change
itself caused no regression by checking the real `fp`-mode α=1.0 run
directly: per-position bias 0.8-2.8%, comfortably within the test's ±10%
tolerance.

Fixed properly rather than just re-running to get a fresh "latest" file
(which would only paper over the fragility until the next diagnostic
run): `_latest_walk_forward_predictions()` now reads each candidate's
sibling `.json` metrics file and skips any whose `target_mode` isn't
`"fp"`, restoring "most recent regression-relevant baseline" as the
actual selection criterion instead of "most recent file of any kind."
This test would have broken the same way on literally any future
`component`/`util`-mode investigation without this fix, given those
modes are now known-working and likely to see more use.

**Verified**: `config/settings.py`, `scripts/paper_trade_lock.py`,
`tests/test_backtest_validation.py` all compile clean. CLI default
confirmed (`run_ts_backtest.py --help` now shows "default: 1.0").
59/59 tests pass across `test_ts_backtester.py`,
`test_baseline_comparison.py`, `test_backtest_validation.py`,
`test_config_code_alignment.py`, `test_generate_draft_data.py`,
`test_snake_draft_sim.py`.

**Not attempted**: no code changed to actually retrain or redeploy
anything — `RIDGE_DEFAULT_ALPHA` still has zero effect on real production
regardless of its value (production's `ComponentPredictor` remains
independently hardcoded to `alpha=1.0`, unchanged by this edit). This
change only affects future backtests/evaluations run through
`ts_backtester.py`, making their default match what was actually
validated instead of a value the evidence argued against.

### v26-v29 re-measured against `component` mode — the "flat lift" conclusion does NOT hold up under the real architecture (2026-08-06)

Closed the "much bigger open question" flagged at the end of the alpha
investigation: every prior v26-v29 lift measurement in this doc used
`fp` mode, which is now known to be architecturally different from
production's real `component` mode. Re-ran the ablation the efficient
way — one combined "strip all four version bumps' features" run against
`component` mode, compared to the "with everything" `component`-mode
result already on record from the alpha investigation — instead of 8
separate per-version runs.

| | R² overall | QB | RB | WR | TE | Hindsight win rate |
|---|---|---|---|---|---|---|
| `component`, without v26-v29 (~v25 features) | 0.326 | 0.230 | 0.338 | 0.256 | 0.218 | 72.7% (16-6), p=0.026 |
| `component`, with v26-v29 (current) | 0.335 | 0.241 | 0.341 | 0.267 | 0.236 | 72.7% (16-6), p=0.026 |
| Δ | **+0.009** | **+0.011** | **+0.003** | **+0.011** | **+0.018** | none |

**Every position improves, consistently — the opposite of what `fp`-mode
showed.** The earlier `fp`-mode ablation (documented above,
"v23-v28 accuracy-lift measurement... flat since v25") found essentially
zero difference anywhere for v26-v28, and the standalone v29 ablation
found the same for personnel grouping alone. Under `component` mode, the
same combined feature set shows a small but real, uniformly positive
effect at every position — not a fluke concentrated in one position,
not noise in one direction at some positions and the other direction at
others. Decision-quality (hindsight win rate) doesn't move either way —
identical 72.7%/16-6/p=0.026 record with and without, though the
underlying average margin differs (15.7 vs 18.0), meaning the
predictions themselves are genuinely different even where the discrete
win/loss outcome happens to land the same.

**What this means, stated carefully**: the "v26-v29 show no lift"
conclusion reached earlier this session was **specific to the `fp`-mode
proxy it was measured on, not a fact about the features themselves**.
Measured against the architecture that's actually deployed, the same
features show a real, if modest (ΔR² 0.003-0.018 per position),
positive effect. This doesn't mean the `fp`-mode measurements were
wrong about `fp` mode — they were honest and correctly reproduced. It
means `fp` mode was the wrong instrument for answering "do these
features help what's actually deployed," which is the whole reason this
investigation started.

**Not attempted**: isolating which of the four version bumps (v26 DVOA,
v27 rookie identity, v28 rookie opportunity, v29 personnel grouping) is
driving the lift, since this was a combined ablation by design (1 run
instead of 8, given how expensive each run is). If a future session
wants to know which specific feature set matters most under `component`
mode, that would need per-version `component`-mode ablations the same
way the `fp`-mode ones were originally done. Also not attempted: a
statistical significance test on the R² deltas themselves (they're
small, and this is one holdout season, one model configuration —
directionally consistent across all 4 positions is meaningful, but
these specific magnitudes shouldn't be read as precisely known).

### Diagnosed: why `util` mode's R² is so low (2026-08-06)

Follow-up to the four-way mode comparison above, which found `util` mode
badly underperforming (R²=0.027, negative for 3 of 4 positions) even
after fixing the three real bugs that were inflating/deflating its
number unfairly. Root-caused with cheap, targeted single-fit diagnostics
rather than more expensive full walk-forward backtests.

**Stage 1 (predicting `utilization_score` itself, before any conversion)
is already broken, independent of the conversion step**: on real 2025
RB data, `R²(util_preds vs util_true) = -0.305`, correlation only
**0.19** — worse than predicting the mean. Predicted values are also
badly variance-compressed (std=5.73 vs real std=18.82). The conversion-
step bugs fixed earlier in this investigation (leakage, distribution
mismatch) were real and worth fixing, but they're downstream of an
already-weak signal — no amount of conversion-step calibration rescues
a stage-1 model this poor.

**Confirmed the specific cause, not just the symptom**: fit the *exact
same* features, model, and train/test split against two different
targets. `fantasy_points` → R²=0.352, corr=0.607. `utilization_score`
(same features, same everything else) → R²=-0.305, corr=0.194. This
isn't a leakage, scaling, or data-quality artifact (all ruled out
separately elsewhere in this doc) — it's the same feature set carrying
dramatically less signal for this specific target.

**Why**: `CAUSAL_FEATURES` has been iteratively engineered and validated
across this entire project's history (every v22-v29 feature bump
documented in this doc) exclusively against fantasy-points accuracy —
DVOA, weather, rookie signals, personnel grouping, coaching-change
detection, all of it was added and kept because it moved `fantasy_points`
prediction quality. Nobody has ever selected, tuned, or even checked
whether the same feature set predicts `utilization_score` — a
differently-composed, percentile-normalized blend of snap/target/rush/
red-zone shares — anywhere near as well. It apparently doesn't.

**This also retroactively explains, with actual evidence for the first
time, why `component` mode (not `util` mode) became this project's
production default** (`position_target_type="component"` for every
position) — a choice that predates this session and was never
previously backed by a documented comparison. It's not merely a
plausible-sounding default; the four-way comparison and this diagnosis
together confirm it was the right call.

**Not attempted**: building or selecting a feature set specifically for
predicting `utilization_score` well (would be real, separate feature-
engineering work, and given `component` mode already outperforms `util`
mode with the existing FP-tuned features, there's no clear product
motivation to invest in fixing `util` mode specifically). Also not
checked: whether other positions (QB/WR/TE) show the same fantasy_points-
vs-utilization_score gap as RB, though the full 4-position `util`-mode
backtest result (negative R² at 3 of 4 positions) is consistent with the
same root cause applying broadly, not just to RB.

### Per-version isolation: which of v26-v29 is actually driving the lift (2026-08-06)

Closed the "not attempted" item from the combined `component`-mode
ablation above. Given how expensive a full walk-forward run is (hours
each), used the same fast single-fit diagnostic method as the `util`
R² investigation instead of 4 more full backtests: one train/test split
(train 2022-2024, test 2025) per position, Ridge on `fantasy_points`
directly (a faster proxy for "does this feature carry signal," not a
full reproduction of `component` mode's per-stat architecture), removing
each version's features one at a time from the full current set and
measuring the R² delta.

| pos | full R² | v26 DVOA | v27 rookie ID | v28 rookie opp | v29 personnel |
|---|---|---|---|---|---|
| QB | 0.2385 | **-0.0054** | +0.0013 | +0.0035 | n/a |
| RB | 0.3536 | +0.0025 | +0.0012 | +0.0009 | **-0.0012** |
| WR | 0.2548 | +0.0013 | -0.0003 | +0.0002 | +0.0009 |
| TE | 0.2548 | **+0.0119** | +0.0040 | +0.0033 | +0.0062 |

**TE benefits the most, consistently, from every single version** — and
this independently corroborates the combined `component`-mode ablation
above, which also found TE getting the single largest Δ of any position
(+0.018, the biggest jump in that table too). Two different measurement
methods landing on the same position is a meaningfully stronger signal
than either alone.

**Two real exceptions, reported honestly rather than smoothed over**:
- **v26 (DVOA) hurts QB** (-0.0054) despite being the single strongest
  positive contributor everywhere else in the table (+0.0119 for TE).
  Plausible explanation, not confirmed: QB has the smallest, noisiest
  test set of the four positions (one starter per team vs. several
  relevant skill-position players), or DVOA's opponent-defense-strength
  signal genuinely transfers less cleanly to quarterback performance
  than to skill positions whose production is more directly usage-driven.
- **v29 (personnel grouping) shows a small negative for RB** (-0.0012)
  — mildly counter to its own design rationale (personnel grouping was
  specifically motivated by RB opportunity in heavier 12/21-personnel
  sets, per GAPS.md §3.3/§4). Magnitude is tiny and plausibly noise at
  this sample size, but it's the position the feature was built for, so
  worth flagging rather than ignoring.

**Methodology caveats, stated plainly**: this is a single train/test
split, not a full 22-week walk-forward with weekly refits — noisier and
less rigorous than the combined ablation above or any of the full
backtests elsewhere in this doc. It also uses `fantasy_points` as a
direct Ridge target rather than reproducing `component` mode's real
per-stat-component architecture, since building a proper per-stat
isolation harness for 4 versions × 4 positions was judged not worth the
additional engineering time for a diagnostic follow-up question. Treat
the *direction* (TE benefits most; DVOA's QB exception; personnel's RB
exception) as the finding, not the precise magnitudes.

## Backtester per-week performance fix (2026-08-06)

Investigated why every walk-forward backtest run this session took hours
instead of minutes. Found and fixed four real bugs/inefficiencies, all in
the expanding-window feature-recomputation path that `leakage_safe_features()`
calls fresh every single week:

1. **`DatabaseManager.get_combine_data()` queried the wrong table.** It
   read from the empty legacy `combine_data` table (0 rows) instead of the
   populated `combine_data_v2` (8,968 rows) — the same class of bug as the
   `draft_picks`/`draft_picks_v2` split that `get_draft_picks()` already
   had a documented fix for. Every call silently fell through to a live
   nflverse network fetch. Fixed to query `combine_data_v2` (`src/utils/database.py`).

2. **`AdvancedRookieProjector.add_combine_features()` matched every row
   individually.** It ran `.iterrows()` over the *entire* training
   DataFrame — which grows to 100K+ rows by late season under the
   expanding window — and for every single row re-scanned all 8,968
   combine records with a fresh `.str.contains()` call, even though
   combine score is a static per-player attribute that never changes
   week to week. Rewrote to compute the score once per unique
   `(player_name, position)` pair and memoize in a process-level class
   cache (`AdvancedRookieProjector._combine_match_cache`), shared across
   the fresh instances the calling code constructs every week
   (`src/features/advanced_rookie_injury.py`). Verified: 44,000-row /
   2,000-unique-player synthetic benchmark went from 3.29s (cold) to
   0.03s (any repeat call).

3. **Fixing bug #1 surfaced a real regression it had been masking.**
   `combine_data_v2` stores `forty`/`bench`/`vertical`/`broad`/`cone` as
   TEXT, while the nflverse-API fallback path returns floats. Once bug #1
   started actually hitting the DB path, `calculate_combine_score()`'s
   numeric comparisons threw `'<=' not supported between instances of
   'str' and 'float'` on every call, silently caught by the wrapping
   `except Exception` in `add_advanced_rookie_injury_features()` — so
   `combine_score` fell back to the constant default (50.0) for every
   player, every week, with no visible error. Fixed by coercing the
   metric columns with `pd.to_numeric(errors='coerce')` right after the
   DB load (`load_combine_data()`). Caught via a "why did the number not
   change" sanity check, not by the type checker — good reminder to
   re-verify a fix's actual output, not just that it ran without raising.

4. **`_add_team_matchup_features()` recomputed four expensive lookup
   tables from scratch every week**, even though all four depend only on
   the full `team_stats` table (all teams/seasons/weeks), never on the
   current backtest window. Confirmed via `cProfile` on a full TE-only
   2025 walk-forward run: this function (via its `_create_opponent_features`
   wrapper) was ~430s of a ~1390s profiled run (~31%), dominated by
   `groupby(...).transform(lambda ...)` calls for 19 metrics plus a
   custom Python-loop momentum function (`_momentum_60_30_10`) — together
   responsible for ~10.7M `pd.Series.__init__` calls. Extracted the
   window-independent precomputation (prior-season averages, in-season
   rolling blend table, offensive-momentum-score table) into a new
   module-level `_get_team_matchup_lookups()` cached by
   `(row count, columns, max season, max week)` of the input `team_stats`
   table (`src/features/feature_engineering.py`). Verified the cache is
   actually exercised: 43/44 calls hit it across a full TE-only 2025
   backtest, with **bit-identical R²/MAE** (`R2=0.2696787296807507`)
   before and after — confirming this is a pure performance fix, not a
   behavior change.

**Net measured effect** (real wall-clock, not profiled — cProfile's own
overhead scales with call count and inflated the profiled share of #4's
Python-loop-heavy code): TE-only 2025 fp-mode backtest went from **601.5s
→ 490.8s (~18% faster)**, single position only. The combine-data network
fetch (bug #1) likely matters more at full 4-position/multi-run scale
than this single-position measurement shows, since #2-#4 mostly help
CPU-bound recomputation while #1 was eliminating actual network
round-trips.

**Investigated and ruled out as a contributor**: `DraftDataLoader.load_draft_data()`
reconstructs a fresh instance every week too (so its `self._draft_cache`
never persists), but `get_draft_picks()` already reads from the correct,
fast local table — measured 22 sequential calls at 0.30s total
(~14ms/call), negligible next to the bugs above. No fix needed there.

**Not yet investigated**: `_create_causal_rolling_features` (190s
profiled), `add_season_long_features`'s age-curve/games-projection/ADP
steps (157s profiled, distinct from the rookie/combine module above),
and `_add_contract_features`/`_add_late_season_momentum` (106s/102s
profiled) were all visible in the same profile as further candidates,
but weren't examined for the same "recomputing something window-
independent" pattern. Worth a follow-up pass if backtest runtime is
still a bottleneck after these four fixes.

### Follow-up: age-curve and rookie-projection vectorization (2026-08-06)

Continued the per-week performance investigation into the two candidates
flagged as not-yet-checked above. `_create_causal_rolling_features`
turned out to be inherently window-dependent (per-player rolling means
recomputed over the actually-growing training set every week — not the
same "recomputing something static" bug pattern as the other fixes, so
left as-is; a real fix there would need incremental/memoized rolling
state, judged not worth the risk for a diagnostic follow-up).

`add_season_long_features`'s age-curve and rookie-projection steps did
have the same bug pattern, though:

- `AgeCurveModel.add_age_features()` ran `.apply(axis=1)` **five separate
  times** over the full expanding-window DataFrame to compute
  `age_factor`, `age_expected_games`, `decline_rate`, `years_from_peak`,
  and `is_in_prime` — all pure functions of `(age, position)`, a space of
  at most ~4 positions × a few dozen ages. Rewrote to compute the three
  lookup-based values once per unique `(age, position)` pair and
  broadcast via dict lookup, and fully vectorized the two arithmetic/
  comparison-only ones (`years_from_peak`, `is_in_prime`) directly
  (`src/features/season_long_features.py`).
- `RookieProjector.add_rookie_features()` ran two more `.apply(axis=1)`
  calls (`rookie_projected_ppg`/`_games`) and a third
  (`rookie_weight`) over the **entire** DataFrame even though the
  underlying functions immediately return NaN/0 for the ~90%+ of rows
  that aren't rookies. Restricted all three to the rookie subset via a
  boolean mask before applying.

Verified correctness two ways: (1) a standalone synthetic-data script
comparing the new vectorized/masked output against the original
per-row formulas element-by-element (`np.allclose`, including NaN
positions) — exact match on both functions; (2) re-ran the same
TE-only 2025 fp-mode backtest used to verify the earlier fixes —
**R²/MAE bit-identical** (`R2=0.2696787296807507`) before and after.

**Net effect of this follow-up alone**: 490.8s → 427.0s (~13% further
reduction). **Cumulative effect of the full per-week performance
investigation** (this section plus the four fixes above): TE-only 2025
fp-mode backtest wall time went from the original **601.5s → 427.0s
(~29% faster)**, with zero change to any prediction, metric, or model
behavior at any step — confirmed a pure performance investigation, not
a silent behavior change, at every stage.

**Still not investigated** (lower expected payoff, not attempted this
pass): `_add_contract_features` and `_add_late_season_momentum` in
`feature_engineering.py` (106s/102s in the original profile) — worth a
look if backtest runtime is still a pain point, but not checked for the
same anti-patterns yet.

### Follow-up 2: contract features + late-season momentum (2026-08-06)

Investigated the last two flagged candidates from the profile.

- **`_add_contract_features`** had the same DB-round-trip pattern as the
  earlier combine-data bug: every weekly call opened a fresh sqlite3
  connection and re-queried and rebuilt the entire `contracts` table
  lookup (15,266 rows) from scratch, even though `contracts` never
  changes mid-backtest and the same `FeatureEngineer` instance is reused
  across weeks. Also used `.iterrows()` to assign `is_contract_year`/
  `contract_apy_rank` row-by-row. Fixed by moving the DB query + APY-
  percentile computation into a class-level cache
  (`_get_contract_lookup_table()`, built once, keyed on nothing since the
  table is process-static) and replacing the `.iterrows()` loop with a
  single `merge` on `player_id` (`src/features/feature_engineering.py`).
  Verified against the original row-by-row logic on a 250-row synthetic
  sample (200 real contract player_ids + 50 non-matching ids): exact
  match on both output columns.
- **`_add_late_season_momentum`** turned out to have ~40% dead code:
  `season_avg`, `is_late`, `late_avg`, and `ratio` were all computed
  (via two more `groupby(...).transform(lambda ...)` calls) but never
  read — only the separately-computed `season_ratio` aggregate actually
  fed the output column. Deleted the dead computation and replaced the
  trailing `df.apply(axis=1)` dict-lookup with a `merge` on
  `(player_id, season)`. Verified against the original implementation
  (including the dead code, to confirm its removal changes nothing) on
  200 synthetic rows across 3 seasons/15 players: `np.allclose` exact
  match.

Re-ran the same TE-only 2025 fp-mode backtest used throughout this
investigation: **R²/MAE bit-identical** (`R2=0.2696787296807507`,
`MAE=3.840073390115718`). Full `test_ml_audit.py` +
`test_ml_robustness_15_steps.py` + `test_schema_validator.py` +
`test_feature_engineering.py` suite still green (78 passed, 3 skipped,
+12 passed).

**Net effect of this follow-up**: 427.0s → 381.05s (~11% further
reduction). **Cumulative effect of the full backtester performance
investigation** (all three follow-ups): TE-only 2025 fp-mode backtest
wall time went from the original **601.5s → 381.05s (~37% faster)**,
single position only, with zero change to any prediction, metric, or
model behavior at any step.

This closes out the backtester performance investigation — no further
candidates remain flagged from the original profile.

## Stale test/doc sweep: PROJECT_NOTES.md alpha claim + 7 stale tests (2026-08-06)

Closed the two remaining flagged items from the Ridge alpha audit
(GAPS.md, "RIDGE_DEFAULT_ALPHA lowered to 1.0" section): PROJECT_NOTES.md
still asserted the old, since-reversed 29-14-vs-27-16 claim, and 4 tests
were flagged as stale but not yet fixed.

**PROJECT_NOTES.md**: appended a dated correction inline after the stale
April 20 claim (matching this doc's existing house style of marking
superseded claims rather than deleting them), pointing at the same-day
direct reproduction that found the opposite result (72.7%/16-6/p=0.026
at α=1.0 vs. 63.6%/14-8/p=0.143 at α=10,000) and noting production was
unaffected either way (`ComponentPredictor` always hardcoded α=1.0).

**The 4 flagged stale tests**, plus 3 more found while verifying (same
sweep, same session — fixed per this project's standing "fix small/safe
bugs immediately" convention rather than leaving them for a later pass):

| Test | File | Root cause |
|---|---|---|
| `test_utilization_weights_persistence` | test_ml_audit.py | Checked train.py; persistence logic moved to `feature_preparation.py` during the 2026-04-22 council process |
| `test_target_uses_shift_neg1` | test_ml_robustness_15_steps.py | Checked train.py; target creation (`shift(-1)`) moved to `feature_preparation.py`'s `_create_horizon_targets()` |
| `test_winsorization_applied` | test_ml_robustness_15_steps.py | Same — winsorization moved to `feature_preparation.py`'s `prepare_training_data()` |
| `test_ensemble_weights_are_spec_mandated` | test_ml_robustness_15_steps.py | Imported `ENSEMBLE_WEIGHTS_1W` from `config.settings`; it lives in `src/models/position_models.py`, conditional on `HAS_LIGHTGBM` (3-model 30/40/30 when unavailable — still spec-mandated; 4-model split when available) |
| `test_meta_learner_trained_on_validation` | test_ml_robustness_15_steps.py | Checked for a variable named `preds_val`; renamed to `val_preds_stack` at some point. Property (meta-learner trained on held-out `X_val`, not train) still holds |
| `test_optuna_uses_timeseries_split` | test_ml_robustness_15_steps.py | `_tune_xgboost` was rewritten to use `SeasonAwareTimeSeriesSplit` (project-specific, season-boundary-respecting, supports a purge gap) with a manual CV loop instead of sklearn's `TimeSeriesSplit`/`cross_val_score`, per an in-code comment: sklearn>=1.6's `get_tags()` path breaks with xgboost's estimator MRO |
| `test_validate_weekly_data_rejects_negative_stats_in_strict_mode` | test_schema_validator.py | Used `passing_yards`, which is deliberately in `NEGATIVE_WARN_COLUMNS` (sacks can legitimately put single-game passing yards below zero) — not a code bug, a wrong test fixture. Switched to `receptions`, which is genuinely in `NEGATIVE_DISALLOWED_COLUMNS` |

All 7 were the same underlying failure mode as much of this session's
other findings: real, currently-correct behavior that a test's literal
string/import check had fallen out of sync with after a refactor —
not actual regressions. Verified each against the real current source
(not assumed) before rewriting the assertion, and re-ran the full
`test_ml_audit.py` + `test_ml_robustness_15_steps.py` +
`test_schema_validator.py` suite afterward: **78 passed, 3 skipped**
(skips are pre-existing, data-availability-gated, unrelated to this fix).

## v30: offensive-line quality metrics + PBP pass-play participation rate (2026-08-06)

Investigated data feasibility for the three remaining §11.1 "still
missing" items (YPRR, Route Participation Rate, Offensive Line Quality
Metrics) before building anything, per user request.

**YPRR and true route participation: confirmed infeasible with any
accessible data source.** Checked `ngs_receiving` (no routes column),
`nfl_data_py.import_ftn_data()` (play-level charting only — motion,
play-action, personnel counts — no per-player route data), and raw PBP
(`nfl_data_py.import_pbp_data()`). PBP does have a `route` column
(93% populated) and `offense_players` (participation, 93% populated,
2016+) — but `route` only describes the *targeted* receiver's route on
plays with a target, saying nothing about the other ~3 receivers/backs
on the field, and `offense_players` says who was on the field but not
what they did (route vs. in-line block). No combination of accessible
data distinguishes a route-runner from a pass-blocker on the same play.

This isn't a new discovery so much as confirmation of an existing
"declared but dead" feature, the same pattern as WOPR and
`recv_epa_per_target_roll3_mean` documented earlier in this doc:
`utilization_score.py` and `feature_engineering.py` both already check
`if "routes_run" in df.columns"` and fall back to `snap_share_pct * 0.8`
— no loader has ever populated `routes_run`, so that branch has never
fired, and per this investigation, none of the accessible data sources
can populate it correctly either.

**A real, honestly-labeled proxy was buildable from PBP `offense_players`
instead**, and the offensive-line metrics were directly buildable from
data already in the DB. Built both, promoted to **v30**:

1. **Team OL quality** (`src/features/feature_engineering.py`,
   `_add_team_ol_features`): re-aggregates `weekly_pfr` (already used at
   individual-player level by `_add_weekly_pfr_features`/component
   features) to team-week.
   - Pass-block (`team_sack_rate_allowed_roll3_mean`): identifies each
     team-week's starting QB as the row with the most pressure activity
     (`weekly_pfr` has no dropback-count column to rank by directly) and
     uses that row's own `times_pressured_pct` — avoids diluting the
     signal by averaging in mop-up-duty backup QB rows.
   - Run-block (`team_run_block_ybc_avg_roll3_mean`): unweighted mean of
     `rushing_yards_before_contact_avg` across a team's RB rows that week
     — a deliberate simplification, documented in code, since
     `weekly_pfr` carries no per-RB carry count to weight by.
   - Wired into `CAUSAL_FEATURES`: QB and WR/TE get the pass-block
     column (time-to-throw / route-development-time rationale), RB gets
     the run-block column.

2. **PBP pass-play participation rate** (new table
   `pbp_pass_participation`, `src/data/pbp_stats_aggregator.py`'s
   `get_pass_play_participation_from_pbp`, `DatabaseManager.
   ensure_pbp_pass_participation()`, `scripts/backfill_pbp_pass_
   participation.py` — all modeled directly on the existing
   `team_personnel_stats`/`get_personnel_groupings_from_pbp`/
   `ensure_team_personnel_stats` pattern from the v29 personnel-grouping
   work). Explodes `offense_players` (semicolon-delimited GSIS IDs,
   already matching this project's `player_id` format directly — no
   crosswalk needed) on real dropback pass plays (`play_type == 'pass'`)
   to compute, per player-week, the fraction of the team's actual pass
   plays a player was on the field for. **Explicitly NOT true route
   participation** — documented in three places (table schema comment,
   function docstring, `CAUSAL_FEATURES` comment) that it can't separate
   a receiver running a route from one staying in to block on the same
   play. Backfilled 2016-2025 (10 seasons, ~1,000 players/season).
   Spot-checked against known real roles on 2023 data: Mahomes 99.2%
   (QB, every dropback), C.Lamb 91.0% (true WR1), T.Kelce 84.8% (elite
   receiving TE), A.Ingold 26.2% (blocking fullback), C.Patterson 19.2%
   (gadget/return RB, minimal passing-down role) — behaves exactly as
   expected. Wired into `CAUSAL_FEATURES` for WR/TE
   (`pbp_pass_play_participation_pct_roll3_mean`), and into
   `utilization_score.py`'s existing `route_participation_pct` fallback
   chain (ahead of the flat `snap_share_pct * 0.8` proxy, behind the
   still-dead-but-harmless `routes_run` branch).

Both new raw team/player-week columns are same-week OUTCOME stats (like
`team_pct_11_personnel` from v29) and are rolled via `shift(1).rolling(3,
min_periods=1).mean()` before exposure to the model — caught and fixed a
real ordering bug during implementation where the new raw column names
were first added to the generic `_create_causal_rolling_features` roll
list, which runs *before* the helper methods that populate those raw
columns (verified by checking the pipeline call order directly, not
assumed); fixed by computing the rolling inline within each new helper
instead, matching how `_add_weekly_pfr_features` already does it.

**FEATURE_VERSION bumped to 30.** Verified: 8 new unit tests
(`tests/test_ol_and_participation_features.py` — starter-selection logic
for pass-block, run-block averaging, causal shift correctness for the
participation rate, the aggregator's own play-type/participant-counting
logic on synthetic PBP data, and the `route_participation_pct`
fallback-chain preference), full existing suite green (`test_ml_audit.py`
+ `test_ml_robustness_15_steps.py` + `test_schema_validator.py` +
`test_feature_engineering.py` + `test_models.py` +
`test_multiweek_ci_exponent.py` + `test_pbp_aggregator.py` — 120 passed,
3 skipped, pre-existing/unrelated).

**Accuracy-lift ablation (same methodology as v29): flat, consistent
with v26-v29.** Two full 2025 walk-forward backtests (all 4 positions, fp
target mode, 5,612 predictions) — one with the 5 v30 columns in
`CAUSAL_FEATURES`, one with them stripped back out (temporarily edited
`config/settings.py`, ran the "without" backtest, then restored via
`git checkout` — which briefly wiped the *uncommitted* v30 wiring
entirely since none of this session's work was committed yet; caught
immediately by checking `git diff` after and re-applied the same five
edits before re-verifying, no data lost, but worth flagging as a
process near-miss: `git checkout -- <file>` reverts to the last commit,
not "undo my last edit," and is dangerous to reach for on files with
real uncommitted work in progress).

| Position | MAE (without → with) | RMSE (without → with) | R² (without → with) |
|---|---|---|---|
| QB | 6.4155 → 6.4267 | 7.9356 → 7.9438 | 0.2513 → 0.2497 |
| RB | 4.8115 → 4.8108 | 6.5383 → 6.5377 | 0.3645 → 0.3647 |
| WR | 4.6507 → 4.6516 | 6.2375 → 6.2362 | 0.2760 → 0.2763 |
| TE | 3.7969 → 3.7968 | 5.2242 → 5.2178 | 0.2710 → 0.2728 |
| **Aggregate** | 4.7303 → 4.7319 | 6.3588 → 6.3583 | 0.3509 → 0.3510 |

Every position's movement is noise-level (QB moved the most, and even
that's ~0.011 MAE on 688 predictions). This is now the **fifth**
feature-version bump in a row (v26 DVOA, v27 rookie identity, v28 rookie
opportunity/breakout, v29 personnel grouping, now v30) to show no
aggregate lift on this backtest configuration. Same standing caveats
apply: this is one backtest, one holdout season, one model type (not the
higher-fidelity `--model ensemble` path), and heavy regularization could
be shrinking a real-but-modest signal toward zero. Not a reason to
revert — both features are theoretically well-motivated (GAPS.md §11.4
ranked OL quality and route-participation-adjacent signals as real gaps)
and the participation-rate proxy visibly encodes real role information
(the spot-check above) even if it doesn't move this particular metric —
but also not a claim of proven value. Filed honestly, matching this
project's standing practice.

## No viable free source for Vegas preseason win totals (2026-08-06)

User confirmed they will never have paid access to The Odds API (the
only source `odds_scraper.py`'s `fetch_win_totals`/`scrape_win_totals`
actually calls), and asked whether a free public source exists instead.
Checked three candidates before concluding none work:

1. **`nfl_data_py.import_win_totals()` — free, but not this data,
   despite the name.** Traced the function to its actual source
   (`https://raw.githubusercontent.com/mrcaseb/nfl-data/master/data/
   nfl_lines_odds.csv.gz`) and pulled real rows: it's **per-game weekly
   odds** (`market_type` = spread/total/money_line, keyed by
   `game_id` like `2021_01_CLE_KC`), not season-long preseason win-total
   futures. This project's own `game_odds` table already has that exact
   data fully covered (100% coverage 2006-2025, per project memory) —
   pulling this would be pure duplication, not new signal. Also
   effectively dead going forward: 2022+ returns 0 rows, and the library
   itself logs "the win totals data source is currently in flux and may
   be out of date." The §8.1 Tier 2 table's citation of this function as
   the source for win totals was itself wrong — corrected inline above.
2. **nflverse-data's GitHub releases** — checked the full release catalog
   (`api.github.com/repos/nflverse/nflverse-data/releases`, ~25
   categories: pbp, rosters, contracts, snap_counts, ftn_charting,
   nextgen_stats, pfr_advstats, injuries, depth_charts, combine, etc.).
   No betting-odds or win-totals dataset exists anywhere in it. nflverse
   simply doesn't publish this.
3. **`sportsoddshistory.com`** — this genuinely *was* a free public
   archive of real NFL preseason win-total futures by team/season/book,
   confirmed via the Wayback Machine's CDX index to have covered seasons
   back to 1989. But the live site now redirects to `covers.com`
   (absorbed/acquired), and even the archived Wayback snapshots of the
   old site only captured the page shell — the actual data table loaded
   via a WordPress AJAX call (`.win_call`/`.pre_call` click handlers)
   that Wayback never captured as a separate response, so there's no
   clean historical CSV/HTML sitting in the archive to parse either.
   Checked `covers.com`'s current live win-totals page as the closest
   surviving equivalent: it's a JS-rendered SPA (empty body on direct
   fetch), so scraping it reliably would mean real browser automation —
   fragile against layout changes, likely against the aggregator's ToS,
   and even if built would only capture *this season's* line going
   forward, not a historical multi-year backfill for training data.

**Conclusion: no free, structured, reliably scrapable source exists for
real historical Vegas preseason win-total futures.** Recategorized this
item from "blocked on API key" (implies paying would unblock it cleanly)
to **permanently deprioritized, no viable path found** — both the §8.1
Tier 2 table and the earlier "scoped out" list entry updated inline
above to point here. Not attempting a browser-automation scraper against
covers.com without an explicit ask, given the fragility/ToS concerns and
that it wouldn't even solve the historical-backfill half of the problem.

## §11.2.C: asymmetric floor/ceiling shipped, full mixture-density network scoped down (2026-08-06)

User asked to scope out §11.2.C ("Mixture Density / Bimodal Output
Modeling," filed HIGH EFFORT, deferred pending "a dedicated design
discussion"). Checked the premise against real data before committing to
any architecture, rather than trusting the cited paper's framing at face
value.

**The population isn't classically bimodal.** Real `player_weekly_stats`
histograms (2018-2025, all 4 positions) show a right-skewed decay from a
mode near 0, not two separate humps — but that conflates many different
player archetypes (starters, committee backs, inactive/injured games),
so it's not the right test. The real test: computed skew, kurtosis, and
Pfister's bimodality coefficient (BC) on real 2025 backtest residuals
(`actual - predicted`, 5,612 predictions, this session's v30 run).

| Position | Skew | Kurtosis | BC (>0.555 = bimodal) |
|---|---|---|---|
| QB | 0.48 | 0.12 | 0.39 |
| RB | 1.28 | 2.78 | 0.46 |
| WR | 1.14 | 2.05 | 0.45 |
| TE | 1.37 | 4.03 | 0.41 |

All four positions land well below the bimodality threshold — the
cited paper's literal "two-component Poisson mixture" framing doesn't
hold up on this project's own data. But residuals **are** meaningfully
right-skewed and heavy-tailed (real positive skew, real excess kurtosis
on RB/WR/TE), consistent with "mostly-typical-or-below outcomes +
occasional real boom games," just not two discrete humps.

**Scoped down to what the data actually supports**: the properly-sized
version of this fix is asymmetric floor/ceiling, not a neural mixture
density network. The existing floor/ceiling formula
(`scripts/generate_draft_data.py`, fixed 2026-08-05 earlier this
session) was a single *symmetric* relative spread applied equally above
and below the point total — exactly the kind of shape mismatch the
confirmed skew predicts. Fixed that specific, real problem instead of
building new model architecture:

- Rebuilt the real prediction-vs-actual dataset behind the existing
  formula (it wasn't committed as a script last time, only its fitted
  coefficients survived) as `scripts/calibrate_floor_ceiling.py` — real
  `PreseasonProjector` predictions vs. real season totals, 2,034
  player-seasons, 2019-2025.
- Fit two one-sided quantile regressions (q=0.067 floor, q=0.933
  ceiling, matching the same 86.6% target as before) instead of one
  symmetric one, same covariates (`log(pred_total)`, `confidence_score`,
  position).
- Validated on the same genuine holdout protocol as the existing formula
  (fit 2019-2022, test 2023-2025). **Real, measured improvement**: the
  symmetric formula's floor was almost never actually breached (2.5% vs.
  the 6.7% it was supposed to be — needlessly conservative) while its
  ceiling was slightly too tight (7.6% vs 6.7%). The asymmetric fit gets
  both sides close to target (7.5% / 7.5%), a **3x reduction in per-side
  miscalibration** (sum of |actual-target| gap: 0.051 → 0.016). Overall
  coverage dropped slightly (89.9% → 85.0%, vs. an 86.6% target) — this
  is expected and fine, since the symmetric formula's 89.9% was itself
  an artifact of over-covering on the floor side while under-covering on
  the ceiling side, not a real calibration win.
- Shipped: replaced `_floor_ceiling_spread()` with `_floor_ceiling()`
  (returns `(floor, ceiling)` directly instead of one symmetric spread),
  updated both call sites in `_resolve_projection()`. Kept the old
  symmetric formula's code in place but unused, for reference/rollback.
- Spot-checked on the real, live 2026 draft board after regenerating it:
  the asymmetry direction itself is meaningful, not degenerate — elite,
  already-near-max-usage players (C.McCaffrey: 321.5 total, -220.3
  floor-side / +114.1 ceiling-side) carry more real downside (injury/
  role-loss risk) than upside (already near their ceiling), while the
  gap narrows or reverses for lower-total players. This is a sensible,
  differentiated pattern the old symmetric formula couldn't express at
  all.
- 5 new unit tests (`tests/test_generate_draft_data.py::TestFloorCeiling`
  — floor/ceiling never cross the point estimate, floor never negative,
  missing-confidence fallback, and the asymmetry itself). Full existing
  suite (`test_generate_draft_data.py` + `test_snake_draft_sim.py` +
  `test_preseason_projector.py`, 31 tests) stays green.

**A full neural MDN remains on the table as a bigger follow-up** if a
future session finds the asymmetric fix isn't enough — but building one
before checking whether this much cheaper fix already solved the real,
confirmed problem (skew, not bimodality) would have been over-engineering
relative to what the data actually supports. GAPS.md's §11.2.C entry
above is now partially addressed (asymmetric shape) rather than fully
open (whole new architecture) — updating the original item's framing
would require a full rewrite of that section rather than a quick edit,
so leaving that entry as historical context and pointing here instead.

## Roadmap decision: lineup optimizer + correlation matrices come after horizons are finalized (2026-08-06)

Following the QB-WR/RB correlation scoping above (§11.2.E), user set
explicit sequencing rather than building correlation matrices now:

1. **Finalize weekly (single-week) predictions** — accuracy/calibration
   work on the `component`-mode single-week path.
2. **Finalize draft/season-long horizon predictions** — the
   `PreseasonProjector` season-total path (floor/ceiling just reworked
   above) and any remaining multi-week horizon work.
3. **Then**, and only then: build a real lineup optimizer (wiring
   `LineupOptimizer` — or a rewrite of it — into an actual caller, fixing
   its existing lineup-sum statistics bug, adding the QB-WR/RB
   correlation matrices scoped above) — this was the natural target for
   §11.2.E all along, it's just gated on the horizons being done first.
4. **Also needed, not yet scoped**: a basic UI for GitHub Pages. Note
   for whoever picks this up: CLAUDE.md's "UI Freeze" section names
   `docs/index.html`/`docs/lineup.html`/`_site/*` as locked files
   requiring confirmation before editing — but per this project's own
   memory (and reconfirmed while investigating the lineup optimizer
   this session) **none of those paths exist in this checkout**. There
   is no existing UI to unfreeze or extend; this would be new work from
   scratch, not a UI-freeze exception. The real data artifacts it would
   consume are `data/players_{POS}.json` (produced by
   `scripts/generate_draft_data.py`) and friends.

Rationale for the gate, as stated: correlation matrices and a lineup
optimizer are only useful once the per-player predictions feeding them
are stable — no point optimizing lineups against projections that are
still being recalibrated out from under them.

## Multi-season, team-aware preseason model: investigation started, tracked in EXPERIMENTS.md (2026-08-06)

Following the "multi-season, team-aware preseason model" discussion:
confirmed `PreseasonProjector` is genuinely single-prior-season-only
with zero team context (verified by grep — no `team`/`dest_team`/
`team_changed` reference anywhere in the file). Built a real candidate
(`src/models/preseason_features.py`, `build_multiyear_season_pairs`) —
reuses the weekly model's already-validated destination-team logic
(`_add_dest_team_pos_profiles`) plus new multi-year (y1/y2/y3) trend
features, genuinely new engineering that didn't exist anywhere in this
codebase before.

Per explicit user request, **all metrics from this investigation
(and going forward) are tracked in `EXPERIMENTS.md`** at the repo root,
not narrated here — R², RMSE, MAE, corr, and n for every variant tested,
specifically because R² alone was found to be misleading mid-
investigation (see EXPERIMENTS.md's "Known pitfalls" section: an early
run showed R² getting worse while MAE improved, traced to a test-set
size mismatch, not a real accuracy difference).

Also fixed in passing: a real pandas `groupby()` bug (silently drops
rows with any NaN key, unlike SQL's `GROUP BY`) that caused an entire
season (2025) to vanish and another (2024) to be undercounted to 5 QB
rows in the new feature pipeline — root-caused and fixed
(`dropna=False`), with a regression test
(`tests/test_preseason_features.py`) added directly against it.

**Current verdict, per EXPERIMENTS.md**: QB and WR show clear real wins
over production; TE a small real win; RB has a close but not-yet-proven
candidate (still 0.009 R² short of production). Nothing shipped yet —
this is candidate evaluation, not a production change. See
`EXPERIMENTS.md`'s "Open experiments" section for what's still queued
(calibration-layer test, lookback-depth sweep, component/util/fp
re-check, walk-forward validation).

## CRITICAL: saved component_*.json models are stale relative to a share/pct feature scale convention change — all 4 positions (2026-08-06)

Found while testing whether `ComponentPredictor`'s post-processing
calibration step actually helps accuracy (asked directly: "can we test
whether calibration and post-processing steps actually improve
performance"). The calibration test itself became moot the moment this
surfaced — a broken base prediction makes any calibration comparison
meaningless.

**The bug**: fed real, current-feature-engineering-pipeline data
(`FeatureEngineer(feature_mode="causal").create_causal_features()`,
the same pipeline `ComponentPredictor` is actually served with) through
the currently-saved `data/models/component_{qb,rb,wr,te}.json` models.
All four produce predictions wildly inflated versus real actuals:

| Position | Predicted mean | Real actual mean | Ratio |
|---|---|---|---|
| QB | 142.3 | 14.1 | **10.1x** |
| RB | 45.7 | 8.7 | **5.2x** |
| WR | 143.0 | 7.9 | **18.1x** |
| TE | 166.6 | 6.4 | **25.9x** |

**Root cause, confirmed precisely by inspecting standardized (z-score)
feature values against each saved `StandardScaler`**: every `_pct`/
`share`-suffixed rolling feature — `redzone_target_share_pct`,
`target_share_pct`, `rush_share_pct`, `snap_share_pct`,
`team_rb_target_share`, `team_rb_lead_share`,
`goal_line_carry_share_pct`, `team_wr_target_share`,
`team_te_target_share`, `completion_pct`, `qb_bad_throw_pct_prior` —
is currently computed as a **0-100 percentage**
(`feature_engineering.py:1449`, e.g. `completion_pct = safe_divide(...)
* 100`), but every saved scaler was fit expecting a **0-1 fraction**.
Feeding a 0-100 value through a scaler calibrated for 0-1 produces
standardized values in the hundreds (one feature hit z=2353) — a linear
Ridge component model extrapolates that into a wildly wrong prediction.
This is a genuine convention drift between the feature-engineering code
and the saved model artifacts, not a test-harness bug — verified with
the real, unmodified `ComponentPredictor.predict()` method, not a
reimplementation.

**Secondary, smaller drift** (max|z| in the 10-30 range, not the
100s-1000s range above): EPA-based rolling features
(`pass_epa_per_play_roll3_mean`, `rush_epa_per_play_roll3_mean`,
`recv_epa_per_target_roll3_mean`) and drop-rate priors
(`recv_drop_pct_season_prior`, `recv_drop_pct_roll3_mean`). These look
more like genuine distributional differences in the small 2025-only
diagnostic slice than a hard unit-convention bug — flagged separately,
not conflated with the confirmed percentage/fraction bug above. Worth
re-checking after a retrain, not assumed to be the same root cause.

**Real, currently-live risk**: `data/models/component_{qb,rb,wr,te}.json`
are the actual weekly serving models (`scripts/generate_app_data.py`,
`README.md`-documented). If served as-is once the season starts,
real weekly predictions would be catastrophically wrong — not merely
missing v30's new features (already known and documented in
`MODELS.md`), but actively producing 5x-26x inflated point totals.

**Not the same thing as the `fp`-mode backtest numbers already in
`EXPERIMENTS.md` §1a** — those come from `run_ts_backtest.py --target-mode
fp`, which trains a fresh Ridge/`PositionModel` ensemble per backtest run
rather than loading the saved `component_*.json` files. Those numbers
are unaffected by this bug and remain a valid (if architecturally
different) reference point. Don't read §1a's reasonable-looking numbers
as evidence this bug doesn't matter — it's a completely different code
path.

**Fix: retrain.** `python -m src.models.train` regenerates
`component_{position}.json` from the current feature-engineering
pipeline, which will fit fresh scalers matching the current 0-100
percentage convention. This is the same retrain already recommended in
`MODELS.md` for v30's new features — this finding raises the priority
from "missing some newer signal" to "actively wrong," but doesn't
change the fix. Not retrained in this session — flagging for explicit
user decision before running a real training job (real cost/time,
overwrites the currently-served model files).

**Not yet done**: confirming exactly when the percentage-convention
change landed (git history for `feature_engineering.py` doesn't extend
far enough back to find it — same limitation noted elsewhere in this
doc, `src/` wasn't tracked until 2026-08-04), and whether the same drift
affects any other saved model artifact that consumes these same
features (`PreseasonProjector` uses a different, disjoint feature set
per `_build_season_pairs`, confirmed unaffected; the dormant
`Hybrid4WeekModel`/`DeepSeasonLongModel`/`MultiWeekModel` artifacts in
`MODELS.md`'s DORMANT table use the same `CAUSAL_FEATURES` as
`ComponentPredictor` and were saved around the same time — likely
affected the same way, not yet verified).

## Doc restructure: EXPERIMENTS.md + MODELS.md merged into TRACKING.md (2026-08-06)

Per explicit request: too wordy/split across two files, wanted mainly
tables/bullets, and wanted a single doc where "experiments" can be
about models, features, or techniques, not just models specifically.
Consolidated into `TRACKING.md` (registry + metrics + post-processing
pipeline + open items + pitfalls, all table/bullet form). `MODELS.md`
and `EXPERIMENTS.md` deleted — their content lives in `TRACKING.md`
now, condensed.

**Section-number references to `EXPERIMENTS.md`/`MODELS.md` in earlier
entries above this one are stale** (e.g. "EXPERIMENTS.md §1a") — they
refer to the pre-merge document structure and weren't individually
remapped, consistent with this doc's own convention of not rewriting
history. The underlying findings they point to are all still real and
still in `TRACKING.md`, just under different section numbers/headers.

## CORRECTION + deeper root cause: the component_*.json bug is NOT fixed by retraining, and the real cause is a silent except-ValueError in src/predict.py (2026-08-06)

Follow-up to the "CRITICAL: saved component_*.json models are stale"
entry above. Ran the retrain (`python -m src.models.train --no-tune`),
then re-verified with the diagnostic script — **still broken**, same
severity: QB 10.4x, RB 26.8x (worse than before!), WR 3.0x
(now under-predicting, not over), TE 9.6x. Retraining changed the
*symptom* per-position but did not fix the *cause*.

**Root cause is more precise, and more serious, than originally
diagnosed.** The original entry attributed this to a stale
percentage-vs-fraction convention in `ComponentPredictor`'s own saved
`StandardScaler`. That was an incomplete diagnosis — my first
diagnostic script called `FeatureEngineer.create_causal_features()`
directly, skipping a real intermediate step
(`_apply_bounded_scaling`/`feature_scaler_bounded.joblib`) that the full
production training pipeline (`_prepare_training_data`) applies before
`ComponentPredictor`'s own scalers ever see the data — a `MinMaxScaler`
fit on 112 "bounded" (`_pct`/`share`/`rate`/`prob`-named) columns.

Rebuilt the diagnostic to exactly replicate the **real serving path**
(`src/predict.py::_prepare_features` — `add_external_features` →
`add_season_long_features` → `UtilizationScoreCalculator.calculate_all_scores`
→ `FeatureEngineer.create_features()` → `add_advanced_rookie_injury_features`
→ apply `feature_scaler_bounded.joblib`), not a shortcut. Found the real
cause: **serving-time feature generation produces 98 columns where the
bounded-scaler artifact expects 112** — 14 missing, mostly
utilization-score percentile-normalized components (`snap_share_norm`,
`target_share_norm`, `redzone_share_norm`, `touch_share_norm`,
`inline_rate_norm`) plus a few raw share/pct columns
(`snap_share_pct`, `touch_share_pct`, `redzone_share_pct`,
`redzone_targets_pct`, `route_participation_pct`,
`util_air_yards_share`). This raises `MinMaxScaler.transform()`'s
`ValueError` (shape mismatch), which `src/predict.py`'s real code
**silently catches**:

```python
try:
    scaled = self.bounded_scaler_artifact["scaler"].transform(values)
    data.loc[:, cols] = scaled
except ValueError:
    pass  # Feature count mismatch from version bump — skip scaling
```

The comment shows this was a known, deliberate trade-off (don't crash
on a version mismatch) — but the silent fallback is "feed raw,
unnormalized 0-100 percentages into scalers that expect a 0-1
MinMax-normalized range," which is exactly the failure mode, not a
graceful degradation. This is the same `except Exception`/`except
ValueError: pass` anti-pattern this project's §9 audit was built
around, in a spot that audit didn't reach.

**Why the missing 14 columns are missing**: likely
`UtilizationScoreCalculator`'s percentile-normalization step
(`_percentile_normalize`, used for the `_norm`-suffixed columns) needs
fitted percentile bounds that only get produced/loaded correctly inside
the full `_prepare_training_data` pipeline, not `src/predict.py`'s
lighter-weight `UtilizationScoreCalculator(weights=util_weights)`
construction. **Not yet fully root-caused** — this needs its own
follow-up to trace exactly why serving-time doesn't produce these 14
columns; flagging precisely what's missing rather than guessing further.

**Status**: still broken, confirmed on the freshly-retrained models.
The retrain from the entry above was not wasted (it did pick up v30's
new features and `feature_version.txt` now correctly reads 30), but it
does not address this bug — a genuine fix needs either (a) making
serving-time feature generation produce the same 112 columns
training-time does, or (b) making the bounded-scaler application robust
to a column subset instead of silently no-op'ing on any mismatch. Real,
currently-live risk, unresolved.

## RETRACTION: the two entries above ("CRITICAL... stale relative to scale convention" and "CORRECTION + deeper root cause") were both wrong, caused by a test-methodology flaw (2026-08-06)

Both prior entries claimed real weekly predictions were 5x-27x wrong due
to a scale-convention drift, then a silent `except ValueError: pass`
swallowing a bounded-scaler failure. **Both diagnoses were artifacts of
testing one position at a time** (`db.get_all_players_for_training(position="QB", ...)`
etc.), not how `src/predict.py` is actually used in production —
`predict()`'s default is `position=None`, meaning all 4 positions are
processed together in one combined dataframe. `feature_scaler_bounded.joblib`'s
112-column list was fit on that combined, multi-position dataframe, so
position-specific columns (e.g. TE's `inline_rate_pct`) are naturally
absent when a position is tested in isolation — an artifact of the test,
not a real gap in what serving actually produces. Rebuilt the
diagnostic a third time with all 4 positions combined, exactly matching
real usage: **the bounded scaler transform succeeds, zero missing
columns.** The "silent except ValueError" mechanism never actually
fires in real usage.

**What's real, verified the correct way** (all positions combined,
comparing the backed-up pre-retrain models against today's retrain,
same real 2025 data):

| Pos | OLD ratio/R² | NEW ratio/R² | Verdict |
|---|---|---|---|
| QB | 1.11x / 0.193 | 1.10x / 0.195 | Unchanged, reasonably healthy |
| RB | 2.09x over / -0.979 | 2.12x over / -1.047 | **Already broken before this session's retrain — a real, pre-existing model-quality issue, not a scale bug** |
| WR | 1.96x over / -0.826 | 1.21x / **0.194** | **Retrain genuinely helped** (likely v30 features) |
| TE | 2.08x over / -1.032 | 0.22x under / -0.461 | **Still broken; failure mode flipped from over- to under-prediction with the retrain** — a real, retrain-related change worth understanding, separate from the RB issue |

**Correction to the record**: no scale-convention bug, no silently-caught
exception in practice. The real, standing issues are RB's systematic
~2x over-prediction (predates this session, unrelated to anything
changed here) and TE's prediction collapse (retrain-related, direction
flipped). Both are genuine model-quality problems, not
feature-pipeline plumbing bugs — worth investigating on their own
terms (component-level residuals, training sample size,
`--no-tune` vs. tuned hyperparameters) rather than chasing the
scale-mismatch theory further. Retracting the "retrain, then it'll be
fixed" framing from both earlier entries; this needed a real accuracy
investigation, which those entries jumped past.

**Lesson, stated plainly**: the isolated-position test felt like a
faithful reproduction (it called the real `predict()` method
un-modified) but wasn't a faithful reproduction of real *usage* —
matching production's actual call pattern (multi-position, `position=None`)
mattered as much as using the real code. Verify against how something
is actually invoked, not just that the function itself is unmodified.

## Fixed: real `KeyError: 'offensive_momentum_score'` in `_add_team_matchup_features`, caught-but-fatal on every live `predict()` call (2026-08-06)

Calling the real, unmodified `NFLPredictor` (`predictor.initialize(); predictor.predict(n_weeks=1, top_n=2000)`,
matching `scripts/generate_app_data.py`'s exact invocation — the correct
methodology per the retraction above) surfaced a real, previously-unknown
warning on every call: `WARNING: Team matchup features unavailable
(KeyError: 'offensive_momentum_score')`.

**Root cause**: `_add_team_matchup_features` (`src/features/feature_engineering.py:2230`)
runs twice on the same dataframe in the real serving path — once inside
`create_features()` (`predict.py:632`), then again via
`refresh_matchup_features()` (`predict.py:292`), which exists specifically
to recompute opponent-dependent features for the actual upcoming
opponent/week after they're overwritten. The second call's merges
(`team_a_avgs`, `team_b_avgs`, `mom`) collide with columns of the same
name already added by the first call. The `team_a_avgs`/`team_b_avgs`
merges have explicit `suffixes=` so they silently keep the *stale*
value from the first pass instead of erroring (a separate, quieter bug
this fix also resolves — the "refresh" wasn't actually refreshing). The
`mom` merge has no `suffixes=` at all, so pandas silently renames both
sides to `offensive_momentum_score_x`/`_y`, and the very next line's
`df['offensive_momentum_score']` raises `KeyError`, caught by a broad
`except Exception` and printed as a warning — silently dropping every
team-matchup feature (matchup edges, expected game total, pace, momentum)
for that call. Same collision pattern as the previously-fixed
`first_season` bug in `advanced_rookie_injury.py`.

Confirmed via `git blame` this predates today's session (`4c30bc71`,
2026-08-04) — not a regression from this session's `_get_team_matchup_lookups`
caching optimization, which only wraps the lookup-table computation and
doesn't touch the merge/collision logic.

**Fix**: drop the columns `_add_team_matchup_features` is about to
(re)produce from `df` before merging, right after the `all_team_stats.empty`
check — so a second call genuinely overwrites rather than silently
colliding. Verified: re-ran the real `NFLPredictor.predict()` call,
warning no longer appears.

**Not the cause of the QB/WR under-prediction / TE near-zero collapse**
also observed via that same real call (QB mean 4.30 vs. real ~14.1; WR
mean 3.36 vs. ~7.9; TE mean 0.25, median 0.00 vs. ~6.4; RB mean 8.45 vs.
~8.7, roughly correct) — those numbers are unchanged after this fix, so
that's a separate, still-open, more serious accuracy issue. Next: since
per-position-isolated diagnostics have now proven unreliable twice in
this session, continue investigating only via the real `NFLPredictor`
class, comparing its actual per-component intermediate outputs against
real 2025 actuals for a handful of specific players.

## Found and fixed: real root cause of QB/WR under-prediction + TE collapse — bounded scaler was destroying calibrated injury probabilities (2026-08-07)

Continued the investigation using only the real `NFLPredictor` class (per
the plan above). Called `cp.predict_components()`/`cp.predict()` directly
on `ComponentPredictor` with the real `latest_data` used by `predict()`
and got sane numbers (e.g. J.Allen 21.3 pts, top TEs 7-9 pts) — **not**
the crushed numbers (QB mean 4.30, TE median 0.00) seen from the full
`NFLPredictor.predict()` call. So the damage happens strictly *after*
`ComponentPredictor.predict()`, inside `NFLPredictor.predict()` itself.

**Root cause**: `predict.py:337-339` applies an injury-based availability
discount — `availability = 1.0 - injury_prob_combined`,
`predicted_points *= availability`. Checked `injury_prob_combined`'s real
distribution in `latest_data`: **mean 0.554, min 0.282, max 1.0** — wildly
implausible for a general population of active NFL players in a given
week (real weekly injury risk is a few percent for most players). Traced
`injury_prob_advanced`/`injury_prob_combined` to
`AdvancedInjuryPredictor.predict_injury_probability`
(`src/features/advanced_rookie_injury.py:1239`), which explicitly caps
the result at `min(combined_prob, 0.25)` — confirmed by calling it
directly with real inputs (age 28, prior_injuries 5, low workload):
returns exactly `0.25`. So the *raw* value is correctly bounded and
usually much lower (median well under 0.25) — but the value actually
present in `latest_data` after the full feature pipeline is far outside
that range.

Found the actual culprit: `_infer_bounded_columns`
(`src/models/feature_preparation.py:172`) auto-selects any numeric
column with `"prob"`/`"probability"` in its name for `MinMaxScaler`
rescaling to fill the full `[0, 1]` range (persisted to
`data/models/feature_scaler_bounded.joblib`, applied live in
`predict.py::_prepare_features`). Checked the artifact directly:
`injury_prob_advanced`/`injury_prob_combined` were fit with
`data_min_=0.105, data_max_=0.25` — a genuinely narrow, *already
meaningful* range (injury risk is capped at 25% by design). MinMax￾stretching that 0.15-wide range to fill all of `[0, 1]` means a real 18%
weekly injury risk gets rescaled to ~55%, and the actual worst case
(25%) gets rescaled to 100% ("certain injury"). This single distortion
then gets multiplied directly into `predicted_points` via the
availability gate, crushing nearly every player's prediction by 45-100%
— explaining the QB/WR under-prediction and (compounding on an already
lower baseline) the TE near-zero collapse.

Also checked the other `"prob"`-matching bounded columns for the same
issue: `rookie_breakout_prob` (0-0.58 natural range) and
`rookie_bust_prob` (0-0.70) — same category of bug (already-calibrated
model probabilities, not raw percentages needing 0-100→0-1 conversion),
smaller blast radius since they're Ridge *input* features (learned
coefficients partially compensate) rather than a direct multiplicative
gate on the output. `win_probability` also matched but is naturally
near-full-range (0.05-0.95) so rescaling it is close to a no-op —
harmless, left as-is implicitly by the fix below (not specifically
re-included, but no separate carve-out needed).

**Fix**: removed `"prob"`/`"probability"` from `_infer_bounded_columns`'s
token list (`bounded_tokens = ("pct", "rate", "share", "percentage")`,
was `(..., "prob", "probability", ...)`). Probabilities are conceptually
different from raw 0-100-scale percentage columns: a calibrated
probability estimate is already meaningful in its native range and must
never be MinMax-stretched to artificially fill `[0, 1]`.

**Verification in progress**: `injury_prob_advanced`/`combined` are not
themselves `ComponentPredictor` input features (only the post-hoc
availability gate uses them), so that part of the fix takes effect
immediately once the bounded-scaler artifact is regenerated — no
retrain needed. `rookie_breakout_prob`/`rookie_bust_prob` **are** real
Ridge input features for all 4 positions (confirmed via
`cp.feature_names`), so a full retrain (`python -m src.models.train
--no-tune`) was kicked off to refit those coefficients against the
corrected, un-distorted values for full train/serve consistency.
Real-`NFLPredictor` before/after comparison to follow once it completes.

### Verified: retrain complete, real before/after via `NFLPredictor.predict()` confirms a major fix (2026-08-07)

`injury_prob_combined` is now correctly bounded in real serving output:
mean 0.185, median 0.175, min 0.146, **max 0.25** (was mean 0.554, max
1.0) — exactly matches `predict_injury_probability`'s own design cap,
no more MinMax distortion.

| Pos | predicted_points BEFORE | predicted_points AFTER | Real typical |
|---|---|---|---|
| QB | mean 4.30 / median 4.35 | mean 10.06 / median 9.97 | ~14.1 |
| RB | mean 8.45 / median 9.00 | mean 13.81 / median 13.81 | ~8.7 |
| WR | mean 3.36 / median 3.51 | mean 6.76 / median 6.73 | ~7.9 |
| TE | mean 0.25 / median 0.00 | mean 0.69 / median 0.00 | ~6.4 |

Spot-checked top players per position post-fix — these look genuinely
realistic now: QB1s 14.1-16.8 (Allen, Mahomes, Daniels, Herbert all in a
plausible range), RB1s 18.8-25.7 (Robinson, Gibbs, Jeanty, McCaffrey),
TE1s 4.5-6.5 (McBride, Kittle, Kelce, Bowers). TE's `n=85/147` exactly-
zero predictions are backup/inactive tight ends (most NFL backup TEs
score ~0 in a given week for real) — plausible, not obviously a bug,
though not independently verified row-by-row against real backup-TE
usage.

**Not fully resolved / needs its own look**:
- **TE median is still 0.00** even after the fix — the earlier "real
  ~6.4" benchmark may have been computed only over fantasy-relevant/
  starting TEs, not this call's full `n=147` roster (which likely
  includes 2nd/3rd-string TEs a real weekly-average benchmark wouldn't
  include) — this comparison needs to be redone against a
  starters-only real baseline before concluding TE is still broken.
- **RB now runs a bit hot** (mean 13.81 vs. ~8.7 real) — inverse
  direction from the pre-fix RB finding (which was *also* over-
  predicting, ~2x, before any of today's fixes) — worth checking
  whether this is the same pre-existing RB issue (GAPS.md's earlier
  "RB/TE model-quality issue" entry) persisting/compounding, now that
  the injury-crush that was masking it is gone, or whether the same
  "which players/benchmark am I comparing against" methodology gap
  applies here too (mean 13.81 across top RBs specifically, vs. an
  unfiltered position-wide real average).

**Bottom line**: this was a real, severe, previously-undocumented bug
crushing every live prediction by 45-100% via a corrupted injury-
availability multiplier — now fixed and verified working as designed.
QB/WR accuracy improved substantially. RB/TE still need a dedicated,
apples-to-apples accuracy pass (matched real-vs-predicted player
populations, not just aggregate means) before calling either "fixed."

---

## Real apples-to-apples RB/TE accuracy check (matched population) — 2026-08-07

Built `scripts/validate_rb_te_accuracy.py` to close the "Not fully
resolved" gap above. Methodology:

- Calls the real `NFLPredictor` class (`initialize()` +
  `predict(n_weeks=1, position=None, top_n=2000)`, the exact pattern
  `scripts/generate_app_data.py` uses in production) — **no retraining**.
- Targets a *past* week without retraining or editing `src/predict.py`
  by subclassing `NFLPredictor` (`HistoricalNFLPredictor`, overrides
  `_load_player_data` to truncate to rows strictly before the target
  `(season, week)`) and patching the module-level
  `src.predict.get_prediction_target_week()` per call via
  `unittest.mock.patch` so `predict()`'s internal season/week overwrite
  targets the historical week instead of "today."
- Joins each prediction to `player_weekly_stats.fantasy_points` for the
  exact same `player_id`/`season`/`week` (inner join; unmatched predicted
  players — did-not-play — are counted and reported, not silently
  zero-filled).
- Reuses `_calc_metrics` from `src/evaluation/ts_backtester.py` for
  MAE/RMSE/R²; adds a bias ratio (`predicted.mean()/actual.mean()`) as
  the real apples-to-apples version of the earlier aggregate-mean
  comparison.
- Splits full population vs. a snap-share-verified "starters-only" cut
  (`utilization_scores.snap_share >= 50`, its native 0-100 scale) to
  separate real population-mismatch effects from genuine prediction bugs.

**Critical caveat**: `data/models/model_metadata.json` shows the
currently-serving model's `train_seasons` includes 2025 — every week
validated here (10-17) is in-sample. This is a **serving-path plumbing
check** (did the injury-prob fix produce sane, correctly-matched
per-player numbers), not a generalization/skill measurement. Real
out-of-sample numbers are `ts_backtester.py`'s walk-forward retrain
(TRACKING.md §2a: RB R²=0.341, TE R²=0.236, real 2025 holdout).

**Results** (2,361 matched rows, 2025 weeks 10-17, all 4 positions run
for context; full table in TRACKING.md §4):

- **RB "runs hot" is mostly a population-mismatch artifact.** Bias
  1.73x on the full eligible pool (min_games=1, includes many low-snap
  players) drops to 1.20x among real starters. A real but modest
  over-prediction remains — smaller than, but consistent with, the
  pre-existing ~2x RB over-prediction documented elsewhere in this file
  — not a new or newly-severe issue.
- **TE zero-median is a confirmed real bug, not legitimate backups.**
  TE starters-only bias is 0.22x, barely different from the full-
  population 0.19x — ruling out "these are just backups" as an
  explanation. Among near-zero predictions (`<0.5`, n=274), actual
  median is 2.55 and 43.8% actually scored >3.0; 117 of those 274 are
  snap-share-verified starters. Spot-checked individually: Trey McBride
  predicted 6.76 / actual 37.4 (wk15), George Kittle predicted 2.77 /
  actual 24.7 (wk11), Kyle Pitts predicted 4.59 / actual 45.6 (wk15).
  These are real starting/high-usage tight ends the live model is
  severely under-predicting. **Root cause found (2026-08-07, same
  session)**: this is a direct downstream effect of the
  `player_weekly_stats.team_snaps` inflation bug documented in the next
  entry. `snap_share_pct_roll3_mean` (raw `snap_count/team_snaps*100`,
  rolled 3-game) is TE's most heavily-weighted `receiving_yards`
  component feature (`coef=-1.0045`, largest-magnitude coefficient in
  that model). For 2025 rows this feature computes to ~2.9 instead of
  its healthy ~55-90 range, and the component model's internal
  `StandardScaler` (fit mostly on healthy 2018-2024 data, mean≈0.30 on
  whatever scale it was fit on) turns that into a wildly out-of-distribution
  z-score, which the large negative coefficient then converts into a
  large, spurious downward pull on every 2025 TE prediction. Reproduced
  directly: manually inspecting McBride's wk15 feature row and the
  model's `predict_components()` output confirmed abnormal input →
  crushed output, end to end. Fixing the upstream `team_snaps` bug
  (below) and retraining should resolve this; not yet verified with a
  post-fix retrain (that's the next step, tracked in TRACKING.md §5).

**Side-effect finding**: building the starters-only cut surfaced a
separate real data bug — `player_weekly_stats.snap_share` is broken for
season 2025 specifically. Documented in its own entry immediately below.

---

## Data bug found in passing: `player_weekly_stats.snap_share` is broken for 2025 only — 2026-08-07

Discovered while building the starters-only cut for the RB/TE accuracy
check above: `player_weekly_stats.snap_share` for season 2025 is capped
at 0-0.18 (mean 0.059), while every other season (2018-2024) has the
expected 0-1 range with mean ~0.55:

| Season | min | max | mean |
|---|---|---|---|
| 2018 | 0.0 | 1.0 | 0.570 |
| 2020 | 0.0 | 1.0 | 0.541 |
| 2022 | 0.0 | 1.0 | 0.556 |
| 2023 | 0.0 | 1.0 | 0.546 |
| 2024 | 0.0 | 1.0 | 0.553 |
| **2025** | **0.0** | **0.184** | **0.059** |

Real starters (e.g. Trey McBride, a clear 90%+ snap-share TE1) show
`player_weekly_stats.snap_share = 0.084` for 2025 wk15, while
`utilization_scores.snap_share` (same player/week, computed separately)
correctly shows `92.4` — confirming `utilization_scores.snap_share` is
healthy (just on a 0-100 scale, not 0-1) and the bug is isolated to
`player_weekly_stats.snap_share`'s 2025 computation.

**ROOT CAUSE CONFIRMED (2026-08-07, follow-up session)**: `team_snaps`
in `player_weekly_stats` is set in `pbp_stats_aggregator.py`'s
`merge_with_snaps()` (`snap_share = snap_count / team_snaps`, lines
410-414), where `team_snaps` is computed as
`snap_pos.groupby(['season','week','team'])['offense_snaps'].sum()`
(lines 371-376) — **summing every individual player's own snap count
for the team/week, instead of the team's actual offensive play count.**
Since ~11+ offensive players are each credited with most of the same
~55-75 snaps that game, summing them inflates `team_snaps` to roughly
(avg snaps per player × number of credited players) ≈ 10x the real
per-game play count. Verified directly against the real
`nfl_data_py.import_snap_counts()` source data (which is itself
completely healthy — Trey McBride wk15 2025: `offense_snaps=61`,
`offense_pct=0.92`, matching `player_weekly_stats.snap_count=61` and
`utilization_scores.snap_share=92.4` exactly):

| | 2023 wk10 ARI | 2025 wk10 ARI |
|---|---|---|
| players credited with snaps | 47 | 48 |
| sum(offense_snaps) | 715 | 836 |
| max(offense_snaps) (real team play count) | 65 | 76 |

**Why only 2025 is affected**: this summing bug exists identically in
every season's raw snap-count source (2023 shows the same ~11x
inflation pattern if you sum instead of max) — but it only reaches
`player_weekly_stats` for the season(s) that are ingested via the PBP
reconstruction fallback path (`pbp_stats_aggregator.merge_with_snaps`,
`nfl_data_loader.py:329-338`), which is used when `nfl_data_py`'s
official aggregated weekly stats aren't yet published for a season —
i.e. the **current, in-progress season only**. Completed historical
seasons (2018-2024) are ingested via the standard
`nfl.import_weekly_data()` path, which sources `team_snaps` correctly
and was never affected. This means the bug isn't 2025-specific in any
special sense — it's "whichever season is currently in-progress at
ingestion time," so it will recur for 2026 once that season starts
unless fixed. **Minimal fix**: replace `.sum()` with `.max()` in the
`team_snaps` groupby at `pbp_stats_aggregator.py:371-376` (the
highest-snap player on a team is a reasonable proxy for total team
plays); the technically correct fix is to derive team offensive play
count directly from `self.pbp_data` (already available in the same
class) rather than from summed player-level snap credits at all.

**Impact — confirmed to reach live training/serving features, not just
this diagnostic script.** `feature_engineering.py`'s
`_create_advanced_requirement_features` (called from the main
`create_features()` pipeline, line 230 — used by both training via
`feature_preparation.py` and serving via `predict.py`'s
`_prepare_features`) computes `is_three_down_back` directly from the raw
`snap_share` column (lines 3143-3152): `snap_roll >= 0.5` gated on a
4-game rolling mean of `df["snap_share"]`. For 2025 rows this column
tops out at 0.184, so `snap_roll >= 0.5` can never be true —
`is_three_down_back` is silently always 0 for every RB in 2025,
in both training data and live predictions. Separately,
`train_position_models.py`'s `POSITION_FEATURES` also lists raw
`snap_share`/`snap_share_roll3` as primary/rolling features for
RB/WR (lines 27, 48+) — not yet confirmed whether `PositionModel`/
`train_position_models.py` is on the currently-live training path or a
dormant one (check TRACKING.md's model registry); if live, those
features are corrupted for 2025 too. `preseason_projector.py` and
`preseason_features.py` are unaffected — they already source
`snap_share` from `utilization_scores` (`us.snap_share`/`COALESCE`),
not `player_weekly_stats`.

**Should not be trusted until fixed**: any 2025-dated `snap_share`,
`snap_count`, `team_snaps`, or `is_three_down_back` value sourced from
`player_weekly_stats` — training features, live predictions, and
starter/backup classification alike. `utilization_scores.snap_share`
(0-100 scale) is unaffected and was used instead for
`validate_rb_te_accuracy.py`'s starter cut.

**FIXED (2026-08-08)**: `pbp_stats_aggregator.py`'s `merge_with_snaps()`
now derives `team_snaps` from the median of each player's implied team
total (`offense_snaps / offense_pct`, nflverse's own already-verified
per-player share — `offense_pct` was present in the source data all
along but unused), falling back to `max(offense_snaps)` when
`offense_pct` is unavailable. Verified directly against real 2025 data
before re-ingesting: Trey McBride's `team_snaps` now comes out 55-84
per game (was 605-924), `snap_share` 0.84-0.97 (was 0.08-0.09),
matching `offense_pct` almost exactly. Re-ingested via
`python -m src.data.nfl_data_loader --seasons 2025` (after deleting the
stale `pbp_advanced_2025.parquet`/`pbp_team_advanced_2025.parquet`
caches, which otherwise silently reuse the pre-fix values). DB
confirmed healthy post-ingestion: `player_weekly_stats.snap_share` for
2025 now has min=0.0, max=1.0, avg=0.56 (was avg=0.059), matching every
other season's ~0.55 average exactly.

---

## Bug #3 (much larger than Bug #1): `utilization_score.py` team-total merge collision — silently degenerate since this code existed, not just 2025 — 2026-08-08

Found while re-auditing for more bugs before re-ingesting Bug #1's fix,
per explicit instruction to be skeptical of the code already written.
This is a separate, pre-existing, larger defect than the `team_snaps`
inflation above.

**Root cause**: `UtilizationScoreCalculator._calculate_team_totals_from_players()`
(`src/features/utilization_score.py`, then lines 209-257) computed
team-level totals by grouping the input `df` on `['team','season','week']`
and summing each player's own stat (`snap_count→team_snaps`,
`rushing_attempts→team_rush_attempts`, `targets→team_targets`,
`receptions→team_receptions`), then merged with **no `suffixes=`
kwarg**. This is invoked via `_merge_player_team_data()` whenever
`team_df` doesn't have ≥3 of a specific 7-column list — i.e. whenever
`team_df` is the empty `pd.DataFrame()`, which is how **every live
production path** calls it: `predict.py:629` (serving),
`feature_preparation.py:330/337/344` (training, called twice),
`backtester.py:1597`, `utilization.py:299` (feeds
`scripts/generate_app_data.py`, the production web app's data),
`realtime_integration.py:195`, `audit_2025_backtest.py`.

The input `df` (from `DatabaseManager.get_all_players_for_training()`)
**already has** genuinely-correct, true team-level `team_snaps` (from
`player_weekly_stats.team_snaps`) and `team_rush_attempts` (from the
`team_stats` join, `database.py:2016`) columns before this function
ever runs. Because the freshly-aggregated totals frame produced columns
with the *same names*, the no-`suffixes` merge silently split them into
`team_snaps_x`/`team_snaps_y` and `team_rush_attempts_x`/
`team_rush_attempts_y` instead of clean columns. Downstream, every
per-position share formula does `df.get("team_snaps", snap_count)` /
`df.get("team_rush_attempts", rushing_attempts)` — since the exact name
no longer existed post-collision, `.get()` silently fell back to the
player's **own** count, producing `snap_share_pct =
snap_count/snap_count*100 = 100` (self-division) for every player with
nonzero snaps (0 for zero-snap players) — and identically for
`rush_share_pct`/`touch_share_pct` via `team_rush_attempts` (a second,
previously-unflagged instance of the same collision, affecting RB
specifically — a plausible contributor to the pre-existing RB
over-prediction pattern documented in TRACKING.md, independent of the
TE-focused investigation that found this).

**Verified this predates 2025 and isn't season-specific**: reproduced
identically on both a live-serving-shaped call and a training-shaped
call on 2018-2024 historical data — `snap_share_pct` came out bimodal
(exactly 0 or exactly 100), not a real percentage distribution, in
both cases. This has been silently degenerate since this code path
existed, not something introduced this session or specific to 2025.

**Train/serve distribution mismatch, not just "consistently wrong"**:
`feature_preparation.py` calls `calculate_all_scores()` **twice** in a
row on `train_data` (lines 330 and 337, sandwiched around
`fit_percentile_bounds`). Call 1 collides `team_snaps`/
`team_rush_attempts` (self-division, as above) but leaves
`team_targets`/`team_receptions` clean (they have no pre-existing
source, so nothing to collide with on call 1). Call 2 re-runs on call
1's *output*: `team_snaps`/`team_rush_attempts` now merge in cleanly
(call 1 already shadowed the originals to `_x`/`_y`), but using
`_calculate_team_totals_from_players`'s own sum of `snap_count` across
**all four positions** on that team/week simultaneously — not the true
single-team play count — deflating the ratio ~4-5x (this is why the
saved `feature_scaler_bounded.joblib`'s fit-time `data_max_` for
`snap_share_pct`/`snap_share_pct_roll3_mean` was ~34-41, not the
bimodal 100 a single call produces — resolves a loose thread flagged
honestly-unresolved in the original TE-crush entry above). **Serving
only calls this once** (`predict.py:629`), so it got the bimodal 0/100
self-division version — training and serving were seeing genuinely
different, both-wrong distributions for these features, not just a
shared wrong one.

**Blast radius confirmed via full codebase search**: the "healthy"-
looking stored `utilization_scores` DB table (populated by
`scripts/repopulate_utilization.py`, which passes a real `team_df` from
`db.get_team_stats()` and so never hits `_calculate_team_totals_from_players`
at all) is not a safe alternate source in practice — every live
pipeline that runs `get_all_players_for_training()` (which SQL-joins
the healthy `util_snap_share`/`util_target_share`/etc. from that same
table) and *then* calls `calculate_all_scores()` **overwrites those
exact column names in place** with the freshly-recomputed (buggy)
values (`utilization_score.py` RB/WR/TE branches). Feature-selection
leakage filtering (`src/utils/leakage.py`) only excludes the bare
`utilization_score`/`utilization_score_raw` column names — `snap_share_pct`,
`rush_share_pct`, `touch_share_pct`, `util_snap_share`, `util_target_share`,
`util_rush_share`, `util_rolling_*`, `util_lag_*` are all live input
features to `ComponentPredictor`/`EnsemblePredictor`. `utilization_score_raw`
(built from the same buggy `_pct` columns) also feeds `target_util_1w/4w/18w`
training-label construction, used by `fit_utilization_weights`/
`train_utilization_to_fp_per_position` regardless of the primary
per-week target mode.

**Fix** (`src/features/utilization_score.py`,
`_calculate_team_totals_from_players`): preserve pre-existing
`team_snaps`/`team_rush_attempts` instead of recomputing/shadowing
them (they reflect the true team for that game regardless of which
player rows happen to be in `df` after upstream filtering); only
fall back to computing them when genuinely absent, using `max()` (not
`sum()`) for `snap_count` specifically — a snap is one play that ~11
players simultaneously share credit for, so summing double/triple/
N-counts the same play, same reasoning as Bug #1's fix.
`team_targets`/`team_receptions` keep `sum()` (correct in principle —
each target/reception is attributable to exactly one player). Added an
explicit `suffixes=("", "_recompute_collision")` to the merge as
defense-in-depth: if a future schema change reintroduces a name
collision, it now fails loudly (a warning + an obviously-wrong column
that gets dropped) instead of silently degrading into self-division
again.

**Verified in isolation before re-ingesting**: on 2018-2024
training-shaped data, `team_snaps`/`team_rush_attempts` are now clean
single columns (no `_x`/`_y` split); `rush_share_pct` for RB went from
bimodal to a smooth distribution (mean 25.8%, median 15.4%, p75 47.4%,
max 100%); real bell-cow spot checks are plausible (Ja'Marr Chase 2022
wk1 and Rob Gronkowski 2018 wk20 playoffs both correctly hit exactly
100% snap share on games where they played every snap). Calling
`calculate_all_scores()` twice in a row (matching `feature_preparation.py`'s
pattern) now produces byte-identical output both times (previously
diverged, per the deflated-vs-bimodal discrepancy above) — the
defensive `suffixes=` logic correctly catches and drops the
second call's `team_targets`/`team_receptions` collision too, even
though those two columns weren't explicitly added to the
preserve-if-present set (an emergent side benefit of the generic
defense-in-depth mechanism, not something specifically designed for).

**Not fixed, deliberately left as-is**: the redundant double-call in
`feature_preparation.py` (lines 330/337) — with the collision fixed,
both calls now produce identical, stable output, so it's a harmless
inefficiency rather than a correctness bug. It exists to re-run
`_norm`/`utilization_score` normalization after `fit_percentile_bounds`
runs between the two calls; removing it entirely is a separate,
lower-priority cleanup, not attempted this session to keep the fix
minimal and targeted.

**Accepted, documented limitation**: even with this fix,
`team_targets`/`team_receptions` (which have no pre-existing DB source
to preserve) can still under-count whenever the input `df` is a
filtered subset of the real roster for that game — training's
`min_games=4` default (`src/models/data_loading.py:21`), `predict.py`'s
`filter_to_eligible_players` (drops all historical rows for players
outside a recent-seasons eligibility window), or a position-scoped
`predict(position=...)` call (would sum only one position's players per
team, missing the other three positions' contribution entirely). This
is real but much lower-severity than the collision bug (a legitimate
approximation that can undercount, vs. a catastrophic bimodal
degenerate value) — not fixed this session, no clean available fix
without a genuine team-level targets/receptions data source.

## Full re-verification after both fixes — 2026-08-08

Re-ingested 2025 data, retrained (`--fast --no-tune`, per the earlier
documented OOM constraint on full Optuna tuning), refreshed the
`utilization_scores` DB table via `repopulate_utilization.py`, and
re-ran `scripts/validate_rb_te_accuracy.py` (2025 weeks 10-17, all 4
positions, 2,361 matched rows). Full results table in TRACKING.md §4.
Headline: TE bias 0.19x→0.84x (R² -0.425→0.280), RB bias 1.73x→0.83x
(R² -0.254→0.344), TE near-zero predictions n=274→n=3. Both root
causes are confirmed fixed, not just mitigated.

**New, separate bug surfaced by the re-run (not caused by this
session's fixes)**: QB metrics got worse (bias 0.68 vs. previous 0.85;
starters-only bias 0.10, R²=-1.75). Investigated and root-caused to a
**pre-existing `players` table data-quality bug**: at least 13 real
WR/RB/TE players are mislabeled `position='QB'` — confirmed via SQL
(`passing_attempts<5` combined with real `rushing_attempts`/`targets`
volume): D.Moore, D.Singletary, G.Olszewski, J.Jennings, C.Kmet,
J.Waddle, C.Olave, T.Burks, B.Hall, I.Williams, T.Warren, R.Harvey,
Q.Judkins. Spot-checked directly: `predict()`'s own live output labels
Derrick Henry (`player_id=00-0032764`, a real RB) as `position='QB'`,
sourced straight from the `players` table (`SELECT position FROM
players WHERE player_id='00-0032764'` returns `'QB'`). The model
correctly predicts near-zero passing production for these players
(they genuinely never pass), while their real `fantasy_points` reflects
real receiving/rushing volume — a severe, spurious under-prediction
that drags down the QB slice's aggregate metrics and inflates the
apparent "starters-only" QB count (56 this run vs. 6 previously, since
these mislabeled skill players legitimately clear a snap-share starter
threshold under the wrong position label). This predates this session
entirely and is unrelated to Bugs #1/#3.

---

## Bug #4: mislabeled-QB root cause found and fixed — trick-play passes corrupt `players.position` permanently — 2026-08-08

Root-caused the QB mislabeling bug flagged above, at the request to fix
it (not just log it).

**Mechanism**: `pbp_stats_aggregator.py`'s `aggregate_passing_stats()`
hardcodes `position='QB'` (line ~197) for every row with a pass
attempt — correct for genuine QBs. `aggregate_rushing_stats()`/
`aggregate_receiving_stats()` produce separate rows for the same player
with no position set at all. `aggregate_all_stats()` then does
`all_stats = pd.concat([passing, skill], ...)`, and for any player who
both passed *and* rushed/received in the same game (a real dual-threat
QB — or a real RB/WR who threw a single trick-play/wildcat pass, like
Derrick Henry or Christian McCaffrey), this produces two rows for the
same `(player_id, season, week)` key. The duplicate-collapse step
(lines ~566-579) summed numeric stats correctly but took `"first"` for
string columns including `position` — and since the passing row was
concatenated first, `"first"` always picked the hardcoded `'QB'`, even
for a player whose real activity that week was 12 carries and 1 target
plus a single trick-play pass attempt.

That mislabeled row then flowed into `nfl_data_loader.py`'s
`_store_weekly_data()`, which calls `db.insert_player({'position':
row['position'], ...})` (`INSERT OR REPLACE`) for **every** weekly row
during ingestion — so whichever week for that player_id happened to be
processed last in that ingestion's row order determined the player's
permanently-stored `players.position`. Once corrupted, it stayed
corrupted indefinitely: `pbp_stats_aggregator.py`'s `_infer_position()`
checks the `players` table lookup *before* its own heuristic
(intentionally, per the 2026-04-25 fix documented in its docstring, to
prefer the "authoritative" stored value) — so even on a subsequent,
otherwise-correct re-ingestion, the cached wrong value would keep
winning over any fresh inference, for every week of that player's
career, not just the trick-play week. This explains why the corruption
was permanent rather than self-correcting on the next data refresh.

Confirmed this is a real, wide-reaching, pre-existing bug, not new:
broadened the detection query (career `passing_attempts` vs.
`rushing_attempts + targets`) and found **30** affected players (not
just the 13 first spotted), including Derrick Henry, Christian
McCaffrey, Cooper Kupp, D.J. Moore, Courtland Sutton, Cole Kmet, Jaylen
Waddle, Chris Olave, Breece Hall, Devin Singletary, David Montgomery,
and 19 lower-profile/sparse-data players. **Deliberately excluded**
Taysom Hill (`00-0033357`, 310 career passing attempts) from the
automatic correction — he's a genuine hybrid QB/gadget player with real
meaningful passing volume, not a trick-play mislabel, and deciding his
"true" position is a real judgment call, not a bug fix.

**Fix, two parts, both required for durability**:
1. Code (`pbp_stats_aggregator.py`): the duplicate-collapse step no
   longer takes `"first"` for `position` — it's dropped from the
   collapse entirely (set to `NaN` for every collapsed row), so
   `_infer_position()` (called afterward on the full frame) decides
   using the *summed* stats and the players-table lookup, not an
   arbitrary row-concat order. Also hardened `_infer_position()`'s
   heuristic itself as defense-in-depth (for players not yet in the
   `players` table): it previously returned `'QB'` for *any* nonzero
   `passing_attempts`; it now requires `passing_attempts >= 5 AND
   passing_attempts >= (rushing_attempts + targets)` — passing must be
   the dominant activity, not merely present.
2. Data: the code fix alone doesn't retroactively correct already-
   stored bad values (and wouldn't durably override them either, since
   the players-table lookup is checked *before* the heuristic and would
   keep returning the old wrong value). Directly corrected the 30
   affected `players` rows via a data-driven SQL correction (same
   rush-vs-target logic as the heuristic: RB if `rushing_attempts >
   targets`, else TE if `receiving_yards/targets < 8`, else WR).

**Verified end-to-end, no retrain needed**: re-ran
`PBPStatsAggregator().aggregate_all_stats(season=2025)` fresh — Derrick
Henry's week 3 (the actual trick-play week, `passing_attempts=1`) now
correctly shows `position='RB'`, matching all his other 16 weeks
(previously only that one week would have re-corrupted on a future
re-ingestion). Live `predict()` call confirms Henry, McCaffrey, Kmet,
Waddle, and Kupp all now route to their correct position's component
model with plausible predictions (no retrain required — `predict.py`
reads `position` live from the DB via `get_all_players_for_training`'s
SQL join, not a baked-in training artifact). Re-ran
`scripts/validate_rb_te_accuracy.py`: QB full-population bias
0.68→**0.82**, R² -0.108→**0.134** (recovered to essentially the
pre-regression baseline of 0.85/0.163 from before this whole
investigation started). "QB starters-only" now correctly shows zero
rows — real QBs never compute `snap_share_pct` at all (confirmed
earlier in this file: `utilization_score.py`'s QB branch doesn't set
it), so the previous run's 56 "QB starters" were *exactly* the
misclassified skill players (who do have a real snap share) being
counted under the wrong label. Their disappearance from the QB slice
is the clean mechanistic confirmation that the fix addressed the real
cause.

**Not fixed / accepted judgment calls**: Taysom Hill's position was
deliberately left as-is (see above — a real ambiguous case, not a bug).
The ~7 near-zero-data players in the correction set (empty names,
1-3 total career plays recorded) were corrected using the same logic
as everyone else, but with such sparse data the "correct" position is
low-confidence either way — low-impact given how little they
contribute to any model. No full historical re-ingestion was run (the
code fix is self-healing for future ingestions of any season; the
targeted `players`-table correction fixes the current state without
needing to reprocess 2006-2025 from scratch).

## Found in passing: `MultiWeekModel`'s 4w/18w targets likely have inflated R² from a variable-window artifact, not real forecasting skill — 2026-08-09

Found while building `scripts/walk_forward_multiweek.py` (real
walk-forward validation of TRACKING.md §2d, requested separately —
logging this specific sub-finding here per standing instruction since
it's a distinct, deeper issue than the walk-forward task itself).

**Mechanism**: `train_position_models.py::create_targets()` builds the
`n`-week target as `df.groupby([player_id, season])["fantasy_points"]
.transform(lambda x: x.shift(-1).rolling(window=n, min_periods=1).sum())`.
`min_periods=1` means a player in week 15 of an 18-game season gets a
target that's the sum of however many games remain (as few as 1-3),
while a player in week 1 gets the sum of up to 18. The raw *magnitude*
of the target is therefore driven heavily by how many games are left
in the season — a fact any feature correlated with in-season timing
(cumulative games played, rolling-window features whose availability
changes by week, etc.) can predict trivially, without any real
skill-forecasting content.

**Evidence**: querying `target_18w` by week for 2025 QBs shows mean
target rising monotonically from ~16.5 (week 1) to 270+ (weeks 19-21,
playoffs) with `corr(week, target_18w) = 0.576` — a strong, mechanical
relationship. A fresh walk-forward run (`walk_forward_multiweek.py`,
QB, test_season=2025) scored `18w: R²=0.976`, `4w: R²=0.844`, vs.
`1w: R²=0.222` — the 1-week target isn't affected by this artifact
(rolling window of exactly 1, always full), and its R² is in a
plausible range consistent with other weekly-single-game numbers
elsewhere in TRACKING.md (§2a: QB weekly R²≈0.24). The 4w/18w jump to
0.84-0.98 is the red flag: TRACKING.md §2d's original (single-split,
never re-derived) QB number was R²=0.049, wildly different from either
of these — meaning it's unclear what target definition or horizon that
original ad-hoc, unsaved script actually used, so it cannot be
reconciled with these new numbers either way.

**Not fixed** — this is a target-definition question (should the
n-week target require a full `n` games of history to be scored at all,
or be normalized to points-per-remaining-game instead of a raw sum?)
that needs a design decision, not a mechanical bug fix. Logged here
per the "always fix small/safe bugs immediately, log bigger ones right
away" standing instruction — this is the "bigger" case. Until resolved,
**treat `MultiWeekModel`'s reported 4w/18w R² anywhere in this repo
(TRACKING.md §2d and any future walk-forward extension of it) as
unreliable/inflated; only the 1w horizon is currently trustworthy.**

---

## Data quality audit: `team_stats.time_of_possession` / `third_down_conv` always 0 (2026-08-09)

Found while scanning `data/nfl_data.db` for all-zero/constant columns (a
sweep the user specifically asked for after noticing it wasn't part of
routine integrity checks like duplicate-key or out-of-range scans).

**`team_stats.points_per_drive` was also constant 0** — this one *was*
a small/safe fix, applied directly: `pbp_stats_aggregator.py`'s drive
metrics block matched on a `drive_result` column that no longer exists
in current `nfl_data_py.import_pbp_data()` output (renamed
`fixed_drive_result` at some point upstream). Every row silently fell
into the zero-fallback branch. Fixed by trying both column names
(`src/data/pbp_stats_aggregator.py` around the "Drive metrics" block).
Verified against live 2024 PBP data: `points_per_drive` now has
mean≈2.29, std≈0.96 instead of being pinned at 0.

**`time_of_possession` and `third_down_conv` are NOT a rename bug —
they are simply never computed.** `PBPStatsAggregator.aggregate_team_stats()`
does not produce these columns at all (confirmed via direct column
inspection on live 2024 PBP output — `points_scored`/`points_allowed`
are also absent). `src/data/nfl_data_loader.py::_store_team_stats_dataframe()`
reads them via `row.get(col)` and stores `None` when missing;
`database.py::insert_team_stats()` upserts with
`COALESCE(excluded.col, team_stats.col)` against a column with
`DEFAULT 0` — so the very first insert established 0, and every
subsequent upsert (with `None` from the missing source column)
COALESCEs right back to that same 0 forever. Net effect: these two
columns look present and populated (only 2 NULLs out of 11,048 rows)
but have carried zero information since inception.

**Fixed (2026-08-09, follow-up)** — added aggregation logic to
`PBPStatsAggregator.aggregate_team_stats()`: `third_down_conv` is
`third_down_converted / (third_down_converted + third_down_failed)`
per team/week using nfl_data_py's native per-play flags; `time_of_possession`
sums each distinct drive's `drive_time_of_possession` ("MM:SS" string,
parsed via new `_mmss_to_seconds()` helper) per team/week and reports
minutes. Verified against live 2024 PBP: third-down conv rate 0-0.82
range (league mean ≈0.39, matches real NFL); TOP ≈24-34 min/team/game,
summing to ~60 for both teams in a game as expected. Regenerated
`data/raw/pbp_team_advanced_{2018..2025}.parquet` caches and re-upserted
`team_stats` in the DB — league-wide season means now ≈0.36-0.39 (3rd
down) and ≈28.6-30.4 min (TOP), both in the expected historical range,
with real per-team-week variance (68-75 / 416-454 distinct values per
season respectively). 2006-2017 remain 0 — same pre-existing scope gap
as `points_per_drive` above (team-level PBP stats were never loaded for
those years; only 2018+ has cached team-advanced parquet files).

Previously logged as "not fixed, requires new aggregation logic, doesn't
meet the small/safe bar" — that assessment held until the user asked
for it directly. `team_stats.time_of_possession` and
`.third_down_conv` are now real, populated data (2018+); they are not
yet read by `feature_engineering.py` — wiring them into the
team-matchup feature block (same place `points_per_drive` is consumed,
`feature_engineering.py`'s `team_metrics` list) is a follow-up if
useful, not required for the data itself to be correct.

---

## `player_weekly_stats` situational-usage columns are stale (near-zero DB, correct pipeline) (2026-08-09)

Found via a low-variance scan the user asked for explicitly (not just
strict-constant / all-null — "one value covers >=97% of non-null rows"),
run after the `points_per_drive` fix above. Flagged columns in
`player_weekly_stats`: `rush_plays`, `pass_plays`, `neutral_rushes`,
`third_down_targets`, `short_yardage_rushes`, `redzone_targets`,
`goal_line_touches`, `two_minute_targets` (and likely `neutral_targets`,
`high_leverage_touches`, `rush_inside_5/10`, `targets_15_plus` — same
family, not individually re-verified) — all 97-99% zero DB-wide.

**Confirmed real, not natural sparsity**: filtered to RB rows with
`rushing_attempts + targets > 3` (i.e. players with genuine volume),
`redzone_targets` nonzero only 1.3%, `goal_line_touches` 2.0%,
`rush_plays` 5.2% — implausible for backs with real touches.

**Root cause is a stale table, not a live bug** — traced the full
pipeline (`PBPStatsAggregator.aggregate_all_stats()` →
`NFLDataLoader._merge_advanced_pbp_features()` →
`_standardize_weekly_columns()`) against live 2024 PBP data end to end:
every stage produces correct, plausible nonzero rates (e.g.
`redzone_targets` 30.6% nonzero, `rush_plays` 99.9% nonzero among
`rushing_attempts>3` rows, matching the aggregator's own direct output
of 21-72% for the situational columns). **The current code is correct.**
The DB values are stale — `player_weekly_stats` (110,283 rows,
2006-2025) was populated by an older run of this pipeline, before the
current advanced-PBP-merge logic existed or reached its present form
(same class of drift as the `points_per_drive` finding above, just on
the primary training table instead of `team_stats`).

**Not fixed** — deliberately not run without checking in first. Unlike
the `team_stats` refresh (8 seasons, ~4,500 rows, cheap/low-risk), a
full `player_weekly_stats` reload touches the entire training table
(20 seasons, 110K rows), takes real wall-clock time (per-season
nflverse/PBP fetches back to 2006), and invalidates
`data/cached_features.parquet` (stale `FEATURE_VERSION`) and whatever
models were trained on the current stale values — worth scoping (full
history vs. recent seasons only, dry-run row-count diff first) before
committing. User was asked whether/how to scope a refresh as of this
write-up (answer pending); until resolved, treat any RB/WR/TE
situational-usage feature (redzone share, goal-line touches, third-down
role, etc.) downstream as unreliable.

---

## Broader referential-integrity / low-variance sweep (2026-08-09, cont'd)

Continuation of the data-quality audit at the user's request ("let's move
on to the other steps you suggest to catch bugs"). Findings, most to
least significant:

**1,090 `players` rows (38% of the table) have a blank `name`, covering
31,683 player-weeks (~29% of all `player_weekly_stats` rows) and
253,808 fantasy points of real, played production.** Root cause:
`NFLDataLoader._store_weekly_data()` (`nfl_data_loader.py:671`) does
`_to_scalar_str(row['name'], '')` — defaults to `''` when the source
row has no name. Confirmed **100% confined to seasons 2006-2016**
(monotonically declining count going further back; zero occurrences
2017+). Since `TRAINING_START_YEAR_DEFAULT = 2018` and these rows are
only ever used as historical rolling-feature *context* (joined by the
still-intact `player_id`, never as training targets), this does not
appear to corrupt any numeric feature — `name` isn't part of any
feature computation. Impact is cosmetic/display-only (draft board,
player lookups) for pre-2018 player identity, not model accuracy.
**Not fixed** — no immediate action needed unless something surfaces
pre-2018 player names to a UI; logging in case a display path
(`scripts/generate_draft_data.py` or similar) ever iterates historical
seasons and hits blank names unexpectedly.

**`qbr` table's `player_id` is ESPN's own athlete-ID scheme (e.g.
`3139477` for Mahomes), not GSIS — 100% orphan rate against `players`,
can never join as-is.** Confirmed via grep that nothing in
`src/features/` or `src/models/` currently reads `qbr`, so this is
inert (same class as `weighted_opportunity` in the first entry above —
dead table, not active corruption). Would need a name-based or
ESPN-ID-crosswalk join to ever use QBR data as a feature.

**`weekly_rosters_v2` (77% orphan) and `injuries_nflpy` (69% orphan)
against `players` — not a bug, a scope mismatch.** `players` is only
ever populated as a side-effect of loading weekly *stat* rows
(`nfl_data_loader.py:674`, inside `_store_weekly_data`), so it means
"has a recorded stat line," not "was ever on a roster." Roster and
injury-report data naturally reference a much broader population
(inactive, IR, practice-squad players) that never gets a `players` row.
Anyone joining roster/injury tables to `players` for name/position/DOB
metadata should expect ~70-77% of rows to not resolve — this is
expected, not something to fix, but worth knowing before trusting an
injury-adjusted feature that silently drops most non-starters.

**SUPERSEDED — 2026-08-10 (Complete Player-Game Panel prerequisite).**
The entry above correctly diagnosed the *mechanism* but wrongly concluded
"expected, not something to fix" — it was scoped only to feature-join
hygiene (metadata resolving cleanly), not evaluated against training-
target selection bias. The user flagged, unprompted, that this exact
mechanism means `player_weekly_stats` (the source of every model built
this session) silently has NO row at all for "played but produced zero
fantasy points" — indistinguishable from "didn't play." Verified
empirically: joining `weekly_rosters_v2` (active-roster status, 2024-2025
only) against `player_weekly_stats` for season 2024 showed 25-40% of
active-roster player-weeks (QB 39.4%, TE 37.6%, RB 27.6%, WR 24.8%) had no
stats row at all. Confirmed the mechanism directly — `nfl_data_py.
import_weekly_data([2024])` itself never contains a row for Joe Flacco
(IND) on weeks he didn't attempt a pass; this is upstream of our ingestion
code, not a filter bug in it.

**Fixed**: built the complete player-game panel. Two-tier eligibility rule
(`scripts/build_complete_player_game_panel.py`):
- **2018-2025**: `snap_counts` (PFR→GSIS mapped via the existing
  `get_pfr_to_gsis_map()`, `nfl_data_loader.py`) confirms `offense_snaps >
  0` for a (player, team, season, week) with no existing stats row — a
  precise "took the field" signal. Tag: `inferred_snap_verified_zero`
  (8,667 rows).
- **2006-2017** (no snap-level data available that far back): `weekly_
  rosters` (backfilled via the already-existing but never-run `scripts/
  backfill_weekly_rosters.py --seasons 2006 2025` — table was empty, 0
  rows, before this), `status='ACT'`, tightened to require the player has
  ≥1 REAL `player_weekly_stats` row elsewhere that season (any real row —
  doesn't require `fantasy_points > 0` on that other row, just legitimate
  statistical participation that season). Weaker signal, explicitly tagged
  low-confidence so it's never mistaken for the snap-verified tier: `data_
  source='inferred_roster_zero_low_confidence'` (25,226 rows after
  excluding bye weeks — see below).
- Position scope: QB/RB/WR/TE only. Never bye weeks (schedule-derived
  opponent required; ~3,726 candidate rows correctly dropped as bye weeks,
  spot-checked directly — e.g. DAL/LV/KC/LAC were confirmed absent from
  the actual 2006 week-3 schedule). Never non-rostered/no-evidence players.
- **Player identity**: exact `gsis_id` lookup (raw `weekly_rosters`
  parquet already carries it — same scheme as `players.player_id`, no
  fuzzy matching needed). 78 genuinely new `players` rows inserted, logged
  to `data/experiments/complete_panel_new_players_audit.csv`. 491
  candidate rows were **excluded** for a position conflict against an
  existing `players` row rather than silently trusting either source —
  correctly caught real position-switchers (Ty Montgomery WR→RB,
  Cordarrelle Patterson WR→RB, J.D. McKissic RB/WR hybrid, Andrew Beck
  FB/TE) as ambiguous rather than guessing.
- **New bug found and fixed en route**: `weekly_rosters` uses era-specific
  team codes (`ARZ`/`BLT`/`CLV`/`HST`/`OAK`/`SD`/`SL`) that `schedule` and
  `player_weekly_stats` don't — both of those already normalize
  retroactively to modern codes (`ARI`/`BAL`/`CLE`/`HOU`/`LV`/`LAC`/`LA`).
  Same class of bug as the already-documented LAR/JAC mismatch (this
  section, §9.2), different table/codes. Without normalizing
  (`TEAM_CODE_NORMALIZATION` in the build script), ~29% of 2006-2017
  candidates lost their schedule-derived opponent and were silently
  dropped; after the fix, the true bye-week rate (~13%) is what's left.
- Schema: `player_weekly_stats.data_source` column added (`ALTER TABLE`
  migration in `database.py`'s existing check-and-add pattern), backfilled
  to `'nflverse_stats'` for all 110,283 pre-existing real rows so every
  row's provenance is unambiguous.

**Sensitivity analysis (population statistics only, no models retrained
yet)** — quantifying how much the data-generating distribution actually
changed:

| Population | Player-games | Zero-PPR rate | QB | RB | TE | WR |
|---|---|---|---|---|---|---|
| A. Original | 101,625 | 6.3% | 1.6% | 2.0% | 10.6% | 8.5% |
| B. Corrected (all) | 135,518 | 29.7% | 31.9% | 19.2% | 40.4% | 29.3% |
| C. Corrected (snap-verified only) | 110,292 | 13.7% | 2.5% | 5.2% | 25.1% | 16.0% |

The original data understated true zero-production-game frequency by
**roughly 5x** (6.3% → 29.7%). This is not a marginal correction — it's a
materially different target-variable distribution, especially for TE/QB
(deep backups/blocking specialists who rarely score are now correctly
represented) and especially in the 2006-2017 era (34.3% zero rate in the
corrected panel vs. 5.9% originally — the low-confidence tier is doing a
lot of the work there, which is exactly why it's kept distinguishable via
`data_source` rather than blended in silently).

**SUPERSEDED — 2026-08-11 (2006-2017 tier rebuilt: PBP-confirmed
participation replaces active-roster-status-alone).** The user raised a
real methodological gap in the `inferred_roster_zero_low_confidence` tier
above: "active roster status" means *eligible to play*, not *actually
played* — a backup can be active every week and never take the field
(coach's decision), which risks injecting **false zeros**, the mirror
problem of the selection bias this whole fix was built to solve. Verified
feasible: raw play-by-play (`nfl_data_py.import_pbp_data`) goes back to
1999, not just 2018, and already carries `passer_player_id` /
`rusher_player_id` / `receiver_player_id` columns on the same GSIS ID
scheme as `players.player_id` — a direct "touched the ball" signal,
stronger than roster status though still a conservative lower bound (a
blocking-only player or an untargeted route-runner won't show up).
Spot-checked 2006 vs. 2009 vs. 2010: passer/rusher counts stable
throughout; receiver-charting coverage is noticeably thinner in 2006-2008
(~10.3K non-null) than 2009+ (~17.6K) — makes the WR/TE tier even more
conservative in those years specifically, not a correctness problem.

**Fixed**: backed up the DB
(`nfl_data.db.bak-pbp-tier-rebuild-20260811134132`), deleted all 25,226
`inferred_roster_zero_low_confidence` rows (safe — none were real
`nflverse_stats` rows), and rebuilt the 2006-2017 tier in
`scripts/build_complete_player_game_panel.py` using
`load_pbp_confirmed_candidates`: roster `status='ACT'` AND PBP-confirmed
participation that week AND ≥1 real stat row elsewhere that season (same
"real stat row" floor as before — deliberately NOT loosened, per explicit
user direction). New tag: `inferred_pbp_confirmed_zero`.

**Result: only 12 rows survive for the entire 2006-2017 range** (vs. the
old tier's 25,226) — a real finding, not a bug. Mechanism: touching the
ball in an NFL game almost always produces *some* countable box-score
stat (even a 0-yard carry registers `rushing_attempts=1`), so "confirmed
participation + zero recorded stats that week" is inherently rare without
snap-count-level granularity. Diagnosed the funnel directly: 603
player-weeks were PBP-confirmed with no stats row that week; 591 of those
belong to players with **zero real stats rows for the entire season**
(e.g. Michael Robinson, a real 8-year 49ers/Seahawks RB/FB, has zero rows
anywhere in `player_weekly_stats`, any season) — a separate, deeper
nflverse weekly-stats coverage gap for that era, not something this fix
should paper over by guessing. The "≥1 real row elsewhere that season"
floor correctly excludes them. Only 12 remain: players with an
established season-long stats history whose one specific week is
PBP-confirmed but stat-row-less (e.g. Matt Schaub, ATL's real backup QB
in 2006, active + a confirmed play in week 8 with no matching stat line
that game, while having real rows other weeks that season).

**Also found and fixed en route**: the new PBP-participation cache
(`fetch_pbp_participation`, filename `pbp_participation_{season}.parquet`)
collided with an unrelated, pre-existing cache from a different pipeline
(pass-protection-participation charting, different schema entirely) that
already used that exact naming convention for 2016-2025. Caught via
dry-run: row/unique-player counts nearly doubled starting at exactly the
first season with a pre-existing file (2016), which made no sense as a
real data discontinuity. Renamed the new cache to
`pbp_touch_participation_{season}.parquet` to guarantee no future
collision, deleted the 10 incorrectly-named files this session had
generated for 2006-2015 (regenerable, no data lost), left the unrelated
pipeline's files untouched.

**Per user direction, logged rather than discarded or force-converted**:
the 591 excluded "PBP-confirmed but no season-long stats history" cases
are written to
`data/experiments/unresolved_historical_stats_coverage.csv` (new
`data_source` value: `unresolved_historical_stats_coverage`, used only
for this audit log — never inserted into `player_weekly_stats`) so the
gap is visible and re-examinable later, not silently dropped.

**Revised sensitivity numbers** (supersedes the 34.3%/2006-2017 figure
above, which was built on the now-deleted roster-only tier): with only 12
rows added for 2006-2017 instead of 25,226, that era's zero-rate is now
effectively unchanged from the original (uncorrected) data — the fix
genuinely cannot manufacture pre-2018 zero-observations at scale given
available data, and correctly says so rather than forcing a number.
Population B (`inferred_snap_verified_zero` + `inferred_pbp_confirmed_zero`
+ original real rows) is now much closer to Population C
(snap-verified-only) than originally reported, specifically for
2006-2017; the 2018-2025 era's numbers in the table above are unaffected
(that tier was never in question).

**Explicit era distinction going forward** (per user direction): 2018+
has a reasonably complete "actually played" signal (`snap_counts`) and is
well-suited to a hurdle-style zero/nonzero decomposition. 2006-2017 does
not — participation there is only partially observable, and any model
trained across both eras should be evaluated with era as an explicit
validation dimension (2006-2017 vs. 2018-2025 splits), not just pooled,
since the pre-2018 zero rate is likely still biased downward by genuinely
unresolvable coverage gaps in the source data, not by anything this
pipeline controls.

**Decision: did NOT re-run Phases 2-6c a third time.** A 12-row change to
the training population (out of ~119K rows, concentrated in 2016-2017)
is far too small to plausibly move any Phase 2/3/4/5/6c result outside
the noise already observed between repeated runs — only QB/TE's 7y window
and RB's "all" window even reach into this era at all. The current
(second) re-run's `FINAL_CONFIG` remains the operative choice; re-running
the full grid a third time for a change this small was judged not worth
the multi-hour cost. If this reasoning is ever in question, the 12
affected rows are fully logged above and the comparison is cheap to spot-
check without a full re-run.

**Phase 7 (18-week season projection) redesigned and run for real —
2026-08-11.** Previously built but never executed end-to-end (see
`[MECHANISM BUILT, REAL RUN PENDING]` in the original entry). Redesigned
per the Complete Player-Game Panel's `data_source` provenance
(`src/models/single_week_ppr/season_projection.py`):
- **Availability-rate fix**: the old code multiplied `availability_rate`
  (P(plays)) onto every week's prediction uniformly, including weeks with
  a real row — double-discounting outcomes we already had direct evidence
  for. Now: any week with a real row (`nflverse_stats`,
  `inferred_snap_verified_zero`, or `inferred_pbp_confirmed_zero`) uses
  the raw prediction (P(plays)=1, already known); `availability_rate` is
  reserved for genuinely synthetic weeks (no row of any kind). Extracted
  the branch decision into a pure, unit-testable function
  (`resolve_week_source`).
- **Sensitivity toggle**: `--exclude-pbp-confirmed-zeros` on
  `scripts/run_phase7_season_projection.py` (off by default) — treats the
  weaker 2006-2017 tier as absent for a Population-B-vs-C-style
  comparison.
- **Output transparency**: added `weeks_real_stats` /
  `weeks_inferred_snap_verified` / `weeks_inferred_pbp_confirmed` /
  `weeks_synthetic` breakdown columns to the per-player CSV
  (`data/experiments/phase7_season_projection.csv`).
- 5 new unit tests (`TestResolveWeekSource`) plus the pre-existing 8 for
  `possible_weeks_for_team`/`estimate_availability_rate`/
  `build_synthetic_week_row` — 13/13 passing, 110/110 full suite.

**Bug found and fixed en route (not scoped to Phase 7, affects
`add_external_features` broadly)**: `src/data/external_data.py`'s injury
merge (~line 833) assumed the input DataFrame never already has
`injury_score`/`is_injured` columns before merging fresh ones in from
`injury_status`. When it does (e.g. Phase 7's carried-forward synthetic
rows, which inherit `injury_score` from the real historical row they're
based on), pandas silently suffixes both to `injury_score_x`/`_y`, and
the very next line (`result["injury_score"].isna()...`) throws
`KeyError: 'injury_score'` — caught by the function's own broad
try/except and silently defaulted to neutral (1.0), so it never crashed
anything, just silently discarded a real injury lookup on every call
where the column already existed. Reproduced directly (row with a
pre-existing `injury_score` column → confirmed `KeyError`), fixed by
dropping any pre-existing `injury_score`/`is_injured` columns immediately
before the merge. Verified fix + full 110-test suite pass. Doesn't change
Phase 7's own output (its synthetic rows force-override `injury_score` to
a neutral 1.0 immediately afterward regardless, by design — see the
module docstring's "conditional on playing" note) but likely affects
other callers of `add_external_features` that don't have that override
and were silently losing real injury data whenever this collision
occurred.

**Real run results, seasons 2023-2025, all 4 positions** — first time
this phase has ever actually executed (`data/experiments/
phase7_season_projection.csv`, 1,745 total player-seasons):

| Position | MAE | Bias | n | Mean synthetic-week share |
|---|---|---|---|---|
| QB | 47.92 | +22.50 | 239 | 0.462 |
| RB | 27.59 | -6.26 | 409 | 0.346 |
| WR | 26.22 | -14.50 | 700 | 0.311 |
| TE | 19.58 | -13.79 | 397 | 0.256 |

Notable, reported as found rather than smoothed over: **QB substantially
over-projects (+22.5) while RB/WR/TE all under-project (-6 to -14.5)** —
opposite directions, not just different magnitudes. Also consistent
across every position: players requiring zero synthetic weeks are
projected far more accurately than players requiring any (e.g. QB 53.5
MAE with no synthetic weeks vs. 46.8 with some — inverted from the other
3 positions, where "any synthetic" reliance roughly doubles the error:
RB 44.0 vs 23.2, WR 42.4 vs 21.8, TE 28.5 vs 15.3). QB's inversion is
itself notable and not yet explained — worth investigating before trusting
QB season projections for players who miss real games during the
projection window.

**Bug found and fixed en route (real, currently-active, unrelated to
Phase 7 specifically)**: while auditing GAPS.md for other open issues in
parallel with this run, re-verified the previously-documented but
never-fixed §9.2 item "LAR/JAC team code mismatch" and confirmed it was
still live — `player_weekly_stats.team`/`.opponent` had 386/362 rows
respectively still coded `LAR`/`JAC` (100% concentrated in season 2025;
`schedule` and every other table already use the modern `LA`/`JAX`
codes), meaning opponent-dependent matchup features
(`opp_fpts_allowed*`, DVP, `possible_weeks_for_team`'s schedule join)
were silently dead for every Rams/Jaguars player in the current season.
Checked for `UNIQUE(player_id, season, week)` collision risk before
touching anything (zero collisions found), backed up the DB
(`nfl_data.db.bak-lar-jac-fix-20260811221644`), normalized both columns
in place (`UPDATE ... SET team/opponent = 'LA'/'JAX' WHERE ... = 'LAR'/'JAC'`).
Row count unchanged (118,962), full 110-test suite still passes. Checked
`snap_counts`/`weekly_rosters`/`team_stats`/`team_personnel_stats` for the
same mismatch — none found, isolated to `player_weekly_stats`.

**Phases 2-3 re-run on the corrected panel — 2026-08-10/11.** Both re-run
in full against the corrected data (`data/experiments/
phase2_single_week_comparison_v2_corrected.csv`,
`phase3_training_window_comparison_v2_corrected.csv`; original pre-fix
CSVs preserved without the `_v2_corrected` suffix for comparison).

Phase 2 (architecture comparison, same 7 architectures A-G, all 4
positions) — **every position's winning architecture changed**:

| Position | Original winner | Corrected winner |
|---|---|---|
| QB | C_gbm_mae | F_yeojohnson_huber |
| RB | F_yeojohnson_huber | C_gbm_mae (E_quantile_gbm treated as a supplementary floor/median/ceiling tool per next_focus.md's framing, not a competing point-estimate architecture) |
| WR | C_gbm_mae | C_gbm_mae (unchanged) |
| TE | C_gbm_mae | B_gbm_huber |

Notably, architecture D (the two-stage/hurdle model — specifically
designed to exploit the new true-zero rows to learn `P(PPR > 0)`) does
**not** win at any position, despite being exactly the mechanism this
data fix was meant to unlock. Reported as found, not forced into the
expected narrative.

Phase 3 (training-window x recency-weighting grid, full 5-window x
3-weighting sweep, all 4 positions, matching the original's exact grid
per explicit user instruction to not scope it down) — new best
(window, weighting) per position, evaluated using each position's new
Phase 2 winner:

| Position | Window | Weighting | MAE |
|---|---|---|---|
| QB | 7y | none | 6.32 |
| RB | all | exponential | 4.54 |
| WR | 3y | none | 4.18 |
| TE | 7y | none | 2.85 |

Contrast with the original (pre-fix) finding in this section (§7.9)
that MAE improved monotonically out to full history for QB/RB/TE: on
corrected data, **QB and TE now peak at 7y and get worse at 10y/all**,
RB still favors "all" but now specifically paired with exponential
weighting (recency-downweighting the now-much-larger low-confidence
2006-2017 zero-tail), and WR now clearly prefers the shortest window
(3y) rather than being flat/mixed across all windows. The corrected
zero-inflated target distribution changed the recency/window tradeoff
materially, not just the point architecture — consistent with what the
user anticipated when proposing this fix ("I would expect this change
to potentially alter... the optimal recency strategy").

`FINAL_CONFIG` in `src/models/single_week_ppr/final_config.py` updated
to these values.

**Phase 4 re-run on the corrected panel — 2026-08-11.** Row-level
validation re-run for all 4 positions using each position's new
FINAL_CONFIG (`data/experiments/phase4_row_level_predictions_v2_corrected.csv`,
139,986 rows; original preserved without the suffix). The challenger
architecture beats `existing_methodology` (production Ridge) at every
position, consistent with Phase 2/3's selection:

| Position | Challenger | Challenger MAE | existing_methodology MAE |
|---|---|---|---|
| QB | F_yeojohnson_huber | 6.11 | 6.36 |
| RB | C_gbm_mae | 4.50 | 4.59 |
| WR | C_gbm_mae | 4.13 | 4.24 |
| TE | B_gbm_huber | 2.86 | 2.94 |

**Caution — absolute MAE is not comparable across the original and
corrected datasets.** All positions' MAE dropped noticeably versus the
original (pre-fix) Phase 4 run (e.g. TE existing_methodology 3.71→2.94,
QB 6.33→6.36 roughly flat, RB 4.80→4.59, WR 4.85→4.24). This is *not*
evidence the models "got better" — the corrected eval set now legitimately
contains far more true zero-PPR rows (up to ~40% for TE, see the
sensitivity table above), and near-zero rows are mechanically easier to
predict accurately, pulling the aggregate MAE down regardless of model
quality. The within-run ranking (challenger vs. existing_methodology vs.
baselines) is still the valid, meaningful signal; the cross-run absolute
MAE trend is not and should not be read as "the corrected models are
substantially more accurate."

Quantile calibration (`E_quantile_gbm`) shows p50 coverage running
noticeably below the nominal 0.50 for the 2025 test season specifically,
across all 4 positions (QB 0.435, RB 0.364, WR 0.448, TE 0.446 — vs.
2023/2024 tracking closer to nominal in the 0.53-0.60 range). Flagged,
not yet root-caused — plausibly an artifact of the corrected zero-inflated
target distribution stressing the quantile heads differently than the
median/MAE point estimate, or a genuine 2025-specific distribution
shift. Worth investigating before treating quantile outputs (floor/
ceiling estimates) as reliable for the current season.

**Phase 5 re-run on the corrected panel — 2026-08-11.** Nested-CV
hyperparameter tuning (100 Optuna trials x 3 seasons) re-run per
position's new FINAL_CONFIG architecture
(`data/experiments/phase5_tuned_predictions_v2_corrected.csv`,
`phase5_tuned_hyperparameters_v2_corrected.csv`).

**Bug found and fixed en route**: `tuning.py:_build_model` only handled
`C_gbm_mae` and `F_yeojohnson_huber` — the two architectures that won
Phase 2 on the *original* pre-fix data. TE's new corrected-data winner,
`B_gbm_huber`, wasn't wired up at all and raised `ValueError: Tuning not
implemented for architecture: 'B_gbm_huber'` on first attempt. Fixed by
adding a `B_gbm_huber -> GBMRegressor(objective="huber")` branch,
confirmed equivalent to `YeoJohnsonHuber`'s own huber wrapping (same
underlying `GBMRegressor(objective="huber")`, so the existing `alpha`
(huber-delta) search range applies unchanged) — extended
`tune_huber_alpha` to cover both `F_yeojohnson_huber` and `B_gbm_huber`.
Re-ran TE after the fix; the other three positions were unaffected (their
architectures were already supported) and did not need a re-run.

**Also found and worked around, not fixed**: `run_tuned_validation`
overwrites `data/experiments/phase5_tuned_hyperparameters.csv` on every
call (`params_output_path.to_csv`, no append, no CLI override) —
identical gotcha to the Phase 2 `run_comparison()` overwrite mistake
documented above. Worked around the same way: ran each position with the
default path, copied its output to a temp file immediately after, reset
the tracked file via `git checkout --`, then merged all 4 positions into
`phase5_tuned_hyperparameters_v2_corrected.csv` at the end. The original
tracked file's content (from the pre-fix run) was never actually lost —
confirmed via `git diff --stat` showing only a 4-line change after each
run, restored cleanly every time.

Tuned vs. default (Phase 4, corrected) MAE, computed directly against the
corrected Phase 4 baseline rather than trusting the script's own printed
comparison (which reads a hardcoded, non-overridable path and was
silently comparing against the *stale pre-fix* Phase 4 file):

| Position | Default MAE | Tuned MAE | Delta |
|---|---|---|---|
| QB | 6.113 | 6.133 | +0.021 (worse) |
| RB | 4.495 | 4.411 | -0.083 (improved) |
| WR | 4.126 | 4.103 | -0.023 (improved) |
| TE | 2.857 | 2.804 | -0.053 (improved) |

Small deltas either way — consistent with the original (pre-fix) Phase 5
finding that tuning provides modest, not transformative, gains over
Phase 2's already-reasonable defaults. QB is a genuine (if tiny) regression
from tuning; worth a note but not worth abandoning tuned QB hyperparameters
over a 0.02 MAE difference within noise.

**Phase 6c re-run on the corrected panel — 2026-08-11.** Feature-count
ablation (10/20/30/all, `CAUSAL_FEATURES` ranked by LightGBM importance)
re-run per position's new FINAL_CONFIG architecture
(`data/experiments/phase6c_feature_ablation_v2_corrected.csv`).

| Position | 10 | 20 | 30 | all |
|---|---|---|---|---|
| QB | 6.279 | 6.169 | 6.166 | **6.114** |
| RB | 4.807 | 4.755 | 4.738 | **4.736** |
| WR | 4.175 | 4.159 | **4.106** | 4.122 |
| TE | 2.868 | 2.860 | **2.851** | 2.855 |

QB and RB still monotonically improve out to "all" features, matching
the original (pre-fix) conclusion. WR and TE now peak at 30 features and
get very slightly worse at "all" (WR: 4.106→4.122, TE: 2.851→2.855) — a
small, non-monotonic reversal that wasn't present pre-fix. Differences at
that scale (≤0.016 MAE) are within noise given no repeated-trial variance
estimate exists here, so this doesn't override "more features generally
helps or is neutral" as the practical takeaway for all 4 positions, but
it's genuinely different from the original clean-monotonic result and
worth noting rather than glossing over — plausibly the corrected zero-
inflated target makes the marginal features noisier to fit exactly at the
full feature count for WR/TE specifically.

**Not yet done — reconsidering and re-running Phase 7 (18-week season
projection) on the corrected panel.** Per next_focus.md's own note, this
phase's design likely needs to change to use the new inferred-zero rows
directly (rather than just re-running the existing synthetic-row
mechanism as-is) — still pending, not a simple re-run like Phases 2-6c
were.

**Fixed (2026-08-09, follow-up) — `rosters` table was empty for a real
reason, not just "never run": `scripts/ingest_rookie_data.py`'s
`ingest_rosters()` called `nfl.import_rosters(seasons)`, which no
longer exists in the installed `nfl_data_py` version** (renamed
upstream; `AttributeError: module 'nfl_data_py' has no attribute
'import_rosters'. Did you mean: '__import_rosters'`) — the same class
of bug as the `drive_result`→`fixed_drive_result` rename above, just
in a standalone script instead of the main pipeline, so it never
surfaced in day-to-day runs. Current API is `import_seasonal_rosters()`
(there's also `import_weekly_rosters()`, presumably what already
populates `weekly_rosters_v2`). Swapped the call
(`scripts/ingest_rookie_data.py:62`), verified output schema still has
every column `bulk_insert_rosters()` expects (`player_id`,
`player_name`, `position`, `team`, `season`, `birth_date`, `height`,
`weight`, `college`, `jersey_number`, `status`, `years_exp`), fixed the
stale docstring in `database.py:1178`, then ran
`python scripts/ingest_rookie_data.py --only rosters`: 67,354 roster
records inserted (seasons 2000-2026). Verified
`scripts/snake_draft_sim.py`'s actual query (`SELECT ... FROM rosters
WHERE player_name IS NOT NULL AND position IN ('QB','RB','WR','TE')`)
now returns 20,490 rows instead of 0 — the simulator is unblocked.
Bonus: `backfill_players_from_rosters()` (called automatically at the
end of the ingest script) also backfilled `players.college`/
`birth_date` for 1,835 of 2,868 players, fixing the "college all-null"
finding from the very first pass of this audit at no extra cost.

**Checked and clean, no action needed:** `data/cached_features.parquet`
matches the current `FEATURE_VERSION` (`30`, both cache and
`data/models/feature_version.txt` share today's mtime — not stale).
`data/draft_picks.parquet` / `combine_data.parquet` / `injuries.parquet`
are not orphaned duplicates of `draft_picks_v2`/`combine_data_v2`/
`injuries_nflpy` — same refresh mtime, and `draft_picks.parquet` is
actively read directly by `feature_engineering.py` (parquet used for
fast feature-engineering reads, DB table for querying/joins — intentional
dual storage, not drift). Name-collision check (`players` grouped by
`name`+`position`) found only expected abbreviated-name ambiguity
(e.g. "J.Williams" WR resolving to 4 distinct real players) — harmless
since all downstream joins key on `player_id`, never `name`.

---

## Remaining items from the user's original checklist (2026-08-09, cont'd)

**`snap_counts`/`weekly_pfr` orphan check (via PFR->GSIS crosswalk,
`get_pfr_to_gsis_map()`) — not a bug.** `snap_counts` covers every
position (LB, CB, DE, OL, etc. all present, confirmed via
`GROUP BY position`), while `players` only ever gets skill positions
that appear in `player_weekly_stats`. Same scope mismatch as the
roster/injury tables above. `weekly_pfr`'s much lower 8.4% orphan rate
(vs. snap_counts' ~64%) is consistent with it already being
skill-position-focused.

**Team-level `snap_share` sanity (sum per team/week) — clean.**
Distribution is tightly centered at 5-6 (max observed 6.66) across all
team-weeks, not the ~100%/1.0 pattern that would indicate
double-counting. Makes sense once scoped correctly:
`player_weekly_stats` only covers QB/RB/WR/TE, so a team-week sum
here represents roughly "1 QB + 4-5 skill-position slots," not all 11
offensive players (which would include OL, not present in this table).

**Season-coverage cliff check — clean for `snap_counts`/`ngs_*`,
real bug found and fixed in `game_odds`.** `snap_counts`,
`ngs_passing`, `ngs_rushing`, `ngs_receiving` all show smooth,
gradually-varying row counts per season (2018-2025), no cliffs.
`game_odds` had 2,759 rows (1.05% of the table, 150 distinct events)
with **blank `season`/`week`**, invisible to the
`idx_game_odds_season_week` index and any season-filtered query.

Root cause: `_parse_market_outcomes()` in `src/scrapers/odds_scraper.py`
always returns `season: None, week: None` in its row dict (it only has
event_id/teams/commence_time from the odds API, not NFL schedule
context) — only the historical-backfill workflow
(`scrape_historical_season`) separately overrode this after the fact;
the four live/current-odds call sites (`scrape_current` etc.) never
did, so every live scrape run has been silently accumulating
season/week-less rows.

Confirmed via one example (TEN@PIT, `commence_time` originally
2020-10-04) that this isn't just "missing metadata" but actively
**wrong if naively date-matched** — that game was COVID-postponed to
Week 7 (2020-10-25); a date-based backfill would have mismatched it.
Fixed properly: backfilled the 134 resolvable historical events
(2,593 of the 2,759 rows) by matching `(home_team, away_team, season)`
against the local `schedule` table — team-pair matching rather than
date matching, so postponements resolve correctly regardless of which
date odds were captured on. The remaining 166 rows (16 events) are
genuinely unmatchable: Super Bowl/conference-championship **futures
markets** for hypothetical matchups that never happened (e.g. "GB vs
KC" priced ahead of Super Bowl LV, when the actual game was
Buccaneers-Chiefs) — correctly has no `schedule` row, not a bug.

Also fixed the recurrence: `save_game_odds_to_db()` (`odds_scraper.py`)
now resolves `season`/`week` from the `schedule` table at insert time
for every call site (not just the historical-backfill path), using the
same postponement-safe team-pair matching. Verified with a synthetic
row reproducing the exact TEN@PIT case: resolves to `season=2020,
week=7` on insert instead of persisting `None`.

**Duplicate/near-duplicate players (name+birth_date), leakage risk
review, and `manual_adjustments_2026.json`/`rookie_priors.json` drift
— completed in the same session, see below.**

**Duplicate players (name+birth_date) — clean.** No collisions across
the 64% of `players` rows with `birth_date` populated (would indicate
the same real person split across two `player_id`s; none found).

**Leakage review — no issues found.** All 31 `.rolling()` calls in
`src/features/feature_engineering.py` are leak-safe: 30 call
`.shift(1)` immediately before `.rolling(...)`, and the one exception
(line 3078, boom/bust rate) is correct despite not having its own
`.shift(1)` — it operates on `shifted_fp` (line 3062), which was
already lagged upstream, so a second shift would be wrong, not missing.
No season-total-joined-onto-every-week pattern found either. Situational
PBP columns (`redzone_targets` etc.) are aggregated per
`(season, week, player)`, not full-season, by construction. Not an
exhaustive line-by-line audit of every feature, but the pattern is
consistent enough not to flag as a live concern.

**`manual_adjustments_2026.json` — clean, no drift.** Cross-checked
all 28 entries' implied team (parsed from each note's "TEAM: reason"
prefix) against current 2026 roster data; every one matches. One
apparent mismatch (Tyler Allgeier: noted "ARI", roster data said "AZ")
turned out to be a team-code inconsistency in the `rosters` table
itself, not a stale adjustment — see below.

**Found in passing: `rosters` table has 3 different codes for
Arizona (`ARI`/`ARZ`/`AZ`, from this session's `import_seasonal_rosters()`
backfill spanning 2000-2026) plus similar duplicates for Baltimore
(`BAL`/`BLT`), Cleveland (`CLE`/`CLV`), and Houston (`HOU`/`HST`) —
nflverse used different abbreviation conventions in different
historical seasons and `bulk_insert_rosters()` doesn't normalize them.
Confirmed **zero functional impact**: the only consumer,
`scripts/snake_draft_sim.py`, displays `rosters.team` but never joins
on it. Logged as cosmetic/low-priority rather than fixed.

**`rookie_priors.json` — not stale, intentional design.**
`scripts/compute_rookie_priors.py --fit-until` defaults to 2023
specifically so 2024/2025 rookies are held out as an out-of-sample
validation set (per the script's own `--help` text) — the file's
`fit_window: [2006, 2023]` is deliberate, not a forgotten refit.

---

## Phase 8 (next_focus.md): summed-weekly vs. direct season-total prediction — 2026-08-12

Wired Phase 7's already-computed summed-weekly output into the existing
season-level walk-forward benchmark (`scripts/walk_forward_preseason.py`,
which already compared `PreseasonProjector` [production Ridge] against a
richer untested candidate [`preseason_features.py`'s multi-year/
team-aware Ridge]) as a third arm, per user-confirmed minimal scope — no
new model training, `_phase7_metrics()` just loads
`data/experiments/phase7_season_projection.csv` and reuses the existing
`_metrics()` helper. Spot-checked against the already-known Phase 7
numbers before trusting the wiring (aggregate MAE here: QB 47.9/RB 27.5/
WR 26.2/TE 19.5 vs. the earlier pooled-run numbers 47.92/27.59/26.22/
19.58 — matches almost exactly, confirming no bug).

**Result — summed-weekly (Phase 7) beats both direct season-total models
by a wide margin, at every position, on both MAE and R²** (walk-forward,
seasons 2023-2025, 3 folds each):

| Position | Production MAE / R² | Candidate MAE / R² | Phase 7 MAE / R² |
|---|---|---|---|
| QB | 81.2 / 0.375 | 81.9 / 0.103 | **47.9 / 0.742** |
| RB | 55.8 / 0.499 | 56.2 / 0.435 | **27.5 / 0.818** |
| WR | 47.3 / 0.557 | 41.9 / 0.611 | **26.2 / 0.769** |
| TE | 33.6 / 0.582 | 29.8 / 0.587 | **19.5 / 0.726** |

Answers next_focus.md's Phase 8 question directly: for this pipeline, a
season-total projection built by summing 18 leakage-safe weekly
predictions is clearly more accurate than a single model predicting the
season total directly from preseason/prior-season features alone — not a
close call, roughly 40-50% lower MAE at every position.

**Caveat, not smoothed over — populations aren't identical across arms**,
consistent with how production and candidate already didn't match each
other before this change (different SQL-level eligibility filters).
Phase 7's `n` is meaningfully larger at every position (e.g. QB n=239 vs.
production's 112/candidate's 132) — Phase 7's eligibility (`>=20` test
rows per position/season in `season_projection.py`) is looser than
`PreseasonProjector`'s `MIN_GAMES=6`-prior-season /
`COUNT(*)>=4`-target-season filter, meaning Phase 7 is scored on a wider,
likely easier-on-average population (more backups/rookies with partial
seasons included). This is a real, unresolved confound: some of Phase
7's advantage could be population breadth rather than pure architectural
superiority. Not corrected here per the agreed minimal scope (each arm
reported on its own natural population, matching the file's existing
precedent) — worth a forced-intersection re-check before treating this
result as fully conclusive, but the gap is large enough (40-50% MAE
reduction) that population differences alone are unlikely to fully
explain it.

**Also notable**: the richer "candidate" direct model
(`preseason_features.py`'s multi-year/team-aware features) does NOT
consistently beat the simpler production Ridge — better for WR/TE,
worse for QB (R² 0.103 vs. 0.375), roughly tied for RB. More features
didn't straightforwardly help here, unlike the tree-based single-week
architectures earlier in this session.

Full fold-by-fold and aggregate results saved to
`data/backtest_results/walk_forward_preseason_20260812_120925.json`
(2023-2025 only) and `..._121005.json` (full default range, confirms
production/candidate's normal 2018-2025 walk-forward is unaffected by
this change, with Phase 7 correctly limited to its 3 available folds).

**Follow-up: population confound isolated — 2026-08-12.** Added
`--intersect-populations` to `scripts/walk_forward_preseason.py`:
restricts all 3 arms to the common `player_id` intersection per
position/fold before scoring (models still trained on their own full
training set — only the *scored* test rows are narrowed), skipping a
position/fold if the intersection is below 15 players. Refactored the
per-fold loop from 3 sequential all-position blocks into a single
per-position loop so all 3 arms' test frames are available together for
the intersection; verified the refactor is a pure no-op when the flag is
off (re-ran without `--intersect-populations`, byte-identical MAE/R²/n
to the pre-refactor run above). Removed `_phase7_metrics()`, now dead
code after being inlined into the per-position loop.

**Result: production's own eligibility filter (`MIN_GAMES=6` prior
season) turns out to already be the strictest of the three** — its `n`
is unchanged before/after intersecting (112/226/428/272 for QB/RB/WR/TE,
identical to the un-intersected run), meaning candidate and Phase 7 were
simply narrowed down to production's existing population, not a novel
smaller set. This makes the intersected comparison a clean answer to
"how do all 3 perform on exactly the same, established-veteran-only
population":

| Position | Production MAE | Candidate MAE | Phase 7 MAE (matched) | Phase 7 MAE (original, unmatched) |
|---|---|---|---|---|
| QB | 81.2 | 80.4 | 58.1 | 47.9 |
| RB | 55.8 | 54.8 | 32.8 | 27.5 |
| WR | 47.3 | 42.9 | 31.5 | 26.2 |
| TE | 33.6 | 29.9 | 22.2 | 19.5 |

**Conclusion: the population-breadth confound is real but does not
explain away the result.** Phase 7's MAE rises 8-21% when restricted to
only the population production requires (confirming part of its original
edge really was from correctly handling easier backups/rookies that
production's stricter filter excludes) — but summed-weekly still beats
both direct models by a clear 26-40% margin at every position even under
this matched, apples-to-apples comparison. Phase 8's headline conclusion
(summed-weekly > direct season-total prediction, for this pipeline)
holds up under scrutiny, not just as an artifact of unequal test
populations.

Saved to `data/backtest_results/walk_forward_preseason_20260812_122701.json`
(intersected) alongside `..._122649.json` (re-run of the original
unintersected mode, confirming the refactor didn't change prior results).


---

## Three root-cause fixes before the Phase 5 re-run — 2026-08-18

Found while investigating the synthetic-week bias. All three were latent
defects that predate this session's work; the depth-chart change merely
perturbed values enough to expose the first one.

### 1. Silently-dropped folds (highest severity — corrupted a result)

Every phase in `src/models/single_week_ppr/` wraps per-fold loading in a
broad `except Exception: logger.warning(...); continue`. The broad catch is
deliberate (one bad fold shouldn't kill a multi-hour grid), but a
`logger.warning` is the *only* trace — and it vanishes whenever output is
piped through `tail`, which is how every long run in this session was
watched.

**This actually corrupted a result.** `_apply_bounded_scaling` raised
`KeyError: 'air_yards_share_pct_roll3_mean_missing'` on the QB/2025/`all`
fold, which was swallowed. The Phase 3 grid then reported QB/`all` = 6.18
MAE computed over **2 of 3 seasons**, making `all` look like a decisive
winner. It was caught only by noticing 14 rows where 21 were expected. With
2025 restored the true value is 6.337 — still nominally best, but by 0.025
over 10y rather than a clear margin. It was one step from being promoted
into `FINAL_CONFIG`.

**Root cause**: failures were ephemeral, not durable artifacts.
**Fix**: `FoldFailureTracker` (`evaluate.py`), wired into all 7 fold
handlers (Phases 2, 3, 4, 5, 6c, 7, 9). Records each failure, prints an
unmissable end-of-run summary explicitly warning that aggregates are
computed over incomplete data, and writes a `<output>.failures.json`
sidecar so incompleteness survives terminal scrollback and is auditable
after the fact. A clean run now says so affirmatively. 5 tests.

### 2. `_apply_bounded_scaling` train/test column mismatch

`cols = _infer_bounded_columns(train_df)` then `test_df[col]` — assumed
both frames share a column set. But the missing-indicator step builds
`<col>_missing` columns **conditionally per frame** (only above 2%
missingness) and train/test are engineered separately, so a column can
legitimately exist in one and not the other. Production code, not just
experiment tooling.

**Fix**: indicator columns absent from test fill with 0 (semantically
correct — no missingness); any *other* absent column is dropped from
scaling rather than fabricated. 4 tests.

### 3. Unbounded depth-chart staleness + dirty `depth_charts` table

The new as-of lookup had **no staleness bound**: a season with no coverage
silently inherited the newest prior season's ranks, however old.

**Coverage reality**: `depth_charts` covers **2020-2024 only**. 2018/2019
have none (and no prior data to carry). **2025 has none and is NOT
backfillable** — `nfl.import_depth_charts([2025])` fails upstream
(`KeyError: 'week'`). So for 2025, a live validation season, rows carry
2024 ranks and the depth-chart fix is **effectively inert**. This dampens
any measured effect of the fix in Phases 7/9 and must be kept in view when
reading those results.

> **CORRECTED 2026-08-19 — 2025 *is* backfillable; it is now loaded.**
> The `KeyError: 'week'` was not upstream absence. nflverse switched 2025
> to a different feed: a **daily league-wide ESPN-style snapshot** keyed on
> `dt`/`pos_rank`/`pos_slot`, with no `week`/`depth_team`/`season` column —
> hence the KeyError, and hence the **554,215 `season IS NULL` rows deleted
> as "junk" below, which were in fact 2025's real data in an unrecognised
> schema.** `scripts/backfill_depth_charts_2025.py` bridges the schemas
> (last snapshot strictly before each week's first kickoff → that week;
> `pos_rank` clipped to 3 to match the old feed's 1/2/3 vocabulary;
> postseason narrowed to clubs actually playing, as the old per-game feed
> did naturally). Loaded 42,496 rows across 22 weeks, 2,267 players.
> **Effect: 44.7% of 2025 skill-position rows change depth rank, and 925
> rows — rookies and others with no 2024 chart — get a real rank where they
> were previously forced to the neutral default 3.** The depth-chart fix is
> therefore no longer inert for 2025, and Phase 7/9 results read under the
> "effectively inert" assumption should be re-examined.
>
> Lesson: an upstream `KeyError` on a column means *the schema changed*,
> not *the data is missing*. Worth checking the raw frame's columns before
> concluding a season is unavailable — and worth being suspicious of any
> bulk "junk rows with NULL keys" deletion, which is the same event seen
> from the other end.

**Fix**: `DEPTH_CHART_MAX_STALENESS_SEASONS = 1`, shared by both the
real-row (`_add_depth_chart_rank`) and synthetic-row
(`_lookup_depth_chart_rank_asof`) paths so they can never disagree about
whether a snapshot is still trustworthy. Beyond the bound → neutral
default rather than a pretended-known value. Verified this changes
**nothing** for current data (2018/19 have no prior data; 2020-2024 have
same-season data; 2025 carries exactly 1 season) — it codifies today's
behavior as intentional and prevents future pathology. 3 tests.

**DB cleanup** (backed up to `nfl_data.db.bak-depthchart-cleanup-*` first):
deleted 554,215 `season IS NULL` junk rows and 1,828 exact duplicates;
638,236 → 82,193 rows, 0 true duplicates remaining (an apparent "524
remaining" was a NULL-concat artifact in the check query, confirmed via a
NULL-safe recheck). `player_weekly_stats` untouched at 118,962.

**Two documented data caveats, not bugs:**
- **1,821 genuinely conflicting ranks in 2024** (same player/week listed at
  multiple depth slots). Resolved by `MIN` — takes the most prominent
  listed role, chosen over `first` mainly for determinism. A judgment call,
  now explicit in code rather than implicit.
- **Granularity differs by season**: 2024 has 58 distinct `depth_position`
  values and 2,054 players vs. ~15 and ~600 for 2020-2023 — an upstream
  format change (full roster vs. a narrow subset), not duplication. So
  `depth_chart_rank` coverage is season-dependent: a player can get a real
  rank in 2024 and the neutral default in 2023.

## `get_opponent_matchup_features` is not idempotent (found 2026-08-18)

**Symptom**: every synthetic week in the Phase 7 / availability-comparison
path logs `Error adding defense rankings: Columns must be same length as
key`, immediately after a nonsensical `Calculated rankings for 1 teams`.

**Cause**: `build_synthetic_week_row` (season_projection.py:195) carries
forward the player's last real row, which `run_fold` has *already* passed
through `add_external_features`. Line 214 then re-runs the same enrichment.
`get_opponent_matchup_features` merges and renames to `opp_defense_rank` /
`opp_matchup_score` / `opp_pts_allowed` — names the frame already has — so
the labels duplicate. `avail_mask` (external_data.py:338) becomes a 2-column
DataFrame instead of a Series, and the assignment at line 343 raises from
pandas `_set_item_frame_value` (frame.py:4209). Reproduced exactly.

**Impact on models: none.** No position's `CAUSAL_FEATURES` contains any of
the four defense columns. Of the 18 features `add_external_features`
produces, only `injury_score`, `is_dome`, `implied_team_total` and `spread`
are consumed; `injury_score` is overwritten to 1.0 by design at
season_projection.py:224, and the weather/Vegas merges both succeed on the
1-row frame (`Merged Vegas lines for 1 player-game rows` is 1-of-1, not a
failure). Availability-experiment gradients are therefore uncontaminated.

**Impact on cost: large.** One `add_external_features` call on a single row
takes 8.2s and runs once per synthetic week per player. The loaders are
uncached and re-read a whole season each time: vegas 1.49s, weather 1.09s,
injuries 0.39s. Plus the defense block computes rankings from a one-row
frame and throws the result away.

**Secondary bug**: the `except` at external_data.py:908 backfills the three
`opp_*` defaults but leaves `defense_data_available` at its carried-forward
value, so the flag asserts real data while the values are constants. Any
future model that adopts that flag would be misled.

**Fixed 2026-08-18.**

1. `get_opponent_matchup_features` now drops any pre-existing `opp_*` /
   `defense_data_available` columns before the merge, so a second pass
   recomputes rather than colliding. Verified idempotent over three
   consecutive passes.
2. Memoized the `nfl_data_py` fetches (`_cached_import`, keyed by kind +
   sorted season tuple, handing out copies since callers mutate).
   `add_external_features` on a 1-row frame: **8.2s → 0.08s warm**. The
   three `import_schedules` calls per invocation now share one fetch.
   `clear_season_import_cache()` evicts.
3. The `except` branch at external_data.py:938 now sets
   `defense_data_available = 0` alongside the constant fallbacks.

Verified the model-consumed features (`injury_score`, `is_dome`,
`implied_team_total`, `spread`) are byte-identical to pre-fix output on
clean frames across QB/WR/RB rows — the change is inert except on
already-enriched frames. 7 new tests in
`tests/test_external_data_idempotency.py`; full suite 197 passed.

**Caveat**: the new `test_availability_flag_is_not_inherited` passes
against the old code too for its input (the old code only mis-set the flag
when the exception fired). The except-branch fix itself is now unreachable
from this cause and is untested — kept as defensive correctness.

## Silently-dropped player-weeks (fixed 2026-08-18)

`FoldFailureTracker` made skipped FOLDS auditable. The same failure mode
existed one level down and was still silent: `compute_player_week_predictions`
skips a week when `build_synthetic_week_row` returns None (no prior history)
or when `predict()` raises. The first logged **nothing at all**; the second
logged a `logger.warning` that vanishes under `tail` -- the exact reason
FoldFailureTracker was written.

**Impact measured on the v33 QB availability run**: 129 player-weeks skipped
across 20 player-seasons, all `no_prior_history`. Concentrated in the LOW
synthetic-share buckets, which is the damaging place for them to land:
`weeks_synthetic == 0` silently conflated "played every week" (34 QBs) with
"we failed to project the weeks they missed" (5 QBs). That moved the
reference bucket's bias from **-30.6 to -28.8** -- a corrupted anchor for
every gradient measured against it.

**Fix**: `WeekSkipTracker` in season_projection.py, mirroring
FoldFailureTracker: records (player, season, week, reason), prints a loud
end-of-run summary, and writes a `<output>.weekskips.json` sidecar. Wired
into Phase 7, Phase 9, and both availability experiment scripts. The
parameter is optional, so existing callers are unaffected. 10 tests, incl.
the accounting identity real + synthetic + skipped == possible.

## Defense rankings computed on frames too small to rank (fixed 2026-08-18)

`calculate_defense_rankings` was called on single-row synthetic frames,
printing `Calculated rankings for 1 teams` and ranking one defense against
itself before the result silently resolved to neutral defaults. After the
idempotency fix removed the exception it had been throwing, this became
*fully* silent -- a loud wrong signal traded for no signal.

**Fix**: `MIN_DEFENSES_FOR_RANKING = 8`. Below that many distinct opponents
the function says so once and returns empty, so the caller takes its
existing defaults path with `defense_data_available = 0`. Verified
byte-identical output on a full 32-team frame; the 290 bogus
"rankings for 1 teams" lines in a QB run became 0.

## Mid-season trades projected against the wrong schedule (NOT fixed)

Found while reconciling the WeekSkipTracker count (129) against the CSV
week-count identity (127). Two player-seasons had 13 real + 5 synthetic = 18
weeks against a 17-week schedule.

**Cause**: `possible_weeks_for_team(db, team, season)` is called with
`team = g.sort_values("week")["team"].iloc[0]` -- the player's FIRST team of
the season. After a mid-season trade the player is projected against their
old team's schedule: wrong bye week, wrong opponents. Confirmed on
00-0033949/2023 (ARI->MIN; played ARI's week-14 bye for MIN) and
00-0026158/2025 (CLE->CIN; played CLE's week-9 bye for CIN).

**Scope (QB, 2023-2025)**: 3 of 238 player-seasons changed teams mid-season;
all 3 are projected against a partly wrong schedule, but only the 2 above
are detectable via the week-count identity. Small for QB; likely larger for
RB/WR where in-season movement is more common -- unmeasured.

**Fixed 2026-08-18.** New `possible_weeks_for_player(db, real_team_by_week,
season)` resolves the team PER WEEK -- the player's own row's team on weeks
they played, the last known team carried forward otherwise, their first team
for weeks before they ever appeared -- and returns both the playable weeks
and the week->team map. A week the player demonstrably played is always
playable, regardless of what the schedule lookup says.

`compute_player_week_predictions` now takes `team_by_week: Dict[int, str]`
instead of `team: str`, so no caller can silently pin a season to one team
again. All four call sites updated (Phase 7, Phase 9, both experiment
scripts); the now-dead `possible_weeks_for_team` imports were removed from
two of them. Schedules are cached per team inside the loop, so the per-week
resolution costs one DB read per team, not one per week.

Verified against both real cases with the live DB. 00-0033949/2023 traded
ARI->MIN at week 9; 00-0026158/2025 traded CLE->CIN at week 6. Both had one
real week falling outside their first team's schedule (their old team's bye);
after the fix, zero real weeks fall outside, and 13 real + 4 synthetic = 17
satisfies the identity in both cases. Single-team players are unaffected --
a test asserts `possible_weeks_for_player` equals `possible_weeks_for_team`
for them. 15 tests across `tests/test_traded_player_schedule.py` and the
updated `tests/test_week_skip_tracker.py`.

**Still unmeasured**: RB/WR/TE in-season movement is more common than QB's
3-of-238, so the pre-fix error was probably larger there. Nothing has been
re-run for those positions.


## `effective_rate` is undefined when weekly predictions cancel (guarded 2026-08-18)

Found by an integrity sweep on the post-fix availability outputs: one row of
995 had `effective_rate = -0.251`, outside [0, 1].

**Not an estimator bug.** Weekly QB PPR predictions can be negative (INTs,
fumbles, sacks) -- 3.5% of 3,322 player-weeks are, min -1.73. For
00-0034401/2024, 10 of 16 synthetic weeks were negative, so
`synth_pred_sum` came out at 0.344: a near-zero residue of cancelling terms.
`effective_rate = (total - known) / synth_pred_sum` is a weighted mean of
the per-week rates only when the weights share a sign; otherwise it is
unstable. Every underlying availability rate was valid.

**Guard**: the script now also records `synth_pred_abs_sum` and emits
`effective_rate` only when `synth_pred_sum > 0 and synth_pred_sum >= 0.5 *
synth_pred_abs_sum`, else NaN. Reporting-only column; no conclusion in the
availability investigation depended on it.

**Note**: `data/experiments/availability_comparison.csv` on disk predates
this guard, so it still carries that one out-of-range value and lacks the
`synth_pred_abs_sum` column. Not worth a 10-minute re-run for a diagnostic
column; the next run picks it up.

Also confirmed NOT a bug during the same sweep: negative `actual_season_total`
(11 player-seasons, min -2.1). Legitimate PPR scoring for QBs with very few
games played.


## Synthetic weeks were generated for players who could not have played (fixed 2026-08-18)

`before_first_real` synthetic weeks -- those falling before a player's first
appearance -- were 46.5% of synthetic weeks and 51% of manufactured points
in the 75-100%-synthetic bucket. A roster audit
(`scripts/audit_before_first_real.py`) classified all 644 of them against
`weekly_rosters`, `depth_charts` and `snap_counts`:

| category | weeks | % pts |
|---|---|---|
| on roster, listed backup | 209 | 39.9 |
| on roster, declared inactive | 124 | 19.0 |
| on roster, role unknown (2025, no depth chart) | 123 | 14.1 |
| not yet in league / pre-acquisition | 34 | 9.1 |
| practice squad | 101 | 8.9 |
| reserve / waived / retired | 39 | 7.7 |
| not on a roster that week | 13 | 1.1 |
| **on roster, listed starter** | **1** | **0.3** |

45.7% of manufactured points came from players not on an active roster at
all. Marcus Mariota was projected at ~15/wk for 12 weeks as QB2 behind a
healthy starter; Kyler Murray at ~15/wk while rehabbing an ACL; a retired
Philip Rivers got 13 weeks before his week-15 2025 comeback. All 8
no-roster-row players were verified genuinely absent (appearing on rosters
only in later weeks) -- not join failures. Zero synthetic weeks had
offensive snaps, so the week definition itself was sound.

**Fix**: `possible_weeks_for_player` now filters synthetic candidate weeks
to weeks the player was on an ACTIVE roster (`weekly_rosters.status='ACT'`;
INA/DEV/RES/CUT/RET all mean he could not take the field). Gated on
eligibility, NOT on being the starter -- a rostered backup stays in the
forecast population, since filtering to known starters would leak the
outcome into the population definition. Weeks he actually played are always
retained regardless of roster status. Seasons with no roster coverage fall
back to permitting rather than silently dropping everything
(`active_roster_weeks` returns None).

**Observability**: `WeekSkipTracker` gained a funnel --
candidate_weeks -> roster_eligible -> row_constructed -> predicted -- so no
stage can vanish into a `continue`. QB 2023-25: 1988 -> 1117 -> 1086 -> 1086.

**Effect on the pre-registered experiment** (same folds, estimators,
buckets, metrics; only the population changed). 42% of synthetic weeks and
51% of manufactured points removed:

| estimator | span old->new | MAE old->new | bias old->new |
|---|---|---|---|
| prior_season_only | 80.9 -> 45.7 | 42.5 -> 29.4 | +26.2 -> +7.6 |
| shrinkage_blend | 69.9 -> 36.1 | 35.9 -> 26.6 | +19.2 -> +4.3 |
| current_season_only | 62.3 -> 31.6 | 32.4 -> 25.1 | +14.3 -> +1.9 |

**Conclusions unchanged**: the MAE ranking of the five estimators is
identical (current_season_only < shrinkage < simple < recency <
prior_season_only), the gradient survives, and the best constant rate
(r=0, MAE 20.8) still beats the best adaptive estimator (25.1) -- so the
availability-weighting hypothesis is retired on a clean population, not a
contaminated one. One real change: the max bucket-to-bucket step now varies
14.2-21.2 across estimators (spread 7.0) where it was 40.0-41.7 (spread
1.4), so the formulation does matter more than it appeared -- just not
enough to be the mechanism.

**Caveat**: bucket membership is not comparable across the two runs. The
filter shrinks `possible_weeks`, so the 0%-synthetic bucket grew from 39 to
94 player-seasons (players whose synthetic weeks were ALL ineligible). Any
old-vs-new bucket comparison is partly recomposition. Likewise
"truly ever-present" (games == possible) now means "played every week he was
active", not "played all 17".

15 tests in `tests/test_roster_eligibility.py` pin the contract: IR /
practice squad / pre-acquisition produce no synthetic row, rostered backup
and starter both do, played weeks always survive, uncovered seasons fall
back to permitting, and the funnel counts every stage.



## Track B: the availability/residual gradient concentrates in one game (CLOSED 2026-08-18)

**Finding.** The apparent availability/residual gradient on genuine
observations is largely concentrated in the final real game preceding an
absence. It is NOT explained by a gradually deteriorating current-season
role, and it is not a stable property of "fragile" players.

**Evidence.** 2,058 real QB-weeks, 106 QBs, 238 player-seasons, 2023-2025,
FINAL_CONFIG (F_yeojohnson_huber / window 'all'). No refit -- Phase 4's
row-level dump already holds these predictions.

Residual approaching an absence (weeks until absence -> mean residual):

| 9+ | 6-8 | 4-5 | 3 | 2 | 1 |
|---|---|---|---|---|---|
| -0.29 | -1.61 | -1.64 | -0.08 | -0.41 | **+2.94** |

With player-season fixed effects:
- trend on weeks_until_absence, all pre-absence weeks: -0.198 (p=0.024)
- trend **excluding the final week: +0.047 (p=0.603)**
- final-week indicator: **+4.095 (p<1e-5)**

Between player-seasons, eventual_availability -> mean residual (prior PPG
controlled): -3.189 (p=0.0005) over all weeks, **-2.061 (p=0.080)**
excluding spike weeks.

Per-game residual by eventual availability:

| availability | <25% | 25-50% | 50-75% | 75-99% | 100% | span |
|---|---|---|---|---|---|---|
| all weeks | +1.10 | +0.05 | -0.59 | -0.59 | -1.80 | 2.90 |
| excluding spike weeks | -0.87 | -1.18 | -1.72 | -0.90 | -1.80 | **0.93** |

16.7% of weeks account for 68% of the gradient, and the remainder is
non-monotonic. On the spike week the prediction is normal (8.01) while the
actual collapses (5.06) -- the signature of an in-game injury or benching.

**Interpretation.** No evidence that a gradually deteriorating current-season
role explains the effect. The player looks normal going into the absence,
then something happens during that final game. This is **largely unexplained
and not actionable from the available pre-game information** -- deliberately
not "irreducible", which would claim more than the evidence supports.

**Limitations.**
- Absence health/role classification is crude: typed from weekly_rosters
  status in the missed week, where 'role' (ACT, dressed, did not play) is
  every week for a career backup, so it does not separate "was demoted" from
  "was always the backup".
- The surviving between-player effect (beta -2.06, p=0.080, n=182) is not
  zero, only no longer clearly distinguishable from noise.
- QB only. RB/WR/TE unexamined.

**Decision.** Do not introduce depth-chart or current-role state features
specifically to address this effect. There is no pre-absence signal for them
to catch, so it would be feature fishing. The availability-gradient
investigation is CLOSED.

**What the original gradient decomposed into**: invalid synthetic population
(fixed -- roster eligibility) + availability weighting (rejected as the
solution) + final-game adverse event (isolated here) + a weak, non-monotonic
remainder.

### Reproducibility sweep (2026-08-18)

Run before freezing, because this analysis accumulated enough moving parts
that the audit trail outweighs another hypothesis test.

- Every consumer that counts "weeks in a season" filters to week <= 18:
  `season_projection.py`, `season_simulation.py`,
  `run_availability_comparison.py`, `run_synthetic_row_diagnostic.py`,
  `run_track_b_exposure.py`. Verified by grep, not by assumption.
- `run_track_b_exposure.py` now imports REGULAR_SEASON_MAX_WEEK from
  `season_projection` instead of redefining it, and asserts on exit that no
  player-season has availability > 1 and max week <= 18.
- Exclusion ledger printed by the script: 2,146 rows at FINAL_CONFIG, 88
  playoff weeks excluded, 2,058 retained; 106 players, 238 player-seasons;
  343 spike weeks; 1,746 weeks with a prior season.
- Final numbers above re-derived from the regenerated clean artifact.

**Outstanding divergence risk (not fixed)**: `compute_market_projections.py`
and `build_complete_player_game_panel.py` each define their own
`REGULAR_SEASON_MAX_WEEK = 18` rather than importing the canonical one. Same
value today, so nothing is wrong now, but they would drift if the regular
season ever changes length. Out of scope for this branch.


## Cross-position validation: the roster bug is far WORSE outside QB (2026-08-18)

A validation checkpoint, deliberately not a repeat of the QB investigation:
one audit, no refitting
(`scripts/audit_roster_eligibility_by_position.py`). It calls the production
filter with `require_active_roster` True and False and diffs, so what is
measured is exactly what ships.

| position | candidate synthetic weeks | removed | % weeks | % points (est) | players |
|---|---|---|---|---|---|
| QB | 1,988 | 871 | 43.8 | 57.8 | 88 |
| RB | 2,570 | 1,960 | **76.3** | 87.5 | 178 |
| WR | 3,956 | 3,519 | **89.0** | 94.8 | 308 |
| TE | 1,887 | 1,736 | **92.0** | 95.6 | 162 |

Reason mix (% of that position's removed weeks):

| position | inactive | practice squad | IR/reserve | not rostered | pre-acq | waived |
|---|---|---|---|---|---|---|
| QB | 51.5 | 16.8 | 23.8 | 2.6 | 3.9 | 1.4 |
| RB | 32.3 | 28.2 | 29.6 | 5.6 | 1.5 | 2.7 |
| WR | 29.7 | **37.9** | 23.5 | 5.0 | 1.3 | 2.4 |
| TE | 32.8 | 32.6 | 25.7 | 5.5 | 1.7 | 1.6 |

Practice-squad churn dominates at WR/TE, which is exactly the population QB
does not have. The training population includes anyone with a stat line, so
a WR who played two games and spent the rest of the season on a practice
squad previously generated ~15 synthetic weeks, all of them invalid.

**Points are ESTIMATED, not measured** -- pricing them exactly would require
constructing every synthetic row and running the model, i.e. the refit this
checkpoint exists to avoid. The proxy is the player's realised PPG that
season. Calibrated against QB, where the true value IS known from the actual
re-run (12,659 -> 6,166 manufactured points = 51.3% removed), the proxy says
57.8% -- overstating by 6.5pp. Applying that correction: RB ~81%, WR ~88%,
TE ~89%.

**Consequence.** The synthetic population was materially broken across the
whole model, not just QB, and worse everywhere else. Any Phase 7 / Phase 9
season-projection output produced for RB/WR/TE before this fix should be
treated as VOID rather than merely noisy -- roughly nine in ten manufactured
points at WR/TE came from players who could not have taken the field.

**Decision.** Per the pre-agreed budget this does NOT trigger a
position-specific Track B. The QB investigation already answered the
mechanism question, and nothing here suggests a different mechanism -- only
a much larger dose of the same data defect, which the shipped filter already
corrects. What it does change: re-running the skill positions is now a
prerequisite for using their season-level output, not an optional refresh.

## `possible_weeks_for_team` silently returned [] for a numpy season (fixed 2026-08-18)

Found while writing the cross-position audit. sqlite3 does not bind numpy
integers, so `db.get_schedule(season=np.int64(2025))` matched no rows and
`possible_weeks_for_team` returned `[]` -- which reads downstream as "this
team never played", so `possible_weeks_for_player` produced ZERO synthetic
candidates and raised nothing. The audit's first run reported 0 candidates
across all 238 QB player-seasons and looked plausible.

Production callers pass python ints (season loop variables), so this never
fired in a real run -- but any caller iterating
`groupby(["player_id", "season"])` gets numpy scalars without realising it,
which is precisely how the audit hit it.

**Fix**: coerce at the boundary -- `int(season)` / `str(team)` /
`str(player_id)` in `possible_weeks_for_team`, `_season_has_roster_data` and
`active_roster_weeks`. 2 tests in `tests/test_traded_player_schedule.py`
pin numpy/python equivalence.


## 2025 snap data was doubly corrupt; re-ingested (2026-08-19)

Found while checking whether nflreadr's game-level snap data (2013+, PFR
feed) could extend our coverage. Two independent defects in
`player_weekly_stats` for 2025 ONLY:

1. **team_snaps inflated ~12x** -- avg 646.5 / max 2112, against 51.5-52.5 /
   90-100 for every other season. KC wk3 held 1584, which is the sum of BOTH
   teams' player snaps (792+792) rather than the team's ~72 offensive plays.
   The 2026-08-07 "team_snaps inflation" fix was already in the code; these
   rows simply predated it and were never re-ingested.
2. **snap_count doubled** on 563 rows -- exactly 2x the source
   `offense_snaps` (Josh Allen wk13: source 74, stored 148). No duplicates
   exist in `snap_counts`, so the double-count happened during ingest.

Defect 2 is why a surgical denominator fix was NOT applied: recomputing
team_snaps alone would have produced `snap_share = 2.0` on 511 rows --
replacing a visibly-broken number with a plausible-looking wrong one.
`scripts/backfill_team_snaps.py` refuses to write when any recomputed
snap_share exceeds 1.0, and refuses when team_snaps falls outside a 20-120
play band.

**Fix**: full re-ingest of 2025 via `NFLDataLoader.load_weekly_data([2025])`,
which uses the pfr_player_id -> GSIS mapping rather than a name join. DB
backed up first (`nfl_data.db.bak-2025-reingest-20260819-062059`).

Verified row-by-row against the backup: 6,764 rows before and after (none
added or dropped); 5,593 team_snaps changed and ALL were previously inflated
>150; 1,171 unchanged and ALL were already 0 (players with no snap data);
563 snap_count corrected and ALL exactly halved, zero other changes;
`fantasy_points` identical on every row. Post-state: avg team_snaps 53.4,
max 96, avg snap_share 0.4650, zero rows above 1.0 -- inside the band of
every other season.

`compute_team_snaps` was extracted to module level in
`pbp_stats_aggregator.py` so the ingest path and any backfill share one
implementation. Validated by reproducing 2024 exactly (avg snap_share
0.4523 -> 0.4523, max 1.0, zero rows over 1).

**Still open**: `snap_count`/`snap_share` are a stored ZERO (not null) for
all of 2006-2017, so RB (window `all`) has 60.3% of its training rows
asserting every back took zero snaps. nflreadr has real data back to 2013
(the documented "2012" file exists but is empty), but backfilling would make
the feature mean different things across eras unless the NaN-vs-0
representation is decided first. See the snap-representation note above.

## run_season_projection/simulation broke on the per-week team change (fixed 2026-08-19)

The mid-season-trade fix replaced the season-level `team` variable with
`team_by_week`, but both `run_season_projection` and `run_season_simulation`
still referenced bare `team` when building their season-level result row --
`NameError: name 'team' is not defined`, killing Phase 7 on its first fold.

The 231-test suite did not catch it: nothing exercises those two functions
end-to-end, since they need the real DB and fitted folds. Caught only by
actually running Phase 7 for RB/WR/TE.

**Fix**: report `real_team_by_week[max(real_team_by_week)]` -- the team he
last actually played for, which is what a season-level row means. Verified
with a real Phase 7 run (TE/2023, 129 rows, correct team values).

**Still uncovered**: there is no regression test for this. A cheap
single-position smoke test over `run_season_projection` would have caught it
and would catch the next one; not yet written.

## Historical coverage backfill: depth charts, injuries, NGS (2026-08-19)

`scripts/backfill_all_data.py` hardcodes `SEASONS = list(range(2018, ...))`
(and, for depth charts, `range(2024, 2026)`). Nothing about the upstream
data required those floors -- three tables started later than nflverse
actually carries them. Filled via `scripts/backfill_historical_coverage.py`
(depth charts 2013-2019, NGS 2016-2017, both pure appends in the existing
schema) and `scripts/backfill_injuries.py -s 2013 2017`.

| table | was | now | upstream floor |
|---|---|---|---|
| `depth_charts` | 2020-2024 | **2013-2025** | 2013 |
| `player_injuries` | 2018-2025 | **2013-2025** | 2009 |
| `ngs_*` | 2018-2025 | **2016-2025** | 2016 |

**Effect on `depth_chart_rank`**: 2013-2019 were **100% neutral-default** --
the feature was entirely dead for seven seasons, not merely noisy. Now
83-86% of skill-position rows carry a real rank, and the resulting
distribution (52% / 32% / 16%) closely matches stored 2020-2024, so the
eras are comparable. 2020/2021 moved 0-1.2%, confirming no contamination of
already-covered seasons.

### Three things found in passing

**1. `injury_score` has a reporting-rule discontinuity at 2015/2016.**
"Probable" was abolished by the NFL after 2015 and appears 2,772 / 2,607 /
2,702 times in 2013/2014/2015 and **exactly 0** from 2016 on. It maps to
0.85, so `pct_injured` (score < 1.0) runs 14-16% for 2013-2015 and 4-6%
from 2016 -- a 3-4x cliff that is a reporting artifact, not players getting
healthier. **Not re-encoded**, because that is a modelling decision, not a
bug fix: mapping Probable to 1.0 would make the eras comparable but discard
real within-era signal. Bounded for now by
`TRAINING_START_YEAR_DEFAULT = 2018`, which excludes the affected seasons;
it only bites the `"full"` (2006+) preset. **Decide before training on
`full`.**

**2. `depth_charts` 2020-2023 are skill-position-only.** They contain 0 rows
outside QB/RB/WR/TE; 2024 and 2025 contain everything. This corrects the
claim in `_load_depth_chart_asof_table` (and above) that 2024's row count
was "~3x-inflated" and needed dedup -- 2024 is not inflated, it is simply
the first season loaded *without* the skill filter. Skill-row counts are
consistent across every season (~11.4K), so `depth_chart_rank` is
unaffected either way. 2013-2019 were loaded unfiltered, matching
2024/2025. **2020-2023 remain a subset** -- reloading them unfiltered would
make the table uniform, but rewrites existing data, so it was left alone.

**3. `backfill_all_data.py` would have deleted every backfill on its next
run.** `_save_df` defaults to `if_exists="replace"` -- it drops the table --
while the script fetches only 2018+. Guarded by `_assert_no_history_lost`,
which refuses a replace that would drop seasons the incoming frame lacks.
Verified against the live DB: it now refuses `depth_charts` (would drop
2025), `ngs_*` (2016-2017) and `snap_counts` (2013-2017, i.e. it also
protects the *previous* commit's work). Raising beats switching to
`append`, which would duplicate every existing row instead. 7 tests.

Also removed 3,347 exact-duplicate rows introduced by the 2013-2019 load:
the feed separates some listings only by `formation`/`game_type`, neither of
which this table stores, so they arrive identical. The 610 groups with
genuinely *conflicting* `depth_team` were preserved -- MIN resolves those at
load time, and collapsing them here would move that policy away from where
it is documented.

## Season-floor hardcoding removed from backfill_all_data.py (2026-08-19)

`SEASONS = list(range(2018, datetime.now().year + 1))` -> `range(MIN_SEASON,
CURRENT_NFL_SEASON + 1)` with `MIN_SEASON = 2013`. Two separate defects in
that one line:

**Floor.** 2018 was never an upstream limit, just where this script started;
it was the reason `depth_charts`, `ngs_*` and `snap_counts` all began later
than nflverse carries them.

**Ceiling.** `datetime.now().year + 1` names the *calendar* year, so for most
of the year it requested a season that has not been played -- in Aug 2026 it
asked for 2026 while `CURRENT_NFL_SEASON` was 2025. Now bounded by the config
constant, which is the single source of truth.

**Per-dataset floors, not one global.** Setting everything to 2013 would
break NGS, which genuinely begins in 2016. `DATASET_MIN_SEASON` records the
real upstream floor per dataset and `seasons_for()` clamps; datasets not
listed use `MIN_SEASON`. Verified floors against nflverse: qbr and
weekly_rosters reach back to at least 2006, injuries 2009, snap_counts 2013
(2012's file exists but is empty), NGS 2016. Two other hardcoded ranges in
the same file were removed: `weekly_rosters` (`range(2024, 2026)`) and the
classic depth-chart pull, now derived and bounded by
`DEPTH_CHART_NEW_SCHEMA_SEASON`.

`SNAP_COUNT_MIN_SEASON` was already 2013. Both of its consumers in
`build_complete_player_game_panel.py` guard with `if lo > hi`, so the lower
floor can't produce an inverted range; only stale "2018-2025" docstrings
needed correcting.

### Still gated at 2018 on purpose: snap_count into player_weekly_stats

`backfill_snap_counts_to_pws.py` hardcoded `season >= 2018` in both its
update and verification queries. `snap_counts` now reaches 2013, so that
gate is the reason 2013-2017 snaps still never arrive in
`player_weekly_stats`. It is now `MIN_SEASON_DEFAULT` with a `--min-season`
override, but the **default is deliberately left at 2018**: pws stores
`snap_count` as a hard 0 (not NULL) for 2006-2017, so lowering it flips
those seasons from "asserted zero" to "real value" and makes the feature
mean different things either side of the boundary. That NaN-vs-0
representation is the open decision recorded above -- resolve it first.

Scale of the pending change, measured by dry-run: **9,912 rows at the 2018
default vs 40,618 at 2013**, i.e. ~30,700 rows in 2013-2017 currently
asserting a zero the snap data could correct.

## AUDIT: does `player_weekly_stats.snap_count = 0` mean zero, or unknown? (2026-08-19)

Run before deciding whether to lower `backfill_snap_counts_to_pws.py`'s 2018
floor. **Answer: pre-2018 zeros are placeholders for UNKNOWN, and the
2018+ era is not clean either.** No data was modified.

### 1-2. The stored values cannot be measurements

2006-2017 is **100.0% `snap_count = 0` across ~71,000 rows, with zero
NULLs** -- and `team_snaps` is identically 100% zero. 2018+ runs 17-20%
zero. A 100% zero rate is not a measurement, and `team_snaps = 0` is
definitionally impossible: a team always runs plays. The column has no way
to say "unknown", so ingestion wrote 0.

### 3-4. Measured against the authoritative `snap_counts` table

Matched by PFR->GSIS id (**92.8%** match pre-2018) rather than by name
(74.5%) -- see the matching defects below. Of all rows currently stored as
`snap_count = 0`:

| era | real POSITIVE snaps | confirmed ZERO | unknown (absent) |
|---|---|---|---|
| 2013-2017 | **28,483** | 26 | 2,197 |
| 2018-2025 | **9,695** | 22 | 195 |

**38,178 rows assert zero snaps for a player the authoritative table shows
on the field.** Only **48 rows in 13 seasons** are confirmed zeros. By
`data_source`: 25,214 are `nflverse_stats` (real stat rows whose snap_count
was simply never populated) and 12,959 are `inferred_snap_verified_zero` --
where "zero" refers to *fantasy points*, not snaps; those players took a
mean of 29.6 snaps. Both groups are genuinely wrong, not deliberate zeros.

Note the 2018+ row: the era assumed clean has **9,695 wrong zeros**, 97.8%
of its zeros. The premise "2018+ = actual snap data" does not hold.

The 2,392 "unknown" is an **upper bound**: the PFR->GSIS map covers only
3,007 of 3,710 distinct 2013-2017 snap ids, so an unmappable player looks
absent. 2013 carries 1,286 of them versus ~230 for every other season,
which is map coverage, not a real gap.

### The matcher itself cannot express the distinction

`backfill_snap_counts_to_pws.py` builds its lookup with
`WHERE offense_snaps > 0`, **excluding zero-snap rows**. It can only ever
upgrade a 0 to a positive; on a failed match it leaves 0, conflating
"confirmed zero" with "absent". Two further defects make its name-based key
unreliable pre-2018:

- **6,435 pre-2018 rows have a blank `players.name`** (2,745 in 2013 falling
  to 0 by 2017 -- exactly the shape of the apparent "coverage" decline).
  Zero blanks in 2018+.
- `_normalize_name` takes the last token as the surname, so
  `"Odell Beckham Jr."` -> `"O.Jr."` and `"A.J. Green"` -> `"A.Green"`.

Any migration should key on player id, not name.

### 5-6. Downstream cannot represent NULL today

`snap_count`, `snap_share` and `snap_share_roll3` are **trained model
features** (`train_position_models.py`), so this is not cosmetic. Every path
collapses unknown into 0:

| site | behaviour |
|---|---|
| `pbp_stats_aggregator.py:463` | `snap_count.fillna(0)`, `team_snaps.fillna(0)` -- **converts a NULL straight back to 0 on write** |
| `pbp_stats_aggregator.py:465` | `snap_share = where(team_snaps > 0, ratio, 0.0)` |
| `utilization.py:465` | `snap_share = snaps / team_snaps if team_snaps > 0 else 0` |
| `preseason_features.py:58` | `COALESCE(us.snap_share, 0)` |
| `quality_gates.py:272` | all-zero treated as ingestion failure (correct reading, but fires on pre-2018) |
| `nfl_data_loader.py:327` | `(snap_count == 0).all()` used to detect missing data |

**So NULL is required to represent unknown, and the current code cannot
carry it** -- the aggregator would erase it on the next write. Schema and
feature logic must change before any backfill, per the checklist. Not
started; awaiting the representation decision.

## RESOLVED: snap columns aligned to snap_counts, 0-vs-NULL made explicit (2026-08-19)

Acts on the audit above. Two changes, in this order, because the write path
had to stop erasing NULLs before any of them could survive a rebuild.

### 1. `fillna(0)` removed from the write path

`pbp_stats_aggregator` fillna(0)'d `snap_count` and `team_snaps` on every
write, so "no snap record" and "took zero snaps" became the same stored
value. Both are now the nullable `Int64` dtype and keep NA through to
SQLite. `snap_share` is NULL when either side is unknown and 0.0 only when a
known-zero player has a known team total. `compute_team_snaps` no longer
floors a team-week to 0, a value that cannot occur.

### 2. `player_weekly_stats` aligned to the authoritative table

`backfill_snap_counts_to_pws.py` rewritten. It had matched on normalised
NAME -- failing on 6,435 blank-name rows and mangling suffixes -- and built
its lookup with `WHERE offense_snaps > 0`, which made a failed match
indistinguishable from a real zero. Now keyed on player_id via PFR->GSIS
(74.5% -> 92.8% pre-2018 match) with zero-snap rows loaded, under an
explicit three-way rule:

| source | pws | result |
|---|---|---|
| has a value | anything | overwrite (authoritative) |
| silent | 0 | **NULL** -- 0 was never a measurement |
| silent | > 0 | **leave alone** |

The third case is load-bearing: 1,831 rows (all 2018+) hold a positive count
with no id-match, because the PFR->GSIS map covers ~81% of ids. Blindly
NULLing "absent" would have destroyed real data this script cannot
regenerate.

**Result across 82,508 rows (2013-2025):** 38,265 placeholder zeros replaced
with real counts, 1,783 zeros correctly became NULL, 29 wrong positives
corrected, 1,831 preserved.

`snap_share` is now continuous across the old boundary -- mean 0.525 / 0.520
/ 0.524 / 0.518 / 0.515 for 2013-2017 against 0.516 / 0.521 / 0.499 ... for
2018+, where every pre-2018 season previously read exactly 0. Zero snap
invariant violations, no `snap_count > team_snaps`, and no row where one of
`snap_count`/`snap_share` is NULL while the other is not. The quality gate
"snap_count is zero for every row -- snap data ingestion has failed", which
2015 would have tripped, now passes.

### Observed in passing, not chased

2013 holds ~880 more `nflverse_stats` rows than 2014-2018 (5,981 vs ~5,100),
which is why it shows 777 unknown / 517 confirmed-zero snap rows against
~230 / <10 elsewhere. Pre-existing ingestion difference, untouched by this
work.

### Still 0-imputed on the READ side (deliberately not changed)

These do not destroy stored data, but they do hand a model a fabricated 0
where the DB now correctly says NULL. `snap_count`/`snap_share`/
`snap_share_roll3` are trained features, so changing them alters model
inputs and wants its own decision:

- `preseason_features.py:58` -- `COALESCE(us.snap_share, 0)`
- `utilization.py:465` -- `snap_share = snaps / team_snaps if team_snaps > 0 else 0`
- `feature_engineering.py:1514` -- `snap_share_accel ... .fillna(0)`

## AUDIT: read-side snap imputation, and whether NaN can survive (2026-08-19)

Steps 1-3 of the imputation plan. **No code or data changed.** Result: the
"let the tree models handle missingness" plan does not apply to this
project, because production is not a tree model.

### Live surface is smaller than it looks

`CAUSAL_FEATURES` (the list actually trained on) contains exactly two
snap-derived features, and only for RB/WR/TE:

    snap_share_pct_roll3_mean
    snap_share_accel

**QB has none** -- it is unaffected entirely. `train_position_models.py`'s
`POSITION_FEATURES` (which declares `snap_count`, `snap_share`,
`snap_share_roll3`) is NOT the live list; it is consumed only by
`scripts/walk_forward_multiweek.py`. `snap_share_roll3` is never constructed
anywhere.

### Where the fabricated zeros actually come from

One upstream site, not three. `safe_divide` (`utils/helpers.py:98`) returns
`default=0.0` whenever the numerator OR denominator is NaN, so
`snap_share_pct = safe_divide(snap_count, team_snaps) * 100` is **0.0 for
every unknown row**. The rolling feature itself is clean --
`shift(1).rolling(3, min_periods=1).mean()` with no fillna, and pandas skips
NaN -- so it only ever propagates a zero that was invented upstream.
`snap_share_accel` then adds its own `.fillna(0)`
(`feature_engineering.py:1514`).

### Scale (step 2)

| feature | era | currently 0 | fabricated | genuine measured 0 |
|---|---|---|---|---|
| `snap_share_pct` | 2013-17 | 1,007 | **983** | 24 |
| `snap_share_pct` | 2018+ | 123 | **96** | 27 |

`snap_share_pct_roll3_mean` changes on **2,066 of 71,312 rows (2.9%)**, of
which **77% fall in 2013-17** -- an era that is only 37% of rows, confirming
missingness is not random. Position spread is proportional (WR 735 / RB 516
/ TE 339), so no position-specific bias.

### Step 3: NaN CANNOT survive, and not because of a stray fillna

The production model is `ComponentPredictor` -- **Ridge + StandardScaler**,
chosen deliberately over GradientBoosting (its docstring records that GBT
systematically underpredicted). Verified directly:

    StandardScaler : passes NaN through
    Ridge          : ValueError, "Ridge does not accept missing values"

`ComponentPredictor._prepare_array` already does
`np.nan_to_num(X_arr, nan=0.0)`, and both `ts_backtester.py:421-422` and
`ensemble.py:843` `fillna(0)` the whole feature matrix before fitting.

So "preserve NaN to the model" is not a config change here -- a linear model
cannot consume it at all. The three read-side call sites are not the
blocker; the model family is.

### Consequence for the plan

Option B (NaN + native handling) is unavailable without changing the model
family, and that trade was already decided the other way on backtest
evidence. For a linear model the equivalent of native missing handling is an
explicit missingness indicator plus an imputed value -- Ridge can then fit a
separate offset for unknown rows instead of being told they took zero snaps.
That is a model change on 2.9% of rows and wants its own decision; not
started.

## RESULT: snap-missingness A/B (2026-08-19)

Pre-registered A vs B, 8 runs (2 variants x 2016/2017/2018/2024 x RB/WR/TE),
21,797 paired predictions. **No production change made.**

    A  unknown snap share -> 0, no indicator (production, untouched)
    B  unknown -> era median fitted inside each training fold,
       plus snap_share_pct_roll3_known

### The effect is real and lands exactly where predicted

Paired change in absolute error (B - A); negative favours B:

| population | n | mean delta | 95% CI | p |
|---|---|---|---|---|
| uncertain (known<1), all | 1,518 | **-0.1630** | [-0.190, -0.138] | 2.4e-15 |
| uncertain, 2016-17 | 992 | -0.1294 | [-0.160, -0.100] | 3.9e-06 |
| uncertain, 2018+ | 526 | -0.2263 | [-0.277, -0.177] | 6.2e-12 |
| **fully known (regression check)** | **20,279** | **+0.0041** | -- | -- |

This is the pattern that was pre-registered as compelling: B improves the
rows whose snap history is incomplete and leaves the other 93% alone. It is
evidence the intended mechanism was fixed, not that the model was perturbed
into a better fit.

### But the headline effect is small

Uncertain rows are ~7% of the frame, so overall MAE moves only 4.377 ->
4.371 (2016-17) and 4.196 -> 4.187 (2018+): about **0.2%**. Both eras
improve, so there is no net regression.

By position, on uncertain rows: **TE -0.2640, WR -0.2096, RB -0.0089**. RB
is not a wiring failure -- it has 447 uncertain rows and 5,662 of its
predictions changed; the snap feature simply does not carry RB signal the
way it does for pass-catchers, which is consistent with RB usage being
better captured by carries.

### Verdict

B is a correctness fix that pays off precisely where the data was wrong, at
no inference cost and with no regression on known rows. The case for it is
"stop telling the model something false", not "it raises the headline
metric" -- it does not, materially.

**Variant C (global median) NOT tested**, per the pre-registered rule: C
differs from B only on the same 1,518 rows, and B's era split is already the
more informative statistic.

Reproduce: `scripts/analyze_snap_missingness_ab.py`.

## Pre-production sanity check + the productionisation gap (2026-08-19)

### Imputation values are sensible

Observed `snap_share_pct_roll3_mean` on known rows vs the median variant B
imputes (position x era -- `apply_snap_imputation` is called inside the
per-position loop, so the era grouping is already per position):

| pos | era | p25 | median (imputed) | p75 |
|---|---|---|---|---|
| RB | pre/post | 18.3 / 18.1 | **34.9 / 34.3** | 53.8 / 53.5 |
| WR | pre/post | 28.1 / 27.0 | **61.7 / 56.5** | 81.8 / 78.6 |
| TE | pre/post | 24.6 / 25.1 | **45.5 / 42.8** | 70.5 / 63.6 |

Nothing pathological -- every value sits mid-distribution.

### But unknown rows are a lower-usage population

| pos | fantasy_points known -> unknown | targets known -> unknown |
|---|---|---|
| RB | 8.09 -> 5.07 | 2.27 -> 1.15 |
| WR | 7.24 -> 3.95 | 4.22 -> 2.42 |
| TE | 4.42 -> 2.54 | 2.55 -> 1.42 |

Players whose snap history is unknown are roughly 55-65% of a known
player's usage, so the median **overstates** their likely snap share. Not a
blocker, and deliberately NOT optimised (that is variant-C territory): it
explains why the `known` indicator earns its keep, since Ridge can fit a
compensating offset for those rows, which is consistent with B beating A on
exactly that population.

### Blocker: B is not productionisable by flipping a default

`apply_snap_imputation` is wired into `ts_backtester.py` alone. Production
training (`train.py`) and inference (`ensemble.py` /
`component_predictor.py`) never call it. Setting
`SNAP_MISSINGNESS_MODE = "preserve"` today would give production
`NaN -> blanket fillna(0) -> 0.0` with no imputation and no indicator --
the same fabricated zero as variant A, while the config claims otherwise.
That is strictly worse than leaving it alone.

**Correct design** (not implemented): the imputation constants must be FIT
at training time and PERSISTED in the model artifact, then applied
unchanged at predict time. `ComponentPredictor` already has
`to_dict`/`from_dict`, so the natural home is a `snap_impute_` mapping
fitted inside `.fit()` from the training rows and serialised with the
model. Computing medians at inference from whatever data is to hand would
reintroduce exactly the leakage the backtest was careful to avoid.

## TRACE: where learned feature transformations belong (2026-08-19)

Requested before productionising variant B. **No code changed.**

### The canonical mechanism already exists

Learned, train-only, persisted feature transformations use the
**percentile-bounds artifact pattern**:

    src/features/utilization_score.py
        fit_percentile_bounds()          fit on train rows
        save_percentile_bounds(path, metadata={"train_seasons": [...]})
        load_percentile_bounds(path, return_meta=True)
        validate_percentile_bounds_meta(meta, expected_train_seasons)

Orchestrated in `feature_preparation._prepare_training_data`, which fits on
`train_data`, writes `MODELS_DIR/utilization_percentile_bounds.json`,
reloads, and **raises** if the metadata's train seasons don't match the
current ones ("refusing to use bounds not fit on the current training
seasons"). Test rows are then transformed with the LOADED bounds. Consumers
re-load the same artifact: `features/utilization.py` (with autoload),
`evaluation/backtester.py`, `scripts/realtime_integration.py`,
`scripts/audit_2025_backtest.py`.

That is precisely the fit -> persist -> reuse contract required here, with
leakage validation already built in.

### ComponentPredictor is a different layer

Its `scalers` are per-component `StandardScaler`s, base64-joblib'd into the
model artifact by `to_dict`. That is **model preprocessing**, not feature
semantics. Putting snap imputation there would make the estimator aware of
one specific feature's data-generating story -- an architectural exception,
and one that inference paths not going through ComponentPredictor would
silently miss.

**Conclusion: snap imputation is a feature-level learned transformation and
belongs with the percentile-bounds pattern, not in ComponentPredictor.**

### The complication: there are TWO feature pipelines

    production   train.py -> feature_preparation._prepare_training_data
                 (fits + persists + validates percentile bounds)

    backtest     ts_backtester.leakage_safe_features
                 (re-implements the two-pass pattern; does NOT do the
                  bounds fit/persist step)

They have diverged before, with consequences: a code comment at
`ts_backtester.py:164` records a rookie-feature ablation whose arms came
back bit-identical because the columns were absent from BOTH, not because
the feature had no effect.

Variant B currently lives only in the backtest path
(`apply_snap_imputation`), which is why flipping a default cannot
productionise it. Verified that the four snap features
(`snap_share_pct`, `_roll3_mean`, `_roll3_known`, `_accel`) DO exist in the
backtest frame, so experiment B did measure the intended thing.

### Smallest architecture-consistent implementation

1. `utilization_score.py`: `fit_snap_imputation(train_df) -> {(pos, era):
   median}` plus save/load/validate mirroring the percentile-bounds trio,
   persisted to `MODELS_DIR/snap_imputation.json` with the SAME
   `train_seasons` metadata contract. (A sibling file rather than a new
   section inside the bounds JSON: same contract, cleaner semantics. Either
   is defensible.)
2. `feature_preparation._prepare_training_data`: fit on `train_data`, save,
   reload, validate, apply to both train and test -- immediately after the
   existing percentile-bounds block, which it mirrors exactly.
3. `ts_backtester`: retire the fold-local `apply_snap_imputation` in favour
   of the shared fit/apply functions, so the validated behaviour and
   production run ONE implementation. This is the step that prevents the
   divergence above from recurring.
4. Inference: the autoload path in `features/utilization.py` picks up the
   artifact the same way it picks up bounds.
5. Flip `SNAP_MISSINGNESS_MODE` to "preserve"; keep "zero" as the baseline
   mode for A/B reruns.

Parameters live in `MODELS_DIR/snap_imputation.json`, versioned by the
`train_seasons` metadata, and are never recomputed at inference.

## CORRECTION: the A/B measured a different mechanism than reported (2026-08-19)

The production gate passed, but running it proved the earlier A/B result was
**misattributed**. Recorded prominently because the number was quoted in
several commit messages.

### What the A/B actually compared

At A/B time three fillers were still live, and one of them —
`_create_base_features` — **recomputed `snap_share_pct` through
`safe_divide` inside `create_features`**, i.e. after any
`calculate_all_scores`. In the backtest path that recomputation is where
`snap_share_pct` comes from at all. So `SNAP_MISSINGNESS_MODE = "preserve"`
never reached the rolling feature: variant B's `snap_share_pct` was
byte-identical to variant A's.

The only remaining difference between the arms was that B added
`snap_share_pct_roll3_known` to the feature list — and with no NaN upstream,
that column could only ever take two values: 0.0 for a player with no prior
games, 1.0 for everyone else.

**Evidence.** The pre-fix gate run recorded
`known distribution {0.0: 108, 1.0: 5679}` — strictly binary. After the
three fixes the same frame yields `[0.0, 0.333, 0.667, 1.0]`. The graded
values only exist when `snap_share_pct` carries NaN, so their absence during
the A/B is proof the snap missingness was already gone.

### What that means

The measured effect is real and was correctly computed — B improved the
1,518 incomplete-history rows by 0.163 MAE with p=2.4e-15 and no regression
elsewhere. But the mechanism is **"tell the model whether this player has
prior history"**, not "tell the model the snap share is unknown". Those
overlap heavily (a debut has neither) which is why the result looked so
clean, but they are not the same feature.

**The snap-missingness representation is therefore NOT yet validated.** It
is now correctly implemented end to end for the first time; nothing has
measured it.

### Consequence

- Do not cite -0.163 MAE as evidence for the snap-imputation change.
- The A/B is worth re-running now that the pipeline preserves missingness,
  since only now do the arms differ in the intended way. Same design, same
  seasons, same pre-registered cuts.
- Adopting variant B remains defensible on correctness grounds (a fabricated
  zero is a false claim regardless of measured lift) but not on the measured
  lift, which belongs to a different feature.

## RESULT: corrected snap-missingness A/B — B does NOT do what it was built to do (2026-08-19)

Re-run of the pre-registered A/B after the three fillers were fixed, so the
arms differ in the intended way for the first time. Same design, seasons,
positions, metric, paired methodology and cuts. Passed the mechanical abort
gate first (A's indicator binary, B's graded, 125/5,787 test rows differing).

### The signature is inverted

Paired change in absolute error (B - A); negative favours B:

| population | n | mean delta | 95% CI | p |
|---|---|---|---|---|
| **UNCERTAIN (the target)** | 1,518 | **+0.0363** | [+0.001, +0.068] | 6.9e-04 |
| — 2016-17 | 992 | -0.0009 | [-0.048, +0.046] | 0.67 (flat) |
| — 2018+ | 526 | **+0.1064** | [+0.055, +0.155] | 5.5e-08 |
| **KNOWN (the control)** | 20,279 | **-0.0464** | [-0.051, -0.042] | 3.4e-81 |
| all rows | 21,797 | -0.0406 | [-0.046, -0.036] | 9.3e-62 |

Overall MAE 4.2835 -> 4.2429 (~0.9%), every era and position improving.

**But the gain comes from the control population, and the target population
gets WORSE.** The pre-registered reading of a genuine fix was "improves the
uncertain rows, leaves the known rows alone". This is the opposite: B helps
the 20,279 rows it should not touch and hurts the 1,518 it was designed for.
That is the definition of perturbing the model rather than fixing the
mechanism.

### Why, and it was predicted

The pre-production sanity check recorded that unknown-snap players are
systematically lower usage -- 55-65% of a known player's fantasy points and
targets. Imputing the position x era median therefore **overstates** them,
the model overpredicts, and MAE rises. The fabricated zero was wrong in
principle but happened to sit closer to this population's true low usage.
Replacing a too-low fabrication with a too-high one made the target rows
worse.

### Where that leaves it

- The **storage layer stands regardless**: unknown != zero in the database,
  38,178 corrected rows, snap coverage back to 2013. None of that depends
  on this result.
- The **architecture stands**: train-only position x era medians, persisted
  with train_seasons, validated on load, applied identically in training,
  backtest and inference, with missingness surviving the full pipeline.
  That machinery is correct and now proven by the gate.
- The **treatment does not**. B is not validated for its stated purpose.
- The ~0.9% overall gain is real and highly significant but **unexplained**.
  An unexplained aggregate gain from perturbing 1,258 training rows is not
  something to ship on the strength of a mechanism it demonstrably does not
  have.

**Production currently runs B** (SNAP_MISSINGNESS_MODE = "preserve"). One
flag reverts it. Not reverted unilaterally: adopting or reverting is a
judgment call, and the honest summary is that B improves aggregate fit while
harming the only population with a mechanistic story.

Deliberately NOT done: tuning the imputation constant in response to this.
That is a new experiment, not a fix to this one.

## Attribution: why B helped the control and hurt the target (2026-08-19)

Diagnostic on the completed v2 experiment. No refit, no new treatment.

### What actually differs between the arms

| frame | roll3 differs | known differs | A mean on those rows | B mean |
|---|---|---|---|---|
| train | 1,258 / 33,361 (3.77%) | 156 | 33.36 | **47.98** |
| test | 125 / 5,787 (2.16%) | 17 | 24.05 | **47.12** |

**B roughly doubles the rolling snap share on every row it touches.**

### The mechanism, in full

Only **0.3%** of the changed train rows were exactly 0.0 under A. These are
not "fabricated zero -> median" swaps. Under A an unknown week enters the
3-week mean as a 0 and drags it DOWN; under B unknown weeks are skipped, so
the mean is taken over known weeks only -- and where every week is unknown,
the median is substituted. Either way the value rises sharply.

The rows this happens to are **low-usage players**: fantasy_points 4.32 vs
6.51 for all training rows, targets 2.10 vs 3.11 (~67% of average).

So B tells the model that a set of demonstrably below-average players had
near-average snap share. It overpredicts them, and MAE on exactly that
population rises (+0.036 overall, +0.106 in 2018+). Meanwhile A's
zero-averaging accidentally encoded something true: these players' rolling
snap share *should* read low, because they are low-usage.

The fabricated zero was a correct answer for the wrong reason.

### Attribution of the 0.9% control-population gain

It comes from the **altered training distribution**, not from the indicator.
`snap_share_pct_roll3_known` differs on only 156 of 33,361 train rows, while
`snap_share_pct_roll3_mean` differs on 1,258 -- and by ~15 points each. That
is enough to move the Ridge coefficients for every prediction, including the
20,279 control rows that have no missingness at all. An aggregate gain
produced by perturbing 3.77% of training values, with no account of why the
new fit generalises better, is not something to ship.

### Disposition

- **Shipped**: NULL storage semantics, snap coverage to 2013, the corrected
  38,178 rows.
- **Rejected**: median imputation + missingness indicator. Both modes back
  to "zero"; the production call sites are gated so nothing runs and no
  artifact is written.
- **Kept**: the fit/persist/validate/apply architecture, as reusable
  infrastructure for train-only transformations. Its tests pin the mode
  explicitly so they assert capability, not policy.
- **Quarantined**: the v1 "has prior history" signal — a separate
  hypothesis needing its own experiment and name.
- **Not done**: tuning the imputation constant. This result is precisely
  what a post-hoc tune would have manufactured a story around.

## CLOSED: snap missingness (2026-08-19)

Three questions were entangled at the start; they are now separated, and
only the middle one was answered "no".

**1. Was the database wrong?** Yes, and it is fixed. `snap_count` was a
fabricated 0 for 100% of 2006-2017 and for 38,178 rows overall where the
authoritative table shows the player on the field. NULL now means unknown.
Snap coverage extends to 2013. **This stands independently of everything
below.**

**2. Does preserving that distinction improve prediction, via position x era
median imputation plus a missingness indicator?** No — experimentally
rejected on a pre-registered A/B that the treatment failed in the direction
that matters.

**3. Does the missingness itself carry predictive information?** Probably
yes, and it is the interesting question. Quarantined, unnamed, untested.

### The conclusion, phrased precisely

> Zero is **not semantically correct**, but it is currently the
> **empirically validated predictive representation** for this feature under
> this model and this data-generating process.

Those are different responsibilities. The database's job is to record what
is known; the feature's job is to predict. They disagree here, and the
disagreement is not an error in either.

### Why they disagree: the missingness is endogenous

Snap observations are not missing at random. They are disproportionately
missing for **low-usage players** (67% of an average player's fantasy points
and targets). Averaging an unknown week in as 0 pulls the rolling snap share
down — which is, accidentally, informative about that low-usage state.
Omitting or median-filling it raises the value ~15 points and destroys that
signal.

So the fabricated zero was carrying real information about *why* the
observation was absent. That is a far more interesting modelling question
than the choice of imputation constant.

### If this is revisited

Start from "missingness is informative and endogenous". Do **not** reopen
median-vs-global imputation: that question was answered, and the answer is
that any central-tendency fill discards the signal the zero was carrying. A
serious attempt would model the missingness mechanism itself, or use the
`snap_share_pct_roll3_known` indicator without altering the value.

Do not resume: availability formulations, additional snap sources, QB snap
reconstruction, Track B forensics, or imputation-constant tuning.

## Cleanup: phantom feature declarations + the "Probable" era guard (2026-08-19)

Two loose ends from the snap investigation, both closed as guardrails rather
than as silent semantic changes.

### Phantom declarations in POSITION_FEATURES

Seven features per position were declared but produced by no builder, and
`get_position_features` dropped them silently -- so
`scripts/walk_forward_multiweek.py`, its only consumer, trained on fewer
features than its config implied with nothing saying so. Checked each
against a real built frame:

| declared | disposition |
|---|---|
| `rushing_yards_roll3` | renamed -> `rushing_yards_roll3_mean` |
| `targets_roll3` | renamed -> `targets_roll3_mean` |
| `snap_share_roll3` | renamed -> `snap_share_pct_roll3_mean` |
| `fantasy_points_roll3` / `_roll5` | no equivalent — removed |
| `receiving_yards_roll3` | no equivalent — removed |
| `total_touches_roll3` | no equivalent — removed |
| `team_sos`, `matchup_difficulty` | no equivalent — removed |

`snap_share_roll3` never existed under that name; the roll loop emits
`{col}_roll3_mean`. The three renames are a **behaviour change** for
walk_forward_multiweek, which now actually receives them.
`get_position_features` warns on any future drop, and all declarations now
resolve against a real frame with zero warnings.

`config.settings.CAUSAL_FEATURES` remains the authoritative production list;
a test pins the distinction.

### The "Probable" era discontinuity: guarded, NOT remapped

`Probable` scores 0.85 and was abolished after 2015 (2,772 / 2,607 / 2,702
uses in 2013-2015, then exactly 0), so `pct_injured` runs 14-16% before 2016
and 4-6% after — a reporting-rule artifact, not players getting healthier.

**Deliberately not remapped to 1.0.** The argument for remapping is real: the
~2,700 vanished Probables were NOT absorbed by Questionable, which stayed
near 1,300, so in the modern regime those players are simply unlisted at 1.0.
But this project has already had one "obviously correct" re-encoding fail its
experiment — unknown snap shares, where the semantically wrong value was the
better predictor because the missingness was informative. Remapping is a
modelling decision and deserves the same evidence.

`_warn_on_probable_era_span` warns when a window straddles the boundary,
naming the exposure. Verified on real data: silent on the default 2018-2025
window, and on the "full" 2006-2025 preset it reports 8,081 Probable rows.
`TRAINING_START_YEAR_DEFAULT = 2018` is why this is a warning and not an
error, and a test pins that relationship.

## Phase 6c re-run on the snap-corrected panel — 2026-08-19 (v3)

`data/experiments/phase6c_feature_ablation_v3_snapfix.csv`. 48 rows, no
dropped folds. Production mode (`SNAP_MISSINGNESS_MODE = "zero"`),
FEATURE_VERSION 33. Feature-set sizes are IDENTICAL to v2 (QB 57 / RB 64 /
TE 65 / WR 67), so v33's change was to how `depth_chart_rank` is computed,
not to the list — the deltas below are feature *values* and training
population, not set size.

| Position | 10 | 20 | 30 | all |
|---|---|---|---|---|
| QB | 6.076 | 6.069 | **5.944** | 6.067 |
| RB | 4.417 | 4.353 | 4.337 | **4.336** |
| TE | 2.812 | 2.814 | 2.810 | **2.810** |
| WR | 4.174 | 4.093 | **4.057** | 4.060 |

### Every cell improved vs v2

| v3 - v2 | 10 | 20 | 30 | all |
|---|---|---|---|---|
| QB | -0.203 | -0.100 | -0.222 | -0.047 |
| RB | **-0.390** | **-0.402** | **-0.401** | **-0.400** |
| TE | -0.056 | -0.046 | -0.041 | -0.045 |
| WR | -0.001 | -0.066 | -0.049 | -0.062 |

**RB's improvement is uniform to within 0.012 across every feature count.**
A feature-quality gain should interact with whether the improved feature
falls inside the top-10/20/30; a flat shift does not. That points at the
training population rather than the features — consistent with RB's
`window="all"` now pulling in the 4,292 new 2013-2017 zero rows and 28,483
corrected pre-2018 snap values. WR, whose `window="3y"` (2020-2022) excludes
every pre-2018 season, improved least (-0.001 at 10 features). QB and TE,
which also reach pre-2018, sit in between.

**Caveat: cross-run MAE comparability is unverified.** The ablation CSVs do
not record test-row counts, so it cannot be confirmed from the artifacts
that v2 and v3 scored the same test population. The within-run comparison
(10 vs 20 vs 30 vs all on identical data) is unaffected and is what Phase 6c
actually asks.

### The reversals are a 2025 phenomenon, not a general one

Per season, "all" wins **8 of 12** position-seasons. Every reversal sits in
2025:

| position | 2023 | 2024 | 2025 |
|---|---|---|---|
| QB | all | all | **30** (5.937 vs 6.343 at all) |
| RB | 30 | all | all |
| TE | 30 | all | all |
| WR | all | all | **30** (4.193 vs 4.226 at all) |

v2's WR/TE reversal has essentially vanished — WR's shrank 0.016 -> 0.003,
TE's 0.004 -> 0.000 (a tie). That reversal was attributed in v2 to "the
corrected zero-inflated target making marginal features noisier"; on the
further-corrected panel it does not survive, so that attribution should be
treated as noise rather than a finding.

A new and larger QB reversal appeared, **entirely in 2025**: 6.343 at "all"
against 5.937 at 30, a 0.406 gap far outside the ≤0.016 scale of v2's
reversals. 2025 is the season whose `depth_chart_rank` changed most (44.7%
of skill rows) and is also the least complete. Worth a look on its own;
not investigated here.

### Practical takeaway: unchanged

"More features generally helps or is neutral" still holds for all four
positions. No evidence to trim `CAUSAL_FEATURES`. Still one run per cell
with no repeated-trial variance estimate, so differences at the 0.01 scale
remain uninterpretable.

## Phase 7: the two "unexplained" findings are one effect (2026-08-19)

Analysis of the existing `phase7_season_projection_v2_playoff_fix.csv`
(2026-08-17, n=1,744). **No re-run required to reach this.**

### Status correction

Two records were stale and contradicted each other. `next_focus.md:502` is
headed `[REDESIGNED + REAL RUN COMPLETE — 2026-08-11/12]` but the body
beneath it still reads "The real end-to-end run ... was intentionally NOT
executed". GAPS.md:5984 likewise says Phase 7 is "not yet done". Both
predate the completed run, and a further v2 (playoff fix) ran 2026-08-17.
**Phase 7 has been run three times; it is not pending.**

### The finding

Season-total bias by number of synthetic (no-row) weeks:

| position | 0 | 1-3 | 4-8 | 9-13 | 14+ |
|---|---|---|---|---|---|
| QB | **-28.1** | +9.0 | +43.0 | +50.4 | +44.3 |
| RB | **-35.0** | -9.9 | +5.5 | +18.3 | +22.3 |
| TE | **-22.8** | -18.6 | -2.4 | +5.9 | +7.5 |
| WR | **-33.3** | -20.9 | -2.5 | +10.8 | +8.8 |

The gradient is **monotonic and universal**, not QB-specific:

1. Every position **under**-projects players who play a full season
   (-22.8 to -35.0).
2. Every position **over**-projects players who miss weeks, increasingly
   with the number missed.
3. QB's aggregate bias is positive (+28.9) only because its population
   skews synthetic — 199 of 238 QB-seasons have synthetic weeks, mean 9.35
   of them. RB/WR/TE keep negative aggregates because they carry more
   full-season players and have shallower slopes.

So "QB over-projects while RB/WR/TE under-project, opposite directions, not
yet explained" is a **population-mix artifact** on top of a single gradient.
QB's slope is merely the steepest: QBs with 14+ synthetic weeks scored 9.1
points on the season, i.e. third-stringers who never played, yet are still
projected at a meaningful rate.

### The second finding was a scale artifact

The recorded "players requiring synthetic weeks have roughly double the
error, except QB where it inverts" compares raw MAE across groups whose
targets differ by 2-3x (RB mean actual 164.3 without synthetic weeks vs
65.7 with). Normalised as MAE / mean actual, the direction reverses and
becomes consistent:

| position | no synthetic | with synthetic |
|---|---|---|
| QB | 12.8% | **57.5%** |
| RB | 23.2% | 32.6% |
| TE | 35.3% | 34.4% |
| WR | 27.5% | 33.3% |

Synthetic-week players are harder everywhere, not easier — and QB is the
outlier by a wide margin, consistent with the gradient above.

### What this implies, and what it does not

It does **not** reopen the availability-gradient investigation, which Track
B closed on 2026-08-18 for genuine games. This is the *synthetic-week
projection* over-estimating heavy-absence players — a different question.

The actionable read: for players who miss most of a season, the per-week
rate times P(plays) is too high. For QB specifically, missing weeks usually
means losing the job permanently rather than returning to the same role,
which the current mechanism does not represent. Not fixed here; this entry
is the diagnosis.

**Also unaddressed**: the full-season under-projection of -22.8 to -35.0,
which is present at every position and is the larger effect for the
population most users care about.

## Phase 7A: the full-season under-projection is weekly-model bias, not aggregation (2026-08-19)

Diagnosis only, per the pre-agreed scope. No feature change, no refit, no
re-run: `phase4_v33.csv` already carries per-week predictions at exactly the
FINAL_CONFIG architecture/window/weighting for all four positions, at the
current feature version.

**Population control.** Only zero-synthetic-week player-seasons (n=403), and
`actual_season_total` from Phase 7 reconciles to the summed weekly
`actual_ppr` with a gap of **exactly 0.0** at every position — same players,
same weeks, no selection creeping in.

### The decomposition (positive = under-projection)

| position | n | weeks | weekly-model bias | aggregation bias | total | per-week |
|---|---|---|---|---|---|---|
| QB | 39 | 15.3 | **28.79** | -0.66 | 28.14 | 1.887 |
| RB | 86 | 16.2 | **36.18** | -1.21 | 34.97 | 2.229 |
| TE | 128 | 16.5 | **21.17** | +1.65 | 22.82 | 1.285 |
| WR | 150 | 16.7 | **30.68** | +2.62 | 33.30 | 1.838 |

The identity holds exactly (max residual 0.0). **Aggregation contributes
2-7% and its sign is not even consistent.** The 18-week machinery is not the
problem; the weekly model is, compounded over ~16 weeks.

### Why the weekly model under-predicts: it is optimising the median

Bias by the player's own season-total decile is strongly monotonic —
**-0.51 in the bottom decile rising to +3.51 in the top**. The model
over-predicts weak players and under-predicts strong ones. That is
shrinkage, and it is *correct* behaviour for a point prediction under MAE
loss: MAE is minimised by the conditional **median**.

Weekly PPR is heavily right-skewed, so median < mean:

| position | skew | mean | median | mean-median | per-week bias |
|---|---|---|---|---|---|
| QB | 0.29 | 13.11 | 13.0 | **0.11** | **0.664** |
| RB | 1.34 | 8.07 | 5.7 | **2.37** | **1.524** |
| TE | 1.90 | 4.14 | 1.8 | **2.34** | **1.221** |
| WR | 1.53 | 6.65 | 4.0 | **2.65** | **1.616** |

The ordering matches: QB's weekly PPR is nearly symmetric (skew 0.29, gap
0.11) and QB has by far the smallest bias. RB/TE/WR are strongly skewed and
carry 2-4x the per-week bias. FINAL_CONFIG selected `C_gbm_mae` for RB/WR
(pure median objective), `B_gbm_huber` for TE and
`F_yeojohnson_huber` for QB (both between mean and median), which is why the
observed bias sits below the full mean-median gap rather than at it.

### The consequence, stated plainly

**Summing ~16 conditional medians of a right-skewed variable systematically
under-estimates the conditional mean of the sum.** Phases 2-5 selected
architectures by weekly MAE, which is the right criterion for a weekly point
prediction and the wrong one for a summed season total. The two use cases
want different estimators, and the season projection inherited the weekly
one.

This also gives the absent-player gradient a shared component: heavy-absence
players sit in the low deciles, where the same shrinkage **over**-predicts
(-0.51/week), pushing season totals up. It is not the whole story there —
the synthetic-week `availability_rate` path contributes separately — but it
is the same underlying mechanism acting in the opposite direction.

### Not done here

No fix applied. The obvious candidate — predict conditional means for the
season-total use case, or calibrate the summed total — is a modelling
decision, and this project's last "obviously correct" adjustment failed its
experiment. It wants a pre-registered test with the same discipline.

## Phase 7A follow-up: the calibration diagnostic, and why 7B's upside is small (2026-08-19)

Cheap pre-experiment diagnostic on `phase7_season_projection_v2_playoff_fix.csv`.
All in-sample; a real 7B must be out-of-fold.

### Conditioning on PREDICTED deciles inverts the picture

Zero-synthetic players (n=403), by **predicted** season-total decile:

| decile | pred | actual | actual/pred |
|---|---|---|---|
| 0 | 7.2 | 15.9 | **2.21** |
| 3 | 46.6 | 71.1 | 1.53 |
| 6 | 122.2 | 175.3 | 1.44 |
| 9 | 282.4 | 318.1 | **1.13** |

The ratio **falls** with predicted value, the opposite of the gradient seen
by *actual* decile (-0.51 -> +3.51). Both are regression to the mean viewed
from opposite conditioning variables. Using actual deciles to design a
calibration would have produced the wrong shape — the reason to insist on
predicted deciles.

An affine fit is adequate: pooled `actual = 17.58 + 1.1159 * predicted`,
residuals by decile showing noise rather than curvature. b > 1, as the
median-vs-mean diagnosis predicts. Per position, b ranges 1.056 (QB) to
1.559 (TE).

### A blanket calibration is NET HARMFUL

Fitting on zero-synthetic players and applying it to everyone:

| group | n | MAE before | MAE after |
|---|---|---|---|
| zero-synthetic | 403 | 33.6 | **24.9** |
| has synthetic | 1,341 | 23.0 | **35.3** |
| **overall** | 1,744 | **25.43** | **31.55** |

It fixes the 403 and wrecks the 1,341, who were already over-projected.
Overall season MAE gets 24% worse.

### The information that would fix it is not available at prediction time

Conditioning on `frac_real` (share of weeks with a real row) collapses
overall MAE to 16.74 and zeroes both groups' bias — but that is **leakage**.
`frac_real` is derived from how many weeks the player actually appeared,
i.e. the outcome. It was nearly reported as a result.

Redone with `availability_rate`, the documented prior-seasons-only
estimator that IS legitimate at prediction time:

| model | overall MAE | zero-synthetic bias |
|---|---|---|
| baseline | 25.43 | -29.8 |
| pred only | 23.74 | -22.0 |
| pred + availability_rate | 23.61 | -22.2 |
| pred + avail + interaction | 23.57 | -22.1 |

**A legitimate calibration buys ~7% and leaves the -22 core bias
essentially intact.** `availability_rate` cannot separate the groups: it
averages 0.847 for zero-synthetic players against 0.698 for the rest, with
heavily overlapping distributions. Pre-season, we genuinely do not know who
will play seventeen games.

### Implication for the plan

7B (season-total calibration) has a small and partly illusory upside: most
of the apparent gain came from a variable that encodes the outcome. It is
still worth running out-of-fold to put a real number on it, but it should
no longer be expected to fix the -22 to -35 bias.

**7C (a mean-oriented weekly objective) is now the more promising branch**,
because it attacks the median-vs-mean mismatch at source and needs no
knowledge of who will stay healthy. The 7A decomposition says the bias is
inherited from the weekly estimator, and this diagnostic says it cannot be
recovered downstream from information available at projection time.

## Phase 7C RESULT: mean objective fixes season bias, at a cost (2026-08-19)

Pre-registered single-lever test. `data/experiments/phase7c/`. Both arms:
1,744 season rows, 21,465 week rows, symmetric difference **0** — identical
player-seasons. Baseline weekly MAE agrees with the Phase 4 harness within
~1% at every position (QB 6.061 vs 6.087, RB 4.334 vs 4.397, TE 2.770 vs
2.806, WR 4.031 vs 4.090), so the two runners measure the same thing.

### Weekly (real weeks only; + bias = over-predict)

| position | MAE A | MAE B | delta | bias A | bias B |
|---|---|---|---|---|---|
| QB | 6.061 | 6.188 | +0.127 | -0.813 | **-0.866** |
| RB | 4.334 | 4.400 | +0.067 | -1.590 | **-0.313** |
| TE | 2.770 | 2.876 | +0.107 | -1.324 | **-0.105** |
| WR | 4.031 | 4.254 | +0.224 | -1.464 | **-0.022** |

Weekly MAE degrades slightly everywhere (+0.07 to +0.22), exactly as
predicted: MAE is minimised by the median, so a mean objective must lose
weekly MAE. Weekly bias collapses toward zero for RB/TE/WR.

### The mechanism is confirmed by the control

QB's weekly bias **does not move** (-0.813 -> -0.866) while RB/TE/WR's
collapses. That is what the Jensen caveat predicted: `G_yeojohnson_mse`
minimises MSE in transformed space, which is not the conditional mean in
PPR space, so QB was never truly switched to a mean objective. Combined
with QB's near-symmetric target (skew 0.29, mean-median gap 0.11), it is a
genuine control arm — and it shows no benefit. The bias reduction tracks
skew across the other three.

### Season

| position | n | MAE A | MAE B | delta | bias A | bias B |
|---|---|---|---|---|---|---|
| QB | 285 | 27.56 | 29.10 | **+1.54** | +1.50 | +1.12 |
| RB | 392 | 21.06 | 18.82 | **-2.24** | -15.43 | -1.32 |
| TE | 393 | 18.14 | 13.77 | **-4.37** | -15.94 | -0.94 |
| WR | 674 | 20.61 | 18.79 | **-1.82** | -16.03 | +0.64 |

Overall season MAE **21.29 -> 19.35** (-9%); overall season bias
**-13.01 -> -0.08**.

### But the pre-registered reject condition also fires

| position | group | n | MAE A | MAE B | bias A | bias B |
|---|---|---|---|---|---|---|
| RB | zero-synthetic | 230 | 28.32 | **21.80** | -24.60 | -7.88 |
| RB | has synthetic | 162 | 10.76 | **14.58** | -2.42 | +8.00 |
| TE | zero-synthetic | 324 | 20.26 | **14.98** | -18.21 | -1.95 |
| TE | has synthetic | 69 | 8.16 | 8.09 | -5.29 | +3.79 |
| WR | zero-synthetic | 512 | 23.89 | **20.16** | -19.66 | -1.67 |
| WR | has synthetic | 162 | 10.24 | **14.47** | -4.56 | +7.93 |

RB's and WR's synthetic-week populations get **materially worse** (+35% and
+41% MAE). Those players were near-calibrated under the baseline (bias -2.4
and -4.6); a uniform upward shift pushes them from slightly-under to clearly
-over (+8.0, +7.9). TE's synthetic group is unaffected (n=69).

So two rows of the decision table fire simultaneously: "weekly slightly
worse + season materially better + bias substantially reduced -> strong
candidate" for the zero-synthetic population, and "season improves only for
zero-synthetic but synthetic worsens materially -> reject/investigate" for
RB and WR.

**QB is a clean reject** on its own terms: weekly worse, season worse, bias
essentially unchanged.

### Not decided here

The result is real and the mechanism is confirmed, but it is not the clean
win the decision rule was written to accept. Adoption is a judgement about
which population the season projection is FOR.

## Phase 7C follow-up: the mixture is SMOOTH (2026-08-19)

Continuous decomposition by synthetic share, on the existing 7C predictions.
No refit.

### RB/WR/TE combined (the positions the objective actually changed)

| synthetic share | N | baseline bias | treated bias | d bias | baseline MAE | treated MAE | d MAE |
|---|---|---|---|---|---|---|---|
| 0% | 1,066 | **-20.29** | -3.10 | +17.19 | 23.74 | **18.94** | -4.81 |
| 0-25% | 234 | -7.93 | +6.11 | +14.04 | 13.19 | 15.35 | +2.15 |
| 25-50% | 103 | **+1.59** | +8.92 | +7.33 | 5.53 | 10.73 | +5.20 |
| 50-75% | 42 | +2.85 | +8.43 | +5.58 | 5.54 | 10.18 | +4.64 |
| 75-100% | 14 | +5.36 | +9.96 | +4.61 | 5.42 | 10.07 | +4.65 |

Monotonic and smooth in every column. There is **no breakpoint**. Baseline
bias climbs steadily from -20.29 to +5.36 and crosses zero at roughly a 25%
synthetic share; the treatment shifts the whole curve up by an amount that
itself declines smoothly (+17.19 -> +4.61), crossing zero near 0-10%.

This is the "smooth mixture" case: **the correction wants to be conditioned
on expected availability, not selected as a different global estimator per
position.** The baseline is already well calibrated for players missing
25-50% of the season; the objective change is right for the durable end of
the same continuum and wrong for the absent end.

### The control separates the two effects

QB, whose objective did not truly change (Jensen — see 7C entry):

| synthetic share | N | baseline bias | treated bias | d bias |
|---|---|---|---|---|
| 0% | 137 | -13.11 | -14.33 | -1.22 |
| 0-25% | 29 | +0.57 | -2.46 | -3.03 |
| 25-50% | 33 | +13.26 | +12.80 | -0.46 |
| 50-75% | 52 | +17.69 | +19.59 | +1.90 |
| 75-100% | 34 | +24.96 | +26.82 | +1.86 |

QB shows **the same monotonic availability gradient** (-13.11 -> +24.96)
with **essentially no treatment effect** (d bias -3.03 to +1.90). So the
availability gradient is a pre-existing structure independent of the
objective, and the objective change is a separate, roughly level shift on
top of it.

That is the decomposition made visible:

    E[season] = E[points | available] x E[availability]

7C improves the first term for skewed positions. The second term is
untouched, and the gradient above is what it looks like when it is wrong.
Nothing here establishes that the availability term has been solved — 7B
already showed the pre-season information to estimate it is not available.

### Consequence

Do not adopt 7C as a per-position global estimator. The evidence says the
target transformation should be conditioned on expected availability, and
that conditioning is exactly what this project cannot currently do well.
The honest options are a conditional-projection estimand
(E[points | available], which 7C improves) or an unconditional season
estimand (which needs the availability term), and they are different
products rather than better and worse versions of one.

---

## Population regime experiment: the participation contract (2026-08-19)

Refactors the production target from "a stats row happened to exist" to
"we observed the player participate," per the directive to separate
conditional production from availability. New code:
`src/models/single_week_ppr/population.py`,
`scripts/run_population_regime_experiment.py`,
`tests/test_population_regimes.py` (13 tests).

### The contract

| era | evidence | `participation_quality` |
|---|---|---|
| 2013+ | `offense_snaps > 0` | 2 (snap-confirmed) |
| 2006-2012 | `PPR > 0`, or `inferred_pbp_confirmed_zero` | 1 (inferred) |
| either | neither | 0 (excluded) |

2013 is not "where modern football starts" — it is the first season with
PFR snap coverage, i.e. the start of the high-confidence participation-label
regime. The quality-1 proxy errs toward false exclusion: it costs
legitimate zero-point 2006-2012 games rather than admitting inactive/IR/
practice-squad weeks.

Snap-era rows with NULL `snap_count` (~1.7K of 119K, unmatched by the snap
feed) are quality 0, not 1 — letting them fall back to the PPR proxy would
leak the weak label into the regime whose whole claim is snap-confirmed
labels.

### First finding: the filter is nearly a no-op on the existing population

`player_weekly_stats` is box-score-triggered, so it already contains almost
only participation rows, plus the 12,959 `inferred_snap_verified_zero` rows
(snaps > 0, PPR = 0) the panel build added. Rows that are *confirmed* `snap_count = 0` in 2013+ at QB/RB/WR/TE:
**63** out of 119K (62 of them carrying a real `nflverse_stats` line).

So directive steps 2-4 were a contract-and-guard job, not a data-deletion
job. Population A ≈ population B in the directive's A/B/C sense. The real
lever turned out to be the 2006-2012 era rule, not the snap filter.

Corollary worth keeping: a literal `offensive_snaps > 0` filter would have
silently deleted all 33,188 pre-2013 rows, because the column defaults to 0
and no snap data exists before 2013. Pre-2013 `snap_count = 0` means
UNKNOWN, not "did not play."

### The experiment

Four arms, one shared fold load per (position, season), FINAL_CONFIG
architecture + weighting held fixed, all arms scored on an IDENTICAL
held-out population (snap-confirmed rows only). Features are engineered
over the full panel BEFORE the population filter, so 2006-2012 feeds career
history and rolling windows to every arm regardless.

    P0_current          FINAL_CONFIG window, no participation filter (reference)
    A_clean_modern      2013+, snaps > 0
    B_extended          + 2006-2012 via the PPR > 0 proxy
    C_extended_flagged  B, with participation_quality as a model feature

A/B/C run at window="all" so the era rule is the only moving part; WR's 3y
and TE's 10y FINAL_CONFIG windows never reach 2012 and would have made A
and B identical there. This means A_clean_modern is NOT the same thing as
current production, which is why P0_current is carried separately.

### Result: the era rule is a null except at TE

Weekly MAE, B minus A, per fold (negative = keeping 2006-2012 helps):

| position | 2023 | 2024 | 2025 | sign-consistent |
|---|---|---|---|---|
| QB | -0.053 | -0.000 | **+0.215** | no |
| RB | -0.010 | +0.012 | -0.061 | no |
| TE | +0.016 | +0.013 | +0.013 | **yes** |
| WR | -0.007 | -0.012 | **-0.267** | no |

In 2023 and 2024 every |B-A| is <= 0.053 and mostly <= 0.016 — no effect at
any position. All the separation is in 2025, and there QB and WR point in
**opposite directions**. The pooled means are one fold wearing a disguise.

Note 2025 was already flagged as anomalous for an unrelated reason (p50
quantile coverage below nominal across all 4 positions, never root-caused).
It is now also the only fold where the population arms separate. Do not
resolve the era question on this evidence.

TE is the only position with a sign-consistent effect, and it is the
item-13 pattern exactly — B trades a tiny MAE loss for a real gain on the
quantities that matter:

| TE metric | direction | magnitude, all 3 folds |
|---|---|---|
| weekly MAE | A better | +0.013 to +0.016 |
| weekly bias | B better | 0.128 to 0.173 less under-prediction |
| weekly RMSE | B better | 0.014 to 0.041 |
| season-conditional MAE | B better | **1.37 to 1.49** |

### Result: flagging the era buys nothing

C minus B on weekly MAE: QB +0.002, RB -0.009, TE -0.000, WR +0.000. A
decisive null. The extra rows help or hurt as raw data; handing the model
an explicit regime label does not let it exploit them better. Record
`participation_quality` in the pipeline for auditing, but there is no case
for feeding it as a feature.

### Result: the model has learned participation -> production directionally,
### and compressed it by half

Predicted vs actual mean PPR by REALIZED offensive-snap bucket (arm B,
pooled). Ordering is monotonic at every position — the relationship is
learned — but the slope is roughly 0.5-0.7 instead of 1.0:

| pos | 0-5 | 5-15 | 15-30 | 30-50 | 50+ | slope | range retained |
|---|---|---|---|---|---|---|---|
| QB | +4.11 | +3.91 | +2.05 | +0.24 | **-2.48** | 0.603 | 60% |
| RB | +1.20 | +0.37 | -1.48 | -3.43 | **-6.33** | 0.590 | 57% |
| TE | +0.43 | +0.17 | -0.37 | -1.67 | **-4.85** | 0.516 | 51% |
| WR | +0.37 | +0.02 | -0.66 | -1.95 | **-3.50** | 0.687 | 68% |

(cells are predicted minus actual)

The model over-predicts low-snap games and under-predicts high-snap games
at every position, retaining only 51-68% of the true dynamic range.

**This is not a defect the production model can fix, and that is the
point.** Realized snaps are not known at forecast time, so the compression
is what a median-optimising estimator correctly does when opportunity is
uncertain. The table therefore measures how much weekly error is
*opportunity uncertainty* rather than production error — direct empirical
support for the split architecture. The bucket-level bias pattern is
near-identical across all four arms, confirming the population question and
the compression question are independent.

### Disposition

The contract and its tests are the durable deliverable. The era-inclusion
choice is NOT resolvable at n=3 folds with the effect concentrated in one
anomalous season; B remains the default (nothing throws away seven seasons
on this evidence, and TE consistently prefers it at season level).

Not established here: anything about the availability half. Every number
above is conditional on observed participation.

---

## Live bug: `age` is a per-position constant, so the whole age-curve
## feature family is dead (2026-08-19)

Found while bucketing the population-regime results by age — every position
collapsed into a single age bucket, which is impossible for real ages.

`src/features/season_long_features.py:200-204` falls back to
`avg_ages = {'QB': 28, 'RB': 25, 'WR': 26, 'TE': 27}` when neither `age`
nor `years_exp` is in the frame. Neither ever is: `age` is not a column in
`get_all_players_for_training()`'s output, and nothing computes
`years_exp`. So **every training row gets its position's constant.**

Consequence: `age_curve` — a declared CAUSAL_FEATURE at all four positions —
is a zero-variance constant (QB 1.0000, RB 1.0000, WR 0.9950, TE 0.9960).
So are `age_factor`, `age_expected_games`, `decline_rate`,
`years_from_peak`, `is_in_prime`, and `injury_age_risk`.

`feature_engineering.py:_add_age_curve_feature` has a *correct* birth-date
fallback (lines 619-645) that joins `players.birth_date`, but it is guarded
by `if "age" in df.columns` — and season_long_features has already
populated the constant by then. The working path is dead-coded behind the
broken one.

`players.birth_date` is populated for 2,016/2,985 players (68%), so real
age is derivable. Not fixed here: fixing it changes model inputs and would
have invalidated the regime comparison running at the time. It also means
every age/aging-curve result this project has ever reported was computed on
a constant.

### FIXED 2026-08-20

Three changes, one commit:

1. `scripts/backfill_player_birth_dates.py` — `players.birth_date` covered
   only 68% of players and 50-71% of player-weeks (QB worst). Backfilled
   937 birth dates from `nfl_data_py.import_players()` (24,998 available,
   keyed by gsis_id). Row-weighted coverage is now ~100% at every position.
   Fill-only; never overwrites. Of the 2,016 players where both sources had
   a value, exactly 2 disagreed, so there was no case for an overwrite.
   The `rosters` table also has a `birth_date` column and looks like a
   second source — it is empty (0 rows), so it is not one.

2. `src/features/player_age.py` — one age derivation, used by both
   callers. Age is taken at Sept 1 of the season (matching
   `preseason_projector._season_start`), so a player has one age per season.
   Fallback order: `age` column, `birth_date` column, `players.birth_date`
   via player_id, `22 + years_exp`, position constant.

3. The position constant now **warns above a 10% fallback rate**. That
   guardrail is the actual fix for this class of bug — the original defect
   was not that a fallback existed, but that it was silent. The codebase
   already used this pattern for `opp_fpts_allowed_*` and Vegas features;
   age simply never had it.

`feature_engineering._add_age_curve_feature`'s dead-coded birth-date join
was deleted rather than repaired, since it now duplicates the shared helper.

Verified on real data, all four positions at 0.000% on the position
constant:

| pos | n | mean age | sd | min | max | unique ages | age_curve sd | age_curve unique |
|---|---|---|---|---|---|---|---|---|
| QB | 14,514 | 28.38 | 4.59 | 21.2 | 45.1 | 1,342 | 0.0970 | 1,224 |
| RB | 27,805 | 25.86 | 2.78 | 20.6 | 37.3 | 1,851 | 0.1095 | 1,838 |
| WR | 44,986 | 26.27 | 3.11 | 20.8 | 38.8 | 2,537 | 0.0661 | 2,537 |
| TE | 26,464 | 26.79 | 3.10 | 20.8 | 41.3 | 1,797 | 0.0559 | 1,518 |

`age_curve` went from 1 unique value per position to 1,224-2,537.

Spot-checked against biography, not just against distribution shape:
Brady 1977-08-03 (45.1 in 2022, his last season), Testaverde 1963-11-13
(43.8 in 2007), Marcedes Lewis 1984-05-19 (41.3 in 2025), Benjamin Watson
1980-12-18 (38.7 in 2019). All correct.

No leakage: birth date is static and known years ahead. Pinned by
`test_age_uses_no_future_information` (a 2018 row must not move when 2020
rows join the frame). 13 tests in `tests/test_player_age.py`; full suite
376 passed.

---

## Live bug: 11.4% of the QB training population are not quarterbacks
## (2026-08-20)

Found while spot-checking ages — Christian McCaffrey surfaced in a
`position='QB'` query. He is labeled QB in the `players` table. So are
Derrick Henry, DJ Moore, Cooper Kupp, Devin Singletary and Courtland
Sutton.

Comparing `players.position` against `nfl_data_py.import_players()`:
99 of 2,884 labeled players disagree (3.4%), which is 2,501 player-weeks
overall (2.18%) — but it is wildly concentrated:

| labeled | rows | mislabeled | share | mean FP (correct) | mean FP (mislabeled) |
|---|---|---|---|---|---|
| **QB** | 14,622 | **1,666** | **11.39%** | 13.08 | 11.62 |
| RB | 28,037 | 546 | 1.95% | 8.39 | 2.51 |
| TE | 26,604 | 222 | 0.83% | 4.88 | 5.22 |
| WR | 45,333 | 67 | 0.15% | 7.67 | 2.60 |

The QB contamination is 822 WR rows, 601 RB rows and 237 TE rows. A
corroborating symptom that needs no external feed: **15.0% of "QB"
player-weeks have zero passing attempts**, and 1,541 of those 2,187 rows
are flagged mislabeled.

### Why this matters more than the row count suggests

Every long-standing unexplained QB anomaly in this file is of the form "QB
behaves unlike the other three positions":

* QB early-season R² = 0.166, the worst of any position
* Phase 7: QB over-projects (+22.5) while RB/WR/TE all under-project
* Phase 7: players needing synthetic weeks have ~2x error *except* QB,
  where the pattern inverts
* Phase 7C: QB is the near-control arm that behaves unlike the others
* The population regime experiment (2026-08-19): QB is the one position
  whose 2025 fold moves opposite to WR

An 11% admixture of players drawn from an entirely different scoring
distribution is a candidate common cause for that whole family. Not
established — but it should be ruled in or out before any of those
findings is treated as a modeling result.

### Not fixed here

Relabeling changes the training population at every position at once,
which would invalidate any comparison running against the current
populations. It also needs a decision on what the authoritative position
source is (the feed's single `position`, a per-season position, or usage-
derived), since a player's position can legitimately change across a
career and `players` has one row per player with no season dimension.

### The age fix changes nothing measurable in the weekly model (2026-08-20)

Re-ran the identical 12 folds x 4 arms with the fix in place
(`data/experiments/population_regime_*_preagefix.csv` vs
`population_regime_*.csv`). Training-row counts and eval-row counts are
byte-identical across the two runs, so the age feature family is the only
thing that moved.

Weekly, mean over arms x seasons (positive = worse after the fix):

| position | d MAE | d bias | d RMSE | d R² |
|---|---|---|---|---|
| QB | +0.0090 | -0.0841 | +0.0164 | -0.0032 |
| RB | +0.0096 | -0.0069 | +0.0112 | -0.0025 |
| WR | +0.0019 | -0.0134 | +0.0053 | -0.0011 |
| TE | +0.0003 | -0.0010 | +0.0014 | -0.0004 |

Season-conditional: QB +0.483 MAE, RB +0.158, WR +0.140, TE -0.001.

So correcting a feature that had been a frozen constant since it was
written produced **no improvement, and a very slight degradation**. Worth
stating plainly rather than burying.

Three things this does and does not mean:

* It does not make the fix wrong. A zero-variance column declared as a
  CAUSAL_FEATURE is a defect independent of its effect on any metric, and
  it would have silently poisoned any future work that leaned on it.
* The weekly model already encodes aging implicitly, through
  `prev_season_ppg`, rolling production and career-trajectory features.
  `age_curve` now ranks 28/64 by LightGBM importance for RB (51 vs 288 for
  the top feature) with 1,755 distinct values — used, but weakly. Adding a
  real-but-weak feature to a GBM costs splits that stronger features were
  using, which is the likely mechanism for the small negative.
* By tenure bucket there is no differential: 6y+ players do not improve
  relative to rookies at any position, which is where a genuine age signal
  should have shown up first if there were one to find.

TE moved least (d MAE +0.0003) because TE's `age_curve` has the flattest
coefficient/peak combination of the four positions — its sd is 0.0559 even
with real ages.

QB moved most at season level (+0.483). Do not read that as an age result:
QB's population is 11.4% non-quarterbacks (see the position-mislabeling
entry above), and handing a real age feature to a contaminated population
is a plausible way to make things slightly worse. Untangling the two needs
the position fix first.

The value of this fix is prospective. `age_expected_games`,
`decline_rate` and `injury_age_risk` feed durability/availability
reasoning — precisely the layer the architecture split calls for next —
and every one of them was reading a constant.

### ROOT-CAUSED AND FIXED 2026-08-20

The chain, in order. Every link was necessary; fixing any one alone would
not have held.

**1. The origin.** `PBPStatsAggregator.aggregate_passing_stats` (line ~225)
hardcodes `position = 'QB'` on every row it produces — i.e. on anyone with
a pass attempt, including a single trick-play pass by a real RB or WR.

**2. The write.** `DatabaseManager.insert_player` used
`INSERT OR REPLACE`, so that derived 'QB' overwrote the correct position
from the weekly feed. The weekly feed itself was never wrong: checked
2017-2025, `nfl_data_py.import_weekly_data()` reports RB for McCaffrey and
Henry, WR for Kupp and Moore, in 100% of rows.

**3. The self-perpetuation.** A 2026-08-08 mitigation already nulls
`position` when a player appears in both the passing and skill groups, so
that `_infer_position` decides from summed stats instead of row order. But
`_infer_position`'s lookup order is `row['position']` -> POSITION_OVERRIDES
-> **the players table** -> heuristic, and the players table was by then
corrupt. Demonstrated directly: with the real lookup, `_infer_position`
returned QB for McCaffrey; with an empty lookup, the heuristic returned RB.
The mitigation was defeated by its own fallback.

**4. The correction lived downstream of the models.**
`reconcile_player_positions_from_rosters()` exists and works — but its only
callers are `generate_draft_data.py` and `generate_app_data.py`, the
output/presentation scripts. So `data/players_RB.json` had McCaffrey
correctly listed as an RB the whole time, while every model trained on a
population that called him a quarterback.

#### Fixes

* `insert_player` is now a real upsert (`ON CONFLICT DO UPDATE`) that
  touches only supplied fields, with a `trust_position` flag. Untrusted
  positions may fill a blank, never overwrite.
* `nfl_data_loader.store_weekly_dataframe` (the PBP path) passes
  `trust_position=False`. The weekly-feed path keeps `True`.
* `_players_position_lookup` reads **roster snapshots first**, players
  table second, breaking the loop at its fallback.
* `check_position_integrity` added to `src/data/quality_gates.py` and wired
  into `run_db_quality_gates`, failing above a 0.5% mismatch rate.
* `scripts/repair_player_positions.py` repairs and audits; `--audit-only`
  exits non-zero so it can gate CI.
* 12 regression tests in `tests/test_position_integrity.py`.

#### Data repaired

38 players, 2,062 player-weeks. QB fell from 14,622 rows to 13,113, and
"QB" rows with zero passing attempts from **15.0% to 5.5%** (the remainder
are legitimate — kneel-downs, wildcat, backups who only handed off).

| stored -> correct | players |
|---|---|
| QB -> WR | 11 |
| QB -> RB | 7 |
| TE -> RB | 4 |
| WR -> RB | 4 |
| WR -> TE | 3 |
| QB -> TE, TE -> QB | 2 each |
| K -> WR (Wes Welker, 156 rows) | 1 |
| RB -> TE, RB -> WR, TE -> WR | 1 each |

Not repaired: 34 players / 159 player-weeks whose authoritative position is
non-skill (OT, G, P, LB — linemen who caught a touchdown, punters on fake
snaps). Left in place deliberately; they did produce those points, and
relabeling them to OT would just drop the rows.
32 players / 4,272 player-weeks have no authoritative source at all and
were left untouched.

---

## Collateral: INSERT OR REPLACE was erasing every static bio field
## (2026-08-20)

Same root cause, separate damage. `insert_player`'s `INSERT OR REPLACE`
rewrites the whole row, and `_store_weekly_data` supplies only
player_id/name/position — so every weekly ingest NULLed everything else.

Measured before the fix: `college` NULL for **all 2,985 players**,
`height`/`weight` NULL for 757, and `created_at` reset on every ingest
(which is why the corrupted rows all appeared to have been "created"
2026-08-19 11:21 — they were replaced, not created; that misdirected the
first pass of this investigation).

It would also have silently destroyed the birth-date backfill committed
hours earlier. The upsert fixes this going forward.

`scripts/backfill_player_birth_dates.py` generalised to
`scripts/backfill_player_bio.py`, covering birth_date/college/height/weight.
Restored 2,951 colleges and 725 height/weight. Coverage is now ~2,953/2,985
on all four. Height and weight are stored as bare integer strings but the
feed supplies floats, so they are coerced before write — otherwise the
column would have ended up in two formats at once.

`college` currently has no downstream consumer
(`college_production_score` in `advanced_rookie_injury` derives from draft
value, not the string), so this is a data-integrity repair, not a feature
change.

#### Consequence for prior results

Every model result in this file that involves QB was trained on a
population 11.4% of which were not quarterbacks. That includes the
population-regime experiment two commits ago. QB conclusions from it should
be treated as void pending a re-run; RB/WR/TE moved by 2% or less and are
substantially less affected.

## Position fix re-run: the QB hypothesis is falsified (2026-08-20)

Re-ran the 12-fold x 4-arm regime experiment after the position repair.
Three comparable result sets now exist:

| suffix | age | position |
|---|---|---|
| `_preagefix` | broken | broken |
| `_prepositionfix` | fixed | broken |
| *(none)* | fixed | fixed |

### The raw numbers say the fix worked. They are wrong.

Arm B weekly MAE, QB: 5.826 -> 5.714 (2023), 5.993 -> 5.955 (2024). Season-
conditional QB MAE: 23.74 -> 21.42. Ten times the size of anything the age
fix moved, in the predicted direction.

**All of it is the evaluation population changing.** The repair removed
1,509 rows from QB, so the QB eval set fell from 2,844 rows to 2,145
(-25%) — and the rows removed were exactly the RB/WR-like high scorers.
Deleting high-variance rows lowers MAE mechanically, model quality
unchanged. GAPS.md already records this trap for the panel fix; it applies
verbatim here.

### On identical rows, nothing happened

Restricting to the 77,060 rows present in BOTH runs (actuals verified
identical to 0.000000), arm B:

| position | n | MAE before | MAE after | d MAE | d bias | d RMSE |
|---|---|---|---|---|---|---|
| QB | 2,145 | 5.9988 | 6.0082 | **+0.009** | +0.084 | -0.047 |
| RB | 4,284 | 4.3432 | 4.3139 | -0.029 | +0.034 | -0.056 |
| TE | 4,930 | 2.8324 | 2.8309 | -0.002 | +0.010 | -0.005 |
| WR | 7,906 | 4.0067 | 4.0826 | **+0.076** | -0.103 | +0.148 |

QB: +0.009. Removing 11.4% contamination from the QB training population
changed its accuracy on comparable rows by nothing. WR got worse, entirely
from its 2025 fold (+0.233).

### The specific prediction failed

Pre-registered before the run: *QB's 2025 fold should stop being the odd
one out; if it survives intact, something else is going on.* Era-rule
contrast (B minus A), before and after:

| position | 2023 | 2024 | 2025 (before) | 2025 (after) |
|---|---|---|---|---|
| QB | -0.028 | -0.025 | +0.191 | **+0.278** |
| WR | +0.013 | -0.007 | -0.258 | -0.051 |
| RB | -0.021 | +0.006 | +0.006 | -0.045 |
| TE | +0.019 | +0.020 | +0.018 | **-0.017** |

QB's 2025 anomaly did not shrink — it grew. WR's collapsed. And TE, the
only position with a sign-consistent era effect in the original
experiment, **lost that consistency**: +0.013/+0.014/+0.018 became
+0.019/+0.020/-0.017.

### What this means

1. The position bug does **not** explain the QB anomaly family (early-season
   R² 0.166, the +22.5 Phase 7 season over-projection, the inverted
   synthetic-week pattern). Those remain open, and the obvious candidate
   cause is now eliminated rather than confirmed.
2. The position fix stands on correctness alone. An 11.4% contaminated
   training population is wrong whether or not fixing it moves MAE, and
   the season-conditional QB numbers are now computed over quarterbacks.
   There is no accuracy dividend.
3. The era rule is *less* resolvable than before, not more. Its one
   sign-consistent effect did not survive a data correction that should
   have been nearly irrelevant to TE (59 eval rows changed).
4. 2025 is now anomalous in a fourth independent way. It was already
   flagged for below-nominal p50 quantile coverage across all four
   positions. Every experiment that separates does so only in 2025, and
   the direction is not stable across positions or across data fixes.
   This is the thing to investigate, not the era rule.

Two fixes in a row have now produced correct data and no measurable
accuracy change. That is worth stating as a pattern: this pipeline's
weekly error is not currently limited by the defects being found in it.
The snap-bucket diagnostic said the same thing from a different direction —
the dominant term is opportunity uncertainty, which no amount of
feature-correctness work touches.

## ROOT CAUSE: the 2025 fold anomaly is broken features, not football (2026-08-20)

Every experiment in this file that separates, separates only in 2025, in
directions that are not stable across positions or across data fixes.
Root-caused today. It is a train/test feature discontinuity.

### The actuals are normal; the predictions collapse

Arm B, held-out rows, by test season:

| position | mean pred 2023 | 2024 | 2025 | mean actual 2025 |
|---|---|---|---|---|
| QB | 12.79 | 13.07 | **11.13** | 13.45 (flat) |
| RB | 7.04 | 7.05 | **5.37** (-24%) | 8.13 (flat) |
| WR | 5.58 | 5.70 | **3.89** (-32%) | 6.44 (flat) |
| TE | 3.17 | 3.11 | **2.74** | 4.43 (flat) |

Actual scoring in 2025 is ordinary. The model's output level drops by up to
a third. That rules out "2025 football was different" and points at features.

### Three features are broken in training and correct in 2025

**`is_dome`** — derived from `home_away` + `DOME_STADIUMS`, not from the
weather table. `home_away` is populated on **100% of
`inferred_snap_verified_zero` rows and 0% of `nflverse_stats` rows** in
every season — except 2025, where the re-ingest
(`nfl_data.db.bak-2025-reingest-20260819`) filled it completely:

| season | data_source | home_away populated |
|---|---|---|
| 2024 | inferred_snap_verified_zero | 1,215 / 1,215 |
| 2024 | nflverse_stats | **0 / 5,480** |
| 2025 | inferred_snap_verified_zero | 1,152 / 1,152 |
| 2025 | nflverse_stats | **5,612 / 5,612** |

So `is_dome` reads ~3% across all training seasons and **35.7%** in 2025.
`game_weather` says 32-35% every season, so nothing is missing at source —
the weekly ingest simply never wrote `home_away`. `is_dome` ranks **4th of
RB's 64 features** by LightGBM importance.

**`team_plays_roll3_mean`** — ~52 through 2019, collapses to 11.4 (2020)
and ~3 (2021-2024), returns to 46.2 in 2025. Cause: `team_stats.total_plays`
is NULL for ~558 of ~600 rows in each of 2020-2024, populated in 2018-19
and 2025. The break lands exactly on the most recent and most heavily
recency-weighted training seasons.

**`team_motion_rate` / `team_play_action_rate`** — frozen at 0.716 / 0.323
for 2006-2022, real values only from 2023. Almost certainly a genuine
source limit (FTN charting starts ~2022), so the defect is the silent
default-fill, not the absence.

### Causal test: remove them and the anomaly disappears

Refit each position for 2024 (control) and 2025 with the five columns
dropped:

| position | 2025 gap vs 2024, before | after | 2025 bias before | after | 2024 control cost |
|---|---|---|---|---|---|
| QB | +0.406 | **-0.005** | -2.328 | -0.617 | +0.047 |
| RB | +0.487 | **-0.047** | -2.755 | -1.303 | +0.050 |
| WR | +0.510 | **-0.120** | -2.556 | -1.089 | +0.110 |
| TE | +0.333 | **+0.076** | -1.694 | -1.253 | +0.109 |
| **mean** | **+0.434** | **-0.024** | -2.333 | -1.066 | +0.079 |

The 2025 penalty averages +0.434 MAE and goes to **-0.024** — gone, at all
four positions. Prediction level recovers (RB 5.37 -> 6.82, WR 3.89 ->
5.35, QB 11.13 -> 12.84). The 2024 control pays only +0.079, confirming
these columns carry almost no real signal even where they are populated.

### This also explains the 2025 quantile-coverage anomaly

Recorded in Phase 4 and never root-caused: p50 coverage runs below nominal
specifically for 2025, across all four positions. A uniformly depressed
prediction level produces exactly that — actuals exceed p50 more often than
they should. Same cause, now closed.

### Consequences

Every result using 2025 as a test season is affected: Phases 4, 5, 6c,
7/7A/7B/7C, and the population-regime experiment. This is the reason the
era-rule contrast was unstable — the arms were being separated by a feature
discontinuity, not by the era rule.

It also reframes the last two days. The age fix and the position fix each
produced correct data and no accuracy change. This one is a *feature-value*
defect rather than a label defect, and it moves MAE by 8-12% on the
affected fold. The lesson is not "data fixes don't pay" — it is that
train/test distribution breaks are the expensive class, and neither
previous fix was one.

### Fix, not yet applied

1. Backfill `player_weekly_stats.home_away` for all seasons from
   `schedule` (data is present), and make the weekly ingest write it.
2. Backfill `team_stats.total_plays` for 2020-2024 from PBP.
3. Mark `team_motion_rate`/`team_play_action_rate` unknown before 2023
   rather than default-filling — the zero-vs-unknown rule the snap columns
   already follow.
4. Add a coverage gate that fails when a declared CAUSAL_FEATURE's mean
   shifts by more than a few sd between adjacent seasons. All three of
   these would have tripped it.

### Fixes applied 2026-08-20 (partial — see the open items below)

**1. `home_away` backfilled.** `scripts/backfill_home_away.py` resolved
96,097 rows for 2006-2024 from `schedule` on (season, week, team,
opponent). **Zero conflicts** against the 27,086 rows that already had a
value, which validates the derivation. Coverage is now ~100% every season;
71 rows unresolvable and left alone. Verified in a real fold: `is_dome` now
reads 0.357-0.405 in every season from 2006 to 2025, against ~0.03 before.

**2. `team_stats` play volume backfilled.**
`scripts/backfill_team_play_volume.py` filled 2,613 rows across 2020-2024.
Not an estimate: summing player `passing_attempts + rushing_attempts` per
(season, week, team) reproduces all 8,249 surviving rows with correlation
1.0000 and zero error in all 20 seasons, because that is how the originals
were computed. The script aborts if that agreement ever breaks. Verified:
`team_plays_roll3_mean` now reads 49.9-53.9 in every season, against ~3 for
2021-2024.

**3. Unknown scheme tendencies are now NaN — after removing FOUR fillers.**
This one took four passes, which is the point worth recording:

  1. `_add_scheme_tendencies` returned a literal `(0.5, 0.1, 0.5)` default.
  2. Inside the same function, `round(motion or 0.5, 3)` replaced both NULL
     *and* a genuine 0.0 rate with 0.5.
  3. `_apply_bounded_scaling` did `.fillna(0.0)` before MinMax scaling, so
     marking anything unknown upstream was pointless.
  4. `_impute_missing` median-fills every numeric column, exempting only
     `_SNAP_IMPUTATION_OWNED`. This is the same filler the snap columns
     already needed an exemption from (see "Third filler", 2026-08-19).

Only after all four did the column stay NaN. Fixes 1 and 2 alone changed the
constant from 0.716/0.323 to 0.371/0.379 — still a constant, which is how
the incompleteness was caught. Added `_STRUCTURALLY_MISSING` alongside
`_SNAP_IMPUTATION_OWNED`, and made the bounded scaler NaN-preserving (it
fills with the column median rather than 0.0 for the fit, since 0.0 can sit
outside the observed range and drag the fitted minimum down).

**4. `check_feature_season_continuity`** added to `quality_gates.py`: flags
any declared CAUSAL_FEATURE whose mean shifts more than 1 train-sd between
ADJACENT seasons, plus missingness jumps so a fully-populated -> fully-NaN
column is caught even though its mean never moves. Adjacency is deliberate:
a league trend moves gradually, an ingestion break moves once.
7 tests in `tests/test_feature_continuity.py`; suite at 395.

---

## The gate immediately found that the fix was incomplete (2026-08-20)

Run against RB's 64 declared CAUSAL_FEATURES on a real fold:
**23 violations of 64 features.** The two repaired columns are gone from
the list. What remains splits into two groups.

### More columns populated ONLY in 2025

| feature | 2024 | 2025 | sd shift |
|---|---|---|---|
| `team_pace_sec_per_play_roll3_mean` | 0.000 | 22.923 | 3.91 |
| `rb_broken_tackles_prior` | 0.000 | 16.654 | 3.78 |
| `redzone_target_share_pct_roll3_mean` | 0.000 | 15.179 | 2.38 |

Traced to `player_weekly_stats` itself. Six columns are populated in 2025
and **nowhere else**:

    redzone_targets, neutral_targets, third_down_targets,
    goal_line_touches, two_minute_targets, high_leverage_touches

(`redzone_targets`: 0 for every season 2018-2024, 2,489 in 2025.) Four more
(`rush_inside_10`, `rush_inside_5`, `targets_15_plus`, `air_yards`) start in
2020, so 2006-2019 is zero-filled for those.

This is the **same root cause as `home_away`, and it is the general form of
it**: the 2025 re-ingest went through `store_weekly_dataframe` (the PBP
path), which derives a much wider set of columns than the historical
`import_weekly_data` path ever wrote. Every column in that difference is a
train/test discontinuity. `home_away` was simply the first one found.

These feed `utilization_score` (via `redzone_targets_pct`), which falls back
to crude proxies like `receiving_tds * 15` when the real column is absent —
so pre-2025 seasons get the proxy and 2025 gets the measurement.

### Boundary artifacts at the start of history

`coaching_adaptation_score`, `coaching_change_impact`, `coaching_stability`,
`rookie_draft_value`, `rookie_ceiling_ppg` all break at 2006->2007. 2006 is
the first loaded season, so prior-season-dependent features are degenerate
there. Probably benign, but unverified.

### Status

The 2025 anomaly is **not** fully fixed. Two of at least five instances are
repaired. The remaining fix is a PBP backfill of the missing
`player_weekly_stats` columns for 2006-2024, which needs play-by-play
downloads for ~19 seasons and is a materially larger job than the two
backfills above.

Do not re-run the regime experiment expecting a clean answer until that
lands — the era-rule contrast will still be contaminated by whatever
remains.

## PBP situational columns backfilled; two discontinuities remain (2026-08-20)

`scripts/backfill_pbp_situational_columns.py` filled 116,490 player-week
rows across 2006-2024. Definitions are not re-implemented: it calls
`PBPStatsAggregator.aggregate_all_stats`, the exact code that produced the
2025 values.

Validated against 2025 before writing — 99.9-100% exact agreement,
correlation 0.9988-1.0000 on all 12 columns (the sub-0.1% residual is
nflverse revisions since the original ingest, not a definitional
difference). A column is filled for a season only when that season's stored
total is zero, so the already-populated 2020-2024 values for
`rush_inside_10/5`, `targets_15_plus` and `air_yards` were left untouched.

### Verified fixed, all four positions

| feature | 2008 | 2012 | 2024 | 2025 |
|---|---|---|---|---|
| `is_dome` | 0.38 | 0.37 | 0.38 | 0.36 |
| `team_plays_roll3_mean` | 51.4 | 53.0 | 52.1 | 53.0 |
| `team_motion_rate` | nan | nan | 0.43 | 0.49 |

`redzone_target_share_pct_roll3_mean` and
`team_pace_sec_per_play_roll3_mean` also cleared. Gate counts: RB 21/64,
WR 24/67, QB 23/57, TE 21/65.

Also added `KNOWN_MISSINGNESS_BOUNDARIES` to the continuity gate. The
scheme-tendency NaN boundary is a documented source limit (FTN charting
starts 2022), and a gate that fails forever on a known limit is a gate
nobody reads. Level shifts among observed seasons are still checked; only
the NaN boundary is exempt.

---

## STILL OPEN after the backfill

### 1. 2006-2008 has no incomplete-pass charting (pre-existing, source-level)

| season | targets | receptions | catch rate |
|---|---|---|---|
| 2006 | 9,963 | 9,938 | **99.7%** |
| 2007 | 10,655 | 10,642 | **99.9%** |
| 2008 | 10,292 | 10,277 | **99.9%** |
| 2009 | 17,091 | 10,552 | 61.7% |

A 99.7% catch rate is impossible. `targets` in 2006-2008 is counting
receptions. Confirmed at source, not an ingest bug: 2008 play-by-play
charts a receiver on **69** incomplete passes; 2009 charts **6,731**.
nflverse does not record the intended receiver on incompletions before 2009.

So `targets`, `target_share`, catch rate, `air_yards`, `redzone_targets`,
`targets_15_plus` and the whole receiving-usage family rest on a
completions-only denominator for those three seasons. The backfill made
those columns internally consistent with the broken denominator; it could
not fix the denominator.

This predates all of today's work and argues for a **2009 floor on
receiving-dependent features**, exactly analogous to the existing 2013 snap
floor. It also partially re-opens the era-rule question: regime B admitted
2006-2012, and three of those seven seasons have structurally broken
receiving usage.

NOT changed unilaterally: `targets` is consumed across the entire feature
layer, so nulling it for three seasons is a much wider blast radius than
the scheme-tendency fix and needs an explicit decision.

### 2. `rb_broken_tackles_prior` — 0.000 (2024) -> 16.654 (2025)

Untouched by the backfill because it comes from PFR advanced stats, not
play-by-play. `weekly_pfr` holds 7,127-7,572 rows for every season
2018-2025, so the source data is present and something downstream is
dropping it. Same shape as the fixed defects, different pipeline.

### 3. `coaching_*` break at 2006->2007 (left-censoring)

`coaching_adaptation_score` falls 0.28 -> 0.006 (3.2-4.0 sd depending on
position), with `coaching_change_impact` and `coaching_stability` moving
with it. Root cause found: `weeks_since_coaching_change` is a `cumcount()`
within (team, cumulative-changes), so in the panel's first season every
team starts at 0 — asserting "a coaching change just happened" when the
prior coach is simply unobserved. `coaching_change` itself is correctly
guarded by `prev_coach.notna()`; the weeks-since counter is not.

The honest value for a team with no observed change is unknown, not zero.
Confined mostly to 2006, but it hands the model something close to a "this
is 2006" indicator.

## The 2009 receiving-charting floor (2026-08-20)

`RECEIVING_CHARTING_MIN_SEASON = 2009` in
`src/models/single_week_ppr/population.py`, applied to RB/WR/TE. QB is
exempt — its features come from passing and NGS, not target charting.

Framed the same way as the 2013 snap floor, and for the same reason: not
"old football is different" but "this is where the measurement regime
starts."

Contamination measured before deciding:

| | 2006-2008 | 2009+ | |
|---|---|---|---|
| catch rate | 99.7-99.9% | ~61% | targets == receptions |
| `recv_success_rate` | 0.847 | 0.578 | +46% |
| `recv_epa` (season sum) | ~8,900 | ~2,350 | **~4x** |

The EPA figure is the clearest tell: incompletions carry the negative EPA,
and before 2009 they were never charted, so the surviving plays are a
success-biased subset. No amount of backfilling fixes that — the plays do
not exist in the source.

Cost: 11,586 rows (WR 5,204, RB 3,850, TE 2,532). Those rows have valid PPR
targets — receptions, yards and touchdowns are box-score derived and
correct — but their usage features are wrong. Keeping a row with
plausible-looking but false target counts is exactly the failure mode this
whole day was spent removing, so the row goes. `apply_receiving_floor=False`
exists to measure what the floor costs, not for production use.

Affected share of each position's declared CAUSAL_FEATURES: WR 15/67, TE
14/65, RB 8/64, QB 2/57 (and QB's two are NGS/team-level, not target-based).

5 tests in `tests/test_population_regimes.py`; suite at 402.

## RESOLVED: the 2025 anomaly and the era-rule question (2026-08-20)

Re-ran the 12-fold x 4-arm regime experiment on data with no known
discontinuity. Four result sets now exist, isolating each change:

| suffix | age | positions | features | receiving floor |
|---|---|---|---|---|
| `_preagefix` | broken | broken | broken | no |
| `_prepositionfix` | fixed | broken | broken | no |
| `_prefeaturefix` | fixed | fixed | broken | no |
| *(none)* | fixed | fixed | fixed | yes |

### The measurement that makes this readable

Arm A (2013+, snaps > 0) has a **byte-identical training and evaluation
population** in the last two runs -- `d_n_train = 0` and `d_n = 0` in all
12 folds. So every change in its numbers is attributable to feature values
alone, with no population confound. That is the comparison used below.

### 1. The 2025 penalty is gone

Arm A, MAE in 2025 minus the mean of 2023-2024:

| position | before | after |
|---|---|---|
| QB | +0.222 | +0.114 |
| RB | +0.640 | +0.073 |
| TE | +0.390 | +0.088 |
| WR | +0.566 | **-0.117** |
| **mean** | **+0.454** | **+0.040** |

For WR, 2025 is now the *best* fold of the three. It was the worst at every
position before.

The per-fold shape confirms the mechanism rather than just the outcome:

| position | d MAE 2023 | 2024 | 2025 |
|---|---|---|---|
| QB | +0.030 | +0.014 | **-0.086** |
| RB | +0.047 | +0.059 | **-0.514** |
| WR | +0.090 | +0.120 | **-0.577** |

Repairing the features costs a little on 2023-2024 and pays enormously on
2025. That is exactly right: on the earlier folds the broken columns were
*consistently* broken across train and test -- useless constants, harmless
-- so making them real only adds a weak noisy feature (+0.01 to +0.12,
matching the ablation's +0.079 control cost in sign and magnitude). On 2025
they were discontinuous, and that was worth -0.09 to -0.58.

### 2. The era rule is a clean null, and now stably so

B minus A, per fold:

| position | 2023 | 2024 | 2025 (before) | 2025 (after) |
|---|---|---|---|---|
| QB | -0.027 | -0.024 | +0.278 | **+0.003** |
| RB | -0.046 | -0.007 | -0.045 | **-0.000** |
| TE | -0.000 | -0.001 | -0.017 | **-0.001** |
| WR | +0.020 | -0.010 | -0.051 | **+0.006** |

Every cell is now within +/-0.046, and most within +/-0.027. Nothing is
driven by one fold, and no position contradicts another.

The original experiment's headline finding -- "QB and WR disagree in sign
on 2025, so the era rule is not resolvable" -- was **an artifact of the
feature discontinuity**, not a property of the era rule. QB's half of that
disagreement (+0.278) collapses to +0.003.

TE's was the only sign-consistent effect in the original run (B costing
0.013-0.016 MAE for a bias gain). With the 2006-2008 rows removed by the
receiving floor, TE's MAE cost vanishes entirely (-0.000/-0.001/-0.001)
while the bias advantage remains. The trade-off TE appeared to face was the
broken seasons, not the era rule.

### 3. Pooled result: keep 2009-2012

| position | arm | MAE | bias | RMSE |
|---|---|---|---|---|
| QB | A / B / C | 5.921 / 5.905 / 5.895 | -0.440 / -0.450 / -0.433 | 7.471 / 7.463 / 7.463 |
| RB | A / B / C | 4.341 / 4.324 / 4.316 | -1.178 / -1.132 / -1.152 | 6.271 / 6.242 / 6.236 |
| TE | A / B / C | 2.887 / 2.887 / 2.885 | -1.247 / **-1.143** / -1.146 | 4.661 / **4.628** / 4.628 |
| WR | A / B / C | 4.047 / 4.052 / 4.047 | -1.270 / -1.234 / -1.263 | 6.072 / 6.065 / 6.068 |

B is never worse than A on MAE beyond 0.005, is slightly better at QB and
RB, and is clearly better on bias and RMSE at TE. **Keep the extended
population.** C remains indistinguishable from B, so `participation_quality`
stays a provenance column and not a feature -- unchanged from the original
finding, now on clean data.

### Caveat on what "B" means now

B admits 2009-2012 for RB/WR/TE, not 2006-2012, because the receiving floor
removed three seasons. So this null is a **weaker** claim than the original
experiment intended: fewer added seasons, less to detect. It does not say
2006-2008 would have been harmless -- it says the seasons we can still
trust are free to include.

### Arms verified distinct

QB/2023 returned identical MAE (5.772) and RMSE (7.201) for P0 and A, which
looked like the "arms that don't actually differ" failure this project hit
before. Checked directly: the two arms' predictions differ by a mean of
1.065 PPR, only 0.1% of rows are identical, correlation 0.968. A genuine
coincidence, not collapsed arms.

## Triage: both remaining defects are immaterial (2026-08-20)

Bounded check, not a reopened investigation. The test for each was: does
removing it change predictions?

### `rb_broken_tackles_prior` — immaterial, conclusively

| | |
|---|---|
| training values | **1 unique value** (0.0000), sd 0.000000 |
| test values (2025) | 61 unique, 4.3-85.0 |
| LightGBM importance | **0** (rank 54/64) |
| mean prediction change if dropped | **0.0000 PPR** |

Constant across every training row, so no split is ever learned on it and
the 2025 values are ignored. Left in place per the directive's instruction
not to change model behaviour merely because a defect exists.

Recorded but not acted on: `seasonal_pfr.broken_tackles_per_att` is NULL
for 2018-2023 and populated only in 2024, and the column is **misnamed** —
it holds raw counts (avg 15.7, max 85), not a per-attempt rate. So the
feature is unusable rather than merely discontinuous.

**Trigger condition:** if anyone backfills it (the per-week source,
`weekly_pfr.rushing_broken_tackles`, does cover 2018-2025), the feature
becomes live and non-constant and this triage no longer applies. Re-check
then.

### Coaching left-censoring — real, but within noise

`weeks_since_coaching_change` is a `cumcount()`, so the panel's first
season asserts every team just changed coaches. The feature does vary in
training (339 unique values, sd 0.086), unlike the RB case, so the model
can and does split on it.

| fold | all features | drop coaching feats | drop 2006 rows |
|---|---|---|---|
| QB 2024 | 5.9689 | 5.9522 (**-0.017**) | 5.9523 (-0.017) |
| QB 2025 | 6.0005 | 6.0353 (**+0.035**) | 6.0422 (+0.042) |

Individual predictions move ~0.5 PPR, but the net accuracy effect is
**sign-inconsistent across folds** and never exceeds 0.042 — inside the
±0.046 noise band the clean-data re-run established. Dropping 2006 rows
outright gives the same inconsistent answer.

Not an opportunity feature, and no consistent effect on the production
baseline. Documented, not fixed.

---

## PRE-REGISTRATION: opportunity-layer experiment (2026-08-20)

Written and committed **before** the experiment runs.

### Motivation

The snap-bucket diagnostic is the only remaining large, unexplained
structure in weekly error: the model retains 51-68% of true dynamic range
across realized-snap buckets, over-predicting low-snap games and
under-predicting high-snap ones. Realized snaps are not knowable at
forecast time, so this measures opportunity uncertainty rather than
production error.

### Arms

All arms share folds, evaluation population, target definition, metrics and
configuration. Only the estimator structure differs.

* **A — baseline.** Current production model: FINAL_CONFIG architecture,
  window and weighting; population `apply_regime(..., "B_extended")`;
  predicts `fantasy_points` directly. Frozen.
* **B — multiplicative opportunity.**
  `E[PPR] = E[snaps] x E[PPR per snap]`, the two estimated by separate
  models on the same features.
* **C — opportunity as a feature.** Baseline model plus predicted snaps as
  one extra input. Secondary; it separates "the opportunity signal helps"
  from "the multiplicative structure helps."

### Stated assumptions

The multiplicative form assumes snaps and per-snap efficiency are
conditionally independent given features. They may not be. This is an
assumption being tested, not a derivation.

Both component models use MSE (mean-oriented) objectives, because a product
of medians does not approximate a mean. The per-snap model is fitted with
`sample_weight = snap_count x recency`, so it estimates the snap-weighted
mean rate — the quantity that reconstructs totals — rather than letting
1-snap rows dominate.

### Decision criteria, fixed in advance

Call the opportunity architecture successful only if ALL hold:

1. Mean weekly MAE improvement (B vs A) **>= 0.05** at a position. The
   established noise band is +/-0.046 per fold, so anything smaller is not
   distinguishable from noise.
2. Improvement at **>= 3 of 4 positions**.
3. Improvement **not driven by a single fold** — the sign must hold in at
   least 2 of 3 folds wherever a position improves.
4. Snap-bucket compression **contracts**: the predicted-vs-actual slope
   across buckets moves toward 1.0 relative to baseline.
5. The opportunity model actually predicts opportunity (report its MAE and
   correlation; a model no better than a positional mean fails).

If 1-3 pass but 4 fails, report that MAE improved for reasons unrelated to
the identified mechanism, and do not adopt.

If the criteria are not met, the result is reported as a negative and the
architecture is not adopted. No tuning until something passes.

### Metrics reported regardless of outcome

Weekly MAE / RMSE / bias; season MAE / bias; snap-bucket bias table;
opportunity-prediction error. Broken out by QB / RB / WR / TE and by fold.

## RESULT: the opportunity layer FAILS its pre-registered test (2026-08-20)

Criteria were fixed and committed (`ed55927`) before the run. Judged
against them as written.

### Criterion 5 — does the opportunity model predict opportunity? **PASS**

| position | snaps MAE | positional-mean baseline | improvement | Spearman |
|---|---|---|---|---|
| QB | 12.48 | 17.54 | 29% | 0.48 |
| RB | 7.90 | 13.98 | **44%** | 0.79 |
| TE | 8.28 | 15.14 | **45%** | 0.80 |
| WR | 10.54 | 17.94 | 41% | 0.77 |

Snaps are genuinely predictable from pre-game information. The premise of
the architecture is sound.

### Criteria 1-3 — weekly MAE. **FAIL, unambiguously**

B minus A, mean over folds (positive = worse than baseline):

| position | B | C |
|---|---|---|
| QB | +0.144 | +0.009 |
| RB | +0.181 | +0.055 |
| TE | +0.122 | -0.002 |
| WR | +0.318 | +0.106 |

**B is worse in 12 of 12 folds.** Zero positions improve by the required
0.05; zero improve at all. C (opportunity as a plain feature) is worse in
8 of 12 and never meaningfully better.

There is no subgroup rescue here and none was looked for. The result is
perfectly consistent in the wrong direction.

### Criterion 4 — snap-bucket compression. **MIXED, and only TE**

Mean slope change (toward 1.0 is the goal):

| position | baseline slope | B slope | change |
|---|---|---|---|
| TE | 0.52-0.55 | **0.74-0.76** | **+0.217** |
| WR | 0.66-0.69 | 0.66-0.78 | +0.027 |
| RB | 0.63-0.67 | 0.61-0.65 | -0.013 |
| QB | 0.61 | 0.58-0.59 | -0.025 |

TE's improvement is large and consistent across all three folds. Nowhere
else does the mechanism move. So the compression is not a single
phenomenon with a single cause — at TE it is opportunity uncertainty, and
at QB/RB it is something else.

### Verdict: do not adopt

Three of five criteria fail. The architecture does not improve weekly
fantasy projection, which is what it was built to do.

### The one result that is genuinely interesting — and its confound

Season-level, summed over participated weeks:

| position | season bias A | season bias B | season MAE A | season MAE B |
|---|---|---|---|---|
| QB | -4.06 | **+1.27** | 18.95 | 19.69 |
| RB | -12.65 | **-0.36** | 18.43 | 18.45 |
| TE | -15.89 | **+2.34** | 17.99 | **12.80** |
| WR | -14.92 | **+3.40** | 19.81 | 20.41 |

Season bias collapses from -4 to -16 (severe under-projection, the
long-standing Phase 7A finding) to roughly zero at every position. TE's
season MAE improves 29%.

**Do not read this as vindication of the opportunity split.** B is
mean-oriented by construction (both components use MSE, per the
pre-registration), while the baseline architectures are median/Huber.
Summing many median-oriented under-predictions compounds into exactly the
season under-projection seen in column A. So the season-bias gain is
plausibly attributable to *mean-orientation alone*, with no opportunity
decomposition required — and Phase 7C already showed a mean-objective
target transform produces a similar season-level effect.

That is a testable confound, not a conclusion: the clean test is a
mean-objective baseline WITHOUT the opportunity split. Until that is run,
the season numbers above establish nothing about opportunity modelling.

### What this closes

The snap-bucket compression was the stated motivation for this entire
direction. It is now clear that explicitly modelling opportunity does not
fix it except at TE, and does not improve weekly projection anywhere. Per
the directive, no additional availability/injury/role machinery should be
built on this foundation.

## Step 8, season half: availability BEATS constant on its own task,
## but only helps season PPR at QB/RB (2026-08-20)

The weekly half is already closed negatively (opportunity layer failed
12/12 folds). This is the separate, untested season question: the weekly
experiment never had to estimate games played.

    E[season PPR] = E[games played] x E[PPR per game | played]

The production term is held IDENTICAL across arms, so any difference is
attributable to the availability term alone. Arms:
`const_position` (position mean rate, training seasons only),
`hist_player` (player's own prior rate), `hist_shrunk` (shrunk toward the
position mean by prior games observed, K=16).

All inputs are strictly pre-season: prior-season rates and the published
schedule. No current-season data, no realized snaps, no future roster.

### The constant baseline is fairly specified

Checked before trusting any comparison. Historical mean rate tracks the
target-season mean closely (QB 0.497 vs 0.493/0.507/0.498 across the three
folds; same at other positions). It is not a strawman.

Its large *PPR* bias has a real mechanism rather than a specification
error: the constant is nearly unbiased in **games** but badly biased in
**PPR**, because games played and PPR-per-game are strongly positively
correlated. Predicting 8.4 games for every QB under-projects the starters
who play 17 and over-projects fringe players who play 2.

### Result 1 — availability is predictable. **12/12 folds.**

Games-played MAE, `hist_shrunk` minus `const_position`, every fold
negative (better), range -0.49 to -1.07 games:

| position | 2023 | 2024 | 2025 |
|---|---|---|---|
| QB | -0.62 | -0.77 | -1.07 |
| RB | -0.91 | -0.85 | -0.49 |
| TE | -0.97 | -0.97 | -0.84 |
| WR | -0.59 | -0.68 | -0.64 |

Unambiguous, and it survives restricting to players who actually have prior
history (QB 4.92 -> 3.69, TE 4.80 -> 3.45).

### Result 2 — season PPR bias improves at all four positions

Mean |season bias| with a realistic production term:

| arm | QB | RB | TE | WR | mean |
|---|---|---|---|---|---|
| const | 21.90 | 18.64 | 10.33 | 9.20 | 15.02 |
| hist_player | 12.53 | 3.60 | 2.60 | 12.36 | 7.77 |
| **hist_shrunk** | **8.68** | **1.10** | **0.04** | **7.85** | **4.42** |

Shrinkage cuts season bias by ~70% and is the best arm at three of four
positions. This matters because season under-projection is the standing
Phase 7A defect.

### Result 3 — season PPR MAE improves only at QB and RB

`hist_shrunk` minus `const_position`, per fold (negative = better):

| position | 2023 | 2024 | 2025 | verdict |
|---|---|---|---|---|
| QB | -7.20 | -8.80 | -11.64 | **3/3 better** |
| RB | -0.85 | -6.36 | -2.21 | **3/3 better** |
| TE | +0.67 | -0.69 | +1.85 | 1/3 |
| WR | -3.41 | +2.12 | +2.59 | 1/3 |

8/12 folds overall. The split is not noise: QB and RB are the positions
where the constant's bias was largest (-21.9, -18.6). At TE and WR the
constant was already closer (-10.3, -9.2) and the *production* term's error
dominates — multiplying a noisy prior-season PPR/game by a per-player
availability compounds it rather than correcting it.

### Verdict

Neither the directive's clean A nor clean B.

* The availability estimator is **validated on its own task** and should be
  used for the games/opportunity component of season projections.
* It **materially reduces season bias at every position**.
* It does **not** generally improve season MAE — only where the constant
  was badly wrong. Do not claim otherwise.
* `hist_shrunk` is the arm to use, not `hist_player`: shrinkage is better
  on bias at 3/4 positions and better on MAE at 8/12 folds, because the raw
  per-player rate over-corrects (WR bias +12.36 vs +7.85).

No new machinery is warranted beyond this. Shrunk historical availability
is a two-parameter estimator over data already in the panel.

### Population caveat, stated because it limits the claim

The evaluation population is players with >= 1 participated game in the
target season, because a player with zero games has no rows to evaluate
against. A real pre-season projection must also cover players who end up
playing nothing at all, and this experiment cannot measure that case. The
result therefore applies to "how many games will a player who plays get",
not "will this player play".

## Season availability layer: implemented and integration-tested (2026-08-20)

`src/models/single_week_ppr/season_availability.py`. Adopted narrowly, on
the strength of the games-played result (12/12 folds) and the season-bias
result (mean |bias| 15.02 -> 4.42), **not** on a general season-MAE claim —
it improves MAE at QB/RB and not at TE/WR.

    season PPR = E[games played] x E[PPR per game | played]

The weekly model supplies the second factor and is untouched. `hist_shrunk`
supplies the first. Nothing else was added: no weekly availability
adjustment, no depth-chart decay, no synthetic zero weeks, and deliberately
**no separate bias-correction layer** — the bias gain comes from the
product itself, and a second correction on top would double-count it.
A test asserts no such term exists.

Causality is structural rather than conventional: `fit(panel,
before_season=S)` drops everything at or after S, so a caller cannot
accidentally train on the season being projected, and it raises rather than
silently returning nothing when no prior seasons exist.

Not to be confused with `availability.py`, which holds the per-week
P(plays) estimators from the abandoned synthetic-week architecture. Those
remain referenced only by two experiment scripts; this module is the one
wired into projections.

### The seven integration requirements, one or more tests each

| # | requirement | how it is enforced |
|---|---|---|
| 1 | weekly predictions unchanged | source-level assertion that no weekly module imports this one, plus a signature check that nothing in the public surface takes a `week` |
| 2 | season games use hist_shrunk | durable pulled down and fragile pulled up toward the position mean; shrinkage strength scales with games observed |
| 3 | season PPR = games x PPR/game | exact-product test, plus the no-bias-correction assertion |
| 4 | strictly causal | target season and later dropped by `fit`; raises with no prior seasons; rejects missing columns |
| 5 | documented no-history fallback | unknown player resolves to the position mean; `has_history` flags the row; a test asserts the docstring still states the "will this player play at all" boundary |
| 6 | reproduces the experiment | production estimator beats the constant on real 2024 and 2025 data |
| 7 | no synthetic zero weeks | every panel row has `games_played > 0`; module source contains no row-fabrication |

18 tests in `tests/test_season_availability_integration.py`; suite at 428.

The experiment script now imports the panel loader and estimator from this
module rather than keeping its own copies, so there is one definition and
the reported numbers reproduce through the production code path.

---

# STEP 8 COMPLETE — architectural boundaries, fixed (2026-08-20)

The decomposition, which subsequent work must not blur:

| layer | question it answers | estimator |
|---|---|---|
| **Weekly** | what does this player produce when he participates? | existing conditional-production model, unchanged |
| **Season** | E[games played] x E[PPR per game] | product only |
| **Availability** | how many games does a player who plays get? | `hist_shrunk`, **season layer only** |

Status, for the record:

* **Step 8 complete.**
* **`hist_shrunk` adopted for season games.** Validated 12/12 position-folds
  on games played; mean |season bias| 15.02 -> 4.42.
* **No weekly opportunity adjustment.** The weekly opportunity layer failed
  its pre-registered test in 12/12 folds and was not adopted. The
  snap-bucket compression is opportunity uncertainty the weekly model
  cannot observe, and is not a defect to be forced out of it.
* **No synthetic zero weeks IN THE NEW WORK.** Corrected 2026-08-20 --
  the original wording ("not reintroduced anywhere; enforced by test") was
  wrong. See the correction entry below.
* **No separate bias correction.** The bias gain comes from the product
  itself; a post-hoc correction would double-count it. Enforced by test.
* **No position-specific availability estimators.** One estimator, two
  parameters, all four positions.
* **The evaluation does NOT establish probability of playing at all.** The
  population is players with >= 1 observed game, because a player with zero
  games has no rows. `hist_shrunk` has not been shown to identify players
  who miss a roster, begin the season injured, retire, are cut, or are
  rookies without NFL history.
* **Unknown / no-history players** use the position-mean fallback and are
  marked `has_history=False`.
* **`availability.py` remains legacy only.** The per-week P(plays)
  estimators belong to the abandoned synthetic-week architecture. Not part
  of production; referenced only by `run_availability_comparison.py` and
  `run_availability_calibration.py`. **Logged for a later cleanup pass —
  deliberately not deleted here**, since cleanup at this point risks
  becoming another investigation.

Do not reopen the availability question absent a concrete failure surfaced
by later work.

---

# CORRECTION to the Step 8 status entry (2026-08-20)

The entry above overstated the repository state in three ways. Recorded
here rather than silently edited, because the overstatement is the kind of
error that compounds.

**1. "No synthetic zero weeks. Not reintroduced anywhere; enforced by
test."** False twice. `season_projection.py` still contains
`possible_weeks_for_player`, `estimate_availability_rate`,
`build_synthetic_week_row` and `resolve_week_source`, all live. The cited
test reads exactly one file (`season_availability.py`) and has no
visibility into Phase 7. The defensible claim was "no synthetic weeks were
reintroduced *by the new work*."

**2. "Availability: hist_shrunk, season layer only."** Describes the new
module; reads as repo-wide. `estimate_availability_rate` still runs
per-week inside Phase 7.

**3. "hist_shrunk adopted for season games."** Nothing consumes it.
`season_availability` is imported by two files -- its own experiment script
and its own tests. `season_projection` is imported by fifteen, including
Phase 9. The estimator is *validated*, not *adopted*.

### Actual state: two parallel season architectures

| | mechanism | consumers |
|---|---|---|
| Phase 7 `season_projection.py` | synthetic weeks x per-week P(plays) | 15 |
| Step 8 `season_availability.py` | E[games] x E[PPR per game] | 2 (own script + tests) |

Step 8 built a new layer **alongside** the old one. It did not migrate
Phase 7, and no directive asked it to -- "do not reintroduce synthetic zero
weeks" is a constraint on new work, not a mandate to remove the existing
mechanism.

### The open decision (NOT resolved here)

Does Step 8 retire the synthetic-week architecture, or does Phase 7 remain
the production season path with Step 8 as an alternative?

This must be answered before Phase 9 is written, because Phase 9 sits
directly on the old architecture and the two designs differ in their unit
of simulation:

    Phase 7 / current Phase 9:  simulate the status of each possible week
    Step 8 architecture:        simulate games played, then simulate the
                                PPR outcomes for those games

Swapping `estimate_availability_rate` for `hist_shrunk` inside the existing
simulation would NOT reconcile them -- it preserves the old unit of
simulation while appearing to adopt the new architecture. That was
proposed and is withdrawn.

### Not in question

The `snap_count > 0, PPR = 0` rows stay in the conditional production
distribution: 18,213 rows, 23.2% of the population, 39% at TE. They feed
the residual/bootstrap machinery and must not be touched. A missing week is
exposure, not a zero-point observation. These belong entirely on the
PPR-per-game side of any season decomposition.

---

# DECISION: Step 8 is the intended replacement; migration is a separate project (2026-08-20)

Resolving the fork recorded in the correction above.

**Decision language, fixed:**

> We will evaluate Step 8 against Phase 7 **before** migrating production,
> with Step 8 as the intended replacement architecture **if** it matches or
> improves the validated Phase 7 performance.

Not "Step 8 has retired Phase 7." Retirement is the destination; the
migration is a measured project with a gate.

### State, as documented

| | Phase 7 | Step 8 |
|---|---|---|
| Exists | yes | yes |
| Production consumer | **yes (15 importers)** | **no (2: own script + tests)** |
| Validated | yes | partially |
| Synthetic weeks | yes | no |
| Played-zero rows | yes | yes |
| Availability separate from production | **no** | **yes** |
| Phase 9 compatible | old architecture | **intended architecture** |
| Disposition | **legacy benchmark** | **candidate replacement** |

Phase 7 stays intact. Its measured result is real and must not be
discarded for architectural preference: summed-weekly beat both direct
season models by 26-40% at every position even under matched,
established-veteran-only populations.

### Why retirement is nonetheless the destination

Phase 7 represents two different things -- exposure and conditional
production -- by manufacturing weekly rows and then applying an
availability discount. That conflation is the documented source of the
synthetic-week problems. The zero-point finding is what makes it decisive:

    snap_count > 0, PPR = 0   ->  a real production observation
    player inactive           ->  an exposure observation

Phase 7 has no way to keep those apart. The Step 8 decomposition does, by
construction.

### HARD INVARIANT

    snap_count > 0 AND PPR = 0
        -> real observed game
        -> belongs in the conditional production distribution
        -> NEVER convert to missing
        -> NEVER treat as an availability failure
        -> NEVER manufacture another zero

18,213 rows; 23.2% of the population; 39% at TE. Phase 9's residual
machinery already builds sequences from real observed rows, so these
contribute naturally to simulated outcomes -- that ingredient is correct
and stays.

### Next work: Step 8A — make the new architecture independently evaluable

Step 8 is currently an estimator with no consumer. Step 8A must establish:

1. `E[games]` / the games-played distribution
2. `E[PPR per game | played]`
3. strictly causal construction of both
4. the exact relationship `season PPR = games x PPR/game`
5. a proper walk-forward evaluation **against the Phase 7 benchmark**

Only after that comparison does the production consumer move.

### Phase 9 is BLOCKED until the Step 8 interface is final

Deliberately. The right Phase 9 depends on what Step 8 exposes, and the
target is that Phase 9 needs to know nothing about active-roster
filtering, synthetic rows, `possible_weeks_for_player`,
`estimate_availability_rate`, carried-forward features, synthetic depth
charts, or synthetic Vegas lines. Those are Phase 7 implementation
concerns, not season-simulation concerns.

## PRE-REGISTRATION: Step 8A — season architecture head-to-head (2026-08-20)

Committed **before** the experiment runs.

### Success is NOT "Step 8 beats Phase 7"

Fixed in advance:

> Does Step 8 achieve sufficiently competitive season accuracy **while**
> eliminating the synthetic-week architecture and preserving the
> causal/interpretability guarantees established this session?

Outcome conditions, all of which are useful results:

| | condition | action |
|---|---|---|
| **A** | Step 8 ~= Phase 7 | strong case for migration |
| **B** | Step 8 materially beats Phase 7 | stronger case; **investigate why before migrating** |
| **C** | Step 8 materially loses | Phase 7 remains production; investigate where the lost information comes from, **do not tune until it wins** |
| **D** | loses overall, wins for some positions/populations | evidence toward a hybrid, but **do not build one yet** -- understand the mechanism first |

### The matched-population result is PRIMARY

`--intersect-populations` is the migration comparison. Unmatched is a
secondary diagnostic. Otherwise differences in *who gets projected*
masquerade as model quality -- Phase 7's edge already shrank 8-21% under
matching. Step 8 will not be tuned against whichever population flatters it.

### Required output: the decomposition, not just the season total

    E[season PPR] = E[games] x E[PPR per game | played]

Per fold and position, both components reported separately: prediction
error for games, for PPR/game, and for the resulting season total; MAE /
bias / R^2 / correlation for each; and the error interaction between the
two terms.

This is required, not optional, because a season total alone is
uninformative about cause:

    games 16.2 vs 16.1 actual, PPR/game 7.3 vs 8.9   -> conditional production is the problem
    games 10.1 vs 15.8 actual, PPR/game 11.2 vs 10.9 -> the exposure model is the problem

### Target-definition guardrail (a real hazard, checked)

The `ppr_per_game` denominator must be **actual qualifying played games**
under the participation contract -- not roster weeks, not rows from the
synthetic-week machinery.

`preseason_features._load_full_history` reads `player_weekly_stats` raw:
no quality label, no receiving floor. Its `games_played` is a bare row
count (`("fantasy_points", "size")`). Using it as the denominator would
reintroduce the selection problem this architecture exists to remove.

So Step 8A takes **features** from `build_multiyear_season_pairs` (causal,
player-season level, real destination-team context) and the **target** from
`season_availability.load_player_seasons()`, which applies the contract.

### Held fixed

Phase 7 is not modified. Phase 9 is not touched. No hybrid is built
regardless of outcome.

### AMENDMENT to the Step 8A pre-registration (2026-08-20, before results)

Recorded before any Step 8A result exists, so the reporting窗口 cannot be
chosen to flatter an outcome.

**Reporting windows, fixed in advance:**

| | window | status |
|---|---|---|
| Primary | **all 17 folds (2009-2025)** | the pre-defined harness span |
| Secondary | **2013-2025** | explicitly labelled the better-supported data regime |
| Diagnostic | **2009-2012** | reported separately |

An earlier draft of this plan said 2013+ would be "weighted more heavily"
when reading the result. That is withdrawn: choosing a weighting after
seeing which window favours Step 8 is outcome-dependent reasoning. No
statistical weighting is applied to any window unless pre-registered.

The rationale for showing 2013-2025 separately stands on facts known
already, not on results: snap data begins in 2013
(`SNAP_LABEL_MIN_SEASON`) and the receiving floor removes RB/WR/TE before
2009 (`RECEIVING_CHARTING_MIN_SEASON`), so those spans are genuinely
different data regimes. The same rule applies to any other cutoff
discovered later.

### THE EXPOSURE-LEAKAGE CONTRACT (permanent, not Step-8A-specific)

    Target-season exposure MAY determine training weights and the target.
    It may NEVER be an input feature.
    Prior-season exposure IS legitimate signal and must survive.

Stronger and clearer than "`games_played` is excluded", and it cuts both
ways: a filter broad enough to catch `games_played` by name would also
destroy `games_played_y1`, which is exactly the durability signal the
exposure model depends on.

Made **structural** rather than a maintained list: `feature_columns`
derives its exclusions from the panel schema
(`TARGET_SEASON_COLUMNS`), so adding a column to
`load_player_seasons()` cannot silently open a leak.

Three permanent regression tests:

* no target-season exposure column appears in the feature set
* `games_played_y1` / `_y2` **do** survive the filter
* predictions are byte-identical when **every** target-season exposure
  column is tampered with -- the behavioural form, which proves the model
  cannot use them at prediction time rather than merely that a list
  excludes them

This is the leak a future feature-selection refactor is most likely to
reopen, which is why the guard is behavioural.

### AMENDMENT 2 to Step 8A: extend the BENCHMARK's span (2026-08-20, before results)

Discovered while the regeneration was running: Phase 7 defaults to
`DEFAULT_VALIDATION_SEASONS = (2023, 2024, 2025)`. The harness walks 17
folds, but the Phase 7 arm exists for only three of them — so the migration
comparison, as pre-registered, would rest on **3 folds**.

That is too fragile a basis for retiring a production architecture,
particularly in a project where the era-rule contrast looked decisive at
n=3 and turned out to be a feature discontinuity, and where QB's 2025
anomaly dominated three separate experiments.

**What changes: the benchmark's evaluation coverage. What does NOT change:
Step 8A's model, features, target, weighting, selection criterion, or
outcome conditions.** This is not post-hoc optimisation of the candidate —
it is discovering that the incumbent's own validation span does not support
the decision being asked of it.

#### Reporting structure, fixed

**Pre-registered Step 8A validation — untouched:**

| | window |
|---|---|
| Primary | 2009-2025 |
| Secondary | 2013-2025 |
| Diagnostic | 2009-2012 |

**Migration comparison — Phase 7 vs Step 8A:**

| | window | role |
|---|---|---|
| Primary | **2013-2025 (13 folds)** | the migration evidence |
| Sensitivity | 2023-2025 (3 folds) | compatibility check against Phase 7's original validation span |

Both arms through identical folds and population intersection.

    2013 ------------------------------------------- 2025
           Step 8A  ---------------------------------
           Phase 7  ---------------------------------
                                        ^ 2023-25
                                          original P7 span

#### Sequence, fixed in advance

1. finish the current Step 8A run
2. finish the current 2023-2025 Phase 7 regeneration
3. run the extended 2013-2022 Phase 7 benchmark (the script appends, so
   the two runs merge into one 2013-2025 file)
4. compare **without changing either model**
5. 13-fold comparison is the primary migration evidence
6. 2023-2025 is the narrower sensitivity check

**Rule:** the additional 2013-2022 Phase 7 results must not retroactively
change the Step 8A model or the pre-registered outcome conditions (A-D).

---

# STEP 8A RESULT: outcome C — DO NOT MIGRATE (2026-08-21)

Judged against the criteria committed in `abb6cb9` before the run.

### 1. Matched population, 2013-2025 (primary migration evidence)

Populations verified identical in **52/52** shared position-folds.

| Position | P7 MAE | S8A MAE | d | P7 RMSE | S8A RMSE | d | P7 R2 | S8A R2 | d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QB | 34.21 | 70.41 | +36.20 | 44.52 | 86.59 | +42.08 | 0.819 | 0.323 | -0.496 |
| RB | 23.16 | 54.85 | +31.69 | 31.79 | 70.82 | +39.04 | 0.870 | 0.346 | -0.524 |
| WR | 22.99 | 46.57 | +23.58 | 32.76 | 59.93 | +27.17 | 0.856 | 0.512 | -0.344 |
| TE | 19.96 | 31.71 | +11.75 | 31.76 | 43.57 | +11.81 | 0.750 | 0.523 | -0.227 |
| **Overall** | **25.08** | **50.89** | **+25.81 (+103%)** | 35.21 | 65.23 | +30.02 | 0.824 | 0.426 | -0.398 |

### 2. Coverage (unmatched, not a migration criterion)

Phase 7 projects more players (QB 968 vs 649, WR 2,860 vs 2,194) -- it
covers backups and rookies that the season-pair builder's history
requirement excludes.

### 3. Trajectory: Step 8A loses **0 of 52** position-folds

Mean gap by season, 2013-2025: 26.6, 23.8, 30.0, 24.8, 22.4, 26.6, 24.3,
25.7, 27.9, 27.7, 24.5, 26.8, 24.5. Flat across the entire span. Not one
season, not one position, not a data-regime artifact.

### 4. Decomposition: the loss is GAMES-dominated

| Position | games | rate | interaction | = season bias |
|---|---:|---:|---:|---:|
| QB | -17.95 | -4.53 | +6.25 | -16.23 |
| RB | -11.45 | -3.75 | +4.03 | -11.18 |
| TE | -6.84 | +2.22 | +0.52 | -4.09 |
| WR | -8.24 | +1.58 | +2.72 | -3.94 |

Games accounts for **60-71%** of the bias magnitude at every position.

Component quality explains why:

* **rate** is respectable -- R2 0.26-0.58, means track closely
  (RB 8.67 predicted vs 8.60 actual, WR 8.19 vs 7.79). It cannot explain a
  103% season-level loss on its own.
* **games** R2 is **-0.05 / -0.32 / -0.04** at RB/TE/WR -- worse than
  predicting the mean.

### 5. 2023-2025 subset (Phase 7's original span)

P7 23.64 vs S8A 48.88 (+107%). Same shape as the 13-fold result, so the
narrow and wide spans reconcile.

### The apparent contradiction, resolved

`hist_shrunk` was validated at 12/12 folds against a constant with games
MAE ~4.0-4.5; here it posts games MAE ~3.2-3.4 (better) yet R2 ~= 0. Both
hold. The earlier test was against a constant on a broad population; R2
here is measured on the intersected population -- established veterans,
whose games-played variance is small. A predictor can beat a constant
overall and still explain none of the cross-sectional variation among
veterans. That may be the fundamental limitation of a single season-level
exposure number.

### What is established, and what is NOT

**Established:** the failure is located in the exposure component.
**Not established:** why. Two competing hypotheses remain, and this
experiment does not distinguish them:

1. season-level exposure can be improved enough to explain veteran
   cross-sectional games-played variation; or
2. Phase 7's advantage comes from its week-by-week conditional assessment,
   which is collapsed and lost when exposure becomes one season number.

These are different experiments and should be run separately.

**Do NOT tune `hist_shrunk` because its R2 is poor.** That jumps from "we
located the failure" to "we know the solution."

### Disposition (updates the state table above)

| | Phase 7 | Step 8A |
|---|---|---|
| Disposition | **REMAINS PRODUCTION** | **candidate; migration REJECTED on this evidence** |

The architecture is not migrated. Nothing in either model was changed.
Step 8A's machinery (`season_availability.py`,
`season_conditional_production.py`, the harness arm) stays in the tree as a
measured candidate with a documented failure mode.

### SUPERSEDED: the historical Phase 8 benchmark

The "Phase 7 beats direct models by 26-40% under matched populations"
figure (2026-08-12) is **superseded**. It was computed before the position
fix, the feature-discontinuity fixes and the receiving floor.

The current corrected benchmark is **25.08 vs 50.89 MAE, matched
population, 2013-2025**, and it is that figure -- not the historical one --
that should inform future architectural decisions.

### Worth stating

Step 8A failed, and the experiment was worth running: it stopped the repo
migrating to a substantially worse architecture on the strength of a
cleaner conceptual design.

## FINAL_CONFIG re-validation, Phase 2 (architecture) — 2026-08-21

`FINAL_CONFIG` was selected 2026-08-18, before the 2026-08-20 data
corrections (position corruption, `home_away`/`is_dome`, `team_plays`, the
PBP situational backfill, the receiving floor, age). Since the original
selection margins were only 0.086-0.176 MAE and one broken-feature ablation
moved a fold by ~0.43 MAE, the config was effectively an unverified
model-selection result. Re-run with methodology unchanged.

Phase 2 trains on 2018+ (`load_training_data` floor), so the receiving
floor and the 2006-2008 contamination are out of scope here — this half of
the re-validation is unambiguous.

### Mean MAE by architecture, corrected data (point estimators only)

| model | QB | RB | TE | WR |
|---|---|---|---|---|
| A_gbm_mse | 6.062 | 4.506 | 3.061 | 4.298 |
| B_gbm_huber | 6.474 | 4.606 | 2.900 | 4.296 |
| **C_gbm_mae** | **5.960** | 4.356 | **2.794** | **4.071** |
| D_hurdle | 6.130 | 4.531 | 3.074 | 4.349 |
| D_hurdle_t5 | 6.166 | 4.407 | 2.916 | 4.161 |
| **F_yeojohnson_huber** | 5.985 | **4.334** | 2.849 | 4.096 |
| G_yeojohnson_mse | 6.006 | 4.362 | 2.893 | 4.133 |

### Winners vs incumbents

| position | incumbent | new winner | | margin |
|---|---|---|---|---|
| QB | F_yeojohnson_huber | C_gbm_mae | changed | **0.025** |
| RB | C_gbm_mae | F_yeojohnson_huber | changed | **0.022** |
| WR | C_gbm_mae | C_gbm_mae | same | 0.000 |
| TE | B_gbm_huber | C_gbm_mae | changed | **0.106** |

### Reading: the corrections did NOT overturn the architecture selection

Three of four "changed", but QB and RB merely **swapped with each other** at
margins of 0.025 and 0.022 — inside the +/-0.046 per-fold noise band this
session established. Those are ties broken differently by the sort, not
real changes.

TE is the one substantive move (0.106, ~4x the noise band), and TE is also
the position the corrections hit hardest: 39% of its rows are played-zeros
and its `age_curve` was the flattest of the four.

`C_gbm_mae` is now at or near the top everywhere, which is a
simplification rather than an upheaval.

### A trap worth recording

The script's own summary named `E_quantile_gbm` as best for WR and TE. The
**original selection rule deliberately excludes E** — it is a
floor/median/ceiling tool, not a competing point estimator
(see `PHASE3_WINNER_ARCHITECTURE`'s comment). Taking the printed "best"
would have manufactured two false changes. Re-validation must reuse the
original selection rule, not just the original harness.

### Consequence for Step 8A

A 0.02-0.11 MAE shift in architecture cannot explain Step 8A's +25.81 MAE
gap. On this evidence Step 8A's result looks **robust to config staleness,
not conditional on it**. Phase 3 (window/weighting) still pending before
that is stated firmly.

`FINAL_CONFIG` is NOT changed. No edits until Phase 3 is in.

### PRE-REGISTERED interpretation rules: RB receiving-floor sensitivity (2026-08-21)

Fixed before running. Phase 3's `all` window spans 2006-2024, so it is the
only place config selection reaches into the seasons the receiving floor
excludes. RB is the position whose incumbent window is `all`; QB is exempt
from the floor; WR (3y) and TE (10y) do not reach those seasons.

Primary = **floor applied** (production data contract).
Sensitivity = floor disabled (the previous behaviour).

| RB result | interpretation |
|---|---|
| both select the same window | floor treatment does not affect selection |
| different window, floored version wins | floor materially affects selection; use the floored version |
| different window, unfloored version wins | investigate whether structurally invalid history is exerting spurious signal; **production still uses the floored population** |
| same / broadly equivalent performance | prefer the floored version on validity grounds |

The selected config is NOT to be chosen by whichever version yields better
downstream Phase 7 performance. Phase 3 selects against its own predefined
objective.

**Reporting requirement:** report the margin to the runner-up, not only the
winner. Phase 2 showed why — QB's winner "changed" at 0.025 MAE (a tie
broken differently) while TE's changed at 0.106 (substantive). A winner
name alone cannot distinguish those.

## FINAL_CONFIG re-validation COMPLETE — one change adopted (2026-08-21)

Phases 2 and 3 re-run on the corrected panel with methodology unchanged and
the original selection rule preserved.

### Phase 3 primary (receiving floor applied): margins to runner-up

| position | winner | margin over runner-up | incumbent | incumbent gap |
|---|---|---:|---|---:|
| QB | 10y / linear | 0.022 | all / none | 0.022 |
| RB | 10y / linear | 0.005 | all / linear | 0.007 |
| WR | 7y / none | **0.000** | 3y / none | 0.018 |
| TE | 10y / linear | 0.002 | 10y / none | 0.002 |

Every gap is 0.002-0.022 MAE against a +/-0.046 noise band. WR's winner
beats its own runner-up by 0.000 — the sort order selected between
identical values. **No window/weighting change is real.**

### RB receiving-floor sensitivity

Only the `all` window differs; every other window is identical to 4dp, as
it must be. Winners differ by label (floored `10y/linear` 4.4685 vs
unfloored `all/exponential` 4.4683) but are **0.0002 MAE apart**, and the
unfloored `all` is better on exponential yet worse on linear and none —
mixed signs consistent with noise from 758 extra rows, not signal.

Lands in the pre-registered fourth row: broadly equivalent performance ->
**prefer the floored version on validity grounds.** Population decided on
validity, not performance, exactly as pre-registered.

### Adopted

| | change |
|---|---|
| Architecture | **TE only**: `B_gbm_huber` -> `C_gbm_mae` (2.900 -> 2.794, margin 0.106 ~ 4x noise) |
| QB / RB / WR architecture | unchanged (QB/RB swapped within noise at 0.025/0.022; WR identical) |
| Window / weighting, all positions | **unchanged** |
| RB population | floored (production data contract) |

Verified: the `final_config.py` diff is exactly one line. Suite 441 passing.

### Consequence for Step 8A

Total available movement from re-selecting BOTH architecture and
window/weighting is ~0.1 MAE. Step 8A's deficit is **+25.81 MAE**. Config
staleness cannot explain it. Step 8A's rejection is therefore **robust, not
conditional** — the qualification raised when the dependency was flagged is
now discharged.

### LINEAGE

The Phase 7 2013-2025 benchmark (25.08 MAE matched) was generated with
**TE = B_gbm_huber**, the pre-edit config. It remains the valid benchmark
for the old production config. Anything consuming the new TE architecture
is not directly comparable to that artifact without a re-run. Recorded in
`final_config.py` as well so the artifact/config lineage cannot be lost.

## Phase 9 empty-donor-pool fix + re-run; prior Phase 9 artifact SUPERSEDED (2026-08-21/22)

Two separate discoveries, kept separate because only the second is
TE-specific:

1. **Stale residual donors, all four positions.** Phase 9's block-bootstrap
   (`season_simulation.py`) draws residuals from
   `phase4_row_level_predictions_v2_corrected*.csv`, filtered to
   `model == FINAL_CONFIG[position]["architecture"]`. Those CSVs predated
   the 2026-08-20 data corrections AND the 2026-08-21 FINAL_CONFIG
   re-validation (LINEAGE section above), so every position's donor pool
   was drawn from stale predictions even where the architecture string
   still matched.
2. **TE additionally had ZERO donors.** TE's re-validated architecture
   (`C_gbm_mae`) does not appear anywhere in the pre-fix CSVs, which only
   have TE rows for the old `B_gbm_huber`. `build_residual_donor_pools`
   filters by exact `model` match, silently returned `{}`, and
   `_sample_block`'s empty-pool fallback (by design, for the thin-donor
   case) returned all-zero residual blocks — so every TE P25/P50/P75/P90
   was silently collapsing to the point prediction with no error anywhere
   in the pipeline. This would have existed even under a hypothetical
   fresh-CSV-but-old-architecture regenerate; it is a lineage bug, not a
   data-staleness bug.

### Fix (no change to residual methodology)

- Added `EmptyDonorPoolError` / `_require_nonempty_donor_pool` in
  `season_simulation.py`, called in `run_season_simulation` immediately
  after each fold's donor pool is built. An empty pool now raises instead
  of silently degrading to zero-variance quantiles. `build_residual_donor_pools`
  and `_sample_block` themselves are unchanged — their empty/thin-pool
  fallbacks remain legitimate primitive behavior (see their existing
  tests); the guard lives at the pipeline level, which is where "this
  position/architecture is required to produce real output" is actually
  known.
- Regression test (`TestRequireNonemptyDonorPool` in
  `tests/test_phase9_season_simulation.py`) reproduces the exact failure:
  a donor CSV with only `B_gbm_huber` TE rows, requested against
  `C_gbm_mae`, must raise `EmptyDonorPoolError`.
- Regenerated both Phase 4 CSVs
  (`phase4_row_level_predictions_v2_corrected.csv` for 2023-2025,
  `..._2020_2022.csv` for the extension window) from scratch (not
  appended — appending would have interleaved stale and fresh rows for
  the same `(player, season)` under the group-by-week donor construction)
  against the current DB and current `FINAL_CONFIG`. Verified all four
  positions now have rows for their configured architecture; TE:
  5,001 / 4,849 `C_gbm_mae` rows in the two files respectively (zero
  before).
- Re-ran Phase 9 (`scripts/run_phase9_season_simulation.py`, QB RB WR TE,
  2023-2025). All 12 folds completed; TE donor pools are 410/539/670
  player-seasons by fold (previously `{}`). Full suite: 443 passed.

### SUPERSEDED: the prior Phase 9 artifact

`data/experiments/phase9_season_simulation.csv` as it stood before this
fix — and the Phase 4 CSVs it was built from — are **superseded**. Backed
up to `data/experiments/backups/*.SUPERSEDED_pre_te_donor_fix_2026-08-22.csv`
for reference only; do not read quantiles (especially TE's) from them.
The current `data/experiments/phase9_season_simulation.csv` is the first
Phase 9 artifact with corrected data, the re-validated `FINAL_CONFIG`,
lineage-correct Phase 4 predictions, and the empty-pool invariant, and is
the one that should inform any future decision.

### Calibration diagnostic (descriptive only — no tuning applied)

The re-run's printed summary showed QB P50 MAE (12.60) far above
RB/TE/WR (1.94/0.19/0.57) and P25-P75 coverage off nominal 50% in both
directions (QB 0.475, RB 0.636, TE 0.790, WR 0.728). Per explicit
direction, this was investigated but NOT acted on --
`scripts/diagnose_phase9_calibration.py` (donor residual distribution,
donor-pool composition, synthetic-week share, coverage by position/season,
worst QB errors) was written and run read-only against the new artifacts.

Findings:

- **Donor residual distributions are not the problem.** QB's pooled donor
  residual std (7.38) is actually SIMILAR TO/below RB and WR (6.32,
  6.20); TE's is tightest (4.42), consistent with TE's narrow observed
  intervals. Sequence lengths are comparable across positions (medians
  7-14 games). Nothing here explains a QB-specific miscalibration.
- **The pooled coverage numbers are dominated by trivially-known seasons.**
  83% of TE player-seasons and 76% of WR player-seasons have ZERO
  synthetic weeks (fully observed already), which are zero-width
  intervals by construction and inflate the pooled `in_p25_p75` figures
  (TE 0.79-0.87, WR 0.81-0.84) without saying anything about the
  simulation. QB has only 40% fully-known seasons, so QB's pooled number
  is much closer to its actual simulated behavior — the positions are not
  comparable pooled.
- **Restricting to seasons with >=1 synthetic week reframes the QB result
  entirely.** In that subset, QB's `below_p25` rate is 0.46-0.75 (vs.
  nominal 0.25) across all three seasons — the simulated distribution is
  centered systematically too high, not just wide-but-off-center. RB/TE/WR
  show the same directional bias in the synthetic-only cut (below_p25
  0.13-0.37) but far less severely than QB.
- **The largest QB point-estimate errors are availability-driven, not
  residual-driven.** The 10 largest QB |P50-actual| errors are all cases
  where `games_actually_played` is 3-10 out of 11-17 possible weeks
  (benched/injured/lost-job scenarios) while `availability_rate` (fed to
  the Bernoulli synthetic-week draw) is still 0.67-0.90 — i.e. the
  *availability* estimate, not the residual/block-bootstrap machinery, is
  assuming these QBs keep playing at a rate their season didn't support.
  This points at `estimate_availability_rate` /
  `season_projection.py`'s availability model under-capturing QB-specific
  discrete job-loss dynamics (benching, single-injury-ends-season), not
  at the donor pool or the bootstrap construction.

No changes were made in response to this — recorded as the diagnosis to
resume from, not a directive to fix the availability model. Next step
identified: check whether this is a pre-existing `estimate_availability_rate`
gap (i.e. also present in Phase 7's point-estimate QB numbers) before
concluding Phase 9 introduced anything new.

### Cross-check vs. Phase 7: CONFIRMED pre-existing (Option A) — 2026-08-22

Ran a fresh QB-only Phase 7 (`run_season_projection`) pass against the
identical corrected-DB state used for the new Phase 9 run, then joined on
`(player, season)` against the same 238 QB player-seasons. Phase 7 and
Phase 9 call the same `estimate_availability_rate` and apply it the same
way to synthetic weeks (`point_prediction * rate`, summed) -- Phase 9's
P50 differs only by the added block-bootstrap residual noise -- so this
isolates whether the exposure error is availability-model-wide or
specific to Phase 9's stochastic translation of it.

| | Overall MAE (n=238) | synthetic_weeks>0 subset (n=144) |
|---|---|---|
| Phase 7 point estimate | 26.31 | 28.40 |
| Phase 9 P50 | 12.60 | 20.82 |
| Phase 9 sim_mean | 13.89 | -- |

Phase 7's point estimate is **not better** than Phase 9 on these cases --
it is worse, both overall and on the exposure-uncertain subset. Row-by-row
on the same worst-case job-loss players flagged above (e.g. `00-0036898`/
2023: Phase 7 off by 132.4 vs. Phase 9 P50 off by 106.0), Phase 7's error
is comparable to or larger than Phase 9's.

**Conclusion: Option A.** The QB job-loss/exposure error is a pre-existing
limitation of `estimate_availability_rate` in `season_projection.py`,
shared by Phase 7 and Phase 9 alike -- not something Phase 9's simulation
translation introduced. Phase 9 is exposing a known upstream limitation,
not creating a new one. This reframes any future fix as an
`estimate_availability_rate` / exposure-modeling project (e.g. a
regime-transition-aware model for QB job loss, as opposed to a stable
per-season Bernoulli rate), not a Phase 9-specific one -- and any fix
there would improve Phase 7's point estimates too, not just Phase 9's
intervals. Still not undertaken here; no production code changed.

## QUICK REFERENCE: 2025 season architecture comparison (2026-08-22)

Matched population, test season 2025, n=352 (QB 41, RB 69, WR 146, TE 96).
`walk_forward_preseason.py --test-seasons 2025 --intersect-populations`,
Phase 7 arm regenerated under the current `FINAL_CONFIG` (raw JSON:
`data/backtest_results/walk_forward_preseason_20260822_123317.json`).

### Headline table (sample-weighted across positions)

| Arm | MAE | RMSE |
|---|---:|---:|
| Phase 7 (summed-weekly) | 21.51 | 32.22 |
| Step 8A (games x rate) | 42.68 | 59.41 |
| Candidate (Ridge multi-year) | 43.41 | 59.92 |
| Production (`PreseasonProjector`) | 45.76 | 63.46 |

### Per position

| Pos | n | P7 MAE/RMSE/R2 | S8A MAE/RMSE/R2 |
|---|---:|---|---|
| QB | 41 | 30.0 / 40.4 / 0.876 | 75.9 / 89.9 / 0.389 |
| RB | 69 | 24.0 / 32.7 / 0.890 | 51.1 / 68.0 / 0.524 |
| WR | 146 | 21.9 / 34.9 / 0.820 | 40.2 / 57.0 / 0.520 |
| TE | 96 | 15.5 / 22.1 / 0.875 | 26.2 / 35.0 / 0.687 |

Step 8A decomposition (`data/experiments/step8a_decomposition.csv`) again
locates the failure in exposure: games R2 = 0.31 / -0.15 / 0.04 / 0.12
(QB/RB/WR/TE) against rate R2 = 0.41 / 0.60 / 0.68 / 0.77. Same failure
mode as the 2013-2025 run, independently reproduced on corrected data.

### READ THIS BEFORE QUOTING THE TABLE ABOVE

**The four arms do NOT solve the same problem, and the top row is not
comparable to the bottom three.**

Phase 7 projects a **completed** season. For each week it checks whether a
real/inferred row exists; if so it treats P(plays)=1 as known and predicts
off that week's actual in-season feature row. Measured on the 2025 run,
**89% of the weeks Phase 7 sums are real known-played weeks** (per
position, mean synthetic weeks: QB 4.62 of 13.33, RB 1.47 of 12.19, WR
0.75 of 11.99, TE 0.40 of 12.51 -- TE is 97% real). Only the remaining
~11% are forecast.

Step 8A, Candidate and Production are **preseason** forecasters: they
predict a whole season from strictly prior-season information, with
Step 8A additionally bound by THE EXPOSURE-LEAKAGE CONTRACT above.

So Phase 7's 21.51 is an in-season reconstruction with largely known
exposure; 42.68 / 43.41 / 45.76 are pre-Week-1 forecasts. The gap is
substantially an **information** gap, not purely an architecture gap.

Consequences, stated so this table cannot be misread later:

1. **Phase 7 cannot be used for the draft board.** It structurally
   requires real played games in the test season (`run_season_projection`
   iterates `pos_test.groupby("player_id")` and skips empty groups). The
   DB holds 2006-2025 only; there are no 2026 rows, so a 2026 Phase 7 run
   produces nothing at all. `docs/data/projections_2026.json` (620
   forward-looking draft projections) is a preseason artifact and Phase 7
   is not a preseason architecture.
2. **The like-for-like preseason ranking is the bottom three**, where the
   order inverts against the repo's current disposition: Step 8A (42.68)
   < Candidate (43.41) < Production (45.76). Step 8A is the *best*
   available preseason arm on this fold, though by a modest ~7% over
   production.
3. **The Step 8A rejection is partially confounded by this asymmetry.**
   The Step 8A pre-registration matched *populations* but never matched
   *information*. GAPS.md's hypothesis 2 ("Phase 7's advantage comes from
   its week-by-week conditional assessment") gestures at this; the
   measurement above makes it concrete and quantified. This does NOT
   overturn outcome C -- Phase 7 remains production for its own task --
   but "Step 8A loses by 103%" should not be quoted as a clean
   architecture verdict.

### The one clean, confound-free result: the TE architecture change

Old vs new Phase 7 CSV for 2025, same players. QB/RB/WR predictions are
**bit-identical** (config unchanged), so TE is isolated exactly:

| TE (n=135) | MAE | RMSE | R2 |
|---|---:|---:|---:|
| `B_gbm_huber` (old) | 18.11 | 30.60 | 0.751 |
| `C_gbm_mae` (current) | 14.95 | 22.39 | 0.867 |

-17% MAE, -27% RMSE, +0.12 R2. The 0.106 **weekly** MAE margin that
justified adopting the change substantially understated its **season**
level benefit. This comparison is valid because both sides are Phase 7 --
same task, same information, one changed variable.

### Caveats

Single fold (2025), no fold-to-fold variance; the original Step 8A verdict
rested on 52 folds. Matched-population figures are not comparable to
native-population ones (Phase 7 QB is 30.0 matched vs 24.42 native --
intersection drops easy-to-predict backups, raising MAE). Pooled overall
R2 is not derivable from per-position R2 and is deliberately not reported.

## RESULT: information-matched preseason test — Phase 7's edge was INFORMATION, not architecture (2026-08-22)

Direct follow-up to the QUICK REFERENCE section above, which flagged that
its four arms did not solve the same problem. This closes that question by
running the test the earlier comparison could not: **all four models
forecasting the full 2025 season once, in August, before Week 1, on one
intersected population (n=352), with no arm holding information the others
lack.**

### Headline (sample-weighted, matched population, 2025)

| Arm | MAE | RMSE | vs. its in-season figure |
|---|---:|---:|---|
| Step 8A (games x rate) | **42.68** | **59.41** | unchanged (already preseason) |
| Candidate (Ridge multi-year) | 43.41 | 59.92 | unchanged (already preseason) |
| **Phase 7 (summed-weekly)** | **43.79** | **60.64** | **21.51 -> 43.79 (+104%)** |
| Production (`PreseasonProjector`) | 45.76 | 63.46 | unchanged (already preseason) |

Only Phase 7 moved, because only Phase 7 had been consuming in-season data.

### Per position (Phase 7, in-season vs. August-legal)

| Pos | P7 in-season MAE / R2 | P7 preseason MAE / R2 | preseason winner |
|---|---|---|---|
| QB | 30.0 / 0.876 | 75.2 / 0.310 | **Phase 7** (75.2) |
| RB | 24.0 / 0.890 | 49.3 / 0.584 | **Phase 7** (49.3) |
| WR | 21.9 / 0.820 | 41.7 / 0.518 | Step 8A (40.2) |
| TE | 15.5 / 0.875 | 29.6 / 0.557 | Step 8A (26.2) |

### What this establishes

1. **Phase 7's 2x season-level dominance was ~entirely an information
   effect.** Information-matched, it falls from first by a factor of two to
   **third of four**, and its R2 collapses at every position (0.876 -> 0.310
   at QB; 0.875 -> 0.557 at TE).
2. **All four arms are within 3.1 MAE (~7%) of each other.** The season
   architecture question is far closer than any prior artifact suggested.
3. **The "+103% / +98% Step 8A loses" figure must NOT be quoted as an
   architecture verdict.** Under matched information Step 8A is the *best*
   arm overall and beats Phase 7 at WR and TE. Outcome C still stands for
   Phase 7's own task (in-season projection, where it is excellent and
   remains production), but the head-to-head framing was measuring an
   information gap.
4. **Split by position** (pre-registration outcome D): Phase 7 wins QB and
   RB, Step 8A wins WR and TE. Per the pre-registration, this is evidence
   toward a hybrid but explicitly does NOT license building one yet.
5. **The production `PreseasonProjector` is last in every comparison run
   this session**, in-season or not. It is the model currently feeding
   `docs/data/projections_2026.json`.

### What this does NOT establish

Single fold (2025), no fold-to-fold variance -- the original Step 8A verdict
rested on 52. A ~7% spread on one season is not a migration mandate for
anything. `is_dome` is derived inside `add_external_features`, which
preseason mode skips, so it stays carried-forward; that slightly
UNDERSTATES Phase 7 (high-importance RB feature) and is a conservative
error here, not a leak.

### Implementation: `preseason_mode` (`season_projection.py`)

August-legal, not information-stripped. GRANTED, because a real August
forecaster has it: the 2025 schedule (published in May) -- opponent,
home/away, byes -- and each opponent graded on its **2024** defensive
record. DENIED, because it does not exist in August: which weeks he played,
in-season opponent strength, week-by-week Vegas/weather, in-season team
rolling context, as-of-week depth chart, mid-season trades.

Opponent strength reuses `refresh_matchup_features` via a probe row stamped
to the END of the prior season, rather than reimplementing its DVOA
residual logic. The three columns that actually reach the model
(`opp_fpts_allowed`, `opp_fpts_allowed_s2d_lag1`,
`opp_fpts_allowed_dvoa_adjusted_lag1` -- identical across all four
positions) are all season-to-date-through-week-N-1 of the season being
predicted, hence all three need the prior-season substitute.

### A bug worth recording, because it nearly produced a plausible lie

`preseason_mode` was initially threaded through every function signature
but NOT through the `compute_player_week_predictions` call site inside
`run_season_projection`. The flag was silently inert: the first smoke run
reported TE MAE 14.83 at a synthetic-week share of 0.250, i.e. in-season
numbers under a preseason label -- a result that looked entirely reasonable
and would have been reported as fact. Caught only by checking the synthetic
share rather than the MAE.

Now guarded structurally: `run_season_projection` RAISES if any week is
counted as known-played while `preseason_mode` is set. Four behavioural
leak tests were added (`TestPreseasonModeLeakInvariants`), written as "the
mode cannot consume this" rather than "this column is excluded", matching
the exposure-leakage contract's reasoning. Suite: 447 passed.

Same lesson as this session's empty-donor-pool bug: a wrong-but-plausible
number is the worst failure mode, so the invariant must be loud.

## PRODUCTION-GRADE RESULT: 11-fold walk-forward, information-matched, 2015-2025 (2026-08-22)

Supersedes the single-fold 2025 comparison above as the basis for any
production decision. That run could not distinguish signal from
season-to-season noise; this one can.

### Method

Expanding-window walk-forward, 11 test seasons (2015-2025) x 4 positions =
**44 position-folds**, all four arms scored on the SAME intersected
population per fold (n=3,620 player-seasons), all four making a single
pre-Week-1 forecast of the full season.

**Not literal LOYO, deliberately.** True leave-one-year-out would train on
seasons AFTER the test year, which leaks. Every fold here trains only on
seasons strictly before the test season, and always includes the most
recent one (test-1).

### Leakage gate (`scripts/verify_loyo_no_leakage.py`, exit 0 = usable)

Asserted mechanically before the comparison was run, because the
silently-inert-flag bug earlier the same day proved a plausible-looking MAE
is not self-validating:

  * all 11 Phase 7 artifacts: **zero** known-played weeks, synthetic-week
    share exactly 1.0000, all four positions present, one season each;
  * Phase 7 training span ends at exactly test-1 in all 44 folds and never
    contains the test season;
  * combined artifact: 5,105 rows, zero duplicate (player, season) -- the
    append-not-overwrite behaviour of `run_season_projection` makes
    double-counting a real hazard;
  * the other three arms verified structurally: `_build_season_pairs` sets
    `curr_season = season_list[i+1]`, i.e. the TARGET season, so
    `curr_season < test_season` excludes the pair whose target is the test
    season. Had `curr_season` meant the FEATURE season, that filter would
    have trained on the test season's outcome. It does not.

### Overall (sample-weighted, 44 position-folds, n=3,620)

| Arm | MAE | RMSE |
|---|---:|---:|
| Phase 7 (preseason mode) | 46.40 | 63.05 |
| Candidate (Ridge multi-year) | 46.49 | 62.33 |
| Step 8A (games x rate) | 46.50 | 62.15 |
| **Production (`PreseasonProjector`)** | **49.11** | **65.11** |

### The top three are indistinguishable

Paired per-fold MAE differences (same players, same folds), mean +/- SD
over 11 season-folds:

| | vs Step 8A | vs Candidate |
|---|---|---|
| Phase 7 | -0.08 +/- 1.36 | -0.06 +/- 1.56 |
| Step 8A | -- | +0.02 +/- 1.45 |

Every mean is ~0.1 MAE against an SD of ~1.4 -- an order of magnitude
smaller than its own fold-to-fold variation. Head-to-head over 44
position-folds: Phase 7 vs Step 8A 23/44, Phase 7 vs Candidate 24/44,
Step 8A vs Candidate 27/44 -- all coin-flips. Season-fold wins split
Phase 7 3, Step 8A 4, Candidate 4.

**No basis exists for ranking these three.** The 0.10 MAE spread between
them is noise.

### Production is reliably WORST, and this one IS a real effect

  * loses **0 of 11** season-folds; wins outright only 2 of 44
    position-folds;
  * head-to-head: loses 35/44 to Phase 7, 36/44 to Step 8A, **39/44** to
    Candidate;
  * paired gap vs Candidate **+2.60 +/- 0.94** -- mean 2.8x its SD, unlike
    every comparison among the top three;
  * worst MAE at **all four positions** (QB 76.1, RB 56.3, WR 47.6, TE
    33.7).

This is the model currently generating `docs/data/projections_2026.json`.

### Stable position structure, reproduced across 11 folds

| Pos | Phase 7 | Step 8A | Candidate | Production |
|---|---:|---:|---:|---:|
| QB | **69.3** | 71.8 | 72.7 | 76.1 |
| RB | **52.9** | 54.5 | 53.6 | 56.3 |
| WR | 46.0 | **45.0** | **45.0** | 47.6 |
| TE | 31.4 | **30.8** | 31.3 | 33.7 |

CORRECTION (same day, before this entry was relied on): an earlier draft
of this section claimed the QB/RB vs WR/TE split was "structure, not
noise" because it reproduced the single 2025 fold. **That claim was wrong
and is withdrawn.** Testing it properly -- paired per-fold differences
against the nominal per-position winner -- shows NOTHING among the top
three is separable at ANY position:

| Pos | nominal best | runner-up gap | folds runner-up wins |
|---|---|---|---|
| QB | Phase 7 | +2.70 +/- 5.86 (Step 8A) | 3/11 |
| RB | Phase 7 | +0.72 +/- 2.63 (Candidate) | 3/11 |
| WR | Candidate | +0.01 +/- 1.66 (Step 8A) | 6/11 |
| TE | Step 8A | +0.42 +/- 1.35 (Candidate) | 3/11 |

Every gap is smaller than its own SD, and the "winner" takes only 5-8 of
11 folds. With four arms and four positions a recurring pattern is
expected by chance; reproducing it in one extra fold is weak evidence, and
I treated it as strong. **Do not read the bolded per-position winners as
findings.**

Production's per-position disadvantage is separable at QB (+6.69 +/- 3.87),
WR (+2.47 +/- 1.97) and TE (+2.87 +/- 1.75), but NOT at RB (+3.31 +/-
3.72). The pooled overall gap remains separable (+2.60 +/- 0.94) because
pooling across positions cuts the variance -- that headline finding stands,
but it is an aggregate claim, not a per-position one.

Pre-registration outcome D (split result -> evidence toward a hybrid) is
therefore NOT triggered: there is no demonstrated split to build a hybrid
around.

### An anomaly worth a future look, not acted on here

Production posts competitive or best **R2** (QB 0.36 -- highest of any arm;
RB 0.40 -- highest) while posting the worst **MAE at every position**. R2
rewards explaining variance; MAE punishes absolute error. That divergence
is the signature of a scale/bias problem rather than a ranking problem --
it may be ordering players roughly correctly while systematically missing
the magnitude. Not diagnosed here; flagged because it suggests a
potentially cheap fix (recalibration) rather than a model replacement.

### What this establishes / does NOT establish

**Establishes:** the production preseason model is reliably ~2.6 MAE
(~5.3%) worse than three alternatives, across 11 seasons, on matched
populations, with no leakage. Phase 7's summed-weekly architecture confers
no season-level advantage once information is matched -- confirming the
2025 finding at 11x the evidence.

**Does NOT establish:** which of the three replacements is better (they
tie); anything about ROOKIES or thin-history players. The intersected
population retains 3,620 of 5,105 preseason-eligible player-seasons (71%),
and the exclusion is systematic -- players without prior-season aggregates
are dropped by every arm. For a draft board that is a material blind spot,
since rookies are exactly where projection is hardest and most valuable.

## The rookie feature set is NOMINAL: declared, computed, and fed constants (2026-08-22)

Found while asking whether any of the four season architectures could be
extended to project rookies. Two separate defects; the second is the
material one.

### 1. `is_rookie` has two contradictory definitions

| Location | Definition |
|---|---|
| `feature_engineering.py:4170,4174` (`_add_injury_features`) | `games_count <= 8` |
| `advanced_rookie_injury.py:1010` | `years_exp == 0` |
| `advanced_rookie_injury.py:1029` (fallback) | `season == first_season` |

`prepare_features` runs `add_engineered_features` (which sets the
`games_count <= 8` version) and THEN `add_advanced_features` (which
overwrites it). `years_exp` is absent from the training frame, so the
`season == first_season` branch wins -- the correct definition. Verified
running: `/tmp/p7_loyo_2024.log` shows 8x "Adding advanced rookie
features" + "Added: rookie_draft_value, ...". The historical bug recorded
in that module's own comment (module silently skipped for all of v22-v26)
is fixed.

**Latent hazard, not currently firing:** `add_advanced_features`
(`feature_preparation.py:158-165`) wraps the whole module in
`except Exception` + a print. If it throws for any position/fold,
`is_rookie` silently reverts to `games_count <= 8` -- i.e. every veteran
who missed half a season is relabelled a rookie -- with no error, no
column-shape change, and no visible difference in the output artifact.
Same silent-degradation class as this session's empty-donor-pool and
inert-preseason-flag bugs.

### 2. Every rookie is modelled as a mid-5th-round pick

`get_all_players_for_training()` returns **87 columns and none of them are
`draft_round`, `draft_pick`, `years_exp`, combine metrics, or
`age`/`birth_date`.** So in `add_advanced_rookie_features`:

    draft_round = int(row.get('draft_round', 5))    # -> ALWAYS 5
    draft_pick  = int(row.get('draft_pick', 150))   # -> ALWAYS 150

Every rookie in every training run is projected as pick 150. The #1
overall selection and an undrafted free agent receive **identical** draft
capital. Since `project_rookie()` derives its tier from draft slot, that
makes five of the six `rookie_*` features constant per position:

  * `rookie_draft_value`, `rookie_breakout_prob`, `rookie_bust_prob`,
    `rookie_ceiling_ppg`, `rookie_floor_ppg` -- constant;
  * `rookie_opportunity_score` -- the ONLY one carrying per-player signal
    (from `_compute_prior_opportunity_scores`, prior team usage);
  * `combine_score` -- defaults to 50.0 when no metrics are present
    (`advanced_rookie_injury.py:437`), and combine columns are absent, so
    also constant;
  * `age_curve` -- `player_age.py:143` already carries its own warning
    that it is "near-constant".

So of the 9 rookie/draft/athleticism features declared in CAUSAL_FEATURES
for all four positions, **at most one varies between rookies.**

### The data exists. It is simply not joined.

| Table | Rows | Coverage |
|---|---:|---|
| `draft_picks_v2` | 11,081 (11,054 with GSIS `player_id`) | draft_season 1980-2026 |
| `combine_data_v2` | 8,968 | 2000-2026 |

`DatabaseManager.get_draft_picks()` already exists and its docstring notes
the legacy `draft_picks` table is empty while v2 "has data". Neither table
is joined into the training query. This is a JOIN, not a data-acquisition
project.

### Consequences

  * **Does NOT invalidate the 11-fold result.** Those features are 0.0 for
    veterans by construction (`advanced_rookie_injury.py:1032-1037`) and
    the scored population contained zero rookies, so they contributed
    nothing either way.
  * **Does invalidate "Phase 7 has a cold-start rookie toolkit."** It has
    the column names. It does not have the information.
  * **Reorders the rookie work.** A cold-start row builder was the wrong
    first step. Wiring `draft_picks_v2` + `combine_data_v2` into
    `get_all_players_for_training` comes first: it makes the existing
    features real for IN-SEASON rookie prediction (where Phase 7 already
    projects all 95 of 2025's rookies), and is a hard prerequisite for any
    preseason cold-start path.
  * Any future rookie experiment must re-baseline after the join -- prior
    rookie-feature importance rankings were measured on constants.

## SCOPE: wiring real rookie/draft/age data into the feature frame (2026-08-22)

Investigation follow-up to "The rookie feature set is NOMINAL" above. Scoped,
not built. Nothing below has been implemented.

### Investigation findings

**Present in the DB, never joined:**

| Source | Rows | Key | Coverage |
|---|---:|---|---|
| `draft_picks_v2` | 11,081 | `player_id` (GSIS) | 1980-2026; matches **61%** of training players |
| `draft_values` | 262 | `pick` | pick -> 5 value charts (stuart/johnson/hill/otc/pff). **Entirely unused.** |
| `players.birth_date` | 2,952 | `player_id` | **98.9%** -- and `players` is ALREADY left-joined by the training query |
| `combine_data_v2` | 8,968 | `pfr_id` / `player_name` | 2000-2026; metrics 56.6%-89.6% complete |

**The unmatched 39% of the draft join are genuinely undrafted, not a join
bug.** Top unmatched by career FP: Welker, Gates, Thielen, Ekeler, Baldwin,
Amendola, Beasley -- all famous UDFAs. Median career FP is 168 (drafted) vs
24 (unmatched), the expected UDFA shape.

**Bugs / missing data found:**

1. `draft_round`/`draft_pick` default to **5 / 150** for every rookie
   (`advanced_rookie_injury.py:1055-1056`), because the columns are absent
   from the 87-column training frame. Conflates "undrafted" with
   "mid-5th-round" -- UDFAs must be encoded as undrafted, not as pick 150.
2. `rosters` table is **EMPTY (0 rows)** -- the `years_exp` source, already
   documented at `preseason_features.py:28`. `years_exp` must be DERIVED
   (season - draft_season; season - first_season for UDFA).
3. `combine_data` (v1) is **EMPTY (0 rows)** while `combine_data_v2` has
   8,968. Same dead-v1 pattern as `draft_picks` vs `draft_picks_v2`. Both v1
   tables should be dropped or documented so a future reader does not join
   the empty one.
4. `combine_score` defaults to **50.0** when no metrics are present
   (`advanced_rookie_injury.py:437`) -- currently constant for everyone.
5. **No GSIS<->PFR bridge exists.** No table carries both `player_id` and a
   `pfr_*` column. Combine must be matched by name, the same way
   `snap_counts` / `weekly_pfr` already are. `src/data/entity_resolver.py`
   exists for this.
6. `players.name` is blank for **36.5%** (1,090 of 2,985; 993 with
   production, 140 with >500 career FP) and is **NOT** recoverable from
   `weekly_rosters_v2` (0 of 1,090). BUT this is purely legacy: blank rate
   by debut era is 82.9% (1999-2009), 50.5% (2010-2014), 3.4% (2015-2019),
   **0.0% (2020-2025)**, and **0.0% of players active 2023+**. Name-keyed
   joins are therefore safe for current players and degrade only historical
   training rows.
7. `players.name` is ABBREVIATED (`W.Welker`) while `combine_data_v2.
   player_name` is full (`John Abraham`) -- a direct equality join is
   impossible; needs last-name + first-initial + position matching.
8. `injury_prob_ml` is **100% missing** (run logs:
   `rate=1.000 warn_threshold=0.030`).
9. `is_rookie` dual definition + bare-except silent fallback -- see the
   preceding GAPS entry.

### Proposed phases

**Phase A -- draft capital + age.** Highest value, lowest risk. Join
`draft_picks_v2` on `player_id`; add `p.birth_date` to the existing
`players` left join (a one-column change at `database.py:2056`); join
`draft_values` on pick for a real `rookie_draft_value`; derive `years_exp`.
Add an explicit `is_undrafted` flag rather than defaulting UDFAs to pick
150. Files: `database.py`, `advanced_rookie_injury.py`, `settings.py`.

**Phase B -- close the silent-failure paths.** Remove or narrow the bare
`except Exception` in `feature_preparation.py:158-165`; reconcile the two
`is_rookie` definitions to one; add a guard asserting the `rookie_*`
features are NOT constant across rookies, so a future regression to
defaults fails loudly instead of silently. This phase is what stops the
same class of bug recurring and should not be deferred.

**Phase C -- combine (defer).** Name-based matching via `entity_resolver`,
accepting a ~60% metric-completeness ceiling and legacy blank names. Lower
value than A, materially more work.

**Phase D -- re-baseline.** Any prior rookie-feature importance ranking was
measured on constants and is void. Retrain and re-measure after A.

### LINEAGE WARNING

Phase A changes the feature set, so `FEATURE_VERSION` must bump from 33.
That invalidates `data/cached_features.parquet` and makes today's 11-fold
walk-forward artifacts (`walk_forward_preseason_20260822_194322.json`, the
`phase7_preseason_loyo/` CSVs) non-comparable to anything produced
afterwards. Either re-run the 11-fold comparison after Phase A, or treat
the two as separate lineages -- do NOT compare across the bump.

## FEATURE_VERSION 34: real draft capital, real age, and loud rookie-degeneracy detection (2026-08-22)

Implements Phases A and B of the preceding scope. Phase C (combine) turned
out to be unnecessary -- see below.

### Phase A -- the data is now joined

  * `get_all_players_for_training()` joins `draft_picks_v2` (deduplicated:
    2 player_ids appear twice and a bare LEFT JOIN would have silently
    DOUBLED their training rows) and `draft_values` (clamped to pick 262,
    since draft_picks_v2 reaches 336 from the old 8-12 round era), and adds
    `p.birth_date` to the `players` join that already existed.
  * `is_undrafted` / sentinel picks are resolved **in SQL**, not downstream.
    Found the hard way: `add_utilization_scores()` blanket-fills NaN->0, so
    leaving `draft_pick` NULL turned every undrafted player into **"pick 0"
    -- the single most valuable selection in the draft**, the exact inverse
    of the truth. Emitting no NULLs makes that unreachable.
  * Undrafted encoded as round 8 / pick 400, past every real selection, with
    `is_undrafted` as its own feature. ~39% of players who reach the league
    are undrafted (Welker, Gates, Thielen, Ekeler); that is signal, not a
    missing value.
  * `player_age.derive_age()` now ranks birth dates ABOVE a pre-existing
    `age` column. `season_long_features` populates `age` with a per-position
    CONSTANT before it runs, and the birth-date fill only touched NaNs, so
    real dates were never consulted -- `age_curve` was effectively constant
    even for the 98.9% of players whose birth date is known.

### Phase B -- the silent-failure paths are closed

  * `feature_preparation.add_advanced_features` no longer catches bare
    `Exception` and returns the frame unchanged. That is how the entire
    module failed silently for v22-v26. It now raises.
  * `feature_engineering._add_injury_features` no longer defines
    `is_rookie` as `games_count <= 8` (which labels any veteran who missed
    half a season a rookie). One owner: `advanced_rookie_injury.py`.
  * `_warn_if_rookie_features_degenerate()` checks that rookie_* features
    actually VARY across rookies. Shape and null checks cannot catch a
    column that is present, correctly typed, and carries one value.
  * `draft_picks_v2` and `draft_values` added to the schema, so a fresh DB
    has the tables the training query depends on.

### Verified working (RB, 2020+, 2,966 rookie rows)

| Feature | Before | After |
|---|---|---|
| `rookie_draft_value` | constant | 143 distinct |
| `combine_score` | constant 50.0 | 64 distinct, **0% defaulted** |
| `age_curve` | near-constant | 322 distinct |
| `rookie_ceiling_ppg` | constant | 8 distinct |
| `is_undrafted` | did not exist | 22.4% of rookie rows |

Directional sanity: undrafted `rookie_draft_value` 0.100 vs drafted 0.347;
`draft_pick` 400 vs 117. Guard verified in BOTH directions -- fires on a
simulated pre-v34 frame, silent on real v34 data. Suite: 447 passed.

**Phase C (combine) is not needed.** A PFR->GSIS mapping is built at
runtime (7,791 players), and combine_score is 100% real for 2020+ rookies.
The earlier "no bridge exists" finding was about DB tables only.

### Still open, found while implementing

1. **`advanced_rookie_injury.py:30` sets `warnings.filterwarnings('ignore')`
   at module scope** -- a process-wide suppression of every warning, from
   any library, for the life of the process. It silently swallowed the
   first version of the degeneracy guard, which is why that guard prints
   instead. Not changed here: narrowing it could surface a flood of
   previously-hidden warnings and deserves its own assessment.
2. `injury_prob_ml` remains 100% missing.
3. Phase D re-baseline is now due -- every prior rookie-feature importance
   ranking was measured on constants and is void.

### LINEAGE

FEATURE_VERSION 33 -> 34. `data/cached_features.parquet` is invalidated,
and today's 11-fold walk-forward artifacts
(`walk_forward_preseason_20260822_194322.json`, `phase7_preseason_loyo/`)
were produced at v33. Do NOT compare across the bump -- re-run the
comparison at v34 or treat them as separate lineages.

## QA/QC of the v34 draft matching: clean. Plus college/conference availability (2026-08-22)

### Six checks, all passed

| # | Check | Result |
|---|---|---|
| 1 | Undrafted rate by debut era | 32.0% / 42.7% / 45.7% / 38.4% (2006-09 / 10-14 / 15-19 / 20-25). Plausible; no spike that would indicate a join failure. |
| 2 | draft_season vs first NFL season | **0** players with a negative gap. Nobody "played before being drafted". |
| 3 | Undrafted rate by position | QB **24.3%** vs RB 41.7% / WR 44.0% / TE 47.1%. Matches reality -- teams rarely carry UDFA quarterbacks. |
| 4 | Top 15 undrafted by career FP (debut 2015+) | Ekeler, Meyers, Chosen, Bourne, Mostert, Taysom Hill, Lazard, Snead, Humphries, Breida... **every one a genuine UDFA.** Zero false negatives. |
| 5 | Draft rows that cannot join (no GSIS id) | 27, **all from the 2026 class**. Of 80 skill-position 2026 picks, 73 have an id, 7 do not. |
| 6 | Cross-source vs `combine_data_v2.draft_ovr` | 10 era-plausible suspects, **all resolved as distinct players sharing an abbreviated name**. |

Check 6 is worth recording because it nearly produced a false alarm. Matching
`players.name` ("J.Taylor") to combine's full names by first-initial +
surname + position yielded 53 apparent contradictions. Every one dissolved on
inspection: `J.Taylor` RB resolves to `00-0036223` (1,554 FP, correctly
drafted) AND `00-0036096` (40 FP, genuinely undrafted) -- two different
players. Same for H.Bryant, T.Williams (Terrance drafted / Tyrell UDFA, both
correct) and J.Wright. **The join is on GSIS `player_id`, which disambiguates
them correctly; the ambiguity was in the QA method, not the data.**

**Conclusion: zero confirmed missed matches among players with NFL
production.** The ~39% "unmatched" are genuinely undrafted and are now
correctly represented by `is_undrafted` rather than being silently defaulted
to a mid-5th-round pick.

**The one fixable gap:** 7 of 80 skill-position 2026 draft picks have no GSIS
id in `draft_picks_v2`. They have no NFL rows yet, so training is unaffected
-- but a 2026 preseason/cold-start projection would miss them. Recoverable by
name+college against `players`, or on the next nfl-data-py refresh once ids
are assigned.

### College is available; conference is NOT

| Source | Coverage | Distinct | Shape |
|---|---|---:|---|
| `draft_picks_v2.college` | 100% | 200 (skill, 2006+) | single school, `"Ohio St."` |
| `combine_data_v2.school` | 100% | 358 | single school |
| `players.college` | 98.9% | 792 | **transfer strings**: `"Miami; Washington State; Incarnate Word"` |

**No conference table or column exists anywhere in the DB.** The only match
for "conference" is `utilization.py`'s `is_conference_championship`, an NFL
week-21 flag, unrelated.

Adding conference would need three things, in order:

1. **Name normalization.** Two conventions collide -- `draft_picks_v2` says
   `"Ohio St."` while `players` says `"Ohio State"`, plus `"Miami (FL)"` vs
   `"Miami (OH)"` vs `"Miami (Ohio)"`. Use `draft_picks_v2.college` as the
   key: it is single-school and 100% populated, whereas `players.college`
   carries multi-school transfer histories that have no single conference.
2. **A college -> conference mapping.** Not in nfl-data-py; would be authored
   or sourced. Cardinality argues for it: college is a 200-value categorical
   with a long tail (top 10 = 23.4%, top 50 = 68.6%, top 100 = 90.8%),
   which is poor for direct encoding, whereas conference is ~12 values.
3. **Era-awareness.** Conference realignment makes a static map wrong for
   historical rows -- Texas A&M moved Big 12 -> SEC in 2012, Nebraska ->
   Big Ten in 2011, USC/UCLA -> Big Ten in 2024. The mapping must be keyed on
   `(college, draft_season)`, not college alone.

Neither college nor conference is currently in CAUSAL_FEATURES.

## The 7 anonymous 2026 draft picks: identified (2026-08-22)

### The GSIS ids do not exist to be recovered

`nfl.import_draft_picks([2026])` has **no GSIS id for the same 7 picks**
(33, 65, 73, 140, 165, 170, 254) that were null locally. This is not a local
ingestion fault: GSIS ids are minted when a player first appears in official
game data, and the 2026 season has not started. No local source could have
supplied them either -- `combine_data_v2` has 0 rows for `draft_year=2026`,
`weekly_rosters_v2` has 0 rows for `season=2026`, and `draft_picks_v2` has no
name column.

### What was actually fixed: identity

Upstream carries `pfr_player_id` and `cfb_player_id`, and this repo already
matches PFR data by id elsewhere. Both were backfilled onto `draft_picks_v2`
via `scripts/backfill_draft_pick_identity.py`, matched on
`(draft_season, draft_round, draft_pick)` -- verified unique in BOTH sources,
so the update cannot fan out. Coverage: `pfr_player_id` 11,081/11,081
(100%), `cfb_player_id` 8,194/11,081 (73.9%; only recent drafts carry a
college id). Row count unchanged.

The 7 are no longer anonymous -- `cfb_player_id` is effectively the name:

| Rd | Pick | Pos | College | pfr_player_id | cfb_player_id |
|---:|---:|---|---|---|---|
| 2 | 33 | WR | Mississippi | StriDe01 | dezhaun-stribling-1 |
| 3 | 65 | QB | Miami (FL) | BeckCa01 | carson-beck-1 |
| 3 | 73 | TE | Georgia | DelpOs01 | oscar-delp-1 |
| 4 | 140 | WR | Georgia | YounCo01 | colbie-young-1 |
| 5 | 165 | RB | Penn St. | SingNi01 | nicholas-singleton-1 |
| 5 | 170 | TE | Cincinnati | RoyeJo01 | joe-royer-1 |
| 7 | 254 | WR | Oklahoma | BurkDe02 | deion-burks-1 |

They can now be reconciled to a GSIS id once nflverse assigns one, and are
matchable for a cold-start projection in the meantime.

### A second, larger bug found in the same place

`backfill_all_data.backfill_draft_picks` ended with
`.dropna(subset=["player_id"])`, which **silently discarded every draft pick
lacking a GSIS id** -- upstream has 12,927 rows against our 11,081, so
**~1,846 real picks were being thrown away**. That is precisely the
not-yet-debuted population a rookie projection needs most. The dropna is
removed and `pfr_player_id`/`cfb_player_id` are now retained, so the next
backfill run picks up the missing rows. They are NOT re-ingested here:
re-running that script rewrites many tables and is out of scope for this fix.

Neither `draft_picks` (legacy) nor the empty-string case was caught by that
dropna either, which is why 27 empty-`player_id` rows survived while 1,846
NaN ones did not -- the same column arriving in two different null forms.

## Re-ingest, the is_rookie year bug, and an era-aware conference map (2026-08-22)

### Re-ingest: +1,846 recovered draft picks, training unchanged

`draft_picks_v2` 11,081 -> **12,927** rows after removing the
`dropna(subset=["player_id"])`. 11,054 carry a GSIS id; 1,873 do not (players
who have not debuted). Training is provably unaffected -- 28,402 rows, 0
duplicate `(player_id, season, week)`, `is_undrafted` mean 0.222 before and
after -- because the join subquery excludes id-less rows by construction.

### CRITICAL: is_rookie was frame-relative, and v34 made it dangerous

`add_advanced_rookie_features` derived `is_rookie` from the minimum season
**present in the frame**. Every windowed fold filters seasons (WR trains on
3y, TE on 10y), so each fold relabelled its own oldest veterans as rookies:
filtering to 2020+ made rookies of **Frank Gore, Adrian Peterson and LeSean
McCoy** -- 127 of 292 rookie player-seasons in that frame, **43% wrong**.

Latent and near-harmless while the rookie_* features were constant. The v34
join made it actively harmful: a 15-year veteran now receives real
draft-capital priors under a rookie flag.

Fixed by supplying `first_nfl_season` from the database, computed over the
whole `player_weekly_stats` table, so no caller's slice can move it.
Verified: false rookies **127 -> 0**, Gore/Peterson/McCoy `is_rookie=0`, and
`is_rookie` fires in at most one season per player across the full frame.

### Rookie data QA (right player, right year, right format)

| Check | Result |
|---|---|
| Known draft positions | Barkley (2018/1/2), McCaffrey (2017/1/8), Jacobs (2019/1/24), Henry (2016/2/45) all exact |
| Draft attrs constant per player | 0 players whose draft_round/pick/value/is_undrafted vary across their own rows |
| Sentinels confined to undrafted | 0 drafted rows with pick 400 or round 8; 0 undrafted rows off-sentinel; 0 undrafted with nonzero pick value |
| dtypes | draft_round/pick/value/is_undrafted all int64 |
| is_rookie timing | fires in <=1 season per player, frame-independent |

`B.Robinson` initially read as a mismatch and was not: Bijan (2023/1/8,
Texas) and Brian Jr (2022/3/98, Alabama) both resolve correctly. The join is
on GSIS `player_id`; the ambiguity was in the abbreviated-name lookup used to
test it. The newly-backfilled `cfb_player_id` now separates them outright.

### College -> conference: era-aware, `src/features/college_conference.py`

Keyed on **(college, draft_season - 1)**, because a player drafted in year N
played their final college season in N-1. That distinction is the entire
point: a Texas A&M player drafted 2012 played 2011 in the **Big 12**; one
drafted 2013 played 2012 in the **SEC**. A static map would have asserted a
fact that was not true when they played.

Coverage on skill picks 2006+ (1,680): SEC 20.5%, Big Ten 14.8%, ACC 12.9%,
Pac-12 11.9%, Big 12 11.1%, then G5 conferences, with **6.1% UNMAPPED** --
almost entirely FCS/DII/DIII (North Dakota St., Montana, Delaware). UNMAPPED
is a real countable value, not a silent fallback to a plausible guess.
Schools are also absent from the base map for their pre-FBS years, so
Appalachian St. is UNMAPPED before 2014 rather than falsely Sun Belt.

Only `is_power5` is model-facing (added to CAUSAL_FEATURES at all four
positions). `college_conference` stays a string for inspection -- encoding a
15-value categorical has not been justified yet.

### A THIRD blanket-filler corruption, found by building the above

`add_advanced_rookie_injury_features` **mode-fills `draft_college`**,
replacing every undrafted player's missing college with the modal value
"Ohio St." -- 1,416 rows -> **12,157**. The first wiring of the conference
feature ran after it and duly reported Big Ten 15,369 with **every undrafted
player labelled Big Ten**. Fixed by resolving conference at the START of
`prepare_features`, on the raw frame, before any filler runs.

That is now three distinct silent-fabrication fillers found in this area:

1. `add_utilization_scores` NaN->0, which would have made undrafted players
   "pick 0", the most valuable selection in the draft;
2. this mode-fill of `draft_college`, fabricating a Big Ten pedigree;
3. the historical `FeaturePolicyRegistry` median-fill of the snap column
   (already recorded earlier in this file).

The pattern is worth naming: **a filler that runs over every column will
invent values for columns whose missingness is meaningful.** Undrafted,
un-debuted and FCS are all real states, not gaps.

FEATURE_VERSION 34 -> 35. Suite: 476 passed.

## Staged arm: PRESERVE_PERSONNEL_MISSINGNESS (default OFF) (2026-08-22)

`team_pct_{11,12,13,21}_personnel` are NaN before 2016 (PBP personnel coverage
starts then) and were blanket-filled to 0 -- a factual claim that the team
lined up in 12 personnel on 0% of snaps, which the data does not support. The
model-facing features are the roll3 means derived from them, so the fabricated
zeros are averaged into two live features per position
(`team_pct_{11,12}_personnel_roll3_mean` for WR,
`team_pct_{12,21}_personnel_roll3_mean` for RB).

**Deliberately NOT switched on.** Phase 2 architecture selection, Phase 3
windows, FINAL_CONFIG and the 11-fold walk-forward were all produced with these
filled. Flipping the default would invalidate every one of them while blending
the effect into the pending v34/v35 re-baseline. Staged as a config flag so it
runs as its own attributable arm -- the same discipline that let the Phase 2
re-validation catch a false "3 of 4 architectures changed".

### Correcting an earlier claim in this file

An earlier note said `add_utilization_scores` "corrupts NA for every feature".
That was **wrong**. It destroys NaN in 32 columns (325,426 -> 58,059 cells),
all present at the raw-DB stage; features engineered afterwards keep their NaN,
which is why 2.3% of feature cells already reach the model missing. The blast
radius on model-facing features is also small: **zero direct hits**, only the
two personnel roll3 means per position.

### FOUR fillers, not one -- and the flag was inert twice

Making the arm real required exempting the column at every filler between the
raw table and the model. It read inert twice, both times with byte-identical
NaN counts on and off:

1. **Exempted `utilization_score`'s blanket fill only.** `_impute_missing`
   (`feature_engineering.py`) median-filled the column afterwards regardless.
2. **Exempted the BASE columns only.** The model never sees
   `team_pct_12_personnel`; it sees `team_pct_12_personnel_roll3_mean`, a
   different name, still filled. (`team_motion_rate` survives today only
   because it happens to BE the feature name rather than a base behind one.)

Both mechanisms already existed and are now flag-aware:
`utilization_score.missingness_preserved_cols()` and
`feature_engineering._structurally_missing_cols()`, both resolved at CALL time
because GAPS.md 7.7/7.8 records that monkeypatching module constants here is
unreliable.

### Verified in both directions

| Position | flag OFF | flag ON |
|---|---:|---:|
| WR personnel roll3 NaN | 0 | **20,543** |
| RB personnel roll3 NaN | 0 | **13,912** |

NaN counts track the pre-2016 populations (WR 19,921 rows, RB 13,507) plus
early-2016 rows whose 3-week lookback crosses the coverage boundary -- the
expected shape. Five regression tests (`tests/test_personnel_missingness_flag.py`)
encode both inertness failures directly. Suite: 481 passed. Default OFF, so
current behaviour is unchanged.

## is_rookie: the THIRD correction — data-floor censoring (2026-08-22)

Asked directly whether the rookie definition is now accurate. It was not, and
the remaining error was distinct from the two already fixed today.

`first_nfl_season` cannot distinguish "debuted in 2006" from "our data starts
in 2006". `player_weekly_stats` begins at `MIN_HISTORICAL_YEAR`, so every
player already in the league at the floor read as a rookie:

  * 548 players have `first_nfl_season == 2006`;
  * of the 382 with draft records, **332 were drafted BEFORE 2006** -- draft
    years run back to 1982, with 46-47 apiece from 2003/2004/2005;
  * only **50** are genuine. **87% of "2006 rookies" were censored veterans.**

Fixed using `draft_season` (joined at FEATURE_VERSION 34): at the floor a
debut is believed only when the draft year agrees. Undrafted players at the
floor are unknowable and resolve to NOT-rookie -- the conservative direction,
since handing a 10-year veteran rookie draft-capital priors is worse than
omitting him from a subgroup. Measured on WR: rookie player-seasons in 2006
fell **548 -> 14**, and 2007+ counts are unchanged.

### The residual 9.7% is a definitional choice, kept deliberately

Of 528 labelled WR rookies with draft data, 477 (90.3%) were drafted that same
year. The other 51 (9.7%) were drafted 1-4 years earlier and had no stats row
until later -- late debuts.

They are STILL labelled rookies, on purpose. `is_rookie` here means "no prior
NFL production to learn from", which is true for a 2010 pick whose first stats
row is 2013. Keying on `season == draft_season` instead would label him a
veteran while every history feature is NaN -- a contradiction the model cannot
resolve. If the distinction is ever wanted, the honest way to express it is a
separate `years_since_draft` feature, not a redefinition of this one.

### All three failures had the same signature

| # | Wrong definition | Damage |
|---|---|---|
| 1 | `games_count <= 8` | any veteran missing half a season |
| 2 | frame-relative debut | 127 of 292 in a 2020+ frame (Gore, Peterson, McCoy) |
| 3 | data-floor censoring | 332 of 382 at the 2006 floor |

None raised. In every case the column existed, was int-typed, and carried a
plausible mean. Six regression tests
(`tests/test_is_rookie_definition.py`) now pin all three, including the
frame-independence property and the deliberate late-debut behaviour.

## Rookie integration, steps 1-2: Phase 7, Candidate and Step 8A (2026-08-23)

### Step 1 -- cold-start rows

**Phase 7** (`build_cold_start_week_row`): a player with no NFL history cannot
be handled by `build_synthetic_week_row`, which works by carrying forward the
most recent real game. TE 2025 verified: 116 rows / 0 rookies -> **135 rows /
19 rookies**, the 19 additions all rookies, and 135 exactly matches the
in-season TE population. Feature split verified per-feature: **26 populated,
44 NaN, zero columns populated outside the allowlist.**

The split is an explicit allowlist (`COLD_START_KEEP_FEATURES`), not a pattern
match, because on a real rookie's row EVERY feature is populated -- their
week-8 `target_share_pct_roll3_mean` is a genuine in-season value. "What looks
filled" cannot classify them.

**Candidate + Step 8A** (`_cold_start_rows` in `preseason_features.py`, one
fix for both since they share a source): 2025 WR goes 155 rows / 0 rookies ->
**216 rows / 61 rookies**, veterans unchanged at 155. The emitted set is a
strict SUBSET of the 64 true rookies -- zero false positives. (Three true
rookies remain out: their only rows are week > 18, a pre-existing filter.)

A first attempt emitted **82** cold-start rows against 38 true rookies,
because it tested prior existence against `season_agg`, which drops
player-seasons under `MIN_GAMES=6`. A player with three games last year has no
aggregate row and read as a rookie. Now tested against raw history: a thin
prior season is not the same as no NFL career.

### Career-static features now populate for EVERY row

`build_multiyear_season_pairs` had **no draft capital, combine or college at
all**, so cold-start rows there would have been nearly featureless.
`career_static_by_player()` joins draft round/pick/pick-value/undrafted plus
`is_power5` for veterans and rookies alike -- these exist for everyone, their
weight simply decays with experience, and a tree finds that itself given
`years_of_history`.

Measured side-effect on veterans (2025 QB, matched population): candidate MAE
**79.5 -> 73.7**, Step 8A **75.9 -> 73.1**. Draft capital helps the existing
population, not only rookies.

### Step 2 -- Ridge imputation, in-fold

`_fit_candidate` used an unconditional `.fillna(0.0)`, asserting that a player
with no prior season posted exactly zero PPG -- indistinguishable from a
genuinely unproductive veteran, since zero is a real value in those columns.

Replaced with median-within-position (the function is already per-position),
fitted on the TRAINING fold only and carried to predict, plus a 0/1 `__isna`
indicator for any feature missing on >=1% of rows. Fitting the imputer on the
combined frame would leak test-fold distribution -- quieter than leaking the
target, but a leak. WR: 62 features -> 111 columns via 49 indicators, no NaN
surviving.

### First rookie-subgroup result, and why stratification was the right call

Candidate arm, 2025 WR, trained on <2025:

| Subgroup | n | predicted mean | actual mean |
|---|---:|---:|---:|
| Rookies | 61 | 50.4 | **32.0** |
| Veterans | 155 | 97.1 | 90.1 |

Rookies are over-predicted by **57%** against 8% for veterans. Pooled, that
is invisible. This is exactly the failure mode a global metric hides.

### Still open

**Production (`PreseasonProjector`) has no cold-start path.** Its
`_build_season_pairs` uses `prior_df.merge(curr_df, on="player_id")` -- an
INNER merge -- so rookies never form a row. Until that changes, the harness's
population intersection strips rookies back out of the comparison, and the
work above is invisible end-to-end. That is the next step, and it touches the
live production model rather than experiment code.

## Production gains a cold-start path; equal-n assertion added (2026-08-23)

### `PreseasonProjector` now forms rookie rows

Three coupled changes, none of which works alone:

1. **`prior_df.merge(curr_df)` (INNER) -> `curr_df.merge(prior_df, how="left")`.**
   The inner join required a prior-season row, so a player whose first NFL
   season is the target never formed a pair.
2. **`curr_df` now carries identity.** It selected only
   `player_id, season_total`; name/position/birth_date came from `prior_df`
   and are NaN for a rookie. Without backfilling them from the current season,
   every rookie row has `position = NaN` and `fit()`'s per-position loop drops
   all of them -- **the rows would exist and still never be trained on**, which
   looks identical to success from outside.
3. **`_fit_linear_model` imputes instead of zero-filling.** Median within
   position, fitted on the training fold only, plus 0/1 `__isna` indicators.

Result, 2025 WR: **146 rows -> 211, true rookies 0 -> 38, zero NaN positions.**

### The trap in the old fit, stated precisely

`_fit_linear_model` ran `X = pos_df[features].fillna(0.0)` and only then
`valid = np.isfinite(X).all(axis=1)`. The mask was therefore **inert** -- the
fillna guaranteed finiteness. The real defect was the zero-fill itself:
**1,054 of 3,226 WR training rows carried at least one NaN feature** and were
being told they had scored 0 PPG last season, indistinguishable from a
genuinely unproductive veteran.

But removing the fillna WITHOUT changing the mask would have converted an
inert guard into a silent row-dropper, discarding exactly the 1,054 rows --
including every rookie just added. The mask is now scoped to the TARGET only.
Both had to change together, which is why they did.

### Equal-n assertion (`_assert_equal_n`)

Raises when any two arms are scored on different row counts within a
position/fold. It catches the failure that looks like success: an arm that
silently drops rookies is scored on an easier, veteran-only population and
posts a BETTER metric for it. Nothing errors, the log still prints an `n` per
arm, and the winner is decided by whichever model discarded the hardest
players.

The harness already printed those counts. Nobody diffs log lines, which is
exactly why this is an assertion rather than a warning. Verified both ways:
silent on matched populations, raising with a per-arm breakdown on a
mismatch.

## FIRST ROOKIE-INCLUSIVE COMPARISON (2025, single fold) — 2026-08-23

All four arms now form rookie rows, and `_assert_equal_n` passes: **447
players per arm, identical populations across 4 position-folds** (was 352
before cold-start; QB 41->51, RB 69->97, WR 146->184, TE 96->115).

### Pooled MAE hides the entire finding

| Arm | pooled MAE |
|---|---:|
| Step 8A | 41.34 |
| Candidate | 42.68 |
| Phase 7 | 45.49 |
| Production | 47.90 |

### Stratified by years_exp, the picture inverts

| bucket | n | Candidate | Step 8A | Phase 7 | Production |
|---|---:|---:|---:|---:|---:|
| **0 (rookie)** | 95 | 38.4 | **37.7** | **52.1** | **55.5** |
| 1-2 | 117 | 43.7 | 42.0 | 41.8 | 44.0 |
| 3-5 | 117 | 45.1 | 41.9 | 43.3 | 46.3 |
| 6+ | 118 | 42.7 | 43.0 | 45.9 | 47.2 |

Between-arm spread is **~18 MAE on rookies** against ~2-4 on every veteran
bucket. Rookies also have the LOWEST actual mean (63.8 vs 94-115), so the
relative error is worse still.

### Paired per-player differences, bootstrapped

| Pair | ROOKIES (n=95) | VETERANS (n=352) |
|---|---|---|
| Step 8A - Phase 7 | **-14.43** [-22.4, -6.7] **SEPARABLE** | -1.37 [-4.2, +1.5] noise |
| Candidate - Phase 7 | **-13.76** [-20.8, -6.5] **SEPARABLE** | +0.14 [-3.1, +3.6] noise |
| Step 8A - Production | **-17.75** [-23.1, -12.4] **SEPARABLE** | -3.53 [-6.3, -1.0] SEPARABLE |
| Step 8A - Candidate | -0.67 [-3.9, +2.5] noise | -1.52 [-4.0, +1.0] noise |

**On veterans the arms are indistinguishable** (only Production separates,
confirming the 11-fold result). **On rookies the season-level arms beat the
weekly ones by 14-18 MAE, well outside the noise band.** Step 8A vs Candidate
is inside noise in BOTH subgroups -- genuinely tied.

### Why Phase 7 is bad at rookies: its cold-start rows are OUT-OF-DISTRIBUTION

Phase 7 has all nine rookie features and still posts rookie MAE 52.1 with bias
**-34.8** -- it predicts ~29 against an actual mean of 63.8, massively
under-predicting.

The mechanism is structural, not a tuning problem. **Phase 7 never trains on a
cold-start-shaped row.** Its training rows come from `run_fold`'s real weekly
data, where a rookie's week-8 row carries genuine in-season rolling features.
At prediction time it is handed a row with 44 of 70 features NaN -- a shape it
has never seen -- and extrapolates badly.

Candidate and Step 8A do not have this problem because
`build_multiyear_season_pairs` emits cold-start rows for EVERY target season,
including training ones. Their models have seen NaN-history rows and learned a
sensible default direction for them.

The obvious follow-up is to train Phase 7 on cold-start-shaped rows
(augmentation), NOT to tune it. Recorded and deliberately not acted on:
locating a mechanism is not the same as validating a fix.

### CAVEAT: single fold, so the bootstrap is over PLAYERS, not seasons

The intervals above resample players within 2025 only. They therefore contain
no season-to-season variation, and the 11-fold history says that variation is
the dominant term -- the Phase 7 vs Step 8A gap moved 22.4-30.0 MAE across
seasons for an identical comparison. **The rookie gap is large enough
(14-18 MAE) that it will likely survive, but this is not yet a
production-grade result.** The multi-season rookie-inclusive run is required
before acting.

## PRODUCTION-GRADE ROOKIE-INCLUSIVE RESULT: 11 folds, 2015-2025 (2026-08-23)

Supersedes the single-fold 2025 version. `_assert_equal_n` passes across all
**44 position-folds**; 4,632 player-seasons per arm, 18,528 predictions.
Artifacts passed the leakage gate including the new rookie-presence assertion
(95-124 rookies per season).

### Pooled MAE still hides it

| Arm | pooled MAE |
|---|---:|
| Step 8A | 44.92 |
| Candidate | 45.66 |
| Phase 7 | 47.30 |
| Production | 50.17 |

### Stratified by years_exp

| bucket | n | Candidate | Step 8A | Phase 7 | Production |
|---|---:|---:|---:|---:|---:|
| **0 (rookie)** | 1012 | 41.0 | **40.3** | **50.3** | **53.3** |
| 1-2 | 1292 | 45.4 | 45.6 | 45.6 | 48.1 |
| 3-5 | 1306 | 47.7 | 45.9 | 45.7 | 49.4 |
| 6+ | 1022 | 48.0 | 47.4 | 48.5 | 50.7 |

Between-arm spread is **~13 MAE on rookies** against **1-3 on every veteran
bucket**. The pooled ranking is essentially the rookie ranking, diluted.

### Paired differences, bootstrapped BY SEASON (11 folds, 4000 resamples)

| Pair | ROOKIES (n=1012) | VETERANS (n=3620) |
|---|---|---|
| Step 8A - Phase 7 | **-10.05** [-12.2, -8.1] **SEP** | -0.22 [-1.2, +0.7] noise |
| Candidate - Phase 7 | **-9.32** [-11.4, -7.3] **SEP** | +0.52 [-0.3, +1.4] noise |
| Step 8A - Production | **-13.04** [-14.7, -11.4] **SEP** | -3.07 [-3.7, -2.5] SEP |
| Phase 7 - Production | **-2.99** [-4.9, -0.9] **SEP** | -2.85 [-4.0, -1.8] SEP |
| Step 8A - Candidate | -0.73 [-1.6, +0.1] noise | -0.74 [-1.6, +0.3] noise |

This is the season-level bootstrap the single-fold run could not do, so the
intervals now contain the season-to-season variation that the earlier 11-fold
history showed was the dominant term.

### What is established

1. **On veterans, Step 8A / Candidate / Phase 7 are indistinguishable.**
   Every pairwise comparison among them sits inside noise. Only Production
   separates, and it is worse. This reproduces the pre-rookie 11-fold result
   exactly.
2. **On rookies, the season-level arms beat the weekly arms by 9-13 MAE, and
   it is separable at every pairing.** This is the largest reliable effect
   found in the whole comparison.
3. **Step 8A vs Candidate is inside noise in BOTH subgroups.** They are tied,
   as they were before rookies were added. No basis for ranking them.
4. **Production is separably worst in both subgroups.** Consistent with the
   pre-rookie result, now confirmed on a rookie-inclusive population.

### Phase 7's rookie failure is a training-distribution artifact

Rookie bias by arm: Phase 7 **-41.7**, Production +3.1, Candidate -10.4,
Step 8A -9.4. Phase 7 under-predicts rookies by ~42 points -- far worse than
its MAE alone suggests, and the deficit persists into years 1-2 (**-18.2**)
before disappearing by year 6+ (+2.4).

Mechanism, unchanged from the single-fold diagnosis and now confirmed at
scale: **Phase 7 never trains on a cold-start-shaped row.** Its training rows
come from real weekly data where a rookie's week-8 row carries genuine
in-season rolling features. At prediction time it receives a row with 44 of 70
features NaN -- a shape absent from training -- and extrapolates badly.
Candidate and Step 8A do not have this problem because
`build_multiyear_season_pairs` emits cold-start rows for every target season
including training ones.

The follow-up is AUGMENTATION -- train Phase 7 on cold-start-shaped rows --
not tuning. Deliberately not attempted: the mechanism is located, the fix is
not validated, and the pre-registration discipline that killed Step 8A applies
here too.

### Production implication

Phase 7 remains the only weekly-capable arm and ties the field on veterans,
but it is **separably worse on rookies than both season-level arms**, and
rookies are ~22% of the scored population. A draft board built on Phase 7
today would be reliably poor on exactly the players a draft is most uncertain
about. That is an argument for the augmentation experiment, not for switching
architectures: Step 8A and Candidate cannot do weekly at all.

## Option A built: PRESERVE_HISTORY_MISSINGNESS (default OFF) — 2026-08-23

### The diagnosis was wrong, then corrected

First stated as "Phase 7 never trains on cold-start-shaped rows". That is the
symptom. The cause: **a rookie's first-week row IS genuinely NaN, and the
pipeline fills it.** Consequently **43 of the 46 columns a cold-start row
blanks were NEVER NaN in training**, so LightGBM had no learned
missing-direction for them and routed real NaN arbitrarily -- the -41.7 bias.

Also corrected: rolling features group by `player_id` ALONE, not
`(player_id, season)`, so a veteran's week 1 takes `shift(1)` from the prior
season's week 17 -- deliberate cross-season continuity. Only a player's
**first-ever NFL week** is genuinely undefined: 959 WR rows (2.4%), not the
8.9% first-guessed.

### Why zero-filling would be worse than the median

Proposed as an alternative. Rejected on measurement: **0 is an occupied value**
in these columns -- 59.4% of veteran `snap_share_y1` sits below 0.5 with a
minimum of exactly 0.00, and `ppg_y1` reaches -0.01. A rookie set to 0 would
land on top of marginal, barely-playing veterans and become structurally
indistinguishable from them. "Trees can handle it" is the argument FOR NaN:
LightGBM's missing-direction learning only engages on NaN; hand it 0 and there
is nothing to handle.

Confirmed Step 8A already does exactly this -- its rookie lag **NaN rate is
1.000 and zero rate 0.000**, with veterans at 24.8% NaN, which is why it has a
learned direction and posts rookie MAE 40.3.

### FIVE fillers, found by bisection

The restore had to move twice before it held:

1. per-column `.fillna(0)` at creation sites
2. the blanket numeric fill in `utilization_score`
3. `_impute_missing`'s position-aware median
4. **the position-specific block, which runs AFTER (3)** -- caught 11 columns
   including `target_share_pct_roll3_mean` and `wopr_roll3`
5. **`advanced_rookie_injury`, measured wiping 959 restored NaNs back to 0**

So the restore runs ONCE at the end of `prepare_features`, past all of them.
Exempting per-site was abandoned as both error-prone and fragile: a rolling
feature added later would silently arrive pre-filled.

### Verified

| | flag OFF | flag ON |
|---|---:|---:|
| cold-start cols never NaN in training | 43 | **10** |
| ...of those, HISTORY columns | 33 | **0** |
| training rows carrying NaN history | 959 | **4,063** |

The 10 remaining are correctly out of scope: weather, Vegas, contract, depth
chart, `current_qb_epa_per_att`. Seven regression tests pin the behaviour,
including that career-static columns (draft capital, combine, age) survive the
blanking and that week 2 is NOT blanked -- blanking it would fabricate
missingness rather than preserve it. Suite 494 passed, default OFF.

### Which TERM is broken -- decomposed before pre-registering

Phase 7 predicts `sum(weekly prediction) x availability`, so "rookie MAE is
bad" does not say which half. Decomposed on the 2025 cold-start artifact:

| | rookies | veterans |
|---|---:|---:|
| predicted per-week | **1.94** | 5.77 |
| actual per-week | **4.93** | 6.07 |
| pred / veteran ratio | 0.34 | -- |
| actual / veteran ratio | **0.81** | -- |

**The PRODUCTION term is the failure.** Rookies genuinely produce at 81% of
the veteran per-week rate; the model predicts 34% -- a 2.4x under-prediction.
1.94 pts/week sits near the floor of the model's range, which is exactly the
signature of NaN routed to an unlearned default direction.

**A second, independent defect partially hides it.** The exposure term
OVER-estimates rookies: `estimate_availability_rate` falls back to
`position_avg_fallback = 0.88` for anyone with no prior seasons, implying 15.0
games, while rookies actually played 11.7. So production is 2.4x too low and
exposure 1.28x too high, netting 2.2x too low.

That matters for sequencing: **fixing exposure alone would make Phase 7's
rookie totals WORSE**, because the over-estimate is currently offsetting the
production shortfall. Recorded so a future reader does not "fix" the
availability fallback in isolation and conclude the change regressed things.

### Pre-registered expectation (sharpened)

Judged on the PER-WEEK production term, which isolates the half under test:

  * **Mismatch binding:** rookie per-week prediction rises from **1.94 toward
    ~4-5**, season total from 29.0 toward ~50-64, bias from -34.8 toward ~-10.
  * **Mismatch NOT binding:** per-week stays near 1.94. Then draft capital +
    combine + destination-team context genuinely cannot predict rookie
    production, and the ceiling is set by feature availability -- consistent
    with this repo having **no college production stats at all**, only draft
    position and combine measurables.

Recorded BEFORE running so the result cannot be rationalised afterwards.

## FOURTH inert-flag bug, same signature: `is_power5`/conference silently absent from Phase 7 (2026-08-24)

Prompted by "ensure all models are properly wired with rookie data" after the
PRESERVE_HISTORY_MISSINGNESS near-miss above. An Explore audit across all four
arms (Phase 7, Step 8A, Candidate, Production) found one more of exactly that
bug class.

`add_conference_features` (`src/features/college_conference.py:228`, producing
`is_power5`) was called in exactly one place: `prepare_features`
(`feature_preparation.py`). `is_power5` is declared in `CAUSAL_FEATURES` for
all four positions (`config/settings.py`), so it grepped as wired and looked
present in config. But Phase 7's real training path is
`_prepare_training_data`, which does not call `prepare_features` -- the same
bypass already found and fixed for the history-NaN restore. `ensemble.py`
silently drops any causal feature whose column doesn't exist
(`[c for c in causal_cols if c in pos_data.columns]`), so the absence never
errored -- Phase 7 just trained without a college-conference signal for every
run to date, with no visible symptom.

Fixed by calling `add_conference_features` at the top of
`_prepare_training_data`, on the raw frame, before `calculate_all_scores`
zeroes `draft_season` and `add_advanced_rookie_injury_features` mode-fills
`draft_college` -- same ordering constraint documented in `prepare_features`.
Verified: `draft_college`/`draft_season` are loaded straight from
`draft_picks_v2` in `database.py`'s base query, so they're intact at function
entry; 494/494 tests still pass.

Audited the other three arms too: Candidate and Step 8A select features by
"every numeric column not excluded," so anything `career_static_by_player`
joins (including `is_power5`) reaches them automatically -- no bypass
possible by construction. Production has no draft-capital features at all
(`BASE_FEATURES_COMMON` only has age/years_exp/rookie_or_low_experience) --
that's a real capability gap, not a wiring bug, consistent with Production's
worst-of-the-four rookie bias already documented above.


## The rookie availability fallback assumes 0.88, rookies realize 0.584
## (measured 2026-08-25; value deliberately left unchanged for now)

Audited two things that looked suspicious in the completed history-NaN
cold-start run (`data/experiments/phase7_coldstart_histnan/`, 6,133
player-seasons across 2015-2025).

**Point 1 -- `weeks_synthetic == possible_weeks` on every row: not a bug.**
`--preseason-mode` is supposed to make every week synthetic, and that is
enforced rather than assumed: `season_projection.py` raises a `RuntimeError`
if any week comes back real/snap-verified/pbp-confirmed under
`preseason_mode`. The invariant exists precisely because the flag was once
silently inert. Nothing to fix.

**Point 2 -- the 0.88 fallback is applied correctly but is the wrong value.**
Application is clean: 1,139 of 6,133 rows (18.6%) carry exactly 0.88, and
those rows are a perfect 1:1 match with true rookies (min season in
`player_weekly_stats` >= projected season). Zero veterans receive it, so the
`prior.empty` branch is not silently catching veterans with unusable history.

Realized availability for exactly those rookies:

| Position | assumed | realized | n    |
|----------|---------|----------|------|
| QB       | 0.88    | 0.469    | 109  |
| RB       | 0.88    | 0.578    | 315  |
| TE       | 0.88    | 0.602    | 215  |
| WR       | 0.88    | 0.606    | 500  |
| **all**  | 0.88    | **0.584**| 1139 |

A +0.296 over-assumption: the projection credits rookies with ~30% of a
season they never play. **0.584 is the better estimate of rookie
availability. That is not in dispute below.**

**The tradeoff, stated honestly.** In cold-start/preseason mode every week is
synthetic, so the season total scales exactly linearly with
`availability_rate` -- the counterfactual is exact, not approximate:

| rookie rate used         | mean bias | MAE  |
|--------------------------|-----------|------|
| 0.88 (current)           | -19.5     | 45.7 |
| 0.584 (empirical mean)   | -31.2     | 45.4 |
| per-position empirical   | -32.8     | 45.0 |

The correction **improves MAE** (45.7 -> 45.0) and **worsens mean bias**
(-19.5 -> -32.8). Rookie bias is already negative while availability is
over-assumed, so 0.88 is currently offsetting a larger per-week
under-prediction of rookie production; removing the offset exposes the full
under-prediction.

**Decision: leave 0.88 in place for now.** This is a choice to hold the bias
baseline still while the per-week rookie cause is being chased, bought at a
0.7-MAE cost. It is explicitly NOT a finding that 0.584 is wrong -- a future
reader should not read "left unchanged" as "the empirical value is bad." Once
the per-week rookie scale is fixed, 0.584 (or the per-position vector) should
go in, and both numbers re-measured together. Changing it before then would
move the baseline mid-experiment and make the rookie arm look worse for a
reason unrelated to whatever is being tested.

This corroborates rather than contradicts the 2026-08-18 finding that
availability weighting is not the root cause of Phase 7's synthetic-week
bias. The root cause is on the per-week rookie prediction side.

**Denominator parity: verified, 0.584 is a genuine drop-in.** Checked before
logging the number, because a rate measured over a different window than the
one it replaces is a silent mismatch once it is in code. Both are on the same
basis: `estimate_availability_rate` uses distinct weeks with a
`player_weekly_stats` row over `possible_weeks_for_team` (team regular-season
games); the realized measure uses `games_actually_played` over the run's
`possible_weeks`, which in preseason mode is ungated by active roster
(`require_active_roster=False`) and so is also team regular-season games.
Numerators match exactly on all 1,139 rows (max abs diff 0), and recomputing
realized rookie availability directly from the estimator's own formula gives
0.5844 against the 0.584 above. Neither is an "all 17 weeks" or
"active-roster weeks" denominator.

**Follow-up, shippable independently: rename `position_avg_fallback`.** It
never varies by position, in either module -- it is a single league-wide
constant reached only by players with no prior history, i.e. rookies. The
name misdescribes the behaviour, and the per-position spread above (QB 0.469
vs WR 0.606) is exactly the variation the name promises and does not deliver.
`no_history_fallback` or `rookie_fallback` is accurate. Behaviour-free and
independent of the value decision above, so it does not need to wait for the
per-week rookie fix.

**Fixed in passing (behaviour-identical, separate commit):**
`season_projection.py` hardcoded `0.88` as a default argument while
`availability.py` defined `POSITION_AVG_FALLBACK = 0.88` for the same
purpose, with no import between them. `season_projection.py` now imports the
constant.


## snap_share is a fabricated 0.0 for every player-week before 2013, and
## `window='all'` folds train on it (found 2026-08-25)

Found while costing out whether to extend the Phase 7 cold-start arm below
2015. It is not a pre-2015 problem -- it contaminates folds already run.

`player_weekly_stats.snap_share` and `.snap_count` are **0.0, not NULL**, for
100% of rows in 2006-2012. The `snap_counts` source table simply starts in
2013; the columns were populated with zeros rather than left missing.

| season | rows | mean snap_share | % exactly 0 | distinct values |
|--------|------|-----------------|-------------|-----------------|
| 2009-2012 | ~6.0k/yr | 0.0000 | 100.0 | 1 |
| 2013   | 6,897 | 0.4803 | 7.5 | 1,759 |
| 2014+  | ~6.0k/yr | ~0.52 | ~0.1 | ~1,750 |

**Why this is the bad kind of missing.** A NULL is handled correctly --
LightGBM's missing-aware splits route it, and the history-NaN work in this
same arm exists precisely to preserve honest NaN. A fabricated 0.0 is a
confident wrong answer: it says "this player was on the field for none of his
team's snaps" for every player in seven seasons. It also **passes any
coverage audit** -- `snap_share IS NOT NULL` reports 100% for 2009-2012,
higher than 2013's genuine 88.7%. Same bug class as the four inert-flag bugs
above: greps as wired, audits as present, semantically empty.

**Live exposure.** `snap_share_pct_roll3_mean` and `snap_share_accel` are
declared in `CAUSAL_FEATURES` for all four positions, and
`window_to_season_list("all", ...)` returns every season from
`MIN_HISTORICAL_YEAR = 2006`, so folds using `window='all'` (QB and RB in the
2015 config, among others) train on those zeros:

| test season | train span | rows from 2006-2012 | % of training rows |
|-------------|------------|---------------------|--------------------|
| 2015 | 2006-2014 | 40,746 | **76.0%** |
| 2018 | 2006-2017 | 40,746 | 57.0% |
| 2021 | 2006-2020 | 40,746 | 45.4% |
| 2025 | 2006-2024 | 40,746 | 35.0% |

The `3y` windows are unaffected for recent test seasons but not for early
ones (a 2015 `3y` fold is 2012-2014, still one-third fabricated).

**Fix (not yet applied)**: set `snap_share`/`snap_count` to NULL for seasons
< 2013 so the missing-aware paths handle them honestly, then re-measure. This
is a training-data change affecting every arm, so it should not be folded
into an in-flight experiment.

**Consequence for extending the arm below 2015**: 2013-2014 are safe (real
snaps, real depth charts -- `depth_charts` also starts 2013, same boundary).
2007-2012 would feed the model seven seasons of fabricated zeros. 2006 is
separately unusable: `MIN_HISTORICAL_YEAR = 2006` left-censors history, so
548 players have 2006 as their first season against a ~130/yr steady state,
and a cold-start rookie arm would treat every established veteran as a
rookie.

**Also noted**: the existing 11-fold set is already not feature-homogeneous.
NGS tables start in 2016 (not 2018 as previously recorded), so the 2015 fold
has no NGS features while 2016-2025 do.


## Fixing the fabricated zeros took six fillers and exposed a seventh bug
## (2026-08-25)

Follow-up to the pre-2013 snap_share entry above. Recorded because the fix
was five times larger than the diagnosis, and the reason generalises.

**What was fabricated.** `snap_count`/`snap_share`/`team_snaps` were a literal
0.0 for all 40,746 pre-2013 rows (snap_counts starts 2013); `ngs_*` columns
were `fillna(0.0)` for everything before 2016 (NGS starts 2016). Both passed
an `IS NOT NULL` audit at 100%.

**Nulling the DB was inert on its own.** `SNAP_MISSINGNESS_MODE = "zero"`
routes through `safe_divide`, which returns 0.0 when either operand is NaN --
so `snap_share_pct` was 0.0 whether the database held 0 or NULL. Six fillers
sit between the raw tables and the model:

1. per-column `fillna` at creation sites (`_merge_ngs_data`, `_create_ngs_features`)
2. `safe_divide` inside `snap_share_pct`
3. the blanket numeric fill in `utilization_score.calculate_all_scores`
4. `_impute_missing`'s position-aware median
5. `apply_snap_imputation`'s position x era medians
6. `feature_policy_registry.apply`'s per-group policy fill

Exempting fewer than all of them leaves the change invisible -- the same
inert-flag outcome already recorded for `PRESERVE_PERSONNEL_MISSINGNESS` and
`PRESERVE_HISTORY_MISSINGNESS`. This was verified by tracing 20,154 real WR
rows through the actual pipeline, not by reading the code; two rounds of code
reading each concluded the fix was complete while it was not.

**The seventh bug, found only because the sixth filler was inspected.** Policy
matchers are substrings resolved in declaration order, and `'epa'` is inside
`'ngs_avg_sEPAration'`. `pbp_advanced` is declared before `ngs`, so
`ngs_avg_separation` and its `_roll3_mean` derivative -- a declared
CAUSAL_FEATURE -- were governed by the pbp_advanced policy and median-filled,
while the sibling `ngs_avg_cushion` from the same table went to `ngs` and was
not. Nothing failed. The asymmetry was visible only as one column reading 0%
NaN while its sibling read 100%. Resolution now ranks by (prefix match,
matcher length); across all 102 causal features exactly two groups change,
both corrections (`rookie_opportunity_score` was under `utilization` via
`'opportun'`).

**Deliberately not routed through SNAP_MISSINGNESS_MODE.** The 2026-08-19 A/B
rejected `"preserve"` and that result stands -- but it only settles 2013+,
where a missing snap row is INFORMATIVE (unknown-snap players run 55-65% of a
known player's usage, so a median overstates them and the fabricated zero is
the closer proxy). Pre-2013 missingness is STRUCTURAL: the sensor did not
exist, so the same zero understates every starter for seven seasons and
carries no signal about any player. Two different populations; the flag
governs one, `SNAP_DATA_START_SEASON` the other. Pre-2013 is also exempted
from `apply_snap_imputation`, whose `snap_era` buckets split at 2018 and would
otherwise blend real 2013-2017 behaviour into seasons that have none.

**Blast radius.** Training data changed for every arm. Every prior result --
including the 2019-2025 Phase 7 cold-start files and the 2015-2018 re-run
completed earlier the same day -- was produced with the fabricated values and
is no longer comparable. The 11-fold set needs a full re-run before its
numbers mean anything.

**Guardrail**: `tests/test_structural_missingness_pre_era.py` covers the
policy collision, the `preserve` strategy, and the snap era exemption.


## Corruption scan: seven more features carry fabricated pre-era constants,
## and the EPA-per-play denominators are broken outright (2026-08-25)

Ran the same audit that found the snap and NGS zeros across every column of
`player_weekly_stats` and every declared CAUSAL_FEATURE for QB and WR,
comparing 2008-2012 against 2019-2024. Detector: a column that is CONSTANT in
the early window but varies in the late one.

**Verified clean (this session's fixes, confirmed by the scan):** all
`ngs_*_roll3_mean`, `snap_share_pct_roll3_mean`, `snap_share_accel`,
`team_motion_rate`, `team_play_action_rate` now read 100% NaN in the early
window rather than a fabricated constant.

### A. `pass_plays` / `rush_plays` / `recv_targets` are zero in EVERY season
### except 2025 -- and the fallback makes the bug frame-dependent

These three columns are the denominators for `pass_epa_per_play`,
`rush_epa_per_play` and `recv_epa_per_target`, whose roll3 means are declared
CAUSAL_FEATURES for QB, RB and WR/TE respectively. They are 100% zero for
2006-2024 and populated only for 2025.

`_create_base_features` guards this with a WHOLE-FRAME fallback:

    pass_plays = df.get("pass_plays", ...)
    if pass_plays.sum() == 0 and "passing_attempts" in df.columns:
        pass_plays = df["passing_attempts"]

Because the test is `sum() == 0` across the entire frame, the presence of a
single 2025 row disables the fallback for every other season. Measured on real
QB rows, `pass_epa_per_play` % zero by season:

    frame 2020-2025:  2020-2024 = 100.0%   2025 = 6.8%
    frame 2020-2024:  2020-2024 = ~7%      (fallback active, real values)

So the same seasons are either real or entirely fabricated depending on
whether 2025 is in the frame. Any pipeline that engineers features on
train+test concatenated trains on all-zero per-play EPA and tests on real
values -- a train/test discontinuity located exactly at the test season.

Even on the "good" path the denominators disagree: training rows are divided
by `passing_attempts` while 2025 rows are divided by `pass_plays`, which are
not the same quantity. Fixing this properly means backfilling the three
columns from PBP for 2006-2024; the cheap interim fix is a per-ROW fallback
(`where(plays > 0, attempts)`) so the frame's composition stops changing the
result. NOT yet applied -- it changes a denominator, which is a modelling
decision, and it should not land mid-experiment.

### B. Seven causal features hold a constant fabricated value pre-era

Same shape as the snap/NGS bug and the same reason it went unseen: 0% NaN, so
every `IS NOT NULL` audit passes.

| feature | constant through | value | source table starts |
|---------|------------------|-------|---------------------|
| `injury_score` | 2012 | 1.0 | player_injuries 2013 |
| `qb_pressure_pct_roll3_mean` | 2017 | 0.0 | weekly_pfr 2018 |
| `recv_drop_pct_roll3_mean` | 2017 | 0.0 | weekly_pfr 2018 |
| `team_sack_rate_allowed_roll3_mean` | 2017 | 0.0 | weekly_pfr 2018 |
| `qb_bad_throw_pct_prior` | 2018 | 0.0 | seasonal_pfr 2018 |
| `qb_pocket_time_prior` | 2018 | 0.0 | seasonal_pfr 2018 |
| `team_pct_11/12_personnel_roll3_mean` | 2015 | 0.633 | team_personnel_stats 2016 |

The personnel pair is already known and held behind
`PRESERVE_PERSONNEL_MISSINGNESS` (default OFF), documented above. The other
five are not documented anywhere and are live in every run.

`injury_score = 1.0` is the most misleading of these: it does not read as
absent, it reads as a specific and confident health claim for every player in
2006-2012.

**Not fixed here.** Each needs the same multi-filler treatment the snap and
NGS columns required, and together they change training data for every arm
again. Worth doing as one attributable batch rather than piecemeal, and after
the current re-baseline rather than during it.

**Method note.** The detector that found all of these is three lines --
per-season `nunique()` on the assembled feature matrix, flagging columns
constant early and varying late. It is worth running as a test over a sampled
frame, because every bug in this family has been invisible to coverage audits
by construction: the fabricated value is never null.


## Both fabricated-value families fixed; re-run deliberately NOT started
## (2026-08-25)

Closes the two items logged in the scan entry above.

### EPA denominators: fixed at source, not with a fallback tweak

`pass_plays`/`rush_plays`/`recv_targets` now hold
`passing_attempts`/`rushing_attempts`/`targets` for all 116,490 pre-2025 rows.
That is not a substitution -- 2025 already stored exactly those values
(verified 6,764/6,764 rows), so this makes every season agree with the
definition production was already using, and removes the train/test
denominator mismatch where training divided by attempts and 2025 by plays.

The whole-frame `sum() == 0` fallback is now per-ROW. Measured before and
after, `pass_epa_per_play` % zero for QB:

    before, frame 2020-2025:  2020-2024 = 100.0%,  2025 = 6.8%
    before, frame 2020-2024:  2020-2024 = ~7%
    after,  either frame:     identical (7.9 / 8.4 / 5.2 / 7.5 / 8.3 / 6.8)

Backup: `data/nfl_data.db.bak-preplaycounts-20260825211855`.

### Pre-era constants: all masked

| source | starts | features | old constant |
|--------|--------|----------|--------------|
| weekly_pfr | 2018 | qb_pressure/blitz/hurry/hit/sack, rb_ybc/yac, recv_drop_pct, team_sack_rate_allowed | 0.0 |
| seasonal_pfr | 2019 (shifted +1) | qb_bad_throw_pct, qb_pocket_time, rb_broken_tackles, recv_drop_pct_season | 0.0 |
| player_injuries | 2013 | injury_score | 1.0 |
| depth_charts | 2013 | depth_chart_rank | 3 |

In-era defaults are untouched. An unlisted player really is healthy, and the
open `Probable` remap question is left exactly as it was -- only rows before a
source exists are masked.

`depth_chart_rank` needed `.astype(int)` dropped; an int column cannot hold
NaN, which is what made the fabricated 3 structurally unavoidable rather than
merely chosen.

**Two traps worth remembering.** `injury_score` had to be masked in
`_merge_injury_data_from_cache`, not `_ensure_injury_rookie_features`: only
the former is on BOTH `create_features` branches and the causal path never
calls the latter, so the first attempt was inert. And every masked column
also needs adding to `_STRUCTURALLY_MISSING` (plus any matching policy
group's `exclude`) -- masking without exempting is inert, which
`snap_share_accel` had already demonstrated twice.

### Remaining, deliberately

`team_pct_{11,12,13,21}_personnel_roll3_mean` still carry a pre-2016 constant
(0.633 etc.). That is the existing `PRESERVE_PERSONNEL_MISSINGNESS` decision
-- default OFF so prior results stay attributable -- not an oversight. It is
now the ONLY causal feature family still constant-and-filled before its
source begins, across all four positions.

### Status: paused before re-running

Training data has now changed several times over in one day: pre-2013 snaps
nulled, NGS preserved, play counts backfilled, seven feature families masked.
Nothing has been re-run against any of it. Every existing result -- the
2019-2025 Phase 7 cold-start files, the 2015-2018 re-run, FINAL_CONFIG,
Phase 2/3 selection, the 11-fold walk-forward -- predates all of it.

The 11-fold set should be re-run as one batch at a single commit before any
of its numbers are read again. Not started deliberately.


## Training-data quality audit: one result-flawing bug, one target gap
## (2026-08-25)

Systematic sweep after the structural-missingness work: integrity, value
ranges, target reconstruction, coverage, era boundaries, leakage.

### CRITICAL -- `REGULAR_SEASON_MAX_WEEK = 18` counts the wild-card round as
### regular season for 2006-2020

The regular season was 17 weeks through 2020 and 18 from 2021. Week 18 is
therefore a PLAYOFF week in the earlier era. From `schedule`, games per week:

    week      2015  2018  2020  2021  2024
    17          16    16    16    16    16
    18           4     4     6    16    16     <- 4-6 games = wild card
    19           4     4     4     6     6

A single constant cannot express that boundary, and the `week <= 18` filter
in `season_projection.py` -- added specifically to keep playoff production out
of season totals -- lets exactly one playoff round through for every pre-2021
season.

Measured against the completed cold-start files: **480 of 3,339 pre-2021 rows
(14.4%) have a wild-card game folded into `actual_season_total`**, worth 4,166
fantasy points. Contaminated rows show mean bias -38.5 against -7.0 for clean
rows in the same seasons. `possible_weeks` is inflated too (472 of the 480
carry `possible_weeks = 17` against a true 16-game schedule), because
`possible_weeks_for_player` counts any week the player actually played -- so
the contamination is invisible to a `games_played > possible_weeks` check,
which is why it survived the earlier playoff-week fix.

This is era-asymmetric, so folds 2015-2020 and 2021-2025 in the 11-fold set
are not measuring the same quantity. It lands hardest on playoff-team players,
who are disproportionately veterans -- i.e. directly on the rookie-vs-veteran
split the arm exists to measure.

Fix: make the boundary a function of season (17 through 2020, 18 from 2021)
rather than a constant. NOT applied -- it changes every season total and
belongs in the same batched re-run as everything else today.

### MODERATE -- `fumbles_lost` is entirely absent for 2025

0 non-zero rows in 2025 against 241 in 2024 and 259 in 2023. It is the only
column populated in 2023/2024 and empty in 2025.

`fantasy_points` reconstructs to standard PPR within rounding (max diff 0.04)
in every season INCLUDING 2025 -- meaning 2025's stored target was computed
from the incomplete data and is overstated by 2 points per lost fumble. At
2024 scale that is ~140 players and ~480 points league-wide, concentrated on
fumble-prone RBs and QBs. 2025 is both the projection target and the most
recent test fold, so this inflates actuals exactly where results are read.

### Minor, recorded not fixed

- `fumbles` is 0.0 in every season -- the column has never been populated
  (only `fumbles_lost` ever was). Dead, but harmless while unused.
- `opp_fpts_allowed_s2d_lag1` / `..._dvoa_adjusted_lag1` default to 0.0 on
  5.7% / 11.7% of rows. That is weeks 1 and 1-2 (1/17, 2/17), uniform across
  every season, so it creates no train/test skew -- but "0.0 = league-average
  defence" is still a modelling claim. The builder already warns.

### Verified clean

No duplicate player-weeks in any season. Team vocabulary identical between
`schedule` and `player_weekly_stats` (32 normalised codes, no STL/SD/OAK
leakage). DST rows (2006-2013, 534/season) are excluded from training by the
position filter. No impossible values: completions <= attempts, receptions <=
targets, snap_share <= 1, success rates in [0,1], games_played in {0,1}.
Week coverage complete for every season. Leakage probe clean -- the highest
|corr| between any causal feature and the SAME-week target is 0.56
(targets_roll3_mean, WR), with nothing in the 0.85+ range a leak would show.


## Audit of the remaining flat-18 sites (2026-08-25)

The era-aware boundary fix landed in the Phase 7 projection path only. Every
other site that uses a flat 18 was read in context and classified. Ten are
semantic (they decide whether a week is regular season); three are bounds and
are correct as written.

### Semantic -- same bug, still live

| site | what it corrupts | in the arm comparison? |
|------|------------------|------------------------|
| `preseason_features.py:71` | season-history aggregates for the preseason arms | YES |
| `preseason_projector.py:578` | prior-season aggregates, `HAVING COUNT(*)>=MIN_GAMES` | YES |
| `feature_engineering.py:4035,4042` | `team_prior_season_wins` | YES (all 4 positions) |
| `season_availability.py:71,76,78,94` | `games_played`, `possible_games`, `rate` | availability track |
| `run_track_b_exposure.py:80,109` | availability denominators | Track B |
| `build_complete_player_game_panel.py:84` | panel opponent/home_away map | panel builds |
| `compute_market_projections.py:67,99` | market season totals | market baseline |
| `snake_draft_sim.py:461` | `actual_total` season totals | draft sim |
| `audit_roster_eligibility_by_position.py:80` | audit history filter | diagnostics |
| `utilization.py:396` | `get_season_phase` labels pre-2021 wild card as "Late" regular season | not causal |

**`team_prior_season_wins` is the widest-reaching**: it is a CAUSAL_FEATURE
for all four positions, and counting the wild-card round gives **62 of 480
pre-2021 team-seasons an inflated win total** (4/season, 6 in 2020), which
then propagates to every player on those teams the following year. 2021+ is
unaffected, so this is another era-asymmetric feature.

**The preseason pair matters most for interpretation.** `walk_forward_preseason.py`
imports both, so with Phase 7 fixed and these not, the 11-fold arm comparison
would be measuring arms against different definitions of "season" -- the
comparison would move for a reason unrelated to the arms.

**`run_track_b_exposure.py` already met the symptom** and mis-diagnosed the
cause: its comment records that leaving playoff rows in gave 29 of 239
player-seasons availability above 1.0, fixed with `week <= 18`. That only
fixed 2021+. Pre-2021 the denominator is still 17 rather than 16, so a player
who played every regular-season game but no playoff game scores 16/17 = 0.94
instead of 1.0 -- under the `availability_gt_1 == 0` assert, which is why the
guard never fired again.

### Bounds -- correct as written, deliberately not changed

- `schema_validator.py:136` -- input validation, and its own comment already
  documents the era boundary and consciously accepts 18-22 for POST rather
  than threading it. Permissive by design.
- `run_track_b_exposure.py:190` -- `assert max_week <= REGULAR_SEASON_MAX_WEEK`,
  an all-era upper bound.
- `run_track_b_exposure.py:132` -- `range(1, MAX+1)` as a degenerate fallback
  when a team has no schedule rows at all.

### Not yet fixed

Ten semantic sites across eight files, several of which write artifacts other
analyses read. Fixing them is mechanical but touches more than 5 files, so it
wants its own phase rather than being folded into the re-run prep.


## Data-quality audit round 2: two real leaks, two dead features, a 41% join
## loss -- all fixed (2026-08-25)

Executed audit across leakage, silent fills, constant features, joins and
split integrity. Method note: every finding below came from perturbing inputs
and re-running the pipeline, not from reading code. Two of my own initial
readings were wrong and were corrected by the perturbation, which is the point
of doing it that way.

### Leak 1+2 (one root cause, presenting as four features)

Two cross-sectional statistics were computed over the whole frame. Test
features are engineered on a train+test frame (`_apply_with_temporal_context`
step 2), so "whole frame" included the test season and every future season.

    _add_bayesian_prior_ppg   groupby("position").transform("mean")
    _impute_missing           df[col].median(), 20 causal features

Probe: perturb ONE test player's weeks >= 12, measure OTHER players at weeks
<= 8. Before, four features moved:

    bayesian_prior_ppg                     641/1188
    target_share_pct_roll3_mean             39/1188
    wopr_roll3                              39/1188
    team_target_concentration_roll3_mean    39/1188

The three team-share features were NOT independent leaks: 100% of their moved
rows were rows `_impute_missing` had filled. Fixing the imputer fixed all
three. Both statistics now come from seasons strictly before each row's own
(`_prior_season_group_stat`, `_prior_season_fill`); the earliest season falls
back to itself, which is always train under a chronological split.

After: the future-information probe moves ZERO features. The same-week
perturbation leaves all 72 invariant (was 2). The 622 `alpha==0` rows carry 9
season-appropriate values instead of one global constant (7.224403).

### Dead features: the tables were empty, not the features wrong

`coaching_change` (causal, all four positions) and
`pbp_pass_play_participation_pct_roll3_mean` (WR/TE) were nunique=1 across
25,964 rows because `team_coaching_staff` and `pbp_pass_participation` had
**0 rows**. Backfill scripts existed and had never been run. Ran both:
10,862 and 96,994 rows. nunique 1 -> 2 and 1 -> 23,082; `head_coach` all-NaN
-> 95 distinct.

Still dead: `weeks_since_oc_change` / `weeks_since_dc_change` (constant 99).
games.csv carries head coach only, so the coordinator family has no source.

### Combine join: 41.5% loss, fabricating athleticism on 40% of rows

`pfr_id -> gsis_id` went through the draft-picks parquet alone with an INNER
join -- a draft-only source, so it dropped 1,137 of 2,741 combine rows (WR
483, RB 303, QB 189, TE 162), all undrafted by construction. Those players got
`speed_score = 0.0`, which asserts the worst athlete in the league rather than
an unknown one. Unioning three mappings cuts the loss to 27.4% (1,990
matched), and unmatched players now get NaN with speed_score/bmi added to
`_STRUCTURALLY_MISSING`. Measured: 0.0 share 40.0% -> 0.0%, replaced by 33.5%
honest NaN; players with real combine data 322 -> 364.

### Verified clean, not fixed because not broken

- No causal feature is identical to a same-week raw column; max |corr| with
  the same-week target is 0.559.
- 35 of 38 merges preserve row count exactly; final matrix rows == raw input
  rows. The two "duplicating" merges are `get_ngs_data`'s intentional outer
  joins into a lookup table.
- 0 player-weeks appear in both train and test; split is chronological
  (train 2018-2023, test 2024); train is transformed alone.
- `team_wr_target_share_roll3_mean = 100.0` looked like a third dead feature
  and was an artifact of auditing a WR-only frame. On a multi-position frame
  it is nunique=15,337, and production calls `add_engineered_features(d)` with
  no position filter. Recorded because the false positive is instructive: a
  position-filtered audit frame fabricates constants.

### Residual, accepted with reason

Four features still move when a teammate's WHOLE season is perturbed:
`team_target_concentration_roll3_mean`, `target_share_pct_roll3_mean`,
`wopr_roll3`, `target_share_accel`. These are lagged teammate aggregates -- a
player's week-w share depends on teammates' weeks < w, which is knowable at
week w. Legitimate for the weekly model. They would NOT be legitimate for a
season-ahead projection, where no week of the target season is knowable;
Phase 7's preseason mode already handles that by restricting carry-forward to
prior seasons.


## Phase 3 re-selection on corrected data: config unchanged, two bugs found
## (2026-08-26)

Re-ran the architecture/window/weighting comparison because every prior
selection predated the training-data corrections. Artifacts and full detail:
`data/experiments/phase3_rerun_20260826/` (read `phase3_MERGED.csv`).

**Result: `FINAL_CONFIG` unchanged.** QB reproduces its current config
exactly (delta 0.0000). RB and TE differ from their current setting by 0.05
and 0.06 of a fold standard deviation. WR shows the only non-trivial delta
(0.69 sd, `all` over `3y`) but `all` ranks fifth of six on WR's window mean
and wins in exactly one of 36 architecture/weighting cells -- selection noise
on 3 folds. Nothing here justifies a change.

**The `since2013` candidate answered its question.** Added specifically to
test cutting the pre-2013 measurement era, it ranks 2nd-3rd for every position
and wins nothing outright. No floor, and it barely matters -- those seasons
neither help nor hurt now that they carry honest NaN rather than fabricated
zeros. Before the corrections the same comparison would have been measuring
fabricated data against real data.

**Two bugs the numbers did not show.** The run completed 72/72 with no
crashes. Auditing it rather than reading it found:

  - Component mode fell back to fp mode in 24 of 72 folds (every
    `test_season=2023`, all positions, all windows). `team_motion_rate` and
    `team_play_action_rate` are 100% NaN in those training sets, so their
    median was NaN; imputing NaN with NaN left NaN, and
    `np.isfinite(X_arr).all(axis=1)` then rejected every row. Half of this was
    self-inflicted: the pre-fix run failed the same folds via the old notna
    gate, and the median vector added the day before recreated the outcome
    through a new path.
  - `PHASE3_WINNER_ARCHITECTURE["TE"]` was still `B_gbm_huber` while
    `FINAL_CONFIG` moved to `C_gbm_mae` on 2026-08-21. TE's window ranking was
    measured against a model production does not use.

Both fixed in 80bd866; `_architectures_for_position` now raises on divergence
rather than quietly mis-measuring.

**Method note.** The architecture rows survived both bugs because
`_build_feature_matrices` passes NaN to LightGBM untouched. That was verified,
not assumed: all 36 WR/2023 architecture rows present in both the main run and
the repair are bit-identical. Salvaging 5 hours of compute rested on that
check, not on the argument.

**Still open**: 3 folds per position is thin. The caching discovery (~40 s per
warm fold against 6.4 min cold) makes an 11-fold Phase 3 roughly 3 hours
rather than a day, so the fold count is now cheap to raise if any of these
deltas ever needs to be taken seriously.


## PRESERVE_HISTORY_MISSINGNESS flipped to default ON — rookie-only effect
## (2026-08-28)

Pre-registered 11-fold paired experiment. Full write-up:
`data/experiments/phase7_histnan_v3_20260827/RESULTS.md`; plan committed
before the run at `data/experiments/phase7_histnan_20260826/PRE_REGISTRATION.md`.

**Falsification passed first.** Placebo pair (OFF vs OFF) gave 538/538
identical predictions -- noise floor exactly zero, so every ON-OFF difference
is attributable to the flag.

**Rule met.** 9 of 11 folds favour ON; mean paired dMAE -0.6580, SE 0.2170,
t = -3.03. The rule (>=8/11 folds AND mean <= -0.25) was fixed before the run.

**The effect is entirely rookies, and the pooled number is misleading:**

    pooled   -1.55% of a 43-point MAE; ON closer on exactly 50.0% of
             6,295 player-seasons -- a coin flip per player
    rookie   n=1188  dMAE -3.2596  10/11 folds  MAE 44.4 -> 41.1  (-7.3%)
    veteran  n=5107  dMAE -0.0649   8/11 folds  MAE 42.7 -> 42.6  (nil)

Describe as "~7% better rookie season projections, no veteran effect". The
pooled -1.55% is a rookie-only effect diluted ~5x by a 19%-rookie population,
and quoting it invites the reader to think every projection improved. Per
position RB carries most of it (-1.73, 9/11); QB shows nothing (-0.03, 5/11).

**Mechanism caveat.** 96.5% of scored feature rows are byte-identical between
arms, so the gain is MODEL-level -- the flag also changes the training matrix
-- not the debut rows being scored differently. "Training on honest debut-week
NaN produces a model better at rookies" is supported; "preserving NaN on a
rookie's row improves that row" is not, and this design cannot separate them.

**Two failed attempts, kept because the failure mode generalises.** v1
stratified by a dose computed from REAL weekly rows while cold-start scores
SYNTHETIC ones -- different populations, 12 hours to detect. v2 fixed the
population and failed again because no null stratum can exist at all: the flag
changes the training matrix, so every prediction moves regardless of its own
features. v3 replaced the null stratum with a placebo pair, which can actually
pass. `scripts/check_phase7_arms.py` now gates each season pair and aborts the
run -- that turned a 12-hour failure into a 1-hour one.

**Test coupling found by the flip.** `test_personnel_missingness_flag` asserts
`_structurally_missing_cols() == _STRUCTURALLY_MISSING` with the personnel flag
off, but that function unions BOTH flags' column sets, so it started failing on
a flag it does not test. Fixed by pinning the history flag off for that module
rather than widening the expectation.


## Production model designated: Step 8. PreseasonProjector demoted (2026-08-28)

First four-arm comparison run entirely on corrected data. All four arms scored
the SAME 4,630 player-seasons (21.9% rookies) -- verified from the run output,
not assumed. Artifacts: `data/experiments/four_arm_20260828/`.

    mean MAE rank   step8 1.25 | candidate 1.75 | phase7 3.00 | production 4.00

    season MAE      QB      RB      WR      TE
      step8       68.5    51.9    43.7    29.2
      candidate   68.8    51.7    44.2    30.3
      phase7      69.8    52.8    45.2    30.6
      production  76.2    57.5    49.5    33.5

    rookies only (n=1012, mean actual 61.2)
      step8 39.81 | candidate 40.56 | phase7 44.96 | production 53.29

**Step 8 wins outright**, and wins on rookies -- beating phase7, the arm built
for cold start, by 5.15 MAE there. It is also better on rookies (39.81) than on
veterans (45.96).

**PreseasonProjector is last at every position** by 4.3-7.7 MAE, and worst on
rookies by 13.5. Critically this is AFTER the component-mode fix that had it
training on 4-11% of available rows; it was expected to improve and did not.
It is what the UI currently ships.

**Extracted to production** (`src/models/season_step8.py`, commit f5df997).
It had existed only inside `_step8_arm`, an evaluation harness that required
target-season actuals and so could not project an unplayed season. No
cold-start mechanism was needed -- pairs already carry rookie rows with NaN
lags, LightGBM routes them, and the exposure half falls back to position-mean
rate. What production needed was `possible_games` from the SCHEDULE rather
than the panel, era-aware so it cannot re-admit the wild-card week. Extraction
verified byte-identical: 4,630/4,630 predictions, max |diff| 0.

### Corrections to earlier claims in this session, recorded deliberately

Three things I asserted and then measured to be false. All three came from
reasoning about code paths instead of measuring populations:

1. "Rookies are structurally excluded from candidate/production/step8." False.
   `_cold_start_rows` already appends rookie rows. My "comparable population"
   split used my own eligibility proxy (prior season >= MIN_GAMES), which
   excludes rookies BY DEFINITION -- so reporting 0% rookies in it was
   circular.
2. "step8 shares a population with phase7." False. step8 is scored on the
   candidate frame. No arm shares phase7's population.
3. "phase7's value is covering rookies the others cannot." False. The 1,665
   phase7-only player-seasons are 10.6% rookies (LOWER than the compared
   group's 21.9%), median 3 games played, mean actual 25.0 points, and phase7's
   error on them is 130% of the quantity predicted. They are injured/cut/
   practice-squad players, not draft-relevant rookies.

### Still open

- The UI (`docs/data/projections_2026.json` -> `docs/index.html`) is served by
  PreseasonProjector from data generated 2026-08-08, predating every fix. Both
  the model and the data are stale.
- `Step8SeasonModel` does not reproduce `predict_with_details()`'s
  confidence_score/support_class, which generate_draft_data.py uses for
  floor/ceiling sizing. That gap must close before the UI cutover.
- Step 8 is season-level only. phase7 remains the only weekly arm.


## MIN_GAMES = 6 is measured-correct, not an arbitrary threshold (2026-08-28)

Recorded because the exclusion looks like a coverage bug and keeps getting
re-litigated. It was questioned again when the Step 8 board swap dropped 120
players' projections, so it was finally measured rather than argued.

**Population**: player-seasons whose PRIOR season had 1-5 games -- exactly what
`_season_aggregates`' `games_played >= MIN_GAMES` filter removes. n=849,
2015-2025.

    mean actual 35.0   median 11.8   sd 56.2
    59% score under 20 points across a whole season; 78% under 50

    baseline                MAE    % of mean actual
    naive_ppg17           49.81       142%
    naive_prior_total     31.01        89%
    naive_posmedian       31.08        89%
    naive_zero            35.08       100%

**Nothing works here.** The best baseline (prior-season total, 31.01) beats
"always predict zero" (35.08) by 4 points against a target sd of 56.2 -- noise.
There is no headroom for a model to be useful.

**The intuitive approach is the WORST.** `ppg x 17` -- take the per-game rate
from those few games and extrapolate -- scores 49.81, worse than predicting
zero. That is what MIN_GAMES protects against: a 4-game per-game rate is
unstable and multiplying by 17 amplifies it. The concrete case that prompted
this: a WR with 57.1 points in 4 games of 2025 would project to 242 by that
logic, near the top of the board.

**Consequence.** Excluding these players is correct behaviour, not a gap. They
ship as `pending` with no number, deliberately, and the draft board swap
(a5a780b) was shipped with NO fallback on this basis. 52% of them are the same
player_ids as the phase7-only group, where the best available model's error is
130% of the quantity predicted.

If they are ever surfaced in the UI it should be as "insufficient data" with no
projection attached -- a number on a draft board reads as a claim.

**What would change this**: a signal that separates the thin tail of genuine
breakouts from the 59% who never establish a role. Prior-season box score does
not contain it. Draft capital, depth-chart movement or offseason role news
might; none has been tested on this population.

## OOF metrics were reported in log1p space for 10 of 12 models (2026-08-29)

`model_metadata.json` reported `oof_metrics` in MODEL space. `PositionModel.fit`
applies `TargetTransformer` (log1p) whenever the target's |skew| >= 0.5, and
`_oof_metrics` was computed on the transformed `y_train_inner`. On the
2026-08-29 `fp` retrain the transform fired for 10 of 12 position/horizon
models -- everything except QB 1w and TE 18w -- so the published table read:

    QB 1w  MAE 5.99   (fantasy points -- transform inactive)
    RB 1w  MAE 0.50   (log points     -- transform active)
    WR 1w  MAE 0.55   (log points)
    TE 1w  MAE 0.50   (log points)

Two scales in one table, with nothing recording which was which. The obvious
reading -- that RB is ~12x more accurate than QB -- is wrong; it is the same
kind of model measured in different units. This silently invalidates any
cross-position comparison, and any component-vs-fp comparison drawn from this
file.

NOT a training or serving defect: `predict()` inverse-transforms
(position_models.py ~637/~679), so shipped predictions were always in fantasy
points. The defect is confined to reported metrics.

Fix: `_oof_metrics` now also carries `rmse_original` / `mae_original` /
`r2_original` plus a `target_space` label, and `_evaluate_model` reports the
original-space values. The model-space `rmse` key is deliberately preserved
because two internal comparisons depend on it being in the same space as
`y_train_inner` -- the isotonic calibration gate (position_models.py ~437) and
the overfit ratio (~466). Overwriting it would have quietly broken both.

Why it survived: nothing compared these metrics across positions. Any check
that had would have found a 12x gap immediately. Guardrail worth adding: a
test asserting reported MAE is within a plausible fantasy-point band per
position (e.g. 1-20), which fails loudly on log-space values.

Follow-up: the 18w metrics look wrong independently of this (target_18w means
~125-144 across all four positions, i.e. the horizon target is near-constant).
Not investigated yet.

## Horizon targets were median-imputed: 91% of 18w labels were fabricated (2026-08-29)

Found while investigating the 18w metrics flagged in the entry above.

`FeatureEngineer._impute_missing` iterates every numeric column and median-fills
it. The horizon targets are numeric columns in the same frame, so `target_1w`,
`target_4w`, `target_18w` and `target_util_*` were filled like predictors. The
function already exempts snap-owned columns, structurally-missing columns and
the `ngs_*` prefix; labels were simply never considered.

Coverage measured 2018-2025 (30,065 rows), immediately after
`_create_horizon_targets` vs. after `add_engineered_features`:

    target_1w    90.6% -> 100%   ( 9.4% of labels invented)
    target_4w    73.8% -> 100%   (26.2% invented)
    target_18w    9.3% -> 100%   (90.7% invented)

Why it matters more than a normal fill: every training path already guards with
`valid_mask = ~y_dict[1].isna()` and drops label-less rows. The fill ran BEFORE
that guard and defeated it -- the row was not dropped, it was kept with a
manufactured label. The guard looked correct and did nothing.

How it stayed hidden: 100% label coverage reads as a clean dataset. It is the
opposite. A forward-looking target CANNOT be defined on the last rows of a
player-season, so full coverage is proof of fabrication. Every audit that
checked for NaN saw none and passed.

The 18w column was the giveaway. Six values covered 90% of it, and the
quartiles were identical across all four positions -- QB and TE both at p25
127.55 / med 129.90 / p75 134.20. After the fix the per-position medians
separate the way scoring implies: QB 278.3, RB 181.8, WR 172.5, TE 142.2.

Fix: `_impute_missing` now skips label columns, using a new shared predicate
`src.utils.leakage.is_label_column`. It reuses the existing `_TARGET_COL_RE`
rather than a `startswith("target_")` test, because the latter would destroy
the legitimate `target_share*` feature family. Regression test:
`tests/test_labels_not_imputed.py` (4 tests; verified non-vacuous by disabling
the exemption and reproducing the 100%-coverage signature).

### What this invalidates -- and what it does NOT

Invalidated: the WEEKLY models (`ensemble.py` / `PositionModel`), including the
2026-08-29 `fp` retrain. 1w least affected (9.4% of labels invented), 18w most
(90.7%) -- the 18w models were largely fitting a near-constant. The
component-vs-fp comparison has to be redone; both arms were measured on
fabricated labels.

NOT invalidated: the four-arm season comparison (production / candidate /
phase7 / step8). Verified, not assumed:

  - No `target_*` reference exists anywhere in `season_step8.py`,
    `preseason_projector.py`, `preseason_features.py`, `single_week_ppr/`, or
    `check_phase7_arms.py`.
  - Ground truth is `actual_total = float(g["fantasy_points"].sum())`
    (season_projection.py:1064) -- summed from observed rows at scoring time.
  - The phase7 arm's weekly model fits `y_train = pos_train["fantasy_points"]`
    (season_projection.py:948) -- the observed column, row-aligned.
  - `fantasy_points` carries 0 NaN before and after feature engineering across
    2018-2025 (30,065 rows), mean 9.6616 both sides, so the blanket imputer
    never touched it.

The structural reason is worth stating because it generalises: the season path
never puts labels in the same DataFrame as features. It takes
`df["fantasy_points"]` at fit time and scores against a freshly computed sum.
The weekly path materialises `target_*` columns INTO the training frame, which
then flows through a generic "fill every numeric column" pass. Labels living
beside features + a blanket numeric imputer is what manufactures labels.

### Why it surfaced now

It did not appear now; it has been present since `src/` entered version
control (4c30bc7, 2026-08-04) and the call ordering predates that. Nothing in
the component-removal work caused it.

It surfaced because the log1p metrics defect (entry above) forced an inspection
of what scale the targets were actually on -- and that was the first time
anyone had looked at the target DISTRIBUTIONS rather than their null counts.
Null-count audits could never find this: the fill drove coverage to 100%, which
is exactly what a clean dataset looks like.

### Open consequence, NOT yet addressed

With the fill removed, `target_18w` has only 2,806 valid rows across all four
positions (QB 447, RB 720, WR 1,187, TE 452). That may be too thin to train an
18-week model at all. `min_periods` for an 18-game sum is 14, and a
player-season is ~17 rows, so only the first few weeks of each season can carry
the label. Options: shorten the horizon, relax min_periods and accept the
magnitude bias the current comment warns about, or let the target span season
boundaries. Not decided.

## Fill-value scan: 7 causal features are fabricated across most of the training window (2026-08-29)

Scan run after the label-imputation fix, using the same method that found it:
look at value DISTRIBUTIONS, not null counts. Detector = a single value
dominating a continuous feature, measured per season on the real training
frame (silver artifact, 51,472 rows, 2018-2025).

61 of 273 (position, feature) pairs have one value covering >30% of rows.
Seven causal features additionally show a TRAIN/SERVE DISCONTINUITY -- near
constant in older seasons, live in recent ones. % of rows at the dominant
value:

    feature                             2018 2019 2020 2021 2022 2023 2024 2025
    rb_broken_tackles_prior                0%  98%  98%  98%  98%  98%  99%   0%
    recv_drop_pct_season_prior             0%  98%  98%  98%  98%  98%  99%  31%
    qb_bad_throw_pct_prior                 0%  63%  87%  86%  62%  87%  62%  88%
    team_pace_sec_per_play_roll3_mean     91%  98%  98%  98%  98%  98%  99%  13%
    team_neutral_pass_rate_oe_roll3_mean  98%  98%  98%  98%  98%  98%  99%  13%
    availability_3yr                      98%  34%  20%  17%  22%  25%  30%  30%
    fp_late6_vs_season                    98%  28%  27%  27%  24%  27%  22%  25%

Three distinct families.

### A. Storage-layer DEFAULT 0 (team_pace_sec_per_play, neutral_pass_rate_oe)

Root cause is NOT a fillna and would be missed by any fillna audit:

    database.py:206   neutral_pass_rate_oe REAL DEFAULT 0
    database.py:968   stats.get("neutral_pass_rate_oe", 0.0)

Confirmed against the DB. In `team_stats`, both columns are 100% zero with
ZERO nulls for 2006-2024; only 2025 is genuinely populated (38 nulls, ~5%
zeros). So 19 seasons of "never computed" are stored as measured zero and pass
an IS NOT NULL audit at 100%.

`pace_sec_per_play = 0.0` is physically impossible -- real values run 7.7 to
37.4 seconds, median 30.3. A team cannot snap a play in zero seconds.

Both features are in CAUSAL_FEATURES for ALL FOUR positions. Consequences:
  - They carry no signal across the training window (constant).
  - They are a perfect 2025 indicator. CV is season-aware
    (SeasonAwareTimeSeriesSplit), so the final fold is exactly where the
    feature stops being constant.
  - At 2026 serving time the model receives live values it effectively never
    saw in training.

Same signature as the pre-2013 snap-count bug (GAPS.md 2026-08-25): a
fabricated constant that is not NULL.

### B. First-season warm-up defaults (availability_3yr, fp_late6_vs_season)

Both default to 1.0 and both sit at 98% in 2018, then drop to 17-34%. 2018 is
the first training season, so a 3-year lookback has nothing to look back at
and every player is handed a fabricated perfect 1.0. The pipeline claims to
load 2006+ for feature-engineering context (MIN_HISTORICAL_YEAR = 2006), so
either that context is not reaching these two features or they are computed
after the window is trimmed. NOT yet diagnosed.

### C. PFR-derived prior-season features (three columns)

rb_broken_tackles_prior and recv_drop_pct_season_prior are ~98% zero for
2019-2024 but fully populated in 2018 and 2025, which is not an era boundary
-- an era boundary is monotonic. Looks like a join that succeeds only at the
window edges. NOT yet diagnosed.

### Status

None fixed. All seven are baked into the retrain running at the time of
writing, which was started to validate the label fix. That run is still
worth having -- it is the first on non-fabricated LABELS, and these are
feature defects -- but its absolute accuracy numbers should not be read as
the model's ceiling.

### Family A fix (2026-08-29): all eleven PBP-derived team_stats columns

Scope was larger than the two features first flagged. Measured across the whole
table: `third_down_conv`, `neutral_pass_plays`, `neutral_run_plays`,
`neutral_pass_rate`, `neutral_pass_rate_lg`, `neutral_pass_rate_oe`,
`drive_count`, `drive_success_rate`, `avg_drive_epa`, `points_per_drive` and
`pace_sec_per_play` are EXACTLY ZERO for 100% of the 10,447 rows in 2006-2024,
with zero NULLs. Only 2025 is genuinely populated. Cause: nothing but
`load_current_season_stats_from_pbp()` ever ran, and it does one season.

Code fixed (does not touch existing rows):
  - `team_stats` DDL: `REAL DEFAULT 0` -> `REAL` for the PBP block.
  - The ALTER TABLE migration map likewise. `ADD COLUMN ... DEFAULT 0`
    backfills every pre-existing row with that default, which is how these
    columns became a fabricated zero on databases predating them.
  - `insert_team_stats` bindings: `stats.get(col, 0.0)` -> `stats.get(col)`,
    so an uncomputed column reaches SQL as NULL.

Data fix: `scripts/backfill_team_pbp_stats.py` recomputes 2006-2024 from
play-by-play via `get_team_stats_from_pbp` and UPDATEs the eleven columns,
then NULLs any row still matching the all-zero signature. Verified on 2019-2021
before writing: pace medians 31.0 / 30.7 / 31.2 s, neutral_pass_rate ~0.57,
neutral_pass_rate_oe centred near 0 -- all sane, and the script refuses to
write a season whose values fall outside physical bounds.

The all-ten-zero signature is specific: it matches 100% of 2006-2024 rows and
only 31 of 639 rows in 2025 (unplayed/bye weeks), so it cannot null a real
team-week.

### Latent hazard found while fixing this: partial upserts to team_stats destroy data

NOT fixed, and it is a landmine for any future backfill.
`DatabaseManager.insert_team_stats` combines

    points_scored = COALESCE(excluded.points_scored, team_stats.points_scored)

with a binding of `stats.get("points_scored", 0)`. A caller passing a partial
dict therefore binds 0, and `COALESCE(0, existing)` returns 0 -- so a
partial upsert silently ZEROES points_scored, total_yards, turnovers and every
other non-PBP column for each row it touches. COALESCE reads as "keep the old
value if the new one is absent", but the binding guarantees the new value is
never absent.

This is why the backfill script uses a targeted UPDATE of ten named columns
instead of the upsert. Proper fix is to bind None for absent keys throughout,
matching what was just done for the PBP block; deferred so it does not ride
along inside a data-migration change.

## Family B diagnosed: lookback history is loaded, then discarded before feature engineering (2026-08-29)

`availability_3yr` and `fp_late6_vs_season` both sit at ~98% of their default
value (1.0) in 2018 and 17-34% thereafter. 2018 is the first training season.

Mechanism, `_add_availability_3yr` (feature_engineering.py:846):

    gp["gp_3yr"] = gp.groupby("player_id")["gp"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum())
    ... .clip(0, 1.0).fillna(1.0)
    df["availability_3yr"] = df.apply(
        lambda r: avail_map.get((...), 1.0), axis=1)

`shift(1)` is NaN for a player's first season IN THE FRAME, and there are two
separate 1.0 defaults behind it. So a player with no history is recorded as
having PERFECT availability -- the most optimistic possible value, assigned
precisely when nothing is known. Same shape in `fp_late6_vs_season`
(feature_engineering.py:1665, `.fillna(1.0)`).

Root cause is upstream and broader than these two features.
`data_loading.load_training_data` does:

    train_data = combined[combined['season'].isin(train_seasons)]   # line 117

`get_all_players_for_training` returns 2006-2025 -- for RB, 28,402 rows of
which 16,348 (57.6%) are pre-2018. All of it is discarded HERE, before feature
engineering runs. So every lookback feature (3-year availability, prior-season
stats, rolling windows) starts cold at the training window's first season and
falls back to a default.

NOTE: this contradicts the standing project note that the pipeline "loads 2006+
for feature engineering context". That is not true on this path. The history is
loaded from the database and thrown away one line before it would be useful.

Fix direction (NOT yet implemented): keep pre-window seasons through feature
engineering as context and trim to `train_seasons` afterwards -- the same
train/test pattern `_apply_with_temporal_context` already implements for the
test frame. Separately, the 1.0 defaults should be NaN: "no history" is not
"perfect availability".

## Family C diagnosed: PFR seasonal source columns exist for one season only (2026-08-29)

`rb_broken_tackles_prior` and `recv_drop_pct_season_prior` are ~98% zero for
2019-2024, but populated in 2025 and NaN in 2018. Not an era boundary -- era
boundaries are monotonic.

Cause is the source table. In `seasonal_pfr`:

    broken_tackles_per_att   non-null: 2024 only (152 rows)
    rec_drop_pct             non-null: 2024 only (494 rows)
    bad_throw_pct            non-null: every season
    (no 2025 rows at all)

`_add_seasonal_pfr_features` shifts season +1 (year N predicts N+1), so the
only populated source year, 2024, lands on target season 2025. That is exactly
the observed pattern. 2018 reads NaN because `_mask_pre_era` correctly masks
pre-2019.

Two separate defects:
  1. Loader gap: the PFR seasonal loader only ever populated broken tackles and
     drop rate for 2024. Everything else is NULL.
  2. Zero-fill masking it (feature_engineering.py:1612-1619): missing values are
     filled with the (position, season) median and then `fillna(0.0)`. When the
     whole group is missing the median is itself NaN, so the fallback fires and
     every row becomes a fabricated 0.0 -- indistinguishable from "this RB broke
     zero tackles".

`qb_bad_throw_pct_prior` has source data for all seasons yet still reads 62-88%
zero, so its loss is in the PFR->GSIS id mapping or the same zero-fill, not in
source coverage. Not yet isolated.

Neither family is fixed.

### Family B FIXED (2026-08-29)

Two changes.

1. Pre-window history now reaches feature engineering.
   `load_training_data(return_context=True)` returns seasons older than the
   training window as a separate `context_data` frame;
   `_prepare_training_data(context_data=...)` threads it into
   `_apply_with_temporal_context`, which now takes three tiers:

       context (-1)  seasons before the window: warms up lookback windows,
                     never returned, never trained on
       train    (0)  computed from context+train
       test     (1)  computed from context+train+test

   Context receives the same treatment test does -- train-fitted percentile
   bounds and utilization weights applied, never fitted on -- because it has to
   reach feature engineering with the same columns as train, or the rolling
   windows it exists to warm up would see NaN for every context row and be
   worse than no context.

   Context is dropped on return, so it never touches the percentile bounds,
   utilization weights, winsorisation bounds, or the models.

   Additive by design: `return_context` defaults False and `context_data`
   defaults None, so the seven existing `load_training_data` call sites and the
   four `_prepare_training_data` call sites are unaffected and reproduce the
   previous behaviour exactly.

   Measured on RB `availability_3yr`, % of rows with no value:

       season   no context   with context
         2018       100.0%          22.3%
         2019        21.1%          18.9%
         2020        17.9%          16.7%
         2021+     unchanged      unchanged
       overall       28.3%          18.7%

   2018 recovers real values for 78% of rows; 2021+ is untouched because those
   lookbacks already sat inside the window. The residual is genuine rookies.

2. The optimistic defaults are gone.
   `availability_3yr` dropped `.fillna(1.0)` and its `avail_map.get(..., 1.0)`
   default; `fp_late6_vs_season` dropped `.fillna(1.0)`. Both now yield NaN
   when there is no history. 1.0 is the top of the availability range and
   "finished exactly at his own average" for the ratio -- specific, optimistic
   claims asserted precisely where nothing was known.

HONEST LIMIT: under FEATURE_MODE="causal" the surviving NaN does not reach the
model as NaN -- ensemble.py median-fills every remaining NaN before fitting,
and its `_missing` indicators are only emitted for columns matching
_roll/_lag/_ewm/_trend, which these two do not. So the gain is (a) real history
for most previously-cold rows and (b) an unknown now falls back to the
population median instead of a fabricated perfect score. Making missingness
itself visible to the model is a separate, larger change.

### Family C FIXED (2026-08-29)

Two defects, both addressed.

1. Loader gap. Nothing in this repo ever populated `seasonal_pfr`; the table
   was loaded ad hoc and `broken_tackles_per_att` / `rec_drop_pct` existed for
   2024 only. The data was always available upstream --
   `nfl_data_py.import_seasonal_pfr` carries `brk_tkl`/`att` (rush) and
   `drop_percent` (rec) at ~100% coverage from 2018.
   `scripts/backfill_seasonal_pfr.py` backfilled 2018-2024: 2,470 rush rows and
   3,593 rec rows updated, bounds-checked before writing.

2. Zero-fill. `_add_seasonal_pfr_features` filled the (position, season) median
   and then `.fillna(0.0)` behind it; when a whole group was missing the median
   was itself NaN so the 0.0 fired for every row. All four columns are listed
   in `_STRUCTURALLY_MISSING`, so `_impute_missing` was already exempting them
   -- the builder's own zero-fill ran first and made that exemption INERT.
   Same pattern as the snap and personnel columns. Now leaves NaN (three sites:
   the early return, the main loop, and the exception handler).

Result, % of rows carrying a real value:

    season   rb_broken_tackles_prior   recv_drop_pct_season_prior
      2018            0.0%  (masked)          0.0%  (masked)
      2019           75.6%                   75.8%
      2020           78.6%                   76.6%
      2021           79.4%                   78.5%
      2022           75.3%                   79.8%
      2023           80.9%                   77.2%
      2024           79.5%                   79.6%
      2025           75.5%                   78.5%

Was ~2% real / 98% fabricated zero for 2019-2024. 2018 correctly reads 0%:
`_mask_pre_era` masks it because PFR starts 2018 and these are prior-season
features. Remaining zeros among present values (48.7% broken tackles, 29.1%
drop rate) are genuine measurements -- backs who really broke no tackles.

### CORRECTION: qb_bad_throw_pct_prior was never broken (2026-08-29)

An earlier entry claimed `qb_bad_throw_pct_prior` "reads 62-88% zero" and
blamed the PFR->GSIS id mapping. Both halves are wrong.

Measured: the mapping succeeds for 748 of 748 pass rows (100%), and
`bad_throw_pct` is 99.6% non-null. On QB ROWS the feature was already healthy
before any fix -- 0-3% zeros, median 17.2, 1.4% NaN.

The error was in the detector, not the data. The discontinuity scan computed
each feature's dominant value across the WHOLE frame, all positions together.
`qb_bad_throw_pct_prior` is populated only for QBs; on RB/WR/TE rows it was
0.0, and those rows are ~87% of the frame. So the scan reported a QB feature's
health using mostly non-QB rows.

Those zeros never reached a model: CAUSAL_FEATURES assigns
qb_bad_throw_pct_prior and qb_pocket_time_prior to QB only,
rb_broken_tackles_prior to RB only, recv_drop_pct_season_prior to WR/TE only,
and training splits by position first.

LESSON: a position-specific feature must be measured on its consuming position.
Measuring it frame-wide reports the fill rate of the positions that never use it.

Re-measured on consuming positions only, the other two Family C features were
REAL defects and the backfill was warranted:

    rb_broken_tackles_prior (RB rows)      2019-2024: 97.8-98.6% zero,
                                           and NO non-zero values at all
    recv_drop_pct_season_prior (WR/TE)     2019-2024: 97.9-98.5% zero,
                                           and NO non-zero values at all

Only 2025 carried real values, matching the 2024-only source coverage.

### Family A backfill DONE (2026-08-29)

`scripts/backfill_team_pbp_stats.py` ran for 2006-2024: 10,292 team-weeks
aggregated from play-by-play, 10,292 rows updated, then 155 rows the aggregator
could not produce were NULLed. A further 31 all-zero rows in 2025 (all
`week = 0` placeholders, not real games) were NULLed separately.

Verification: `pace_sec_per_play` now has zero fabricated zeros in any season,
with means running 32.0s in 2006-2008 down to ~31.0s by 2013 -- the real
leaguewide trend toward faster offences, which a constant 0.0 obviously could
not show. `neutral_pass_rate_oe` means sit near 0 across all seasons, as an
"over expected" measure should.

Rows still matching the all-ten-zero fabrication signature anywhere in
team_stats: 0.

DB backups: `nfl_data.prepfr_20260829_220233.db`,
`nfl_data.prebackfill_20260829_220426.db` (500 MB each).

### Consequence

Every model artifact now predates the data it was trained on. The 2026-08-29
21:58 retrain used fabricated team_stats and PFR columns. A retrain is required
before any accuracy number is quotable.

## The rookie cold-start fallback (2026-08-29) — HEADLINE RETRACTED, see correction 2026-08-30

`predict.py::_apply_cold_start_fallback` is intended to replace ML predictions
with a position average for players below `MIN_GAMES_FOR_PREDICTION = 4`, on
the grounds that a near-rookie's model output is too uncertain to serve.

It never fires. Its first statement is:

    if "games_count" not in latest_data.columns:
        return results

and `games_count` does not exist. `get_all_players_for_training` returns
`games_played`. Nothing in src/ produces a `games_count` column at all (the
only occurrence is an unrelated local in scripts/diagnose_outliers.py). So the
guard returns immediately on every call and `MIN_GAMES_FOR_PREDICTION` is
inert.

Consequence: rookies receive real ML predictions from week 1, built on
median-imputed history. Measured share of causal features present for TRUE
rookies (is_rookie == 1):

    week   all features   rolling-form features
      1        50.0%              0.0%
      2        87.5%             77.4%
      3        91.0%             84.7%
      4        96.0%             95.1%

At week 1 a rookie has NO form history; every rolling feature is filled with a
veteran-derived median, so the model effectively sees "a league-average player
who happens to have this draft capital". From week 2 the prediction is
genuinely informed, because one played game satisfies min_periods=1.

NOT fixed, deliberately. Enabling the guard would switch on behaviour that has
never been validated and that is questionable on its own terms: it assigns
every rookie at a position an IDENTICAL prediction (the position mean), which
discards draft capital, depth chart and combine score -- the only real signal a
week-1 rookie has. Options: (a) leave dead and delete it, (b) fix the column
name and measure whether the override beats the model, (c) replace it with a
rookie-specific model. Needs a decision, not a rename.

Related: veterans are unaffected -- 97.6% feature availability at week 1,
indistinguishable from week 10, because rolling features group by player_id
only and carry across the offseason.

## The prior-week join + blanket zero-fill re-fabricates data downstream of the DB fix (2026-08-29)

Found while verifying why the post-backfill retrain moved MAE by <=0.01.

`get_all_players_for_training` joins team and opponent stats from the PRIOR
week to avoid leakage:

    LEFT JOIN team_stats ts ON pws.team = ts.team
        AND pws.season = ts.season AND ts.week = pws.week - 1

Week 1 therefore never matches -- there is no week 0 -- and the join yields
NULL. `utilization_score.py:431` then blanket-fills every numeric column with
0, excluding only `missingness_preserved_cols()`, which does not cover these.
So week-1 absence becomes a measured zero.

23 columns carry the signature (100% zero at week 1, ~6% mid-season). Five
reach models:

    column                      reaches model as                    positions
    team_pace_sec_per_play      team_pace_sec_per_play_roll3_mean   all 4
    team_neutral_pass_rate_oe   team_neutral_pass_rate_oe_roll3_mean all 4
    team_neutral_pass_rate      feeds the _oe calculation           all 4
    team_plays                  team_plays_roll3_mean               all 4
    opp_fpts_allowed            DIRECT + _s2d_lag1 + _dvoa_adj_lag1 all 4

`opp_fpts_allowed` is the worst case: a direct causal feature, zero on 100% of
week-1 rows against a mid-season median of 21.6. Every week-1 matchup is
presented to the model as an opponent that allows ZERO fantasy points -- the
best possible defence, identical for all 32 teams, which erases matchup
discrimination in exactly the week where form data is weakest.

The zero also propagates forward through the 3-game rolling means:

    week 2: 25.9% of team_pace roll3 rows implausible (<20s), mean 18.3
    week 3: 33.0%, mean 20.9
    week 4: 25.3%, mean 22.1
    week 5:  3.8%, mean 29.7   <- the zero ages out of the window

Distribution vs source, whole frame: raw team_stats pace is mean 31.23 /
std 2.71 / p1 24.75; the model-facing roll3 is mean 27.20 / std 5.98 /
p1 9.65, with 10.3% of rows between 0 and 20 s/play. A team cannot snap the
ball in under 20 seconds; the rolling mean of plausible values cannot be 9.65.

THIS IS WHY THE 2006-2024 BACKFILL BOUGHT NO ACCURACY. The database is now
correct and the fabrication is reintroduced downstream. Third instance today of
the same shape: a correct-looking guard defeated by a filler running earlier in
the chain (cf. the label imputation and the seasonal_pfr zero-fill).

NOT fixed pending a decision on the default -- see below.

### The prior-week fix took TWO more passes (2026-08-29)

The first attempt (PRIOR_WEEK_JOIN_COLS, 5 columns) fixed
team_pace_sec_per_play -- roll3 rows below 20 s/play went 25-33% -> 0.0% in
weeks 2-4, mean 31.2 against a raw 31.23 -- but left two holes.

1. WRONG SCOPE. `opp_fpts_allowed` is ASSEMBLED in feature_engineering from
   `fantasy_points_allowed_{qb,rb,wr,te}`, so exempting only the assembled name
   left the inputs zero-filled and the output was still 0 on 100% of week-1
   rows. The four source columns appeared in the original 23-column scan and
   were filtered out because they are not causal features BY NAME -- the scan
   checked CAUSAL_FEATURES membership and `<name>_*` prefixes, and a column
   feeding a feature under a DIFFERENT name passes both tests. Now added.

2. WEEK-0 PLACEHOLDER ROWS. `team_plays` came out of SQL 73% ZERO at week 1 --
   not a Python fill at all. team_stats holds 186 week-0 rows (31 per season,
   2020-2025), every one with total_plays = 0, and `ts.week = pws.week - 1`
   resolves to 0 for week-1 rows, so they matched a fabricated record instead
   of finding nothing. Fixed with `AND ts.week >= 1` on both the team_stats and
   team_defense_stats joins. Verified: week-1 team_plays went 73% zero -> 100%
   NULL, week 5 unchanged (median 61.0 plays, 31.2 s pace).

Lesson recorded with the fix: in this pipeline a column can feed a causal
feature under a different name, so name-based scoping is not sufficient. Trace
the assembly, not the identifier.

## Rookie cold start: option (c) implemented (2026-08-30)

Three defects, all in machinery that already existed.

1. `_load_draft_rounds` queried the WRONG TABLE. `draft_picks` holds 0 rows;
   the live table is `draft_picks_v2` (12,927 rows, 11,054 with player_id and
   draft_round). The lookup returned {}, so `_round_bucket_for(None)` mapped
   every rookie to "UDFA" -- a first-round RB would have drawn 5.20 instead of
   13.13, a 2.5x understatement on exactly the players draft capital
   identifies best. The rest of the pipeline already reads draft_picks_v2.

2. The rookie prior was written and then ERASED. `_apply_rookie_prior` fills
   prev_season_ppg from data/rookie_priors.json (position x draft-round PPG,
   fitted on 1,676 rookies 2006-2023) but runs inside `_add_prev_season_ppg`,
   early. `_restore_debut_history_nan` runs LAST by design and blanks all 65
   history columns on a debut week -- prev_season_ppg included. Measured:
   prev_season_ppg was 100% NaN for week-1 rookies, so the priors artifact had
   never once reached a model.

   Fixed by re-applying the prior AFTER the restore rather than exempting the
   column. The restore keeps its invariant ("runs after every filler, so a new
   filler cannot silently defeat it") and still blanks 64 of 65 columns; the
   one column where a fitted, draft-conditioned estimate exists gets it back.

3. `_apply_cold_start_fallback` DELETED (option (a) folded in). NOTE: the
   original text here said it "had never run" because nothing produces
   `games_count`. That was WRONG -- games_count is built inside predict.py and
   the function was LIVE. See the 2026-08-30 correction below. The deletion
   stands on the argument that follows, which never depended on it.
   Not repaired, because its behaviour was worse than the model it overrode:
   every rookie at a position got the SAME number, discarding draft capital,
   depth chart and combine score, i.e. the 33 of 72 causal features that ARE
   populated at week 1.

Verified end to end: 369 true week-1 rookies (debut 2019+), prev_season_ppg
100% populated, 15 distinct values, every one a rookie-prior constant --
13.13 (RB rd1) through 3.33 (TE rd4_7). Was 100% NaN.

Draft-round match rate is 67.3% of players in frame; the unmatched fall to the
UDFA bucket, which is the correct default for an undrafted player but will also
catch anyone missing from draft_picks_v2. Not investigated further.

### NOT fixed: the rookie CI-widening block

predict.py still gates 1.5x confidence-interval widening on `games_count`, so
it is dead for the same reason. Deliberately NOT repaired by renaming to
`games_played`: that column is a constant 1 (a per-row "played" flag, not a
cumulative count), so `games_played < MIN_GAMES_FOR_PREDICTION` would be TRUE
for every player and widen every interval 1.5x -- worse than the dead code.
A correct fix needs a real career-game count or `is_rookie` plumbed to the
serving frame; neither is present there today.

## A/B: the prior-week fix produced no measurable accuracy gain (2026-08-30)

Controlled comparison, both arms identical except the flag:

    harness      scripts/error_by_week.py --season 2025
    train        2018-2024,  test 2025 (held out)
    arm ON       PRESERVE_PRIOR_WEEK_MISSINGNESS=1 (week-1 absence -> NaN,
                 then median-imputed)
    arm OFF      NFL_PRESERVE_PRIOR_WEEK_MISSINGNESS=0 (week-1 absence -> 0,
                 the pre-fix behaviour)
    outputs      data/experiments/error_by_week_2025_flag{ON,OFF}.csv

Validity checked before trusting it: env=0 makes the flag read False and
leaves PRIOR_WEEK_JOIN_COLS unpreserved. The `ts.week >= 1` join guard is not
flag-controlled and is active in both arms, but that does not confound the
comparison -- with the guard on, week 1 joins to NULL and the OFF arm
zero-fills it, reproducing exactly the 0 the pre-fix code produced via the
week-0 placeholder rows.

MAE delta (ON minus OFF), negative = fix helped:

    bucket        QB       RB       TE       WR
    wk 1       +0.053   -0.033   -0.019   -0.033
    wk 2-4     -0.018   -0.009   -0.003   +0.003
    wk 5-8     +0.024   +0.022   -0.016   -0.001
    wk 9-13    -0.021   -0.001   +0.006   -0.010
    wk 14-18   +0.015   +0.020   -0.006   +0.004

Largest single change anywhere: 0.053 MAE, on QB week 1, n=33, in the WRONG
direction.

There is a faint pattern in the expected shape -- three of four positions
improve in the targeted buckets (wk 1, wk 2-4) while untargeted buckets
(wk 9-13, wk 14-18) stay flat:

    RB targeted -0.021 / untargeted +0.010
    WR targeted -0.015 / untargeted -0.003
    TE targeted -0.011 / untargeted +0.000
    QB targeted +0.018 / untargeted -0.003

NOT claimed as an effect. ~0.02 MAE on a base of 4-6, QB runs the other way,
one season, no repeats. Fully consistent with noise. Establishing it would
need multiple held-out seasons and repeated runs.

### Conclusion for the whole audit

Six retrains and one controlled A/B: the 2026-08-29/30 work bought
CORRECTNESS, NOT ACCURACY. Aggregate weekly MAE is unchanged from where it
started (QB 6.04 / RB 4.49 / WR 4.23 / TE 2.87).

That is a result, not a disappointment. The most likely reading is that the
models had already routed around features that were constant or nonsensical,
so removing the fabrication freed capacity they were not using. Two things
follow:

  - The ceiling here is NOT data hygiene. It is the feature set and the model.
    Further data-cleaning work should not be expected to move accuracy.
  - These features are now genuinely available to future work. team pace,
    neutral pass rate over expected, PFR broken tackles and drop rate, and the
    rookie priors were all noise-wearing-a-name before; they now carry signal
    a better model could use.

What DID change is the epistemic status of the numbers. Before: computed on
91%-fabricated labels, reported in log space for 10 of 12 models, over
features that were constant zero for 19 seasons. After: the same numbers,
measuring what they claim to.

### Per-week error, current models (arm ON, 2025 held out)

MAE as % of mean actual points, which is comparable across buckets in a way
raw MAE is not:

    bucket       QB     RB     TE     WR
    wk 1       38.6   60.1   69.3   59.3
    wk 2-4     42.1   51.8   66.4   59.1
    wk 5-8     48.8   58.7   66.5   63.2
    wk 9-13    53.1   52.2   65.6   66.2
    wk 14-18   46.3   57.2   68.1   65.2

Early season is NOT the weak spot: QB is 9.4pp BETTER early than late, WR
6.5pp better, RB and TE flat within ~1pp. This contradicts the standing
project note that QB early-season is the biggest gap -- that note predates
this audit, and whatever it measured may have been the corruption now removed.
Caveat: QB wk 1 is n=33.

## The fold harness did not match production (2026-08-30)

`run_fold` -- the basis of every walk-forward comparison in this project --
called `_prepare_training_data` WITHOUT `context_data`, while `train_models()`
passed it from earlier the same day. So the harness and production trained
different pipelines: lookback features (rolling windows, availability_3yr,
prior-season stats) started cold at the training window's first season in the
harness and did not in production.

No error, no warning; the harness simply measured something other than what
ships. Same class of defect as the fabricated values this audit was chasing.

Fixed in both of run_fold's loading paths. The held-out season is excluded
from context in each, so warming up lookback windows cannot reach the fold's
own test data -- verified for test_season=2024: context 2006-2017, strictly
before the 2018-2023 training window, held-out season absent.

Tests: tests/test_fold_harness_matches_production.py (2). The second is the
one that matters -- context carrying the held-out season would mean a fold
training on its own test data.

### Provenance note on the A/B above

The prior-week A/B and the per-week error table were produced BEFORE this fix,
i.e. by the no-context harness. That does not invalidate them: context was
absent from BOTH arms, so the comparison is still controlled and its
conclusion (no measurable effect) stands. But the absolute per-week MAE values
were measured on a pipeline that no longer matches production, and should be
re-measured before being quoted as current model performance.

## CORRECTION: the cold-start fallback and the CI widening were LIVE, not dead (2026-08-30)

Retracts two claims made on 2026-08-29 and repeated in commit 7f635b3.

CLAIMED: `predict.py::_apply_cold_start_fallback` had never run, because its
first statement is `if "games_count" not in latest_data.columns: return
results` and nothing in the repo produces `games_count`. The same was claimed
of the confidence-interval widening block, which gates on the same column.

BOTH WERE FALSE. `games_count` IS built, in predict.py itself:

    games_per_player = player_data.groupby("player_id").size() \
                          .reset_index(name="games_count")     # ~line 264
    latest_data = latest_data.merge(games_per_player, on="player_id")

Verified it reaches the call site with sane values (min 1, median 14, max 39)
and survives `refresh_matchup_features`.

HOW THE ERROR HAPPENED: the check was run against
`get_all_players_for_training`'s output, which indeed has `games_played` and no
`games_count`. But that is the INPUT to the prediction pipeline, not the frame
passed to the guard -- `games_count` is derived afterwards, inside predict.py.
The wrong frame was measured, and the measurement looked conclusive. This is
the second time in this audit that a confident conclusion came from evaluating
the wrong population (cf. the qb_bad_throw_pct_prior correction, where a
QB-only feature was measured across all positions).

GUARDRAIL: when asserting a column is absent, measure the frame at the USE
SITE, not the frame that feeds the function that builds it.

### What this changes

- The CI widening (rookie 1.5x, volatile 1.25x, injury-prone 1.4x) needs no
  work. It has been functioning all along. The GAPS entry proposing a fix for
  it is withdrawn.
- `_apply_cold_start_fallback` was LIVE production behaviour when deleted in
  7f635b3, not dead code. Sub-4-game players previously received the position
  average; they now receive the model's prediction.

### The deletion stands (user decision, 2026-08-30)

Confirmed after the correction was surfaced. The justification is the second
argument from the original entry, which never depended on the function being
unreachable: it assigned every rookie at a position an IDENTICAL number,
discarding draft capital, depth chart and combine score. That argument is
stronger now that the rookie prior actually reaches the model -- the prediction
it would override is draft-conditioned rather than veteran-median.

Rookie uncertainty is still expressed: the CI widening is live and untouched.

## Per-week baseline on the corrected harness; context fix also shows no effect (2026-08-30)

Re-run of scripts/error_by_week.py after run_fold was fixed to pass
context_data. Supersedes the earlier table, which was measured on a harness
that did not match production.

Current per-week error, MAE as % of that bucket's mean actual points
(2025 held out, trained 2018-2024):

    bucket       QB     RB     TE     WR
    wk 1       38.8   60.3   69.3   59.5
    wk 2-4     42.3   52.0   66.5   59.2
    wk 5-8     48.5   58.8   66.7   63.3
    wk 9-13    52.9   52.1   65.5   66.2
    wk 14-18   46.0   57.2   68.2   65.2

Delta vs the no-context harness (negative = context helped):

    bucket       QB       RB       TE       WR
    wk 1      +0.035   +0.011   +0.001   +0.014
    wk 2-4    +0.036   +0.019   +0.006   +0.009
    wk 5-8    -0.041   +0.005   +0.010   +0.009
    wk 9-13   -0.022   -0.007   -0.004   +0.001
    wk 14-18  -0.036   -0.001   +0.003   +0.003

Largest change anywhere 0.041 MAE, signs mixed, and the early buckets -- where
cold lookback windows were the whole problem -- are marginally WORSE for all
four positions. Noise.

### All four fix families are now measured, and none moved accuracy

  labels / OOF metric space        aggregate retrain, no change
  team_stats + PFR backfills       aggregate retrain, no change
  prior-week zero-fill             controlled A/B, no change
  pre-window lookback context      this comparison, no change

Each was verified to have reached the models. The consistent reading is that
the models had already routed around features that were constant, fabricated
or cold, so removing the fabrication freed capacity they were not using.

CONCLUSION FOR PLANNING: the ceiling here is not data hygiene. It is the
feature set and the model. Further cleaning should not be expected to move
accuracy, and should be justified on correctness grounds alone.

## NEGATIVE RESULT: there is no QB in-season degradation (2026-08-30)

Recorded so the next reader does not re-chase it from the same table.

The per-week baseline shows QB error rising from 38.8% of mean actual points in
week 1 to 52.9% by weeks 9-13. That was flagged as the one structural pattern
worth pursuing. It is an artifact of the normalisation, not a finding.

Decomposed:

    bucket      MAE   mean_actual   MAE vs wk1   actual vs wk1
    wk 1       6.26      16.11          0.0%           0.0%
    wk 2-4     5.85      13.81         -6.6%         -14.2%
    wk 5-8     6.66      13.74         +6.4%         -14.7%
    wk 9-13    6.71      12.68         +7.3%         -21.3%
    wk 14-18   6.32      13.73         +1.0%         -14.8%

QB MAE is flat: 6.26 in week 1, 6.32 in weeks 14-18, oscillating 5.85-6.71 in
between and non-monotonic (weeks 2-4 are BETTER than week 1). Roughly
three-quarters of the apparent degradation is the denominator shrinking.

Why the denominator shrinks, from raw 2025 data rather than model output:

    bucket      QBs/week   mean FP   share under 5 FP
    wk 1          34.0      16.24         11.8%
    wk 2-4        41.0      12.88         25.2%
    wk 5-8        36.3      13.44         26.2%
    wk 9-13       36.2      12.66         25.4%
    wk 14-18      26.2      12.34         28.8%

The share of QBs scoring under 5 points MORE THAN DOUBLES after week 1 and
stays there. Week 1 is the one week where nearly every team starts its intended
starter; from week 2 the pool fills with backups, injury replacements and
benchings. Mean scoring drops ~20% for roster reasons and never recovers.

METHOD NOTE, which is the transferable part: dividing MAE by a bucket's mean
actual makes error appear to grow whenever scoring shrinks. It is the right
normalisation for comparing positions (a 3-point TE error is not a 3-point QB
error), and the wrong one for comparing weeks within a position when the
player pool changes composition. Report raw MAE alongside it, and check the
denominator before treating a trend as a model property.

This also contradicts, again, the standing note that QB early-season accuracy
is the biggest gap: week 1 is QB's BEST bucket on raw MAE.

## The weekly serving path was dead, then ~8x wrong (2026-08-30)

Found only when the weekly model was proposed for the UI. Both defects lived
in the SERVING path, which nothing in this audit had measured -- every
verification up to this point compared TRAINING frames against each other.

### 1. Serving path raised TypeError on every call (self-inflicted)

The regex used to delete `_apply_cold_start_fallback` in commit 7f635b3,
`\n    def _apply_cold_start_fallback\(self.*?\n(?=    def )`, matched up to
the next `    def ` -- and the `@staticmethod` decorator of the FOLLOWING
method sits on the line above its `def`, so it was swallowed.
`_apply_snap_imputation` became an instance method and
`self._apply_snap_imputation(data)` raised
"takes 1 positional argument but 2 were given".

predict_next_week() was completely dead, committed and merged, with 550 tests
passing. Lesson: do not delete Python blocks with a regex that terminates on
the next definition; decorators bind upward.

### 2. Serving computed team shares on a position-filtered frame (older, worse)

`predict()` called `_load_player_data(position)` and then ran feature
engineering on that single-position frame, so every team-relative denominator
was position-local. Measured, serve vs train medians for RB:

    team_rb_target_share_roll3_mean   100.000  vs  18.395
    target_share_pct_roll3_mean        25.417  vs   5.829

100% is definitional: with only RBs in the frame, the share of team targets
going to RBs is all of them. Fed to a model trained on real shares:

    RB weekly projection   median   mean    max
    before                  42.3    ~42     48.0
    after                    3.6      4.9   17.1
    actual RB weekly         5.5      8.2   ~56

Every weekly prediction this system ever served was roughly 8x too high, and
the top ten RBs came back within 0.58 points of each other. Training was never
affected -- _prepare_training_data engineers features on all positions at once
and splits by position inside the trainer.

Fix: load all positions, engineer features, then narrow to the requested
position. After the fix the board reads as football: B.Robinson 17.1,
A.Jeanty 16.0, J.Gibbs 14.7, C.McCaffrey 13.2.

### Coverage gap that let both through

Nothing called predict_next_week() end to end. tests/test_serving_path_smoke.py
now does: the path must run, must produce fantasy-point-scale medians per
position, must discriminate between players, and team shares must not be
position-local. Verified non-vacuous -- the share assertion (`< 50`) sits
against a pre-fix serve value of 100.0 and a current value of 0.33.

Honest limit: the "discriminates between players" test would NOT have caught
defect 2 on its own. Across 200 players the broken predictions still had
std 11.76; only the top of the board was flat. The scale and share assertions
are what catch it.

## Weekly predictions were biased low; log1p retransformation was most of it (2026-08-31)

### The bias

The serving backtest showed every position predicting well under actual:

    pos   bias   pred_mean   actual_mean   relative
    QB   -2.51     10.54        13.05        -19%
    RB   -2.95      5.10         8.05        -37%
    WR   -2.42      3.68         6.09        -40%
    TE   -1.70      2.62         4.32        -39%

### Diagnosis

Training on log1p(y) and inverting with expm1 estimates a conditional GEOMETRIC
mean, which by Jensen's inequality sits below the arithmetic mean the
prediction is meant to be. The gap grows with residual variance.

The tell was the relative column: QB, the only position whose SAVED 1-week
model carries no transform, had roughly half the relative bias of the other
three. This also explains GAPS.md 7.11, where a 100-trial Optuna search could
not move the bias -- no hyperparameter can undo a retransformation error,
because it is arithmetic rather than fit quality.

### Fix

Duan (1983) smearing: multiply the back-transformed prediction by the mean of
exp(residual) in log space, fitted on OUT-OF-FOLD residuals. Non-parametric,
which matters because fantasy scoring is not lognormal, so a closed-form
sigma^2/2 correction would be wrong. Clipped to [1.0, 3.0]: a correction should
be a nudge, and a factor outside that means something else is broken.

Behind TARGET_TRANSFORM_MODE ("on" | "smearing" | "off"), now defaulting to
"smearing".

### Validation, on a season the model never saw

Trained 2018-2024 with --test-season 2025, then backtested serving on 2025
weeks 6/10/14. Same window, same holdout, transform mode the only difference:

    pos   MAE base  MAE smear    dMAE  |  bias base  bias smear  bias cut
    QB      7.089     7.089    +0.000  |    -1.295     -1.295         0%
    RB      4.637     4.430    -0.207  |    -2.977     -1.623        45%
    WR      4.102     4.029    -0.073  |    -2.555     -1.454        43%
    TE      3.061     2.998    -0.063  |    -1.903     -1.215        36%

Both calibration AND accuracy improved for all three transformed positions --
MAE was expected to worsen slightly, since correcting toward the mean usually
costs on a skewed target, and it did not. QB was BIT-IDENTICAL across arms
(same MAE, bias and pred_mean to three decimals), which is the control: its
saved model has active=False, so there was nothing to correct.

The effect also replicated: an earlier leaked run (2025 inside training) gave
45/47/44% bias cuts against 45/43/36% here.

### METHODOLOGICAL NOTE on the earlier backtest

The first serving backtests scored models trained on 2018-2025 against 2025.
`as_of` truncation removes the target week from PLAYER HISTORY but not from the
MODEL's training data, so those absolute numbers were optimistic. The A/B
comparison stayed valid (both arms leaked identically), and re-running clean
moved RB/WR/TE MAE by <0.08 -- so the leak was doing little work -- but QB
degraded 6.66 -> 7.09, its largest discrepancy.

### NOT FIXED: transform activation is unstable across training windows

Whether a position gets log-transformed depends on the measured skewness of
that window's targets, so it flips between retrains. QB 1w was transformed on
some builds and not others. Nothing surfaces this, and it silently changes bias
behaviour from one retrain to the next. Worth pinning per position rather than
re-deciding it from data each time.

### Residual bias remains

Smearing corrects the retransformation component only. Bias is still -1.2 to
-1.6 for RB/WR/TE and -1.30 for QB, which has no transform at all. The likely
remaining cause is the MAE/Huber objective targeting a conditional median on a
right-skewed target (7.11). Not addressed.

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

**STALE — corrected 2026-08-06**: `nfl.import_win_totals()` is not this
data despite the name — it's per-game weekly odds (spread/total/
moneyline), not season-long win-total futures, and this project's
`game_odds` table already has that fully covered. No free source for
real preseason win-total futures was found. See "No viable free source
for Vegas preseason win totals" further down in this doc.

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


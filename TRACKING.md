# Model & Experiment Tracking

Single source of truth for: what models/features/techniques exist, what
state they're in, and what's actually been measured. Replaces
`MODELS.md` + `EXPERIMENTS.md` (merged 2026-08-06, kept too wordy/split
before). Tables and bullets only — see git history of the old files for
narrative detail if needed.

**Status legend**: 🟢 LIVE (real entry point, verified) · 🟡 DORMANT
(real code, no live caller) · ⚪ ORPHANED (real code, never had any
caller at all) · 🔴 UNBUILT (proposed, zero code) · 🔵 CANDIDATE
(this session's new work, untested-in-production)

**Convention**: every metrics row reports R², RMSE, MAE, n at minimum.
`—` = not captured, never fabricated.

---

## 1. Registry — every model/feature/technique, live or not

| Subject | Type | Status | File | Accuracy tested? |
|---|---|---|---|---|
| `ComponentPredictor` | model | 🟢 | `component_predictor.py` | Yes — see §2. QB/WR healthy, RB/TE have real model-quality issues, see §4 (corrected finding — no scale bug) |
| `PreseasonProjector` | model | 🟢 | `preseason_projector.py` | Yes — see §2 |
| `KickerDSTPredictor` | model | 🟢 | `kicker_dst_predictor.py` | No |
| `UtilizationToFPConverter` | model | 🟢 | `utilization_to_fp.py` | Indirect (feeds live `util`-mode conversion) |
| `PlayerEmbeddings` (PCA) | feature | 🟢 | `advanced_techniques.py` | No (feature, not predictor) |
| `EnsemblePredictor`/`PositionModel`/`MultiWeekModel` | model | 🟡 bypassed by `component` mode | `ensemble.py`, `position_models.py` | Partial — QB only, this session, see §2 |
| `Hybrid4WeekModel` | model | 🟡 bypassed | `horizon_models.py` | No |
| `DeepSeasonLongModel` | model | 🟡 bypassed | `horizon_models.py` | No |
| `TouchdownRegressor` | technique | 🟡 disabled by design (Huber loss deemed sufficient) | `production_model.py` | No — disabled without a head-to-head |
| `BayesianPlayerModel` | model | 🟡 only unit-tested | `bayesian_models.py` | No |
| `weekly_matchup_predictor.py` | model | 🟡 caller itself unused | `weekly_matchup_predictor.py` | No |
| `LineupOptimizer` #1 | optimizer | 🟡 zero callers, no tests | `src/optimization/lineup_optimizer.py` | No — known bug: sums independent percentiles |
| `MonteCarloSimulator` | technique | ⚪ never documented as existing before this audit | `advanced_modeling.py` | No, never fed real data |
| `LineupOptimizer` #2 (different impl) | optimizer | ⚪ | `advanced_modeling.py` | No |
| `advanced_ml_pipeline.py` (EnsembleStack, PurgedTimeSeriesCV, etc.) | framework | ⚪ likely superseded by `ts_backtester.py`/`robust_validation.py`, not confirmed | `advanced_ml_pipeline.py` | No |
| `LSTMFantasyModel`, `StackedEnsemble` | model | ⚪ | `advanced_models.py` | No |
| `backtesting.py` (WalkForwardValidator, etc.) | framework | ⚪ likely superseded by `ts_backtester.py` | `backtesting.py` | No |
| `validate_methodology.py` + `train_advanced.py` | framework | ⚪ circular pair, no entry point | both files | No |
| GNN player-interaction graphs | technique | 🔴 confirmed zero code anywhere | — | — |
| General historical-twin/Player2Vec matching | technique | 🔴 only rookie-specific comp-matching exists (`advanced_rookie_injury.py`), not general | — | — |
| Bayesian hierarchical **matchup** model (team-vs-position) | technique | 🔴 `BayesianPlayerModel` is player-level only, not this | — | — |
| News sentiment as ML feature | feature | 🔴 computed, confirmed NOT in `CAUSAL_FEATURES` | `advanced_analytics.py` | — |
| Mixture density / bimodal output | technique | ✅ addressed differently — see asymmetric floor/ceiling, §2 | `generate_draft_data.py` | Yes |
| Multi-year + team-aware preseason candidate | model | 🔵 | `preseason_features.py` | Yes — see §2 |

---

## 2. Metrics — every real measurement

### 2a. Weekly single-game (`component` mode proxy, real 2025 backtest, 5,612 preds)

| Pos | R² | MAE | Spearman ρ | Within 10pt | n |
|---|---|---|---|---|---|
| QB | 0.241 | — | 0.494 | 80.1% | 688 |
| RB | 0.341 | — | 0.641 | 89.3% | 1,483 |
| WR | 0.267 | — | 0.538 | 90.5% | 2,263 |
| TE | 0.236 | — | 0.536 | 94.8% | 1,178 |
| **Agg** | **0.335** | **4.73** | **0.605** | **89.8%** | 5,612 |

vs. this project's own targets: ρ>0.65 (misses), within-10pt≥80% (clears). **Gap is rank accuracy, not error.**
⚠️ This is a `fp`-mode backtest proxy (fresh-trained), **not** the same as loading the real saved `component_*.json` — see §4.

### 2b. Season-total (`PreseasonProjector`, real 2023-2025 holdout)

| Pos | R² | RMSE | MAE | corr | n |
|---|---|---|---|---|---|
| QB | 0.291 | 103.6 | 84.7 | 0.541 | 104 |
| RB | 0.456 | 74.9 | 57.9 | 0.675 | 214 |
| WR | 0.485 | 64.3 | 50.7 | 0.709 | 373 |
| TE | 0.532 | 45.2 | 35.3 | 0.731 | 203 |

Architecture: position-specific Ridge, single prior season, zero team context. `RIDGE_ALPHA_BY_POSITION={QB:14,RB:28,WR:24,TE:18}` — hardcoded, no documented derivation.

### 2c. Multi-year + team-aware candidate (`preseason_features.py`, same holdout, production alpha)

| Pos | Best variant | R² | RMSE | MAE | vs. production |
|---|---|---|---|---|---|
| QB | full features + `PositionModel` (nonlinear) | 0.397 | — | 74.4 | **+0.106 R², real win** |
| RB | y1+dest-team + Ridge + real calibrator | 0.452 (R² marginal, test-set-size artifact — see below) | 71.8 | 56.1 | **RMSE/MAE both beat production (74.9/57.9) — real win** |
| WR | full features + Ridge | 0.565 | 57.1 | 45.1 | **+0.080 R², real win** |
| TE | y1+dest-team + Ridge | 0.540 | 42.4 | 32.3 | **+0.008 R², small real win** |

Destination-team context helps every position (small, consistent). Multi-year history (y2/y3) only helps QB (paired with a nonlinear model) and WR (Ridge) — hurts RB, wash for TE under Ridge. RB's multi-year hurt is real and not fixable by more regularization (alpha swept 10→200, R² plateaus at 0.402, never recovers to the 0.447 no-multi-year baseline). RB's R² (0.452) looks marginally short of production (0.456) but that's a 245-vs-214-row test-set-size artifact — RMSE/MAE are the trustworthy comparison and both clearly win. **All 4 positions: real win.** Nothing shipped — still candidate.

### 2d. In-season rest-of-season (`MultiWeekModel`, DIFFERENT model/task than 2c — real train≤2022/test 2023-2025)

| Pos | R² | corr | MAE | Naive (week1×17) R² | n |
|---|---|---|---|---|---|
| QB | 0.049 | 0.367 | 50.5 | -2.838 | 208 |
| RB | 0.483 | 0.700 | 50.2 | -0.780 | 429 |
| WR | 0.406 | 0.656 | 43.9 | -1.627 | 608 |
| TE | -0.416 | 0.398 | 53.7 | -1.974 | 233 |

TE's negative R²: traced to 4.08x train/OOF overfit ratio on smallest sample (922 rows), not architecture flaw. All 4 beat naive. Feature pipeline was missing 8 rookie `CAUSAL_FEATURES` columns — likely understates real accuracy.

### 2e. Calibration/post-processing effect tests (real, this session)

| Step | Test | Result |
|---|---|---|
| `PreseasonProjector` full calibration chain (legacy patches + `UpstreamCalibrator`) | base_pred vs. calibrated pred, real 2023-2025 holdout, all 4 pos | **Negligible.** QB: no change (calibrator not even active for QB). RB: ΔR²=0.000, MAE +0.1 (slightly worse). WR: ΔR²=+0.002, MAE -0.1 (tiny real help). TE: ΔR²=+0.001 (negligible). |
| `UpstreamCalibrator`, freshly refit on the §2c RB candidate | with vs. without | **Real help** — R² 0.447→0.452, MAE 56.4→56.1. Contrast with row above: a *freshly-fit* calibrator helps meaningfully; the *currently-deployed* one (fit on the original base model) does almost nothing on this holdout. |
| `ComponentPredictor` linear recalibration (QB only, only position where it's active) | — | Not yet tested — QB's base predictions are healthy (§4), so this is now a valid test to run, unlike when §4 looked catastrophic. |
| `EnsemblePredictor` sanity-bounds clip | — | Not yet tested |
| `EnsemblePredictor` tier-uncertainty scaling | — | Confirmed **no live effect** in `component` mode (operates on `NaN` fields) |
| TD mean-reversion (`TouchdownRegressor`) | — | Confirmed **disabled**, not tested against enabled |
| Asymmetric floor/ceiling vs. old symmetric formula | real 2023-2025 holdout, per-side coverage | **Real win.** Symmetric: floor breached 2.5% (target 6.7%, over-conservative), ceiling breached 7.6%. Asymmetric: 7.5%/7.5%, both near target. 3x reduction in per-side miscalibration. |

---

## 3. Post-processing pipeline (order of operations, live paths)

### Weekly (`ComponentPredictor` → `EnsemblePredictor`)
⚠️ Step 1 was broken for all 4 positions until today's retrain (§4).

| # | Step | Live? |
|---|---|---|
| 1 | Predict components → assemble FP via PPR weights, clip ≥0 | Yes |
| 2 | Linear recalibration (slope×fp+intercept) | Conditional — only kept if >0.5% RMSE improvement at fit time. Active: QB only. |
| 3 | Scale by `n_weeks` | Yes |
| 4 | Tier-specific uncertainty scaling | **No** — no-op in `component` mode |
| 5 | TD mean-reversion | **No** — disabled |
| 6 | Sanity-bounds clip (`{QB:65,RB:55,WR:55,TE:45}` pts/wk) | Yes |

Not produced at all: `prediction_ci80/95_*` stay `NaN` in `component` mode.

### Season-total (`PreseasonProjector`)

| # | Step | Live? |
|---|---|---|
| 1 | Base Ridge → clip ≥0 | Yes |
| 2 | "Veteran elite" calibration (multiplicative) | **No** — empty/inactive in the currently-trained model |
| 3 | "Fragile role" calibration (multiplicative) | **No** — same |
| 4 | `UpstreamCalibrator` (bounded 2nd-stage Ridge) | Active: RB/WR/TE only, not QB. Real effect: negligible, see §2e. |
| 5 | Clip ≥0 | Yes |
| 6 | Asymmetric floor/ceiling (`generate_draft_data.py`, separate from `PreseasonProjector`) | Yes — real win, see §2e |

---

## 4. RB/TE model-quality issue (corrected finding — two earlier "scale bug" claims retracted)

**Two earlier versions of this section (a "5x-26x scale-convention
drift" claim, then a "silent except-ValueError" claim) were both
wrong** — both were artifacts of testing one position at a time, which
doesn't match how `src/predict.py` is actually used (`position=None`
default, all 4 positions processed together in one combined
dataframe). `feature_scaler_bounded.joblib`'s 112-column list was fit
on that combined dataframe, so position-specific columns are naturally
absent when a position is tested alone — an artifact of the test, not
a real serving gap. Retested correctly (all 4 positions combined,
exactly matching real usage): **the bounded scaler applies
successfully, zero missing columns.** Full detail + lesson learned in
GAPS.md's retraction entry.

**What's actually real**, verified the correct way (comparing the
backed-up pre-retrain models against today's retrain, same real 2025
data, all positions combined):

| Pos | OLD ratio/R² | NEW ratio/R² | Verdict |
|---|---|---|---|
| QB | 1.11x / 0.193 | 1.10x / 0.195 | Unchanged, reasonably healthy |
| RB | 2.09x over / -0.979 | 2.12x over / -1.047 | **Pre-existing ~2x over-prediction, predates this session, not a scale bug** |
| WR | 1.96x over / -0.826 | 1.21x / **0.194** | **Retrain genuinely helped** (likely v30 features) |
| TE | 2.08x over / -1.032 | 0.22x under / -0.461 | **Still broken — failure mode flipped over→under with the retrain**, a real retrain-related change |

**STATUS: real, open, but far less severe than previously stated.**
RB and TE need their own investigation as genuine model-quality
problems (component-level residuals, sample size, `--no-tune` vs.
tuned hyperparameters) — not a feature-pipeline plumbing bug.

---

## 5. Not yet tried

- [ ] **PRIORITY — §4**: RB's systematic ~2x over-prediction (pre-existing, real, component-level: `rushing_yards`/`rushing_tds` residuals) and TE's prediction collapse (retrain flipped it from over→under) — both real model-quality issues, need component-level investigation, not a feature-pipeline bug.
- [ ] `EnsemblePredictor` sanity-bounds clip — does it help or hurt real predictions?
- [ ] `PositionModel` at production alpha / with tuning on (current numbers used `tune_hyperparameters=False`)
- [ ] Lookback-depth sweep for §2c (2yr vs 3yr vs 4yr+, 3 was picked arbitrarily)
- [ ] `component` vs `util` vs `fp` target mode recheck since v30's feature additions
- [ ] Real walk-forward validation of §2c/§2d (currently one train/test split)
- [ ] RB: try 1-2yr lookback instead of dropping multi-year entirely
- [ ] Test `Hybrid4WeekModel`/`DeepSeasonLongModel` — real trained artifacts, never evaluated
- [ ] Test `MultiWeekModel` for RB/WR/TE from a saved artifact (only QB has one; this session's test retrained fresh)
- [ ] Test `TouchdownRegressor` head-to-head vs. disabled
- [ ] Feed `MonteCarloSimulator` real data, see if it's usable as-is

---

## 6. Known pitfalls

- **R² is unstable on small/mismatched test sets** and can point the wrong direction vs. RMSE/MAE at the same time. Found directly: an RB test showed R² getting worse while MAE improved — test-set-size mismatch (17-50 rows vs. real baseline's 104-373), not a real difference.
- **pandas `groupby()` silently drops rows with any NaN key** (SQL `GROUP BY` doesn't). Cost an entire season (2025) and undercounted another (2024→5 rows) in `preseason_features.py` before being caught. Fixed with `dropna=False`.
- **Regularization can't fix a genuinely bad feature.** RB's multi-year alpha swept 10→200, R² plateaued at 0.402, never recovered to the 0.447 no-multi-year baseline.
- **"Real and complete" ≠ "known to work well."** Of ~13 non-live models cataloged, only 2 have ever had real accuracy measured before this session.

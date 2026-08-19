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

Everything below except the asymmetric floor/ceiling was **removed
from the codebase on 2026-08-07** per explicit user decision, once the
real-data tests confirmed each step was either negligible, dead, or
disabled — kept here as the historical record of why.

| Step | Test | Result | Outcome |
|---|---|---|---|
| `PreseasonProjector` full calibration chain (legacy patches + `UpstreamCalibrator`) | base_pred vs. calibrated pred, real 2023-2025 holdout, all 4 pos | **Negligible.** QB: no change (calibrator not even active for QB). RB: ΔR²=0.000, MAE +0.1 (slightly worse). WR: ΔR²=+0.002, MAE -0.1 (tiny real help). TE: ΔR²=+0.001 (negligible). | **Removed.** `preseason_projector.py` rewritten to a single fixed Ridge per position (`position_specific_ridge` feature set) — `UpstreamCalibrator`, `VeteranEliteCalibration`, `FragileRoleCalibration`, and the whole multi-variant selection/draft-sim-gate framework deleted. `confidence_score`/`support_class` kept (feed the floor/ceiling formula below). |
| `UpstreamCalibrator`, freshly refit on the §2c RB candidate | with vs. without | **Real help** — R² 0.447→0.452, MAE 56.4→56.1. Contrast with row above: a *freshly-fit* calibrator helps meaningfully; the *currently-deployed* one (fit on the original base model) does almost nothing on this holdout. | **Removed** along with the rest of the calibrator machinery — the freshly-fit-vs-deployed gap wasn't worth the ~1000 lines of variant-selection code required to keep re-deriving it live. |
| `ComponentPredictor` linear recalibration (QB only, only position where it's active) | raw vs. calibrated, real serving data, post-injury-fix retrain | **Real, modest help.** QB: slope=0.919, intercept=+2.05 — mean 11.05→12.20 (+1.1 avg, up to +1.9), moving predictions closer to the ~14.1 real benchmark. | **Removed** per explicit user request (`_maybe_fit_calibration`/`_target_fantasy_points`/`self.calibration` deleted from `component_predictor.py`). Live QB predictions dropped back ~1pt (median 9.97→9.18) as a direct, expected consequence — traded away knowingly, not a bug. |
| `EnsemblePredictor` sanity-bounds clip | — | Not yet tested | **Kept** — not part of "calibration," it's a hard safety clip on the final output, out of scope for this removal. |
| `EnsemblePredictor` tier-uncertainty scaling | — | Confirmed **no live effect** in `component` mode (operates on `NaN` fields) | **Removed** (`_apply_tier_uncertainty`, `TIER_UNCERTAINTY_MULTIPLIERS`, `_get_utilization_tier` deleted from `ensemble.py`) — dead code, no measured effect to lose. |
| TD mean-reversion (`TouchdownRegressor`) | — | Confirmed **disabled**, not tested against enabled | **Removed** (`TouchdownRegressor` class deleted from `production_model.py`, its already-commented-out call site in `ensemble.py` deleted, `validate_methodology.py`'s check for it deleted) — was already inert. |
| Asymmetric floor/ceiling vs. old symmetric formula | real 2023-2025 holdout, per-side coverage | **Real win.** Symmetric: floor breached 2.5% (target 6.7%, over-conservative), ceiling breached 7.6%. Asymmetric: 7.5%/7.5%, both near target. 3x reduction in per-side miscalibration. | **Kept** — the only step (`scripts/generate_draft_data.py`) with a measured real accuracy win. |

### 2f. Target-mode comparison (fp vs util vs component, v30 features, 2025 holdout, real)

First real apples-to-apples 3-way, 4-position comparison — prior record
(`data/models/qb_target_choice.json`) was QB-only and predates
`FEATURE_VERSION=30` and the 2026-08-08 bug fixes. Ran
`scripts/run_ts_backtest.py --season 2025 --target-mode {fp,util,component}`
(same harness/season/model as §2a, ridge, all defaults) — all 3 runs
produced identical n=5,612 (same eligible-player population, differ only
in target mode).

| Pos | fp R² / MAE | util R² / MAE | component R² / MAE |
|---|---|---|---|
| QB | 0.244 / 6.44 | 0.184 / 6.87 | 0.242 / 6.45 |
| RB | 0.380 / 4.76 | -0.046 / 6.64 | 0.380 / 4.76 |
| WR | 0.283 / 4.69 | 0.006 / 5.65 | 0.285 / 4.66 |
| TE | 0.279 / 3.82 | 0.011 / 4.49 | 0.280 / 3.81 |
| **Agg** | **0.358 / 4.73** | **0.096 / 5.81** | **0.359 / 4.72** |

**`util` is clearly worse across every position** (agg R² 0.096 vs.
~0.358, RB even goes negative) — predicting a utilization score and
converting to FP loses real signal versus predicting points-relevant
targets directly. **`fp` and `component` are statistically tied**
(agg R² 0.358 vs. 0.359, MAE 4.73 vs. 4.72, per-position deltas all
≤0.002 R²) — confirms production's current all-`component` config
(`config/settings.py`'s `MODEL_CONFIG["position_target_type"]`) is a
real, validated choice under the current feature set, not a stale
default. **No config change warranted** — `component` retains the
inherent advantage of also producing the individual stat-line
breakdown (pass_yds, rush_tds, etc.) that `fp` mode can't, at no
accuracy cost.

fp-mode run also served as this task's sanity check against §2a: n=5,612
matched exactly, MAE 4.73 matched exactly, agg R² 0.358 vs. §2a's 0.335
(small real drift, consistent with the 2026-08-08 data fixes landing
since §2a was measured — not a tooling regression).

### 2g. Real walk-forward validation of §2c (season-total, expanding window, `scripts/walk_forward_preseason.py`)

§2b/§2c were a single pooled 2023-2025 holdout — exactly the kind of
one-shot split §6 warns is unstable on small samples. Reran as a real
expanding-window walk-forward: for each test season, fit fresh on every
season strictly before it, score on that season alone, repeat. First
pass used every season back to 2007 (`--min-train-seasons 3`) and
surfaced the pitfall directly — 2011/2012 folds (trained on only
2006-2010, thin/pre-modern-feature data) scored R²=-0.61 and -2.55,
wildly unstable. Restricted to the modern-feature era relevant to
production today (test seasons 2019-2025, 7 folds) for the numbers
below. **Candidate here is a single Ridge-on-full-features fit per
position** (the §2c WR/TE winning architecture) — it does not
reproduce QB's nonlinear `PositionModel` or RB's `UpstreamCalibrator`
variants, which were never captured in a reusable script; out of scope
for this pass.

| Pos | production R² (mean±std) | candidate R² (mean±std) | production MAE | candidate MAE | n_folds |
|---|---|---|---|---|---|
| QB | 0.319±0.105 | 0.287±0.162 | 79.4 | 77.7 | 7 |
| RB | 0.446±0.05 | 0.405±0.11 | 55.9 | 55.5 | 7 |
| WR | 0.488±0.06 | 0.509±0.06 | 50.6 | 48.5 | 7 |
| TE | 0.508±0.049 | 0.512±0.07 | 37.6 | 34.0 | 7 |

**Production numbers hold up well under real walk-forward** — closely
match §2b's single-split figures (e.g. RB 0.446 vs. 0.456, WR 0.488 vs.
0.485), confirming §2b wasn't a lucky split. **WR/TE candidate wins are
real but smaller than originally reported** (WR +0.021 R² here vs.
+0.080 in §2c; TE +0.004 vs. +0.008) — still a consistent, real
(if modest) edge. **RB's original "win" mostly evaporates**: R² is now
clearly worse for the candidate (0.405 vs. 0.446), MAE is only
marginally better (55.5 vs. 55.9) — this directly confirms §6's
pitfall warning that the original RB number was a test-set-size
artifact, not a real win. **QB's Ridge-only candidate underperforms
production** (0.287 vs. 0.319) — expected, since it's missing the
nonlinear `PositionModel` architecture §2c's QB win actually depended
on; this run doesn't test that variant, so it neither confirms nor
refutes the original QB claim. Full per-fold data in
`data/backtest_results/walk_forward_preseason_20260809_031743.json`.

### 2h. Real walk-forward validation of §2d (in-season rest-of-season, expanding window, `scripts/walk_forward_multiweek.py`)

§2d was a single train≤2022/test-2023-2025-pooled split. Reran as a
real walk-forward (test seasons 2021-2025, 5 folds each), training
`MultiWeekModel` fresh per fold via the same target/feature helpers as
`train_position_models.py` (never calling `.save()` — that would have
overwritten the live production `.joblib` artifacts; verified via
before/after md5 that no production files were touched).

| Pos | 1w R² (mean±std) | 4w R² (mean±std) | 18w R² (mean±std) |
|---|---|---|---|
| QB | 0.240±0.021 | 0.848±0.011 | 0.970±0.009 |
| RB | 0.209±0.138 | 0.559±0.054 | 0.199±0.038 |
| WR | 0.163±0.017 | 0.485±0.013 | 0.156±0.051 |
| TE | 0.160±0.012 | 0.506±0.021 | 0.229±0.033 |

**Found a real, separate issue while building this**: the 4w/18w
targets are variable-length rolling sums (`min_periods=1`) that shrink
near season-end, so target *magnitude* correlates with in-season
timing almost mechanically (`corr(week, target_18w)=0.576` for 2025
QBs) — any feature proxying "games remaining" can predict that
magnitude without real forecasting skill. QB's 18w R²=0.970 is likely
inflated by exactly this (full mechanism logged in GAPS.md's
2026-08-09 entry). This doesn't fully explain the picture on its own —
RB/WR/TE's 18w R² (0.16-0.23) is much lower than QB's despite the same
target construction applying to all four — but it's reason enough to
**not trust 4w/18w R² at face value**. **Only the 1w horizon is
unaffected** (fixed window of exactly 1, always fully populated) and
its numbers (0.16-0.24) are in a plausible, stable range consistent
with §2a's weekly figures. §2d's original single-number-per-position
table (QB R²=0.049, RB 0.483, WR 0.406, TE -0.416) can't be reconciled
against any of the three horizons above — the original ad-hoc script
was never saved, so which horizon/target definition it used is
unknown. Treat §2d's original numbers as superseded by this table, and
treat this table's 4w/18w columns as directional only until the target
construction is fixed (tracked in GAPS.md, not yet fixed — needs a
design decision, not a mechanical patch). Full per-fold data in
`data/backtest_results/walk_forward_multiweek_20260809_091323.json`.

---

## 3. Post-processing pipeline (order of operations, live paths)

**2026-08-07: simplified.** Every calibration/scaling step below except
the asymmetric floor/ceiling was removed from the code (see §2e for
why, per-step). These tables now describe the current, much shorter
pipelines.

### Weekly (`ComponentPredictor` → `EnsemblePredictor`)

| # | Step | Live? |
|---|---|---|
| 1 | Predict components → assemble FP via PPR weights, clip ≥0 | Yes |
| 2 | Scale by `n_weeks` | Yes |
| 3 | Sanity-bounds clip (`{QB:65,RB:55,WR:55,TE:45}` pts/wk) | Yes |

Not produced at all: `prediction_ci80/95_*` stay `NaN` in `component` mode.

### Season-total (`PreseasonProjector`)

| # | Step | Live? |
|---|---|---|
| 1 | Base Ridge (fixed `position_specific_ridge` feature set, no variant selection) → clip ≥0 | Yes |
| 2 | Asymmetric floor/ceiling (`generate_draft_data.py`, separate from `PreseasonProjector`) | Yes — real win, see §2e |

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

**Follow-up — fixed one real bug, found the symptom picture doesn't
match this table (2026-08-06):** calling the real `NFLPredictor` class
directly (`predictor.initialize(); predictor.predict(n_weeks=1,
top_n=2000)` — the actual `scripts/generate_app_data.py` call pattern)
surfaced and fixed a genuine bug: `_add_team_matchup_features` ran
twice on the same dataframe (`create_features()` then
`refresh_matchup_features()`), the second call's merges silently
collided with columns from the first call, and `offensive_momentum_score`
raised a caught `KeyError` on every real call, dropping all team-matchup
features for that call. Fixed by dropping stale columns before
re-merging (GAPS.md 2026-08-06 entry). Confirmed pre-existing since
2026-08-04, not a regression from this session's earlier caching work.

That same real call's per-position prediction stats **don't match the
OLD/NEW table above**: QB mean 4.30 (real ~14.1, badly under), RB mean
8.45 (real ~8.7, roughly correct), WR mean 3.36 (real ~7.9, under), TE
mean 0.25/median 0.00 (real ~6.4, majority predicted zero — worse than
the table shows). The momentum-score fix didn't change these numbers.
**Not yet reconciled**: the RB-over/TE-under pattern above came from a
manually-reconstructed pipeline call (proven unreliable twice already
this session — see the retraction above); this new QB/WR-under,
TE-collapsed pattern comes from the real class and supersedes it, but
the mechanistic explanation (dominant rookie-feature coefficients) found
earlier hasn't been re-derived against this real data yet. Next step:
investigate using ONLY the real `NFLPredictor` class output from here on.

**Root cause found and fixed (2026-08-07)**: the QB/WR under-prediction
and TE collapse were driven by `_infer_bounded_columns`
(`feature_preparation.py`) auto-selecting `injury_prob_advanced`/
`injury_prob_combined` — genuine, already-calibrated probabilities
capped at 0.25 by design — for MinMax rescaling meant for raw 0-100
percentage columns. That stretched a real 0.10-0.25 range to fill all of
`[0,1]`, and `predict.py`'s `predicted_points *= (1 - injury_prob_combined)`
availability gate then crushed nearly every prediction by 45-100%. Fixed
by removing `"prob"`/`"probability"` from the bounded-column token list;
also caught `rookie_breakout_prob`/`rookie_bust_prob` (same bug, smaller
effect since they're Ridge *inputs* not a direct output multiplier). Full
retrain done for train/serve consistency on those two features. Full
root-cause writeup + verification in GAPS.md's 2026-08-07 entries.

| Pos | predicted_points BEFORE fix | AFTER fix + retrain | Real typical |
|---|---|---|---|
| QB | mean 4.30 / median 4.35 | mean 10.06 / median 9.97 | ~14.1 |
| RB | mean 8.45 / median 9.00 | mean 13.81 / median 13.81 | ~8.7 |
| WR | mean 3.36 / median 3.51 | mean 6.76 / median 6.73 | ~7.9 |
| TE | mean 0.25 / median 0.00 | mean 0.69 / median 0.00 | ~6.4 |

`injury_prob_combined` now correctly bounded live (mean 0.185, max
0.25, was mean 0.554, max 1.0). Top-player predictions post-fix look
realistic (QB1s 14-17, RB1s 19-26, TE1s 4.5-6.5). **Still open**: TE
median is still exactly 0.00 (n=85/147 are likely legitimate backup
TEs, not re-verified against a real starters-only baseline), and RB
now runs a bit hot (13.81 vs ~8.7) — possibly the same pre-existing RB
over-prediction issue re-surfacing now that the injury-crush that was
masking it is gone, possibly a benchmark-population mismatch (top RBs
vs. all RBs). Needs a real, apples-to-apples (matched player
population) accuracy pass, not just aggregate means — tracked in §5.

**RESOLVED — real apples-to-apples check (2026-08-07, `scripts/validate_rb_te_accuracy.py`):**
Called the real `NFLPredictor` class (same `initialize()`+`predict(n_weeks=1,
top_n=2000)` pattern as `generate_app_data.py`, no retraining) against 8 past
2025 weeks (10-17), joined every prediction 1:1 to the real `fantasy_points`
actual for that exact player_id/season/week (2,361 matched rows), and split
by a snap-share-verified starter cut (`utilization_scores.snap_share >= 50`,
on its native 0-100 scale — see GAPS.md's snap_share data-bug entry for why
`player_weekly_stats.snap_share` could not be used for this).

| Pos | Cut | n | MAE | RMSE | R² | bias (pred/actual) |
|---|---|---|---|---|---|---|
| RB | full population | 618 | 8.42 | 9.45 | -0.254 | 1.73 |
| RB | starters-only | 204 | 7.18 | 8.79 | 0.017 | **1.20** |
| TE | full population | 510 | 5.08 | 7.34 | -0.425 | 0.19 |
| TE | starters-only | 301 | 6.37 | 8.73 | -0.618 | **0.22** |

**RB "runs hot" — mostly a population-mismatch artifact, not a new bug.**
Bias drops from 1.73x (full eligible pool, including many low-snap
players) to 1.20x among real starters — a real but modest over-prediction,
consistent with (smaller than) the pre-existing ~2x RB over-prediction
documented earlier in this section. Not fully closed, but far less severe
than the raw aggregate-mean comparison suggested.

**TE zero-median — CONFIRMED REAL BUG, not legitimate backups.** Among
TE predictions `< 0.5` (n=274), actual `fantasy_points` median is 2.55
and 43.8% actually scored >3.0 points; 117 of those 274 are snap-share-
verified starters (`util_snap_share >= 50`). Spot-checked individual rows
confirm this is severe, not noise: Trey McBride predicted 6.76 / actual
37.4 (wk15), George Kittle predicted 2.77 / actual 24.7 (wk11), Kyle
Pitts predicted 4.59 / actual 45.6 (wk15). TE starters-only bias (0.22x)
is essentially unchanged from the full-population bias (0.19x) — this is
not a population artifact, it's a genuine live prediction defect
affecting real starting tight ends. Root cause not yet investigated —
tracked as a new, sharper item in §5 (was previously a hedge, now a
confirmed bug).

**FULLY RESOLVED (2026-08-08)**: root-caused to two compounding bugs
(full mechanism and code changes in GAPS.md's 2026-08-08 entries):

- **Bug #1**: `pbp_stats_aggregator.py`'s `merge_with_snaps()` inflated
  `player_weekly_stats.team_snaps` ~10x for 2025 by summing every
  player's individual snap credit instead of the team's real play
  count. Fixed by deriving `team_snaps` from the median of each
  player's implied team total (`offense_snaps/offense_pct`, nflverse's
  own verified per-player share), falling back to `max(offense_snaps)`.
- **Bug #3** (found while re-auditing before re-ingesting, per explicit
  request to be skeptical of more bugs in this area — much larger than
  Bug #1): `UtilizationScoreCalculator._calculate_team_totals_from_players()`
  (`utilization_score.py`) recomputed `team_snaps`/`team_rush_attempts`
  via groupby-sum and merged with no `suffixes=` kwarg. Since both
  columns already existed as true team-level values on the input `df`
  (from `player_weekly_stats`/`team_stats` joins), the merge silently
  split them into `team_snaps_x`/`team_snaps_y`, and every per-position
  share formula's `df.get("team_snaps", snap_count)` fallback then
  silently self-divided (`snap_share_pct = snap_count/snap_count = 100%`)
  — a degenerate bimodal 0/100 pattern present in **both training and
  serving, for every position and every season back to 2018**, not
  2025-specific. `team_rush_attempts` collided the same way, breaking
  RB's `rush_share_pct`/`touch_share_pct` too — a second, previously
  unflagged instance of the same bug, and a plausible contributor to
  the pre-existing RB over-prediction pattern documented earlier in
  this section. Fixed by preserving pre-existing `team_snaps`/
  `team_rush_attempts` instead of shadowing them, with `max()` (not
  `sum()`) as the fallback aggregator for snap-like columns.

Re-ingested 2025 data, retrained (`--fast --no-tune`), refreshed the
`utilization_scores` DB table, and re-ran `validate_rb_te_accuracy.py`
(2025 weeks 10-17, all 4 positions, 2,361 matched rows):

| Pos | Cut | n | MAE | RMSE | R² | bias (pred/actual) | vs. pre-fix bias |
|---|---|---|---|---|---|---|---|
| RB | full | 566 | 4.48 | 6.65 | 0.344 | 0.83 | was 1.73 |
| RB | starters-only | 191 | 6.77 | 9.40 | -0.189 | 0.68 | was 1.20 |
| WR | full | 884 | 4.52 | 6.42 | 0.213 | 0.82 | was 1.06 |
| TE | full | 495 | 3.57 | 5.26 | 0.280 | 0.84 | was 0.19 |
| TE | starters-only | 293 | 4.29 | 6.22 | 0.190 | 0.78 | was 0.22 |

TE near-zero predictions (`<0.5`) collapsed from n=274 to **n=3**. TE
and RB full-population R² both went from negative to positive. This is
the real fix, not just a milder version of the earlier
population-mismatch explanation — both root causes are now confirmed
and closed.

**New regression surfaced, not caused by this fix**: QB metrics got
*worse* (full bias 0.68 vs. previous 0.85, R² -0.108 vs. 0.163;
starters-only bias 0.10, R² -1.75). Root-caused to a **separate,
pre-existing `players` table bug**: at least 13 real WR/RB/TE players
(D.Moore, D.Singletary, C.Kmet, J.Waddle, C.Olave, B.Hall, and others —
confirmed via `passing_attempts<5` combined with real `rushing_attempts`/
`targets` volume) are mislabeled `position='QB'` in the `players` table.
The model correctly predicts near-zero passing production for them
(since they never pass), while their real `fantasy_points` reflects
genuine receiving/rushing production — a severe, spurious
under-prediction that drags down the QB slice's aggregate metrics. This
predates this session and is unrelated to Bugs #1/#3; it's surfaced now
because the QB predictions are noisier this run, and because it directly
inflates the "starters-only" QB count (56 vs. the previous run's 6,
since these mislabeled skill players legitimately clear a snap-share
starter threshold under a QB label). Logged as a new item in §5; not
fixed in this session (needs its own investigation into the `players`
table position-assignment/roster-sync logic in `nfl_data_loader.py`).

**Accepted, documented limitations** (not blocking, see GAPS.md
2026-08-08 entries for full detail): `team_targets`/`team_receptions`
(no pre-existing DB source) can still under-count when the input
population is filtered (`min_games`, `filter_to_eligible_players`, or a
position-scoped `predict(position=...)` call) — a real but much
lower-severity issue than the collision bug. The redundant double-call
to `calculate_all_scores()` in `feature_preparation.py` (lines 330/337)
was left in place — with the collision fixed, both calls now produce
identical, stable results, so it's a harmless inefficiency rather than
a correctness bug; removing it is a separate future cleanup.

**QB regression FIXED (2026-08-08, same session)**: root-caused and
fixed the mislabeled-QB bug (full mechanism in GAPS.md). Fixed both the
code (`src/data/pbp_stats_aggregator.py`) and the 30 already-corrupted
`players` table rows. Verified live (no retrain needed — `predict.py`
reads `position` straight from the DB): Derrick Henry, Christian
McCaffrey, Cole Kmet, Jaylen Waddle, and Cooper Kupp all now correctly
route to their real position's component model with plausible
predictions. Re-ran `validate_rb_te_accuracy.py`: QB full-population
bias 0.68→**0.82**, R² -0.108→**0.134** (recovered to essentially the
pre-regression baseline of 0.85/0.163). "QB starters-only" now
correctly shows zero rows — real QBs never compute `snap_share_pct` at
all (`utilization_score.py`'s QB branch doesn't set it), so the
previous run's 56 "QB starters" were *exactly* the misclassified
skill players (who do have a real snap share) being counted under the
wrong label; their removal from the QB slice is the clean, mechanistic
confirmation that the fix addressed the actual cause, not a
coincidental improvement.

---

## 5. Not yet tried

- [x] ~~real, apples-to-apples RB/TE accuracy check post-injury-fix~~ — done 2026-08-07, see §4. RB "hot" was mostly population mismatch (1.20x bias among real starters, not 1.73x). TE zero-median is a **confirmed real bug**, not legitimate backups.
- [x] ~~root-cause TE's near-zero live predictions~~ — done 2026-08-07 (same session), see GAPS.md. TE's `receiving_yards` component model's top feature (`snap_share_pct_roll3_mean`, coef=-1.0) is fed a ~10x-deflated value for every 2025 row, a direct downstream effect of the `team_snaps` bug below.
- [x] ~~fix + verify `pbp_stats_aggregator.py`'s team_snaps inflation~~ — done 2026-08-08. Fixed, re-ingested, retrained, verified.
- [x] ~~re-run `scripts/validate_rb_te_accuracy.py` post-fix~~ — done 2026-08-08, see §4. TE bias 0.19→0.84, RB bias 1.73→0.83, both R² negative→positive.
- [x] ~~root-cause + fix Bug #3 (`utilization_score.py` team-total collision)~~ — done 2026-08-08, see §4 and GAPS.md. Was silently degenerate for every position/season since this code existed, not just 2025.
- [x] ~~mislabeled QB players~~ — fixed 2026-08-08, see GAPS.md. Root-caused, fixed at the code level (`pbp_stats_aggregator.py`), and corrected 30 already-corrupted `players` rows directly. QB metrics recovered fully post-fix (bias 0.68→0.82, R² -0.108→0.134).
- [ ] Lower priority: `team_targets`/`team_receptions` under-count when input population is filtered (`min_games`, `filter_to_eligible_players`, position-scoped queries) — accepted limitation, see GAPS.md 2026-08-08 entry.
- [ ] Lower priority: redundant double-call to `calculate_all_scores()` in `feature_preparation.py` (lines 330/337) — harmless now that the collision is fixed (both calls produce identical output), but still wasteful; safe cleanup for later.
- [ ] `EnsemblePredictor` sanity-bounds clip — does it help or hurt real predictions?
- [ ] `PositionModel` at production alpha / with tuning on (current numbers used `tune_hyperparameters=False`)
- [ ] Lookback-depth sweep for §2c (2yr vs 3yr vs 4yr+, 3 was picked arbitrarily)
- [x] ~~`component` vs `util` vs `fp` target mode recheck since v30's feature additions~~ — done 2026-08-09, see §2f. `util` is clearly worse (agg R² 0.096 vs. ~0.358); `fp`/`component` are statistically tied. Production's all-`component` config confirmed correct, no change made.
- [x] ~~Real walk-forward validation of §2c/§2d (currently one train/test split)~~ — done 2026-08-09, see §2g and §2h.
- [ ] RB: try 1-2yr lookback instead of dropping multi-year entirely
- [ ] Test `Hybrid4WeekModel`/`DeepSeasonLongModel` — real trained artifacts, never evaluated
- [ ] Test `MultiWeekModel` for RB/WR/TE from a saved artifact (only QB has one; this session's test retrained fresh)
- [ ] Test `TouchdownRegressor` head-to-head vs. disabled
- [ ] Feed `MonteCarloSimulator` real data, see if it's usable as-is
- [ ] Joint sweep of training-years-included × `recency_decay_halflife` (config/settings.py:217-221, currently 1.5 seasons, horizon-aware 1.5/2.5/3.5 for 1w/4w/18w) — more years of history plus a shorter half-life could behave differently than fewer years at the current half-life; not yet tested as a joint grid.

---

## 6. Known pitfalls

- **R² is unstable on small/mismatched test sets** and can point the wrong direction vs. RMSE/MAE at the same time. Found directly: an RB test showed R² getting worse while MAE improved — test-set-size mismatch (17-50 rows vs. real baseline's 104-373), not a real difference.
- **pandas `groupby()` silently drops rows with any NaN key** (SQL `GROUP BY` doesn't). Cost an entire season (2025) and undercounted another (2024→5 rows) in `preseason_features.py` before being caught. Fixed with `dropna=False`.
- **Regularization can't fix a genuinely bad feature.** RB's multi-year alpha swept 10→200, R² plateaued at 0.402, never recovered to the 0.447 no-multi-year baseline.
- **"Real and complete" ≠ "known to work well."** Of ~13 non-live models cataloged, only 2 have ever had real accuracy measured before this session.

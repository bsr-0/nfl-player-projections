# Model & Strategy Inventory

Full catalog of every prediction model, optimizer, and simulation
strategy that exists in this codebase, and — critically — whether it
actually runs in production or is dormant/orphaned code that looks
current but isn't. Built 2026-08-06 per explicit user request, after
repeatedly finding real models sitting unused and undocumented this
session (`MultiWeekModel`, `LineupOptimizer`, and — found while
building this doc — several more). Keep this current: whenever a model
or strategy is added, moved, wired in, or retired, update its row here.

**How "LIVE" was verified, not assumed**: for each entry, traced actual
import/call chains from a real, documented entry point (`README.md`'s
documented commands, or a script another live script/README calls) —
not just "a test file imports it" or "it has a docstring." Verified
`git diff`-clean against the last commit before writing this, to
confirm nothing below was changed while investigating.

## Verification: nothing promoted to production this session

- `config/settings.py`: `position_target_type` still
  `{"QB": "component", "RB": "component", "WR": "component", "TE": "component"}`
  — unchanged.
- `scripts/generate_draft_data.py`: `_resolve_projection()` still calls
  the original `PreseasonProjector`, not the new
  `build_multiyear_season_pairs` candidate.
- `src/models/preseason_features.py` (today's new candidate code): zero
  callers anywhere outside its own test file — confirmed via grep.
- No diff in the last commit touching any of `generate_draft_data.py`,
  `preseason_projector.py`, `component_predictor.py`, `train.py`, or
  `config/settings.py`.

---

## LIVE (actually serves real predictions today)

| Model / Strategy | File | Entry point | What it does |
|---|---|---|---|
| `ComponentPredictor` | `src/models/component_predictor.py` | `scripts/generate_app_data.py`, `src/models/train.py` (via `position_target_type="component"`) | Predicts individual stat components (pass yards, rush TDs, etc.) per position, assembles fantasy points via PPR weights. **The real weekly prediction architecture** — confirmed real 2025 accuracy in `EXPERIMENTS.md` §1a. |
| `PreseasonProjector` | `src/models/preseason_projector.py` | `scripts/generate_draft_data.py`'s `_resolve_projection()` | Season-total draft-board projection from single-prior-season stats. **Drives the live draft board today.** Real accuracy in `EXPERIMENTS.md` §1b. Confirmed gaps (single-season only, zero team context) motivated today's candidate work. |
| `KickerDSTPredictor` | `src/models/kicker_dst_predictor.py` | `scripts/generate_app_data.py` (`README.md`-documented: `python scripts/generate_app_data.py`) | Kicker and DST predictions — a separate pipeline from the QB/RB/WR/TE skill-position models covered everywhere else in this doc. |
| `UtilizationToFPConverter` | `src/models/utilization_to_fp.py` | `train.py`, `ensemble.py`, `ts_backtester.py`, `feature_preparation.py` | Converts utilization-score predictions to fantasy points. Live infrastructure — used whenever `util`-mode conversion is exercised, including by the (currently non-default) `util` target mode. |
| `PlayerEmbeddings` | `src/models/advanced_techniques.py` | `feature_preparation.py`'s `_prepare_training_data()` | PCA-based player embeddings, a real feature (not a standalone predictor) computed during the full production training pipeline. |
| `robust_validation.py` utilities | `src/models/robust_validation.py` | `train.py`, `train_advanced.py` | Validation utilities used during real training. |
| `data_loading.py` / `data_quality.py` | `src/models/` | `train.py`, `src/evaluation/*` | Live data-loading/quality-gate infrastructure, not standalone models. |

---

## Post-processing steps applied after raw predictions (2026-08-06)

Order matters — each row runs after the one above it.

### Weekly (`ComponentPredictor` → `EnsemblePredictor`, live)

| # | Step | Where | Live effect? |
|---|---|---|---|
| 1 | Predict components, assemble FP via PPR weights, clip ≥0 | `component_predictor.py::_predict_total_fp` | Yes |
| 2 | Linear recalibration (`slope × fp + intercept`) | same, `self.calibration` | Conditional — only kept if it beats raw RMSE by >0.5% at fit time; often a no-op |
| 3 | Scale by `n_weeks` | `ensemble.py::predict` | Yes |
| 4 | Tier-specific uncertainty scaling | `ensemble.py::_apply_tier_uncertainty` | **No** — only touches `prediction_std`/CI fields, which stay `NaN` in `component` mode. Live only for the dormant `position_models` path. |
| 5 | TD mean-reversion (`TouchdownRegressor`) | `ensemble.py::_apply_td_regression` | **No** — call site commented out |
| 6 | Sanity-bounds clip (`{QB:65, RB:55, WR:55, TE:45}` pts/wk, scaled by `n_weeks`) | `ensemble.py::predict` | Yes — final clamp |

**Not produced at all today**: confidence intervals (`prediction_ci80/95_*`) stay `NaN` for the live `component`-mode path.

### Season-total (`PreseasonProjector`, live — drives the draft board)

| # | Step | Where | Live effect? |
|---|---|---|---|
| 1 | Base position-specific Ridge → `base_pred`, clip ≥0 | `preseason_projector.py::predict_with_details` | Yes |
| 2 | "Veteran elite" calibration (multiplicative, hardcoded conditions) | same, `legacy_veteran_elite_calibration` | Yes, for matching rows |
| 3 | "Fragile role" calibration (multiplicative, hardcoded conditions) | same, `legacy_fragile_role_calibration` | Yes, for matching rows |
| 4 | `UpstreamCalibrator` — 2nd-stage Ridge on raw_pred + confidence/support features, bounded to a max % adjustment, damped for low confidence | same, `upstream_calibrators.calibrate` | Yes, if it beat train MAE/bias at fit time |
| 5 | Clip ≥0 | same | Yes |
| 6 | Asymmetric floor/ceiling (quantile-regression spread, this session's work) | `scripts/generate_draft_data.py::_floor_ceiling` | Yes — separate from `PreseasonProjector` itself |

---

## DORMANT (real, complete, often trained — but unreachable from any live entry point)

**Accuracy-tested column is the honest-reporting point of this table**:
most of this code has *never* had its real-world accuracy measured
against held-out data, at all, ever — "real and complete" is not the
same as "known to work well."

| Model / Strategy | File | Why dormant | Evidence it's real, not a stub | Accuracy tested? |
|---|---|---|---|---|
| `EnsemblePredictor` / `PositionModel` / `MultiWeekModel` | `src/models/ensemble.py`, `position_models.py` | Production hardcodes `position_target_type="component"` for every position; `ComponentPredictor` is checked first in `EnsemblePredictor.predict()`'s branch logic, so these are bypassed even when loaded. | `data/models/multiweek_qb.joblib` exists (85MB, real trained artifact, all 18 week-horizons, saved 2026-08-01) — but only for QB; RB/WR/TE never got a `multiweek_*.joblib`. | **Yes, partially** — real head-to-head test this session (`EXPERIMENTS.md` §3), QB only (RB/WR/TE artifacts don't exist so those were retrained fresh for the test, not evaluating the saved artifact). Mixed real results, RB clearly beats naive baseline, TE overfits on small sample. |
| `Hybrid4WeekModel` | `src/models/horizon_models.py` | Same `component`-mode bypass. | `data/models/hybrid_4w_{qb,rb,wr,te}.joblib` — real trained artifacts, all 4 positions, LSTM+ARIMA, saved 2026-08-01. | **No.** Zero mentions anywhere in GAPS.md before this document. Never evaluated against real held-out data. |
| `DeepSeasonLongModel` | `src/models/horizon_models.py` | Same `component`-mode bypass. | `data/models/deep_18w_{qb,rb,wr,te}` — real trained artifacts, all 4 positions, residual feedforward net, saved 2026-08-03. | **No.** Same as above — trained, saved, never evaluated. |
| `TouchdownRegressor` | `src/models/production_model.py` | Its only call site, `EnsemblePredictor._apply_td_regression()`, is explicitly commented out in `ensemble.py` (`# results = self._apply_td_regression(...)`) with a rationale comment: Huber loss already provides outlier robustness, so this was deliberately disabled, not forgotten — but the class and its logic remain, real and unused. | Real implementation, opportunity-based expected-TD mean reversion. | **No.** Disabled before/without a real accuracy comparison against the Huber-loss-only baseline that replaced it — the rationale comment is a design argument, not a measurement. |
| `BayesianPlayerModel` | `src/models/bayesian_models.py` | Zero references anywhere except `tests/test_rookie_projections.py`. Never imported by any production or training script. | 721-line file, real implementation (full Bayesian + simplified variants). | **No.** Only exercised by unit tests checking it runs, not that it predicts well. |
| `weekly_matchup_predictor.py` | `src/models/weekly_matchup_predictor.py` | Only caller is `scripts/generate_weekly_projections.py`, which itself has zero references anywhere (not in README, not called by any other script) — dormant two levels deep. | 382 lines, real implementation. | **No.** |
| `LineupOptimizer` (#1) | `src/optimization/lineup_optimizer.py` | `optimize_lineup()` has no caller anywhere in the codebase, no test coverage. Found and flagged earlier this session. | Salary-cap knapsack, cash/GPP strategies, real (if buggy — sums independent player percentiles for lineup floor/ceiling, a real statistical error found and not yet fixed). | **No** — no test coverage at all, so not even confirmed to run correctly, let alone accurately. |

---

## ORPHANED (no live caller, no dormant-but-intentional caller either — appear to be abandoned prototypes, found while building this doc)

These weren't previously documented as existing at all. Substantial,
real code (400-1400 lines each), zero references anywhere outside
themselves.

| File | Lines | What's in it | Notable overlap with documented gaps |
|---|---|---|---|
| `src/models/advanced_modeling.py` | 745 | `MonteCarloSimulator` (skew-normal outcome sampling, injury-probability mixture, matchup adjustment), a **second, different** `LineupOptimizer` (objective-based: expected/floor/ceiling/sharpe, supports superflex), `UserProfile`, `ModelComparisonFramework` | **GAPS.md §11.2.D ("Monte Carlo Game Simulation") is filed as "not attempted."** A real, working Monte Carlo simulator already exists — dormant, never fed real data, but not "not attempted" in the sense GAPS.md implies. Needs re-scoping: is this usable as-is, or does it need rework? |
| `src/models/advanced_ml_pipeline.py` | 1,388 | `EnsembleStack`, `PurgedTimeSeriesCV`, `SeasonHoldoutCV`, `AdvancedFeatureSelector`, `UncertaintyQuantifier`, `TargetEngineer`, `AdvancedFeatureEngineer`, `RobustnessChecker`, `ModelMonitor`, `AdvancedMLEvaluator` | Large validation/monitoring framework — likely superseded by what `ts_backtester.py` and `robust_validation.py` actually do live, but not confirmed identical; worth a real diff before assuming full duplication. |
| `src/models/advanced_models.py` | 758 | `TimeSeriesValidator`, `LSTMFantasyModel`, `StackedEnsemble`, `VegasLinesIntegration` | A standalone LSTM model, distinct from `horizon_models.py`'s `Hybrid4WeekModel` (also LSTM-based) — two separate LSTM implementations exist, neither live. |
| `src/models/backtesting.py` | 448 | `BacktestResult`, `WalkForwardValidator`, `SeasonBasedValidator`, `FantasyBacktester`, `ModelComparison` | Likely superseded by `src/evaluation/ts_backtester.py` (confirmed live, used throughout this session) — not confirmed identical, not deleted, just unused. |
| `src/models/validate_methodology.py` + `train_advanced.py` | 739 combined | Reference each other; `train_advanced.py`'s only external references are `GAPS.md` (documentation) and itself | Circular pair, no real entry point into either. |

**Not evaluated for correctness or quality — cataloged as existing, not
vetted.** Before reusing any of these (e.g. for the GAPS.md §11.2.D
Monte Carlo item), read the actual code and check it against current
data schemas — it may predate recent feature/schema changes and not
run as-is. **Accuracy tested: No, for all five entries** — none of
these have ever been run against real data, let alone measured for
accuracy; "orphaned" is a strictly weaker claim than "dormant" above
(dormant = built with an intended live path that got bypassed; orphaned
= no evidence a live path was ever wired at all).

---

## UNBUILT (proposed in GAPS.md §11.2, zero code exists anywhere — verified by direct search, not assumed from the "not attempted" label)

| Proposal | GAPS.md section | Real status |
|---|---|---|
| Player Interaction Graph Networks (GNN) | §11.2.A | **Confirmed zero code.** Searched for `GATv2`, `GraphAttention`, `GNN`, `graph_neural`, `PlayerInteractionGraph` anywhere in `src/` — no matches. Genuinely unbuilt, not just unwired. |
| Player Embeddings / Historical Twin Matching | §11.2.B | **Partially exists, easy to conflate with what doesn't.** `PlayerEmbeddings` (`src/models/advanced_techniques.py`, LIVE, PCA-based) is a *training feature*, not the "find the 5 most similar historical players and use their trajectories as a projection input" system §11.2.B actually describes. The real analog — comp-player matching by draft capital + combine similarity — exists but **only for rookies** (`src/features/advanced_rookie_injury.py`), not the general Player2Vec/Baller2Vec-style system for all players. The general version is unbuilt. |
| Bayesian Hierarchical Matchup Model | §11.2.F | **Partially exists, different scope than proposed.** `BayesianPlayerModel` (`src/models/bayesian_models.py`, DORMANT, see table above) does player-level random-effects shrinkage (James-Stein / MCMC) — real, but it is not the team-vs-position matchup model §11.2.F describes (simultaneous team offensive strength + team defensive strength vs. position + player-within-team effects). That specific extension is unbuilt. |
| News Sentiment as ML Feature | §11.1.J (feature, not §11.2, but same "unclear if wired" pattern) | **Ambiguous, unresolved — flagged in GAPS.md, still not confirmed either way.** `MODEL_CONFIG["enable_news_sentiment"] = True` and `NewsSentimentAnalyzer` exist, but `news_sentiment`/`news_volume` are confirmed **not** in `CAUSAL_FEATURES` (checked directly). Computed, possibly unused — GAPS.md already flagged this as "evaluation needed" and it's still open. |

Already covered elsewhere, not re-listed here: Mixture Density (§11.2.C)
— addressed this session via the asymmetric floor/ceiling fix, see
GAPS.md's write-up, real MDN remains a further option; Monte Carlo Game
Simulation (§11.2.D) — real code exists, see ORPHANED table above;
QB-WR/RB Correlation Matrices (§11.2.E, misnumbered in GAPS.md as under
11.1) — scoped this session, computed real correlations, blocked on a
real lineup-optimizer consumer per the user's own sequencing decision
(see GAPS.md's "Roadmap decision" entry).

---

## EXPERIMENTAL / CANDIDATE (today's work — explicitly not a production decision)

| Model / Strategy | File | Status | Accuracy tested? |
|---|---|---|---|
| Multi-year + team-aware preseason candidate | `src/models/preseason_features.py` | Real, tested (`tests/test_preseason_features.py`), metrics tracked in `EXPERIMENTS.md` §2. **Zero production callers — confirmed above.** | **Yes** — real 2023-2025 holdout, full R²/RMSE/MAE/corr per position in `EXPERIMENTS.md`. Mixed results: QB/WR/TE beat `PreseasonProjector`, RB does not yet (0.009 R² short of production, best candidate found). Nothing decided or shipped. |

---

## Summary: what's actually running right now

For the live 2026 draft board and any weekly serving: **`PreseasonProjector`** (season totals) and **`ComponentPredictor`** (weekly, once the season starts) are the only two skill-position models in the loop, plus **`KickerDSTPredictor`** for K/DST. Everything else in this document — roughly a dozen files, several thousand lines, multiple real trained model artifacts on disk — is either deliberately disabled, architecturally bypassed, or was never wired in at all.

**Accuracy-tested tally, across every section above**: of the ~13
non-live models/strategies cataloged (dormant + orphaned + today's
candidate), only **2** have ever had real accuracy measured against
held-out data — `MultiWeekModel` (partially, QB only, this session) and
today's preseason candidate. Everything else — `Hybrid4WeekModel`,
`DeepSeasonLongModel`, `TouchdownRegressor`, `BayesianPlayerModel`,
`weekly_matchup_predictor.py`, both `LineupOptimizer`s,
`MonteCarloSimulator`, and the four orphaned framework files — is
real, unvalidated code. "Exists" and "works well" are not the same
claim anywhere in this document.

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

## DORMANT (real, complete, often trained — but unreachable from any live entry point)

| Model / Strategy | File | Why dormant | Evidence it's real, not a stub |
|---|---|---|---|
| `EnsemblePredictor` / `PositionModel` / `MultiWeekModel` | `src/models/ensemble.py`, `position_models.py` | Production hardcodes `position_target_type="component"` for every position; `ComponentPredictor` is checked first in `EnsemblePredictor.predict()`'s branch logic, so these are bypassed even when loaded. | `data/models/multiweek_qb.joblib` exists (85MB, real trained artifact, all 18 week-horizons, saved 2026-08-01) — but only for QB; RB/WR/TE never got a `multiweek_*.joblib`. Real head-to-head test this session (`EXPERIMENTS.md` §3): mixed real results, RB clearly beats naive baseline, TE overfits on small sample. |
| `Hybrid4WeekModel` | `src/models/horizon_models.py` | Same `component`-mode bypass. | `data/models/hybrid_4w_{qb,rb,wr,te}.joblib` — real trained artifacts, all 4 positions, LSTM+ARIMA, saved 2026-08-01. |
| `DeepSeasonLongModel` | `src/models/horizon_models.py` | Same `component`-mode bypass. | `data/models/deep_18w_{qb,rb,wr,te}` — real trained artifacts, all 4 positions, residual feedforward net, saved 2026-08-03. |
| `TouchdownRegressor` | `src/models/production_model.py` | Its only call site, `EnsemblePredictor._apply_td_regression()`, is explicitly commented out in `ensemble.py` (`# results = self._apply_td_regression(...)`) with a rationale comment: Huber loss already provides outlier robustness, so this was deliberately disabled, not forgotten — but the class and its logic remain, real and unused. | Real implementation, opportunity-based expected-TD mean reversion. |
| `BayesianPlayerModel` | `src/models/bayesian_models.py` | Zero references anywhere except `tests/test_rookie_projections.py`. Never imported by any production or training script. | 721-line file, real implementation (full Bayesian + simplified variants). |
| `weekly_matchup_predictor.py` | `src/models/weekly_matchup_predictor.py` | Only caller is `scripts/generate_weekly_projections.py`, which itself has zero references anywhere (not in README, not called by any other script) — dormant two levels deep. | 382 lines, real implementation. |
| `LineupOptimizer` (#1) | `src/optimization/lineup_optimizer.py` | `optimize_lineup()` has no caller anywhere in the codebase, no test coverage. Found and flagged earlier this session. | Salary-cap knapsack, cash/GPP strategies, real (if buggy — sums independent player percentiles for lineup floor/ceiling, a real statistical error found and not yet fixed). |

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
run as-is.

---

## EXPERIMENTAL / CANDIDATE (today's work — explicitly not a production decision)

| Model / Strategy | File | Status |
|---|---|---|
| Multi-year + team-aware preseason candidate | `src/models/preseason_features.py` | Real, tested (`tests/test_preseason_features.py`), metrics tracked in `EXPERIMENTS.md` §2. **Zero production callers — confirmed above.** Mixed real results: QB/WR/TE show wins over `PreseasonProjector`, RB does not yet. Nothing decided or shipped. |

---

## Summary: what's actually running right now

For the live 2026 draft board and any weekly serving: **`PreseasonProjector`** (season totals) and **`ComponentPredictor`** (weekly, once the season starts) are the only two skill-position models in the loop, plus **`KickerDSTPredictor`** for K/DST. Everything else in this document — roughly a dozen files, several thousand lines, multiple real trained model artifacts on disk — is either deliberately disabled, architecturally bypassed, or was never wired in at all.

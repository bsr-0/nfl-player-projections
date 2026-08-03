# NFL Player Projections — Project Notes

Consolidated reference for past lessons, audits, council decisions, and open items.

---

## System Overview

Position-specific ML models predicting fantasy points for 1–18 weeks ahead. Key layers:

- **Data**: `nfl-data-py` / nflverse → SQLite (`data/nfl_data.db`), 2006–2025
- **Features**: 33–37 hand-curated causal features per position (FEATURE_VERSION = 21)
- **Models**: Component prediction (predict stat lines, assemble FP); Ridge α=10,000 as walk-forward production default
- **Evaluation**: Walk-forward expanding-window backtest, Spearman ρ as primary metric
- **Config single source of truth**: `config/settings.py`
- **Training**: `python -m src.models.train`

---

## Requirements (Original Spec)

Position-specific models required because cross-position models show 15–20% lower accuracy.

**Three prediction horizons:**

| Horizon | Architecture | RMSE Target |
|---------|-------------|-------------|
| 1-week | RF + XGBoost + Ridge ensemble | 6–8 pts |
| 4-week | Hybrid LSTM-ARIMA | 8–10 pts |
| 18-week | Deep feedforward (98+ layers) + ensemble | 12–15 pts |

**Utilization Score components:** snap share, target/touch share, red zone %, high-value touch rate → normalized 0–100.

**Success criteria (1-week):** RMSE within 10% of expert consensus, Spearman ρ > 0.65, beat naive baselines by >25%, 70%+ of predictions within 7 pts.

---

## Early Dashboard Audit (Jan–Feb 2026)

Issues found and fixed in `scripts/analytics_dashboard.py`:

1. **Hardcoded Super Bowl teams** — "Patriots vs Seahawks" shown year-round. Fixed with dynamic `generate_dynamic_predictions()` using last 2 seasons of actual data.
2. **Perpetual Feb 9 date** — Fixed with auto-incrementing Super Bowl number and current-year date.
3. **Uninformative charts** — Replaced 26-position raw counts with 3 actionable panels: NFL Evolution (25yr trend), Elite Concentration (scarcity by position), Position Depth (reliability).
4. **No ML best practices** — Created `ml_pipeline.py` with `NFLDataPreprocessor`, `TimeSeriesValidator` (expanding window, fit-on-train-only scaler, forward-fill).
5. **Missing column crashes** — Fixed `calculate_utilization_scores()` to handle column renames with safe fallbacks.

---

## Data Fixes (Applied 2026-05-02)

**Vegas Lines backfill (2006–2017):** Schedule table previously only had 2020–2025 with zero spread/total values. Backfilled via `scripts/backfill_vegas_lines.py` — 5,431 games, 100% coverage.

**Team code normalization:** OAK→LV, SD→LAC, STL→LA (historical codes), LAR→LA in `team_stats`.

**Snap count backfill:** `snap_count`/`team_snaps` were zero across all rows. Script `scripts/backfill_snap_counts_to_pws.py` matched 41,660/43,118 rows (96.6%) for 2018–2025 via first-initial + last-name + team + season + week. Pre-2018 remain zero (nflverse unavailable).

**v9 features added:** `snap_share_pct_roll3_mean` (RB, TE), NGS completion % above expected (QB), NGS rush yards over expected (RB), NGS avg separation (WR/TE), decayed draft capital.

---

## Production Model Metrics (2025 Holdout, v9 Models)

| Position | RMSE | MAE | R² | Spearman ρ | ≤7pt% | ≤10pt% |
|----------|------|-----|-----|-----------|-------|--------|
| QB | 7.11 | 5.84 | 0.085 | 0.334 | 65.4% | 83.8% |
| RB | 6.97 | 5.31 | 0.224 | 0.488 | 74.2% | 86.8% |
| WR | 5.73 | 4.29 | 0.243 | 0.464 | 83.1% | 92.1% |
| TE | 5.56 | 4.05 | 0.118 | 0.405 | 84.4% | 92.3% |

Low R² is expected — these metrics were generated at v9 (9–14 features); current model (v21, 33–37 features) has not been re-benchmarked on a full holdout. Spearman ρ (0.33–0.49) is the relevant fantasy metric.

---

## Walk-Forward Backtest Findings (April 2026 Council Process)

### Baseline results (Ridge α=1, April 10 2026)

| Metric | Overall | QB | RB | WR | TE |
|---|---|---|---|---|---|
| R² | 0.269 | 0.092 | 0.258 | 0.257 | 0.152 |
| Pearson r | 0.520 | 0.325 | 0.512 | 0.513 | 0.399 |
| Bias | +3.2% | +2.1% | −5.3% | +7.5% | +4.0% |

**Key insight on variance compression:** For RB/WR, `std(pred)/std(actual) ≈ r`. This is mathematical, not a regularization bug. The only fix is lifting correlation via better features — not reducing shrinkage.

### Alpha sweep (April 20 2026)

Raised `RIDGE_DEFAULT_ALPHA` from 1.0 to **10,000** in `config/settings.py`. Cross-season result: 29-14 (67.4%) hindsight win rate vs 27-16 (62.8%) at α=1 (p=0.016, n=43 weeks).

### Phase 1 — Vegas lines (April 21 2026)

Vegas features had been silently pinned at constants (`implied_team_total=23.0`, `spread=0.0`) due to `except Exception: pass`. Fixing this lifted overall R² by +0.004, QB r by +0.031–0.034 on 2025. Cross-season hindsight: 67.4% → **69.8% (30-13, p=0.007, ROI +25.6%)**.

**Lesson:** The "+0.004 R²" is a bug fix, not a feature addition. Vegas had been declared in CAUSAL_FEATURES for months; it just wasn't working.

### Phase 2 — Opponent defense (April 21 2026)

Added season-to-date expanding lag opponent defense feature alongside the existing single-week version. Result: max RB r lift **+0.0015** vs +0.02 kill threshold → **rejected**. Root cause: the single-week `opp_fpts_allowed` was already carrying the signal; the s2d version is collinear.

Bonus fix: `player_weekly_stats.opponent` was 100% empty for 2025 — opponent defense was silently dead that whole season. Partially backfilled via schedule table (93.1% filled). **LAR (200 rows) and JAC (186 rows) remain empty** — team code mismatch between `player_weekly_stats` (`LAR`/`JAC`) and `schedule` (`LA`/`JAX`) caused the join to silently skip both teams. Opponent-based features for Rams and Jaguars players in 2025 are still dead.

### Phase 3 — Injury status at lock time (April 22 2026)

Phases 1+2+3 together: **+0.004 cumulative R²** vs +0.02 workstream threshold → **kill criterion fired**. Three "highest-confidence" features (Vegas, injuries, opp defense) turned out to be already-declared-but-silently-dead; fixing the bugs added modest but real lift.

### Phase 4 — Ensemble walk-forward (April 22 2026)

Pre-registered kill gate #3 ("runtime > 4× Ridge") fired: both ensemble runs died after ~12.8h wall clock (~650–700s per week, 13× Ridge). **Production default stays Ridge α=10,000.**

Residual bug (Vegas bypass): considered fixed — Vegas loading is now centralized through `FeatureEngineer._create_vegas_game_script_features()` with structured warning logging instead of silent pass. **New residual bug found:** `PositionModel.compute_ensemble_diversity()` calls `self._prepare_features(X)` at `src/models/position_models.py:640` — that method does not exist (should be `self._prepare_input(X)`). Will crash with `AttributeError` if called.

### Causal features audit (April 24 2026)

Walk-forward confirmed leakage-safe. Found silent feature dropout: 4 declared share features (`target_share_pct`, `rush_share_pct`, `snap_share_pct`, `air_yards_share_pct`) were never computed because `utilization_scores` is empty in the DB. Fixed by computing shares from raw stats in `_create_base_features`. Per-position R² lifted +0.06 to +0.20.

H2H kill gate moved from 70.73% → **75.61%** (bootstrap p5: 58.5% → **65.85%**).

`snap_share_pct` deliberately excluded — `snap_count`/`team_snaps` are zero-filled for all seasons (data never populated). **Corrected after snap backfill (2026-05-02).**

### Bootstrap validation (April 22 2026)

10,000 resamples of 43-week H2H record: p5 = 58.14% vs −110 break-even 52.38% → **kill criterion does not fire**. 99.19% of resamples exceed breakeven. Caveat: at 1.8× DFS cash break-even (55.56%), p5 clears by only 2.6 pp.

### Draft product kill (April 24 2026)

Pre-draft `--ranking week1` sim loses to ADP on both 2024 (−9.0%) and 2025 (−15.2%) after fixing a name-match bug. ADP bakes in offseason news, camp injuries, depth-chart moves that the pipeline does not have.

**Pure ML pre-draft ranking (`--ranking week1`) is shelved** — it loses to ADP and has no offseason signal. The user-facing draft product (`docs/index.html`) is a blended board that anchors to ADP and applies calibrated adjustments on top; that remains active. Start/sit (75.6% H2H, p5=65.85%) is the sole ML-performance claim.

---

## Predictive Ceiling Summary

**The core problem:** Ridge walk-forward R²≈0.27, loses to blended trailing-average heuristic (R²=0.279). Model adds no value over a simple blend until R² exceeds 0.279.

**What was ruled out as root cause:**
- Two-stage UtilizationToFPConverter compounding error — ruled out (backtester trains directly on fantasy_points)
- Hard prediction caps in ensemble — ruled out (empirical max predictions far below caps)
- Huber loss flattening tails — N/A for Ridge backtest
- Feature percentile clipping — minor contributor only

**What remains:** Weak feature signal. Fix requires more predictive features, not regularization tweaks.

---

## Agent Directive V7 Audit (March 2026)

Revised compliance: **~52%** (5 PASS, 16 PARTIAL, 4 FAIL). Key findings:

**Critical gaps:**
1. Confidence intervals miscalibrated — 73% actual coverage at 90% nominal (17pp error)
2. No decision optimization layer — system predicts FP but no lineup optimizer, start/sit engine, or abstention logic
3. ~~CI test failures swallowed — some test stages non-blocking~~ **Fixed** — all test stages in `rubric-compliance.yml` are now blocking (verified 2026-07-31)
4. No data pipeline resilience — no DAG, no idempotency, no schema validation

**Strengths:**
1. Temporal integrity deeply embedded — `src/utils/leakage.py`, `ts_backtester.py` enforces chronological splits
2. Comprehensive fantasy metrics — Spearman ρ, tier accuracy, boom/bust, VOR, within-N
3. Rigorous ML audit suite — `tests/test_ml_audit.py` (7 phases including poison feature injection)
4. Production monitoring — prediction drift, feature drift (KS test), RMSE degradation alerts
5. Experiment tracking — append-only JSONL with git commit hashes

**Priority fixes (from audit):**
- Fix 73%/90% calibration gap (quantile regression or larger conformal calibration set)
- Add VOR-based start/sit recommendation with abstention threshold
- ~~Make all CI stages blocking~~ Done
- Add dataset checksums to experiment ledger
- Schema validation on data ingestion

---

## Known Limitations by Horizon

### 1-week (start/sit)
- No weather data for outdoor games
- Injury status integration (added Phase 3 but small lift)
- No primetime adjustment
- No tier-specific uncertainty bounds

### 5-week (trade analysis)
- No schedule strength analysis
- No coaching/scheme change detection
- No trade deadline team-change adjustment

### 18-week (draft/season-long)
- Pure ML pre-draft ranking shelved — ADP beats model; blended board (`docs/index.html`) remains the user-facing product
- No games-played projection (assumes 17)
- No age/decline curves
- No suspension risk

### Data gaps
- Snap counts: pre-2018 unavailable
- NGS: 2018–2025 only
- Real air yards: estimated
- Red zone data: estimated from TDs

---

## Future Data Sources (Priority Order)

High impact: PFR advanced stats, contracts/incentives, depth charts (daily), FTN charting data, combine measurables. Medium: weather API, beat reporter injury news, college stats for rookies, ADP from FantasyPros.

---

## Methodology Notes

**Utilization Score weights:**

| Position | Snap% | Target% | Rush% | Air Yards | Red Zone | Other |
|----------|-------|---------|-------|-----------|----------|-------|
| RB | 30% | 20% | 25% | — | 15% | 10% goalline |
| WR | 25% | 30% | — | 25% | 15% | 5% route |
| TE | 25% | 35% | — | 20% | 20% | — |
| QB | — | — | 20% | — | 25% | 35% dropback, 20% pressure |

**WOPR:** `1.5 × Target Share + 0.7 × Air Yards Share`

**Walk-forward CV:** Expanding window, weekly refit, per-fold feature recompute, train-only scaler fit. Never random splits on time series.

**Feature acceptance rules:** stable importance via RFE, no impossible information (enforced by `src/utils/leakage.py`), production availability (computable from historical data), walk-forward improvement tested.

---

## Recurring Bugs / Patterns to Watch

1. **Silent fallback pattern** — `except Exception: pass` blocks that silently zero-fill features. Vegas, injuries, and opponent defense all suffered this. Any new data integration should log structured warnings instead.
2. **Empty DB columns declared as features** — `utilization_scores` table empty → 4 share features never computed. Always verify feature values are non-zero before declaring them in CAUSAL_FEATURES.
3. **`train.py` import shadowing** — `json`/`np` imported inside if-blocks shadowing outer scope. Fixed 2026-05-02.
4. **Full Optuna training OOM** — Crashes at ~exit 139. Use `--no-tune` or `--fast --no-tune` for all non-production runs.
5. **`component_models[pos] = None`** — Component mode sets trained_models to None. Most call sites in `train.py` (lines 206, 366, 424, 843, 1157) guard correctly. **Live unguarded crash** in `src/evaluation/ablation.py:152` — accesses `multi_model.models` without a None check.
6. **Opponent column partially backfilled** — 2025 is 93.1% filled. **LAR and JAC remain 100% empty** (team code mismatch: `LAR`/`JAC` in `player_weekly_stats` vs `LA`/`JAX` in `schedule`). Opponent features for Rams and Jaguars players in 2025 are still dead.
7. **Archived backtest results invalid** — `data/backtest_results/archived/` contains a run showing R²=0.996, which is implausible (realistic ceiling is 0.2–0.4). Generated by an older `TimeSeriesBacktester.leakage_safe_features()` that computed train features from a combined train+test block, contaminating train-period rolling stats. Fixed; do not use those results as a baseline.

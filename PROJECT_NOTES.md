# NFL Player Projections — Project Notes

Consolidated reference for past lessons, audits, council decisions, and open items.

---

## System Overview

Position-specific ML models predicting fantasy points for 1–18 weeks ahead. Key layers:

- **Data**: `nfl-data-py` / nflverse → SQLite (`data/nfl_data.db`), 2006–2026
- **Features**: 39–50 hand-curated causal features per position (FEATURE_VERSION = 25, as of 2026-08-04 — was 21 when this line was last accurate; bumped 22→25 across several sessions, see GAPS.md for what each version added)
- **Models**: Component prediction (predict stat lines, assemble FP) is the actual production path for all positions (`position_target_type = "component"`). Ridge α=10,000 is still the default for non-component mode, but production bypasses it — see "Component Prediction Architecture" below.
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

## Production Model Metrics (2025 Holdout, v9 Models) — STALE, v9

| Position | RMSE | MAE | R² | Spearman ρ | ≤7pt% | ≤10pt% |
|----------|------|-----|-----|-----------|-------|--------|
| QB | 7.11 | 5.84 | 0.085 | 0.334 | 65.4% | 83.8% |
| RB | 6.97 | 5.31 | 0.224 | 0.488 | 74.2% | 86.8% |
| WR | 5.73 | 4.29 | 0.243 | 0.464 | 83.1% | 92.1% |
| TE | 5.56 | 4.05 | 0.118 | 0.405 | 84.4% | 92.3% |

**This table is v9 (9–14 features) and stale.** Current model is v25
(39–50 features per position, as of 2026-08-04). **Current production
metrics are unknown** — no v22–v25 holdout benchmark has been run; only
"feature computes correctly, non-degenerate, doesn't crash" has been
verified for the v23/v24/v25 additions, not accuracy lift. This is the
next validation step (see GAPS.md's "STATUS BRIEF" for the running list of
what's been verified vs. what's still an open measurement).

---

## Walk-Forward Backtest Findings (April 2026 Council Process)

### Baseline results (Ridge α=1, April 10 2026)

| Metric | Overall | QB | RB | WR | TE |
|---|---|---|---|---|---|
| R² | 0.269 | 0.092 | 0.258 | 0.257 | 0.152 |
| Pearson r | 0.520 | 0.325 | 0.512 | 0.513 | 0.399 |
| Bias | +3.2% | +2.1% | −5.3% | +7.5% | +4.0% |

**Key insight on variance compression:** For RB/WR, `std(pred)/std(actual) ≈ r`. This is mathematical, not a regularization bug. The only fix is lifting correlation via better features — not reducing shrinkage. As of this writing the documented fix ("lift correlation via better features") has not produced a measured R² lift in several months — largely because nobody has re-run the R²/Spearman benchmark since v9 despite multiple feature additions (v22–v25). Whether the theoretical ceiling has actually been reached with available data, versus simply never re-measured, is unknown until that benchmark runs.

### Alpha sweep (April 20 2026)

Raised `RIDGE_DEFAULT_ALPHA` from 1.0 to **10,000** in `config/settings.py`. Cross-season result: 29-14 (67.4%) hindsight win rate vs 27-16 (62.8%) at α=1 (p=0.016, n=43 weeks).

**STALE — reversed 2026-08-06, see GAPS.md §"Skeptical audit of RIDGE_DEFAULT_ALPHA=10,000".** A same-day, same-season, same-codebase direct reproduction found the opposite result: α=1.0 beat α=10,000 72.7% (16-6, p=0.026, ROI +30.9%) vs 63.6% (14-8, p=0.143 not significant, ROI +14.5%). The α=10,000 change also turned out to have **zero effect on real production** the whole time — `ComponentPredictor` (what's actually served) hardcodes `alpha=1.0` independently of this constant; only the `ts_backtester.py` evaluation tool ever read it. `RIDGE_DEFAULT_ALPHA` was lowered back to `1.0` in `config/settings.py` on 2026-08-06 after this finding, and a duplicate hardcoded copy in `scripts/paper_trade_lock.py` was fixed to reference the constant instead of a separate literal.

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

**Pure ML pre-draft ranking (`--ranking week1`) is shelved** — it loses to ADP and has no offseason signal. **The reason, precisely:** ADP embeds offseason signals the model has zero access to — camp reports, depth-chart battles, coaching changes, free-agency context. This is a **data gap, not a model gap**. The correct fix is ingesting those signals (see GAPS.md §3 for the market/ADP-bias removal work and §3.1 for the coaching-staff data now wired in as a first step in that direction — head-coach identity + change detection, done 2026-08-04), not blending predictions with ADP. As of 2026-08-04, `preseason_ecr` and the post-processing market-anchor system have both been fully removed from the pipeline (see GAPS.md §3 "STATUS BRIEF") — the projector is pure ML now; whether that changes the ADP-vs-model comparison above has not been re-tested since the removal.

The user-facing draft product (`docs/index.html`) is a blended board that anchors to ADP and applies calibrated adjustments on top; that remains active. Start/sit (75.6% H2H, p5=65.85%) is the sole ML-performance claim.

---

## Predictive Ceiling Summary — UPDATED 2026-08-04, ceiling cleared

**Original problem (through ~v9-v21 features):** Ridge walk-forward R²≈0.27, loses to blended trailing-average heuristic (R²=0.279). Model added no value over a simple blend until R² exceeded 0.279.

**What was ruled out as root cause (at the time):**
- Two-stage UtilizationToFPConverter compounding error — ruled out (backtester trains directly on fantasy_points)
- Hard prediction caps in ensemble — ruled out (empirical max predictions far below caps)
- Huber loss flattening tails — N/A for Ridge backtest
- Feature percentile clipping — minor contributor only

**What remained:** Weak feature signal. Fix required more predictive features, not regularization tweaks.

**2026-08-04 update: the ceiling has been cleared.** A `run_ts_backtest.py`
walk-forward run on the 2025 holdout, Ridge α=10,000 (production default),
with the full v22–v25 feature set live, measured **overall R²=0.345** —
above the 0.279 heuristic ceiling, and every position individually
improved over the last-measured baseline (QB 0.092→0.223, RB 0.258→0.364,
WR 0.257→0.276, TE 0.152→0.263). This is the first time this measurement
has been re-run since v9/v21; the fix that finally worked was exactly
"more predictive features, not regularization tweaks" as predicted above
— the accumulated v22–v25 work (dest-team features, ADP/market-bias
removal, weather/WOPR/red-zone allocation, team target-allocation/tempo,
head-coach-identity change detection) collectively did it. See GAPS.md's
"STATUS BRIEF" → "Next: validation testing" section for the full numbers
and caveats (single-season, single-model-type measurement; doesn't isolate
which specific change(s) drove the lift).

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
- Weather data: **added 2026-08-04** — `wind_speed_mph`, `is_dome`, `precipitation_flag`, `temperature_bucket` in CAUSAL_FEATURES for all positions. Was ingested into `game_weather` and stored for a long time before this but never reached the feature pipeline.
- Injury status integration (added Phase 3 but small lift)
- No primetime adjustment
- No tier-specific uncertainty bounds

### 5-week (trade analysis)
- No schedule strength analysis
- Coaching/scheme change detection: **added 2026-08-04** — `coaching_change`, `coaching_adaptation_score`, `coaching_stability`, `coaching_change_impact` are now real (head-coach identity from nflverse), in CAUSAL_FEATURES for all positions. OC/DC-level detection remains unavailable (no reliable structured 20-year data source found).
- No trade deadline team-change adjustment

### 18-week (draft/season-long)
- Pure ML pre-draft ranking shelved — ADP beats model; blended board (`docs/index.html`) remains the user-facing product
- No games-played projection (assumes 17)
- No age/decline curves
- No suspension risk

### Data gaps
- Snap counts: pre-2018 unavailable
- NGS: 2018–2025 only
- Real air yards: `ngs_avg_intended_air_yards` (NGS, 2018+) is real, not estimated, and is in CAUSAL_FEATURES for WR/TE. Estimated-from-target-depth air yards is only used as a fallback outside that window.
- Red zone data: as of v23 (2026-08-04), `redzone_targets` and `rush_inside_5` are real per-player counts from PBP (not estimated from TDs) — `redzone_target_share_pct_roll3_mean` and `goal_line_carry_share_pct_roll3_mean` are computed directly from them. Team-level red-zone play-calling tendency (pass vs. rush inside the 20/10) beyond per-player share is still not modeled.

---

## Future Data Sources (Priority Order)

High impact: ~~PFR advanced stats~~ (done — `qb_pressure_pct`, `recv_drop_pct`, `qb_bad_throw_pct`, `qb_pocket_time`, `rb_ybc_avg`/`rb_yac_avg`, `rb_broken_tackles_prior` all in CAUSAL_FEATURES), ~~contracts/incentives~~ (done — `is_contract_year`, `contract_apy_rank`), ~~depth charts~~ (done — `depth_chart_rank`), FTN charting data / personnel grouping (11/12/13%) — still not ingested, real gap (PBP `offense_personnel` is the better source, only available from a certain year onward, not yet wired), combine measurables (`combine_data_v2` table exists in DB with 8,968 rows, confirmed still unused in the pipeline as of 2026-08-04). Medium: ~~weather API~~ (done), beat reporter injury news, college stats for rookies, ADP from FantasyPros, Vegas preseason win totals (scraper exists in `odds_scraper.py` but the `win_totals` market has never actually been scraped into the DB — `game_odds` only has h2h/spreads/totals).

---

## Component Prediction Architecture

The **actual production path** (`position_target_type = "component"` for
all four positions) predicts individual stat lines directly — e.g. for RB:
`rushing_yards`, `rushing_tds`, `receptions`, `receiving_yards`,
`receiving_tds` — each as its own Ridge model (α fitted via CV), then
assembles fantasy points from the predicted components using the scoring
formula in `SCORING`/`SCORING_HALF_PPR`/`SCORING_STANDARD`. This is more
interpretable and auditable than predicting fantasy points as a single
black-box target, and was chosen over the older utilization-score →
FP-converter path (below) specifically for that reason.

When `position_target_type = "component"`, the older
utilization/FP-ensemble path (Ridge α=10,000 on a single FP or utilization
target) is **bypassed entirely** — it still exists in the codebase and is
the default for non-component mode, but production training doesn't call
it. `trained_models[pos]` is set to `None` in component mode; several call
sites in `train.py` guard for this correctly, but see Recurring Bug #5
above for one unguarded crash site.

## Utilization Weight Optimization

The `UTILIZATION_WEIGHTS` table in `config/settings.py` (snap%/target%/
rush%/red-zone%/etc. per position) is **only the default/fallback**. Every
training run calls `fit_utilization_weights()`
(`utilization_weight_optimizer.py`) to fit position-specific weights from
actual data, and those fitted weights — not the hardcoded config values —
are what gets used for that run's utilization scores. The static table
below is documentation of the *starting point*, not the weights actually
in effect on any given trained model. `WARNING: Applied utilization weight
floor for {position}: {component} (min=0.03)` in training logs means the
optimizer wanted to push a component's weight below the floor and got
clamped — check `model_metadata.json` / the training log for the actual
fitted weights on a specific run rather than assuming this table.

**Utilization Score weights (defaults only, see above):**

| Position | Snap% | Target% | Rush% | Air Yards | Red Zone | Other |
|----------|-------|---------|-------|-----------|----------|-------|
| RB | 30% | 20% | 25% | — | 15% | 10% goalline |
| WR | 25% | 30% | — | 25% | 15% | 5% route |
| TE | 25% | 35% | — | 20% | 20% | — |
| QB | — | — | 20% | — | 25% | 35% dropback, 20% pressure |

## Draft Prep Mode (June–August training window)

In the offseason (roughly June through the start of the season), the
current season has no games played yet, so the normal "train on past
seasons, test on current season's played weeks" split has no test data.
`is_draft_prep_window()` (`src/utils/nfl_calendar.py`) detects this window
and `DataManager.get_train_test_seasons()` substitutes an empty/placeholder
test set for the upcoming season instead of erroring. Two things depend on
this carve-out and will break in August if it's ever removed without
updating both:
- `_prepare_training_data()`'s in-season guard (`src/models/data_loading.py`)
  — mirrors the same exemption; without it, `current_season_has_weeks_
  played()` reads misleadingly `True` in August (the *prior* season had
  weeks played, it just ended months ago) and the guard raises.
- `feature_preparation.py`'s `_apply_bounded_scaling()` — the empty-test-set
  path needs `test_df` to stay empty through the whole pipeline; a
  pre-existing bug (fixed 2026-08-04) indexed `test_df[cols]`
  unconditionally one line below an `if not test_df.empty` guard for the
  same dataframe, crashing on any draft-prep training run.

Draft prep mode is why `python -m src.models.train` in August trains on
`[2018 ... 2025]` with test season `2026` and an empty test dataframe —
this is expected, not a bug.

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
6. **Opponent column team-code mismatch — FIXED 2026-08-04.** The root cause was worse than originally documented here: `entity_resolver.py`'s `TEAM_CODE_ALIASES` had `JAX→JAC`/`LA→LAR` backwards relative to `schedule`'s convention, applied on every data refresh via `nfl_data_loader.py`'s `resolver.build_keys()`. This meant `team_stats`/`team_defense_stats`/`player_weekly_stats` had **both** codes coexisting for the same team across different weeks (not just `LAR`/`JAC` vs `LA`/`JAX` as two static, separately-coded tables). Fixed by flipping the alias direction; `scripts/normalize_lar_jac_team_codes.py` cleaned up existing rows. A near-identical bug was found and fixed the same day for `OAK`/`LV` (2018-2019 Raiders) while backfilling coaching-staff data — same lesson, different teams. `opp_fpts_allowed` is now 100% populated for both teams in every affected season.
7. **Archived backtest results invalid** — `data/backtest_results/archived/` contains a run showing R²=0.996, which is implausible (realistic ceiling is 0.2–0.4). Generated by an older `TimeSeriesBacktester.leakage_safe_features()` that computed train features from a combined train+test block, contaminating train-period rolling stats. Fixed; do not use those results as a baseline.
8. **Declared-in-`CAUSAL_FEATURES`-but-never-actually-computed** — a specific flavor of #2 found repeatedly in the 2026-08-04 session: `wopr_roll3` and `recv_epa_per_target_roll3_mean` were declared for WR/TE and computed... but only in the full-mode pipeline (`_create_rolling_features`), never in the causal pipeline's `_create_causal_rolling_features`, so they silently didn't exist for the production (causal) training path. Same failure mode, different root cause each time (WOPR: wrong pipeline calls it; `opp_fpts_allowed` in a naive test harness: missing a required LEFT JOIN, not an actual bug). **The general lesson:** before trusting any GAPS.md/PROJECT_NOTES.md claim about a feature's status, run `create_causal_features()` (via the real `get_all_players_for_training()` loader, not a partial hand-built one — that itself caused a false-positive "dead feature" reading once) on real data and check non-null rate + variance directly, per position. Don't trust the declaration list alone.
9. **Historical-vs-current team-code drift** — the `LAR`/`JAC` and `OAK`/`LV` bugs (#6 above) share a pattern worth watching for with any new team-keyed table: nflverse sources are not internally consistent about whether they use the code that was accurate *at the time* (`OAK`, `JAX`) or the current franchise code retroactively applied (`LV`, `JAC`/`LAR` depending on source). Any new per-team ingestion should immediately spot-check for team-code duplication across seasons (`GROUP BY team, season` and look for two codes representing the same franchise in different rows) before trusting the join.
10. **Row-order-dependent groupby on a differently-sorted frame** — `CoachingChangeDetector._detect_hc_changes`/`_compute_coaching_tenure` (`src/features/advanced_analytics.py`) did `df.groupby("team")[...].shift(1)`/`.cumsum()` directly on the per-player training frame, which `add_coaching_change_features` itself sorts by `player_id` first — so the "previous row" for a team-group shift was often a different player's unrelated season/week, not the true prior team-week. Silent and undetected because the class had never been fed real data before 2026-08-04. Fixed by computing on a `(team, season, week)`-deduplicated, properly-sorted table and merging results back by key. **Watch for this pattern anywhere a per-team or per-position aggregate is computed via `groupby(...).shift()`/`.cumsum()` on a frame that isn't already sorted by that same key** — it will silently produce plausible-looking but wrong values rather than erroring.

## Finalized Phase 2 Calibration and Acceptance Plan

This plan governs the next Phase 2 evaluation. It does not retroactively change the current preregistered result: raw `hist_gbm` remains the incumbent, its predictive-performance requirements passed, and its original per-position calibration gate remains failed for QB/RB. It also does not authorize any Phase 7 serving change.

### 1. Freeze the future protocol before inspecting results

Create a versioned preregistration file containing the database/version, target definitions, walk-forward folds, candidate models, baselines, calibration method, binning method, bootstrap settings, confidence level, non-inferiority margin, multiple-comparison rule, cold-start policy, unknown-label policy, and pass/fail logic.

The proposed future calibration rule is: compare `ECE(candidate) - ECE(position_rate)` using paired within-position bootstrap confidence intervals; use a predeclared margin (initial proposal `+0.005`), 95% confidence, and simultaneous control across required positions. A candidate is calibration-non-inferior only when the upper confidence bound is no greater than the fixed margin for every required position. The margin and rule must be approved before the run and may not be selected after viewing results.

### 2. Validate data, labels, and pairing

- Rebuild or validate `canonical_player_weeks`.
- Keep unknown rows out of supervised labels; retain confirmed zero-snap rows as negatives.
- Resolve duplicate player-week rows deterministically and fail on unresolved conflicts.
- Require player, season, week, team, position, fold, train cutoff, actual label, candidate probability, and baseline probability in OOF output.
- Enforce one-to-one pairing between candidate and baseline rows.
- Verify no player-week appears in both training labels and outer test predictions.
- Report row counts by season and position.
- Fail loudly on duplicate keys, missing fold metadata, or mismatched pairs.

### 3. Establish a baseline hierarchy

Evaluate global rate, position rate, smoothed position rate, prior-season hierarchical position/team rate, previous-game, rolling-3, and rolling-5 baselines. All baselines must be fit using only information available before the relevant test fold. The position-rate baseline remains the primary calibration reference, but stronger smoothed baselines reveal whether the comparison is overly conservative.

### 4. Evaluate the meaningful-participation classifier

For the 10% target, report Brier, log loss, ROC AUC, PR AUC, ECE, adaptive ECE, maximum calibration error, calibration slope/intercept, reliability diagrams, and Brier reliability/resolution/uncertainty decomposition.

Required slices are all rows, every position, cold-start, has-history, season, team, roster-backed versus fallback team rows, label source, and history-depth groups. Every slice must include sample size and effective sample size.

Repeat calibration analysis with fixed 10-bin and 20-bin schemes, quantile bins, adaptive bins, and minimum-support bins. A candidate must not pass solely because one arbitrary binning scheme favors it.

### 5. Evaluate conditional opportunity

The second target is snap share conditional on meaningful participation. Report MAE, RMSE, median absolute error, signed error, continuous calibration, prediction interval coverage if available, and error by position, season, cold-start status, history depth, and actual snap-share range. Compare against rolling-3 and rolling-5 baselines using paired uncertainty intervals.

Also evaluate the combined unconditional opportunity estimate:

`P(meaningful participation) × E(snap share | meaningful participation)`

This separates success of the two stages from success of the combined opportunity signal.

### 6. Evaluate calibration candidates correctly

Retain raw GBM as the incumbent. Evaluate position-specific Platt, cross-fitted isotonic, pooled one/three/five-season isotonic, exponentially weighted pooled calibration, beta calibration, regularized beta calibration, empirical-logit calibration, hierarchical/partial-pooling calibration, and residual shrinkage toward the position-rate baseline.

Every calibrator must use strictly earlier data, separate fit and held-out calibration rows, minimum-support rules, explicit fallback logging, and persisted calibration metadata. Isotonic candidates must persist knots, fitted values, flat regions, large jumps, support counts, and monotonicity checks.

### 7. Perform full curve and bin audits

For every fold, season, position, and candidate, persist bin interval, count, mean predicted probability, observed rate, absolute gap, and ECE contribution. Inspect the dominant `0.9–1.0` bin separately from the `0.6–0.8` mid-range.

For QB specifically, inspect actual isotonic step-function knots and fitted values, fit/held-out row counts, knot counts, flat regions, jumps, sparse thresholds, and raw-fallback behavior. Repeat for other positions if QB exposes instability.

The audit must answer whether a calibration method preserves the already-accurate top bin, fixes mid-range error, creates new majority-mass error, or merely shrinks probabilities toward a season-specific prior.

### 8. Bootstrap uncertainty and non-inferiority

Use paired OOF rows. Within each position, resample player-week rows with replacement, recalculate candidate and baseline ECE, and store the paired difference. Use at least 1,000, preferably 2,000, replicates with a fixed seed. Produce confidence intervals and the probability that the difference is non-positive.

Apply the same paired procedure to Brier, log loss, and conditional MAE. Do not use overlapping confidence intervals as the decision rule; use the preregistered difference and margin.

### 9. Control multiple comparisons

Nominate one primary candidate before evaluation. Treat other candidates as secondary or exploratory. Predefine whether simultaneous position control uses a max-statistic bootstrap, Bonferroni adjustment, or another family-wise rule. Do not select the best calibration window or candidate after seeing test results without documenting model-selection correction.

### 10. Test temporal and subgroup stability

Report per-season ECE, calibration slope/intercept, top-bin gap, mid-range gap, and sign of calibration error. Include recent-season and rolling-window summaries. Stress-test unusual seasons, rule changes, COVID-era data, roster-source changes, and distribution shifts.

Check cold-start, established players, role, team, roster source, label provenance, and observation-quality subgroups. Require minimum effective sample sizes before interpreting position-specific results.

### 11. Verify feature provenance and causal safety

Confirm that unknown labels are not negatives, snap labels are authoritative, teammate features use lagged outcomes, current-week outcomes/status/injury/depth information is not introduced, team fallback rates are reported, and structural missingness is preserved.

Run perturbation tests: change future outcomes and verify past features do not change; change a teammate's current-week outcome and verify focal competition features do not change; alter current-week labels and verify classifier inputs remain unchanged.

### 12. Apply a fixed decision matrix

The final matrix must separately mark pass/fail for Brier/log loss versus simple baselines, ECE non-inferiority, confidence-interval requirements, cold-start reporting, unknown-label handling, conditional MAE versus rolling-3, temporal stability, subgroup stability, leakage checks, causal walk-forward checks, and the complete test suite.

Do not accept a candidate on aggregate improvement alone. If no candidate passes, retain raw `hist_gbm`, describe it as useful for ranking/discrimination rather than fully calibrated probability output, and do not apply a post-hoc transformation.

### 13. Separate Phase 2 from Phase 3 and Phase 7

Phase 2 acceptance does not establish PPR improvement. Any serving promotion requires a separate Phase 3 matched-population experiment with reliable effect estimates, valid matching, sensitivity analyses, leakage checks, and downstream PPR/decision utility improvement. A Phase 2 offline opportunity model may be accepted without being permitted to alter Phase 7 serving predictions.

### 14. Required artifacts

The finalized run should write a preregistration JSON, evaluation JSON, OOF predictions, calibration bootstrap JSON, calibration bins CSV, calibration curves JSON and plots, season-position metrics, opportunity metrics, cold-start metrics, feature-provenance checks, causal walk-forward checks, a final Phase 2 decision JSON, and a manifest containing database/code versions, fold definitions, calibration settings, seeds, replicates, margin, and the explicit Phase 7 non-modification statement.

### 15. Execution order

1. Freeze and approve preregistration.
2. Validate data, labels, folds, and pairings.
3. Run leakage and causal checks.
4. Generate OOF predictions.
5. Generate calibration curves and bins.
6. Compute point metrics.
7. Compute bootstrap intervals.
8. Evaluate temporal and subgroup stability.
9. Evaluate conditional and combined opportunity.
10. Apply the fixed decision matrix.
11. Write the final manifest and decision.
12. Only then decide whether a separate Phase 3 integration experiment is justified.

The central separation is: discrimination, calibration, and downstream fantasy utility are different questions and require different evidence and gates.

# Comprehensive Phase 2 Calibration and Acceptance Plan

## 1. Preserve the current decision

Before starting the next evaluation:

- Keep raw `hist_gbm` as the current Phase 2 incumbent.
- Do not promote teammate, Platt, isotonic, or pooled-isotonic variants.
- Do not modify Phase 7 serving predictions.
- Preserve the original preregistered gate for the completed run.
- Record that raw GBM passed predictive-performance requirements but failed the original per-position calibration requirement for QB and RB.

Any revised acceptance rule applies only to a future evaluation and must be finalized before inspecting its results.

## 2. Define the future acceptance protocol first

Create a versioned preregistration file, for example:

`data/experiments/participation_system/phase2/preregistration_v2.json`

It should specify:

- Evaluation dataset and database version.
- Target definition:
  - meaningful participation: offensive snap share ≥10%;
  - conditional opportunity: snap-share percentage conditional on participation.
- Walk-forward fold construction.
- Minimum training-season count.
- Candidate models.
- Baselines.
- Calibration method.
- Number of calibration bins.
- Bootstrap method.
- Random seed.
- Number of replicates.
- Confidence level.
- Non-inferiority margin.
- Multiple-comparison rule.
- Cold-start policy.
- Missing/unknown-label policy.
- Final pass/fail logic.

A proposed initial calibration rule:

- Metric: 10-bin ECE, with adaptive-bin sensitivity analysis.
- Difference: `ECE(candidate) − ECE(position_rate)`.
- Bootstrap: paired, within-position, 1,000 or preferably 2,000 replicates.
- Confidence level: 95%.
- Illustrative margin: `+0.005`.
- A candidate is calibration-non-inferior if the upper bound of the 95% confidence interval is no greater than `+0.005` for every required position.
- The margin must be approved and recorded before the next run; it must not be selected after seeing results.

Because there are four positions, define whether the rule is:

- strict per-position control;
- simultaneous confidence intervals;
- or a multiplicity-adjusted family-wise rule.

The safest initial choice is simultaneous control across positions, such as a max-statistic bootstrap or Bonferroni-adjusted intervals.

## 3. Verify the evaluation data and pairing

Before fitting any candidate:

- Rebuild or validate `canonical_player_weeks`.
- Confirm unknown rows are excluded from supervised labels.
- Confirm confirmed zero-snap rows remain valid negatives.
- Confirm duplicate player-week rows are resolved deterministically.
- Confirm each OOF row has:
  - player ID;
  - season;
  - week;
  - position;
  - team;
  - fold/test season;
  - training cutoff;
  - actual label;
  - candidate probability;
  - baseline probability.
- Require one-to-one pairing between candidate and baseline rows.
- Check that no player-week appears in both a model’s training labels and its outer test predictions.
- Verify fold boundaries and training-season cutoffs.
- Record row counts by season and position.

The evaluator should fail loudly on duplicate keys, missing fold metadata, or mismatched candidate/baseline rows.

## 4. Establish the baseline hierarchy

Do not use only one weak baseline. Compare against:

1. Global participation rate.
2. Position-rate baseline.
3. Smoothed position-rate baseline.
4. Hierarchical position/team rate, using only prior seasons.
5. Previous-game indicator.
6. Rolling-3 participation rate.
7. Rolling-5 participation rate.
8. Raw GBM.
9. Teammate-competition GBM.
10. Calibrated GBM variants.

The position-rate model should remain the principal calibration reference, but stronger hierarchical baselines will show whether the comparison is unfairly conservative or whether the GBM truly adds reliable information.

All baselines must be fit using only data available before each test fold.

## 5. Evaluate the meaningful-participation classifier

### Primary metrics

For the 10% target, calculate:

- Brier score.
- Log loss.
- ROC AUC as a secondary discrimination metric.
- PR AUC because participation is imbalanced.
- ECE.
- Adaptive ECE.
- Maximum calibration error.
- Calibration slope.
- Calibration intercept.
- Reliability diagrams.
- Brier decomposition into:
  - reliability;
  - resolution;
  - uncertainty.

### Required slices

Report every metric for:

- all rows;
- QB;
- RB;
- WR;
- TE;
- cold-start rows;
- players with history;
- season;
- team;
- roster-backed versus fallback team rows;
- participation-label source;
- high- and low-history groups.

Do not rely only on aggregate averages. Include sample size and effective sample size in every slice.

### Calibration bin sensitivity

Repeat calibration analysis with:

- fixed 10 equal-width bins;
- fixed 20 equal-width bins;
- quantile bins;
- adaptive bins with minimum support;
- minimum bin sizes such as 100 and 250 rows.

A candidate should not pass only because one arbitrary binning scheme favors it.

## 6. Evaluate the conditional opportunity model

The second Phase 2 target is snap share conditional on meaningful participation.

Report:

- MAE.
- RMSE.
- Median absolute error.
- Mean signed error.
- Calibration of predicted continuous share.
- Prediction interval coverage if intervals are added.
- Error by position.
- Error by season.
- Error for cold-start versus established players.
- Error by actual snap-share range.
- Error compared with rolling-3 and rolling-5 baselines.

The primary acceptance requirement remains:

- candidate MAE must beat rolling-3 snap share;
- improvement should be evaluated with paired bootstrap confidence intervals;
- large degradation in any position should be reported even if aggregate MAE improves.

Also test the combined opportunity estimate:

`P(meaningful participation) × E(snap share | meaningful participation)`

This evaluates whether the two-stage system produces a useful unconditional expected snap share.

## 7. Implement calibration candidates correctly

### Raw GBM

Retain as the incumbent and reference model.

### Platt calibration

Use only as a comparator. It must:

- fit on a strictly prior calibration split;
- never fit and score on the same rows;
- be position-specific only if support is adequate;
- fall back explicitly when support is insufficient.

### Isotonic calibration

Use:

- chronological internal calibration split;
- minimum support threshold;
- explicit fallback;
- persisted knots and fitted values;
- monotonicity checks;
- sparse-region diagnostics.

### Pooled calibration

Evaluate pooling over:

- one prior season;
- three prior seasons;
- five prior seasons;
- exponentially weighted prior seasons.

The pooling window must be fixed before evaluation. Do not select the best window after inspecting test results without accounting for model selection.

### Hierarchical or shrinkage calibration

Consider:

- beta calibration;
- regularized beta calibration;
- empirical-logit calibration;
- hierarchical position calibration;
- partial pooling toward the position-rate baseline;
- calibrated residual correction rather than full-probability transformation.

These approaches may be more stable than unconstrained isotonic fitting when the high-probability region already works well.

## 8. Perform the QB curve audit

For every QB fold:

- Persist isotonic input thresholds.
- Persist fitted output values.
- Persist fit-row and held-out-row counts.
- Count knots/steps.
- Confirm monotonicity.
- Identify flat regions.
- Identify jumps larger than a predefined amount.
- Identify thresholds supported by too few calibration rows.
- Plot or tabulate the curve against the identity line.
- Compare raw GBM, calibrated GBM, and observed rates.

Repeat the same audit for RB, WR, and TE if the QB audit finds instability.

The audit must distinguish:

- a genuinely learned stable correction;
- a step function driven by a few rows;
- a mapping that merely shrinks probabilities toward a season-specific base rate.

## 9. Inspect all probability bins

For every candidate and every test season, persist:

- bin interval;
- sample count;
- mean predicted probability;
- observed rate;
- absolute calibration gap;
- contribution to ECE.

Pay special attention to:

- `0.9–1.0`, because it contains most rows;
- `0.6–0.8`, where prior miscalibration was observed;
- sparse low-probability bins;
- position-specific bins.

The report must answer:

- Did the candidate preserve the well-calibrated top bin?
- Did it improve the mid-range?
- Did improvements in small bins come at the cost of majority-mass bins?
- Did any candidate create new overconfidence or underconfidence?

## 10. Bootstrap uncertainty and non-inferiority

Use paired OOF rows.

For each position:

1. Resample player-week rows with replacement.
2. Recalculate candidate ECE.
3. Recalculate baseline ECE.
4. Store the paired difference.
5. Repeat at least 1,000 times, preferably 2,000.
6. Calculate percentile or BCa confidence intervals.
7. Record the probability that the difference is ≤0.
8. Record whether the upper confidence bound is below the preregistered margin.

Use the same approach for:

- Brier differences;
- log-loss differences;
- opportunity MAE differences.

Do not treat overlapping confidence intervals as the decision rule. Use the explicitly preregistered difference and margin.

## 11. Address multiple comparisons

There will be multiple positions, candidates, metrics, and calibration variants. Define in advance whether:

- only one primary candidate is tested;
- candidates are ranked descriptively;
- a family-wise error adjustment is applied;
- or a hierarchical decision sequence is used.

A clean approach:

- nominate one primary candidate before evaluation;
- treat all others as secondary/exploratory;
- apply simultaneous position-level confidence control to the primary gate;
- label all post-hoc candidate comparisons exploratory.

## 12. Test temporal stability

Calibration may fail because a prior-season correction does not transfer.

Report:

- per-season ECE;
- rolling multi-season ECE;
- calibration slope by season;
- calibration intercept by season;
- sign of the calibration gap by season;
- top-bin gap by season;
- mid-range gap by season.

A candidate should not pass solely because it performs well in aggregate while failing repeatedly in recent seasons.

Use stress tests for:

- rule changes;
- shortened seasons;
- unusual injury environments;
- COVID-era seasons;
- roster/data-source changes;
- recent-season distribution shifts.

## 13. Validate label and feature provenance

Confirm:

- no unknown rows are treated as negatives;
- snap labels come only from authoritative sources;
- team grouping does not use current-week outcomes;
- teammate features use only lagged outcomes;
- current-week status/injury/depth information is not silently introduced;
- roster fallback rates are reported;
- feature missingness is preserved and documented.

Run automated causal perturbation tests:

- change future outcomes;
- verify past features do not change;
- change a teammate’s current-week outcomes;
- verify focal player’s current-week competition features do not change;
- alter current-week labels;
- verify classifier inputs remain unchanged.

## 14. Define the final decision matrix

The final table should contain:

| Requirement | Decision |
|---|---|
| Brier versus simple baselines | Pass/fail |
| Log loss versus simple baselines | Pass/fail |
| ECE non-inferiority by position | Pass/fail |
| Confidence interval rule | Pass/fail |
| Cold-start reporting | Pass/fail |
| Unknown-label handling | Pass/fail |
| Conditional MAE versus rolling-3 | Pass/fail |
| Temporal stability | Pass/fail |
| Leakage checks | Pass/fail |
| Test suite | Pass/fail |
| Causal walk-forward checks | Pass/fail |

Acceptance should be binary under the preregistered rule. Do not convert a failed gate into a pass based on subjective overall judgment.

## 15. Define what happens if no candidate passes

If no calibrated candidate passes:

- retain raw `hist_gbm` as the offline incumbent;
- report probabilities as model scores with known calibration limitations;
- do not claim calibrated probabilities;
- avoid applying a post-hoc transformation;
- consider using rank/selection outputs instead of probability thresholds;
- revisit the baseline and gate only through a new preregistration;
- keep Phase 7 unchanged.

If raw GBM is operationally useful despite calibration limitations, document two separate uses:

- ranking/discrimination use;
- probability interpretation use.

Do not treat success in one use as evidence of success in the other.

## 16. Separate Phase 2 from Phase 3 and Phase 7

Phase 2 acceptance does not establish PPR improvement.

Before any serving promotion:

- run the Phase 3 matched-population experiment;
- verify the participation model improves PPR or decision utility;
- require reliable effect estimates and confidence intervals;
- check matched-population validity;
- check sensitivity to player exclusions;
- ensure no leakage from post-outcome variables;
- keep the Phase 7 gate separate.

A Phase 2 model may be accepted as an offline opportunity model without being allowed to alter serving predictions.

## 17. Required final artifacts

The next finalized evaluation should write:

- `phase2_preregistration_v2.json`
- `evaluation.json`
- `oof_predictions.csv`
- `calibration_bootstrap.json`
- `calibration_bins.csv`
- `calibration_curves.json`
- `calibration_curve_plots/`
- `season_position_metrics.csv`
- `opportunity_metrics.csv`
- `cold_start_metrics.csv`
- `feature_provenance.json`
- `causal_walkforward_checks.json`
- `phase2_decision.json`
- `phase2_manifest.json`

The manifest should include:

- database hash or version;
- code commit;
- model names;
- fold definitions;
- calibration settings;
- bootstrap seed;
- bootstrap replicates;
- non-inferiority margin;
- final decision;
- explicit statement that Phase 7 was not modified.

## 18. Recommended execution order

1. Freeze the preregistration.
2. Run data and pairing validation.
3. Run leakage and causal checks.
4. Generate OOF predictions.
5. Generate calibration curves and bins.
6. Compute point metrics.
7. Compute bootstrap intervals.
8. Evaluate temporal and subgroup stability.
9. Evaluate conditional opportunity.
10. Apply the fixed decision matrix.
11. Write the final manifest.
12. Only then consider whether a new Phase 3 integration experiment is justified.

The central principle is to separate three questions:

1. Does the model discriminate better?
2. Are its probabilities calibrated and statistically non-inferior?
3. Does using the model improve downstream fantasy outcomes?

Those questions require separate metrics, separate gates, and separate evidence.

## Phase 2 current decision record (2026-09-18)

Raw `hist_gbm` remains the Phase 2 offline incumbent. It met the completed
run's predictive-performance requirements, but it is not approved as a source
of calibrated probabilities and no teammate, Platt, isotonic, or
pooled-isotonic variant is promoted. Phase 7 serving predictions remain
unchanged.

The current decision highlights the calibration failure for QB and RB. The
preserved completed-run artifact also records failed original per-position ECE
gates for TE and WR; the QB/RB wording does not narrow or supersede that
artifact. Any revised acceptance rule is restricted to a future preregistered
evaluation and must be fixed before its results are inspected.

The machine-readable decision record is
`data/experiments/participation_system/phase2/phase2_decision.json`.

# Model Experiments Log

Running record of every model/feature/config comparison tested, with
full metrics (not just R²) and enough detail to reproduce or extend
each one. Started 2026-08-06 per explicit user request: R² alone isn't
trustworthy enough on its own to make production decisions (confirmed
directly this session — see "Known pitfalls" below), and there are many
more axes left to test (feature sets, lookback depth, model family,
component/util/fp target mode). This file is the answer to "what have
we actually tried and what did it show," kept current as new
experiments run.

**Convention**: every row must report R², RMSE, MAE, corr, and n
(train/test) at minimum. If a metric wasn't captured for a given run
(e.g. an expensive model wasn't worth re-running just to backfill RMSE),
write `n/a (not captured)` — never fabricate a number.

## Known pitfalls (read before trusting any single metric)

- **R² is unstable on small test sets and can point the wrong direction
  even when MAE improves.** Found directly this session: an early
  preseason-model test showed R² getting *worse* while MAE got *better*
  at the same time for RB — turned out to be a test-set size mismatch
  (17-50 rows vs. the real baseline's 104-373), not a real accuracy
  difference. Always check `n_test` before trusting an R² comparison.
- **A pandas `groupby()` silently drops rows where any key is NaN** —
  unlike SQL `GROUP BY`, which groups NULLs together. Cost an entire
  season of data (2025) and undercounted another (2024, down to 5 QB
  rows) in the preseason-candidate feature pipeline before being caught.
  Fixed with `dropna=False` in `src/models/preseason_features.py`. Watch
  for this in any new pandas-based feature assembly, especially against
  `player_weekly_stats`/`players`, where `birth_date` is null for a
  large fraction of rows (100% of 2025, 87% of 2024 in this DB).
- **Regularization strength can't fix a genuinely bad feature.** Swept
  Ridge alpha 10→200 on RB's multi-year feature set; R² plateaued at
  0.402, never recovering to the 0.447 achieved by simply excluding the
  multi-year features. If a feature hurts, more shrinkage isn't always
  the fix — sometimes it needs to not be there.

---

## 1. Current production baselines (real, measured)

### 1a. Weekly single-game prediction (`component` mode — the actual live architecture)
Source: real 2025 walk-forward backtest, `--target-mode fp` proxy for
`component` (same `CAUSAL_FEATURES`), 5,612 predictions, all 4 positions.
Also checked against this project's own `SUCCESS_CRITERIA` targets
(`config/settings.py`).

| Position | R² | RMSE | MAE | Spearman ρ | Within 10 pts | n |
|---|---|---|---|---|---|---|
| QB | 0.241 | n/a | n/a | 0.494 | 80.1% | 688 |
| RB | 0.341 | n/a | n/a | 0.641 | 89.3% | 1,483 |
| WR | 0.267 | n/a | n/a | 0.538 | 90.5% | 2,263 |
| TE | 0.236 | n/a | n/a | 0.536 | 94.8% | 1,178 |
| **Aggregate** | **0.335** | 6.36 | 4.73 | 0.605 | 89.8% | 5,612 |

SUCCESS_CRITERIA targets: Spearman ρ > 0.65 (aggregate misses: 0.605),
within-10pts ≥ 80% (aggregate clears: 89.8%), within-7pts ≥ 70% (clears:
78.7%). **Real gap: rank accuracy, not absolute error** — QB is worst at
ρ=0.494.

### 1b. Season-total prediction (`PreseasonProjector` — drives the live draft board)
Real 2023-2025 holdout (train on seasons ≤2022), recomputed with full
metrics for this table.

| Position | R² | RMSE | MAE | corr | n |
|---|---|---|---|---|---|
| QB | 0.291 | 103.6 | 84.7 | 0.541 | 104 |
| RB | 0.456 | 74.9 | 57.9 | 0.675 | 214 |
| WR | 0.485 | 64.3 | 50.7 | 0.709 | 373 |
| TE | 0.532 | 45.2 | 35.3 | 0.731 | 203 |

Architecture: position-specific Ridge, single prior season only, zero
team context. `RIDGE_ALPHA_BY_POSITION = {QB:14, RB:28, WR:24, TE:18}`
— hardcoded, no documented derivation found (checked docstrings, git
history, internal variant-selection code — none of it compares Ridge
against any other model family, only different Ridge configs).

---

## 2. Multi-year + team-aware preseason candidate

New feature set (`src/models/preseason_features.py`,
`build_multiyear_season_pairs`): each player's own prior 1/2/3 seasons
as separate columns + trend features, plus destination-team 3-season
rolling positional usage (reused from the weekly model's
`_add_dest_team_pos_profiles` logic). Same real 2023-2025 holdout,
train ≤2022, **production's own per-position Ridge alpha** used
throughout for fair comparison to §1b.

| Position | Variant | R² | RMSE | MAE | corr | n_train | n_test |
|---|---|---|---|---|---|---|---|
| QB | y1-only (control) | 0.323 | 98.9 | 78.5 | 0.592 | 437 | 128 |
| QB | y1 + dest-team | **0.385** | 94.3 | 74.4 | 0.634 | 437 | 128 |
| QB | full (multi-year + dest-team) | 0.287 | 101.5 | 80.0 | 0.575 | 437 | 128 |
| RB | y1-only (control) | 0.441 | 72.5 | 57.0 | 0.671 | 982 | 245 |
| RB | y1 + dest-team | **0.447** | 72.1 | 56.4 | 0.676 | 982 | 245 |
| RB | full (multi-year + dest-team) | 0.396 | 75.3 | 58.1 | 0.642 | 982 | 245 |
| WR | y1-only (control) | 0.554 | 57.8 | 45.6 | 0.747 | 1,480 | 400 |
| WR | y1 + dest-team | 0.563 | 57.2 | 44.9 | 0.752 | 1,480 | 400 |
| WR | full (multi-year + dest-team) | **0.565** | 57.1 | 45.1 | 0.756 | 1,480 | 400 |
| TE | y1-only (control) | 0.535 | 42.7 | 32.4 | 0.734 | 791 | 220 |
| TE | y1 + dest-team | **0.540** | 42.4 | 32.3 | 0.737 | 791 | 220 |
| TE | full (multi-year + dest-team) | 0.536 | 42.7 | 32.6 | 0.733 | 791 | 220 |

Bold = best Ridge variant per position. **Pattern: destination-team
context helps every position (small, consistent gain). Multi-year
history (y2/y3) only helps WR (barely) — hurts QB and RB under Ridge,
roughly a wash for TE.**

### Nonlinear model family (`PositionModel` — RF+XGBoost+LightGBM+Ridge
OOF-stacked ensemble, same class `MultiWeekModel` uses), full feature
set, flat alpha (not yet re-run at per-position alpha — R²/MAE only,
RMSE/corr not captured, expensive to retrain):

| Position | R² | MAE | n_test |
|---|---|---|---|
| QB | **0.397** | 74.4 | 128 |
| RB | 0.214 | 62.0 | ~275 |
| WR | 0.486 | 46.6 | ~490 |
| TE | 0.283 | 36.9 | 220 |

**QB is the one position where the nonlinear model on the full
(multi-year + dest-team) feature set clearly beats every Ridge variant**
(0.397 vs. best Ridge 0.385) — multi-year history seems to carry real
signal for QB, but only a nonlinear model extracts it; Ridge can't (full
feature set was QB's *worst* Ridge variant, 0.287). RB/TE get worse with
the nonlinear model — consistent with the small-sample overfitting
already diagnosed for TE (train/OOF RMSE ratio 4.08x) in the
multi-horizon test below.

### Verdict so far, by position (Ridge unless noted)
- **QB**: best = full features + `PositionModel` (R²=0.397, beats
  production's 0.291 by +0.106). Real, clear win.
- **WR**: best = full features + Ridge (R²=0.565, beats production's
  0.485 by +0.080). Real, clear win.
- **TE**: best = y1+dest-team + Ridge (R²=0.540, beats production's
  0.532 by +0.008). Small, real win.
- **RB**: best = y1+dest-team + Ridge (R²=0.447), still **short of**
  production's 0.456 by -0.009. Not yet a win — closest candidate found,
  not proven better. Remaining gap plausibly the calibration layer this
  quick harness skips (see §1b architecture note).

**Not yet shipped anywhere — these are candidate numbers, not a
decision.** See "Open experiments" below before promoting anything.

---

## 3. In-season rest-of-season model (`MultiWeekModel`) — NOT the same experiment as §2, despite both involving "multiple years/weeks"

**Read this before comparing numbers across §2 and §3 — they are
different models answering different questions, and a number being bad
in one says nothing about the other.** §2 is "predict next season's
total, before it starts, using multiple years of the player's past
plus their new team" (Ridge/PositionModel on `PreseasonProjector`-style
features). §3 here is "predict the *rest of the current season*, from
an early-week vantage point, using the live weekly model's own
features" (`MultiWeekModel`, a gradient-boosted ensemble). Different
model, different target, different task. **TE's multi-year §2 result is
positive (0.536, a small win) — the negative number below (-0.416) is
this separate §3 experiment, not §2.**

Predicts the rest of *this* season from an
early-season vantage point (target_18w is season-scoped, needs ≥14 of
18 future games within the same season — can't cross season boundaries,
so this is architecturally not comparable to §1b/§2's "predict next
season before it starts"). Real train ≤2022 / test 2023-2025 split,
`CAUSAL_FEATURES` (same as the live weekly model), no hyperparameter
tuning. RMSE/corr not captured for the naive baseline.

| Position | MultiWeekModel R² | corr | MAE | Naive (this-week×17) R² | MAE | n_test |
|---|---|---|---|---|---|---|
| QB | 0.049 | 0.367 | 50.5 | -2.838 | 101.1 | 208 |
| RB | 0.483 | 0.700 | 50.2 | -0.780 | 89.5 | 429 |
| WR | 0.406 | 0.656 | 43.9 | -1.627 | 90.0 | 608 |
| TE | -0.416 | 0.398 | 53.7 | -1.974 | 73.2 | 233 |

TE's negative R² traced to a specific 4.08x train/OOF RMSE overfitting
ratio (vs. 1.3-1.9x for every other position/horizon combination) —
smallest training set (922 rows) for that horizon, not a fundamental
architecture flaw. **Caveat on this whole table**: this run's feature
pipeline was missing 8 real `CAUSAL_FEATURES` columns (rookie-related:
`is_rookie`, `rookie_draft_value`, etc.) that a full production retrain
would include — likely understates real accuracy, not overstates it.

---

## Open experiments (not yet run — the "many more options" to work through)

- [ ] Re-run §2's `PositionModel` rows at production's per-position
      alpha equivalent / with hyperparameter tuning on, not flat/fast
      mode — current numbers used `tune_hyperparameters=False`.
- [ ] Test different lookback depths for QB/WR/TE (2 years vs. 3 vs. 4+)
      — 3 was picked as a reasonable default, never swept.
- [ ] Test the calibration layer (`UpstreamCalibrator` /
      veteran-elite / fragile-role patches) on top of the new feature
      set for RB specifically — the ~0.009 R² gap to production might
      close with calibration alone, without needing new features.
- [ ] `component` vs. `util` vs. `fp` target mode comparison for the
      *weekly* model — partially done in a prior session (found `util`
      mode has a broken stage-1 signal, R²=-0.305 on RB; `component`
      mode confirmed as the right current default) but not re-checked
      since v30's feature additions.
- [ ] Different model families beyond Ridge/PositionModel — e.g. a
      simpler single-model GBM instead of the full 4-way OOF stack, to
      see if `PositionModel`'s complexity is buying anything over a
      plain XGBoost for this task.
- [ ] A real walk-forward version of the §2/§3 evaluations (currently a
      single train/test split, like the rest of this session's
      preseason-projector calibration work) rather than one fixed
      holdout.
- [ ] RB-specific: try 1-2 year lookback instead of dropping multi-year
      entirely, in case 2-years-back (not 3) is the actual cutoff where
      RB signal goes stale.
- [ ] **Never-tested dormant models — see `MODELS.md` for the full
      inventory and why each is dormant.** None of these have real
      accuracy numbers anywhere in this repo:
  - [ ] `Hybrid4WeekModel` (`horizon_models.py`) — real trained
        artifacts exist for all 4 positions, never evaluated.
  - [ ] `DeepSeasonLongModel` (`horizon_models.py`) — same, never
        evaluated.
  - [ ] `MultiWeekModel` for RB/WR/TE using its own saved-artifact
        training config (this session's test retrained fresh rather
        than evaluating a saved artifact, since only QB had one).
  - [ ] `TouchdownRegressor` (`production_model.py`) — disabled based on
        a design argument (Huber loss already handles outliers), never
        actually compared head-to-head.
  - [ ] `BayesianPlayerModel` (`bayesian_models.py`) — only unit-tested
        for running, not for prediction quality.
  - [ ] `MonteCarloSimulator` (`advanced_modeling.py`) — real
        implementation, never fed real player data at all.

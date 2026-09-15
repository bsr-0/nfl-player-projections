# Audit Report — 2026-09-09

> **Status (2026-09-15): all 25 findings resolved.** The body below is the
> original report, kept as written. Don't re-investigate from it -- verify
> against the current code. Resolution by finding (commit):
>
> | # | Commit | | # | Commit |
> |---|---|---|---|---|
> | 1 | `5a514ce` | | 14 | `d148f2f` |
> | 2 | `37ad02d` | | 15 | `4dc1866` (served ensemble now beats every baseline but one blended heuristic, -1.2%) |
> | 3 | `7608895` badge, injury feed backfilled, `d02e824` refresh entrypoint | | 16 | `4dc1866` |
> | 4 | `8ff75bc` | | 17 | `d8d5751` |
> | 5 | `77045ca` | | 18 | `dc02cc6` (file was orphaned, deleted -- not a mistyped number) |
> | 6 | `afce5d3`, then `06073e4` + follow-up for name collisions the first fix introduced | | 19 | `f307bdb` |
> | 7 | `bfc22c2` | | 20 | this commit -- a 2026-09-14 re-check wrongly marked it fixed; `tuning.py` still fed `num_leaves` to the sklearn fallback |
> | 8 | `f2ec505` | | 21 | `f3bd16b` |
> | 9 | `f2ec505` | | 22 | `77045ca` + `e6dd651` |
> | 10 | `6d36642` | | 23 | moot -- `CLAUDE.md` fully rewritten `d6eebab`; no freeze policy exists now (see README) |
> | 11 | `d02e824` -- manual entrypoint by design; no DB for cron/CI to run against | | 24 | `84fdd21` |
> | 12 | `d148f2f` | | 25 | `e6dd651` |
> | 13 | `d148f2f` | | | |
>
> Found while closing these, not in the original report: the Spearman
> "target > 0.65" had been measuring a 50-row cross-position slice (#17);
> `pass_plays`/`rush_plays` were desynced for 2023-24 only (`b1e00ea`);
> nflverse's live injury feed dropped `date_modified` (fails safe); the
> first ADP fix put Jonathan Taylor at ADP 364 via a name collision.

**Question audited:** does this repo deliver a robust, accurate fantasy football prediction system with a well-tested, optimized backend and a UI that gives real value to team owners?

**Short answer:** not yet. The research core is careful and unusually honest, but the product that ships today is a static PPR draft board with duplicated rows and no bye weeks, the weekly ML model has never been shown on a full season to beat a trailing average, and the data write path silently destroys two of the inputs the model depends on. Week 1 of 2026 starts tomorrow and the calendar code thinks it is week 2.

**Method:** six parallel code audits (serving path, features/leakage, data layer, evaluation, UI, hygiene), each finding re-verified by reading the cited lines; full `tests/` run; Playwright load of every Pages page; JSON inspection of `docs/data/`. `data/nfl_data.db` and the weekly `.joblib` models are gitignored and absent, so nothing in the production path could be executed end to end. Findings below are code- and artifact-level.

## Scorecard

| Area | Score | One-line verdict |
|---|---|---|
| Serving path / model core | 5/10 | No target leakage found; output layer ships verifiable errors today. |
| Feature engineering | 5/10 | Rolling machinery is causal; two declared features carry future information. |
| Data layer | 4/10 | Good schema, but the hottest write path re-zeroes snaps and wipes Vegas lines. |
| Evaluation / accuracy claims | 4/10 | Only full-season backtest of the served ensemble type loses to every baseline; headline numbers come from a different model. |
| Tests | 5/10 | 686 pass, but DB-dependent tests error instead of skip, and the suite mutates committed files. |
| UI value to owners | 3/10 | One usable feature (VOR board). No byes, injuries, real ADP, weekly numbers, or full names. |
| Codebase hygiene | 4/10 | ~12.8K LOC dead, six parallel implementations, project memory describes a repo that no longer exists. |
| **Overall** | **4/10** | A strong research notebook; not a robust production system. |

## Critical and High findings (verified)

### Correctness of what ships today

1. **Critical — Calendar is one week early for 2026.** `src/utils/nfl_calendar.py:26-35` takes "first Thursday of September" (Sept 3); real kickoff is Sept 10. Verified: `get_current_nfl_week(2026-09-10)` returns week 2. `generate_app_data.py` will stamp week-1 rows as week 2 and look up week-2 opponents. Same failure in 2027.

2. **High — Duplicate players on the live board.** 19 duplicated `player_id`s across `docs/data/players_*.json`; five appear on two teams with identical projections (T.Lockett TEN+LV, B.Cooks BUF+NO, D.Johnson ARI+NE, …). Cause: `scripts/generate_draft_data.py:448` groups by `(player_id, name, team, position)`, so a 2025 mid-season trade yields two rows, and `apply_current_teams` re-teams both. Propagates to every weekly file. Verified by script.

3. **High — Weekly page is 18 copies of one number.** All `predicted_points` in wk2–wk18 equal wk1 to the cent (488 shared rows, 0 differences). `weekly_meta.json` says `season_prorated` = season total / 17. The banner discloses this, but there is no start/sit, matchup, or range signal for any week. Commit 718c035 measured this basis at −1.1 to −1.6 pts bias for veterans and fixed it only in the private ESPN report.

4. **High — "Rest of season" is a 4-week sum.** `TRAINING_HORIZONS = [1, 4]` (`config/settings.py:747`), but `scripts/generate_app_data.py:115` still requests horizon 18. `MultiWeekModel.predict` (`src/models/position_models.py:1553-1567`) silently picks the closest trained key (4) with no scaling; `predict.py` then divides by 18 for ppg.

5. **High — QB prior-season totals on the board are ~22% low.** Median QB ratio of `prev_season_total_fp` to backtest actual is 0.78 (RB/WR/TE ≈ 1.0); J.Allen shows 226.6 vs 426.8. Not root-caused without the DB. Related: `nfl_data_loader.py:638-657` omits `two_point_conversions` from the label although `SCORING` defines it.

6. **High — "ADP" column is the model's own rank.** `generate_draft_data.py:824` sets `adp = rank`. Values are 1..N and perfectly monotone with projection. The board cannot show market-vs-model value, which is the main draft-day use.

7. **High — Bye, injury flag, and age are hardcoded off.** `generate_draft_data.py:823,836,837` write `None/False/None` for every row; `draft.html` renders the columns as "—". Bye data already exists in the schedule join used by `generate_weekly_data.py:79-96`.

### Data integrity

8. **Critical — Weekly ingest re-zeroes snap counts.** `nfl_data_loader.py:113` (`_to_scalar_int`) maps NaN to 0; `:715-717` applies it to `snap_count`/`team_snaps`; `database.py:840` is `INSERT OR REPLACE`. The NULL-vs-0 invariant pinned by `tests/test_snap_null_semantics.py` is destroyed at write time and only restored by a manual script. The column list also omits `data_source`, so provenance is nulled on re-ingest. No test covers this write path.

9. **Critical — Schedule reload wipes Vegas lines.** `nfl_data_loader.py:879-890` builds schedule rows without `spread_line`/`total_line`; `database.py:1048-1062` REPLACEs with those columns as NULL. `auto_refresh.py:213-217` reloads the schedule for any season newer than local stats (true for 2026 on every run until week 1 lands), and auto-refresh runs at the top of every training entry. Vegas features then collapse to the documented constants (spread 0, implied total 23).

10. **High — Quality gates cannot block anything.** `data_manager.py:390-400` warns and proceeds on FAIL; `train.py` imports `run_quality_gates` but never calls it; `auto_refresh.py:252-262` runs the gate after commit. `quality_gates.py:113-126` has no regular-season filter, so with week 22 ingested it always reports 30 missing teams — which is exactly what `data/models/data_quality_gate_refresh.json` shows (`status: fail`).

11. **High — No orchestrated refresh.** No `.github/`, no cron. Rosters, depth charts, injuries, Vegas, snaps, and props are eight separate scripts, several defaulting to 2025 (`backfill_vegas_lines.py:143`, `backfill_injuries.py:66`, `backfill_weekly_rosters.py:73`). `backfill_all_data.py:369-373` prints FAILED then `Done.` with exit 0.

### Leakage

12. **High — Contract features use the newest contract for every historical row.** `feature_engineering.py:1264-1272` keeps max `yr_signed` per player with no season bound; `:1311-1314` derives `is_contract_year` and `contract_apy_rank` from it. A 2019 row for a player who signed a big 2023 deal reads a 2023 APY percentile. Both are in every position's `CAUSAL_FEATURES`.

13. **High — `injury_score` is not kickoff-filtered in the live path.** Causal path merges `player_injuries` (`feature_engineering.py:4358-4488`), written by `backfill_injuries.py:104-124` with no `date_modified` check; the guard exists only in `external_data.py:181-192`, which the cache overrides (`combine_first` at `:4467`). Serving sets synthetic rows to 1.0 (`season_projection.py:540-567`). `leakage.py:288` documents the opposite.

14. **High — `prev_season_ppg` is not prior-season PPG.** `feature_engineering.py:750-756`: within-season expanding mean, then shifted once per player. Result: week 1 = mean of a few prior-season weeks, week 2 = NaN for every player every season (history-missingness preservation skips the fill), week 3+ = current-season lagged mean. `bayesian_prior_ppg` and `career_year_flag` inherit it. Docstring and `leakage.py:262` misdescribe it.

### Evaluation

15. **Critical — The served ensemble has no full-season holdout, and the only one of its type says it loses.** `data/backtest_results/backtest_2025_20260801.json` (`model_source: production_ensemble`): R² −0.24, Spearman −0.056, RMSE 8.28; loses to trailing-3g (−19.5%), trailing-5g (−23.9%), season-avg (−25.4%); `model_has_real_edge: false`. That run predates the 2026-08-29 retrain and its QB block is visibly broken (342 features, avg prediction 0.81), and nothing in GAPS/TRACKING acknowledges it. `model_metadata.json` for the current model has `test_metrics: {}`; its OOF R² (0.26–0.36) is stacking CV inside the training window with no baseline.

16. **High — Published accuracy numbers come from a different model than the one served.** `scripts/generate_results_page.py:45-50` picks the last `ts_backtest_2025_*.json` regardless of model; that backtester runs Ridge, not the stacked GBM ensemble, and does not record `model_type`. The three-week serving-path check (MAE QB 7.09 / RB 4.43 / WR 4.03 / TE 3.00, hardcoded in `generate_weekly_data.py:58-63`) has no surviving artifact and is contradicted by `docs/data/backtest_2025.json` (6.45 / 4.76 / 4.66 / 3.81).

17. **Medium — Spearman is pooled and top-50-restricted.** `metrics.py:38-47` defaults `top_n=50` on the pooled season. Recomputed from `ts_backtest_2025_20260809_001736_predictions.csv`: within-week ρ QB 0.45 / RB 0.66 / WR 0.55 / TE 0.55; top-24-by-prediction within position-week 0.28. The project's own target is ρ > 0.65. Inactives are excluded by construction, so the backtest never pays for starting a player who did not play.

18. **Medium — `backtest_metrics.json` thresholds accept a negative R²** (`min_overall_r2: -0.2`) and no code reads the file.

### Tests

Run: `python -m pytest tests` with pinned deps, no DB → **686 passed, 8 failed, 22 skipped, 2 errors**. Bare `pytest` at root (as README says) fails collection on `scripts/test_espn_connection.py` (no `espn_api`).

19. **High — Test run mutates committed state.** Tests connect to `DB_PATH` directly; sqlite creates an empty `data/nfl_data.db`, and `DataManager` then overwrote the committed `data/data_availability_cache.json` with `available_seasons: [2026]`. Reverted by hand.
20. **Medium — Real bug exposed by a failing test.** `src/models/single_week_ppr/architectures.py:91-97`: when lightgbm is absent, the sklearn fallback receives LightGBM-only params (`num_leaves`, `colsample_bytree`, …) and raises `TypeError`. The fallback has never worked.
21. **Medium — DB-dependent tests error rather than skip** (10 of 58 files); `tests/test_structural_missingness_pre_era.py:238` calls `pytest.skip` without importing pytest.

### Hygiene

22. **High — ~12.8K LOC dead.** 22 `src/` modules with zero importers (9,180 LOC: `advanced_ml_pipeline.py`, `advanced_models.py`, `advanced_modeling.py`, `bayesian_models.py`, `model_improver.py`, `pipeline.py`, …) plus transitively dead `production_model.py`, `train_advanced.py`. Of 134 scripts, ~25 are demonstrably live; 46 are referenced by nothing at all.
23. **High — Project memory is wrong.** `CLAUDE.md §9` freezes eight files; seven do not exist and tag `ui-stable-v1` does not exist. `GAPS.md` is 783 KB / 13,968 lines and README says "read this first". `PROJECT_NOTES.md:13` still calls component mode production; `settings.py:293` sets every position to `fp` and `ensemble.py:319-328` discards the component JSONs with a warning.
24. **Medium — Scoring is defined twice inside settings** (`SCORING` :134, `PPR_SCORING_WEIGHTS` :694, the latter without 2PC) and re-inlined in four more files. `nfl_data.db` path is rebuilt by hand in 36 places. 450 `except Exception`, 91 followed by `pass`/`continue`; 1,231 `print(` lines in `src/`.
25. **Medium — Broken "runnable" scripts.** `production_retrain_and_monitor.py:157` imports a symbol that does not exist; `run_dashboard.sh` runs a file that does not exist; `compute_confidence_tiers.py:18-19` and `model_metadata.json` hardcode `/Users/benrosen/...`.

## What is genuinely good

- Every player-history rolling feature in the causal path is `shift(1)` before `rolling` on a sorted `(player_id, season, week)` frame; targets are structurally excluded and asserted; `as_of` truncation happens before feature engineering, so the serving path is backtestable.
- `FEATURE_VERSION` mismatch is fatal at load. Step 8's `E[PPR | played] × P(plays) × games` decomposition is sound, and its rookie path is principled.
- Decisions are measured, not asserted: component mode was retired on a 12-of-12-fold loss; the week-1 cold-start rule beat four alternatives with paired-bootstrap CIs and seven nulls were recorded honestly.
- `team_stats` COALESCE upsert, `_assert_no_history_lost`, strict pre-store validation, era-aware NaN masking, and the pre-commit secret hook are all well done.
- The site loads with zero JS errors, works at 390px, and its banners are more candid than most commercial tools.

## Fix order

1. Fix `_season_start` (Thursday after Labor Day) before Sept 10.
2. Dedupe the board by `player_id` and take the current-roster team.
3. Stop `insert_player_weekly_stats` from zeroing NULL snaps; carry `spread_line`/`total_line` through the schedule reload (or COALESCE both). Add a test for each write path.
4. Remove horizon 18 from `generate_app_data.py` or raise when a requested horizon is untrained.
5. Bound contract lookup by `yr_signed <= season`; kickoff-filter `player_injuries` at write time; rename or fix `prev_season_ppg`.
6. Run one full-season walk-forward of the served ensemble against trailing-3g/season-avg/Vegas baselines and record `model_type` in the artifact. Until it wins, keep serving Step 8 and say so.
7. Wire byes, injury status, full names, and real ADP into `players_*.json`; state "PPR" on every page.
8. Make quality-gate failure block training; add one scheduled refresh entry point.
9. Make tests skip on missing DB, never touch `DB_PATH` or committed files; fix the sklearn fallback params.
10. Delete the 25 dead modules and ~46 orphan scripts; replace `GAPS.md` with a 300-line current-state digest; rewrite `CLAUDE.md §9` to the files that exist.

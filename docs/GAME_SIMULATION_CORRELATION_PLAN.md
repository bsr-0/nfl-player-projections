# Game Simulation and Player Correlation Plan

Status (2026-09-27): OOF capture, calibrated marginals, factor-copula
dependence and the rolling backtest driver are built and verified on
synthetic panels; the real-panel run is pending (needs local data). See
"Calibrated simulation v2" at the end of this file and in GAPS.md.
Branch: feature/game-simulation-correlation

## Objective
Extend weekly PPR projections from independent player estimates to coherent game-level outcomes while preserving existing models and leakage-safe validation.

## Workstreams
1. Game-script simulation: sample a correlated game total and margin, derive team scores, adjust pass rate and pace by score state, allocate plays and attempts, then sample player usage conditional on that script.
2. Correlation layer: fit same-game residual covariance from out-of-fold player predictions, apply shrinkage and a positive-semidefinite projection, and draw shared residuals on top of existing point predictions.
3. Integration: add adapters for existing game-outcome artifacts and player outputs, build historical OOF panels, add a production command/output schema, then expose behind a feature flag.

## Data contracts
GameScriptInput contains game IDs, teams, win probability, predicted margin/total, team volume baselines, and uncertainty.
PlayerSimulationInput contains player/team/position, expected fantasy points, uncertainty, usage share, and participation probability.
SimulationResult contains draw ID, scores, team volume, player activity, and fantasy points.

## Leakage and validation gates
- Same seed and inputs are bitwise deterministic.
- Scores, probabilities, and volume are bounded; pass attempts never exceed plays.
- Correlation fitting accepts only caller-supplied OOF residuals and requires at least two games.
- Fitted covariance is positive semidefinite.
- Disabled mode preserves current point predictions exactly.
- Compare future-holdout individual MAE plus joint log/energy score and teammate correlation calibration.
- Enable only if joint calibration improves without unacceptable individual-player degradation.

## Ordered implementation
1. Land database-independent contracts, simulation primitives, and tests (this branch). **Done** -- `game_simulation.py`, `player_correlation.py`, `simulation_adapter.py`, `simulation_schema.py`, `simulation_io.py`, `simulation_evaluation.py`, `simulation_readiness.py`. Five real correctness bugs found and fixed 2026-09-22 against real data shapes (NaN cold-start crash, QB/WR share-pool conflation, all-or-nothing usage-share coverage, plus two pre-existing test bugs) -- see git history same date.
2. Add DB/model-artifact adapters. **Half done.** `simulation_adapter.py` itself is built and unit-tested against hand-built fixtures. What's still missing is real data flowing INTO it -- see "Phase 2 scoping" below.
3. Build OOF residual panels with season cutoffs and audit coverage. **Not started -- scoped 2026-09-23, see below.** This is the actual blocker for everything after it.
4. Add production simulation command and JSON/Parquet output. Schema/IO layer built (`simulation_schema.py`, `simulation_io.py`); the CLI (`scripts/generate_simulation_data.py`) exists but `--production` mode is hard-disabled (raises `RuntimeError`) because the artifacts step 3 would produce don't exist yet.
5. Wire weekly generation and UI behind an explicit flag. Not started; correctly gated on 3-4.
6. Run ablations against independent and shared-residual baselines. Not started; correctly gated on 3-5.

## Phase 2 scoping (2026-09-23): the OOF residual panel gap

Investigated what step 3 above actually requires, before writing any code for
it. Headline finding: **no persisted out-of-fold player-level prediction
artifact exists anywhere in this repo**, for either weekly-model
architecture. `position_models.py` (the model `EnsemblePredictor`/`NFLPredictor`
actually serves -- see GAPS.md's 2026-08-29 `fp`-mode entry) computes OOF
predictions internally for meta-learner stacking and isotonic calibration,
but only ever keeps aggregate metrics; the row-level (predicted, actual)
pairs are discarded once training finishes. `single_week_ppr`'s Phase 4 did
save row-level predictions once (`data/experiments/phase4_row_level_predictions.csv`),
but that's a one-off artifact from a research architecture that was never
wired into serving (it has zero `joblib.dump`/`.save()` calls anywhere --
confirmed while scoping this -- so it can't be the source of "real"
residuals for a simulator meant to sit on top of production). The
correlation layer's stated design ("fit same-game residual covariance from
out-of-fold player predictions") therefore has nothing to fit on yet, for
the model that would actually be simulated.

One adjacent piece already exists and is reusable as-is: game-level OOF
predictions (`data/experiments/game_outcome_oof_predictions.csv`, built by
`scripts/ablate_game_outcome_features.py`, walk-forward with season cutoffs
respected) cover the `game_predictions` half of
`simulation_adapter.game_inputs_from_predictions`'s input. Only the
player-level half is missing.

### Concrete work items, in order

1. **Row-level OOF capture for the served weekly model.** `train.py
   --walk-forward`'s `_run_one_fold` currently returns only aggregate
   `by_position` metrics and discards every row's individual prediction.
   Needs a parallel capture path (mirroring `ablate_game_outcome_features.py`'s
   `OOF_CACHE` pattern) that persists `(player_id, season, week, team,
   opponent, position, predicted_points, actual_points)` per row across the
   walk-forward folds. This re-runs the same multi-hour-per-fold training
   already exercised for the Vegas-fix retrain (see GAPS.md/PROJECT_NOTES
   for that timing) -- expensive, but a one-time cache, not a per-use cost.
2. **Correlation fitting from that panel.** Group by `(season, week, team)`
   for same-game player residual vectors, fit via `fit_role_residual_correlation`
   keyed by role (`role_keys_for_players`-style: `home_WR1`, `away_RB2`,
   etc.), NOT raw `player_id` -- `INDEPENDENT_SIMULATION_AUDIT.md` already
   flags "Player-ID covariance is not portable across lineups" as a named
   risk, and role-keying is the existing, already-built interface for
   avoiding it (`RoleResidualCorrelationModel`). Serialize the result as the
   `role_correlation_artifact` `simulation_readiness.production_readiness()`
   already expects but has never been given.
3. **A real backtest driver.** A script analogous to `evaluate_simulation.py`
   but that builds its own inputs end-to-end for a held-out season: pull
   that season's game + player OOF predictions, adapt via
   `simulation_adapter.py`, run `simulate_players`, score against real
   actuals via `simulation_evaluation.py`'s CRPS/energy/variogram gates.
   This is what would finally answer "does shared-residual simulation beat
   independent per-player prediction" (the plan's own §Leakage-and-validation
   promotion criterion) instead of only proving the code runs.

### Deliberately out of scope for this next step

`INDEPENDENT_SIMULATION_AUDIT.md`'s full component list (possession/drive
modeling, a true stat-line conversion layer, TD-scarcity allocation,
availability/role modeling) is real and each item is individually valid,
but building any of it before knowing whether the cheap "shared Gaussian
residual on top of existing point predictions" version clears its own
promotion bar would be the same mistake Plan A's decision gate exists to
prevent -- prove the cheap version first.

### Open risk carried over from the Vegas-retrain investigation (2026-09-23)

Item 1 above must fit residuals from whichever model is ACTUALLY served,
not a research harness -- this plan deliberately targets `position_models.py`
(`fp` mode), not `single_week_ppr`, for exactly the reason the Vegas-retrain
investigation surfaced: the two architectures are unconnected in code and
have materially different accuracy, and conflating them silently is the
kind of cross-harness mistake GAPS.md has documented more than once (e.g.
the 2026-08-29 log1p-space metric mixing entry). Confirm which model is
current production immediately before building the capture path, don't
assume it's still true by the time this is picked up.
### Work item 1 built (2026-09-25): row-level OOF capture

`src/models/oof_capture.py` + wiring in `train.py`'s walk-forward loop.
Each fold trains on seasons 1..N-1 and predicts season N, so its test-set
predictions are genuinely out-of-fold; the folds concatenate into
`data/experiments/walk_forward_oof_predictions.parquet` with
`(player_id, season, week, team, opponent, position, predicted_points,
actual_points, residual)` plus the fold's training seasons.

`team` and `opponent` are retained specifically for work item 2 -- the
correlation layer groups residuals by `(season, week, team)` to form
same-game vectors, and would otherwise have to re-join them.

The leakage conditions are enforced rather than assumed, because a
contaminated panel makes every downstream number look better than it is and
does so silently: `capture_fold_rows` raises if a fold's test season is also
a training season or if the frame carries rows from any other season, and
`build_panel` raises if two folds predict the same player-week (overlapping
test seasons would double-count rows and reweight every aggregate).

Also included is the segmentation this was originally asked for: rows carry
`prior_weeks_in_panel` (strict shift -- a row never counts itself) and
`is_cold_start` (a player's first appearance in the panel), and
`segment_report()` breaks MAE/RMSE/**bias**/n down by any grouping. Bias is
reported alongside the error magnitudes because a change can leave MAE flat
while shifting the whole distribution, which is precisely what the aggregate
four-number summary hid.

Note the segmentation is panel-relative, not career-relative: it answers
"how much had this model seen of this player by this point in validation",
which is the question segment evaluation asks. A career-relative split needs
pre-panel history and is a different metric -- do not conflate them.

**Not yet populated.** The artifact only appears after a walk-forward run
executes with this code. The two arms running the Vegas A/B started before
it landed, so they will not produce one; the next walk-forward run will.

Remaining for the correlation layer: work item 2 (fit
`fit_role_residual_correlation` keyed by role, not `player_id`) and work
item 3 (the backtest driver). Both now have their input.

### Methodology review (2026-09-25): five problems found and fixed

An adversarial review of the OOF panel design (one lens completed before the
reviewing session hit a rate limit; every load-bearing claim was
independently re-verified against `data/nfl_data.db` before acting on it --
composition percentages, zero-rates, and skew all replicated within
rounding) found five real problems in the first version:

1. **`is_cold_start` confounded with early-season weeks and the shortest
   fold.** Measured: of ~930 panel-defined cold-start rows, 65% are week<=2
   (vs 12% of returning rows) and 63% fall in the 2023 fold, the shortest
   training window; for QB specifically, 73% of the "cold-start" cell was
   2023 opening-day veterans, not new players. `week_bucket` now surfaces
   this in the default segmentation, and `add_career_experience_segments`
   builds the metric the module always claimed to describe (using real
   `player_weekly_stats` history, best-effort, fails open).
2. **No uncertainty on non-independent rows.** `segment_report(..., cluster="player_id")`
   now reports a cluster-bootstrap 95% CI, so a segment table stops looking
   like eight independent verdicts.
3. **No provenance, single overwritable path.** `write_run_panel` stamps
   `git_commit`/`generated_at`/`label` and writes an immutable per-run copy
   under `data/experiments/oof_panels/<run_id>/`, the same discipline commit
   ca4b5e1 added for model artifacts one commit before this module shipped
   without it. `train.py --oof-label` lets a deliberate A/B tag its two runs.
4. **Dropped rows unrecorded.** `fold_coverage` persists offered/captured/
   dropped counts per (fold, position) alongside the panel.
5. **MAE alone on a zero-inflated outcome.** `segment_report` now also
   reports `zero_rate`, `mae_positive` (conditional on a nonzero actual),
   and Spearman rank correlation.

**The missing consumer, also built:** `scripts/compare_oof_panels.py`
inner-joins two run-scoped panels on (player_id, season, week), computes the
per-row paired |error| delta, and reports cluster-bootstrapped segment
deltas -- refusing to report anything if the row overlap is too small to
mean anything. Without this, the panel could describe one run but not
compare two, which was the entire stated purpose.

**Deliberately not automated:** multiple-comparison correction across
segment cells, and a minimum-detectable-effect column. The CI half-width
serves the same purpose a reader needs most (a wide interval says "this cell
can't support a conclusion"); the right correction depends on which cells
are primary vs exploratory for a given comparison, which the tooling cannot
know on its own. Documented in `compare_oof_panels.py`'s output instead.

### Code review findings (2026-09-26): two open risks for work item 2

Reviewed `player_correlation.py` and `game_simulation.py` closely ahead of
fitting real data. The core math (linear shrinkage toward the diagonal,
eigenvalue-floor PSD projection, correlation-then-rescale-by-marginal-sd) is
internally consistent and unaffected by these. Two risks are specific to
role-keying and aren't caught by the existing synthetic-fixture tests, so
they're recorded here to check once work item 2 actually fits real data:

1. **Role-key tie-breaking degrades for low-usage players.**
   `role_keys_for_players` (`game_simulation.py`) ranks players within a
   `(side, position)` group by `usage_share is not None`, then
   `usage_share`, then `mean_fantasy_points`, then `player_id`. Backup/depth
   players with no `usage_share` and similar `mean_fantasy_points` fall back
   to sorting by `player_id` -- an identifier with no relationship to actual
   role. Across many historical games this means a role like `home_RB2` is
   trained on a mix of whichever depth player happened to sort second
   alphabetically/numerically, not a stable role, which dilutes any real
   correlation signal for that slot.

   **First pass (2026-09-26, superseded below):** built
   `role_assignment_diagnostics(game, players)` to flag roles assigned with
   no `usage_share` at all, on the reasoning that `PlayerSimulationInput`
   has no better role-magnitude signal to tiebreak on. True as far as it
   went, but incomplete -- it didn't yet account for how the correlation
   model these roles feed actually gets fit.

   **What that first pass missed, found while reviewing the other session's
   new fitting-side code:** `src/models/residual_calibration.py`'s
   `role_keys_from_panel_game` -- which builds the role-keyed rows
   `fit_sparse_role_residual_correlation` actually fits on, from the OOF
   panel -- ranks by `predicted_points`/`player_id` only. It has no
   `usage_share` to rank on, because `oof_capture.py` never captures that
   column. So `role_keys_for_players`'s `usage_share`-aware ranking wasn't
   just missing a tiebreak improvement -- it was actively **inconsistent
   with the fitting side**: the same real player could rank as `home_WR1` at
   serve time (usage_share available this week) but have been fit into the
   `home_WR2` slot historically (no usage_share existed in the panel), so
   the model's fitted `home_WR2` correlation column would get applied to
   the wrong player. A role label only means anything if fit time and serve
   time agree on how it's assigned; a better serving-time-only signal that
   the fitting side structurally can't see makes this worse, not better.

   **Fixed (2026-09-26, same day):** `_grouped_role_assignments` now ranks
   on `mean_fantasy_points`/`player_id` only, matching
   `role_keys_from_panel_game`'s convention exactly (verified:
   `role_keys_for_players`'s only production caller is `simulate_players`'s
   `RoleResidualCorrelationModel` lookup, so this has no other blast
   radius). `role_assignment_diagnostics` was redefined to match: it now
   flags a role key as diluted only when its occupant has a genuine tie in
   `mean_fantasy_points` with another player in the same `(side, position)`
   group (the literal trigger for the arbitrary `player_id` tiebreak),
   rather than "no `usage_share`" as a proxy for it. Tests updated
   (`tests/test_game_simulation_correlation.py`): a new test pins that
   `usage_share` no longer affects ranking even when it would reorder
   players relative to points, and the diagnostics tests were rewritten
   around genuine point ties instead of `usage_share` presence.

   **If `usage_share` is ever added to `oof_capture.py`'s captured columns,**
   both `role_keys_from_panel_game` and `_grouped_role_assignments` should
   start using it together, not just one -- update both or neither.

   **Still gated on work item 1 (the real OOF panel) for the empirical
   check:** once real per-game residual rows exist, correlate
   `role_assignment_diagnostics`'s per-role tie rate with that role's fitted
   correlation magnitude -- a role with a high tie rate showing near-zero
   fitted correlation is this mechanism, not evidence that depth players are
   truly uncorrelated.
2. **No consistency check between `home_win_prob` and `predicted_margin`.**
   `_draw_margin` (`game_simulation.py`) is internally self-consistent (the
   unconditional expected margin algebraically equals `predicted_margin`
   given `home_win_prob`), but nothing checked whether the two inputs
   plausibly come from the same underlying game. If they disagree (e.g. a
   1% home win probability paired with a strongly positive predicted home
   margin), the solved exponential means become extreme to reconcile them,
   producing a silently bimodal, unrealistic score distribution rather than
   a warning.

   **Fixed (2026-09-26):** `implied_win_probability_from_margin(predicted_margin,
   margin_sd)` (`game_simulation.py`) treats margin as
   `Normal(predicted_margin, margin_sd)` and returns the win probability that
   implies -- an independent second estimate of the same quantity
   `home_win_prob` is. `margin_win_probability_disagreement(game)` returns
   the absolute gap between the two. `simulate_game_scripts` now logs a
   `WARNING` (not a hard failure -- the two heads can legitimately differ
   somewhat, and a raise would break any caller whose inputs disagree only
   mildly) whenever that gap exceeds
   `MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD` (0.20, chosen so existing
   test fixtures' mild disagreements -- e.g. .75 win prob against a 3-point
   margin at `margin_sd=10`, which implies ~.62 -- don't fire, while a 1%
   win probability paired with a double-digit positive margin, which
   implies ~.84, does). Tested
   (`tests/test_game_simulation_correlation.py`): known Phi values including
   the `margin_sd=0` step-function edge case, the disagreement calculation
   on a consistent vs. an inconsistent game, and that `simulate_game_scripts`
   actually emits (or withholds) the log warning via `caplog`.

   Not yet wired to anything beyond a log line -- `simulation_adapter.py`
   does not currently surface or aggregate these warnings across a slate,
   so today this is only visible to someone reading logs for a specific
   game. Aggregating a rate across `simulation_adapter.py`'s real weekly
   game population, once it exists, is a natural follow-up but wasn't done
   here since it doesn't yet have real week-over-week data to be useful on.
3. **`player_inputs_from_predictions` silently collapsed duplicate games for
   the same week/matchup.** `simulation_adapter.py` indexes games by
   `(season, week, frozenset(home_team, away_team))` to route each player
   row to its scheduled game -- deliberately undirected, per its own
   comment, so a `home=A,away=B` row and a `home=B,away=A` row for what
   should be the same real game still match. But nothing checked whether
   two *different* `game_id`s collided on that same key.
   `game_inputs_from_predictions` only rejects an exact duplicate
   `(season, week, home, away)` tuple, so two distinct game_ids for one
   real matchup/week (e.g. a data-quality issue upstream) were not caught
   there either -- the dict assignment silently kept whichever game_id was
   processed last, routing all of that matchup's players to it and leaving
   the other game_id with none (confirmed live: reproduced with two
   `GameScriptInput`s sharing season/week/teams -- the first's players
   vanished with no error, and the final `if players` filter dropped that
   game_id from the output entirely, silently).

   **Fixed (2026-09-26):** the `by_pair` construction loop in
   `player_inputs_from_predictions` now raises `ValueError` naming both
   colliding `game_id`s instead of overwriting silently. This should never
   fire against a real, correct NFL schedule (no two games between the same
   teams in the same week) -- it's a defensive fail-loud guard against
   upstream data-quality bugs, consistent with `game_inputs_from_predictions`'s
   own existing duplicate check a few lines above it. Tested
   (`tests/test_simulation_adapter.py`), verified failing before the fix.

### Calibrated simulation v2 (2026-09-27)

Work items 2 and 3 above are now built, beyond the originally scoped
"shared Gaussian residual": prediction-analog marginals
(`PredictionAnalogCalibration`), a game/team/script factor copula keyed by
canonical role (`FactorCopulaModel`, with a two-parameter team-factor
baseline), and a one-command rolling-origin backtest
(`scripts/evaluate_calibrated_simulation.py`, verified by
`scripts/verify_calibrated_simulation.py`). Design, synthetic evidence,
the command and caveats: GAPS.md, "Calibrated simulation v2".

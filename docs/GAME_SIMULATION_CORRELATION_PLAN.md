# Game Simulation and Player Correlation Plan

Status: Phase 1 (interfaces, deterministic core) built and bug-fixed; Phase 2
(OOF residual panel + real correlation fitting) scoped 2026-09-23, not
started. See "Phase 2 scoping" below for what that actually requires.
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

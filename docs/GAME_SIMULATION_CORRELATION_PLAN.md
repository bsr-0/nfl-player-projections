# Game Simulation and Player Correlation Plan

Status: Phase 1 implementation (interfaces and deterministic core)
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
1. Land database-independent contracts, simulation primitives, and tests (this branch).
2. Add DB/model-artifact adapters after database access is available.
3. Build OOF residual panels with season cutoffs and audit coverage.
4. Add production simulation command and JSON/Parquet output.
5. Wire weekly generation and UI behind an explicit flag.
6. Run ablations against independent and shared-residual baselines.

## Deferred until DB access
Queries, schema mapping, historical OOF fitting, injury/depth-chart resolution, learned allocation, copula comparison, artifact regeneration, and UI output.
# Game Simulation Output Schema

Schema version: game-sim-v1

The simulation output is a JSON object or an equivalent set of Parquet tables:

- game_draws: one row per simulated game and draw;
- player_draws: one row per simulated player and draw;
- player_summary: one row per player/game with UI-ready distribution statistics.

## Top-level metadata

| Field | Type | Meaning |
|---|---|---|
| schema_version | string | Must equal game-sim-v1 |
| seed | integer | Random seed used for reproducibility |
| game_count | integer | Number of simulated games |
| game_ids | string[] | Stable game identifiers |
| model_config | object | Selected game-model families and simulation settings |

## game_draws

| Field | Type | Meaning |
|---|---|---|
| game_id | string | Stable game identifier |
| draw | integer | Zero-based Monte Carlo draw number |
| home_score | float | Simulated home score |
| away_score | float | Simulated away score |
| home_plays / away_plays | float | Simulated offensive plays |
| home_pass_attempts / away_pass_attempts | float | Simulated pass attempts |

Invariants: scores and volumes are nonnegative; pass attempts cannot exceed
plays; a player draw cannot reference a missing game draw.

## player_draws

| Field | Type | Meaning |
|---|---|---|
| game_id | string | Parent game identifier |
| draw | integer | Parent game draw |
| player_id | string | NFL player identifier |
| team | string | Team in the simulated game |
| position | string | QB, RB, WR, or TE |
| active | boolean | Whether the player participated in this draw |
| fantasy_points | float | Simulated PPR points |

The draw-level table is the canonical correlated output. The player point
estimate is not replaced; it is the center supplied by the existing model and
then conditioned on the shared game script.

## player_summary

| Field | Type | Meaning |
|---|---|---|
| game_id / player_id / team / position | string | Player identity |
| draw_count | integer | Number of draws summarized |
| mean / median | float | Distribution center |
| p10 / p25 / p75 / p90 | float | Simulation quantiles |
| std | float | Sample standard deviation |
| prob_zero | float | Probability of zero fantasy points |
| prob_active | float | Participation frequency |
| max | float | Maximum simulated point result |

Summary rows are intended for the weekly site, lineup optimizer, and DFS
decision tools. They must not be used as a replacement for validation on
future held-out seasons.

## Current implementation boundary

The schema and validators are database-independent. The pipeline still needs
a later adapter that writes these payloads after the existing game and player
serving paths run. Participation probabilities, learned teammate residuals,
and historical calibration remain explicit follow-up inputs.

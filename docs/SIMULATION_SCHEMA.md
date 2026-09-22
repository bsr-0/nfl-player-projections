# Game Simulation Output Schema and Production Gates

## Artifact split

The system writes two intentionally different artifacts.

| Artifact | Schema | Location | Purpose |
|---|---|---|---|
| Site summary | game-sim-site-v1 | docs/data/simulation_{season}_wk{week}.json | Compact UI/decision summary |
| Research payload | game-sim-v1 | Local only | Full raw Monte Carlo draws |
| Research tables | game-sim-v1 | data/simulations/{season}/*.parquet | Efficient analysis and calibration |

Raw draw rows are never written to docs/data. A 1,000-draw slate can contain
hundreds of thousands of player rows and is not appropriate for GitHub Pages.

## Raw schema: game-sim-v1

The full payload contains metadata, game_draws, player_draws, and
player_summary. It is validated before publication: game/player references,
draw IDs, summary values, and schema identity must agree exactly.

### game_draws

Required fields: game_id, draw, home_score, away_score, home_plays,
away_plays, home_pass_attempts, away_pass_attempts.

Current simulator also emits:

- home_won
- simulated_margin
- simulated_total

Each draw conserves score: home_score + away_score = simulated_total.
The winner is sampled from home_win_prob; signed margin magnitude is calibrated
to preserve the supplied expected margin before score clipping.

### player_draws

Required fields: game_id, draw, player_id, team, position, active,
fantasy_points.

### player_summary

One row per player/game:

- mean, median, p10, p25, p75, p90, standard deviation
- probability of zero points and participation probability
- maximum simulated point total

## Site schema: game-sim-site-v1

The site payload contains metadata, game_summary, and player_summary only.

game_summary includes simulated home-win probability plus total and home-margin
mean/p10/p50/p90. player_summary is copied from the validated raw payload.

## Current quality status

The generated command is explicitly labeled:

- simulation_status: exploratory_volume_only
- correlation_mode: independent_residuals
- usage_mode: team_volume_only

It is not a production DFS or lineup-optimization simulation yet.

## Required production gates

1. Availability: connect the participation model so draw-level active status
   is a calibrated probability, not the current 1.0 fallback.
2. Usage allocation: provide causal target, carry, and red-zone share priors
   for each player, then conserve team pass attempts/targets/rushes per draw.
   Point-projection means alone cannot identify a player usage distribution.
3. Correlation fitting: train role-based out-of-fold residual correlation
   artifacts at a stable game-role granularity. Do not fit covariance on a
   fixed list of player IDs and apply it to another slate.
4. Calibration: use held-out seasons to calibrate total/margin dispersion,
   team pace/pass-rate response, player quantiles, and teammate correlation.
5. Evaluation: require improvements in joint forecast score/correlation
   calibration without unacceptable per-player MAE degradation before exposing
   the output as a decision tool.

Until all five gates pass, the site artifact is informational engineering
output only, not a recommendation.

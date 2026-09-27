# Simulation Output Schema (game-sim-v2)

## What produces it

`scripts/generate_simulation_data.py` serves the calibrated copula
(`src/models/calibrated_simulation.py`): each player's draws are the served
prediction plus a calibrated marginal deviation, reordered into game-level
dependence by the role-factor copula. It is exactly the backtest's
`calibrated_role_factor` arm. Artifacts come from
`scripts/fit_simulation_artifacts.py` (verified OOF panel → `data/models/simulation/<run_id>/`,
pointed to by `latest.json`); without them the command fails instead of
falling back. Weeks at or before the artifacts' fitted-through week are
refused, because their own outcomes shaped the calibration.

v2 replaces v1, whose score/plays/pass-attempt draws came from the deleted
game-script simulator.

## Artifacts

| Artifact | Schema | Location | Purpose |
|---|---|---|---|
| Site summary | game-sim-site-v2 | docs/data/simulation_{season}_wk{week}.json | Compact summaries |
| Research tables | game-sim-v2 | data/simulations/{season}/*_{games,players,player_draws}.parquet | Local analysis; scoring with `scripts/evaluate_simulation.py` |

Raw draws never go to docs/data.

## Site payload

- `schema_version`, `source_schema_version`, `seed`, `draws_per_player`,
  `game_count`, `game_ids`.
- `model_config`: `simulation_status="calibrated_copula"`, the selected
  marginal candidate and why, `dependence="role_factor"` and its fit status,
  `dependence_evidence` (the attached backtest's primaries and gate, or a note
  that none was attached), artifact run id / fitted-through week / source
  panel SHA-256, the served game fields used, `scope_caveat`, and the player
  row accounting (offered, excluded by reason, simulated, outcome columns dropped).
- `player_summary` (one row per player-game): `served_prediction`, `mean`,
  `median`, `p10`, `p25`, `p75`, `p90`, `std`, `max`, `draw_count`.
- `game_summary` (one row per game): `served_home_win_prob`, `served_margin`,
  `served_total` (the game models' predictions passed through, **not
  simulated**; null when absent) and `fantasy_sums`: home/away team fantasy
  totals, QB+2 stacks (with `player_ids`) and the game fantasy total, each
  with mean/p10/p50/p90.

Summaries are recomputed from the draws on every validation, so they cannot
disagree with them.

## Scope and status

- Draws are conditional on the player appearing; availability is not modelled.
- On real OOF rows (GAPS.md, 2026-09-27) the calibrated marginal was a
  significant CRPS improvement over the legacy pools; the copula's joint gain
  was not significant although its correlations matched held-out ones. The
  site output is informational, not a lineup or DFS recommendation.

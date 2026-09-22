# Independent Statistical Audit: Game Simulation and Correlation

Status: exploratory framework only. This document defines what must exist before
the simulator can be used for ranking, DFS, or lineup decisions.

## Verdict

The original independent-player system cannot be converted into a valid joint
simulator merely by adding Gaussian residuals. A proper solution is a
hierarchical, causal generative system:

1. Game environment: kickoff-known weather, team strength, market/game model,
   then joint home/away score, possessions, pace, and game state.
2. Team opportunity: pass attempts, rushes, targets, red-zone plays, and
   scoring opportunities conditional on the game environment.
3. Player availability: probability of active roster, role tier, and
   participation conditional on news/injury/depth-chart information available
   before kickoff.
4. Player usage: target/carry/red-zone shares drawn jointly within team pools,
   constrained to conserve opportunities.
5. Player conversion: receptions, yards, touchdowns, turnovers, and bonuses
   drawn conditionally on opportunity and matchup; fantasy scoring is computed
   from the simulated stat line, not independently added PPR noise.
6. Dependence: residual factors are conditional on the above layers and
   estimated by stable game roles plus team/opponent/game context. Player-ID
   covariance is not portable across lineups.

The current code has exploratory implementations of layers 1, parts of 2/3/4,
and a role-correlation interface. It does not yet have the data-trained
versions of any of those components and has no stat-line conversion layer.

## Critical missing model components

| Component | Why it matters | Required data/artifact |
|---|---|---|
| Possession/drive process | Final-score difference is not a game script; timing and drives determine trailing pass volume. | PBP-derived possessions, clock state, EPA/success, pace |
| Joint scoring model | Margin, total, and win probability must be calibrated jointly, not reconciled by a heuristic. | Chronological OOF score distribution/model |
| Usage labels | PPR means do not identify target/carry/red-zone distribution. | Causal targets, carries, routes, snaps, red-zone opportunities |
| Receiving and rushing split | RB points can arise from both rush and target pools. | Position/player stat-line components |
| Touchdown allocation | TDs create most ceiling/correlation and must be scarce/conserved. | Team TD opportunities and player red-zone/goal-line rates |
| Availability/role model | A player active but not a starter is different from inactive. | Participation phase outputs, depth-chart tier, injury reports |
| Correlation conditioning | Shared residuals must be conditioned on context already explained by opportunity. | OOF residuals keyed by role, team context, matchup, game state |
| Roster completeness | Simulating only projected players produces false shares and incomplete opponent effects. | Full active/eligible roster including low-projection teammates |
| Team changes/new coaches | Historical shares can be structurally stale. | Causal team/coaching/roster transition priors with shrinkage |

## Required evaluation design

Every evaluation is rolling-origin. For game in season S/week W, fit all
artifacts only on information available before that kickoff. Same-season
features need as-of-week boundaries; season-level artifacts use seasons < S.

### Marginal gates

- player mean MAE/bias versus current independent projection;
- empirical CRPS for each player distribution;
- PIT/rank histogram and 50/80/90% interval coverage by QB/RB/WR/TE;
- coverage by starter tier, injury status, rookie/veteran, and game total.

### Joint gates

- game win Brier/log score, score-total CRPS, and margin CRPS;
- energy score for player vectors;
- variogram score (p=0.5) for dependence;
- empirical teammate/opponent correlation by role pair:
  QB-WR1, QB-TE1, RB-WR1, WR1-WR2, QB-opposing WR1, etc.;
- conservation checks every draw: score, plays, pass/rush, targets, carries,
  red-zone opportunities, and player share totals.

### Decision gates

- compare independent, team-factor-only, role-correlation, and full
  opportunity/stat-line simulators on the same OOF games;
- paired block bootstrap by game/week for score differences;
- keep a final untouched season for confirmation after design selection;
- separately evaluate cash, tournament ceiling, stack, bring-back, and
  ownership-adjusted objectives. Do not tune on the decision metric and then
  report it on the same seasons.

A model is promotable only if it improves multivariate calibration and proper
joint scores without materially degrading player marginals. A better simulated
correlation that worsens predictive distributions is not an improvement.

## Implemented in this branch

- total/margin/win-probability draw contract with conserved scores;
- optional participation and share inputs;
- team opportunity share allocator with invariant checks;
- role-based residual-correlation artifact interface;
- strict output schemas and compact site summaries;
- empirical CRPS, energy score, variogram score, coverage, game calibration,
  and strict actual-outcome coverage evaluator;
- production readiness guard and an OOF evaluation CLI.

## Not implemented because it requires historical DB access

- all causal labels/features and OOF artifact construction;
- possession/drive and stat-line models;
- artifact serialization/loading into production simulation;
- calibration fitting and promotion decision;
- DFS/lineup evaluation and UI decision exposure.

## Statistical references

- Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction,
  and Estimation.
- Scheuerer and Hamill (2015), Variogram-based proper scoring rules for
  probabilistic forecasts of multivariate quantities.
- Allen, Ziegel, and Ginsbourger (2023), Assessing the calibration of
  multivariate probabilistic forecasts.

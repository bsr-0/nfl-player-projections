# Simulation marginals and correlation layer on 2026 weeks 1-4 (2026-10-10)

The 2026-10-07 backtest (`calibrated_simulation_backtest_20261007_raw_actuals`) is
98% 2024-2025: its 12,456 scored player-games include only 292 from 2026 (weeks 2-3),
scored on the OOF panel's pre-repair predictions. This scores the calibrated
simulation on every 2026 player-game through week 4 on the repaired database.

Method (`replay_unblended.py`, `score_2026.py`, `score_corr.py`): replay
`predict(as_of)` for 2026 weeks 1-4 in a scratch copy of the repaired DB, keep the
unblended model prediction (`predicted_points_model`, what the OOF panel and the
calibration are made of) and the served blend (`predicted_points`, what the site's
simulation is centred on), keep players who played the target game (the panel's
rule), draw 1,000 residuals per player from the calibration fitted before 2025
week 22 (`outer_folds/2025/calibration_full_week22.json`, causal for 2026) with the
repo's game seeds, and score against raw PPR points. `is_cold_start` = no stats row
since 2023 before the target week (4.6% of rows). Dependence arm = the 2025-fold
role-factor copula. 1,431 player-games, 64 games.

## Marginals

| | n | MAE | CRPS | cov50 | cov80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backtest 2024-25, weeks 2-22 | 12,164 | 4.427 | 3.074 | 0.489 | 0.787 |
| Backtest 2024-25, weeks 2-5 | 2,703 | 4.414 | 3.128 | 0.480 | 0.790 |
| Earlier 2026 rows (weeks 2-3, pre-repair inputs) | 292 | 5.030 | 3.541 | 0.459 | 0.723 |
| **2026 weeks 1-4, unblended-centred** | 1,431 | 4.548 | 3.200 | 0.486 | 0.774 |
| **2026 weeks 1-4, blend-centred (site)** | 1,431 | 4.445 | 3.290 | 0.504 | 0.767 |

By week (unblended): CRPS 3.38 / 3.25 / 3.07 / 3.09, cov80 0.75 / 0.77 / 0.80 / 0.80.
By position: QB CRPS 4.5, RB 3.1, WR 3.2, TE 2.8; cov80 QB 0.83, RB 0.78, WR 0.75, TE 0.81.

The per-player distributions hold: CRPS is 0.07 above the 2024-25 early-season
level (about one standard error at this n) and coverage is within a point or two of
target. The earlier 292-row 2026 sample looked worse because of its inputs.

## Sums, independent vs role-factor dependence (CRPS; coverage50 / 80)

| Centred on | Sum | Independent | Role factor | Difference (se) |
| --- | --- | --- | --- | --- |
| Unblended | Stack (QB+2) | 9.90; .34 / .67 | 9.77; .41 / .71 | -0.134 (0.047) |
| | Team total | 13.38; .43 / .70 | 13.30; .50 / .81 | -0.080 (0.077) |
| | Game total | 20.44; .45 / .63 | 20.10; .55 / .75 | -0.335 (0.222) |
| Blend (site) | Stack | 10.02; .40 / .65 | 9.95; .48 / .70 | -0.070 (0.049) |
| | Team total | 17.72; .30 / .56 | 17.41; .34 / .64 | -0.312 (0.079) |
| | Game total | 31.33; .20 / .52 | 30.35; .33 / .63 | -0.989 (0.235) |

The correlation layer improves every sum in the same direction as the backtest
(stack CRPS 9.46 -> 9.39 there), most clearly the stack (-0.134, about 2.8 se,
unclustered and on four weeks). Backtest benchmark for the independent arm: stack
9.48, team 13.86, game 21.84.

## The site's simulation is centred on the blend, and the blend is not a total

`generate_simulation_data.py` centres draws on `predicted_points`, the pace-blended
value; the calibration is fitted on the unblended model's residuals for players
who played. The Step 8 pace is a season-total projection / 17, so it already
discounts missed games, and early in the season the blend leans on it (weight
1 - g/(g+3)). On players who played in weeks 1-4 the blend sums to 131.5 points
per game against 168.5 actual (-37, -22%; QB 11.3 vs 14.8 per player), the
unblended model to 161.2 (-7). Player-level coverage is unaffected, but team and
game totals centred on the blend are badly under-covered: 80% intervals hold 56%
of team totals and 52% of game totals (unblended: 70% and 63%; target 80%).
The selection rule is not affected: it scores the blend as served.

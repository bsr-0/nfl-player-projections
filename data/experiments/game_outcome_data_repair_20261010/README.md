# Game outcome models on the repaired 2026 data (2026-10-10)

The live game path (`scripts/predict_upcoming_games.py`) treats unscored schedule
rows as upcoming and builds team form and Elo from scored games. Until 2026-10-09
the local schedule had scores for 2026 week 1 only, and the PBP-derived
`team_stats` columns it reads (`drive_success_rate`, `avg_drive_epa`,
`points_per_drive`, `neutral_pass_rate_oe`) were NULL for weeks 2-4 (GAPS.md,
"Week 6 rehearsal"). So any game prediction served from week 2 on saw one week of
2026 and flagged every game as a cold start.

`compare_states.py` predicts week 5 on copies of three backups: before both fixes
(`nfl_data_pre_schedule_scores_20261009_234228.db`), scores only
(`..._pre_team_pbp_20261009_234609.db`) and both fixed
(`..._pre_job_run_20261010_000934.db`). `replay.py` replays weeks 2-5 as of each
week: fixed data with scores blanked from the target week on, against the stale
state.

## Week 5 (15 games), before fixes -> both fixed

| Model | Mean abs change | Max | Picks flipped |
| --- | ---: | ---: | --- |
| Home win prob, logistic | 0.038 | 0.108 | 0 winners |
| Home win prob, XGB | 0.064 | 0.152 | 4 winners |
| Home win prob, RF | 0.051 | 0.139 | |
| Margin, ridge | 1.31 pts | 2.98 | 4 ATS picks |
| Total, ridge | 0.87 pts | 2.33 | 2 O/U picks |

Scores and team stats contribute about equally. Largest feature moves: Elo
difference (53 points on average), points scored/allowed form, drive EPA and
points per drive, and `is_cold_start` (1 -> 0 for every game).

## Played games, weeks 3-5 (33 games; week 2 inputs are identical in both states)

| | Fixed | Stale | Vegas line |
| --- | ---: | ---: | ---: |
| Brier, logistic / XGB / RF | 0.244 / 0.231 / 0.230 | 0.235 / 0.211 / 0.218 | |
| Winner accuracy, logistic / XGB / RF | 0.55 / 0.67 / 0.64 | 0.48 / 0.58 / 0.55 | 0.55 |
| Margin MAE (ridge) | 7.97 | 7.80 | 7.82 |
| Total MAE (ridge) | 10.48 | 11.40 | 10.35 |
| ATS / O-U record (ridge) | 14-15 / 16-17 | 13-18 / 22-11 | |

Paired Brier difference (fixed - stale): logistic +0.009 [-0.029, +0.048], XGB
+0.020 [-0.012, +0.054]; margin MAE +0.17 [-0.89, +1.30] (bootstrap over games).
Nothing is distinguishable on 33 games. The fixed state is the correct one by
construction (the models were trained on full form and Elo); the replay cannot
say it predicts better yet. Neither state beats the Vegas line on this sample.

`week5_predictions_by_state.csv` and `replay_2026_weeks2_5.csv` are local only (gitignored).

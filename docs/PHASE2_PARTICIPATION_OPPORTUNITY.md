# Phase 2: Participation and Opportunity

## Boundary

Phase 2 predicts pre-kickoff usage independently of fantasy production:

1. `P(meaningful participation)` from objective snap thresholds.
2. `E(offensive snap share | meaningful participation)`.

It does **not** modify Phase 7, create fantasy-point zeros, or manually label
players RB1/WR1. Integration with the PPR model is a later, separately tested
phase using only out-of-fold Phase 2 predictions.

## Population and targets

Only Phase 1 rows with authoritative snap measurements are training labels:

- `confirmed_played` and `confirmed_zero_snaps`: labeled.
- `unknown`: retained for inference and coverage reporting, never made negative.

The pre-registered targets are:

| Target | Definition |
|---|---|
| Any offensive snap | `offense_snaps > 0` |
| Primary meaningful use | `offense_pct >= 10%` |
| Substantial use | `offense_pct >= 25%` |
| Conditional opportunity | normalized `offense_pct`, only for primary-positive rows |

The 10% threshold is primary. The others are sensitivity analyses and must not
be selected after seeing which makes a preferred model look best.

## Causal feature contract

Current-game outcomes never enter model features. Usage and role features are
shifted by at least one player-game:

- previous-game and rolling 3/5-game snap share;
- prior participation and zero-snap counts;
- previous and rolling participation;
- previous team/position snap rank;
- previous roster status;
- cold-start and history-availability indicators;
- schedule week and home/away.

Current status, same-week depth chart, team/opponent identity, PPR, snaps, and
snap share are excluded. The default model also neutralizes injury score. The
optional `--include-pregame-injury` ablation reuses the repository's existing
kickoff-filtered `player_injuries` cache; it drops reports modified after
kickoff. Historical depth-chart features remain excluded: a retrospective
depth chart is leakage, not a role feature.

The previous team/position rank is an objective usage measurement, not a
manual RB1/WR1 label. Human-readable labels may eventually be derived for the
UI from predicted opportunity; they are not training truth.

## Models and baselines

Participation baselines:

- position rate learned from prior seasons;
- previous observed participation;
- rolling-three participation.

Candidates:

- regularized logistic regression;
- histogram gradient boosting.

Conditional opportunity compares rolling-three snap share with a histogram
gradient-boosting absolute-error regressor. Predictions are clipped to `[0,1]`.

## Validation and artifacts

Validation is expanding-season walk-forward. For held-out season `S`, all
training seasons are `< S`. The runner reports Brier score and log loss for
participation, MAE/RMSE for opportunity, calibration bins, label coverage by
season/position, and causal feature ablations.

```bash
python scripts/build_canonical_player_weeks.py --write --seasons 2013 2026
python scripts/run_phase2_participation.py
python scripts/run_phase2_participation.py --predict-season 2026
python scripts/run_phase2_participation.py --predict-season 2026 --predict-week 4
python scripts/run_phase2_participation.py --include-pregame-injury
```

Outputs (gitignored experiment data):

- `data/experiments/phase2/oof_predictions.csv`
- `data/experiments/phase2/evaluation.json`
- `data/experiments/phase2/predictions_2026_week_1.csv` (when requested)

OOF predictions are the only acceptable Phase 2 inputs to a later Phase 3
training experiment. In-sample fitted probabilities must never be supplied to
the historical PPR model.

The optional Week 1 output is a true pre-season prediction: it trains only on
earlier seasons and returns only Week 1 rows. For any later week, pass
`--predict-week W`; that path trains on all earlier seasons plus observed rows
from weeks `< W` in the target season, and predicts Week `W` only. It never
uses an outcome from the target week or a later week.

## Acceptance criteria

Phase 2 can be considered empirically successful only when, by position and in
aggregate:

1. A candidate materially beats simple baselines out of season on Brier score
   and log loss, without worse calibration.
2. Conditional opportunity beats rolling snap share on MAE.
3. Results are not driven only by veterans; cold-start rows are reported.
4. Label coverage has no unexplained season/position discontinuity.
5. All causal-contract and walk-forward tests pass.

If gradient boosting does not reliably beat logistic regression or a rolling
baseline, retain the simpler estimator. No downstream PPR benefit is claimed
until Phase 3 measures it on identical held-out player-weeks.

## Known incomplete inputs

- Pregame injuries: available only through the opt-in, kickoff-filtered cache
  ablation; its incremental out-of-season value must be measured.
- Historical depth charts: excluded until snapshots are demonstrably pregame.
- Routes: the repo has pass-play participation for some seasons, but it is not
  yet part of this first objective snap target and has uneven coverage.
- Rookies: cold starts are explicitly marked; college/draft/preseason priors
  require a separate causality audit before inclusion.
- 2026: predictions can be produced once Phase 1 has roster evidence; outcomes
  remain unlabeled until authoritative snaps arrive.

## Provisional real-data smoke result

The complete harness was also run against the checked-in 2013–2025 raw snap
archive (87,479 QB/RB/WR/TE snap rows). This is a mechanics check, not the final
Phase 2 experiment: it lacks Phase 1's roster-only/unknown population and uses
PFR IDs directly. Across expanding-season folds, the primary 10% target gave:

| Model | Brier | Log loss |
|---|---:|---:|
| Histogram GBM | 0.0838 | 0.2714 |
| Logistic regression | 0.0869 | 0.2830 |
| Rolling-3 baseline | 0.1256 | 1.4282 |
| Previous-game baseline | 0.1387 | 1.8447 |
| Position-rate baseline | 0.1438 | 0.4594 |

Conditional snap-share MAE was 0.1167 for histogram GBM versus 0.1316 for the
rolling-3 baseline. These results show that the code path and causal history
features have signal; they do **not** satisfy the final acceptance gate until
the same run uses the audited `canonical_player_weeks` table and its coverage
report is inspected.

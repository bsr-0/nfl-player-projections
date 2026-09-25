# NFL Player Projections

Machine learning system for projecting NFL fantasy football performance —
weekly and season-long — built on play-by-play and weekly player data from
`nfl-data-py`/nflverse, 2006–present.

## What it does

- **Position-specific models** (QB/RB/WR/TE) predict individual stat lines
  (passing yards, rushing TDs, receptions, etc.), which are assembled into
  fantasy points using standard PPR/half-PPR/standard scoring.
- **Season-total projections** for draft prep, trained directly on
  prior-season aggregate stats (not summed from noisy weekly predictions).
- **Draft tooling**: a spread/availability/VONA advisor that layers model
  insight on top of ADP rather than replacing it.
- Leakage-safe feature engineering (walk-forward validation, causal
  rolling windows, no same-week outcome data as input) — see
  `src/utils/leakage.py`.

Plan A team-share experiments found a small, consistent out-of-fold lift from
fold-local blending with the rolling-3 baseline followed by team/week share
renormalization. Residual learning alone was worse than rolling-3. See
`GAPS.md` §11.4 and the retained metrics under
`data/experiments/plan_a_improvements/`. These are validation artifacts only;
they have not replaced served models. The current priority is to qualify that
candidate against Plan A's per-target, held-out accuracy, segment, constraint,
and served-path comparison gates. A two-stage, position-specific rushing-
attempt arm now supplies replicated held-out improvement for the last target
that the simple blend missed; it is still validation-only. The incomplete QB
DVOA ablation is a separate weekly-model experiment and is not evidence about
Plan A.

The yardage-only reconstruction has a separate reconstruction-aware benchmark
(`scripts/evaluate_plan_a_reconstruction_candidates.py`). It tests
volume-weighted share fitting, blend selection on reconstructed-volume error,
and lagged-team-total forecasts on identical rows. On explicitly bounded
2024/2025 holdouts, the best research arm reduced partial-yardage MAE from
1.071633 (rolling-3) to 1.062207 (paired 95% CI [-0.015270, -0.003516]). This
is validation evidence only; it is not a full-PPR result or a production
promotion until a fresh season and the serialized serving path agree.

Team-total model selection is calibrated within each walk-forward fold rather
than chosen after seeing the holdout. The resulting frozen candidate,
`xgb_reconstruction_blend_teamtotal_blend`, beat rolling-3 across completed
2023–2025 holdouts (MAE 1.070962 vs 1.083105; paired 95% CI
[-0.016845, -0.007383]). It is still a validation artifact until the same
candidate is exported and checked through the serialized serving path.

The frozen reconstruction candidate has now been reloaded from serialized
artifacts and compared on identical valid rows: it beats rolling-3 in 2023,
2024, and 2025, with season-level paired 95% CIs excluding zero. The artifacts
remain under `data/experiments/plan_a_reconstruction_served_artifacts/`; this
does not promote or replace the production weekly models.
Run `scripts/check_plan_a_reconstruction_promotion.py` for the fail-closed
readiness status. It reports `research_ready_not_promotable` until Plan A
also models the missing full-PPR components (receptions, touchdowns, and
passing production).

The underlying share table now contains those full-PPR component histories
(`receptions`, receiving/rushing TDs, passing yards/TDs, and interceptions),
but their models and scoring reconstruction are intentionally still separate
from the validated four-target Plan A artifacts.

Leakage-safe team-total forecasts for these components are evaluated with
`scripts/evaluate_team_total_forecasts.py`; player allocation and full-PPR
reconstruction remain the next layers.

### Current Plan A and Vegas-retrain status (2026-09-24)

The served weekly artifacts were retrained after the Vegas sign correction.
The authoritative 2025 production-harness holdout is
`data/backtest_results/backtest_2025_20260924.json` (overall MAE 4.33).
This must not be compared to `single_week_ppr` research metrics, which use a
different architecture and evaluation population.

The corrected eight-component Plan A validation run scores against raw player
stats. Its fold-local selector reduced PPR MAE from 2.49130 to 2.36984
across 40,559 identical player-weeks (paired 95% CI for the delta:
-0.13532 to -0.10674). Results and row-level scores are under
`data/experiments/full_ppr_raw_truth_20260924/`. The earlier 2.36521 result
used altered actual labels and is superseded. This remains validation-only:
the UI and production weekly serving path do not load it. The shadow serving
path now preserves sparse-event zero fallbacks and supports fold-zero
rolling-3 team totals; the 2023 fold-zero replay matches all 13,404 OOF
predictions with round-trip CSV parsing. Fumbles lost and two-point
conversions are outside this eight-component score.

## Setup

```bash
pip install -r requirements.txt
python -m src.data.nfl_data_loader   # populate data/nfl_data.db from nflverse
```

## Usage

```bash
python -m src.models.train                     # full training run
python -m src.models.train --fast --no-tune     # quick iteration, skips Optuna
python scripts/generate_app_data.py             # weekly/season predictions
python scripts/generate_draft_data.py           # draft board JSON
python scripts/draft_advisor.py --mode spread --season 2025
python scripts/predict_upcoming_games.py --season 2026 --week 2  # win, ATS, and O/U forecasts
python scripts/generate_simulation_data.py --season 2026           # game/player simulation JSON
python scripts/generate_simulation_data.py --season 2026 --write-parquet  # JSON + analysis Parquet
pytest
```

## Keeping the site current (manual, weekly)

Nothing runs on a schedule: the whole pipeline reads `data/nfl_data.db`,
a gitignored local file, so there is no CI or cron to run it from. During
the season, run this once a week after the games (Tuesday is a good
default -- nflverse has usually published by then):

```bash
python scripts/refresh_site_data.py    # chains every step below, in order
```

It runs `src.data.auto_refresh` (rosters / weekly stats / schedule),
`backfill_injuries.py` and `backfill_adp.py` for the current season,
`generate_draft_data.py`, copies `data/players_*.json` into `docs/data/`,
and `generate_weekly_data.py`. It does **not** commit or push -- review
`git status docs/data data/players_*.json` first. Use `--skip` for partial
runs (e.g. `--skip auto_refresh` when the DB is already fresh).

What goes stale if you skip a week: the injury Risk/Probable badge on
`docs/weekly.html` and `injury_flag` on the draft board (the report feed
is per-week), market ADP, and the weekly page's `season_prorated` ->
`weekly_model` switch, which flips automatically once the season's first
completed game rows are ingested.

The `ui-stable-v1` tag exists but points at 2026-05-18 and predates every
site change since; it is not a usable recovery point for the current
pages. There is currently no documented UI freeze policy.

## Project structure

```
config/settings.py   # single source of truth for constants, feature lists, scoring
src/data/             # ingestion (nflverse, PBP aggregation, entity resolution)
src/features/          # feature engineering, leakage-safe rolling/causal features
src/models/            # training, ensemble, component predictors, preseason projector
src/predict.py          # serving-path prediction entry point
scripts/                # data generation, draft tooling, backfills, one-off scripts
tests/                  # pytest suite
data/                   # SQLite DB, trained models, generated JSON/parquet (gitignored)
```

## Project notes

- `PROJECT_NOTES.md` — architecture decisions, known bugs/patterns, past
  audit findings, methodology notes.
- `GAPS.md` — gap-analysis audit and running log of fixes/features shipped
  each session. Read this first when picking up work on the project; it
  has a standing instruction for documenting bugs found along the way.
- `CLAUDE.md` — agent working directives for this repo.
- **Vegas lineage and validation status:** Vegas player features use the
  corrected team-side formulas in `src/data/external_data.py` (home implied
  total `(total + home_spread_line) / 2`, away `(total - home_spread_line) / 2`,
  with the player-team spread convention preserved). The served weekly
  artifacts are not considered corrected until a post-fix retrain completes.
  `scripts/audit_production_artifacts.py` is read-only and reports artifact
  timestamps, feature schema/version, and pre/post-fix lineage. Walk-forward
  CSVs are validation artifacts, not served models; compare them with
  `scripts/compare_vegas_validation.py` before promotion.

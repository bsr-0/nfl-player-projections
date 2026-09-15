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

# NFL Player Performance Predictor

[![Rubric Compliance](https://github.com/benrosen/nfl-player-projections/actions/workflows/rubric-compliance.yml/badge.svg)](https://github.com/benrosen/nfl-player-projections/actions/workflows/rubric-compliance.yml)

A machine learning workflow that predicts NFL player fantasy performance for 1-18 weeks ahead, using historical player data, team statistics, and utilization scores.

## Features

- **6-position support**: ML models for QB, RB, WR, TE plus rolling-average projections for K and DST
- **Flexible prediction window**: Predict performance for next week or entire season (1-18 weeks)
- **Utilization Score integration**: Incorporates opportunity-based metrics for offensive position predictions
- **Automated data pipelines**: Loaders (via nfl-data-py) to refresh historical data to latest results (back to 2006)
- **Team context**: Includes team stats for every team a player has been on
- **Model optimization**: Hyperparameter tuning with Optuna, dimensionality reduction

## Installation

```bash
cd nfl-player-projections
pip install -r requirements.txt
```

## Usage

### 1. Load Data
```bash
# Load specific season range
python -m src.scrapers.run_scrapers --seasons 2020-2025

# Refresh current season only
python -m src.scrapers.run_scrapers --refresh
```

### 2. Train Models
```bash
python -m src.models.train --positions QB RB WR TE
```

### Retraining

**When to retrain:** Any time `FEATURE_VERSION` in `config/settings.py` is bumped (e.g. after adding or removing features), or after loading a new season of data. The model will log a version-mismatch warning at prediction time if the saved artifact is stale.

**Fast retrain** (recommended for iterating — ~8-10x faster, minimal accuracy loss):
```bash
python -m src.models.train --positions QB RB WR TE --fast --no-tune
```

**Production retrain** (full hyperparameter search via Optuna — slow, may OOM on large machines; use `--no-tune` if it crashes):
```bash
python -m src.models.train --positions QB RB WR TE
```

**Single position** (useful for testing a feature change on one position before full retrain):
```bash
python -m src.models.train --positions WR --fast --no-tune
```

Trained artifacts are saved to `data/models/`. After retraining, regenerate predictions:
```bash
python -m src.predict --weeks 18
```

### Diagnosing outlier projections

If a player is projected far above or below expectations, run:
```bash
# Check all positions (default: 18-week horizon, 1.5σ threshold)
python scripts/diagnose_outliers.py

# Tighter threshold or single position
python scripts/diagnose_outliers.py --sigma 1.2 --position WR

# Week-1 view
python scripts/diagnose_outliers.py --weeks 1
```

The script flags each outlier and identifies the most likely driver from a known list of failure modes:
- `preseason_ecr=300` — ADP name match failed; model treats player as undrafted
- `prev_season_ppg≈0` — player missed most/all of prior season
- `team_changed=1` — share features (target/rush/snap%) reflect the old team's system
- `fp_late6_vs_season` near clip bounds — single-game outlier inflating/deflating late-season trend
- `snap_share≈0` — no recent snap data (new signing or ADP lookup miss)
- `injury_score<0.5` — significant injury flag
- `depth_chart_rank≥3` — listed as backup

### 3. Make Predictions
```bash
# Predict next week
python -m src.predict --weeks 1

# Predict full season
python -m src.predict --weeks 18

# Predict specific player
python -m src.predict --player "Patrick Mahomes" --weeks 4
```

## Web App (FastAPI)

A web app served by FastAPI at **http://localhost:8501**. Supports QB, RB, WR, TE with time horizon filters (1-week, 4-week, rest-of-season).

**Recommended: one command** (starts server with data refresh and predictions):
```bash
python run_app.py --refresh --with-predictions
```

Other useful flags:
- `--skip-data` — skip data loading, start app immediately
- `--with-predictions` — regenerate ML predictions before launch (uses cache if fresh)
- `--force-predictions` — always regenerate predictions (ignore cache)
- `--port <N>` — change port (default: 8501)

**Alternatively**, start the API server directly:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8501
```

See `docs/RUN_WITH_NEW_FEATURES.md` and `api/README.md` for details.

## Project Structure

```
nfl-player-projections/
├── run_app.py              # Main entry point (data + predictions + web server)
├── requirements.txt        # Python dependencies
├── config/
│   └── settings.py         # Central configuration (seasons, paths, feature flags)
├── data/
│   ├── raw/                # Raw source data (team stats CSVs)
│   ├── models/             # Trained model artifacts and metadata
│   └── nfl_data.db         # SQLite database with player/team data
├── src/
│   ├── data/               # Data loading, auto-refresh, PBP aggregation
│   ├── scrapers/           # Data collection wrappers (nfl-data-py loaders)
│   ├── features/           # Feature engineering and utilization score
│   ├── models/             # ML model definitions, training, and ensembles
│   ├── evaluation/         # Backtesting, metrics, monitoring, explainability
│   ├── integrations/       # External platform integrations (e.g. ESPN)
│   ├── utils/              # Shared utilities (database, calendar, helpers)
│   ├── predict.py          # Prediction CLI
│   └── pipeline.py         # End-to-end pipeline orchestration
├── api/                    # FastAPI backend
├── scripts/                # Standalone scripts (compliance, secrets scan, analytics)
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks (model validation)
└── docs/                   # Documentation
```

## Publishing to GitHub / Security

- **Secrets**: Never commit API keys, passwords, or `.env` files. Use environment variables and copy `.env.example` to `.env` for local config (`.env` is gitignored).
- **Optional features** that need env vars: email alerts (`SMTP_*`), PostgreSQL migration (`DATABASE_URL`). See `.env.example`.
- **Already tracked files**: If you previously committed `.env` or secrets, remove them with `git rm --cached .env` and rotate any exposed keys.
- **Pre-push check**: Run `python scripts/scan_secrets.py` to scan staged files, or `python scripts/scan_secrets.py --all` to scan the repo. A GitHub Action runs the same scan on every push and pull request.

## Rubric Compliance Gate

The repository includes a CI gate that verifies key fantasy-system requirements (position-specific architecture, multi-horizon model contracts, feature/evaluation surface checks, and monitoring artifacts).

Run locally:

```bash
# Human-readable report
python scripts/check_rubric_compliance.py

# JSON output (CI-friendly)
python scripts/check_rubric_compliance.py --json

# Strict mode: fail if monitoring artifacts are missing
python scripts/check_rubric_compliance.py --require-artifacts
```

Run the core regression tests for the gate:

```bash
pytest -q tests/test_rubric_compliance_checker.py tests/test_metrics_evaluator.py tests/test_production_retrain.py
```

## Data and Mid-Season Updates

- **Auto-refresh**: Running the pipeline (train or `scripts/generate_app_data.py`) triggers an auto-refresh so the current NFL season's completed weeks are loaded when available.
- **Schedule updates**: Schedule data is refreshed from nfl-data-py on `--refresh`. New seasons are loaded automatically when nfl-data-py publishes them.
- **Train/test**: The latest available season is held out as the test set; training uses all prior seasons.
- **Data loading**: `src/data/nfl_data_loader.py` uses PBP fallback when weekly data has fewer weeks than the current NFL week, so in-season data stays up to date.

## Utilization Score Methodology

The Utilization Score (0-100) is the **primary prediction target** for offensive positions. Models predict future utilization; rankings and app display use predicted utilization. The score measures player opportunity and usage:

### RB Utilization Score
- **60-69**: ~12.2 PPG, 70%+ finish as RB2/RB3
- **70-79**: ~15.1 PPG, strong RB2 upside
- **80+**: Elite usage, RB1 potential

### WR Utilization Score
- Target share, air yards share, red zone targets
- Route participation rate

### TE Utilization Score
- Target share relative to position
- Red zone involvement
- Inline vs slot usage

### QB Utilization Score
- Adjusted for rushing involvement
- Red zone opportunity rate

### Kicker (K) Projections
- Rolling averages of FG made (by distance: 0-39, 40-49, 50+), XP made/missed
- Home/away adjustment, team scoring context
- Scoring: FG 0-39 = 3 pts, FG 40-49 = 4 pts, FG 50+ = 5 pts, XP = 1 pt, miss = -1 pt

### Defense/Special Teams (DST) Projections
- Rolling averages of sacks, interceptions, fumble recoveries, defensive/special teams TDs
- Points allowed bracket scoring (shutout = 10 pts, 35+ allowed = -4 pts)
- Home field advantage adjustment

## Model Architecture

Each position uses an ensemble of:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Ridge Regression (linear baseline)

With dimensionality reduction via:
- Recursive Feature Elimination (RFE)
- PCA for correlated features
- Feature importance-based selection

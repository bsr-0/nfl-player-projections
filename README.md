# NFL Player Performance Predictor

A machine learning workflow that predicts NFL player fantasy performance for 1-18 weeks ahead, using historical player data, team statistics, and utilization scores.

## Features

- **Multi-position support**: Separate optimized models for QB, RB, WR, TE
- **Flexible prediction window**: Predict performance for next week or entire season (1-18 weeks)
- **Utilization Score integration**: Incorporates opportunity-based metrics for better predictions
- **Automated data pipelines**: Scrapers to refresh historical data to latest results
- **Team context**: Includes team stats for every team a player has been on
- **Model optimization**: Hyperparameter tuning with Optuna, dimensionality reduction

## Installation

```bash
cd nfl-predictor
pip install -r requirements.txt
```

## Usage

### 1. Refresh Data
```bash
python -m src.scrapers.run_scrapers --seasons 2020-2024
```

### 2. Train Models
```bash
python -m src.models.train --positions QB RB WR TE
```

### 3. Make Predictions
```bash
# Predict next week
python -m src.predict --weeks 1

# Predict full season
python -m src.predict --weeks 18

# Predict specific player
python -m src.predict --player "Patrick Mahomes" --weeks 4
```

## Web app (FastAPI + React SPA)

A dark-theme single-page app with **position** and **time horizon** filters and one interactive chart of projected fantasy points (matchup-aware; tooltips show opponent and utilization).

**Recommended: one command** (builds frontend if needed, starts server):
   ```bash
   python run_app.py --refresh --with-predictions
   ```
   Then open **http://localhost:8501**. Use `--skip-data` to skip data load, or `--with-predictions` to regenerate ML predictions before launch.
**Alternatively**, run API and frontend separately:
1. Build frontend: `cd frontend && npm install && npm run build && cd ..`
2. Start API (serves built SPA at `/`): `python -m uvicorn api.main:app --host 0.0.0.0 --port 8501`

See `docs/RUN_WITH_NEW_FEATURES.md`, `api/README.md`, and `frontend/README.md` for details.

## Project Structure

```
nfl-predictor/
├── data/
│   ├── raw/              # Raw scraped data
│   ├── processed/        # Cleaned and feature-engineered data
│   └── models/           # Trained model artifacts
├── src/
│   ├── scrapers/         # Data collection modules
│   ├── features/         # Feature engineering and utilization score
│   ├── models/           # ML model definitions and training
│   ├── evaluation/       # Model evaluation and testing
│   └── utils/            # Shared utilities
├── api/                  # FastAPI backend for SPA
├── frontend/             # React + Vite SPA
├── tests/                # Unit and integration tests
└── config/               # Configuration files
```

## Publishing to GitHub / Security

- **Secrets**: Never commit API keys, passwords, or `.env` files. Use environment variables and copy `.env.example` to `.env` for local config (`.env` is gitignored).
- **Optional features** that need env vars: email alerts (`SMTP_*`), PostgreSQL migration (`DATABASE_URL`). See `.env.example`.
- **Already tracked files**: If you previously committed `.env` or secrets, remove them with `git rm --cached .env` and rotate any exposed keys.
- **Pre-push check**: Run `python scripts/scan_secrets.py` to scan staged files, or `python scripts/scan_secrets.py --all` to scan the repo. A GitHub Action runs the same scan on every push and pull request.

## Data and Mid-Season Updates

- **Auto-refresh**: Running the pipeline (train or `scripts/generate_app_data.py`) triggers an auto-refresh so the current NFL season’s completed weeks are loaded when available (e.g. 2025 weeks before today).
- **Schedule updates**: Schedule data is refreshed from nfl-data-py on `--refresh`. New seasons (e.g. next year's schedule when released in spring) are loaded automatically when nfl-data-py publishes them.
- **Train/test**: The latest available season is held out as the test set; training uses all prior seasons.
- **Data loading**: `src/data/nfl_data_loader.py` uses PBP fallback when weekly data has fewer weeks than the current NFL week, so in-season data stays up to date.

## Utilization Score Methodology

The Utilization Score (0-100) is the **primary prediction target**. Models predict future utilization; rankings and app display use predicted utilization. The score measures player opportunity and usage:

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

## Model Architecture

Each position uses an ensemble of:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Ridge Regression (linear baseline)

With dimensionality reduction via:
- Recursive Feature Elimination (RFE)
- PCA for correlated features
- Feature importance-based selection

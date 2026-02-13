# Running the App with the New Features

This guide covers how to run the NFL predictor app using the new features: **matchup-aware predictions**, **team stats from player data**, **injury/rookie features**, **missing-data imputation**, and **schedule loading (nfl-data-py first)**.

---

## Prerequisites

- Python 3.9+ with dependencies installed:
  ```bash
  cd nfl_predictor_claude
  pip install -r requirements.txt
  ```
- For **player and schedule data** the app uses **nfl-data-py** (and optionally scrapers). No extra API keys are required for the main pipeline.

---

## Quick start (one command)

To load data, refresh from nfl-data-py, and launch the **web app** (FastAPI + React) with ML predictions and matchup info:

```bash
python run_app.py --refresh --with-predictions
```

- `--refresh`: Ensures data is (re)loaded (e.g. from nfl-data-py / auto-refresh).
- `--with-predictions`: Runs the ML pipeline and merges projections (including **upcoming opponent** and **home/away**) into the app cache before opening the app.

Then open **http://localhost:8501** (or the port shown). In the app you’ll see:

- **Position** and **time horizon** filters (1 week, 4 weeks, 18 weeks).
- One **interactive chart** of projected fantasy points; tooltips show matchup (e.g. "vs KC", "@ SF") and utilization.

## Full workflow (step-by-step)

Use this when you want to control each stage (data → train → predictions → app) and ensure all new features are used.

### 1. Load historical data (nfl-data-py)

Load player weekly stats and schedules from nfl-data-py. Schedules are used for matchup-aware prediction; team stats can be derived from players (no separate scrapers required).

```bash
# Default: config range (e.g. 2020 through current NFL season)
python -m src.data.nfl_data_loader

# Specific seasons
python -m src.data.nfl_data_loader --seasons 2022 2023 2024 2025
```

### 2. (Optional) Auto-refresh and backfill team stats

Ensures the current season’s completed weeks and schedule are present, and **backfills team_stats** from player data so team tendency features (pass rate, red zone, etc.) are available even without scrapers.

```bash
python -m src.data.auto_refresh
```

You can run this after step 1 or before training/prediction; it also loads schedules for “upcoming” seasons when available.

### 3. Train models

Training uses the full feature pipeline: **imputation** (no NaN/inf), **injury_score / is_injured / is_rookie** (with defaults when missing), **team/matchup** features (from team_stats and schedule). No extra flags are needed to enable these.

```bash
python -m src.models.train
```

**Retraining with new features:** After you pull code that adds or changes features (e.g. injury/rookie, matchup, imputation), the app and predict pipeline will **only use the new feature set if you retrain**. Old `.joblib` models in `data/models/` were built with the previous feature list; they will either ignore new columns or break if columns were removed. To ensure you run with the current features:

1. **Recommended:** Run training again. It overwrites existing models and writes the current feature version:
   ```bash
   python -m src.models.train
   ```
2. **Optional (clean slate):** Remove cached model files, then train:
   ```bash
   rm -f data/models/model_*.joblib data/models/multiweek_*.joblib data/models/feature_version.txt
   python -m src.models.train
   ```

When models are loaded for prediction, the pipeline checks a saved **feature version** file. If it’s missing or doesn’t match the current code, you’ll see a warning and a reminder to run `python -m src.models.train`.

To train only certain positions:

```bash
python -m src.models.train --positions QB RB WR TE
```

### 4. Generate app data (predictions + matchup)

Builds predictions for 1w, 4w, and 18w using the **upcoming game’s** opponent and home/away (from schedule), then writes `data/cached_features.parquet` (and optionally `data/daily_predictions.parquet`) with **upcoming_opponent** and **upcoming_home_away** for the app.

```bash
# Update cached_features only
python scripts/generate_app_data.py

# Also write daily_predictions.parquet
python scripts/generate_app_data.py --parquet
```

### 5. Run the web app

**Option A – Startup script (recommended)**

```bash
# Use existing cache; no data load (builds frontend if needed, then starts FastAPI)
python run_app.py --skip-data

# Or: refresh data and regenerate predictions, then open app
python run_app.py --refresh --with-predictions
```

The script builds the React frontend (if `frontend/dist` is missing) and starts the FastAPI server. Open **http://localhost:8501** (or the port you set with `--port`).

**Option B – API and frontend separately**

```bash
# Build frontend once
cd frontend && npm install && npm run build && cd ..

# Start API (serves built frontend at /)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8501
```

For frontend development with hot reload, run the Vite dev server (e.g. `cd frontend && npm run dev`) and point it at the API (proxy or CORS); the API runs on port 8501.

---

## What the “new features” do at runtime

| Feature | When it runs | What you see |
|--------|----------------|--------------|
| **Matchup-aware prediction** | `predict()` and `generate_app_data.py` | Predictions use the **next game’s** opponent and home/away (from schedule). Schedule is loaded via nfl-data-py first, scraper fallback. |
| **Upcoming matchup in app** | App reads `cached_features` / `daily_predictions` | “Matchup” column in Live and Upcoming week: “vs KC”, “@ SF”, or “TBD” when no schedule. |
| **Team stats from players** | `auto_refresh` or when DB has no team_stats | `ensure_team_stats_from_players` backfills so team tendency features exist without scrapers. |
| **Injury/rookie features** | `FeatureEngineer.create_features()` | `injury_score`, `is_injured`, `is_rookie` added with safe defaults; used in training and prediction. |
| **Missing-data imputation** | End of `create_features()` | No NaN/inf in numeric features; median (or 0) fill so pipelines don’t break on sparse data. |

---

## Matchup features: data dependencies

For **matchup-based utilization (and fantasy-point) predictions**—e.g. Super Bowl or any upcoming game—the pipeline needs:

1. **Schedule for the target week**  
   The DB must have the schedule for the prediction season and week (e.g. 2025 week 22 for Super Bowl). This is what sets `opponent` and `home_away` per team. Without it, those are empty/unknown and schedule-derived features use neutral defaults.

2. **Team stats for the prediction season**  
   The DB must have **team_stats** for the **prediction season** for **both** the player's team and the opponent (e.g. SEA and NE for 2025). Those feed `team_a_*`, `team_b_*`, and the matchup differentials (`matchup_scoring_edge`, `matchup_pass_diff`, `matchup_rush_diff`, etc.). If team_stats are missing for that season, both sides get neutral defaults and the matchup differentials are zero.

**Summary:** Load the current season (and schedule) via `nfl_data_loader` and/or `auto_refresh`; ensure team_stats are backfilled (e.g. via `auto_refresh` or `aggregate_team_stats_from_players`) for the prediction season. Otherwise predictions still run but use neutral matchup values.

---

## Caching and fast runs

Data and models are **saved** so you don't have to re-download or re-train every time:

| What | Where it's saved | When it's used again |
|------|------------------|----------------------|
| **Player/team data** | `data/nfl_data.db` | Used until you run with `--refresh`. Normal runs skip refresh if the DB already has data. |
| **Trained models** | `data/models/*.joblib`, `utilization_weights.json` | Loaded for prediction; train once with `python -m src.models.train`, then all future prediction runs use these files. |
| **Prediction cache** | `data/cached_features.parquet`, `data/daily_predictions.parquet` | Written by `generate_app_data.py` or `run_app.py --with-predictions`. With `--with-predictions`, regeneration is **skipped** if the parquet is newer than 24 hours (configurable via `--max-prediction-cache-hours`). Use `--force-predictions` to always regenerate. |

**Fastest way to open the app after the first full run:**  
`python run_app.py --skip-data` — skips data check and prediction generation; the API serves from existing parquet.

**Refresh predictions once per day:**  
`python run_app.py --with-predictions` — uses cached predictions if they're under 24 hours old; otherwise regenerates and saves them.

---

## Upcoming week and latest data (e.g. 2026 Super Bowl)

For the app to show the **current NFL upcoming week** (e.g. 2026 Super Bowl, Seahawks vs Patriots) instead of the previous year:

1. **Current-season data**  
   Data for the 2025/2026 season is loaded from nfl-data-py. When weekly stats are not yet available, the pipeline uses **play-by-play (PBP)** aggregation. Run a refresh so the DB has the current season:
   ```bash
   python -m src.data.auto_refresh
   ```
   Or use `python run_app.py --refresh --with-predictions` to refresh and then generate app data.

2. **Schedule for Super Bowl (week 22)**  
   Schedules are loaded via **nfl.import_schedules** (nfl-data-py). Playoff weeks are stored with week 19–22 (Wild Card through Super Bowl). The pipeline maps `game_type` (e.g. SB) to week 22 so `get_schedule(season=2025, week=22)` returns the Super Bowl game. Auto-refresh loads the current season’s schedule when missing; the predictor also calls `ensure_schedule_loaded(pred_season)` before generating matchup features.

3. **Cache behind prediction target**  
   If `cached_features.parquet` has a latest season/week **behind** the calendar’s upcoming week (e.g. cache has 2024 week 18 but the upcoming week is 2025 week 22), `generate_app_data` **rebuilds** feature data from the DB instead of reusing the cache. That way the app can show the current season and upcoming week once 2025 data is in the DB.

4. **Validation**  
   When the prediction target is 2025 week 22 (Super Bowl), `generate_app_data` logs whether the schedule for that week includes the expected teams (e.g. SEA, NE). The API can return an **upcoming_week_label** (e.g. "Super Bowl LX") from `data/upcoming_week_meta.json` when present.

---

## Verifying that new features are used

1. **Predictions include opponent/home_away**
   ```bash
   python -m src.predict --weeks 1 --top 5
   ```
   Output should include `opponent` and `home_away` columns (or empty/unknown when schedule is missing).

2. **App shows Matchup**
   After `generate_app_data.py`, open the app and check the **Live** or **Upcoming week** section for a “Matchup” column.

3. **Tests**
   ```bash
   pytest tests/test_missing_data_and_new_features.py tests/test_feature_engineering.py tests/test_matchup_aware_prediction.py tests/test_schedule_and_team_stats.py -v
   ```

---

## Troubleshooting

- **“No player data available”**  
  Run step 1 (nfl_data_loader) for the desired seasons. The message will suggest the exact command and year range.

- **“No trained models found”**  
  Run step 3 (`python -m src.models.train`). Until then, the app can still run but will use fallback projections (e.g. fantasy_points / fp_rolling).

- **Matchup always “TBD”**  
  Schedule for the prediction week may be missing. Run `python -m src.data.auto_refresh` so the current (and next) season’s schedule is loaded from nfl-data-py when available.

- **Port already in use**  
  Use a different port: `python run_app.py --port 8502`. When running uvicorn directly: `uvicorn api.main:app --port 8502`.

- **"Feature set version mismatch" or "No feature version file found"**  
  Models on disk were trained with an older (or unknown) feature set. Retrain so predictions use the current features: `python -m src.models.train`. Optionally clear old artifacts first: `rm -f data/models/model_*.joblib data/models/multiweek_*.joblib data/models/feature_version.txt`.

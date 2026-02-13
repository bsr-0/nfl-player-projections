# Web App Launch Guide

## CLI Command to Launch Web App

```bash
cd /Users/benrosen/nfl_predictor_claude

# Standard launch (uses cached data)
python run_app.py

# Launch with ML predictions (runs trained models before starting)
python run_app.py --with-predictions

# Launch on custom port
python run_app.py --port 8502

# Skip data check (fast start for testing)
python run_app.py --skip-data
```

Or directly with Streamlit:

```bash
streamlit run app.py --server.port 8501
```

---

## Data Flow to Web App

```
┌─────────────────────────────────────────────────────────────────┐
│  BACKEND                                                         │
├─────────────────────────────────────────────────────────────────┤
│  1. python -m src.data.nfl_data_loader --seasons 2020-2024       │
│     → Populates data/nfl_data.db                                 │
│                                                                  │
│  2. python -m src.models.train                                   │
│     → Trains models, saves to data/models/                       │
│     → Saves utilization_weights.json                             │
│                                                                  │
│  3. python scripts/generate_app_data.py                          │
│     → Runs NFLPredictor (EnsemblePredictor)                      │
│     → Merges projection_1w, projection_5w, projection_18w        │
│     → Saves to data/cached_features.parquet                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  WEB APP (app.py)                                                │
├─────────────────────────────────────────────────────────────────┤
│  load_player_data_with_features()                                │
│    → Reads data/cached_features.parquet (or daily_predictions)   │
│    → Falls back to DB + engineer_all_features if missing         │
│                                                                  │
│  App uses: projection_1w, projection_5w, projection_18w,         │
│            utilization_score, fantasy_points, etc.               │
└─────────────────────────────────────────────────────────────────┘
```

**To ensure ML predictions flow to the app:**

1. Train models: `python -m src.models.train`
2. Generate app data: `python scripts/generate_app_data.py`
3. Launch: `python run_app.py`

Or use `python run_app.py --with-predictions` to combine steps 2 and 3.

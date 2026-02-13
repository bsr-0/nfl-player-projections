# Data and Mid-Season Updates

## Current season (e.g. 2025/26) and auto-refresh

- **Auto-refresh**: When you run training (`python -m src.models.train`) or app data generation (`python scripts/generate_app_data.py`), the pipeline calls `auto_refresh_data()` so the latest available data is loaded. This includes the current NFL season’s completed weeks (e.g. 2025 weeks before today) when available from nfl-data-py.
- **Train/test split**: The latest available season is always the **test** set; all prior seasons are used for training. So when 2025 data exists, 2025 completed weeks are the held-out test set and 2020–2024 (or configured range) are used for training.
- **Data loading**: `src/data/nfl_data_loader.py` uses `get_current_nfl_week()` and a PBP (play-by-play) fallback when weekly data has fewer weeks than the current NFL week, so in-season data stays up to date as new weeks are released.

## No manual steps

You do not need to manually refresh 2025 data. Running the pipeline (train or generate_app_data) will:

1. Check for missing seasons and new weeks for the current season.
2. Load any missing data (including PBP aggregation when weekly data is incomplete).
3. Use that data as the latest historical/test set.

## Documentation

See also:

- `config/settings.py`: `CURRENT_NFL_SEASON`, `SEASONS_TO_SCRAPE`, `TRAINING_YEARS`
- `src/utils/nfl_calendar.py`: `get_current_nfl_season()`, `get_current_nfl_week()`
- `src/data/auto_refresh.py`: `NFLDataRefresher.refresh()`, `auto_refresh_data()`

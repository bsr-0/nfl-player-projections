# Year Parameters and Defaults

Single source of truth for all year/latest-season values. Aligns with the 2025/26 data and "This Week's Edge" plan.

## Config: `config/settings.py`

| Constant | Value / Source | Purpose |
|----------|----------------|---------|
| `MIN_HISTORICAL_YEAR` | `2020` | Earliest season to load/scrape (explicit input constant). |
| `AVAILABLE_SEASONS_START_YEAR` | `2016` | Earliest season for nfl-data-py availability checks. |
| `CURRENT_YEAR` | `datetime.now().year` | Calendar year (for display only; prefer NFL season where relevant). |
| `CURRENT_NFL_SEASON` | `get_current_nfl_season()` from `src.utils.nfl_calendar` | **Latest NFL season** (Sept–Feb: Jan–Aug = previous year, Sept–Dec = current year). |
| `SEASONS_TO_SCRAPE` | `range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1)` | Default load/scrape range. |
| `TRAINING_START_YEAR_DEFAULT` | `2015` | Default first year for training (balanced window). |
| `TRAINING_END_YEAR_DEFAULT` | `CURRENT_NFL_SEASON` | Latest season; used for `end_year` and `test_years`. |
| `TRAINING_YEARS` | `start_year`, `end_year`, `test_years` from above | Training window and test season (dynamic). |
| `TRAINING_WINDOW_PRESETS` | `modern`, `balanced`, `extended`, `full` | All use `TRAINING_END_YEAR_DEFAULT` for `end_year`. |

## Entry Points and Defaults

| Entry point | Year/season parameter | Default |
|-------------|------------------------|---------|
| **run_app.py** | `--from-year`, `--to-year`, `--seasons` | Range: `MIN_HISTORICAL_YEAR` to `CURRENT_NFL_SEASON` (from config). |
| **nfl_data_loader** (`load_all_historical_data`, CLI `--seasons`) | `seasons` | `range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1)`. |
| **schedule_scraper** (CLI `--year`) | `year` | `get_current_nfl_season()` (nfl_calendar). |
| **data_manager** | `get_prediction_season()` | `get_current_nfl_season()` (no duplicated calendar logic). |
| **train.py** | Error message / example | `MIN_HISTORICAL_YEAR`–`CURRENT_NFL_SEASON` from config. |
| **predict.py** / **generate_app_data.py** | Error message | Same as above. |
| **auto_refresh** | Check range | `range(MIN_HISTORICAL_YEAR, current_season + 2)`; `current_season` from nfl_calendar. |
| **optimize_training_years.py** | `test_year` (and saved config) | `TRAINING_END_YEAR_DEFAULT` (config). |
| **evaluate_deep_learning.py** | `test_year` | `TRAINING_END_YEAR_DEFAULT` (config). |
| **analytics_dashboard.py** | Slider "Training Years" | Min 2000, max `CURRENT_NFL_SEASON`, default `(TRAINING_START_YEAR_DEFAULT, CURRENT_NFL_SEASON)`. |
| **app.py** (UI strings) | Train/test text | From `TRAINING_YEARS` (start_year, test_years); "recent" filter uses `MIN_HISTORICAL_YEAR`. |

## Current NFL Season and Week

- **Source:** `src.utils.nfl_calendar` only.
  - `get_current_nfl_season(today=None)` → season year.
  - `get_current_nfl_week(today=None)` → week_num, season, is_playoffs, etc.
  - `is_future_or_current_matchup(data_season, data_week, today=None)` → used for "This Week's Edge".
- **No hardcoded 2025/2026** in calendar or app logic; all dates are computed from `today`.

## Coherence Rules

1. **Loading / scraping:** Default range is always `MIN_HISTORICAL_YEAR` through `CURRENT_NFL_SEASON` (from config).
2. **Training / test:** `end_year` and `test_years` equal `CURRENT_NFL_SEASON` (via `TRAINING_END_YEAR_DEFAULT`) unless overridden.
3. **CLI defaults:** All year arguments default to config or nfl_calendar; help text shows actual default values where possible.
4. **Error messages:** Reference config range (e.g. `MIN_HISTORICAL_YEAR`–`CURRENT_NFL_SEASON`) instead of literal "2020-2024".

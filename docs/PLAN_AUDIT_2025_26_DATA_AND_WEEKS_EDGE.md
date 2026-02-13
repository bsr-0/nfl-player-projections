# Plan Audit: 2025/26 Data and "This Week's Edge" Fix

**Audit date:** 2025-02-02  
**Plan:** `2025-26_data_and_week's_edge_fix_f3538275.plan.md`

This document reviews implementation against the plan’s success criteria, identifies gaps and hardcoded values, and instructs the main agent on required changes.

---

## Success Criteria (from Plan)

1. **"This Week's Edge"** is only presented as the **upcoming** matchup when data’s (season, week) is future or current; otherwise show a clear disclaimer or "latest data" label.
2. **Current NFL season and week** are computed **dynamically** from today (no hardcoded 2025 Super Bowl).
3. **2025/26 (and every current NFL season)** is **automatically** included in loading and refresh when available; data through the **exact current NFL week** is included.
4. When weekly data is missing or incomplete (e.g. mid-season 2025/26), **play-by-play** aggregation is used and merged into the same DB so all processes see complete history through current week.
5. Every place that compiles or filters by season/week uses the **same** notion of "current season" and "current week" and includes current season data by default.

---

## What Is Implemented Well

### 1. Single source of truth for current NFL season/week

- **`src/utils/nfl_calendar.py`** exists and provides:
  - `get_current_nfl_season(today=None)` – Jan–Aug → previous year, Sept–Dec → current year.
  - `get_current_nfl_week(today=None)` – week_num, season, playoffs, Super Bowl, etc., from season start/playoff dates (no hardcoded year).
  - `is_future_or_current_matchup(data_season, data_week, today=None)` – used for UI branching.
- **`app.py`**:
  - `get_current_nfl_week()` delegates to `nfl_calendar.get_current_nfl_week()` (no hardcoded 2025/2026).
  - Sidebar and "This Week's Edge" use `get_current_nfl_season`, `get_current_nfl_week`, and `is_future_or_current_matchup`; labels and disclaimers are shown when data is behind or from a past week.

**Verdict:** Success criteria 1 and 2 are met for the app and nfl_calendar.

### 2. "This Week's Edge" logic

- Compares `(latest_season, latest_week)` to current NFL (season, week).
- If data is behind: "Load {current_nfl_season} season data for current matchups."
- If data is past: "The data below is from a past week (Week X, Season Y)."
- Title switches to "Latest Data We Have" when not future/current.

**Verdict:** Success criterion 1 is met.

---

## Gaps and Required Fixes

### 1. PBP fallback not integrated (Success criterion 4 NOT met)

- **`src/data/pbp_stats_aggregator.py`** exposes `get_weekly_stats_from_pbp(season)` and column alignment with nfl_data_loader schema.
- **Neither `nfl_data_loader` nor `auto_refresh`** call this when weekly data is empty or has fewer weeks than expected.
- **Result:** When `nfl.import_weekly_data([2025])` is empty or missing weeks, the pipeline does **not** fall back to PBP aggregation. Current season through current week is not guaranteed.

**Required:**  
- In **`nfl_data_loader.load_weekly_data()`** (or a single place used by both loader and refresher): after calling `nfl.import_weekly_data(seasons)`, for each season where the result is empty or has fewer weeks than the current NFL week (for that season), call `get_weekly_stats_from_pbp(season)`, standardize columns, merge with any weekly rows (prefer weekly when both have same player-week), and store.  
- In **`auto_refresh.refresh()`**: when loading a season, if after `load_weekly_data` the DB still has no or insufficient weeks for that season, trigger the same PBP fallback (e.g. call into loader’s logic or `get_weekly_stats_from_pbp` + existing store path).  
- Ensure one code path performs "try weekly → if empty/incomplete then PBP → merge → store" so the DB always has the best available view through current week.

### 2. nfl_data_loader: hardcoded years and no current-season default (Criterion 3 & 5)

| Location | Issue | Fix |
|----------|--------|-----|
| `get_available_seasons()` | Returns `list(range(2016, 2026))` – hardcoded 2026 | Return `list(range(2016, get_current_nfl_season() + 2))` (or use shared constant from nfl_calendar). |
| `get_latest_available_season()` | Loops `range(2026, 2015, -1)`; fallback `return 2024` | Use `current_nfl_season + 1` as upper bound; fallback `return get_current_nfl_season()`. |
| `load_all_historical_data(seasons=None)` | `seasons = list(range(2020, 2025))` – 2025 excluded | When `seasons is None`, set `seasons = list(range(2020, get_current_nfl_season() + 1))` (or from config min_year through current NFL season). |
| No PBP fallback | See above | Integrate PBP fallback as in §1. |

**Required:**  
- Import and use `get_current_nfl_season()` from `src.utils.nfl_calendar` in `nfl_data_loader.py`.  
- Replace all fixed 2024/2025/2026 bounds with expressions based on `get_current_nfl_season()`.  
- Add PBP fallback in the load path as described in §1.

### 3. auto_refresh: duplicate season logic and no PBP fallback (Criterion 3, 4, 5)

| Location | Issue | Fix |
|----------|--------|-----|
| `get_current_nfl_season()` | Duplicates Jan–Aug / Sept–Dec logic | Remove; use `from src.utils.nfl_calendar import get_current_nfl_season` and call it. Prefer calling at refresh time, not only at __init__, so long-running processes see up-to-date season. |
| `check_data_availability()` | Uses `self.get_current_nfl_season()` and `range(2020, current_season + 2)` | Keep range logic but derive current_season from nfl_calendar. |
| `refresh()` | No PBP fallback when weekly is empty or has fewer weeks | After loading a season, if `get_latest_week_for_season(season)` is 0 or less than current NFL week for that season, call PBP fallback (e.g. `get_weekly_stats_from_pbp(season)`), standardize, merge with existing weekly, store. |

**Required:**  
- Use `nfl_calendar.get_current_nfl_season()` (and optionally `get_current_nfl_week()` for “expected” weeks).  
- Add PBP fallback in `refresh()` as above; ensure it uses the same storage path as the loader (e.g. loader’s `_store_weekly_data` or equivalent).

### 4. config/settings.py: hardcoded end_year and calendar vs NFL season (Criterion 5)

| Location | Issue | Fix |
|----------|--------|-----|
| `SEASONS_TO_SCRAPE` | `list(range(2020, CURRENT_YEAR + 1))` – calendar year; e.g. Jan 2026 excludes 2025 season | Use current **NFL** season: e.g. `from src.utils.nfl_calendar import get_current_nfl_season` and `list(range(2020, get_current_nfl_season() + 1))`. Avoid circular import (e.g. lazy import or a small settings_nfl.py that imports nfl_calendar). |
| `TRAINING_YEARS` | `"end_year": 2024`, `"test_years": [2024]` | Make dynamic: e.g. `end_year = get_current_nfl_season()`, `test_years = [get_current_nfl_season()]` (or “latest available” from data_manager when needed). |
| `TRAINING_WINDOW_PRESETS` | All presets use `"end_year": 2024` | Same: use `get_current_nfl_season()` for end_year so 2025/26 and beyond are included when available. |

**Required:**  
- Ensure scraping and training defaults include the current NFL season and are not tied to calendar year only.  
- Replace hardcoded 2024 in training windows with the shared current-season utility (with care for import order).

### 5. data_manager.py: hardcoded error message (Criterion 5)

| Location | Issue | Fix |
|----------|--------|-----|
| `get_train_test_seasons()` | `ValueError("... 2020-2024")` | Use "... 2020 through current season" or "... e.g. 2020-2025" so it stays correct every year. |

**Required:**  
- Replace the hardcoded "2020-2024" in the error message with a dynamic or generic phrase (e.g. "through current season" or "e.g. 2020-{current_nfl_season}").

### 6. Scripts and other files: hardcoded years (Criterion 5)

These should be updated to use the shared nfl_calendar (or at least dynamic current season) so they don’t drift:

| File | Issue | Fix |
|------|--------|-----|
| `scripts/realtime_integration.py` | `current_season=2025`, `datetime(2025, 9, 4)`, `datetime(2026, 2, 9)` | Use `get_current_nfl_season()` and `get_current_nfl_week()` (or nfl_calendar’s season start/SB dates) so it works for any year. |
| `scripts/analytics_dashboard.py` | Many `2024`, `2025`, `range(2000, 2025)`, `datetime(2025, 9, 4)`, etc. | Use current NFL season and nfl_calendar for phase/season start; replace fixed ranges with dynamic where it affects “current” behavior. |
| `scripts/email_alerts.py` | `season_start = datetime(2025, 9, 4)` | Use nfl_calendar (e.g. `_season_start(get_current_nfl_season())` or equivalent). |
| `src/data/pbp_stats_aggregator.py` | `load_current_season_stats_from_pbp()` uses `current_year - 1 if month <= 8 else current_year` | Replace with `get_current_nfl_season()` from nfl_calendar for consistency. |
| `src/features/multiweek_features.py` | `current_season = ... else 2024` | Use `get_current_nfl_season()` for fallback when `result['season'].max()` is missing. |
| `src/models/train.py` | Error messages / examples "2020-2024" | Prefer "through current season" or dynamic current season in messages. |
| `run_app.py` | Default seasons `[2023, 2024]` and docstring "2023-2024", "2020-2024" | Default to include current NFL season (e.g. from nfl_calendar or data_manager); update docstrings to say "e.g. 2020–current season". |
| `config/settings.py` | See §4 above. | Already covered. |

**Required:**  
- Audit all scripts and app entrypoints that mention “current” season or week; switch them to `nfl_calendar.get_current_nfl_season()` / `get_current_nfl_week()` (or equivalents).  
- Replace hardcoded 2024/2025/2026 in user-facing messages and defaults with dynamic values or generic wording.

### 7. pbp_stats_aggregator: hardcoded function name

- `load_2025_season_stats()` is a convenience wrapper; the name is year-specific. Prefer keeping `get_weekly_stats_from_pbp(season)` and `load_current_season_stats_from_pbp()` (with current season from nfl_calendar) as the main API, and deprecate or rename `load_2025_season_stats()` to something like `load_season_stats_from_pbp(season=2025)` or remove it in favor of `get_weekly_stats_from_pbp(get_current_nfl_season())`.

---

## Hardcoded Years / Numbers Summary

Places that still hardcode a year or range and can cause wrong “latest” or “current” behavior:

- **nfl_data_loader.py:** 2016, 2020, 2024, 2025, 2026 in `get_available_seasons`, `get_latest_available_season`, `load_all_historical_data`, CLI default.
- **auto_refresh.py:** Duplicate season logic (no literal 2026 but duplicated logic).
- **config/settings.py:** CURRENT_YEAR (calendar), 2024 in TRAINING_YEARS and TRAINING_WINDOW_PRESETS.
- **data_manager.py:** "2020-2024" in error message.
- **scripts/realtime_integration.py:** 2025, 2026 in constructor and date literals.
- **scripts/analytics_dashboard.py:** 2024, 2025, 2026, range(2000, 2025), datetime(2025, 9, 4).
- **scripts/email_alerts.py:** 2025 in season_start.
- **pbp_stats_aggregator.py:** `load_current_season_stats_from_pbp` uses calendar logic; `load_2025_season_stats` name.
- **src/features/multiweek_features.py:** 2024 fallback.
- **run_app.py:** 2023, 2024 in defaults and docstrings.
- **app.py:** Some UI strings still say "2020-2023" / "2024" (e.g. trained on 2020-2023, test 2024); should reflect “train through prior season, test on latest/current” dynamically where possible.
- **train.py, predict.py, pipeline.py, scripts (generate_app_data, run_scrapers, etc.):** Various "2020-2024" or fixed test years in messages or defaults; should be updated to “current season” or dynamic.

---

## Instructions for Main Agent

1. **Implement PBP fallback in the main pipeline**  
   - In `nfl_data_loader` (and/or `auto_refresh`), after attempting weekly load for a season, if the result is empty or has fewer weeks than the current NFL week for that season, call `get_weekly_stats_from_pbp(season)`, align columns with `_standardize_weekly_columns` / schema, merge with any weekly data (weekly wins on overlap), and store via the same path used for weekly data.  
   - Ensure `auto_refresh.refresh()` also triggers this fallback when a season is missing or incomplete.

2. **Use a single source of truth for “current” everywhere**  
   - In `nfl_data_loader`, `auto_refresh`, `config/settings`, `data_manager`, `realtime_integration`, `analytics_dashboard`, `email_alerts`, `pbp_stats_aggregator`, and `multiweek_features`: replace duplicated season/week logic and hardcoded 2024/2025/2026 with `src.utils.nfl_calendar.get_current_nfl_season()` and, where needed, `get_current_nfl_week()`.  
   - Resolve circular imports (e.g. config importing nfl_calendar) via lazy import or a small bootstrap module.

3. **Make defaults include current NFL season**  
   - `load_all_historical_data(seasons=None)` should default to `range(2020, get_current_nfl_season() + 1)` (or configurable min).  
   - `get_available_seasons()` / `get_latest_available_season()` should use current NFL season (and current + 1 where appropriate) instead of 2026/2024.  
   - Training/test defaults and presets in config should use current NFL season for end_year and test_years.

4. **Fix user-facing and error messages**  
   - Replace "2020-2024" and fixed “2024” in errors and docstrings with “through current season” or dynamic current season.  
   - Ensure “This Week’s Edge” and any “latest data” messaging stay correct as implemented; no further change needed there if the above are done.

5. **Optional but recommended**  
   - Add a small integration test or script that, for a given date (e.g. Feb 2, 2026), asserts current season is 2025 and current week is Super Bowl (or similar), and that “This Week’s Edge” and data-load defaults would include 2025.  
   - Document in README or ops runbook that the default workflow runs auto_refresh (or equivalent) so current season through current week is included, and that PBP is used when weekly data is missing.

---

## Verdict

- **Criteria 1 & 2:** Met (nfl_calendar + app “This Week’s Edge” and disclaimers).  
- **Criteria 3, 4, 5:** Not fully met until:  
  - PBP fallback is integrated in loader and auto_refresh,  
  - nfl_data_loader and config/settings default to current NFL season and use nfl_calendar,  
  - auto_refresh uses nfl_calendar and PBP fallback,  
  - and all remaining hardcoded 2024/2025/2026 in defaults and messages are removed or made dynamic.

Implementing the changes above will satisfy and exceed the plan’s success criteria in a robust, maintainable way.

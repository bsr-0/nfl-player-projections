# Missing Data, Injury/Rookie Handling, and Rigorous Testing

## Missing data: root causes and fixes

### Where nulls come from

1. **LEFT JOINs in training data**  
   `get_all_players_for_training` uses LEFT JOINs to:
   - `team_stats` (team tendencies: pass/rush attempts, red zone, etc.)
   - `utilization_scores` (snap/target/rush share)
   - `team_defense_stats` (opponent fantasy points allowed)

   When those tables are empty or a (team, season, week) / (player, season, week) is missing, joined columns are NULL.

2. **Schedule / strength-of-schedule**  
   When schedule or StrengthOfScheduleCalculator fails or has no data for a season, `opponent_rating`, `matchup_difficulty`, and `team_sos` can be NaN. These are filled with neutral defaults (50.0) in `_add_schedule_features` and `refresh_matchup_features`.

3. **Rolling and lag features**  
   Rolling means and lags produce NaN for the first few rows per player. These are cleaned in a final imputation step and in `prepare_training_data` so the model never sees NaN or inf.

### How we fix or handle it

- **Team stats**: Backfill from player-level aggregates when `team_stats` is empty (`ensure_team_stats_from_players` in `database.py`), run from `auto_refresh` so pipelines work without scrapers.
- **Schedule**: nfl-data-py first, scraper fallback; matchup features use neutral defaults when schedule is missing.
- **Final imputation**: In `FeatureEngineer.create_features`, after all feature steps we run:
  - `_ensure_injury_rookie_features`: adds injury/rookie columns with safe defaults if missing.
  - `_check_missing_rate`: logs a warning for any numeric feature with >5% missing (requirement: max 5% per feature acceptable). Does not drop columns.
  - `_impute_missing`: replaces inf with nan, then fills NaN in numeric columns with column median (fallback 0) so every pipeline gets finite inputs.
- **Training**: `prepare_training_data` also does `X.replace([np.inf, -np.inf], np.nan).fillna(0)` so model inputs are always clean.

No separate “broken pipeline” repair is required if auto_refresh and the above steps run; if a source is still missing, imputation and defaults keep the pipeline running. Requirements specify max 5% missing per feature; we flag exceedances in `_check_missing_rate` (warning only). Imputation: column median, then 0.

---

## Rookie and injury data in the ML workflow

### Main pipeline (feature_engineering.py)

- **Injury**: Optional columns `injury_score` (0–1 availability) and `is_injured` (0/1) are ensured in `_ensure_injury_rookie_features`. If absent (e.g. no external injury load), they default to full availability (1.0 and 0). If present (e.g. from `external_data.add_external_features`), values are preserved and clipped. They act as predictors for utilization/performance.
- **Rookie**: `is_rookie` is added in the same step: when `games_count` exists it is “games_count ≤ 8”; otherwise we use a proxy from “first 8 rows per player” in the dataset. This flags limited-sample players so the model can account for higher uncertainty.

These columns are numeric and included in the feature list, so they are used in training and prediction when the main `FeatureEngineer` is used.

### Advanced pipelines

- **Rookie**: `AdvancedRookieProjector` in `advanced_rookie_injury.py` (draft capital, combine, comparables, breakout/bust probability) is used in advanced/experimental flows (e.g. `train_advanced`, scripts), not in the default `train.py` → `FeatureEngineer.create_features` path. The main path’s `is_rookie` and cold-start handling in `predict.py` (position-average for very low games_count) give a simpler, robust use of “rookie” in the core workflow.
- **Injury**: Full injury modeling (probability, recovery trajectory, history) lives in `multiweek_features.InjuryProbabilityModel` and `advanced_rookie_injury.AdvancedInjuryPredictor`. External injury status is merged in via `external_data.add_all_external_features` (used in advanced pipelines). The main pipeline only needs the simple `injury_score` / `is_injured` defaults so that when advanced pipelines add real injury data, the same feature names exist and are used.

### Feature engineering value for utilization

- **Injury**: Using `injury_score` as a feature (and optionally as an availability multiplier) lets the model downweight utilization for players who are less likely to play. Defaults ensure no crash when injury data is missing.
- **Rookie**: `is_rookie` (or low `games_count`) helps the model treat early-career or limited-sample players differently; cold-start in prediction then applies position-average when games_count is below threshold. Together they improve utilization prediction for both injured and rookie players.

---

## Rigorous testing of new features and missing data

### Tests in `tests/test_missing_data_and_new_features.py`

- **No NaN/inf in model inputs**:  
  - `test_create_features_produces_no_nan_or_inf_in_numeric_columns`: after `create_features`, no numeric column contains NaN or inf.  
  - `test_prepare_training_data_returns_clean_X_y`: `prepare_training_data` returns X and y with no NaN and no inf.

- **Injury/rookie features**:  
  - `test_injury_rookie_features_exist_with_defaults_when_missing`: with no injury or games_count input, `injury_score`, `is_injured`, and `is_rookie` exist and have safe, in-range values.  
  - `test_injury_rookie_features_preserve_valid_input`: when `injury_score` / `is_injured` are provided, they are preserved (and clipped) rather than overwritten.

- **Imputation**:  
  - `test_impute_missing_removes_inf`: `_impute_missing` replaces inf with finite values.  
  - `test_impute_missing_fills_nan`: `_impute_missing` fills NaN in numeric columns.

- **Matchup defaults**:  
  - `test_refresh_matchup_features_handles_missing_with_defaults`: `refresh_matchup_features` fills `team_sos`, `matchup_difficulty`, `opponent_rating` when missing (with mocks for schedule/matchup).

Running:

```bash
pytest tests/test_missing_data_and_new_features.py -v
```

along with existing `test_feature_engineering.py` and `test_matchup_aware_prediction.py` ensures that new features are implemented and that missing data and pipeline robustness are covered.

---

## Summary

- **Missing data**: Root causes are LEFT JOINs (team_stats, utilization, defense), optional schedule/SOS, and rolling/lag NaNs. Fixes are team_stats backfill, schedule loading (nfl-data-py first), and a final imputation step plus `prepare_training_data` cleaning so model inputs are never NaN/inf.
- **Rookie and injury**: The main ML workflow uses `injury_score`, `is_injured`, and `is_rookie` with safe defaults in `FeatureEngineer`; advanced rookie/injury models are used in advanced pipelines. Injury and rookie features increase the value of utilization prediction when data is available and degrade gracefully when it is not.
- **Testing**: Dedicated tests enforce no NaN/inf in model inputs, presence and defaults of injury/rookie features, imputation behavior, and matchup default handling, so new features are fully implemented and errors are caught.

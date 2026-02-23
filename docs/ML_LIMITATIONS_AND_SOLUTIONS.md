# ML Limitations and Solutions

From the perspective of a senior machine learning engineer, here are observed limitations in the NFL Predictor codebase and implemented solutions.

---

## 1. Training Year Selection

**Limitation**: Fixed or arbitrary training window (e.g., last 4 seasons). TRAINING_YEARS in config was not used by the main pipeline. Older data may teach outdated patterns; too little data causes high variance.

**Solution**: 
- **Dynamic optimization**: `--optimize-years` runs `select_optimal_training_years()` which evaluates windows (3, 5, 7, 10, 15, all) via time-series CV and selects the window that minimizes test RMSE per position.
- **Use all available by default**: `n_train_seasons=None` uses ALL seasons before the test season (no artificial cap).
- **DB fallback**: When schedule loading returns empty, `get_available_seasons_from_db()` falls back to database to determine available seasons.
- **Caching**: Optimal years saved to `data/models/optimal_training_years.json`.

---

## 2. Test Set Configuration

**Limitation**: Unclear whether the process automatically uses the latest year(s) as the test set.

**Solution**: **Yes.** `DataManager.get_train_test_seasons()` uses `latest_season` (or `max(available)`) as the test season by default. Train = all seasons before test. Predictions are evaluated on the most recent out-of-sample season.

---

## 3. Time-Series Validation

**Limitation**: Basic TimeSeriesSplit; no purge gap or embargo to reduce leakage at train/validation boundaries.

**Solution**:
- **Purged CV**: `RobustTimeSeriesCV(gap_seasons=N)` excludes the last N seasons before the test season from training, reducing temporal leakage. Config: `MODEL_CONFIG["cv_gap_seasons"]`.
- **Season-based folds**: Each fold trains on past seasons only, tests on a future season.
- **Walk-forward backtester**: `WalkForwardValidator` and `FantasyBacktester` support week-by-week and season-by-season walk-forward validation.

---

## 4. Recency Weighting

**Limitation**: All historical samples weighted equally. Recent seasons may be more relevant (rule changes, offensive trends).

**Solution**: Config option `recency_decay_halflife` (e.g., 2.0 = weight halves every 2 seasons). Sample weights can be passed to tree models for future integration. The dynamic training year selector already favors recent data when it performs better.

---

## 5. Data Leakage

**Limitation**: Risk of target or future information leaking into features.

**Solution**: 
- `fantasy_points` (current week) explicitly excluded from features.
- Rolling/lag features use `shift(1)`.
- Targets use `shift(-1)`.
- Defense rankings shifted by +1 week.
- `fp_over_expected`, `expected_fp` excluded.
- Assertion in ensemble: `assert "fantasy_points" not in feature_cols`.

---

## 6. Multicollinearity

**Limitation**: Highly correlated features can destabilize Ridge and inflate variance.

**Solution**: 
- `select_features_simple()` removes pairs with correlation > 0.92.
- `compute_vif()` reports VIF after selection; training logs features with VIF > 10.
- Ridge is relatively robust; tree models are invariant to monotonic transforms.

---

## 7. Utilization Score Weights

**Limitation**: Hardcoded weights; not data-driven.

**Solution**: `utilization_weight_optimizer.fit_utilization_weights()` fits optimal component weights from training data (non-negative least squares). Weights persisted for train/serve consistency.

---

## 8. Cold Start

**Limitation**: Rookies or players with few games excluded or poorly handled.

**Solution**: `_apply_cold_start_fallback()` in predict.py uses position-average projections with elevated uncertainty when `games_count < MIN_GAMES_FOR_PREDICTION`.

---

## 9. Scalability and Reproducibility

**Limitation**: Non-deterministic splits or insufficient seed control.

**Solution**: 
- `random_state=42` in config and models.
- Data sorted by `['season','week']` before splits.
- Optuna uses fixed seed for TPESampler.

---

## Summary of Commands

```bash
# Train with all available years (default)
python -m src.models.train

# Dynamically optimize training years per position
python -m src.models.train --optimize-years

# Override test season
python -m src.models.train --test-season 2024
```

---

## Files Modified/Added

| File | Purpose |
|------|---------|
| `src/utils/training_years_selector.py` | Dynamic training year optimization |
| `src/utils/data_manager.py` | DB fallback, n_train_seasons=None, optimal_years |
| `src/models/robust_validation.py` | Purged CV (gap_seasons) |
| `config/settings.py` | recency_decay_halflife, cv_gap_seasons |
| `src/models/train.py` | --optimize-years, --test-season, integrate selector |

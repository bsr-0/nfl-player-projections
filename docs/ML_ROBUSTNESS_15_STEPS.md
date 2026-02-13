# 15 Crucial Steps for Robust ML Forecasting

A checklist for building a robust, accurate machine learning system that excels on unseen test data. This document maps each step to the NFL Predictor codebase implementation.

## Step 1: Split Data Before Any Preprocessing
**Principle**: Train/validation/test split must occur first. No preprocessing (scaling, imputation, feature engineering that uses global stats) should use information from validation/test sets.

**Implementation**: `train.py` - `load_training_data()` returns train_data and test_data split by season. Utilization scores and feature engineering are applied separately to train and test. Each receives only its own data - no combined statistics. Utilization weights are fit on train only, then applied to both.

## Step 2: Time-Series Appropriate Splitting
**Principle**: Use temporal splits only. Never random split for time-series. Training must always precede test chronologically.

**Implementation**: `DataManager.get_train_test_seasons()` - Test season is the latest; train seasons are all prior. `ModelTrainer` sorts by `['season','week']` before training. `RobustTimeSeriesCV` uses season-based folds.

## Step 3: Fit Scaler on Training Data Only
**Principle**: Scaling parameters (mean, std) must be computed from training data only.

**Implementation**: `position_models.py` - `self.scaler.fit(X_train_inner)` then `transform()` for val. At inference, `_prepare_input()` uses `self.scaler.transform(X_np)` - scaler was fit during training, never on inference data.

## Step 4: Transform All Datasets with Training Parameters
**Principle**: Validation and test sets are transformed using the SAME parameters derived from training. Never re-fit on val/test.

**Implementation**: `position_models.py` - `X_val_scaled = self.scaler.transform(X_val)`. Prediction: `self.scaler.transform(X_np)`. Scaler persisted in `model_data["scaler"]`.

## Step 5: Prevent Feature Leakage
**Principle**: No feature may use information from the target period. Rolling/lag features use `shift(1)`. Targets use `shift(-1)`.

**Implementation**: `feature_engineering.py` - Rolling: `x.shift(1).rolling()`. WoW change: `shift(1).diff()`. Targets: `shift(-1).rolling().sum()`. `external_data.py` - defense rankings shifted: `defense_rankings['week'] = defense_rankings['week'] + 1`.

## Step 6: Cross-Validation with Time-Series Splits
**Principle**: Use TimeSeriesSplit or expanding-window CV. Each fold: train on past, validate on future.

**Implementation**: `position_models.py` - Optuna uses `TimeSeriesSplit(n_splits=MODEL_CONFIG["cv_folds"])`. `RobustTimeSeriesCV` uses season-based folds. `train.py` runs `_run_robust_cv_report()` for Ridge baseline.

## Step 7: Validation Set for Hyperparameter and Ensemble Decisions
**Principle**: Tune hyperparameters and ensemble weights on a held-out validation set, never on training or test.

**Implementation**: `position_models.py` - 80/20 time-based split. Hyperparameter tuning uses `X_train_scaled`. Ensemble weights and meta-learner use `X_val_scaled`, `y_val`.

## Step 8: Early Stopping to Prevent Overfitting
**Principle**: Use validation set for early stopping in gradient boosting. Prevents overfitting on noise.

**Implementation**: `position_models.py` - XGBoost and LightGBM use `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=25`.

## Step 9: Feature Selection Fit on Train Only
**Principle**: Feature selection (importance, correlation filter) must be fit on training data. Same feature set applied to test.

**Implementation**: `ensemble.py` - `select_features_simple(X, y_dict[1], ...)` fit on training X,y. Selected columns passed to `multi_model.fit()`. Model stores `feature_names`. At prediction, `EnsemblePredictor` fills missing columns with 0.

## Step 10: Handle Missing Values Without Leakage
**Principle**: Imputation (median, mean) must use training statistics only. Test uses training-derived fill values.

**Implementation**: `position_models.py` - `fillna(0)` and `np.nan_to_num(X_np, nan=0.0)` - conservative zero fill. No global imputation that would leak. Feature engineering uses `safe_divide` and per-row logic.

## Step 11: Target Outlier Treatment
**Principle**: Winsorize or cap extreme targets to reduce skew. Use percentiles computed on training data only.

**Implementation**: `train.py` - Winsorize at 1st/99th percentile per position on `train_data` only, before model training.

## Step 12: Persist All Preprocessing Artifacts
**Principle**: Scaler, feature selector, feature names must be saved with the model. Inference must apply identical preprocessing.

**Implementation**: `position_models.py` - `model_data` includes `scaler`, `feature_names`, `meta_learner`. `PositionModel.load()` restores them. `_prepare_input()` applies scaler.

## Step 13: Train/Serve Consistency
**Principle**: Same columns, same order, same transformations at serve time. Handle missing columns gracefully.

**Implementation**: `ensemble.py` - `_fill_missing_features()` adds missing columns with 0. `position_models.py` - `X.reindex(columns=self.feature_names).fillna(0)`.

## Step 14: Final Evaluation on Held-Out Test Set
**Principle**: Report metrics on test set only after all model choices are finalized. Never tune on test.

**Implementation**: `train.py` - `_report_test_metrics()` evaluates on `test_data` after training. Test season is separate from train seasons.

## Step 15: Reproducibility
**Principle**: Random seeds fixed. Deterministic splits. Same code produces same results.

**Implementation**: `MODEL_CONFIG["random_state"] = 42`. Optuna `TPESampler(seed=...)`. All model constructors use `random_state`. Data sorted by season, week for deterministic splits.

---

## Additional Safeguards

**Primary variable is utilization, not fantasy_points**: Raw `fantasy_points` (current week) is explicitly excluded from features. The model predicts future fantasy points from opportunity metrics (utilization components, snap share, target share, etc.) and historical performance (rolling/lag of fantasy points - which use `shift(1)`). `ensemble.py` asserts `"fantasy_points" not in feature_cols`.

**Multicollinearity**: `select_features_simple()` removes features with correlation > 0.92. Post-selection VIF is reported during training (`compute_vif()`). Ridge regression is robust to moderate multicollinearity.

**Data-driven utilization weights**: `utilization_weight_optimizer.fit_utilization_weights()` fits optimal component weights from training data (Ridge + nnls for non-negativity). Falls back to config defaults when insufficient data.

---

## Backtesting

Rigorous backtesting uses only the holdout test season; all preprocessing and model artifacts are train-derived. See [BACKTESTING.md](BACKTESTING.md) for: (1) holdout definition and strict unseen test, (2) no leakage (weights/scaling/feature selection fit on train only, applied to test), (3) backtest using the same production pipeline (EnsemblePredictor + same feature prep).

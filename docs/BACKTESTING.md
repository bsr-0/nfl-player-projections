# Backtesting

Rigorous backtesting ensures production models excel on **truly unseen** test data. This document describes how the NFL Predictor backtest is designed to avoid leakage and match the production pipeline.

## Holdout definition

- **Test set**: One holdout season (by default the **latest** available season). This season is used only for final evaluation; it is never used for training, hyperparameter tuning, or fitting any preprocessing.
- **Train set**: All seasons strictly before the test season. Cross-validation and tuning use only train (and an internal validation split within train).

The split is enforced by `DataManager.get_train_test_seasons()` and asserted in `load_training_data()` and in `run_backtest()` so that `test_season not in train_seasons`.

## No leakage

1. **Preprocessing**: Utilization weights are fit on **train only** (and persisted to `data/models/utilization_weights.json`). When backtesting or serving, test/inference data is prepared using these train-derived weights only (no fitting on test).
2. **Scaling and feature selection**: Scalers and feature selectors are fit on training data during model training. At backtest and serve time, the same persisted artifacts are applied (transform only).
3. **Hyperparameters and model choice**: All tuning and model/ensemble decisions use only training (and internal validation) data. The test season is touched only for the final backtest report.

## Production pipeline

Backtesting uses the **same** pipeline as production:

- **Data**: Same `load_training_data()` (positions, min_games, season split) as in training.
- **Features**: Same utilization weights (loaded from disk), same feature engineering (add_engineered_features). No refitting on test.
- **Model**: Persisted production ensemble (`EnsemblePredictor` loading from `data/models/`). Predictions are generated with `predict(player_data, n_weeks=1)` (or the horizon being backtested).

So backtest performance is a direct estimate of how the model will perform in production on unseen data.

## How to run

- **Standalone backtest** (after models are trained):  
  `python -m src.evaluation.backtester --season YYYY`  
  (omit `--season` to use the latest available test season.)  
  Writes `data/advanced_model_results.json` with `backtest_results` for the app.

- **Multi-season** (stability across years):  
  `python -m src.evaluation.backtester --multi-season 3`  
  Runs backtest on the last 3 seasons and reports mean ± std of RMSE, MAE, R².

- **With training**: Running `python -m src.models.train` trains models and runs a full backtest after training, writing `data/advanced_model_results.json` so the app displays production backtest metrics. This is the single source of truth for production model quality.

## Output and reproducibility

Backtest results include:

- **Metrics**: Overall and per-position RMSE, MAE, R², directional accuracy, within-X-points rates, ranking accuracy (top 5/10/20 hit rates).
- **Baseline comparison**: Model vs. a simple rolling 4-week average baseline (RMSE/MAE/R² and % improvement).
- **Config**: `train_seasons`, `test_season`, `backtest_date`, and (when saved) `model_source` so runs are auditable and reproducible.

Results are saved under `data/backtest_results/` and, in app-compatible form, to `data/advanced_model_results.json` (per-position `backtest_results` for the UI).

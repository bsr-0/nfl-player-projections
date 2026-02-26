# Prediction Model Review: Limitations and Fixes for Out-of-Sample Excellence

**Date:** 2026-02-26
**Scope:** 4-week LSTM+ARIMA hybrid (`Hybrid4WeekModel`) and 18-week deep residual feedforward (`DeepSeasonLongModel`) in `src/models/horizon_models.py`, plus their integration via `EnsemblePredictor` and the training pipeline.

---

## Executive Summary

The two horizon models represent solid architectural choices — an LSTM+ARIMA hybrid for medium-range weekly prediction and a residual feedforward network for season-long projection. However, several issues ranging from **critical data leakage** to **missing regularization** will degrade out-of-sample performance. This document identifies 14 issues across 4 severity categories, with concrete fixes for each.

---

## CRITICAL — Will Materially Harm Out-of-Sample Accuracy

### C1. Temporal Data Leakage in Train/Val Splits (Both Models)

**Location:** `horizon_models.py:143,215,548,615`

Both models split data with a simple `split = int(n * 0.8)` index-based cut. This is applied *after* data has been sorted by player and chronologically sequenced, meaning the "validation" set is just the last 20% of chronological rows. While this is better than random splitting, there are two problems:

1. **Within the LSTM model:** Sequences are built per-player and then concatenated across players before splitting. If players have different career lengths, the 80/20 split places some players' later seasons into validation while other players' entire histories remain in training. This is not a clean temporal holdout — it mixes time periods across players.

2. **Within the Optuna tuning objective:** The same 80/20 split is used for *both* hyperparameter selection and early stopping. This means hyperparameters are optimized to overfit the validation set, which is then also used as the stopping criterion. The result is double-dipping: tuned hyperparameters will appear better than they actually generalize.

**Fix:** Use a proper season-based split. Group all data by season, hold out the final 1-2 seasons as validation, and ensure the split respects season boundaries. For Optuna, use a nested inner-CV (the existing `SeasonAwareTimeSeriesSplit` from `position_models.py` is available).

```python
# Example: season-aware split for horizon models
seasons = df['season'].values  # aligned with X rows
unique_seasons = sorted(set(seasons))
val_seasons = set(unique_seasons[-2:])  # last 2 seasons for validation
train_mask = ~np.isin(seasons, list(val_seasons))
val_mask = np.isin(seasons, list(val_seasons))
```

### C2. LSTM DataLoader `shuffle=False` Prevents Stochastic Gradient Descent

**Location:** `horizon_models.py:155,222`

The LSTM training `DataLoader` uses `shuffle=False`. For the LSTM, this means the optimizer sees sequences in exactly the same deterministic order every epoch. This:
- Eliminates the mini-batch stochasticity that prevents local minima entrapment
- Creates systematic bias: the gradient updates in each epoch follow the same trajectory, leading to memorization rather than generalization
- Is especially harmful with only 80 epochs and small NFL datasets (~3K-8K samples per position)

Note: The deep feedforward model correctly uses `shuffle=True` at line 622, so only the LSTM is affected.

**Fix:** Change to `shuffle=True` in both LSTM locations (line 155 and 222).

### C3. ARIMA Produces Static Forecasts — Not True Per-Game Predictions

**Location:** `horizon_models.py:306-343`

The ARIMA model fits once on historical data during `fit()` and stores a single 4-step-ahead forecast per player in `_player_forecast`. At prediction time, it returns this same static forecast regardless of what week is being predicted or what new data is available.

This means:
- **Week 1 prediction = Week 4 prediction** for the same player (identical values)
- The ARIMA component contributes zero discriminative power across time — it's effectively a fancy per-player constant
- The 40% weight given to ARIMA (default `arima_weight=0.4`) dilutes the LSTM's predictions with stale information

**Fix:** The ARIMA component should be re-fit or updated at prediction time using the most recent data, or the architecture should be changed so ARIMA predictions roll forward. Alternatively, reduce ARIMA weight substantially and use it only as a regularizing prior.

---

## HIGH — Significant Accuracy Degradation

### H1. No Gradient Clipping in Neural Networks

**Location:** `horizon_models.py:224-238, 624-638`

Neither the LSTM nor the deep feedforward model applies gradient clipping. NFL fantasy data is inherently noisy with explosive outlier games (e.g., a RB scoring 45+ points). These outliers produce large loss values that create gradient spikes, which can:
- Destabilize LSTM hidden states (LSTMs are especially vulnerable to exploding gradients in deeper stacks)
- Cause the model to "jump" away from a good solution and never recover, especially with patience-based early stopping

**Fix:** Add `torch.nn.utils.clip_grad_norm_` before `optimizer.step()`:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

### H2. No Weight Decay / L2 Regularization in Adam Optimizer

**Location:** `horizon_models.py:224, 624`

Both models use plain `Adam` with no weight decay. For small tabular datasets like NFL stats (~3K-8K samples), neural networks can easily memorize training data. The residual feedforward model has dropout (0.35) and batch normalization, which helps, but the LSTM relies only on dropout (0.25) between layers.

**Fix:** Switch to `AdamW` with weight decay:
```python
optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
```

### H3. No Learning Rate Scheduling

**Location:** `horizon_models.py:224, 624`

Both models use a fixed learning rate throughout training. A large learning rate at the start helps converge quickly, but the same rate near convergence causes oscillation around the optimum — particularly damaging when early stopping is the only training termination criterion.

**Fix:** Add `ReduceLROnPlateau` scheduler:
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# After validation loss computation:
scheduler.step(val_loss)
```

### H4. Deep Feedforward `predict()` Silently Pads or Truncates Features

**Location:** `horizon_models.py:662`

```python
X_in = X[:, :self.n_features] if X.shape[1] >= self.n_features else np.hstack([X, np.zeros(...)])
```

If the prediction-time feature matrix has more or fewer columns than training, the model silently truncates or zero-pads. This means:
- Extra features are discarded (potentially losing valuable signal from new features)
- Missing features are filled with zeros rather than training medians (zero may be far from the feature distribution center, especially after StandardScaler)
- **No error or warning is raised**, so feature misalignment bugs are silently ignored

**Fix:** Validate column alignment explicitly, reindex to training feature order, and fill missing features with the scaler's mean (not zero):
```python
if set(feature_names) != set(expected_features):
    warnings.warn(f"Feature mismatch: {len(feature_names)} provided vs {len(expected_features)} expected")
# Reindex and fill with scaler mean (center of scaled distribution = 0 after transform)
```

### H5. LSTM Sequence Construction Drops Players with Short History

**Location:** `horizon_models.py:186-194, 259-282`

Players with fewer games than `sequence_length` (default: 10) produce zero sequences and are silently excluded from both training and prediction. This systematically removes:
- Rookies (critical for out-of-sample relevance)
- Players who changed teams (reset game count)
- Players returning from long-term injury

At prediction time (`predict()`), these players get `np.nan`, which cascades to the hybrid's fallback logic. The fallback replaces with `traditional_pred`, but if the ARIMA also failed (no history), the player gets neither model's contribution.

**Fix:** For players with `n < sequence_length`, pad their sequences with zeros or use a shorter sequence length. Alternatively, implement a learned embedding for short-history players.

### H6. 18-Week Model's `regression_to_mean_scale` Is Set But Never Used

**Location:** `horizon_models.py:529`

```python
self.regression_to_mean_scale = 0.95
```

This attribute is initialized and persisted but never applied during prediction. Over an 18-week horizon, mean reversion is a dominant force in NFL performance (especially for TDs, big plays, and boom/bust games). Without applying this, the model will overweight recent hot/cold streaks in its predictions.

**Fix:** Apply regression to mean in `predict()`:
```python
league_mean = np.mean(traditional_pred)
deep_pred = deep_pred * self.regression_to_mean_scale + league_mean * (1 - self.regression_to_mean_scale)
```

---

## MEDIUM — Reduces Model Quality

### M1. Fixed LSTM/ARIMA Blend Weights (60/40) Are Not Validated

**Location:** `horizon_models.py:352-353`

The 4-week hybrid model uses a fixed 60% LSTM / 40% ARIMA blend. These weights are not validated against out-of-sample performance. The 18-week model has a `learn_blend_weight()` method that optimally sets deep/traditional blend — the same approach should be applied here.

**Fix:** Add `learn_blend_weight()` to `Hybrid4WeekModel` that optimizes LSTM/ARIMA weights on a held-out validation season using `scipy.optimize.minimize_scalar`.

### M2. Optuna Tuning in `DeepSeasonLongModel` Uses Fixed Architecture (256, 64)

**Location:** `horizon_models.py:555-565`

The deep model's Optuna tuning only varies `dropout`, `learning_rate`, and `n_blocks_per_stage`, but the stage widths are hardcoded at `[(256, n_blocks), (64, n_blocks)]`. This means the architecture search is limited to depth only, missing potentially better width configurations (e.g., 128→32 for smaller datasets).

**Fix:** Include width as a tunable parameter:
```python
width1 = trial.suggest_categorical("stage1_width", [128, 256, 512])
width2 = trial.suggest_categorical("stage2_width", [32, 64, 128])
```

### M3. `Hybrid4WeekModel` Ignores `n_weeks` at Prediction Time

**Location:** `ensemble.py:297-310`

The hybrid model is invoked for weeks 4-8 (`horizon_4w_weeks`), but the `Hybrid4WeekModel.predict()` method doesn't receive `n_weeks`. The ARIMA forecast is always a 4-step-ahead average, even when predicting 6 or 8 weeks. This means:
- Predicting 4 weeks and 8 weeks produces the same ARIMA component
- The model has no way to differentiate prediction horizon within its band

**Fix:** Pass `n_weeks` to `Hybrid4WeekModel.predict()` and adjust `ARIMA4WeekModel.FORECAST_STEPS` accordingly.

### M4. Learned Blend Weight for 18-Week Model Can Be Optimized on Leaked Data

**Location:** `horizon_models.py:670-692`

`learn_blend_weight()` is called from the training pipeline with data that has already been used for model training. The `X`, `y`, and `traditional_pred` arguments should come from a held-out set that was not seen during either the deep model's training or the traditional model's training.

**Fix:** Ensure `learn_blend_weight()` is called only on test/validation data, not training data.

---

## LOW — Minor Improvements for Production Robustness

### L1. No Reproducibility Seeds in PyTorch Training

**Location:** `horizon_models.py` (global)

While `random_state=42` is set for sklearn models throughout the codebase, the PyTorch models have no equivalent seed control. `torch.manual_seed()`, `np.random.seed()`, and CUDA determinism flags are missing, meaning each training run produces different results.

**Fix:** Add at the start of `fit()`:
```python
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### L2. Save/Load Asymmetry in `DeepSeasonLongModel`

**Location:** `horizon_models.py:694-727`

When loading a saved model, `_DeepFeedforwardNet(m.n_features)` is constructed with default `hidden_units=None`, ignoring any custom architecture that may have been tuned by Optuna and saved. If the saved model used a non-default architecture (e.g., different stage widths from tuning), loading it with the wrong architecture will cause `load_state_dict()` to fail or silently produce wrong results if layer dimensions happen to match.

**Fix:** Persist `hidden_units` (or the full architecture specification) in `config.joblib` and reconstruct with the saved configuration:
```python
# In save():
joblib.dump({..., "hidden_units": self._hidden_units, "dropout": self.dropout}, path / "config.joblib")
# In load():
m.model = _DeepFeedforwardNet(m.n_features, hidden_units=cfg.get("hidden_units"), dropout=cfg.get("dropout", 0.35))
```

---

## Summary Table

| ID | Severity | Component | Issue | Impact on OOS |
|----|----------|-----------|-------|---------------|
| C1 | Critical | Both | Temporal leakage in train/val splits | Inflated val metrics, overfitted hyperparameters |
| C2 | Critical | 4W LSTM | DataLoader shuffle=False | Memorization, poor generalization |
| C3 | Critical | 4W ARIMA | Static forecasts, not per-game | 40% of hybrid output is stale |
| H1 | High | Both | No gradient clipping | Training instability from outlier games |
| H2 | High | Both | No weight decay in optimizer | Overfitting on small datasets |
| H3 | High | Both | No learning rate scheduling | Suboptimal convergence |
| H4 | High | 18W Deep | Silent feature pad/truncate | Wrong predictions from misaligned features |
| H5 | High | 4W LSTM | Players with short history dropped | Missing predictions for key players |
| H6 | High | 18W Deep | Regression to mean unused | Overweights recent streaks |
| M1 | Medium | 4W Hybrid | Fixed blend weights not validated | Suboptimal ensemble |
| M2 | Medium | 18W Deep | Limited Optuna architecture search | Potentially suboptimal network |
| M3 | Medium | 4W Hybrid | Ignores n_weeks at prediction | Same output for 4w and 8w |
| M4 | Medium | 18W Deep | Blend weight learned on train data | Overfit blend coefficient |
| L1 | Low | Both | No PyTorch reproducibility seeds | Non-deterministic training |
| L2 | Low | 18W Deep | Save/load architecture mismatch | Silent model corruption on reload |

---

## Recommended Fix Priority

1. **Immediate (C1-C3):** Fix data leakage, LSTM shuffle, ARIMA staleness — these are the primary reasons OOS accuracy would disappoint
2. **Next Sprint (H1-H6):** Add gradient clipping, weight decay, LR scheduling, feature validation, short-history handling, mean reversion
3. **Refinement (M1-M4):** Validate blend weights, expand Optuna search, horizon-aware prediction
4. **Polish (L1-L2):** Reproducibility seeds, save/load consistency

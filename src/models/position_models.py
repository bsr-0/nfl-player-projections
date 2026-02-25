"""Position-specific ML models for NFL player prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib

from sklearn.model_selection import cross_val_score, cross_val_predict, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    TPESampler = None
    HAS_OPTUNA = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODEL_CONFIG, MODELS_DIR, POSITIONS

VALIDATION_PCT = MODEL_CONFIG.get("validation_pct", 0.2)
EARLY_STOPPING_ROUNDS = MODEL_CONFIG.get("early_stopping_rounds", 25)


class TargetTransformer:
    """Log1p target transformation for right-skewed fantasy point distributions.

    Applies log1p(y + shift) during training and expm1 - shift at prediction time.
    The shift ensures all values are positive before log. Skewness is checked
    before applying: if the target is already approximately symmetric (|skew| < 0.5),
    no transformation is applied.
    """
    def __init__(self):
        self.shift = 0.0
        self.active = False  # Whether transformation is actually applied

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        skewness = float(np.nanmean(((y - np.nanmean(y)) / max(np.nanstd(y), 1e-8)) ** 3))
        if abs(skewness) < 0.5:
            self.active = False
            return y.copy()
        self.active = True
        self.shift = max(0.0, -np.nanmin(y) + 1.0)
        return np.log1p(y + self.shift)

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        if not self.active:
            return y_transformed
        return np.expm1(y_transformed) - self.shift
# Ensemble weights: 4-model when LightGBM available, else original 3-model
if HAS_LIGHTGBM:
    ENSEMBLE_WEIGHTS_1W = {"random_forest": 0.20, "xgboost": 0.30, "lightgbm": 0.25, "ridge": 0.25}
else:
    ENSEMBLE_WEIGHTS_1W = {"random_forest": 0.30, "xgboost": 0.40, "ridge": 0.30}


class SeasonAwareTimeSeriesSplit:
    """Season-aware CV splitter that never splits mid-season.

    Each fold's test set consists of complete seasons that are strictly
    after all training seasons, with an optional purge gap.
    Falls back to standard TimeSeriesSplit when season info is unavailable.

    Parameters:
        n_splits: Number of folds.
        seasons: Array of season labels per row (aligned with X).
        gap_seasons: Number of seasons to skip between train and test
            to prevent feature leakage from rolling/lag features.
    """
    def __init__(self, n_splits: int = 5, seasons: np.ndarray = None,
                 gap_seasons: int = 0):
        self.n_splits = n_splits
        self.seasons = seasons
        self.gap_seasons = gap_seasons

    def split(self, X, y=None, groups=None):
        if self.seasons is None or len(self.seasons) != len(X):
            # Fallback: standard TimeSeriesSplit
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X, y, groups)
            return
        unique_seasons = sorted(set(self.seasons))
        n_seasons = len(unique_seasons)
        # Need at least n_splits + 1 seasons (1 test per fold + remainder for train)
        if n_seasons < self.n_splits + 1:
            yield from TimeSeriesSplit(n_splits=self.n_splits).split(X, y, groups)
            return
        # Allocate test seasons: last n_splits seasons each serve as a test season
        for i in range(self.n_splits):
            test_season_idx = n_seasons - self.n_splits + i
            test_season = unique_seasons[test_season_idx]
            # Train on all seasons before test, minus the gap
            train_end_idx = test_season_idx - self.gap_seasons
            if train_end_idx <= 0:
                continue
            train_seasons = set(unique_seasons[:train_end_idx])
            train_idx = np.where(np.isin(self.seasons, list(train_seasons)))[0]
            test_idx = np.where(self.seasons == test_season)[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class _IdentityScaler:
    """Passthrough scaler for backward compatibility with models saved before scaling was added."""
    def transform(self, X):
        return X


class PositionModel:
    """
    Position-specific ML model with hyperparameter tuning.

    Uses 1-week ensemble per requirements: Random Forest (30%) + XGBoost (40%) + Ridge (30%).
    """
    def __init__(self, position: str, n_weeks: int = 1):
        self.position = position
        self.n_weeks = n_weeks
        self.models = {}
        self.best_params = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.ensemble_weights = dict(ENSEMBLE_WEIGHTS_1W)
        self.meta_learner = None
        self.feature_medians = {}  # Training medians for prediction-time imputation
        self._oof_metrics = None  # OOF evaluation metrics (honest, no data leakage)
        self.target_transformer = TargetTransformer()
        self._uncertainty_model = None  # Heteroscedastic error model
        self._uncertainty_model_type = None
        self._uncertainty_scale_factor = 1.0  # Conformal recalibration factor
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            tune_hyperparameters: bool = True,
            n_trials: int = None,
            sample_weight: Optional[np.ndarray] = None,
            seasons: Optional[np.ndarray] = None) -> 'PositionModel':
        """
        Train the position model.

        Uses time-based train/validation split. Season-aware CV splits
        ensure fold boundaries never land mid-season.

        Args:
            X: Feature DataFrame
            y: Target Series
            tune_hyperparameters: Whether to run Optuna tuning
            n_trials: Number of Optuna trials (default from config)
            sample_weight: Optional recency weights
            seasons: Optional array of season labels (same length as X)
                for season-aware CV splits. When provided, OOF stacking
                and hyperparameter tuning respect season boundaries.
        """
        n_trials = n_trials or MODEL_CONFIG["n_optuna_trials"]
        if tune_hyperparameters and not HAS_OPTUNA:
            print("Optuna not installed; skipping hyperparameter tuning and using defaults.")
            tune_hyperparameters = False

        self.feature_names = list(X.columns)
        X_raw = np.asarray(X.values, dtype=np.float64)
        y_np = np.asarray(y.values, dtype=np.float64)

        # Apply target transformation for right-skewed distributions
        self.target_transformer = TargetTransformer()
        y_np = self.target_transformer.fit_transform(y_np)
        if self.target_transformer.active:
            print(f"  Target transformation: log1p applied (skewed distribution)")

        n = len(X_raw)
        split_idx = int(n * (1 - VALIDATION_PCT))
        if split_idx < 50 or n - split_idx < 20:
            split_idx = max(50, n - 50)

        # Compute training medians for prediction-time imputation (replaces zero-fill)
        self.feature_medians = {}
        for i, col_name in enumerate(self.feature_names):
            col_vals = X_raw[:split_idx, i]
            finite_vals = col_vals[np.isfinite(col_vals)]
            self.feature_medians[col_name] = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0

        X_np = np.where(np.isfinite(X_raw), X_raw, 0)
        X_np = np.nan_to_num(X_np, nan=0.0)

        X_train_inner = X_np[:split_idx]
        y_train_inner = y_np[:split_idx]
        X_val = X_np[split_idx:]
        y_val = y_np[split_idx:]
        sw_train = (sample_weight[:split_idx].astype(np.float64) if sample_weight is not None and len(sample_weight) >= split_idx else None)
        seasons_train = seasons[:split_idx] if seasons is not None else None

        print(f"\nTraining {self.position} model for {self.n_weeks}-week prediction...")
        print(f"Training samples: {len(X_train_inner)}, Validation: {len(X_val)}, Features: {len(self.feature_names)}")

        self.scaler.fit(X_train_inner)
        X_train_scaled = self.scaler.transform(X_train_inner)
        X_val_scaled = self.scaler.transform(X_val)

        # Build season-aware CV splitter (respects season boundaries + purge gap)
        gap_seasons = MODEL_CONFIG.get("cv_gap_seasons", 1)
        n_cv_folds = MODEL_CONFIG.get("cv_folds", 5)
        cv_splitter = SeasonAwareTimeSeriesSplit(
            n_splits=n_cv_folds,
            seasons=seasons_train,
            gap_seasons=gap_seasons,
        )

        if tune_hyperparameters:
            import time as _time
            print("Running hyperparameter optimization...", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning Random Forest ({n_trials} trials)...", flush=True)
            self.best_params["random_forest"] = self._tune_random_forest(
                X_train_scaled, y_train_inner, n_trials, seasons=seasons_train
            )
            print(f"  RF tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning XGBoost ({n_trials} trials)...", flush=True)
            self.best_params["xgboost"] = self._tune_xgboost(
                X_train_scaled, y_train_inner, n_trials, seasons=seasons_train
            )
            print(f"  XGBoost tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning Ridge (RidgeCV)...", flush=True)
            self.best_params["ridge"] = self._tune_ridge(
                X_train_scaled, y_train_inner, n_trials, seasons=seasons_train
            )
            print(f"  Ridge tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
            if HAS_LIGHTGBM:
                _t0 = _time.perf_counter()
                print(f"  Tuning LightGBM ({n_trials} trials)...", flush=True)
                self.best_params["lightgbm"] = self._tune_lightgbm(
                    X_train_scaled, y_train_inner, n_trials, seasons=seasons_train
                )
                print(f"  LightGBM tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
        else:
            self.best_params = self._get_default_params()

        model_names = "RF + XGBoost + LightGBM + Ridge" if HAS_LIGHTGBM else "RF + XGBoost + Ridge"
        print(f"Training final models ({model_names}) with {n_cv_folds}-fold season-aware OOF stacking...", flush=True)

        # --- Meta-learner stacking via cross-validated OOF predictions ---
        # Uses SeasonAwareTimeSeriesSplit so fold boundaries never split mid-season,
        # with a purge gap to prevent feature leakage from rolling/lag features.
        n_base = 4 if HAS_LIGHTGBM else 3
        oof_preds = np.full((len(X_train_scaled), n_base), np.nan)
        fold_i = 0
        for train_idx, oof_idx in cv_splitter.split(X_train_scaled):
            fold_i += 1
            print(f"  OOF fold {fold_i}/{n_cv_folds}...", flush=True)
            X_tr_fold, y_tr_fold = X_train_scaled[train_idx], y_train_inner[train_idx]
            X_oof_fold = X_train_scaled[oof_idx]
            sw_fold = sw_train[train_idx] if sw_train is not None else None

            rf_fold = self._train_random_forest(X_tr_fold, y_tr_fold, sample_weight=sw_fold)
            xgb_fold = self._train_xgboost(X_tr_fold, y_tr_fold, X_oof_fold, y_train_inner[oof_idx], sample_weight=sw_fold)
            ridge_fold = self._train_ridge(X_tr_fold, y_tr_fold, sample_weight=sw_fold)

            oof_preds[oof_idx, 0] = rf_fold.predict(X_oof_fold)
            oof_preds[oof_idx, 1] = xgb_fold.predict(X_oof_fold)
            oof_preds[oof_idx, 2] = ridge_fold.predict(X_oof_fold)

            if HAS_LIGHTGBM:
                lgb_fold = self._train_lightgbm(X_tr_fold, y_tr_fold, X_oof_fold, y_train_inner[oof_idx], sample_weight=sw_fold)
                oof_preds[oof_idx, 3] = lgb_fold.predict(X_oof_fold)

        # Fit meta-learner on OOF predictions (rows with all folds filled)
        # RidgeCV tunes alpha automatically for better stacking performance
        oof_valid = ~np.isnan(oof_preds).any(axis=1)
        if oof_valid.sum() >= 20:
            from sklearn.linear_model import RidgeCV as _RidgeCV
            self.meta_learner = _RidgeCV(
                alphas=np.logspace(-2, 2, 20), cv=min(3, n_cv_folds)
            )
            self.meta_learner.fit(oof_preds[oof_valid], y_train_inner[oof_valid])
        else:
            self.meta_learner = None
            print("  Warning: insufficient OOF predictions for meta-learner, using fixed weights")

        # Compute honest OOF metrics before retraining on all data
        if oof_valid.sum() >= 20:
            if self.meta_learner is not None:
                oof_ensemble = self.meta_learner.predict(oof_preds[oof_valid])
            else:
                base_keys = list(ENSEMBLE_WEIGHTS_1W.keys())
                weights = [ENSEMBLE_WEIGHTS_1W.get(k, 1.0 / n_base) for k in base_keys[:n_base]]
                oof_ensemble = np.average(oof_preds[oof_valid], axis=1, weights=weights)
            oof_y = y_train_inner[oof_valid]
            self._oof_metrics = {
                "rmse": float(np.sqrt(mean_squared_error(oof_y, oof_ensemble))),
                "mae": float(mean_absolute_error(oof_y, oof_ensemble)),
                "r2": float(r2_score(oof_y, oof_ensemble)),
                "n_samples": int(oof_valid.sum()),
            }

            # Isotonic calibration: correct systematic biases in OOF predictions
            try:
                from sklearn.isotonic import IsotonicRegression
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(oof_ensemble, oof_y)
                cal_pred = self.calibrator.predict(oof_ensemble)
                cal_rmse = float(np.sqrt(mean_squared_error(oof_y, cal_pred)))
                if cal_rmse < self._oof_metrics["rmse"] * 1.01:
                    print(f"  Isotonic calibration: RMSE {self._oof_metrics['rmse']:.3f} -> {cal_rmse:.3f}")
                else:
                    self.calibrator = None
                    print(f"  Isotonic calibration: no improvement, disabled")
            except Exception:
                self.calibrator = None
        else:
            self._oof_metrics = None
            self.calibrator = None

        # Now train final base models on ALL training data for serving
        self.models["random_forest"] = self._train_random_forest(X_train_scaled, y_train_inner, sample_weight=sw_train)
        self.models["xgboost"] = self._train_xgboost(X_train_scaled, y_train_inner, X_val_scaled, y_val, sample_weight=sw_train)
        self.models["ridge"] = self._train_ridge(X_train_scaled, y_train_inner, sample_weight=sw_train)
        if HAS_LIGHTGBM:
            self.models["lightgbm"] = self._train_lightgbm(X_train_scaled, y_train_inner, X_val_scaled, y_val, sample_weight=sw_train)

        self._base_model_keys = list(self.models.keys())

        # Overfitting diagnostic: compare train RMSE to OOF RMSE
        if self._oof_metrics is not None:
            train_preds_stack = np.column_stack([self.models[k].predict(X_train_scaled) for k in self._base_model_keys])
            if self.meta_learner is not None:
                train_pred = self.meta_learner.predict(train_preds_stack)
            else:
                weights = [self.ensemble_weights.get(k, 1.0 / len(self._base_model_keys)) for k in self._base_model_keys]
                train_pred = np.average(train_preds_stack, axis=1, weights=weights)
            train_rmse = float(np.sqrt(mean_squared_error(y_train_inner, train_pred)))
            oof_rmse = self._oof_metrics["rmse"]
            ratio = oof_rmse / train_rmse if train_rmse > 0 else float("inf")
            self._oof_metrics["train_rmse"] = train_rmse
            self._oof_metrics["overfit_ratio"] = ratio
            if ratio > 1.3:
                print(f"  WARNING: Possible overfitting — train RMSE={train_rmse:.3f}, OOF RMSE={oof_rmse:.3f} (ratio={ratio:.2f})")
            else:
                print(f"  Overfit check OK — train RMSE={train_rmse:.3f}, OOF RMSE={oof_rmse:.3f} (ratio={ratio:.2f})")

        # Conformal calibration: compute residual distribution from OOF predictions
        # (cross-validated, not from the retrained model) for well-calibrated intervals.
        #
        # Using OOF residuals avoids the bias of evaluating the final model on data
        # it was trained on. The OOF predictions come from models trained on fold
        # subsets, so residuals reflect true out-of-sample error.
        if self.meta_learner is not None and oof_valid.sum() >= 20:
            oof_ensemble_pred = self.meta_learner.predict(oof_preds[oof_valid])
        elif oof_valid.sum() >= 20:
            oof_ensemble_pred = np.mean(oof_preds[oof_valid], axis=1)
        else:
            oof_ensemble_pred = None

        if oof_ensemble_pred is not None:
            cal_residuals = np.abs(y_train_inner[oof_valid] - oof_ensemble_pred)
            cal_pred = oof_ensemble_pred
        else:
            # Fallback: use validation set (less ideal but better than nothing)
            val_preds_stack = np.column_stack([self.models[k].predict(X_val_scaled) for k in self._base_model_keys])
            if self.meta_learner is not None:
                cal_pred = self.meta_learner.predict(val_preds_stack)
            else:
                cal_pred = np.mean(val_preds_stack, axis=1)
            cal_residuals = np.abs(y_val - cal_pred)

        self._conformal_residual_std = float(np.std(cal_residuals))

        # Constant-width quantiles (backward-compatible fallback)
        self._conformal_quantiles = {
            0.80: float(np.quantile(cal_residuals, 0.80)),
            0.90: float(np.quantile(cal_residuals, 0.90)),
            0.95: float(np.quantile(cal_residuals, 0.95)),
        }

        self._conformal_hetero = None  # Removed: heteroscedastic conformal adds fitting complexity without clear generalization benefit

        # Heteroscedastic uncertainty model: predict absolute residuals from features.
        # This provides player-specific uncertainty rather than a constant-width interval.
        try:
            if oof_ensemble_pred is not None and oof_valid.sum() >= 200:
                abs_resid = np.abs(y_train_inner[oof_valid] - oof_ensemble_pred)
                # Use a conservative GBM to avoid overfitting residuals.
                unc_model = GradientBoostingRegressor(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                )
                unc_model.fit(X_train_scaled[oof_valid], abs_resid)
                self._uncertainty_model = unc_model
                self._uncertainty_model_type = "gbr_abs_resid"
                print("  Uncertainty model: heteroscedastic GBM trained on OOF residuals")
            else:
                self._uncertainty_model = None
                self._uncertainty_model_type = None
        except Exception as e:
            print(f"  Uncertainty model skipped: {e}")
            self._uncertainty_model = None
            self._uncertainty_model_type = None

        # Post-hoc conformal recalibration of uncertainty estimates.
        # The blended uncertainty from predict_with_uncertainty is typically
        # too narrow (e.g., 73% actual coverage at 90% nominal). We compute
        # a single scalar correction factor from OOF residuals so that
        # nominal coverage matches actual coverage.
        self._uncertainty_scale_factor = 1.0
        if oof_ensemble_pred is not None and oof_valid.sum() >= 50:
            from scipy.stats import norm as _norm_dist

            # Compute blended uncertainty on OOF data (same logic as predict_with_uncertainty)
            oof_ensemble_std = np.std(oof_preds[oof_valid], axis=1)
            conformal_std_fill = np.full_like(oof_ensemble_std, self._conformal_residual_std)

            hetero_std_oof = None
            if self._uncertainty_model is not None:
                try:
                    hetero_std_oof = self._uncertainty_model.predict(X_train_scaled[oof_valid])
                    hetero_std_oof = np.clip(hetero_std_oof, 0.1, None)
                except Exception:
                    hetero_std_oof = None

            if hetero_std_oof is not None:
                blended_std = 0.5 * hetero_std_oof + 0.3 * conformal_std_fill + 0.2 * oof_ensemble_std
            else:
                blended_std = 0.7 * conformal_std_fill + 0.3 * oof_ensemble_std

            # Standardized absolute residuals
            oof_abs_resid = np.abs(y_train_inner[oof_valid] - oof_ensemble_pred)
            standardized = oof_abs_resid / np.maximum(blended_std, 1e-8)

            # Correction factor: make 90% nominal coverage actually achieve ~90%.
            # For a N(0,1) distribution, P(|Z| < 1.645) = 0.90.
            # If our standardized residuals have q90 != 1.645, scale accordingly.
            z_90 = _norm_dist.ppf(0.95)  # 1.6449
            q_90 = float(np.quantile(standardized, 0.90))
            correction = q_90 / z_90 if z_90 > 0 else 1.0

            # Bound to avoid pathological cases
            correction = float(np.clip(correction, 0.5, 5.0))
            self._uncertainty_scale_factor = correction

            # Verify calibration on OOF data
            raw_coverage = float((oof_abs_resid <= z_90 * blended_std).mean())
            cal_coverage = float((oof_abs_resid <= z_90 * blended_std * correction).mean())
            print(f"  Uncertainty recalibration: factor={correction:.3f}, "
                  f"raw 90% coverage={raw_coverage:.1%} -> calibrated={cal_coverage:.1%}")

        self.is_fitted = True
        print(f"Model training complete. Meta-learner stacking {'enabled' if self.meta_learner else 'disabled (fixed weights)'}.")
        print(f"  Conformal calibration: residual_std={self._conformal_residual_std:.3f}, "
              f"q80={self._conformal_quantiles[0.80]:.3f}, q95={self._conformal_quantiles[0.95]:.3f}")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions (inverse-transformed if target transform was applied)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_np = self._prepare_input(X)
        keys = getattr(self, "_base_model_keys", list(self.models.keys()))
        preds_list = [self.models[k].predict(X_np) for k in keys]
        if self.meta_learner is not None:
            raw_pred = self.meta_learner.predict(np.column_stack(preds_list))
        else:
            weights = [self.ensemble_weights.get(k, 1.0 / len(keys)) for k in keys]
            raw_pred = np.average(np.column_stack(preds_list), axis=1, weights=weights)
        # Apply isotonic calibration if available (before inverse transform)
        cal = getattr(self, "calibrator", None)
        if cal is not None:
            raw_pred = cal.predict(raw_pred)
        # Inverse-transform predictions back to original scale
        tt = getattr(self, "target_transformer", None)
        if tt is not None:
            return tt.inverse_transform(raw_pred)
        return raw_pred
    
    def _prepare_input(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and scale input for prediction using training medians for imputation."""
        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X.reindex(columns=self.feature_names)
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        medians = getattr(self, 'feature_medians', {})
        if medians:
            X_clean = X_clean.fillna(medians)
        X_clean = X_clean.fillna(0)  # fallback for any remaining NaN
        X_np = np.asarray(X_clean.values, dtype=np.float64)
        X_np = np.where(np.isfinite(X_np), X_np, 0)
        X_np = np.nan_to_num(X_np, nan=0.0)
        return self.scaler.transform(X_np)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.

        Uses constant-width conformal residual std blended with ensemble
        disagreement for calibrated uncertainty.

        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_np = self._prepare_input(X)

        keys = getattr(self, "_base_model_keys", list(self.models.keys()))
        preds = np.array([self.models[k].predict(X_np) for k in keys])
        if self.meta_learner is not None:
            mean_pred = self.meta_learner.predict(np.column_stack(preds))
        else:
            weights = [self.ensemble_weights.get(k, 1.0 / len(keys)) for k in keys]
            mean_pred = np.average(preds, axis=0, weights=weights)

        # Inverse-transform mean prediction
        tt = getattr(self, "target_transformer", None)
        if tt is not None:
            mean_pred = tt.inverse_transform(mean_pred)

        ensemble_std = np.std(preds, axis=0)
        conformal_std = getattr(self, "_conformal_residual_std", None)
        conformal_std_arr = None
        if conformal_std is not None and np.isfinite(conformal_std) and conformal_std > 0:
            conformal_std_arr = np.full_like(ensemble_std, float(conformal_std))

        # Optional heteroscedastic uncertainty model (trained on OOF residuals)
        hetero_std = None
        unc_model = getattr(self, "_uncertainty_model", None)
        if unc_model is not None:
            try:
                hetero_std = unc_model.predict(X_np)
                hetero_std = np.clip(hetero_std, 0.1, None)
            except Exception:
                hetero_std = None

        # If target was log-transformed, scale std by derivative of expm1
        if tt is not None and tt.active:
            raw_mean = mean_pred + tt.shift
            scale = np.maximum(raw_mean, 1.0)
            ensemble_std = ensemble_std * scale
            if conformal_std_arr is not None:
                conformal_std_arr = conformal_std_arr * scale
            if hetero_std is not None:
                hetero_std = hetero_std * scale

        # Blend uncertainties: heteroscedastic (if available) + conformal + ensemble spread
        if hetero_std is not None and conformal_std_arr is not None:
            std_pred = 0.5 * hetero_std + 0.3 * conformal_std_arr + 0.2 * ensemble_std
        elif hetero_std is not None:
            std_pred = 0.7 * hetero_std + 0.3 * ensemble_std
        elif conformal_std_arr is not None:
            std_pred = 0.7 * conformal_std_arr + 0.3 * ensemble_std
        else:
            std_pred = ensemble_std

        # Apply conformal recalibration factor (computed during fit from OOF data)
        # to correct systematic under/over-coverage of confidence intervals.
        scale_factor = getattr(self, "_uncertainty_scale_factor", 1.0)
        if scale_factor != 1.0:
            std_pred = std_pred * scale_factor

        return mean_pred, std_pred

    def predict_distributional(self, X: pd.DataFrame,
                               boom_threshold: float = 20.0,
                               bust_threshold: float = 5.0) -> Dict[str, np.ndarray]:
        """Predict point estimates plus boom/bust probabilities.

        Uses the Gaussian assumption from ``predict_with_uncertainty`` to
        estimate the probability of exceeding ``boom_threshold`` (boom) and
        falling below ``bust_threshold`` (bust).

        Returns:
            Dict with keys: 'prediction', 'std', 'boom_prob', 'bust_prob'.
        """
        from scipy.stats import norm as _norm

        mean_pred, std_pred = self.predict_with_uncertainty(X)
        # Ensure positive std
        std_safe = np.maximum(std_pred, 1e-6)

        boom_prob = 1.0 - _norm.cdf(boom_threshold, loc=mean_pred, scale=std_safe)
        bust_prob = _norm.cdf(bust_threshold, loc=mean_pred, scale=std_safe)

        return {
            "prediction": mean_pred,
            "std": std_pred,
            "boom_prob": np.clip(boom_prob, 0, 1),
            "bust_prob": np.clip(bust_prob, 0, 1),
        }

    @staticmethod
    def _subsample_for_tuning(X: np.ndarray, y: np.ndarray,
                              max_samples: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data for hyperparameter tuning to keep trial times reasonable.

        Uses stratified temporal sampling: keeps ALL recent data (last 60%)
        and randomly samples from older data to fill the remainder, so tuning
        reflects current NFL patterns while still seeing older-era variation.
        This avoids double-stacking recency bias (recency weights + tail-only
        subsampling) that would over-specialize hyperparameters to a narrow
        temporal window.
        """
        if len(X) <= max_samples:
            return X, y
        # Keep last 60% of max_samples as the recent block
        recent_count = int(max_samples * 0.6)
        older_count = max_samples - recent_count
        X_recent, y_recent = X[-recent_count:], y[-recent_count:]
        # Sample from older data
        X_old, y_old = X[:-recent_count], y[:-recent_count]
        if len(X_old) > older_count:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_old), size=older_count, replace=False)
            idx.sort()  # maintain temporal order
            X_old, y_old = X_old[idx], y_old[idx]
        return np.vstack([X_old, X_recent]), np.concatenate([y_old, y_recent])

    def _tune_random_forest(self, X: np.ndarray, y: np.ndarray, n_trials: int,
                            seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune Random Forest hyperparameters with widened search space.

        Tunes n_estimators jointly with structural params so that the
        interaction between tree count and depth/leaf settings is captured.
        """
        if not HAS_OPTUNA:
            return self._get_default_params().get("random_forest", {})
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        seasons_tune = seasons[-len(X_tune):] if seasons is not None and len(seasons) >= len(X_tune) else None
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        gap = MODEL_CONFIG.get("cv_gap_seasons", 1)
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
                "random_state": MODEL_CONFIG["random_state"],
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
            cv = SeasonAwareTimeSeriesSplit(n_splits=tune_folds, seasons=seasons_tune, gap_seasons=gap)
            scores = cross_val_score(model, X_tune, y_tune, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
            return -scores.mean()
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 60), show_progress_bar=True)
        return study.best_params

    def _tune_xgboost(self, X: np.ndarray, y: np.ndarray, n_trials: int,
                      seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune XGBoost hyperparameters with reduced search space.

        Uses Pseudo-Huber loss (reg:pseudohubererror) for robustness to outlier games.
        Fixes subsample and gamma to reduce the search from 9D to 7D,
        allowing the TPE sampler to converge with the available trial budget.
        """
        if not HAS_OPTUNA:
            return self._get_default_params().get("xgboost", {})
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        seasons_tune = seasons[-len(X_tune):] if seasons is not None and len(seasons) >= len(X_tune) else None
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        gap = MODEL_CONFIG.get("cv_gap_seasons", 1)
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "subsample": 0.8,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "gamma": 0.0,
                "objective": "reg:pseudohubererror",
                "huber_slope": 1.0,
                "random_state": MODEL_CONFIG["random_state"],
                "tree_method": "hist",
                "n_jobs": 1,
            }
            cv = SeasonAwareTimeSeriesSplit(n_splits=tune_folds, seasons=seasons_tune, gap_seasons=gap)
            # Avoid sklearn get_tags() path that breaks with xgboost's estimator MRO
            # under sklearn>=1.6 by running manual CV.
            fold_mse = []
            for train_idx, test_idx in cv.split(X_tune, y_tune):
                model = xgb.XGBRegressor(**params)
                X_train, y_train = X_tune[train_idx], y_tune[train_idx]
                X_test, y_test = X_tune[test_idx], y_tune[test_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                fold_mse.append(mean_squared_error(y_test, preds))
            if not fold_mse:
                return float("inf")
            return float(np.mean(fold_mse))

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 100), show_progress_bar=True)
        return study.best_params

    def _tune_ridge(self, X: np.ndarray, y: np.ndarray, n_trials: int,
                    seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune Ridge alpha using RidgeCV (analytically exact for 1D)."""
        if not HAS_OPTUNA:
            return self._get_default_params().get("ridge", {})
        from sklearn.linear_model import RidgeCV as _RidgeCV
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        seasons_tune = seasons[-len(X_tune):] if seasons is not None and len(seasons) >= len(X_tune) else None
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        gap = MODEL_CONFIG.get("cv_gap_seasons", 1)
        cv_splitter = SeasonAwareTimeSeriesSplit(
            n_splits=tune_folds, seasons=seasons_tune, gap_seasons=gap
        )
        alphas = np.logspace(-2, 2, 50)
        ridge_cv = _RidgeCV(alphas=alphas, cv=cv_splitter, scoring="neg_mean_squared_error")
        ridge_cv.fit(X_tune, y_tune)
        return {"alpha": float(ridge_cv.alpha_)}

    def _tune_lightgbm(self, X: np.ndarray, y: np.ndarray, n_trials: int,
                        seasons: Optional[np.ndarray] = None) -> Dict:
        """Tune LightGBM hyperparameters with Huber loss for outlier robustness."""
        if not HAS_OPTUNA:
            return self._get_default_params().get("lightgbm", {})
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        seasons_tune = seasons[-len(X_tune):] if seasons is not None and len(seasons) >= len(X_tune) else None
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        gap = MODEL_CONFIG.get("cv_gap_seasons", 1)
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "subsample": 0.8,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "objective": "huber",
                "alpha": 1.0,
                "random_state": MODEL_CONFIG["random_state"],
                "verbosity": -1,
                "n_jobs": 1,
            }
            # Avoid sklearn get_tags() path that breaks with third-party
            # estimator MROs under sklearn>=1.6 by running manual CV
            # (same fix as _tune_xgboost).
            cv = SeasonAwareTimeSeriesSplit(n_splits=tune_folds, seasons=seasons_tune, gap_seasons=gap)
            fold_mse = []
            for train_idx, test_idx in cv.split(X_tune, y_tune):
                model = lgb.LGBMRegressor(**params)
                X_train, y_train = X_tune[train_idx], y_tune[train_idx]
                X_test, y_test = X_tune[test_idx], y_tune[test_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                fold_mse.append(mean_squared_error(y_test, preds))
            if not fold_mse:
                return float("inf")
            return float(np.mean(fold_mse))
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 80), show_progress_bar=True)
        return study.best_params

    def _get_default_params(self) -> Dict[str, Dict]:
        """Get default hyperparameters (requirements-aligned)."""
        defaults = {
            "random_forest": {
                "n_estimators": 750,
                "max_depth": 12,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": MODEL_CONFIG["random_state"],
            },
            "xgboost": {
                "n_estimators": 750,
                "max_depth": 7,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "gamma": 0.0,
                "objective": "reg:pseudohubererror",
                "huber_slope": 1.0,
                "random_state": MODEL_CONFIG["random_state"],
                "tree_method": "hist",
                "n_jobs": 1,
            },
            "ridge": {
                "alpha": 5.0,
                "random_state": MODEL_CONFIG["random_state"],
            },
        }
        if HAS_LIGHTGBM:
            defaults["lightgbm"] = {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "objective": "huber",
                "alpha": 1.0,
                "random_state": MODEL_CONFIG["random_state"],
                "verbosity": -1,
                "n_jobs": 1,
            }
        return defaults
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       sample_weight: Optional[np.ndarray] = None) -> xgb.XGBRegressor:
        """Train XGBoost with best params, Huber loss, and early stopping."""
        params = self.best_params.get("xgboost", self._get_default_params()["xgboost"]).copy()
        params["random_state"] = MODEL_CONFIG["random_state"]
        params["tree_method"] = "hist"  # Much faster; equivalent quality
        params["n_jobs"] = 1  # Avoid macOS fork deadlock with sequential position training
        params["objective"] = "reg:pseudohubererror"
        params["huber_slope"] = 1.0
        # XGBoost 3.x moved eval_metric from fit() to constructor
        params["eval_metric"] = "rmse"
        model = xgb.XGBRegressor(**params)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None and len(X_val) >= 10:
            # XGBoost 2.0+: fit() uses callbacks; <2.0 uses early_stopping_rounds (callbacks not supported)
            xgb_version = getattr(xgb, "__version__", "0.0.0")
            use_callbacks = xgb_version >= "2.0.0"
            if use_callbacks:
                try:
                    early_stop = xgb.callback.EarlyStopping(
                        rounds=EARLY_STOPPING_ROUNDS, metric_name="rmse", save_best=True
                    )
                    kw["callbacks"] = [early_stop]
                except (AttributeError, TypeError):
                    use_callbacks = False
            fit_kw = dict(eval_set=[(X_val, y_val)], verbose=False, **kw)
            if not use_callbacks:
                fit_kw["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
            try:
                model.fit(X, y, **fit_kw)
            except TypeError as e:
                err_msg = str(e).lower()
                fit_kw.pop("callbacks", None)
                fit_kw.pop("early_stopping_rounds", None)
                if "callbacks" in err_msg:
                    try:
                        model.fit(X, y, early_stopping_rounds=EARLY_STOPPING_ROUNDS, **fit_kw)
                    except TypeError:
                        model.fit(X, y, **fit_kw)
                elif "early_stopping_rounds" in err_msg:
                    model.fit(X, y, **fit_kw)
                else:
                    raise
        else:
            model.fit(X, y, **kw)
        return model

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray,
                             sample_weight: Optional[np.ndarray] = None) -> RandomForestRegressor:
        """Train Random Forest with OOB early stopping.

        Uses warm_start to incrementally add trees and monitors OOB R²
        to stop when performance plateaus, preventing overfitting.
        """
        params = self.best_params.get("random_forest", self._get_default_params()["random_forest"]).copy()
        target_n = params.pop("n_estimators", 500)
        params["random_state"] = MODEL_CONFIG["random_state"]
        params["n_jobs"] = 1
        params["warm_start"] = True
        params["oob_score"] = True

        step = 50
        model = RandomForestRegressor(n_estimators=step, **params)
        sw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X, y, **sw)

        best_oob = model.oob_score_
        patience = 3
        no_improve = 0

        while model.n_estimators < target_n:
            model.n_estimators = min(model.n_estimators + step, target_n)
            model.fit(X, y, **sw)
            if model.oob_score_ > best_oob:
                best_oob = model.oob_score_
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break

        model.warm_start = False
        return model

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray,
                         X_val: np.ndarray = None, y_val: np.ndarray = None,
                         sample_weight: Optional[np.ndarray] = None):
        """Train LightGBM with best params, Huber loss, and early stopping."""
        params = self.best_params.get("lightgbm", self._get_default_params().get("lightgbm", {})).copy()
        params["random_state"] = MODEL_CONFIG["random_state"]
        params["verbosity"] = -1
        params["n_jobs"] = 1
        params["objective"] = "huber"
        params["alpha"] = 1.0
        model = lgb.LGBMRegressor(**params)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None and len(X_val) >= 10:
            callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
            model.fit(X, y, eval_set=[(X_val, y_val)], callbacks=callbacks, **kw)
        else:
            model.fit(X, y, **kw)
        return model

    def _train_ridge(self, X: np.ndarray, y: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None) -> Ridge:
        """Train Ridge regression with best params (X should already be scaled)."""
        params = self.best_params.get("ridge", self._get_default_params()["ridge"])
        # Ridge does not accept random_state; remove if present
        params.pop("random_state", None)
        model = Ridge(**params)
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        return model
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree models (RF + XGBoost, or legacy XGB+LGB)."""
        importances = pd.DataFrame({"feature": self.feature_names})
        for k in self.models:
            if hasattr(self.models[k], "feature_importances_"):
                importances[k] = self.models[k].feature_importances_
        w = self.ensemble_weights
        importances["combined"] = 0.0
        for k in importances.columns:
            if k not in ("feature", "combined") and k in w:
                importances["combined"] = importances["combined"] + w[k] * importances[k]
        return importances.sort_values("combined", ascending=False)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        
        return {
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "mae": mean_absolute_error(y, predictions),
            "r2": r2_score(y, predictions),
        }
    
    def save(self, filepath: Path = None):
        """Save model to disk."""
        filepath = filepath or MODELS_DIR / f"model_{self.position.lower()}_{self.n_weeks}w.joblib"
        
        model_data = {
            "position": self.position,
            "n_weeks": self.n_weeks,
            "models": self.models,
            "best_params": self.best_params,
            "feature_names": self.feature_names,
            "ensemble_weights": self.ensemble_weights,
            "scaler": self.scaler,
            "meta_learner": self.meta_learner,
            "_base_model_keys": getattr(self, "_base_model_keys", list(self.models.keys())),
            "_conformal_residual_std": getattr(self, "_conformal_residual_std", None),
            "_conformal_quantiles": getattr(self, "_conformal_quantiles", None),
            "_conformal_hetero": getattr(self, "_conformal_hetero", None),
            "_oof_metrics": getattr(self, "_oof_metrics", None),
            "feature_medians": getattr(self, "feature_medians", {}),
            "target_transformer": getattr(self, "target_transformer", None),
            "calibrator": getattr(self, "calibrator", None),
            "_uncertainty_model": getattr(self, "_uncertainty_model", None),
            "_uncertainty_model_type": getattr(self, "_uncertainty_model_type", None),
        }
        
        joblib.dump(model_data, filepath)
        print(f"Saved {self.position} model to {filepath}")
    
    @classmethod
    def load(cls, position: str, n_weeks: int = 1, 
             filepath: Path = None) -> 'PositionModel':
        """Load model from disk."""
        filepath = filepath or MODELS_DIR / f"model_{position.lower()}_{n_weeks}w.joblib"
        
        model_data = joblib.load(filepath)
        
        model = cls(position=model_data["position"], n_weeks=model_data["n_weeks"])
        model.models = model_data["models"]
        model.best_params = model_data["best_params"]
        model.feature_names = model_data["feature_names"]
        model.ensemble_weights = model_data["ensemble_weights"]
        model.scaler = model_data.get("scaler") or _IdentityScaler()
        model.meta_learner = model_data.get("meta_learner")
        model._base_model_keys = model_data.get("_base_model_keys", list(model.models.keys()))
        model._conformal_residual_std = model_data.get("_conformal_residual_std")
        model._conformal_quantiles = model_data.get("_conformal_quantiles")
        model._conformal_hetero = model_data.get("_conformal_hetero")
        model._oof_metrics = model_data.get("_oof_metrics")
        model.feature_medians = model_data.get("feature_medians", {})
        model.target_transformer = model_data.get("target_transformer") or TargetTransformer()
        model.calibrator = model_data.get("calibrator")
        model._uncertainty_model = model_data.get("_uncertainty_model")
        model._uncertainty_model_type = model_data.get("_uncertainty_model_type")
        model.is_fitted = True
        return model


class MultiWeekModel:
    """
    Model that handles predictions for variable week horizons (1-18 weeks).
    
    Uses separate models for different prediction horizons:
    - Short-term (1-3 weeks): More weight on recent performance
    - Medium-term (4-8 weeks): Balanced approach
    - Long-term (9-18 weeks): More weight on season averages and utilization
    """
    
    def __init__(self, position: str):
        self.position = position
        self.models = {}  # {n_weeks: PositionModel}
        self.horizon_groups = {
            "short": [1, 2, 3],
            "medium": [4, 5, 6, 7, 8],
            "long": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        }
    
    def fit(self, X: pd.DataFrame, y_dict: Dict[int, pd.Series],
            tune_hyperparameters: bool = True,
            sample_weight: Optional[np.ndarray] = None,
            seasons: Optional[np.ndarray] = None) -> 'MultiWeekModel':
        """
        Train models for different prediction horizons.

        Args:
            X: Feature DataFrame
            y_dict: Dict mapping n_weeks to target Series
            tune_hyperparameters: Whether to tune hyperparameters
            sample_weight: Optional recency weights (e.g. by season)
            seasons: Optional season labels for season-aware CV splits
        """
        # Train representative models for each horizon group.
        # Prefer 1/4/18 exact horizons (requirements), but gracefully fall back
        # to nearest available targets when a specific horizon column is absent.
        available = sorted(int(k) for k in y_dict.keys())

        def _pick_available(preferred: int, candidates: List[int]) -> Optional[int]:
            subset = [w for w in available if w in candidates]
            if not subset:
                return None
            return min(subset, key=lambda w: abs(w - preferred))

        def _horizon_recency_weights(n_weeks: int) -> Optional[np.ndarray]:
            """Compute horizon-aware recency weights if seasons are provided."""
            if seasons is None or len(seasons) == 0:
                return sample_weight
            base_halflife = MODEL_CONFIG.get("recency_decay_halflife")
            horizon_map = MODEL_CONFIG.get("horizon_recency_halflife", {})
            halflife = horizon_map.get(n_weeks, base_halflife)
            if not halflife:
                return sample_weight
            seasons_arr = np.asarray(seasons, dtype=float)
            max_season = np.nanmax(seasons_arr)
            min_season = np.nanmin(seasons_arr)
            if not np.isfinite(max_season) or max_season <= min_season:
                return sample_weight
            decay = np.power(0.5, (max_season - seasons_arr) / float(halflife))
            w = decay / decay.max()
            if sample_weight is not None and len(sample_weight) == len(w):
                # Combine provided weights with horizon-specific decay.
                w = w * sample_weight
                w = w / np.nanmax(w) if np.nanmax(w) > 0 else w
            return w

        representative_weeks = {
            "short": _pick_available(1, self.horizon_groups["short"] + [1]),
            "medium": _pick_available(4, self.horizon_groups["medium"] + [4]),
            "long": _pick_available(18, self.horizon_groups["long"] + [18]),
        }

        for horizon, n_weeks in representative_weeks.items():
            if n_weeks is None:
                continue
            print(f"\nTraining {horizon}-term model ({n_weeks} weeks)...")

            model = PositionModel(self.position, n_weeks=n_weeks)
            sw = _horizon_recency_weights(n_weeks)
            model.fit(X, y_dict[n_weeks], tune_hyperparameters=tune_hyperparameters,
                      sample_weight=sw, seasons=seasons)

            # Use this model for all weeks in the horizon group
            for week in self.horizon_groups[horizon]:
                self.models[week] = model

        return self
    
    def predict(self, X: pd.DataFrame, n_weeks: int) -> np.ndarray:
        """Make predictions for specified number of weeks."""
        if n_weeks not in self.models:
            # Find closest available model
            available = list(self.models.keys())
            closest = min(available, key=lambda x: abs(x - n_weeks))
            model = self.models[closest]
        else:
            model = self.models[n_weeks]

        # Do not apply naive proportional scaling between horizons.
        # Representative models (1w/4w/18w) are trained on horizon-specific
        # targets (e.g., utilization mean vs fantasy-point sums), so linear
        # week-ratio scaling introduces bias and semantic mismatch.
        return model.predict(X)
    
    def save(self, filepath: Path = None):
        """Save all models."""
        filepath = filepath or MODELS_DIR / f"multiweek_{self.position.lower()}.joblib"
        
        # Save unique models only
        unique_models = {}
        for n_weeks, model in self.models.items():
            model_id = id(model)
            if model_id not in unique_models:
                unique_models[model_id] = {
                    "model": model,
                    "weeks": [n_weeks],
                }
            else:
                unique_models[model_id]["weeks"].append(n_weeks)
        
        from datetime import datetime
        save_data = {
            "position": self.position,
            "unique_models": unique_models,
            "horizon_groups": self.horizon_groups,
            "version_metadata": {
                "saved_at": datetime.now().isoformat(),
                "n_models": len(unique_models),
                "horizons": list(self.models.keys()),
            },
        }
        
        joblib.dump(save_data, filepath)
    
    @classmethod
    def load(cls, position: str, filepath: Path = None) -> 'MultiWeekModel':
        """Load models from disk."""
        filepath = filepath or MODELS_DIR / f"multiweek_{position.lower()}.joblib"
        
        save_data = joblib.load(filepath)
        
        multi_model = cls(position=save_data["position"])
        multi_model.horizon_groups = save_data["horizon_groups"]
        
        # Reconstruct models dict
        for model_data in save_data["unique_models"].values():
            for week in model_data["weeks"]:
                multi_model.models[week] = model_data["model"]
        
        return multi_model

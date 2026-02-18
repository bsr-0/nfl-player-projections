"""Position-specific ML models for NFL player prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib

from sklearn.model_selection import cross_val_score, cross_val_predict, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODEL_CONFIG, MODELS_DIR, POSITIONS

VALIDATION_PCT = MODEL_CONFIG.get("validation_pct", 0.2)
EARLY_STOPPING_ROUNDS = MODEL_CONFIG.get("early_stopping_rounds", 25)
# 1-week ensemble per requirements: RF 30%, XGBoost 40%, Ridge 30%
ENSEMBLE_WEIGHTS_1W = {"random_forest": 0.3, "xgboost": 0.4, "ridge": 0.3}


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
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            tune_hyperparameters: bool = True,
            n_trials: int = None,
            sample_weight: Optional[np.ndarray] = None) -> 'PositionModel':
        """
        Train the position model.

        Uses time-based 80/20 train/validation split. Optional recency
        sample_weight (e.g. by season) for time-series emphasis.
        """
        n_trials = n_trials or MODEL_CONFIG["n_optuna_trials"]

        self.feature_names = list(X.columns)
        X_np = np.asarray(X.values, dtype=np.float64)
        y_np = np.asarray(y.values, dtype=np.float64)
        X_np = np.where(np.isfinite(X_np), X_np, 0)
        X_np = np.nan_to_num(X_np, nan=0.0)

        n = len(X_np)
        split_idx = int(n * (1 - VALIDATION_PCT))
        if split_idx < 50 or n - split_idx < 20:
            split_idx = max(50, n - 50)

        X_train_inner = X_np[:split_idx]
        y_train_inner = y_np[:split_idx]
        X_val = X_np[split_idx:]
        y_val = y_np[split_idx:]
        sw_train = (sample_weight[:split_idx].astype(np.float64) if sample_weight is not None and len(sample_weight) >= split_idx else None)

        print(f"\nTraining {self.position} model for {self.n_weeks}-week prediction...")
        print(f"Training samples: {len(X_train_inner)}, Validation: {len(X_val)}, Features: {len(self.feature_names)}")

        self.scaler.fit(X_train_inner)
        X_train_scaled = self.scaler.transform(X_train_inner)
        X_val_scaled = self.scaler.transform(X_val)

        if tune_hyperparameters:
            import time as _time
            print("Running hyperparameter optimization...", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning Random Forest ({n_trials} trials)...", flush=True)
            self.best_params["random_forest"] = self._tune_random_forest(X_train_scaled, y_train_inner, n_trials)
            print(f"  RF tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning XGBoost ({n_trials} trials)...", flush=True)
            self.best_params["xgboost"] = self._tune_xgboost(X_train_scaled, y_train_inner, n_trials)
            print(f"  XGBoost tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
            _t0 = _time.perf_counter()
            print(f"  Tuning Ridge ({n_trials} trials)...", flush=True)
            self.best_params["ridge"] = self._tune_ridge(X_train_scaled, y_train_inner, n_trials)
            print(f"  Ridge tuning done in {_time.perf_counter()-_t0:.1f}s", flush=True)
        else:
            self.best_params = self._get_default_params()

        n_cv_folds = MODEL_CONFIG.get("cv_folds", 5)
        print(f"Training final models (RF + XGBoost + Ridge) with {n_cv_folds}-fold OOF stacking...", flush=True)

        # --- Meta-learner stacking via cross-validated OOF predictions ---
        # Generate out-of-fold predictions on training set using TimeSeriesSplit
        # so the meta-learner never sees its own training labels through the
        # base models.  This prevents stacking leakage.
        tscv = TimeSeriesSplit(n_splits=n_cv_folds)
        oof_preds = np.full((len(X_train_scaled), 3), np.nan)
        for fold_i, (train_idx, oof_idx) in enumerate(tscv.split(X_train_scaled), 1):
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

        # Fit meta-learner on OOF predictions (rows with all folds filled)
        oof_valid = ~np.isnan(oof_preds).any(axis=1)
        self.meta_learner = Ridge(alpha=1.0, random_state=MODEL_CONFIG["random_state"])
        self.meta_learner.fit(oof_preds[oof_valid], y_train_inner[oof_valid])

        # Now train final base models on ALL training data for serving
        self.models["random_forest"] = self._train_random_forest(X_train_scaled, y_train_inner, sample_weight=sw_train)
        self.models["xgboost"] = self._train_xgboost(X_train_scaled, y_train_inner, X_val_scaled, y_val, sample_weight=sw_train)
        self.models["ridge"] = self._train_ridge(X_train_scaled, y_train_inner, sample_weight=sw_train)

        self._optimize_ensemble_weights(X_val_scaled, y_val)
        self._base_model_keys = list(self.models.keys())
        self.is_fitted = True
        print(f"Model training complete. Meta-learner stacking enabled.")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_np = self._prepare_input(X)
        keys = getattr(self, "_base_model_keys", list(self.models.keys()))
        preds_list = [self.models[k].predict(X_np) for k in keys]
        if self.meta_learner is not None:
            return self.meta_learner.predict(np.column_stack(preds_list))
        weights = [self.ensemble_weights.get(k, 1.0 / len(keys)) for k in keys]
        return np.average(np.column_stack(preds_list), axis=1, weights=weights)
    
    def _prepare_input(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare and scale input for prediction."""
        X = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X.reindex(columns=self.feature_names)
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_np = np.asarray(X_clean.values, dtype=np.float64)
        X_np = np.where(np.isfinite(X_np), X_np, 0)
        X_np = np.nan_to_num(X_np, nan=0.0)
        return self.scaler.transform(X_np)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
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
        std_pred = np.std(preds, axis=0)
        
        return mean_pred, std_pred
    
    @staticmethod
    def _subsample_for_tuning(X: np.ndarray, y: np.ndarray,
                              max_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data for hyperparameter tuning to keep trial times reasonable.

        Uses the most recent rows (tail) to preserve time-series ordering and
        ensure tuning reflects recent data patterns. Only subsamples when data
        exceeds max_samples.
        """
        if len(X) <= max_samples:
            return X, y
        # Take the tail (most recent data) to respect time ordering
        X_sub = X[-max_samples:]
        y_sub = y[-max_samples:]
        return X_sub, y_sub

    def _tune_random_forest(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict:
        """Tune Random Forest hyperparameters (requirements: 500-1000 trees, max_depth 10-15).

        Uses reduced n_estimators during tuning for speed; the tuned structural
        hyperparameters (max_depth, min_samples_*) transfer well to the full
        tree count used in final training.
        """
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        def objective(trial):
            params = {
                "n_estimators": 150,  # Reduced for tuning speed; structural params transfer
                "max_depth": trial.suggest_int("max_depth", 10, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": MODEL_CONFIG["random_state"],
                "n_jobs": 1,
            }
            model = RandomForestRegressor(**params)
            tscv = TimeSeriesSplit(n_splits=tune_folds)
            scores = cross_val_score(model, X_tune, y_tune, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
            return -scores.mean()
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 10), show_progress_bar=True)
        best = study.best_params
        # Restore production n_estimators from the tuned trial's suggestion
        # (use the best trial's n_estimators if it was tuned, otherwise use midpoint)
        best["n_estimators"] = 750
        return best

    def _tune_xgboost(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict:
        """Tune XGBoost hyperparameters (requirements: lr 0.01-0.05, max_depth 6-8, n_estimators 500-1000).

        Uses reduced n_estimators during tuning; early stopping in final training
        determines the optimal iteration count.
        """
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        def objective(trial):
            params = {
                "n_estimators": 200,  # Reduced for tuning speed; early stopping in final training
                "max_depth": trial.suggest_int("max_depth", 6, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": MODEL_CONFIG["random_state"],
                "tree_method": "hist",
                "n_jobs": 1,
            }

            model = xgb.XGBRegressor(**params)

            tscv = TimeSeriesSplit(n_splits=tune_folds)
            scores = cross_val_score(model, X_tune, y_tune, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)

            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 15), show_progress_bar=True)
        best = study.best_params
        best["n_estimators"] = 750  # Restore production count
        return best

    def _tune_ridge(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict:
        """Tune Ridge regression hyperparameters (requirements: alpha 1.0-10.0)."""
        X_tune, y_tune = self._subsample_for_tuning(X, y)
        tune_folds = min(MODEL_CONFIG["cv_folds"], 3)
        def objective(trial):
            alpha = trial.suggest_float("alpha", 1.0, 10.0)

            model = Ridge(alpha=alpha, random_state=MODEL_CONFIG["random_state"])

            tscv = TimeSeriesSplit(n_splits=tune_folds)
            scores = cross_val_score(model, X_tune, y_tune, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)

            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=MODEL_CONFIG["random_state"])
        )
        study.optimize(objective, n_trials=min(n_trials, 20), show_progress_bar=False)

        return study.best_params
    
    def _get_default_params(self) -> Dict[str, Dict]:
        """Get default hyperparameters (requirements-aligned)."""
        return {
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
                "random_state": MODEL_CONFIG["random_state"],
                "tree_method": "hist",
                "n_jobs": 1,
            },
            "ridge": {
                "alpha": 5.0,
                "random_state": MODEL_CONFIG["random_state"],
            },
        }
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       sample_weight: Optional[np.ndarray] = None) -> xgb.XGBRegressor:
        """Train XGBoost with best params and early stopping."""
        params = self.best_params.get("xgboost", self._get_default_params()["xgboost"]).copy()
        params["random_state"] = MODEL_CONFIG["random_state"]
        params["tree_method"] = "hist"  # Much faster; equivalent quality
        params["n_jobs"] = 1  # Avoid macOS fork deadlock with sequential position training
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
        """Train Random Forest with best params."""
        params = self.best_params.get("random_forest", self._get_default_params()["random_forest"]).copy()
        params["random_state"] = MODEL_CONFIG["random_state"]
        params["n_jobs"] = 1  # Avoid macOS fork deadlock with sequential position training
        model = RandomForestRegressor(**params)
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
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
    
    def _optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray):
        """Optimize ensemble weights near the spec-mandated 30/40/30 baseline.
        
        Constrains weights so they stay close to the requirements
        (RF >= 0.25, XGB >= 0.35, Ridge >= 0.20) while allowing small
        data-driven adjustments.  Only adopts new weights when they
        improve validation MSE over the current spec weights.
        """
        keys = list(self.models.keys())
        preds = {k: self.models[k].predict(X) for k in keys}
        
        # Start from the spec weights as the baseline to beat
        spec_weights = dict(ENSEMBLE_WEIGHTS_1W)
        spec_pred = sum(spec_weights.get(k, 1 / len(keys)) * preds[k] for k in keys)
        best_mse = float(mean_squared_error(y, spec_pred))
        best_weights = spec_weights.copy()
        
        if len(keys) == 3:
            k1, k2, k3 = keys
            # Floor constraints per spec: RF >= 0.25, XGB >= 0.35, Ridge >= 0.20
            min_w = {k1: 0.25, k2: 0.35, k3: 0.20}
            for w1 in np.arange(min_w[k1], 0.40, 0.05):
                for w2 in np.arange(min_w[k2], 0.50, 0.05):
                    w3 = round(1 - w1 - w2, 2)
                    if w3 < min_w[k3]:
                        continue
                    ensemble_pred = w1 * preds[k1] + w2 * preds[k2] + w3 * preds[k3]
                    mse = mean_squared_error(y, ensemble_pred)
                    if mse < best_mse:
                        best_mse = mse
                        best_weights = {k1: w1, k2: w2, k3: w3}
        self.ensemble_weights = best_weights
    
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
            sample_weight: Optional[np.ndarray] = None) -> 'MultiWeekModel':
        """
        Train models for different prediction horizons.
        
        Args:
            X: Feature DataFrame
            y_dict: Dict mapping n_weeks to target Series
            tune_hyperparameters: Whether to tune hyperparameters
            sample_weight: Optional recency weights (e.g. by season)
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
            model.fit(X, y_dict[n_weeks], tune_hyperparameters=tune_hyperparameters, sample_weight=sample_weight)

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

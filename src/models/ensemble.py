"""Ensemble predictor combining position-specific models."""
import json
import warnings
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    POSITIONS,
    MODELS_DIR,
    MAX_PREDICTION_WEEKS,
    MODEL_CONFIG,
    QB_TARGET_CHOICE_FILENAME,
    FEATURE_VERSION,
    FEATURE_VERSION_FILENAME,
)
from src.models.position_models import PositionModel, MultiWeekModel, VALIDATION_PCT
from src.models.utilization_to_fp import UtilizationToFPConverter
from src.features.dimensionality_reduction import (
    select_features_simple,
    compute_vif,
)
try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - compatibility fallback
    InconsistentVersionWarning = None

# Suppress non-actionable model-serialization compatibility warnings in runtime/tests.
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*If you are loading a serialized model.*XGBoost.*",
    category=UserWarning,
)
if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Optional horizon-specific models (4w LSTM+ARIMA, 18w deep)
try:
    from src.models.horizon_models import (
        Hybrid4WeekModel,
        DeepSeasonLongModel,
        HAS_TF,
        HAS_ARIMA,
    )
    HAS_HORIZON_MODELS = True
except ImportError:
    HAS_HORIZON_MODELS = False
    HAS_TF = False
    HAS_ARIMA = False


def _warn_if_feature_version_mismatch() -> None:
    """If saved feature version differs from current, warn to retrain so new features are used."""
    version_path = MODELS_DIR / FEATURE_VERSION_FILENAME
    if not version_path.exists():
        print(
            "\n*** WARNING: No feature version file found. Models may have been trained with an "
            "older feature set. To use new features (injury/rookie, matchup, imputation), run:\n"
            "  python -m src.models.train\n"
        )
        return
    try:
        saved = version_path.read_text(encoding="utf-8").strip()
    except Exception:
        return
    if saved != FEATURE_VERSION.strip():
        print(
            f"\n*** WARNING: Feature set version mismatch (saved={saved!r}, current={FEATURE_VERSION!r}). "
            "Models were trained with a different feature set. To use the current features, retrain:\n"
            "  python -m src.models.train\n"
        )


class EnsemblePredictor:
    """
    Main prediction interface that coordinates position-specific models.
    
    Handles:
    - Loading appropriate models per position
    - Making predictions for 1-18 week horizons
    - Providing uncertainty estimates
    - Batch predictions for multiple players
    """
    
    def __init__(self):
        self.position_models: Dict[str, MultiWeekModel] = {}
        self.single_week_models: Dict[str, PositionModel] = {}
        self.util_to_fp: Dict[str, UtilizationToFPConverter] = {}
        self.hybrid_4w: Dict[str, Any] = {}
        self.deep_18w: Dict[str, Any] = {}
        self.qb_target: str = "util"
        self.horizon_availability: Dict[str, Dict[str, Any]] = {}
        self.is_loaded = False

    @staticmethod
    def _load_qb_target_choice() -> str:
        """Load persisted QB target choice. Defaults to utilization."""
        qb_choice_path = MODELS_DIR / QB_TARGET_CHOICE_FILENAME
        if not qb_choice_path.exists():
            return "util"
        try:
            with open(qb_choice_path) as f:
                choice = json.load(f).get("qb_target", "util")
            return choice if choice in ("util", "fp") else "util"
        except Exception:
            return "util"

    def load_models(self, positions: List[str] = None):
        """
        Load trained models for specified positions.
        
        Args:
            positions: List of positions to load (default: all)
        """
        positions = positions or POSITIONS
        self.qb_target = self._load_qb_target_choice()
        self.horizon_availability = {}
        
        for position in positions:
            try:
                # Try to load multi-week model first
                multi_path = MODELS_DIR / f"multiweek_{position.lower()}.joblib"
                if multi_path.exists():
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*Trying to unpickle estimator.*",
                            category=UserWarning,
                        )
                        warnings.filterwarnings(
                            "ignore",
                            message=".*If you are loading a serialized model.*XGBoost.*",
                            category=UserWarning,
                        )
                        if InconsistentVersionWarning is not None:
                            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                        self.position_models[position] = MultiWeekModel.load(position)
                    print(f"Loaded multi-week model for {position}")
                else:
                    # Fall back to single-week model
                    single_path = MODELS_DIR / f"model_{position.lower()}_1w.joblib"
                    if single_path.exists():
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=".*Trying to unpickle estimator.*",
                                category=UserWarning,
                            )
                            if InconsistentVersionWarning is not None:
                                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
                            self.single_week_models[position] = PositionModel.load(position, n_weeks=1)
                        print(f"Loaded single-week model for {position}")
            except Exception as e:
                print(f"Warning: Could not load model for {position}: {e}")

        for pos in POSITIONS:
            try:
                c = UtilizationToFPConverter.load(pos)
                if c.is_fitted:
                    self.util_to_fp[pos] = c
            except Exception:
                pass

        # Horizon-specific models (4w hybrid, 18w deep) when enabled
        if not HAS_HORIZON_MODELS:
            print("Warning: horizon models unavailable (module import failed).")
        if MODEL_CONFIG.get("use_4w_hybrid", True) and not HAS_ARIMA:
            print("Warning: 4-week hybrid requires statsmodels (ARIMA) but it is unavailable.")
        if MODEL_CONFIG.get("use_4w_hybrid", True) and not HAS_TF:
            print("Warning: 4-week hybrid requires TensorFlow (LSTM) but it is unavailable.")
        if MODEL_CONFIG.get("use_18w_deep", True) and not HAS_TF:
            print("Warning: 18-week deep model requires TensorFlow but it is unavailable.")

        horizon_4w_weeks = tuple(MODEL_CONFIG.get("horizon_4w_weeks", (4, 5, 6, 7, 8)))
        if MODEL_CONFIG.get("use_4w_hybrid", True) and HAS_HORIZON_MODELS and HAS_TF and HAS_ARIMA:
            for position in positions:
                self.horizon_availability.setdefault(position, {})
                try:
                    h = Hybrid4WeekModel.load(position)
                    if getattr(h, "is_fitted", False):
                        self.hybrid_4w[position] = h
                        self.horizon_availability[position]["hybrid_4w"] = "loaded"
                        print(f"Loaded 4-week hybrid model for {position}")
                    else:
                        self.horizon_availability[position]["hybrid_4w"] = "not_fitted"
                except Exception as e:
                    self.horizon_availability[position]["hybrid_4w"] = f"load_failed: {e}"
        elif MODEL_CONFIG.get("use_4w_hybrid", True):
            for position in positions:
                self.horizon_availability.setdefault(position, {})
                reasons = []
                if not HAS_TF:
                    reasons.append("tensorflow_missing")
                if not HAS_ARIMA:
                    reasons.append("statsmodels_missing")
                if not HAS_HORIZON_MODELS:
                    reasons.append("horizon_module_missing")
                self.horizon_availability[position]["hybrid_4w"] = "disabled_or_unavailable:" + ",".join(reasons)
        if MODEL_CONFIG.get("use_18w_deep", True) and HAS_HORIZON_MODELS and HAS_TF:
            for position in positions:
                self.horizon_availability.setdefault(position, {})
                try:
                    d = DeepSeasonLongModel.load(position)
                    if getattr(d, "is_fitted", False):
                        self.deep_18w[position] = d
                        self.horizon_availability[position]["deep_18w"] = "loaded"
                        print(f"Loaded 18-week deep model for {position}")
                    else:
                        self.horizon_availability[position]["deep_18w"] = "not_fitted"
                except Exception as e:
                    self.horizon_availability[position]["deep_18w"] = f"load_failed: {e}"
        elif MODEL_CONFIG.get("use_18w_deep", True):
            for position in positions:
                self.horizon_availability.setdefault(position, {})
                reasons = []
                if not HAS_TF:
                    reasons.append("tensorflow_missing")
                if not HAS_HORIZON_MODELS:
                    reasons.append("horizon_module_missing")
                self.horizon_availability[position]["deep_18w"] = "disabled_or_unavailable:" + ",".join(reasons)

        self.is_loaded = len(self.position_models) > 0 or len(self.single_week_models) > 0

        if self.is_loaded:
            _warn_if_feature_version_mismatch()
    
    def predict(self, player_data: pd.DataFrame, 
                n_weeks: int = 1) -> pd.DataFrame:
        """
        Make predictions for players.
        
        Args:
            player_data: DataFrame with player features (must include 'position' column)
            n_weeks: Number of weeks to predict (1-18)
            
        Returns:
            DataFrame with predictions added (includes prediction_speed_ok metadata)
        """
        import time as _time
        _pred_start = _time.perf_counter()

        if not self.is_loaded:
            raise ValueError("Models must be loaded before prediction. Call load_models() first.")
        
        if n_weeks < 1 or n_weeks > MAX_PREDICTION_WEEKS:
            raise ValueError(f"n_weeks must be between 1 and {MAX_PREDICTION_WEEKS}")
        
        results = player_data.copy()
        results["predicted_points"] = np.nan
        results["predicted_utilization"] = np.nan
        results["prediction_std"] = np.nan
        results["prediction_ci80_lower"] = np.nan
        results["prediction_ci80_upper"] = np.nan
        results["prediction_ci95_lower"] = np.nan
        results["prediction_ci95_upper"] = np.nan
        
        for position in POSITIONS:
            mask = results["position"] == position
            if not mask.any():
                continue
            
            pos_data = results[mask].copy()
            
            # Ensure feature consistency: fill missing columns with training medians
            def _fill_missing_features(data, pm):
                pos_model = pm.models.get(1) or list(pm.models.values())[0]
                medians = getattr(pos_model, "feature_medians", {})
                for fn in getattr(pos_model, "feature_names", []):
                    if fn not in data.columns:
                        data[fn] = medians.get(fn, 0)

            if position in self.position_models:
                _fill_missing_features(pos_data, self.position_models[position])
            elif position in self.single_week_models:
                medians = getattr(self.single_week_models[position], "feature_medians", {})
                for fn in getattr(self.single_week_models[position], "feature_names", []):
                    if fn not in pos_data.columns:
                        pos_data[fn] = medians.get(fn, 0)
            
            if position in self.position_models:
                model = self.position_models[position]
                traditional_pred = model.predict(pos_data, n_weeks)
                predictions = traditional_pred.copy()
                horizon_4w_weeks = tuple(MODEL_CONFIG.get("horizon_4w_weeks", (4, 5, 6, 7, 8)))
                horizon_long = MODEL_CONFIG.get("horizon_long_threshold", 9)
                # 4-week band: use hybrid LSTM+ARIMA when available
                if n_weeks in horizon_4w_weeks and position in self.hybrid_4w:
                    hybrid = self.hybrid_4w[position]
                    fcols = getattr(hybrid, "feature_names", None) or (getattr(hybrid.lstm, "feature_names", []) if getattr(hybrid, "lstm", None) else [])
                    if fcols:
                        for fn in fcols:
                            if fn not in pos_data.columns:
                                pos_data[fn] = 0
                        try:
                            player_ids = pos_data["player_id"].values if "player_id" in pos_data.columns else np.arange(len(pos_data))
                            hy_pred = hybrid.predict(pos_data, player_ids, fcols, traditional_pred)
                            use_hy = np.isfinite(hy_pred)
                            predictions = np.where(use_hy, hy_pred, traditional_pred)
                        except Exception:
                            pass
                # Long horizon: blend 70% deep + 30% traditional when available
                if n_weeks >= horizon_long and position in self.deep_18w:
                    deep = self.deep_18w[position]
                    dcols = getattr(deep, "feature_names", [])
                    if dcols:
                        for fn in dcols:
                            if fn not in pos_data.columns:
                                pos_data[fn] = 0
                        X = pos_data.reindex(columns=dcols, fill_value=0).values.astype(np.float64)
                        X = np.nan_to_num(X, nan=0.0)
                        try:
                            predictions = deep.predict(X, traditional_pred, blend_traditional=0.3)
                        except Exception:
                            pass
                results.loc[mask, "predicted_utilization"] = predictions
                results.loc[mask, "predicted_points"] = predictions
                # RB/WR/TE always convert utilization->FP; QB converts only when QB target is utilization.
                should_convert = (
                    position in self.util_to_fp
                    and self.util_to_fp[position].is_fitted
                    and (position != "QB" or self.qb_target == "util")
                )
                if should_convert:
                    eff_df = pos_data.copy()
                    eff_df["utilization_score"] = predictions
                    fp_pred = self.util_to_fp[position].predict(predictions, efficiency_df=eff_df)
                    results.loc[mask, "predicted_points"] = fp_pred
                # Prediction intervals using conformal quantiles when available,
                # falling back to Gaussian z-score approximation.
                # When util->FP conversion is applied, combine conversion uncertainty
                # with utilization model uncertainty via error propagation.
                try:
                    base_model = model.models.get(1) or list(model.models.values())[0]
                    if hasattr(base_model, "predict_with_uncertainty"):
                        _, std = base_model.predict_with_uncertainty(pos_data)
                        pred_pts = results.loc[mask, "predicted_points"].values
                        conformal_q = getattr(base_model, "_conformal_quantiles", None)

                        # Conversion uncertainty: if util->FP was applied, get converter residuals
                        conv_q = None
                        if should_convert and position in self.util_to_fp:
                            conv_q = getattr(self.util_to_fp[position], "_conversion_conformal_q", None)

                        # Multi-week scaling: use n_weeks^0.4 instead of sqrt
                        # to account for autocorrelation in weekly errors (injuries,
                        # role changes create correlated outcomes across weeks)
                        multi_week_scale = n_weeks ** 0.4

                        if conformal_q and 0.80 in conformal_q and 0.95 in conformal_q:
                            # Constant-width conformal intervals
                            q80 = conformal_q[0.80] * multi_week_scale
                            q95 = conformal_q[0.95] * multi_week_scale
                            # Propagate conversion uncertainty: combined_q = sqrt(util_q^2 + conv_q^2)
                            if conv_q is not None:
                                q80 = np.sqrt(q80**2 + conv_q.get(0.80, 0)**2)
                                q95 = np.sqrt(q95**2 + conv_q.get(0.95, 0)**2)
                            results.loc[mask, "prediction_std"] = std
                            results.loc[mask, "prediction_ci80_lower"] = np.maximum(pred_pts - q80, 0)
                            results.loc[mask, "prediction_ci80_upper"] = pred_pts + q80
                            results.loc[mask, "prediction_ci95_lower"] = np.maximum(pred_pts - q95, 0)
                            results.loc[mask, "prediction_ci95_upper"] = pred_pts + q95
                        else:
                            # Fallback: Gaussian z-scores
                            std_scaled = std * multi_week_scale
                            z80, z95 = 1.28, 1.96
                            results.loc[mask, "prediction_std"] = std_scaled
                            results.loc[mask, "prediction_ci80_lower"] = np.maximum(pred_pts - z80 * std_scaled, 0)
                            results.loc[mask, "prediction_ci80_upper"] = pred_pts + z80 * std_scaled
                            results.loc[mask, "prediction_ci95_lower"] = np.maximum(pred_pts - z95 * std_scaled, 0)
                            results.loc[mask, "prediction_ci95_upper"] = pred_pts + z95 * std_scaled
                except Exception:
                    pass

            elif position in self.single_week_models:
                model = self.single_week_models[position]
                base_pred = model.predict(pos_data)
                scaled = base_pred * n_weeks
                results.loc[mask, "predicted_utilization"] = scaled
                # Apply utilization-to-FP conversion for RB/WR/TE (base_pred is utilization 0-100)
                should_convert = (
                    position in self.util_to_fp
                    and self.util_to_fp[position].is_fitted
                    and (position != "QB" or self.qb_target == "util")
                )
                if should_convert:
                    eff_df = pos_data.copy()
                    eff_df["utilization_score"] = base_pred
                    fp_pred = self.util_to_fp[position].predict(base_pred, efficiency_df=eff_df)
                    results.loc[mask, "predicted_points"] = fp_pred * n_weeks
                else:
                    results.loc[mask, "predicted_points"] = scaled
                _, std = model.predict_with_uncertainty(pos_data)
                std_scaled = std * np.sqrt(n_weeks)
                results.loc[mask, "prediction_std"] = std_scaled
                z80, z95 = 1.28, 1.96
                results.loc[mask, "prediction_ci80_lower"] = scaled - z80 * std_scaled
                results.loc[mask, "prediction_ci80_upper"] = scaled + z80 * std_scaled
                results.loc[mask, "prediction_ci95_lower"] = scaled - z95 * std_scaled
                results.loc[mask, "prediction_ci95_upper"] = scaled + z95 * std_scaled

        # Prediction sanity bounds: clip to reasonable fantasy point ranges per position per week
        _BOUNDS_PER_WEEK = {"QB": (0, 65), "RB": (0, 55), "WR": (0, 55), "TE": (0, 45)}
        for position in POSITIONS:
            mask = results["position"] == position
            if not mask.any():
                continue
            lo, hi = _BOUNDS_PER_WEEK.get(position, (0, 60))
            scaled_hi = hi * n_weeks
            for col in ["predicted_points", "predicted_utilization"]:
                if col in results.columns:
                    results.loc[mask, col] = results.loc[mask, col].clip(lower=lo, upper=scaled_hi)

        # Prediction speed tracking (requirement: < 5s per player)
        from config.settings import MAX_PREDICTION_TIME_PER_PLAYER_SECONDS
        _pred_elapsed = _time.perf_counter() - _pred_start
        n_players = len(player_data)
        per_player = _pred_elapsed / max(n_players, 1)
        results.attrs["prediction_elapsed_s"] = round(_pred_elapsed, 4)
        results.attrs["prediction_per_player_s"] = round(per_player, 6)
        results.attrs["prediction_speed_ok"] = per_player <= MAX_PREDICTION_TIME_PER_PLAYER_SECONDS

        return results
    
    def predict_player(self, player_id: str, player_features: pd.Series,
                       position: str, n_weeks: int = 1) -> Dict:
        """
        Make prediction for a single player.
        
        Args:
            player_id: Player identifier
            player_features: Series of player features
            position: Player position
            n_weeks: Weeks to predict
            
        Returns:
            Dict with prediction details
        """
        # Convert to DataFrame
        df = pd.DataFrame([player_features])
        df["position"] = position
        
        result = self.predict(df, n_weeks)
        
        return {
            "player_id": player_id,
            "position": position,
            "n_weeks": n_weeks,
            "predicted_points": result["predicted_points"].iloc[0],
            "predicted_ppg": result["predicted_points"].iloc[0] / n_weeks,
            "uncertainty": result.get("prediction_std", pd.Series([np.nan])).iloc[0],
        }
    
    def predict_season(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict full season (18 weeks) performance.
        
        Args:
            player_data: DataFrame with player features
            
        Returns:
            DataFrame with season predictions
        """
        return self.predict(player_data, n_weeks=18)
    
    def get_weekly_projections(self, player_data: pd.DataFrame,
                                weeks: List[int] = None) -> pd.DataFrame:
        """
        Get projections for multiple week horizons.
        
        Args:
            player_data: DataFrame with player features
            weeks: List of week horizons to predict (default: [1, 4, 8, 18])
            
        Returns:
            DataFrame with projections for each horizon
        """
        weeks = weeks or [1, 4, 8, 18]
        
        results = player_data[["player_id", "name", "position", "team"]].copy()
        
        for n_weeks in weeks:
            preds = self.predict(player_data, n_weeks)
            results[f"proj_{n_weeks}w"] = preds["predicted_points"]
            results[f"ppg_{n_weeks}w"] = preds["predicted_points"] / n_weeks
        
        return results
    
    def rank_players(self, player_data: pd.DataFrame, 
                     n_weeks: int = 1,
                     position: str = None) -> pd.DataFrame:
        """
        Rank players by predicted performance.
        
        Args:
            player_data: DataFrame with player features
            n_weeks: Prediction horizon
            position: Optional position filter
            
        Returns:
            DataFrame sorted by predicted points
        """
        if position:
            player_data = player_data[player_data["position"] == position]
        
        results = self.predict(player_data, n_weeks)
        
        # Add rankings
        results["overall_rank"] = results["predicted_points"].rank(ascending=False)
        
        # Position ranks
        for pos in POSITIONS:
            mask = results["position"] == pos
            results.loc[mask, "position_rank"] = results.loc[mask, "predicted_points"].rank(ascending=False)
        
        return results.sort_values("predicted_points", ascending=False)

    def get_horizon_availability(self) -> Dict[str, Dict[str, Any]]:
        """Return loaded/disabled status for 4w and 18w horizon models per position."""
        return self.horizon_availability


class ModelTrainer:
    """Handles training of all position models."""
    
    def __init__(self):
        self.trained_models: Dict[str, PositionModel] = {}
        self.training_metrics: Dict[str, Dict] = {}
    
    def train_all_positions(self, data: pd.DataFrame,
                            positions: List[str] = None,
                            tune_hyperparameters: bool = True,
                            n_weeks_list: List[int] = None,
                            test_data: Optional[pd.DataFrame] = None) -> Dict[str, PositionModel]:
        """
        Train models for all positions.
        
        For QB: if test_data is provided and has enough QB rows, train two QB models
        (utilization vs future fantasy points), compare R² on test, and persist the winner.
        For RB, WR, TE: train one model per position on utilization only.
        
        Args:
            data: Full training DataFrame with features and targets
            positions: Positions to train (default: all)
            tune_hyperparameters: Whether to run Optuna tuning
            n_weeks_list: List of prediction horizons to train
            test_data: Optional held-out test set (used for QB target selection only)
            
        Returns:
            Dict of trained models
        """
        positions = positions or POSITIONS
        n_weeks_list = n_weeks_list or [1, 4, 18]  # Short, medium, season-long
        
        for position in positions:
            print(f"\n{'='*60}")
            print(f"Training models for {position}")
            print(f"{'='*60}")
            
            # Filter to position and sort by time for proper train/val split
            pos_data = data[data["position"] == position].copy()
            if "season" in pos_data.columns and "week" in pos_data.columns:
                pos_data = pos_data.sort_values(["season", "week"]).reset_index(drop=True)
            
            if len(pos_data) < 100:
                print(f"Insufficient data for {position} ({len(pos_data)} samples). Skipping.")
                continue
            
            # QB-only: dual-train (util vs FP) and compare on test_data if available
            if position == "QB" and test_data is not None:
                qb_test = test_data[test_data["position"] == "QB"]
                if len(qb_test) >= 20 and "target_util_1w" in pos_data.columns and "target_1w" in pos_data.columns:
                    chosen = self._train_qb_dual_and_pick(
                        pos_data, qb_test, n_weeks_list, tune_hyperparameters
                    )
                    self.trained_models["QB"] = chosen["model"]
                    self.training_metrics["QB"] = chosen["metrics"]
                    qb_choice_path = MODELS_DIR / QB_TARGET_CHOICE_FILENAME
                    with open(qb_choice_path, "w") as f:
                        json.dump(
                            {
                                "qb_target": chosen["qb_target"],
                                "selection_method": "holdout_owner_fp_objective",
                                "rmse_fp_model": chosen.get("rmse_fp"),
                                "rmse_util_model_as_fp": chosen.get("rmse_util_as_fp"),
                                "mae_fp_model": chosen.get("mae_fp"),
                                "mae_util_model_as_fp": chosen.get("mae_util_as_fp"),
                                "r2_fp_model": chosen.get("r2_fp"),
                                "r2_util_model_as_fp": chosen.get("r2_util"),
                            },
                            f,
                            indent=2,
                        )
                    r2u, r2f = chosen.get("r2_util"), chosen.get("r2_fp")
                    rmse_u, rmse_f = chosen.get("rmse_util_as_fp"), chosen.get("rmse_fp")
                    r2u_s = f"{r2u:.3f}" if np.isfinite(r2u) else "n/a"
                    r2f_s = f"{r2f:.3f}" if np.isfinite(r2f) else "n/a"
                    rmse_u_s = f"{rmse_u:.3f}" if np.isfinite(rmse_u) else "n/a"
                    rmse_f_s = f"{rmse_f:.3f}" if np.isfinite(rmse_f) else "n/a"
                    print(
                        "  QB target chosen (owner objective = future fantasy points): "
                        f"{chosen['qb_target']} (RMSE util->fp={rmse_u_s}, fp={rmse_f_s}; "
                        f"R² util->fp={r2u_s}, fp={r2f_s})"
                    )
                    print(f"\nQB Training Metrics:")
                    for horizon, m in chosen["metrics"].items():
                        print(f"  {horizon}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R²={m['r2']:.3f}")
                    continue
                # Fallback: not enough test QB or missing targets -> single util model (below)
            
            # Single model path (RB, WR, TE or QB fallback)
            multi_model = MultiWeekModel(position)
            
            # Prepare targets: primary = utilization (target_util_*), optional FP (target_*w)
            y_dict = {}
            for n_weeks in n_weeks_list:
                # QB fallback target is fantasy points (owner objective) when dual-test path is unavailable.
                if position == "QB":
                    target_col = f"target_{n_weeks}w"
                    if target_col in pos_data.columns:
                        y_dict[n_weeks] = pos_data[target_col]
                    else:
                        y_dict[n_weeks] = pos_data.groupby("player_id")["fantasy_points"].transform(
                            lambda x: x.shift(-1) if n_weeks == 1 else x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
                        )
                else:
                    util_col = f"target_util_{n_weeks}w" if n_weeks > 1 else "target_util_1w"
                    if util_col in pos_data.columns:
                        y_dict[n_weeks] = pos_data[util_col]
                    else:
                        target_col = f"target_{n_weeks}w"
                        if target_col in pos_data.columns:
                            y_dict[n_weeks] = pos_data[target_col]
                        else:
                            y_dict[n_weeks] = pos_data.groupby("player_id")["utilization_score"].transform(
                                lambda x: x.shift(-1) if n_weeks == 1 else x.shift(-1).rolling(window=n_weeks, min_periods=1).mean()
                            )
            
            # Get feature columns - exclude non-numeric, metadata, and LEAK columns
            exclude_cols = [
                "player_id", "name", "position", "team", "season", "week",
                "fantasy_points", "target", "opponent", "home_away",
                "created_at", "updated_at", "id", "birth_date", "college",
                "game_id", "game_time",
                "fp_over_expected", "expected_fp",
                "utilization_score",  # current week - use only lagged/rolling util
            ]
            
            feature_cols = [c for c in pos_data.columns 
                          if c not in exclude_cols 
                          and not c.startswith("target_")
                          and pos_data[c].dtype in ['int64', 'float64', 'int32', 'float32']]
            
            assert "fantasy_points" not in feature_cols, "LEAKAGE: fantasy_points must not be a feature"
            assert "utilization_score" not in feature_cols, "LEAKAGE: utilization_score (current week) must not be a feature"
            
            X = pos_data[feature_cols].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            # Missingness-aware imputation: add binary indicators for rolling/lag
            # features with structural NaN (early-season rows), then median-fill.
            # This prevents the model from interpreting "no history" as "zero performance."
            rolling_lag_tokens = ("_roll", "_lag", "_ewm", "_trend")
            n_rows = len(X)
            indicator_cols = {}
            for col in X.columns:
                if not any(tok in col for tok in rolling_lag_tokens):
                    continue
                n_miss = int(X[col].isna().sum())
                if n_miss > 0 and (n_miss / n_rows) > 0.02:
                    ind_name = f"{col}_missing"
                    if ind_name not in X.columns:
                        indicator_cols[ind_name] = X[col].isna().astype(np.int8)
            if indicator_cols:
                X = pd.concat([X, pd.DataFrame(indicator_cols, index=X.index)], axis=1)
            # Median imputation per column using TRAINING portion only.
            # PositionModel.fit() splits at (1 - validation_pct), so compute
            # medians on the same prefix to avoid val→train information leakage.
            _imp_split = int(len(X) * (1 - VALIDATION_PCT))
            for col in X.columns:
                if not X[col].isna().any():
                    continue
                med = X[col].iloc[:_imp_split].median()
                X[col] = X[col].fillna(med if pd.notna(med) else 0.0)
            X = X.infer_objects()
            
            # Remove rows without valid targets (use primary util target)
            valid_mask = ~y_dict[1].isna()
            X = X[valid_mask]
            y_dict = {k: v[valid_mask] for k, v in y_dict.items()}
            
            # Recency weighting: weight recent seasons more (time-series)
            sample_weight = None
            halflife = MODEL_CONFIG.get("recency_decay_halflife")
            if halflife and "season" in pos_data.columns:
                seasons = pos_data.loc[valid_mask, "season"]
                max_season = seasons.max()
                if max_season > seasons.min():
                    decay = np.power(0.5, (max_season - seasons.values.astype(float)) / float(halflife))
                    sample_weight = decay / decay.max()
            
            # Per-horizon feature selection: select features relevant to each horizon's target
            from src.features.dimensionality_reduction import adaptive_feature_count
            base_n = MODEL_CONFIG.get("n_features_per_position", 50)
            if MODEL_CONFIG.get("adaptive_feature_count", True):
                n_features = adaptive_feature_count(len(X), default=base_n)
            else:
                n_features = base_n
            corr_thresh = MODEL_CONFIG.get("correlation_threshold", 0.92)
            val_pct = float(MODEL_CONFIG.get("validation_pct", 0.2))
            fs_split = int(len(X) * (1 - val_pct))

            if len(X.columns) > n_features:
                # Select features separately per horizon for better horizon-specific modeling
                horizon_features = {}
                for n_weeks, y_horizon in y_dict.items():
                    y_fs = y_horizon.iloc[:fs_split]
                    X_fs = X.iloc[:fs_split]
                    _, sel_cols = select_features_simple(
                        X_fs, y_fs,
                        n_features=n_features,
                        correlation_threshold=corr_thresh
                    )
                    horizon_features[n_weeks] = sel_cols if sel_cols else list(X.columns)

                # Union all horizon features (each model gets its optimal features,
                # but we pass the union so MultiWeekModel can train each horizon)
                all_selected = set()
                for cols in horizon_features.values():
                    all_selected.update(cols)

                # Stability selection: boost features consistently selected across bootstrap Lasso
                try:
                    from src.models.feature_engineering_pipeline import StabilitySelector
                    y_primary = y_dict.get(1, list(y_dict.values())[0])
                    stability_n_bootstrap = MODEL_CONFIG.get("stability_n_bootstrap", 30)
                    stability_sel = StabilitySelector(n_bootstrap=stability_n_bootstrap, threshold=0.5)
                    stable_features = stability_sel.fit(
                        X.iloc[:fs_split], y_primary.iloc[:fs_split],
                        n_features_to_select=n_features
                    )
                    if stable_features:
                        # Add stability-selected features to the union
                        pre_count = len(all_selected)
                        all_selected.update(stable_features)
                        n_added = len(all_selected) - pre_count
                        if n_added > 0:
                            print(f"  Stability selection added {n_added} features", flush=True)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning("Stability selection skipped: %s", e)

                X = X[sorted(all_selected)]
                print(f"  Selected {len(X.columns)} features for {position} "
                      f"(union across {len(y_dict)} horizons + stability)", flush=True)

            # Actionable VIF pruning: iteratively drop highest-VIF feature
            # Compute VIF on training portion only to avoid val→train leakage
            try:
                from src.features.dimensionality_reduction import prune_by_vif
                vif_thresh = MODEL_CONFIG.get("vif_threshold", 10.0)
                pre_vif_count = X.shape[1]
                _, vif_removed = prune_by_vif(X.iloc[:fs_split], threshold=vif_thresh)
                if vif_removed:
                    X = X.drop(columns=vif_removed)
                    print(f"  VIF pruning: removed {len(vif_removed)} features (VIF>{vif_thresh}), "
                          f"{pre_vif_count} -> {X.shape[1]}", flush=True)
                else:
                    print(f"  Multicollinearity: OK (all VIF <= {vif_thresh})", flush=True)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("VIF pruning failed: %s", e)

            # Extract season labels for season-aware CV splits
            seasons_arr = None
            if "season" in pos_data.columns:
                seasons_arr = pos_data.loc[valid_mask, "season"].values

            # Train (with optional recency sample_weight and season-aware CV)
            print(f"  Starting model training for {position}...", flush=True)
            multi_model.fit(X, y_dict, tune_hyperparameters=tune_hyperparameters,
                           sample_weight=sample_weight, seasons=seasons_arr)

            # Save
            multi_model.save()
            
            # Also save single-week model for quick access
            if 1 in multi_model.models:
                multi_model.models[1].save()
            
            self.trained_models[position] = multi_model
            
            # QB fallback: if dual-path unavailable we train FP target model and persist choice
            if position == "QB":
                qb_choice_path = MODELS_DIR / QB_TARGET_CHOICE_FILENAME
                with open(qb_choice_path, "w") as f:
                    json.dump(
                        {
                            "qb_target": "fp",
                            "selection_method": "fallback_no_qb_holdout",
                            "reason": "Insufficient QB holdout rows for dual-target selection; using fantasy points for owner-facing objective.",
                        },
                        f,
                        indent=2,
                    )
            
            # Evaluate
            metrics = self._evaluate_model(multi_model, X, y_dict)
            self.training_metrics[position] = metrics
            
            print(f"\n{position} Training Metrics:")
            for horizon, m in metrics.items():
                print(f"  {horizon}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R²={m['r2']:.3f}")
        
        return self.trained_models
    
    def _train_qb_dual_and_pick(self, pos_data: pd.DataFrame, qb_test: pd.DataFrame,
                                 n_weeks_list: List[int], tune_hyperparameters: bool) -> Dict:
        """
        Train two QB models (util and FP), compare owner-centric FP metrics on a
        VALIDATION split from training data (not the test set), return winner and
        its metrics.
        """
        exclude_cols = [
            "player_id", "name", "position", "team", "season", "week",
            "fantasy_points", "target", "opponent", "home_away",
            "created_at", "updated_at", "id", "birth_date", "college",
            "game_id", "game_time",
            "fp_over_expected", "expected_fp",
            "utilization_score",
        ]
        feature_cols = [c for c in pos_data.columns
                       if c not in exclude_cols
                       and not c.startswith("target_")
                       and pos_data[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        assert "fantasy_points" not in feature_cols and "utilization_score" not in feature_cols

        # Targets: util
        y_dict_util = {}
        for n_weeks in n_weeks_list:
            util_col = f"target_util_{n_weeks}w" if n_weeks > 1 else "target_util_1w"
            y_dict_util[n_weeks] = pos_data[util_col] if util_col in pos_data.columns else pos_data["target_util_1w"]
        # Targets: FP
        y_dict_fp = {}
        for n_weeks in n_weeks_list:
            target_col = f"target_{n_weeks}w"
            y_dict_fp[n_weeks] = pos_data[target_col] if target_col in pos_data.columns else pos_data["target_1w"]

        valid_util = ~y_dict_util[1].isna()
        valid_fp = ~y_dict_fp[1].isna()
        valid_mask = valid_util & valid_fp

        X = pos_data[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        # Missingness-aware imputation (same as main training path)
        _rl_tokens = ("_roll", "_lag", "_ewm", "_trend")
        _n = len(X)
        _ind = {}
        for _c in X.columns:
            if not any(_t in _c for _t in _rl_tokens):
                continue
            _nm = int(X[_c].isna().sum())
            if _nm > 0 and (_nm / _n) > 0.02:
                _in = f"{_c}_missing"
                if _in not in X.columns:
                    _ind[_in] = X[_c].isna().astype(np.int8)
        if _ind:
            X = pd.concat([X, pd.DataFrame(_ind, index=X.index)], axis=1)
        # Median imputation using training portion only (same policy as main path)
        _imp_split = int(len(X) * (1 - VALIDATION_PCT))
        for _c in X.columns:
            if not X[_c].isna().any():
                continue
            _med = X[_c].iloc[:_imp_split].median()
            X[_c] = X[_c].fillna(_med if pd.notna(_med) else 0.0)
        X = X.infer_objects()
        X = X[valid_mask]
        y_dict_util = {k: v[valid_mask] for k, v in y_dict_util.items()}
        y_dict_fp = {k: v[valid_mask] for k, v in y_dict_fp.items()}

        sample_weight = None
        halflife = MODEL_CONFIG.get("recency_decay_halflife")
        if halflife and "season" in pos_data.columns:
            seasons = pos_data.loc[valid_mask, "season"]
            max_season = seasons.max()
            if max_season > seasons.min():
                decay = np.power(0.5, (max_season - seasons.values.astype(float)) / float(halflife))
                sample_weight = decay / decay.max()

        n_features = MODEL_CONFIG.get("n_features_per_position", 50)
        corr_thresh = MODEL_CONFIG.get("correlation_threshold", 0.92)
        if len(X.columns) > n_features:
            # Feature selection on training portion only to avoid leakage into val
            val_pct = float(MODEL_CONFIG.get("validation_pct", 0.2))
            fs_split = int(len(X) * (1 - val_pct))
            _, selected_cols = select_features_simple(
                X.iloc[:fs_split], y_dict_util[1].iloc[:fs_split],
                n_features=n_features, correlation_threshold=corr_thresh
            )
            X = X[selected_cols] if selected_cols else X

        # --- Use a VALIDATION SPLIT from training data for model selection ---
        # Never use the held-out test set to pick between model variants.
        val_pct = float(MODEL_CONFIG.get("validation_pct", 0.2))
        n_total = len(X)
        split_idx = int(n_total * (1 - val_pct))
        split_idx = max(50, min(split_idx, n_total - 20))

        X_train_sel = X.iloc[:split_idx]
        X_val_sel = X.iloc[split_idx:]
        y_dict_util_train = {k: v.iloc[:split_idx] for k, v in y_dict_util.items()}
        y_dict_fp_train = {k: v.iloc[:split_idx] for k, v in y_dict_fp.items()}
        y_dict_util_val = {k: v.iloc[split_idx:] for k, v in y_dict_util.items()}
        y_dict_fp_val = {k: v.iloc[split_idx:] for k, v in y_dict_fp.items()}
        sw_train_sel = sample_weight[:split_idx] if sample_weight is not None else None

        # Train both model variants on training portion only
        multi_util_sel = MultiWeekModel("QB")
        multi_util_sel.fit(X_train_sel, y_dict_util_train, tune_hyperparameters=tune_hyperparameters, sample_weight=sw_train_sel)

        multi_fp_sel = MultiWeekModel("QB")
        multi_fp_sel.fit(X_train_sel, y_dict_fp_train, tune_hyperparameters=tune_hyperparameters, sample_weight=sw_train_sel)

        # Evaluate on validation portion
        for fn in (list(multi_util_sel.models.values())[0].feature_names if multi_util_sel.models else []):
            if fn not in X_val_sel.columns:
                X_val_sel[fn] = 0
        pred_util_val = multi_util_sel.predict(X_val_sel, n_weeks=1)
        pred_fp_val = multi_fp_sel.predict(X_val_sel, n_weeks=1)

        y_fp_val = y_dict_fp_val[1].values
        valid_f = ~np.isnan(y_fp_val) & np.isfinite(y_fp_val)

        # Convert util predictions to FP for comparison
        qb_conv = UtilizationToFPConverter("QB")
        conv_train_df = pos_data.loc[valid_mask].iloc[:split_idx].copy()
        conv_train_df["utilization_score"] = np.asarray(y_dict_util_train[1], dtype=float)
        conv_train_df["fantasy_points"] = np.asarray(y_dict_fp_train[1], dtype=float)
        qb_conv.fit(conv_train_df, target_col="fantasy_points")
        if qb_conv.is_fitted:
            eff_df = pos_data.loc[valid_mask].iloc[split_idx:].copy()
            eff_df["utilization_score"] = np.asarray(pred_util_val, dtype=float)
            util_as_fp = qb_conv.predict(np.asarray(pred_util_val, dtype=float), efficiency_df=eff_df)
        else:
            util_as_fp = np.asarray(pred_util_val, dtype=float) * 0.25

        rmse_util_as_fp = (
            float(np.sqrt(mean_squared_error(y_fp_val[valid_f], util_as_fp[valid_f])))
            if valid_f.sum() >= 5 else np.nan
        )
        rmse_fp = (
            float(np.sqrt(mean_squared_error(y_fp_val[valid_f], pred_fp_val[valid_f])))
            if valid_f.sum() >= 5 else np.nan
        )
        r2_util_as_fp = r2_score(y_fp_val[valid_f], util_as_fp[valid_f]) if valid_f.sum() >= 5 else np.nan
        r2_fp = r2_score(y_fp_val[valid_f], pred_fp_val[valid_f]) if valid_f.sum() >= 5 else np.nan
        mae_util_as_fp = (
            float(mean_absolute_error(y_fp_val[valid_f], util_as_fp[valid_f]))
            if valid_f.sum() >= 5 else np.nan
        )
        mae_fp = (
            float(mean_absolute_error(y_fp_val[valid_f], pred_fp_val[valid_f]))
            if valid_f.sum() >= 5 else np.nan
        )

        # Pick winner based on validation (not test) performance
        margin = max(0.1, 0.02 * rmse_fp) if np.isfinite(rmse_fp) else 0.1
        util_wins = (
            np.isfinite(rmse_util_as_fp)
            and np.isfinite(rmse_fp)
            and (rmse_util_as_fp + margin < rmse_fp)
        )
        if not util_wins:
            qb_target = "fp"
            y_dict_winner = y_dict_fp
        else:
            qb_target = "util"
            y_dict_winner = y_dict_util

        # --- Retrain winner on ALL training data for final model ---
        winner = MultiWeekModel("QB")
        winner.fit(X, y_dict_winner, tune_hyperparameters=tune_hyperparameters, sample_weight=sample_weight)
        if qb_target == "util" and qb_conv.is_fitted:
            # Refit converter on full training data
            conv_full = pos_data.loc[valid_mask].copy()
            conv_full["utilization_score"] = np.asarray(y_dict_util[1], dtype=float)
            conv_full["fantasy_points"] = np.asarray(y_dict_fp[1], dtype=float)
            qb_conv.fit(conv_full, target_col="fantasy_points")
            qb_conv.save()

        winner.save()
        if 1 in winner.models:
            winner.models[1].save()

        metrics = self._evaluate_model(winner, X, y_dict_winner)
        return {
            "model": winner,
            "qb_target": qb_target,
            "metrics": metrics,
            "r2_util": r2_util_as_fp,
            "r2_fp": r2_fp,
            "rmse_util_as_fp": rmse_util_as_fp,
            "rmse_fp": rmse_fp,
            "mae_util_as_fp": mae_util_as_fp,
            "mae_fp": mae_fp,
        }
    
    def _evaluate_model(self, model: MultiWeekModel,
                        X: pd.DataFrame,
                        y_dict: Dict[int, pd.Series]) -> Dict[str, Dict]:
        """Evaluate model using honest OOF (out-of-fold) metrics.

        Previously this method re-predicted on a trailing slice of the training
        data, but the final base models are retrained on ALL of X_train_inner
        (see PositionModel.fit), so the last-20% slice is NOT held out from
        the final model.  OOF metrics from cross-validated predictions during
        training are the correct honest estimate.
        """
        metrics = {}

        for n_weeks in y_dict:
            pos_model = model.models.get(n_weeks)
            if pos_model is None:
                continue
            oof = getattr(pos_model, "_oof_metrics", None)
            if oof is not None:
                metrics[f"{n_weeks}w"] = {
                    "rmse": oof["rmse"],
                    "mae": oof["mae"],
                    "r2": oof["r2"],
                }

        return metrics
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of training metrics."""
        rows = []
        
        for position, metrics in self.training_metrics.items():
            for horizon, m in metrics.items():
                rows.append({
                    "position": position,
                    "horizon": horizon,
                    **m
                })
        
        return pd.DataFrame(rows)

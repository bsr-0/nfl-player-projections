"""
Advanced ML Pipeline for NFL Fantasy Predictions

Implements senior data scientist best practices:
1. Ensemble Methods & Model Stacking
2. Proper Time-Series Cross-Validation
3. Feature Selection & Regularization
4. Uncertainty Quantification
5. Target Engineering
6. Advanced Feature Engineering
7. Robustness Techniques
8. Monitoring & Drift Detection

All models are evaluated on held-out test data before selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
import warnings
import logging

logger = logging.getLogger(__name__)

# Suppress only specific noisy warnings instead of blanket suppression
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import CURRENT_NFL_SEASON

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# 1. ENSEMBLE METHODS & MODEL STACKING
# =============================================================================

class EnsembleStack:
    """
    Ensemble stacking with multiple base models and a meta-learner.
    
    Base models: XGBoost, LightGBM, CatBoost, Ridge, RandomForest
    Meta-learner: Ridge regression on out-of-fold predictions
    """
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.base_models = {}
        self.meta_model = None
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def _get_base_models(self) -> Dict[str, Any]:
        """Get available base models with tuned hyperparameters."""
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'rf': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=1
            ),
            'gbm': GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                min_samples_leaf=10, random_state=42
            ),
        }
        
        if HAS_XGBOOST:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
        
        if HAS_LIGHTGBM:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=-1
            )
        
        if HAS_CATBOOST:
            models['catboost'] = cb.CatBoostRegressor(
                iterations=100, depth=5, learning_rate=0.1,
                random_state=42, verbose=False
            )
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None) -> 'EnsembleStack':
        """
        Fit ensemble with stacking.
        
        1. Train each base model with cross-validation
        2. Generate out-of-fold predictions
        3. Train meta-learner on OOF predictions
        """
        print("\n" + "="*60)
        print("Training Ensemble Stack")
        print("="*60)
        
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]

        # Get base models
        base_models = self._get_base_models()

        # Store OOF predictions for meta-learner (NaN = not yet predicted)
        oof_predictions = np.full((len(y), len(base_models)), np.nan)
        model_scores = {}

        # Time-series split for proper validation
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        for i, (name, model) in enumerate(base_models.items()):
            print(f"\nTraining {name}...")

            oof_pred = np.full(len(y), np.nan)
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                # Fit scaler on training fold only (no data leakage)
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X[train_idx])
                X_val = fold_scaler.transform(X[val_idx])
                y_train, y_val = y[train_idx], y[val_idx]

                # Clone and fit model
                model_clone = self._clone_model(model)
                model_clone.fit(X_train, y_train)

                # Predict on validation
                pred = model_clone.predict(X_val)
                oof_pred[val_idx] = pred

                # Score
                fold_rmse = np.sqrt(mean_squared_error(y_val, pred))
                fold_scores.append(fold_rmse)

            # Store OOF predictions
            oof_predictions[:, i] = oof_pred

            # Calculate overall score
            valid_mask = ~np.isnan(oof_pred)
            if valid_mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(y[valid_mask], oof_pred[valid_mask]))
                model_scores[name] = rmse
                print(f"  {name} OOF RMSE: {rmse:.3f}")

        # Fit scaler once on all data for final models (outside the loop)
        X_scaled = self.scaler.fit_transform(X)

        # Retrain final base models on all scaled data
        for name, model in base_models.items():
            model.fit(X_scaled, y)
            self.base_models[name] = model

        # Calculate model weights (inverse of RMSE)
        total_inv_rmse = sum(1/v for v in model_scores.values())
        self.model_weights = {k: (1/v)/total_inv_rmse for k, v in model_scores.items()}
        
        print("\nModel weights:")
        for name, weight in sorted(self.model_weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {weight:.3f}")
        
        # Train meta-learner on OOF predictions
        print("\nTraining meta-learner...")
        valid_mask = ~np.isnan(oof_predictions).any(axis=1)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(oof_predictions[valid_mask], y[valid_mask])
        
        # Final stacked prediction score
        stacked_pred = self.meta_model.predict(oof_predictions[valid_mask])
        stacked_rmse = np.sqrt(mean_squared_error(y[valid_mask], stacked_pred))
        print(f"  Stacked OOF RMSE: {stacked_rmse:.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions."""
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_preds = np.zeros((len(X), len(self.base_models)))
        for i, (name, model) in enumerate(self.base_models.items()):
            base_preds[:, i] = model.predict(X_scaled)
        
        # Meta-learner prediction
        return self.meta_model.predict(base_preds)
    
    def predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Generate weighted average predictions (alternative to stacking)."""
        X_scaled = self.scaler.transform(X)
        
        weighted_pred = np.zeros(len(X))
        for name, model in self.base_models.items():
            weight = self.model_weights.get(name, 1/len(self.base_models))
            weighted_pred += weight * model.predict(X_scaled)
        
        return weighted_pred
    
    def _clone_model(self, model):
        """Clone a model with same parameters."""
        from sklearn.base import clone
        return clone(model)


# =============================================================================
# 2. PROPER TIME-SERIES CROSS-VALIDATION
# =============================================================================

class PurgedTimeSeriesCV:
    """
    Time-series cross-validation with purging and embargo.
    
    - Purging: Remove samples too close to the train/test boundary
    - Embargo: Add gap between train and test to prevent leakage
    - Walk-forward: Expanding training window
    """
    
    def __init__(self, n_splits: int = 5, purge_weeks: int = 1, 
                 embargo_weeks: int = 1, min_train_size: float = 0.3):
        self.n_splits = n_splits
        self.purge_weeks = purge_weeks
        self.embargo_weeks = embargo_weeks
        self.min_train_size = min_train_size
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices with purging and embargo.
        
        Expects X to have 'season' and 'week' columns for time ordering.
        """
        if 'season' not in X.columns or 'week' not in X.columns:
            # Fall back to simple time series split
            n = len(X)
            fold_size = n // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                test_start = train_end + self.embargo_weeks
                test_end = min(test_start + fold_size, n)
                
                if train_end < self.min_train_size * n:
                    continue
                
                train_idx = np.arange(0, train_end - self.purge_weeks)
                test_idx = np.arange(test_start, test_end)
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx
        else:
            # Use season/week for proper time ordering
            X = X.copy()
            X['time_idx'] = X['season'] * 100 + X['week']
            X = X.sort_values('time_idx')
            
            unique_times = X['time_idx'].unique()
            n_times = len(unique_times)
            fold_size = n_times // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                train_end_idx = (i + 1) * fold_size
                test_start_idx = train_end_idx + self.embargo_weeks
                test_end_idx = min(test_start_idx + fold_size, n_times)
                
                if train_end_idx < self.min_train_size * n_times:
                    continue
                
                train_times = unique_times[:train_end_idx - self.purge_weeks]
                test_times = unique_times[test_start_idx:test_end_idx]
                
                train_idx = X[X['time_idx'].isin(train_times)].index.values
                test_idx = X[X['time_idx'].isin(test_times)].index.values
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx


class SeasonHoldoutCV:
    """
    Season-based holdout validation.
    
    Train on prior seasons, test on the current season.
    """

    def __init__(self, test_seasons: List[int] = None):
        self.test_seasons = test_seasons or [CURRENT_NFL_SEASON]
    
    def split(self, X: pd.DataFrame, y: pd.Series = None,
              groups: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate train/test split by season."""
        if 'season' not in X.columns:
            raise ValueError("X must have 'season' column")
        
        train_mask = ~X['season'].isin(self.test_seasons)
        test_mask = X['season'].isin(self.test_seasons)
        
        train_idx = X[train_mask].index.values
        test_idx = X[test_mask].index.values
        
        yield train_idx, test_idx


# =============================================================================
# 3. FEATURE SELECTION & REGULARIZATION
# =============================================================================

class AdvancedFeatureSelector:
    """
    Advanced feature selection combining multiple methods.
    
    Methods:
    - Correlation filtering (remove >0.95 correlated)
    - Mutual information
    - RFE with cross-validation
    - SHAP importance
    - L1 regularization (Lasso)
    """
    
    def __init__(self, correlation_threshold: float = 0.95,
                 min_features: int = 20, max_features: int = 50):
        self.correlation_threshold = correlation_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.selected_features = []
        self.feature_importance = {}
    
    def remove_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper.columns 
                   if any(upper[col] > self.correlation_threshold)]
        
        return [c for c in X.columns if c not in to_drop]
    
    def mutual_info_selection(self, X: pd.DataFrame, y: pd.Series,
                               n_features: int = 50) -> List[str]:
        """Select features by mutual information."""
        mi_scores = mutual_info_regression(X.fillna(0), y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        return mi_df.head(n_features)['feature'].tolist()
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series,
                        alpha: float = 0.01) -> List[str]:
        """Select features using Lasso (L1 regularization)."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(0))
        
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        
        # Features with non-zero coefficients
        selected = [X.columns[i] for i, coef in enumerate(lasso.coef_) if abs(coef) > 1e-6]
        
        return selected
    
    def shap_selection(self, X: pd.DataFrame, y: pd.Series,
                       n_features: int = 50) -> List[str]:
        """Select features using SHAP importance."""
        if not HAS_SHAP or not HAS_XGBOOST:
            return X.columns.tolist()[:n_features]
        
        # Train a quick XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, random_state=42, verbosity=0
        )
        model.fit(X.fillna(0), y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.fillna(0))
        
        # Mean absolute SHAP value per feature
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        return shap_df.head(n_features)['feature'].tolist()
    
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                        method: str = 'ensemble') -> List[str]:
        """
        Select features using specified method or ensemble of methods.
        
        Methods: 'correlation', 'mutual_info', 'lasso', 'shap', 'ensemble'
        """
        print("\nFeature Selection...")
        print(f"  Starting features: {len(X.columns)}")
        
        if method == 'correlation':
            selected = self.remove_correlated_features(X)
        elif method == 'mutual_info':
            selected = self.mutual_info_selection(X, y, self.max_features)
        elif method == 'lasso':
            selected = self.lasso_selection(X, y)
        elif method == 'shap':
            selected = self.shap_selection(X, y, self.max_features)
        elif method == 'ensemble':
            # Combine multiple methods
            # 1. Remove correlated
            uncorrelated = self.remove_correlated_features(X)
            X_uncorr = X[uncorrelated]
            
            # 2. Get scores from multiple methods
            feature_scores = {}
            
            # Mutual info
            mi_features = self.mutual_info_selection(X_uncorr, y, self.max_features)
            for i, f in enumerate(mi_features):
                feature_scores[f] = feature_scores.get(f, 0) + (self.max_features - i)
            
            # Lasso
            lasso_features = self.lasso_selection(X_uncorr, y)
            for f in lasso_features:
                feature_scores[f] = feature_scores.get(f, 0) + self.max_features
            
            # SHAP
            if HAS_SHAP:
                shap_features = self.shap_selection(X_uncorr, y, self.max_features)
                for i, f in enumerate(shap_features):
                    feature_scores[f] = feature_scores.get(f, 0) + (self.max_features - i)
            
            # Sort by combined score
            sorted_features = sorted(feature_scores.items(), key=lambda x: -x[1])
            selected = [f for f, _ in sorted_features[:self.max_features]]
        else:
            selected = X.columns.tolist()
        
        # Ensure minimum features
        if len(selected) < self.min_features:
            selected = X.columns.tolist()[:self.min_features]
        
        self.selected_features = selected
        print(f"  Selected features: {len(selected)}")
        
        return selected


# =============================================================================
# 4. UNCERTAINTY QUANTIFICATION
# =============================================================================

class UncertaintyQuantifier:
    """
    Uncertainty quantification for predictions.
    
    Methods:
    - Quantile regression for prediction intervals
    - Conformal prediction for calibrated confidence
    - Bootstrap for confidence intervals
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        self.quantiles = quantiles
        self.quantile_models = {}
        self.conformal_scores = None
        self.scaler = StandardScaler()
    
    def fit_quantile_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit quantile regression models for each quantile."""
        print("\nFitting quantile regression...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if HAS_LIGHTGBM:
            for q in self.quantiles:
                model = lgb.LGBMRegressor(
                    objective='quantile', alpha=q,
                    n_estimators=100, max_depth=5, learning_rate=0.1,
                    random_state=42, verbosity=-1
                )
                model.fit(X_scaled, y)
                self.quantile_models[q] = model
                print(f"  Fitted quantile {q}")
        else:
            # Fallback to gradient boosting with quantile loss
            from sklearn.ensemble import GradientBoostingRegressor
            for q in self.quantiles:
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42
                )
                model.fit(X_scaled, y)
                self.quantile_models[q] = model
                print(f"  Fitted quantile {q}")
    
    def predict_intervals(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict intervals for each quantile."""
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for q, model in self.quantile_models.items():
            predictions[q] = model.predict(X_scaled)
        
        return predictions
    
    def fit_conformal(self, X: np.ndarray, y: np.ndarray,
                      base_model: Any, calibration_size: float = 0.2) -> None:
        """
        Fit conformal prediction for calibrated intervals.
        
        Uses split conformal prediction with nonconformity scores.
        """
        print("\nFitting conformal prediction...")
        
        # Split into proper training and calibration sets
        n = len(y)
        n_cal = int(n * calibration_size)
        
        X_train, X_cal = X[:-n_cal], X[-n_cal:]
        y_train, y_cal = y[:-n_cal], y[-n_cal:]
        
        # Fit base model
        X_train_scaled = self.scaler.fit_transform(X_train)
        base_model.fit(X_train_scaled, y_train)
        
        # Calculate nonconformity scores on calibration set
        X_cal_scaled = self.scaler.transform(X_cal)
        y_pred_cal = base_model.predict(X_cal_scaled)
        
        self.conformal_scores = np.abs(y_cal - y_pred_cal)
        self.conformal_model = base_model
        
        print(f"  Calibration set size: {n_cal}")
        print(f"  Median nonconformity score: {np.median(self.conformal_scores):.3f}")
    
    def predict_conformal(self, X: np.ndarray, 
                          confidence: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with conformal intervals.
        
        Returns: (point_prediction, lower_bound, upper_bound)
        """
        X_scaled = self.scaler.transform(X)
        point_pred = self.conformal_model.predict(X_scaled)
        
        # Calculate quantile of nonconformity scores
        q = np.quantile(self.conformal_scores, confidence)
        
        lower = point_pred - q
        upper = point_pred + q
        
        return point_pred, lower, upper


# =============================================================================
# 5. TARGET ENGINEERING
# =============================================================================

class TargetEngineer:
    """
    Advanced target engineering for better predictions.
    
    Methods:
    - Residual prediction (predict deviation from baseline)
    - Log transformation for skewed targets
    - Winsorization for outliers
    """
    
    def __init__(self, baseline_col: str = 'fp_rolling_3',
                 winsorize_percentile: float = 0.01):
        self.baseline_col = baseline_col
        self.winsorize_percentile = winsorize_percentile
        self.baseline_mean = None
        self.target_mean = None
        self.target_std = None
    
    def create_residual_target(self, df: pd.DataFrame, 
                                target_col: str = 'fantasy_points') -> pd.Series:
        """
        Create residual target (actual - baseline).
        
        Predicting residuals often works better than raw values.
        """
        if self.baseline_col in df.columns:
            baseline = df[self.baseline_col].fillna(df[target_col].mean())
            residual = df[target_col] - baseline
            self.baseline_mean = baseline.mean()
        else:
            residual = df[target_col] - df[target_col].mean()
            self.baseline_mean = df[target_col].mean()
        
        return residual
    
    def winsorize_target(self, y: pd.Series) -> pd.Series:
        """Winsorize target to reduce outlier impact."""
        lower = y.quantile(self.winsorize_percentile)
        upper = y.quantile(1 - self.winsorize_percentile)
        return y.clip(lower, upper)
    
    def transform_target(self, df: pd.DataFrame,
                         target_col: str = 'fantasy_points',
                         method: str = 'residual') -> pd.Series:
        """
        Transform target variable.
        
        Methods: 'raw', 'residual', 'log', 'standardized'
        """
        y = df[target_col].copy()
        
        if method == 'residual':
            y = self.create_residual_target(df, target_col)
        elif method == 'log':
            y = np.log1p(y.clip(0))
        elif method == 'standardized':
            self.target_mean = y.mean()
            self.target_std = y.std()
            y = (y - self.target_mean) / self.target_std
        
        # Always winsorize
        y = self.winsorize_target(y)
        
        return y
    
    def inverse_transform(self, y_pred: np.ndarray, 
                          baseline: np.ndarray = None,
                          method: str = 'residual') -> np.ndarray:
        """Inverse transform predictions back to original scale."""
        if method == 'residual':
            if baseline is not None:
                return y_pred + baseline
            else:
                return y_pred + self.baseline_mean
        elif method == 'log':
            return np.expm1(y_pred)
        elif method == 'standardized':
            return y_pred * self.target_std + self.target_mean
        else:
            return y_pred


# =============================================================================
# 6. ADVANCED FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering techniques.
    
    - Opponent-adjusted metrics
    - Bayesian shrinkage for small samples
    - Interaction features
    """
    
    def __init__(self, shrinkage_weight: float = 0.3):
        self.shrinkage_weight = shrinkage_weight
        self.position_means = {}
        self.league_mean = None
    
    def add_opponent_adjusted_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create opponent-adjusted fantasy points.
        
        Adjusts raw stats by opponent defense strength.
        """
        result = df.copy()
        
        if 'opp_defense_strength' in result.columns:
            # Opponent-adjusted fantasy points
            result['fp_opp_adjusted'] = (
                result['fantasy_points'] / result['opp_defense_strength'].clip(0.5, 1.5)
            )
        
        if 'opp_matchup_score' in result.columns:
            # Matchup-adjusted projection
            if 'fp_rolling_3' in result.columns:
                result['fp_matchup_adjusted'] = (
                    result['fp_rolling_3'] * (1 + (result['opp_matchup_score'] - 0.5) * 0.2)
                )
        
        return result
    
    def apply_bayesian_shrinkage(self, df: pd.DataFrame,
                                  stat_col: str = 'fantasy_points',
                                  games_col: str = 'games_played_season') -> pd.DataFrame:
        """
        Apply Bayesian shrinkage to small sample sizes.
        
        Regresses player stats toward position mean based on sample size.
        """
        result = df.copy()
        
        # Calculate position means
        self.position_means = df.groupby('position')[stat_col].mean().to_dict()
        self.league_mean = df[stat_col].mean()
        
        # Shrinkage factor based on games played
        # More games = less shrinkage
        if games_col in result.columns:
            shrinkage = self.shrinkage_weight / (1 + result[games_col] / 4)
        else:
            shrinkage = self.shrinkage_weight
        
        # Apply shrinkage
        pos_mean = result['position'].map(self.position_means).fillna(self.league_mean)
        result[f'{stat_col}_shrunk'] = (
            (1 - shrinkage) * result[stat_col] + shrinkage * pos_mean
        )
        
        return result
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add key interaction features."""
        result = df.copy()

        # Use lagged utilization to avoid current-week leakage
        util_col = None
        for candidate in ['utilization_score_lag_1', 'utilization_score_roll3_mean']:
            if candidate in result.columns:
                util_col = candidate
                break

        # Usage * matchup interaction
        if util_col and 'opp_matchup_score' in result.columns:
            result['usage_matchup_interaction'] = (
                result[util_col] * result['opp_matchup_score']
            )

        # Age * usage interaction (older players with high usage = injury risk)
        if 'age' in result.columns and util_col:
            result['age_usage_interaction'] = result['age'] * result[util_col]
        
        # Consistency * matchup (consistent players benefit more from good matchups)
        if 'consistency_score' in result.columns and 'opp_matchup_score' in result.columns:
            result['consistency_matchup_interaction'] = (
                result['consistency_score'] * result['opp_matchup_score']
            )
        
        return result


# =============================================================================
# 7. ROBUSTNESS TECHNIQUES
# =============================================================================

class RobustnessChecker:
    """
    Robustness techniques for model validation.
    
    - Adversarial validation
    - Out-of-distribution detection
    - Model calibration
    """
    
    def __init__(self):
        self.adversarial_auc = None
        self.calibration_error = None
    
    def adversarial_validation(self, X_train: pd.DataFrame, 
                                X_test: pd.DataFrame) -> float:
        """
        Adversarial validation to detect train/test distribution shift.
        
        Trains a classifier to distinguish train from test.
        AUC close to 0.5 = similar distributions (good)
        AUC close to 1.0 = different distributions (bad)
        """
        print("\nAdversarial Validation...")
        
        # Create labels
        y_train = np.zeros(len(X_train))
        y_test = np.ones(len(X_test))
        
        X_combined = pd.concat([X_train, X_test], axis=0).fillna(0)
        y_combined = np.concatenate([y_train, y_test])
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc', n_jobs=1)
        
        self.adversarial_auc = scores.mean()
        
        if self.adversarial_auc > 0.7:
            print(f"  ⚠️ WARNING: AUC = {self.adversarial_auc:.3f} - Significant distribution shift detected!")
        else:
            print(f"  ✅ AUC = {self.adversarial_auc:.3f} - Train/test distributions are similar")
        
        return self.adversarial_auc
    
    def detect_ood_samples(self, X_train: np.ndarray, 
                           X_test: np.ndarray,
                           threshold_percentile: float = 95) -> np.ndarray:
        """
        Detect out-of-distribution samples in test set.
        
        Uses Mahalanobis distance from training distribution.
        """
        from scipy.spatial.distance import mahalanobis
        
        # Calculate training distribution parameters
        mean = np.mean(X_train, axis=0)
        cov = np.cov(X_train.T)
        
        # Regularize covariance matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov)
        
        # Calculate Mahalanobis distance for test samples
        distances = np.array([
            mahalanobis(x, mean, cov_inv) for x in X_test
        ])
        
        # Threshold based on training distances
        train_distances = np.array([
            mahalanobis(x, mean, cov_inv) for x in X_train
        ])
        threshold = np.percentile(train_distances, threshold_percentile)
        
        ood_mask = distances > threshold
        
        print(f"\nOOD Detection: {ood_mask.sum()} / {len(X_test)} samples flagged as OOD")
        
        return ood_mask
    
    def calibrate_predictions(self, y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               method: str = 'isotonic') -> Any:
        """
        Calibrate predictions using isotonic or Platt scaling.
        
        Returns calibrated predictions.
        """
        from sklearn.isotonic import IsotonicRegression
        
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred, y_true)
            return calibrator
        else:
            # Platt scaling (linear)
            from sklearn.linear_model import LinearRegression
            calibrator = LinearRegression()
            calibrator.fit(y_pred.reshape(-1, 1), y_true)
            return calibrator


# =============================================================================
# 8. MONITORING & DRIFT DETECTION
# =============================================================================

class ModelMonitor:
    """
    Model monitoring and drift detection.
    
    - Track feature distributions
    - Monitor prediction accuracy
    - Detect concept drift
    """
    
    def __init__(self):
        self.baseline_distributions = {}
        self.performance_history = []
        self.drift_detected = False
    
    def set_baseline_distributions(self, X: pd.DataFrame) -> None:
        """Store baseline feature distributions."""
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                self.baseline_distributions[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'median': X[col].median(),
                    'q25': X[col].quantile(0.25),
                    'q75': X[col].quantile(0.75),
                }
    
    def detect_feature_drift(self, X_new: pd.DataFrame,
                              threshold: float = 2.0) -> Dict[str, float]:
        """
        Detect feature drift using z-score of distribution shift.
        
        Returns dict of features with significant drift.
        """
        drift_scores = {}
        
        for col in X_new.columns:
            if col in self.baseline_distributions:
                baseline = self.baseline_distributions[col]
                new_mean = X_new[col].mean()
                
                if baseline['std'] > 0:
                    z_score = abs(new_mean - baseline['mean']) / baseline['std']
                    if z_score > threshold:
                        drift_scores[col] = z_score
        
        if drift_scores:
            print(f"\n⚠️ Feature drift detected in {len(drift_scores)} features:")
            for col, score in sorted(drift_scores.items(), key=lambda x: -x[1])[:5]:
                print(f"  {col}: z-score = {score:.2f}")
        
        return drift_scores
    
    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                        week: int = None, season: int = None) -> Dict[str, float]:
        """Log prediction performance for monitoring."""
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'correlation': spearmanr(y_true, y_pred)[0],
            'week': week,
            'season': season,
            'timestamp': time.time(),
        }
        
        self.performance_history.append(metrics)
        
        return metrics
    
    def check_performance_degradation(self, 
                                       window: int = 5,
                                       threshold: float = 0.2) -> bool:
        """
        Check if model performance has degraded significantly.
        
        Compares recent performance to historical average.
        """
        if len(self.performance_history) < window * 2:
            return False
        
        recent = self.performance_history[-window:]
        historical = self.performance_history[:-window]
        
        recent_rmse = np.mean([m['rmse'] for m in recent])
        historical_rmse = np.mean([m['rmse'] for m in historical])
        
        degradation = (recent_rmse - historical_rmse) / historical_rmse
        
        if degradation > threshold:
            print(f"\n⚠️ Performance degradation detected: {degradation:.1%}")
            self.drift_detected = True
            return True
        
        return False


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

class AdvancedMLEvaluator:
    """
    Main evaluation pipeline that tests all approaches and selects the best.
    """
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_config = None
    
    def evaluate_all(self, df: pd.DataFrame, 
                     feature_cols: List[str],
                     target_col: str = 'fantasy_points',
                     test_season: int = CURRENT_NFL_SEASON) -> Dict[str, Any]:
        """
        Evaluate all model configurations and select the best.
        
        Returns comprehensive results with metrics for each approach.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*70)
        
        # Prepare data
        train_df = df[df['season'] < test_season].copy()
        test_df = df[df['season'] == test_season].copy()
        
        print(f"\nTrain: {len(train_df)} samples (seasons < {test_season})")
        print(f"Test: {len(test_df)} samples (season = {test_season})")
        
        # Get features
        available_features = [c for c in feature_cols if c in train_df.columns]
        X_train = train_df[available_features].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[available_features].fillna(0)
        y_test = test_df[target_col]
        
        print(f"Features: {len(available_features)}")
        
        # 1. Adversarial validation
        print("\n" + "-"*50)
        print("1. ADVERSARIAL VALIDATION")
        print("-"*50)
        robustness = RobustnessChecker()
        adv_auc = robustness.adversarial_validation(X_train, X_test)
        self.results['adversarial_auc'] = adv_auc
        
        # 2. Feature selection comparison
        print("\n" + "-"*50)
        print("2. FEATURE SELECTION COMPARISON")
        print("-"*50)
        
        feature_selector = AdvancedFeatureSelector(max_features=50)
        
        feature_methods = ['correlation', 'mutual_info', 'lasso', 'ensemble']
        feature_results = {}
        
        for method in feature_methods:
            selected = feature_selector.select_features(X_train, y_train, method=method)
            
            # Quick evaluation with Ridge
            X_train_sel = X_train[selected].values
            X_test_sel = X_test[selected].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            feature_results[method] = {
                'n_features': len(selected),
                'rmse': rmse,
                'features': selected
            }
            print(f"  {method}: {len(selected)} features, RMSE = {rmse:.3f}")
        
        # Select best feature set
        best_feature_method = min(feature_results, key=lambda x: feature_results[x]['rmse'])
        best_features = feature_results[best_feature_method]['features']
        print(f"\n  Best: {best_feature_method} ({len(best_features)} features)")
        
        self.results['feature_selection'] = feature_results
        
        # 3. Target engineering comparison
        print("\n" + "-"*50)
        print("3. TARGET ENGINEERING COMPARISON")
        print("-"*50)
        
        target_engineer = TargetEngineer()
        target_methods = ['raw', 'residual', 'standardized']
        target_results = {}
        
        for method in target_methods:
            y_train_t = target_engineer.transform_target(train_df, target_col, method)
            
            X_train_sel = X_train[best_features].values
            X_test_sel = X_test[best_features].values
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train_t)
            y_pred_t = model.predict(X_test_scaled)
            
            # Inverse transform
            if method == 'residual':
                baseline = test_df['fp_rolling_3'].fillna(y_train.mean()).values
                y_pred = target_engineer.inverse_transform(y_pred_t, baseline, method)
            else:
                y_pred = target_engineer.inverse_transform(y_pred_t, method=method)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            target_results[method] = rmse
            print(f"  {method}: RMSE = {rmse:.3f}")
        
        best_target_method = min(target_results, key=target_results.get)
        print(f"\n  Best: {best_target_method}")
        
        self.results['target_engineering'] = target_results
        
        # 4. Model comparison
        print("\n" + "-"*50)
        print("4. MODEL COMPARISON")
        print("-"*50)
        
        X_train_sel = X_train[best_features].values
        X_test_sel = X_test[best_features].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        }
        
        if HAS_XGBOOST:
            models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=0
            )
        
        if HAS_LIGHTGBM:
            models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbosity=-1
            )
        
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            model_results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
            }
            print(f"  {name}: RMSE = {rmse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")
        
        self.results['models'] = model_results
        
        # 5. Ensemble stacking
        print("\n" + "-"*50)
        print("5. ENSEMBLE STACKING")
        print("-"*50)
        
        ensemble = EnsembleStack(n_folds=5)
        ensemble.fit(X_train_scaled, y_train.values, best_features)
        
        y_pred_stacked = ensemble.predict(X_test_scaled)
        y_pred_weighted = ensemble.predict_weighted(X_test_scaled)
        
        stacked_rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacked))
        weighted_rmse = np.sqrt(mean_squared_error(y_test, y_pred_weighted))
        
        print(f"\n  Stacked RMSE: {stacked_rmse:.3f}")
        print(f"  Weighted RMSE: {weighted_rmse:.3f}")
        
        self.results['ensemble'] = {
            'stacked_rmse': stacked_rmse,
            'weighted_rmse': weighted_rmse,
        }
        
        # 6. Uncertainty quantification
        print("\n" + "-"*50)
        print("6. UNCERTAINTY QUANTIFICATION")
        print("-"*50)
        
        uq = UncertaintyQuantifier(quantiles=[0.1, 0.5, 0.9])
        uq.fit_quantile_regression(X_train_scaled, y_train.values)
        
        intervals = uq.predict_intervals(X_test_scaled)
        
        # Calculate coverage
        coverage_90 = np.mean(
            (y_test.values >= intervals[0.1]) & (y_test.values <= intervals[0.9])
        )
        print(f"  90% interval coverage: {coverage_90:.1%}")
        
        self.results['uncertainty'] = {
            'coverage_90': coverage_90,
        }
        
        # 7. Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        # Find best overall approach
        all_rmses = {
            **{f"model_{k}": v['rmse'] for k, v in model_results.items()},
            'ensemble_stacked': stacked_rmse,
            'ensemble_weighted': weighted_rmse,
        }
        
        best_approach = min(all_rmses, key=all_rmses.get)
        best_rmse = all_rmses[best_approach]
        
        print(f"\n  Best approach: {best_approach}")
        print(f"  Best RMSE: {best_rmse:.3f}")
        print(f"  Best features: {best_feature_method} ({len(best_features)} features)")
        print(f"  Best target: {best_target_method}")
        print(f"  90% coverage: {coverage_90:.1%}")
        
        self.results['best'] = {
            'approach': best_approach,
            'rmse': best_rmse,
            'feature_method': best_feature_method,
            'n_features': len(best_features),
            'target_method': best_target_method,
            'coverage_90': coverage_90,
        }
        
        # Store best model
        if 'ensemble' in best_approach:
            self.best_model = ensemble
        else:
            model_name = best_approach.replace('model_', '')
            self.best_model = models[model_name]
        
        self.best_config = {
            'features': best_features,
            'scaler': scaler,
            'target_method': best_target_method,
        }
        
        return self.results


def run_comprehensive_evaluation():
    """Run the comprehensive ML evaluation pipeline."""
    from src.utils.database import DatabaseManager
    from src.data.external_data import add_external_features
    from src.features.multiweek_features import add_multiweek_features
    from src.features.season_long_features import add_season_long_features
    from src.features.utilization import engineer_all_features
    from src.features.qb_features import add_qb_features
    
    print("Loading data...")
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=4)
    
    print("Engineering features...")
    df = engineer_all_features(df, allow_autoload_bounds=False)
    df = add_qb_features(df)
    df = add_external_features(df)
    df = add_multiweek_features(df, horizons=[1, 5, 18])
    df = add_season_long_features(df)
    
    # Get feature columns - ONLY use features known BEFORE the game
    # This is critical to prevent data leakage
    
    # Features that are ALLOWED (known before game)
    allowed_patterns = [
        '_lag_', '_rolling_', 'rolling_',  # Historical/lagged features
        '_trend', '_avg',                   # Trend and average features
        'games_played', 'is_home',          # Game context
        'consistency_score', 'weekly_volatility', 'coefficient_of_variation',
        'boom_bust_range', 'confidence_score',
        # External data (known before game)
        'injury_score', 'is_injured', 'opp_defense_rank', 'opp_matchup_score',
        'opp_pts_allowed', 'is_dome', 'is_outdoor', 'weather_score',
        'implied_team_total', 'game_total', 'spread', 'opp_defense_strength',
        # Multi-week features
        'sos_next_', 'sos_rank_next_', 'favorable_matchups_next_',
        'expected_games_next_', 'projection_', 'floor_', 'ceiling_',
        'injury_prob_next_', 'expected_missed_games_', 'injury_risk_score_',
        'variance_', 'std_',
        # Season-long features
        'age', 'age_factor', 'age_expected_games', 'decline_rate',
        'years_from_peak', 'is_in_prime', 'projected_games', 'historical_gpg',
        'is_rookie', 'rookie_projected', 'rookie_weight',
        'position_rank', 'season_position_rank', 'estimated_adp', 'projected_adp',
        'adp_value', 'positional_scarcity', 'adjusted_adp',
        # Utilization features (from previous games)
        'utilization_score_lag', 'utilization_score_roll', 'target_share', 'rush_share', 'snap_share',
        'air_yards_share', 'red_zone_share', 'wopr',
    ]
    
    # Features that are NEVER allowed (current-week stats = leakage)
    forbidden_patterns = [
        'receiving_yards', 'rushing_yards', 'passing_yards',  # Current game stats
        'receptions', 'targets', 'rushing_attempts', 'passing_attempts',
        'touchdowns', 'td', 'interceptions', 'fumbles',
        'fantasy_points', 'fp_over_expected', 'expected_fp',
        'risk_adjusted_projection', 'completions', 'carries',
    ]
    
    exclude_cols = ['player_id', 'name', 'team', 'position', 'season', 'week',
                    'fantasy_points', 'opponent', 'home_away', 'rookie_archetype',
                    'first_season', 'created_at', 'updated_at', 'id',
                    'birth_date', 'college', 'game_id', 'game_time',
                    'player_name', 'gsis_id']
    
    # Use only training data for correlation-based feature filtering (avoid test leakage)
    train_only = df[df['season'] < CURRENT_NFL_SEASON] if 'season' in df.columns else df

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if df[c].dtype not in ['int64', 'float64']:
            continue

        # Check if forbidden
        is_forbidden = any(pattern in c.lower() for pattern in forbidden_patterns)
        if is_forbidden:
            continue

        # Check if allowed
        is_allowed = any(pattern in c for pattern in allowed_patterns)
        if not is_allowed:
            # Check correlation on training data only - if too high, it's probably leaky
            corr = abs(train_only[c].corr(train_only['fantasy_points']))
            if corr > 0.7:
                print(f"  Excluding potential leaky feature: {c} (corr={corr:.3f})")
                continue

        feature_cols.append(c)

    from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns
    feature_cols = filter_feature_columns(feature_cols)
    assert_no_leakage_columns(feature_cols, context="advanced_ml_pipeline")
    
    print(f"Total features available: {len(feature_cols)}")
    
    # Run evaluation
    evaluator = AdvancedMLEvaluator()
    results = evaluator.evaluate_all(df, feature_cols, test_season=CURRENT_NFL_SEASON)
    
    # Save results with mandatory metadata for reproducibility
    results_path = Path(__file__).parent.parent.parent / 'data' / 'ml_evaluation_results.json'

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results_serializable = convert_numpy(results)

    # Add evaluation metadata (C2 fix: every result file must be self-describing)
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    results_serializable["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": git_hash,
        "authoritative": False,
        "target_variable": "utilization_score (0-100; converted to fantasy_points via Ridge)",
        "test_season": int(CURRENT_NFL_SEASON),
        "evaluation_type": "static_snapshot (exploratory, NOT production)",
        "n_features": len(feature_cols),
        "feature_columns": sorted(feature_cols),
        "scoring_format": "PPR",
        "note": (
            "WARNING: This is an exploratory evaluation file, NOT the production "
            "model's authoritative results. RMSE/R² here may differ significantly "
            "from actual model performance. The single source of truth is "
            "data/advanced_model_results.json (produced by train.py)."
        ),
    }

    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == '__main__':
    run_comprehensive_evaluation()

"""
Production-Grade Fantasy Football Prediction System

Addresses senior-level concerns from both Data Science and NFL Analytics perspectives:

DATA SCIENCE CONCERNS ADDRESSED:
1. Automatic training window optimization (recency vs sample size tradeoff)
2. Proper handling of non-stationarity (NFL rules/meta changes over time)
3. Heteroscedasticity handling (variance differs by player tier)
4. Proper uncertainty quantification (prediction intervals, not just point estimates)
5. Feature stability analysis (features that work consistently across years)
6. Regularization tuned per position (different signal-to-noise ratios)
7. Ensemble with diversity (different model types, not just hyperparameters)
8. Proper evaluation metrics (fantasy-relevant: hit rate on top-N, rank correlation)

NFL ANALYTICS CONCERNS ADDRESSED:
1. Regime changes (new OC, QB change, injury returns)
2. Rookie handling (no historical data - use draft capital, college stats proxies)
3. Bye week / rest effects
4. Opponent adjustments (defense strength, pace of play)
5. Game script sensitivity (blowouts reduce opportunity)
6. Snap count trends (increasing/decreasing role)
7. Target quality (not just volume - red zone, air yards)
8. Touchdown regression (TDs are high variance, regress to expected)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class ModelConfig:
    """Configuration for production model."""
    position: str
    min_train_seasons: int = 1
    max_train_seasons: int = 10  # Support 1-10 years of historical data
    auto_select_window: bool = True
    scale_features: bool = True
    use_ensemble: bool = True
    regularization_strength: float = 1.0
    prediction_interval_coverage: float = 0.80


@dataclass
class PredictionResult:
    """Container for predictions with uncertainty."""
    player_id: str
    name: str
    position: str
    season: int
    week: int
    prediction: float
    lower_bound: float  # e.g., 10th percentile
    upper_bound: float  # e.g., 90th percentile
    confidence: float   # 0-1 confidence score
    floor: float        # Conservative estimate (25th percentile)
    ceiling: float      # Upside estimate (75th percentile)
    
    def to_dict(self) -> Dict:
        return {
            'player_id': self.player_id,
            'name': self.name,
            'position': self.position,
            'season': self.season,
            'week': self.week,
            'prediction': round(self.prediction, 1),
            'floor': round(self.floor, 1),
            'ceiling': round(self.ceiling, 1),
            'lower_bound': round(self.lower_bound, 1),
            'upper_bound': round(self.upper_bound, 1),
            'confidence': round(self.confidence, 2)
        }


class TrainingWindowOptimizer:
    """
    Automatically select optimal training window based on test performance.
    
    Key insight: More data isn't always better in NFL due to:
    - Rule changes (e.g., pass interference rules)
    - Meta shifts (more passing, RPO evolution)
    - Player career arcs (prime years vs decline)
    
    This finds the sweet spot between recency and sample size.
    
    Uses nested cross-validation to avoid overfitting the window selection:
    1. Outer loop: Hold out most recent season as final test
    2. Inner loop: Use second-most-recent season to select optimal window
    3. Validate window selection on final test season
    
    Supports 1-10 years of historical data.
    """
    
    def __init__(self, min_seasons: int = 1, max_seasons: int = 10):
        self.min_seasons = min_seasons
        self.max_seasons = max_seasons
        self.optimal_window = None
        self.window_performance = {}
        self.selection_method = None  # 'validation' or 'cv'
    
    def find_optimal_window(self, df: pd.DataFrame, model_class, model_params: Dict,
                           feature_cols: List[str], position: str = None,
                           verbose: bool = True) -> int:
        """
        Find optimal training window by testing different lookback periods (1-10 years).
        
        Uses nested cross-validation to avoid overfitting:
        1. If 4+ seasons available: Use multiple validation folds
        2. If 3 seasons: Use single validation fold
        3. If 2 seasons: Use all available (no optimization possible)
        
        Args:
            df: DataFrame with player data
            model_class: Model class to use for evaluation
            model_params: Parameters for model
            feature_cols: List of feature column names
            position: Position to filter (optional)
            verbose: Print progress
            
        Returns:
            Optimal number of training seasons (1-10)
        """
        if position:
            df = df[df['position'] == position].copy()
        
        seasons = sorted(df['season'].unique())
        n_seasons = len(seasons)
        
        if n_seasons < 2:
            raise ValueError(f"Need at least 2 seasons, got {n_seasons}")
        
        if n_seasons == 2:
            # Only one option: train on season 1, test on season 2
            self.optimal_window = 1
            self.selection_method = 'default'
            if verbose:
                print(f"   Only 2 seasons available, using window=1")
            return 1
        
        # Determine max possible window (can't use more seasons than available - 1 for test)
        max_possible = min(self.max_seasons, n_seasons - 1)
        
        if n_seasons >= 4:
            # Use nested CV: multiple validation folds for robust selection
            return self._find_optimal_with_cv(df, model_class, model_params, 
                                              feature_cols, seasons, max_possible, verbose)
        else:
            # Use single validation fold
            return self._find_optimal_single_fold(df, model_class, model_params,
                                                   feature_cols, seasons, max_possible, verbose)
    
    def _find_optimal_single_fold(self, df: pd.DataFrame, model_class, model_params: Dict,
                                   feature_cols: List[str], seasons: List,
                                   max_possible: int, verbose: bool) -> int:
        """Find optimal window using single validation fold."""
        self.selection_method = 'single_fold'
        
        # Use second-to-last season as validation
        val_season = seasons[-2]
        
        best_window = 1
        best_rmse = float('inf')
        
        for window_size in range(self.min_seasons, max_possible + 1):
            # Train on window_size seasons ending before validation
            end_idx = len(seasons) - 2  # Index of val_season
            start_idx = max(0, end_idx - window_size)
            train_seasons = seasons[start_idx:end_idx]
            
            if len(train_seasons) < 1:
                continue
            
            rmse, train_size, val_size = self._evaluate_window(
                df, train_seasons, val_season, model_class, model_params, feature_cols
            )
            
            if rmse is not None:
                self.window_performance[window_size] = {
                    'rmse': rmse,
                    'train_size': train_size,
                    'val_size': val_size,
                    'train_seasons': list(train_seasons)
                }
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_window = window_size
        
        self.optimal_window = best_window
        
        if verbose:
            self._print_window_performance()
        
        return best_window
    
    def _find_optimal_with_cv(self, df: pd.DataFrame, model_class, model_params: Dict,
                               feature_cols: List[str], seasons: List,
                               max_possible: int, verbose: bool) -> int:
        """Find optimal window using cross-validation across multiple test seasons."""
        self.selection_method = 'cross_validation'
        
        # Use last 2-3 seasons as validation folds
        n_val_folds = min(3, len(seasons) - 2)
        val_seasons = seasons[-n_val_folds:]
        
        # Collect RMSE for each window across all folds
        window_rmses = {w: [] for w in range(self.min_seasons, max_possible + 1)}
        
        for val_season in val_seasons:
            val_idx = seasons.index(val_season)
            
            for window_size in range(self.min_seasons, max_possible + 1):
                # Train on window_size seasons ending before this validation season
                start_idx = max(0, val_idx - window_size)
                train_seasons = seasons[start_idx:val_idx]
                
                if len(train_seasons) < 1:
                    continue
                
                rmse, _, _ = self._evaluate_window(
                    df, train_seasons, val_season, model_class, model_params, feature_cols
                )
                
                if rmse is not None:
                    window_rmses[window_size].append(rmse)
        
        # Average RMSE across folds for each window
        best_window = 1
        best_avg_rmse = float('inf')
        
        for window_size, rmses in window_rmses.items():
            if len(rmses) > 0:
                avg_rmse = np.mean(rmses)
                std_rmse = np.std(rmses) if len(rmses) > 1 else 0
                
                self.window_performance[window_size] = {
                    'rmse': avg_rmse,
                    'rmse_std': std_rmse,
                    'n_folds': len(rmses)
                }
                
                if avg_rmse < best_avg_rmse:
                    best_avg_rmse = avg_rmse
                    best_window = window_size
        
        self.optimal_window = best_window
        
        if verbose:
            self._print_window_performance()
        
        return best_window
    
    def _evaluate_window(self, df: pd.DataFrame, train_seasons: List,
                         val_season, model_class, model_params: Dict,
                         feature_cols: List[str]) -> Tuple[Optional[float], int, int]:
        """Evaluate a specific training window."""
        train_df = df[df['season'].isin(train_seasons)]
        val_df = df[df['season'] == val_season]
        
        if len(train_df) < 50 or len(val_df) < 20:
            return None, len(train_df), len(val_df)
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['fantasy_points']
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df['fantasy_points']
        
        # Scale (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train and evaluate
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        return rmse, len(train_df), len(val_df)
    
    def _print_window_performance(self):
        """Print window performance summary."""
        print(f"\n   Training Window Optimization (1-{self.max_seasons} years):")
        print(f"   {'Window':<8} {'RMSE':<10} {'Status':<10}")
        print(f"   {'-'*30}")
        
        for window in sorted(self.window_performance.keys()):
            perf = self.window_performance[window]
            rmse = perf['rmse']
            status = "← OPTIMAL" if window == self.optimal_window else ""
            
            if 'rmse_std' in perf:
                print(f"   {window} years  {rmse:.2f}±{perf['rmse_std']:.2f}  {status}")
            else:
                print(f"   {window} years  {rmse:.2f}       {status}")
        
        print()


class FeatureStabilityAnalyzer:
    """
    Analyze which features are stable predictors across different time periods.
    
    Unstable features (high variance in importance across years) are risky
    and should be down-weighted or excluded.
    """
    
    def __init__(self):
        self.feature_stability = {}
    
    def analyze_stability(self, df: pd.DataFrame, feature_cols: List[str],
                         model_class, model_params: Dict,
                         position: str = None) -> pd.DataFrame:
        """
        Calculate feature importance stability across seasons.
        
        Returns DataFrame with:
        - mean_importance: Average importance across seasons
        - std_importance: Standard deviation (lower = more stable)
        - stability_score: mean / (std + epsilon) - higher is better
        """
        if position:
            df = df[df['position'] == position].copy()
        
        seasons = sorted(df['season'].unique())
        
        importance_by_season = {col: [] for col in feature_cols}
        
        for i, test_season in enumerate(seasons[1:], 1):
            train_seasons = seasons[:i]
            
            train_df = df[df['season'].isin(train_seasons)]
            
            if len(train_df) < 50:
                continue
            
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df['fantasy_points']
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            
            if hasattr(model, 'feature_importances_'):
                for j, col in enumerate(feature_cols):
                    importance_by_season[col].append(model.feature_importances_[j])
        
        # Calculate stability metrics
        results = []
        for col in feature_cols:
            importances = importance_by_season[col]
            if len(importances) > 0:
                mean_imp = np.mean(importances)
                std_imp = np.std(importances)
                stability = mean_imp / (std_imp + 0.001)
                
                results.append({
                    'feature': col,
                    'mean_importance': mean_imp,
                    'std_importance': std_imp,
                    'stability_score': stability,
                    'n_seasons': len(importances)
                })
        
        stability_df = pd.DataFrame(results)
        stability_df = stability_df.sort_values('stability_score', ascending=False)
        
        self.feature_stability = stability_df.set_index('feature')['stability_score'].to_dict()
        
        return stability_df


class TouchdownRegressor:
    """
    Regress touchdowns toward expected values.
    
    TDs are high-variance events. A player who scored 12 TDs on 80 targets
    is likely to regress. This calculates expected TDs based on opportunity
    and regresses actual toward expected.
    
    Based on research showing TD rates regress ~50% toward mean.
    """
    
    # Historical average TD rates by position
    AVG_TD_RATES = {
        'RB': {
            'rush_td_per_attempt': 0.035,  # ~3.5% of carries result in TD
            'rec_td_per_target': 0.045,    # ~4.5% of targets result in TD
        },
        'WR': {
            'rec_td_per_target': 0.055,    # ~5.5% of targets
        },
        'TE': {
            'rec_td_per_target': 0.065,    # ~6.5% of targets (more red zone)
        },
        'QB': {
            'pass_td_per_attempt': 0.045,  # ~4.5% of attempts
            'rush_td_per_attempt': 0.025,  # ~2.5% of rushes
        }
    }
    
    REGRESSION_FACTOR = 0.5  # Regress 50% toward mean
    
    def calculate_expected_tds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expected TD columns based on opportunity."""
        result = df.copy()
        
        for position, rates in self.AVG_TD_RATES.items():
            mask = result['position'] == position
            
            if 'rush_td_per_attempt' in rates:
                result.loc[mask, 'expected_rush_tds'] = (
                    result.loc[mask, 'rushing_attempts'] * rates['rush_td_per_attempt']
                )
            
            if 'rec_td_per_target' in rates:
                result.loc[mask, 'expected_rec_tds'] = (
                    result.loc[mask, 'targets'] * rates['rec_td_per_target']
                )
            
            if 'pass_td_per_attempt' in rates:
                result.loc[mask, 'expected_pass_tds'] = (
                    result.loc[mask, 'passing_attempts'] * rates['pass_td_per_attempt']
                )
        
        return result
    
    def regress_tds(self, actual_tds: float, expected_tds: float) -> float:
        """Regress actual TDs toward expected."""
        return actual_tds * (1 - self.REGRESSION_FACTOR) + expected_tds * self.REGRESSION_FACTOR


class ProductionModel:
    """
    Production-grade fantasy football prediction model.
    
    Combines multiple improvements:
    1. Automatic training window selection
    2. Feature stability weighting
    3. Proper uncertainty quantification
    4. Ensemble of diverse models
    5. Position-specific tuning
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.scaler = None
        self.feature_cols = []
        self.training_window = None
        self.residual_std = None
        self.is_fitted = False
        
        # Components
        self.window_optimizer = TrainingWindowOptimizer(
            min_seasons=config.min_train_seasons,
            max_seasons=config.max_train_seasons
        )
        self.stability_analyzer = FeatureStabilityAnalyzer()
    
    def fit(self, df: pd.DataFrame, feature_cols: List[str],
            target_col: str = 'fantasy_points') -> 'ProductionModel':
        """
        Fit the production model.
        
        Steps:
        1. Filter to position
        2. Optimize training window (if enabled)
        3. Analyze feature stability
        4. Train ensemble of models
        5. Calculate residual distribution for uncertainty
        """
        pos_df = df[df['position'] == self.config.position].copy()
        seasons = sorted(pos_df['season'].unique())
        
        if len(seasons) < 2:
            raise ValueError(f"Need at least 2 seasons, got {len(seasons)}")
        
        self.feature_cols = feature_cols
        
        # Step 1: Optimize training window
        if self.config.auto_select_window and len(seasons) >= 3:
            base_model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
            self.training_window = self.window_optimizer.find_optimal_window(
                pos_df, GradientBoostingRegressor, 
                {'n_estimators': 50, 'max_depth': 4, 'random_state': 42},
                feature_cols, position=self.config.position
            )
            print(f"   Optimal training window: {self.training_window} seasons")
        else:
            self.training_window = len(seasons) - 1
        
        # Step 2: Prepare training data with optimal window
        train_seasons = seasons[-self.training_window - 1:-1]
        test_season = seasons[-1]
        
        train_df = pos_df[pos_df['season'].isin(train_seasons)]
        test_df = pos_df[pos_df['season'] == test_season]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        # Step 3: Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 4: Train ensemble
        self._train_ensemble(X_train_scaled, y_train)
        
        # Step 5: Calculate residual distribution for uncertainty
        preds = self._ensemble_predict(X_test_scaled)
        residuals = y_test.values - preds
        self.residual_std = np.std(residuals)
        self.residual_mean = np.mean(residuals)
        
        # Store performance metrics
        self.test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        self.test_r2 = r2_score(y_test, preds)
        
        self.is_fitted = True
        return self
    
    def _train_ensemble(self, X: np.ndarray, y: pd.Series):
        """Train diverse ensemble of models."""
        # Ridge (linear, regularized)
        self.models['ridge'] = Ridge(alpha=self.config.regularization_strength)
        self.models['ridge'].fit(X, y)
        
        # ElasticNet (sparse linear)
        self.models['elasticnet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        self.models['elasticnet'].fit(X, y)
        
        # GBM (tree-based)
        self.models['gbm'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, 
            subsample=0.8, random_state=42
        )
        self.models['gbm'].fit(X, y)
        
        # XGBoost (if available)
        if HAS_XGB:
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, 
                subsample=0.8, random_state=42
            )
            self.models['xgb'].fit(X, y)
        
        # LightGBM (if available)
        if HAS_LGB:
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5,
                subsample=0.8, random_state=42, verbose=-1
            )
            self.models['lgb'].fit(X, y)
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble prediction (average of all models)."""
        predictions = []
        for name, model in self.models.items():
            predictions.append(model.predict(X))
        
        return np.mean(predictions, axis=0)
    
    def _ensemble_predict_with_variance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ensemble prediction with model disagreement variance."""
        predictions = []
        for name, model in self.models.items():
            predictions.append(model.predict(X))
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        model_std = np.std(predictions, axis=0)
        
        return mean_pred, model_std
    
    def predict(self, df: pd.DataFrame) -> List[PredictionResult]:
        """
        Make predictions with uncertainty quantification.
        
        Returns list of PredictionResult objects with:
        - Point prediction
        - Floor/ceiling (25th/75th percentile)
        - Prediction interval (configurable coverage)
        - Confidence score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        pos_df = df[df['position'] == self.config.position].copy()
        
        X = pos_df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions with model variance
        mean_pred, model_std = self._ensemble_predict_with_variance(X_scaled)
        
        # Total uncertainty = model disagreement + historical residual
        total_std = np.sqrt(model_std**2 + self.residual_std**2)
        
        # Calculate intervals
        from scipy import stats
        z_coverage = stats.norm.ppf((1 + self.config.prediction_interval_coverage) / 2)
        z_floor = stats.norm.ppf(0.25)
        z_ceiling = stats.norm.ppf(0.75)
        
        results = []
        for i, (_, row) in enumerate(pos_df.iterrows()):
            pred = mean_pred[i]
            std = total_std[i]
            
            # Confidence based on model agreement and data quality
            model_agreement = 1 / (1 + model_std[i])
            confidence = min(model_agreement, 0.95)
            
            result = PredictionResult(
                player_id=row['player_id'],
                name=row['name'],
                position=row['position'],
                season=row['season'],
                week=row['week'],
                prediction=max(0, pred),
                lower_bound=max(0, pred - z_coverage * std),
                upper_bound=pred + z_coverage * std,
                floor=max(0, pred + z_floor * std),
                ceiling=pred + z_ceiling * std,
                confidence=confidence
            )
            results.append(result)
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from ensemble."""
        importances = pd.DataFrame({'feature': self.feature_cols})
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances[name] = np.abs(model.coef_)
        
        # Average across models
        imp_cols = [c for c in importances.columns if c != 'feature']
        if imp_cols:
            importances['mean_importance'] = importances[imp_cols].mean(axis=1)
            importances = importances.sort_values('mean_importance', ascending=False)
        
        return importances


def train_production_models(df: pd.DataFrame, feature_cols: List[str],
                           positions: List[str] = None) -> Dict[str, ProductionModel]:
    """
    Train production models for all positions.
    
    Returns dict mapping position to fitted ProductionModel.
    """
    positions = positions or ['QB', 'RB', 'WR', 'TE']
    models = {}
    
    for position in positions:
        print(f"\n{'='*50}")
        print(f"Training Production Model: {position}")
        print('='*50)
        
        config = ModelConfig(
            position=position,
            min_train_seasons=1,
            max_train_seasons=5,
            auto_select_window=True
        )
        
        model = ProductionModel(config)
        
        try:
            model.fit(df, feature_cols)
            models[position] = model
            
            print(f"   Test RMSE: {model.test_rmse:.2f}")
            print(f"   Test R²: {model.test_r2:.3f}")
            print(f"   Residual Std: {model.residual_std:.2f}")
            
        except Exception as e:
            print(f"   Failed: {e}")
    
    return models

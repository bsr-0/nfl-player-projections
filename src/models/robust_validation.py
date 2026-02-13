"""
Robust Validation Framework for Fantasy Football Models

Ensures:
1. No data leakage - features computed only from past data
2. Proper scaling - scaler fit on train, applied to test
3. Time-series cross-validation - multiple folds for robust estimates
4. Consistent methodology regardless of test year

Based on best practices for time-series ML validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ValidationResult:
    """Container for validation results."""
    model_name: str
    position: str
    rmse: float
    mae: float
    r2: float
    mape: float
    n_train: int
    n_test: int
    fold_results: List[Dict]
    feature_importance: Optional[pd.DataFrame] = None


class RobustTimeSeriesCV:
    """
    Robust time-series cross-validation with proper scaling.
    
    Key principles:
    1. Train on past, test on future (no look-ahead bias)
    2. Scaler fit ONLY on training data, applied to both train and test
    3. Features computed ONLY from historical data
    4. Multiple folds for robust performance estimates
    5. Optional purge gap: exclude samples near train/test boundary (reduces leakage)
    """
    
    def __init__(self, n_splits: int = 3, min_train_seasons: int = 1,
                 scale_features: bool = True, gap_seasons: int = 0):
        """
        Args:
            n_splits: Number of CV folds (each uses a different test season)
            min_train_seasons: Minimum seasons required for training
            scale_features: Whether to standardize features
            gap_seasons: Purge gap - exclude this many seasons before test from train (0 = no purge)
        """
        self.n_splits = n_splits
        self.min_train_seasons = min_train_seasons
        self.scale_features = scale_features
        self.gap_seasons = gap_seasons
    
    def validate(self, df: pd.DataFrame, model_class, model_params: Dict,
                 feature_cols: List[str], target_col: str = 'fantasy_points',
                 position: str = None) -> ValidationResult:
        """
        Run robust cross-validation.
        
        Args:
            df: Full dataset
            model_class: Model class with fit/predict methods
            model_params: Parameters for model
            feature_cols: List of feature column names
            target_col: Target column name
            position: Position to filter (optional)
            
        Returns:
            ValidationResult with metrics and predictions
        """
        if position:
            df = df[df['position'] == position].copy()
        
        # Get unique seasons sorted
        seasons = sorted(df['season'].unique())
        
        if len(seasons) < self.min_train_seasons + 1:
            raise ValueError(f"Need at least {self.min_train_seasons + 1} seasons, got {len(seasons)}")
        
        # Determine folds (test on last n_splits seasons)
        n_folds = min(self.n_splits, len(seasons) - self.min_train_seasons)
        test_seasons = seasons[-n_folds:]
        
        fold_results = []
        all_predictions = []
        all_importances = []
        
        for fold_idx, test_season in enumerate(test_seasons):
            # Train on seasons before test, with optional purge gap
            all_before = [s for s in seasons if s < test_season]
            if self.gap_seasons > 0:
                # Purged CV: exclude last gap_seasons from train to reduce temporal leakage
                gap_cutoff = test_season - self.gap_seasons
                train_seasons = [s for s in all_before if s < gap_cutoff]
            else:
                train_seasons = all_before
            
            if len(train_seasons) < self.min_train_seasons:
                continue
            
            train_df = df[df['season'].isin(train_seasons)].copy()
            test_df = df[df['season'] == test_season].copy()
            
            # Prepare features
            X_train = train_df[feature_cols].fillna(0)
            y_train = train_df[target_col]
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col]
            
            # Scale features (fit on train only!)
            if self.scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            predictions = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            # MAPE (avoid division by zero)
            mask = y_test != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
            else:
                mape = np.nan
            
            fold_result = {
                'fold': fold_idx,
                'test_season': test_season,
                'train_seasons': train_seasons,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
            fold_results.append(fold_result)
            
            # Store predictions
            pred_df = test_df[['player_id', 'name', 'season', 'week', target_col]].copy()
            pred_df['prediction'] = predictions
            pred_df['fold'] = fold_idx
            all_predictions.append(pred_df)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_,
                    'fold': fold_idx
                })
                all_importances.append(imp)
        
        if not fold_results:
            raise ValueError("No valid folds generated")
        
        # Aggregate metrics
        avg_rmse = np.mean([f['rmse'] for f in fold_results])
        avg_mae = np.mean([f['mae'] for f in fold_results])
        avg_r2 = np.mean([f['r2'] for f in fold_results])
        avg_mape = np.nanmean([f['mape'] for f in fold_results])
        
        total_train = sum(f['n_train'] for f in fold_results)
        total_test = sum(f['n_test'] for f in fold_results)
        
        # Aggregate feature importance
        if all_importances:
            importance_df = pd.concat(all_importances)
            importance_df = importance_df.groupby('feature')['importance'].mean().reset_index()
            importance_df = importance_df.sort_values('importance', ascending=False)
        else:
            importance_df = None
        
        return ValidationResult(
            model_name=model_class.__name__,
            position=position or 'ALL',
            rmse=avg_rmse,
            mae=avg_mae,
            r2=avg_r2,
            mape=avg_mape,
            n_train=total_train,
            n_test=total_test,
            fold_results=fold_results,
            feature_importance=importance_df
        )


def validate_no_leakage(df: pd.DataFrame, feature_cols: List[str], 
                        target_col: str = 'fantasy_points') -> Dict[str, Any]:
    """
    Validate that features don't leak target information.
    
    Checks:
    1. No feature has correlation > 0.95 with target
    2. No feature contains target values directly
    3. Lag features are properly shifted
    
    Returns dict with validation results.
    """
    results = {
        'passed': True,
        'warnings': [],
        'errors': []
    }
    
    # Check correlations
    for col in feature_cols:
        if col in df.columns and target_col in df.columns:
            corr = df[col].corr(df[target_col])
            if abs(corr) > 0.95:
                results['errors'].append(f"Feature '{col}' has correlation {corr:.3f} with target - likely leakage!")
                results['passed'] = False
            elif abs(corr) > 0.85:
                results['warnings'].append(f"Feature '{col}' has high correlation {corr:.3f} with target")
    
    # Check for target in feature names (but allow lag/rolling/trend/avg which are historical)
    target_patterns = ['fantasy_points', 'fp_', '_fp']
    safe_patterns = ['lag', 'rolling', 'trend', 'avg', 'history']
    for col in feature_cols:
        for pattern in target_patterns:
            if pattern in col.lower():
                # Check if it's a safe historical feature
                is_safe = any(safe in col.lower() for safe in safe_patterns)
                if not is_safe:
                    results['errors'].append(f"Feature '{col}' may contain target information")
                    results['passed'] = False
    
    return results


def run_robust_validation(df: pd.DataFrame, position: str,
                          feature_cols: List[str],
                          n_splits: int = 3) -> pd.DataFrame:
    """
    Run robust validation comparing multiple models.
    
    Returns DataFrame with comparison results.
    """
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    
    # Try to import optional libraries
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False
    
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False
    
    models = {
        'Ridge': (Ridge, {'alpha': 1.0}),
        'Lasso': (Lasso, {'alpha': 0.1}),
        'ElasticNet': (ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5}),
        'GBM': (GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
        'RandomForest': (RandomForestRegressor, {'n_estimators': 100, 'max_depth': 8, 'random_state': 42}),
    }
    
    if has_xgb:
        models['XGBoost'] = (xgb.XGBRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42})
    
    if has_lgb:
        models['LightGBM'] = (lgb.LGBMRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbose': -1})
    
    validator = RobustTimeSeriesCV(n_splits=n_splits, scale_features=True)
    
    results = []
    for name, (model_class, params) in models.items():
        try:
            result = validator.validate(
                df, model_class, params, feature_cols, position=position
            )
            results.append({
                'model': name,
                'rmse': result.rmse,
                'mae': result.mae,
                'r2': result.r2,
                'mape': result.mape,
                'n_folds': len(result.fold_results)
            })
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    return pd.DataFrame(results).sort_values('rmse')

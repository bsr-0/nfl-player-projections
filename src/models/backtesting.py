"""
Robust Backtesting Framework for Fantasy Football Predictions

Implements proper time-series validation to prevent data leakage:
- Walk-forward validation (train on past, test on future)
- Season-based splits (never train on future seasons)
- Week-by-week evaluation
- Proper feature engineering within each fold

Key Issues Addressed:
1. Data Leakage: Features must be computed ONLY from past data
2. Look-ahead Bias: No future information in training
3. Realistic Evaluation: Mimics actual prediction scenario
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class BacktestResult:
    """Container for backtest results."""
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    feature_importance: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Backtest Results:\n"
            f"  RMSE: {self.metrics['rmse']:.2f}\n"
            f"  MAE: {self.metrics['mae']:.2f}\n"
            f"  R²: {self.metrics['r2']:.3f}\n"
            f"  MAPE: {self.metrics.get('mape', 0):.1f}%\n"
            f"  Samples: {len(self.predictions)}\n"
            f"  Folds: {len(self.fold_metrics)}"
        )


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series data.
    
    This is the CORRECT way to validate fantasy football models:
    - Train on weeks 1-N, predict week N+1
    - Expand training window, predict next week
    - Never use future data in training
    
    Parameters:
        min_train_weeks: Minimum weeks of training data before first prediction
        gap_weeks: Gap between training and test (e.g., 1 = predict next week)
        expanding: If True, use all past data; if False, use rolling window
        window_size: Size of rolling window (only if expanding=False)
    """
    
    def __init__(self, min_train_weeks: int = 4, gap_weeks: int = 0,
                 expanding: bool = True, window_size: int = 8):
        self.min_train_weeks = min_train_weeks
        self.gap_weeks = gap_weeks
        self.expanding = expanding
        self.window_size = window_size
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for walk-forward validation.
        
        Args:
            df: DataFrame with 'season' and 'week' columns
            
        Yields:
            Tuples of (train_df, test_df)
        """
        # Sort by time
        df = df.sort_values(['season', 'week']).reset_index(drop=True)
        
        # Get unique (season, week) combinations
        time_points = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
        time_points = list(time_points.itertuples(index=False, name=None))
        
        splits = []
        
        for i in range(self.min_train_weeks, len(time_points)):
            test_season, test_week = time_points[i]
            
            # Training data: all weeks before test week (minus gap)
            train_end_idx = i - self.gap_weeks - 1
            
            if train_end_idx < self.min_train_weeks - 1:
                continue
            
            if self.expanding:
                train_start_idx = 0
            else:
                train_start_idx = max(0, train_end_idx - self.window_size + 1)
            
            # Get train time points
            train_times = time_points[train_start_idx:train_end_idx + 1]
            
            # Create masks
            train_mask = df.apply(
                lambda row: (row['season'], row['week']) in train_times, axis=1
            )
            test_mask = (df['season'] == test_season) & (df['week'] == test_week)
            
            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
        
        return splits


class SeasonBasedValidator:
    """
    Season-based cross-validation.
    
    Train on complete past seasons, test on future season.
    More realistic for pre-season predictions.
    """
    
    def __init__(self, min_train_seasons: int = 1):
        self.min_train_seasons = min_train_seasons
    
    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate season-based train/test splits."""
        seasons = sorted(df['season'].unique())
        
        splits = []
        
        for i in range(self.min_train_seasons, len(seasons)):
            test_season = seasons[i]
            train_seasons = seasons[:i]
            
            train_df = df[df['season'].isin(train_seasons)].copy()
            test_df = df[df['season'] == test_season].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))
        
        return splits


class FantasyBacktester:
    """
    Main backtesting engine for fantasy football models.
    
    Handles:
    - Proper time-series splits
    - Feature engineering within each fold (no leakage)
    - Model training and evaluation
    - Aggregated metrics
    """
    
    def __init__(self, model_class, model_params: Dict = None,
                 feature_engineer: Callable = None):
        """
        Args:
            model_class: Class of model to use (must have fit/predict methods)
            model_params: Parameters to pass to model constructor
            feature_engineer: Function to engineer features (applied per fold)
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.feature_engineer = feature_engineer
    
    def run_backtest(self, df: pd.DataFrame, 
                     validator: Any,
                     target_col: str = 'fantasy_points',
                     feature_cols: List[str] = None,
                     position: str = None,
                     verbose: bool = True) -> BacktestResult:
        """
        Run full backtest.
        
        Args:
            df: Full dataset
            validator: Validator object with split() method
            target_col: Target column name
            feature_cols: List of feature columns (if None, auto-detect)
            position: Position to filter (optional)
            verbose: Print progress
            
        Returns:
            BacktestResult with predictions and metrics
        """
        if position:
            df = df[df['position'] == position].copy()
        
        splits = validator.split(df)
        
        if verbose:
            print(f"Running backtest with {len(splits)} folds...")
        
        all_predictions = []
        fold_metrics = []
        all_importances = []
        
        for fold_idx, (train_df, test_df) in enumerate(splits):
            if verbose and fold_idx % 5 == 0:
                print(f"  Fold {fold_idx + 1}/{len(splits)}...")
            
            # Apply feature engineering WITHIN fold (prevents leakage)
            if self.feature_engineer:
                train_df = self.feature_engineer(train_df)
                test_df = self.feature_engineer(test_df)
            
            # Get feature columns
            if feature_cols is None:
                exclude = ['player_id', 'name', 'position', 'team', 'opponent',
                          'season', 'week', target_col, 'home_away', 'created_at', 'id']
                feature_cols_fold = [c for c in train_df.columns 
                                    if c not in exclude and train_df[c].dtype in ['int64', 'float64']]
            else:
                feature_cols_fold = [c for c in feature_cols if c in train_df.columns]
            
            # Prepare data
            X_train = train_df[feature_cols_fold].fillna(0)
            y_train = train_df[target_col]
            X_test = test_df[feature_cols_fold].fillna(0)
            y_test = test_df[target_col]
            
            # Train model
            model = self.model_class(**self.model_params)
            
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            except Exception as e:
                if verbose:
                    print(f"    Fold {fold_idx} failed: {e}")
                continue
            
            # Store predictions
            pred_df = test_df[['player_id', 'name', 'position', 'season', 'week', target_col]].copy()
            pred_df['prediction'] = predictions
            pred_df['error'] = pred_df['prediction'] - pred_df[target_col]
            pred_df['abs_error'] = np.abs(pred_df['error'])
            all_predictions.append(pred_df)
            
            # Calculate fold metrics
            fold_metric = self._calculate_metrics(y_test.values, predictions)
            fold_metric['fold'] = fold_idx
            fold_metric['train_size'] = len(train_df)
            fold_metric['test_size'] = len(test_df)
            fold_metrics.append(fold_metric)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                imp = pd.DataFrame({
                    'feature': feature_cols_fold,
                    'importance': model.feature_importances_
                })
                imp['fold'] = fold_idx
                all_importances.append(imp)
        
        # Aggregate results
        if not all_predictions:
            raise ValueError("No successful folds in backtest")
        
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Overall metrics
        overall_metrics = self._calculate_metrics(
            predictions_df[target_col].values,
            predictions_df['prediction'].values
        )
        
        # Average feature importance
        if all_importances:
            importance_df = pd.concat(all_importances)
            importance_df = importance_df.groupby('feature')['importance'].mean().reset_index()
            importance_df = importance_df.sort_values('importance', ascending=False)
        else:
            importance_df = None
        
        if verbose:
            print(f"\nBacktest Complete:")
            print(f"  Total predictions: {len(predictions_df)}")
            print(f"  RMSE: {overall_metrics['rmse']:.2f}")
            print(f"  MAE: {overall_metrics['mae']:.2f}")
            print(f"  R²: {overall_metrics['r2']:.3f}")
        
        return BacktestResult(
            predictions=predictions_df,
            metrics=overall_metrics,
            fold_metrics=fold_metrics,
            feature_importance=importance_df
        )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }


class ModelComparison:
    """Compare multiple models using the same backtest framework."""
    
    def __init__(self, validator: Any):
        self.validator = validator
        self.results = {}
    
    def add_model(self, name: str, model_class, model_params: Dict = None,
                  feature_engineer: Callable = None):
        """Add a model to compare."""
        self.results[name] = {
            'model_class': model_class,
            'model_params': model_params or {},
            'feature_engineer': feature_engineer,
            'result': None
        }
    
    def run_comparison(self, df: pd.DataFrame, 
                       target_col: str = 'fantasy_points',
                       position: str = None,
                       verbose: bool = True) -> pd.DataFrame:
        """Run backtest for all models and compare."""
        
        for name, config in self.results.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing: {name}")
                print('='*50)
            
            backtester = FantasyBacktester(
                model_class=config['model_class'],
                model_params=config['model_params'],
                feature_engineer=config['feature_engineer']
            )
            
            try:
                result = backtester.run_backtest(
                    df, self.validator, target_col, position=position, verbose=verbose
                )
                config['result'] = result
            except Exception as e:
                print(f"  Failed: {e}")
                config['result'] = None
        
        # Create comparison DataFrame
        comparison = []
        for name, config in self.results.items():
            if config['result']:
                row = {'model': name}
                row.update(config['result'].metrics)
                comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('rmse')
        
        if verbose:
            print(f"\n{'='*50}")
            print("Model Comparison (sorted by RMSE)")
            print('='*50)
            print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self) -> Tuple[str, BacktestResult]:
        """Return the best performing model."""
        best_name = None
        best_rmse = float('inf')
        
        for name, config in self.results.items():
            if config['result'] and config['result'].metrics['rmse'] < best_rmse:
                best_rmse = config['result'].metrics['rmse']
                best_name = name
        
        return best_name, self.results[best_name]['result']


def run_quick_backtest(df: pd.DataFrame, position: str = None) -> BacktestResult:
    """
    Run a quick backtest with default settings.
    
    Useful for rapid iteration and testing.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    validator = WalkForwardValidator(min_train_weeks=4, gap_weeks=0)
    
    backtester = FantasyBacktester(
        model_class=GradientBoostingRegressor,
        model_params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    )
    
    return backtester.run_backtest(df, validator, position=position)


# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Load sample data
    from src.utils.database import DatabaseManager
    db = DatabaseManager()
    df = db.get_player_stats()
    
    if not df.empty:
        # Run comparison
        validator = WalkForwardValidator(min_train_weeks=4)
        comparison = ModelComparison(validator)
        
        comparison.add_model('Ridge', Ridge, {'alpha': 1.0})
        comparison.add_model('GBM', GradientBoostingRegressor, 
                            {'n_estimators': 100, 'max_depth': 5})
        
        results = comparison.run_comparison(df, position='RB')
        print(results)

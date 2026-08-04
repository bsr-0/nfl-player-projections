"""
Advanced ML Models for Fantasy Football Prediction

This module implements:
1. LSTM model for sequential/time-series predictions
2. Stacked Ensemble (XGBoost + LightGBM + Ridge)
3. Time-series cross-validation
4. Vegas lines integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

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
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


# =============================================================================
# TIME-SERIES CROSS-VALIDATION
# =============================================================================
class TimeSeriesValidator:
    """
    Proper time-series cross-validation for fantasy football.
    
    Key principle: Never use future data to predict the past.
    Uses expanding window or sliding window approach.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = None, gap: int = 0):
        """
        Args:
            n_splits: Number of CV folds
            test_size: Size of test set in each fold (None = auto)
            gap: Number of samples to skip between train and test (prevents leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              time_column: str = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time-series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            time_column: Column to sort by (e.g., 'week', 'season')
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if time_column and time_column in X.columns:
            # Sort by time column
            sorted_indices = X.sort_values(time_column).index
            X_sorted = X.loc[sorted_indices]
        else:
            X_sorted = X
            sorted_indices = X.index
        
        splits = []
        for train_idx, test_idx in self.tscv.split(X_sorted):
            # Map back to original indices
            train_indices = sorted_indices[train_idx]
            test_indices = sorted_indices[test_idx]
            splits.append((train_indices, test_indices))
        
        return splits
    
    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series,
                       time_column: str = None) -> Dict[str, List[float]]:
        """
        Perform time-series cross-validation and return metrics.
        
        Returns:
            Dictionary with lists of metrics for each fold
        """
        results = {
            'train_rmse': [], 'test_rmse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X, y, time_column)):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]
            
            # Clone and fit model
            model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            model_clone.fit(X_train, y_train)
            
            # Predictions
            train_pred = model_clone.predict(X_train)
            test_pred = model_clone.predict(X_test)
            
            # Metrics
            results['train_rmse'].append(np.sqrt(mean_squared_error(y_train, train_pred)))
            results['test_rmse'].append(np.sqrt(mean_squared_error(y_test, test_pred)))
            results['train_mae'].append(mean_absolute_error(y_train, train_pred))
            results['test_mae'].append(mean_absolute_error(y_test, test_pred))
            results['train_r2'].append(r2_score(y_train, train_pred))
            results['test_r2'].append(r2_score(y_test, test_pred))
        
        return results


# =============================================================================
# LSTM MODEL FOR SEQUENTIAL PREDICTIONS
# =============================================================================
class LSTMFantasyModel:
    """
    LSTM-based model for fantasy football predictions.
    
    Captures sequential patterns in player performance (hot streaks, 
    cold streaks, form trends) that traditional models miss.
    """
    
    def __init__(self, sequence_length: int = 5, lstm_units: int = 64,
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Args:
            sequence_length: Number of past games to use for prediction
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = None
        self.is_fitted = False
        
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
    
    def _build_model(self, n_features: int) -> Sequential:
        """Build the LSTM architecture."""
        model = Sequential([
            # First LSTM layer with return sequences for stacking
            LSTM(self.lstm_units, activation='tanh', 
                 input_shape=(self.sequence_length, n_features),
                 return_sequences=True),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(self.lstm_units // 2, activation='tanh'),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(self.dropout_rate / 2),
            Dense(16, activation='relu'),
            
            # Output layer
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None,
                          player_ids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        For each player, creates sequences of past N games to predict the next game.
        """
        X_seq, y_seq = [], []
        
        if player_ids is not None:
            # Group by player
            unique_players = np.unique(player_ids)
            for player_id in unique_players:
                mask = player_ids == player_id
                X_player = X[mask]
                y_player = y[mask] if y is not None else None
                
                # Create sequences for this player
                for i in range(len(X_player) - self.sequence_length):
                    X_seq.append(X_player[i:i + self.sequence_length])
                    if y_player is not None:
                        y_seq.append(y_player[i + self.sequence_length])
        else:
            # No player grouping - treat as single sequence
            for i in range(len(X) - self.sequence_length):
                X_seq.append(X[i:i + self.sequence_length])
                if y is not None:
                    y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq) if y is not None else None
        
        return X_seq, y_seq
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            player_id_column: str = 'player_id',
            epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.2, verbose: int = 0) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X: Feature DataFrame (should be sorted by player and time)
            y: Target Series
            player_id_column: Column name for player IDs
            epochs: Maximum training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Store feature columns
        self.feature_columns = [c for c in X.columns if c != player_id_column]
        
        # Extract player IDs if available
        player_ids = X[player_id_column].values if player_id_column in X.columns else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X[self.feature_columns])
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values, player_ids)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.sequence_length}")
        
        # Build model
        n_features = X_seq.shape[2]
        self.model = self._build_model(n_features)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }
    
    def predict(self, X: pd.DataFrame, player_id_column: str = 'player_id') -> np.ndarray:
        """Make predictions for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        player_ids = X[player_id_column].values if player_id_column in X.columns else None
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        X_seq, _ = self._create_sequences(X_scaled, player_ids=player_ids)
        
        if len(X_seq) == 0:
            return np.array([])
        
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(path / 'lstm_model.keras')
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }, path / 'lstm_config.joblib')
    
    @classmethod
    def load(cls, path: Path) -> 'LSTMFantasyModel':
        """Load model from disk."""
        path = Path(path)
        
        config = joblib.load(path / 'lstm_config.joblib')
        model = cls(
            sequence_length=config['sequence_length'],
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )
        model.model = load_model(path / 'lstm_model.keras')
        model.scaler = config['scaler']
        model.feature_columns = config['feature_columns']
        model.is_fitted = True
        
        return model


# =============================================================================
# STACKED ENSEMBLE MODEL
# =============================================================================
class StackedEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacked ensemble combining XGBoost, LightGBM, and Ridge regression.
    
    Uses a meta-learner to combine predictions from base models,
    often achieving better generalization than any single model.
    """
    
    def __init__(self, 
                 xgb_params: Dict = None,
                 lgb_params: Dict = None,
                 ridge_alpha: float = 1.0,
                 meta_learner: str = 'ridge',
                 use_cv_predictions: bool = True,
                 n_cv_folds: int = 5):
        """
        Args:
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters
            ridge_alpha: Ridge regression alpha
            meta_learner: Type of meta-learner ('ridge', 'xgb', 'average')
            use_cv_predictions: Use CV predictions for meta-learner training
            n_cv_folds: Number of CV folds for generating meta-features
        """
        self.xgb_params = xgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        self.lgb_params = lgb_params or {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        
        self.ridge_alpha = ridge_alpha
        self.meta_learner = meta_learner
        self.use_cv_predictions = use_cv_predictions
        self.n_cv_folds = n_cv_folds
    
    def _init_base_models(self):
        """Initialize base models."""
        models = {}
        
        if HAS_XGBOOST:
            models['xgb'] = xgb.XGBRegressor(**self.xgb_params)
        
        if HAS_LIGHTGBM:
            models['lgb'] = lgb.LGBMRegressor(**self.lgb_params)
        
        models['ridge'] = Ridge(alpha=self.ridge_alpha)
        
        return models
    
    def _get_cv_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions for meta-learner training."""
        n_samples = len(X)
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            oof_predictions = np.zeros(n_samples)
            oof_mask = np.zeros(n_samples, dtype=bool)
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Clone and fit
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train, y_train)
                
                oof_predictions[val_idx] = model_clone.predict(X_val)
                oof_mask[val_idx] = True
            
            # For samples not in any validation fold, use full model prediction
            if not oof_mask.all():
                model.fit(X, y)
                oof_predictions[~oof_mask] = model.predict(X[~oof_mask])
            
            meta_features[:, model_idx] = oof_predictions
        
        return meta_features
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'StackedEnsemble':
        """
        Fit the stacked ensemble.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_array)
        
        # Initialize base models
        self.base_models = self._init_base_models()
        
        if self.use_cv_predictions:
            # Generate out-of-fold predictions for meta-learner
            meta_features = self._get_cv_predictions(X_scaled, y_array)
        else:
            # Use direct predictions (risk of overfitting)
            meta_features = np.zeros((len(X_scaled), len(self.base_models)))
            for idx, (name, model) in enumerate(self.base_models.items()):
                model.fit(X_scaled, y_array)
                meta_features[:, idx] = model.predict(X_scaled)
        
        # Fit base models on full data
        for name, model in self.base_models.items():
            model.fit(X_scaled, y_array)
        
        # Fit meta-learner
        if self.meta_learner == 'ridge':
            self.meta_learner_ = Ridge(alpha=1.0)
        elif self.meta_learner == 'xgb' and HAS_XGBOOST:
            self.meta_learner_ = xgb.XGBRegressor(n_estimators=50, max_depth=3)
        else:
            self.meta_learner_ = None  # Use simple average

        if self.meta_learner_ is not None:
            self.meta_learner_.fit(meta_features, y_array)
        
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the stacked ensemble."""
        if not getattr(self, 'is_fitted_', False):
            raise ValueError("Model must be fitted before prediction")

        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler_.transform(X_array)
        
        # Get base model predictions
        meta_features = np.zeros((len(X_scaled), len(self.base_models)))
        for idx, (name, model) in enumerate(self.base_models.items()):
            meta_features[:, idx] = model.predict(X_scaled)
        
        # Combine with meta-learner
        if self.meta_learner_ is not None:
            return self.meta_learner_.predict(meta_features)
        else:
            # Simple average
            return meta_features.mean(axis=1)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from base models."""
        importance = {}
        
        if 'xgb' in self.base_models:
            importance['xgb'] = self.base_models['xgb'].feature_importances_
        
        if 'lgb' in self.base_models:
            importance['lgb'] = self.base_models['lgb'].feature_importances_
        
        return importance
    
    def save(self, path: Path):
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'base_models': self.base_models,
            'meta_learner_': self.meta_learner_,
            'scaler_': self.scaler_,
            'params': self.get_params()
        }, path / 'stacked_ensemble.joblib')
    
    @classmethod
    def load(cls, path: Path) -> 'StackedEnsemble':
        """Load ensemble from disk."""
        path = Path(path)
        data = joblib.load(path / 'stacked_ensemble.joblib')
        
        model = cls(**data['params'])
        model.base_models = data['base_models']
        model.meta_learner_ = data.get('meta_learner_', data.get('meta_learner'))
        model.scaler_ = data.get('scaler_', data.get('scaler'))
        model.is_fitted_ = True
        
        return model


# =============================================================================
# VEGAS LINES INTEGRATION
# =============================================================================
class VegasLinesIntegration:
    """
    Integration for Vegas betting lines data.
    
    Vegas lines are highly predictive because they incorporate:
    - Injury information
    - Weather forecasts
    - Historical matchup data
    - Sharp money movements
    
    Free data sources:
    - The Odds API (free tier available)
    - ESPN (game lines)
    - Pro Football Reference (historical)
    """
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: API key for The Odds API (optional, for live data)
        """
        self.api_key = api_key
        self.cached_lines = {}
    
    def fetch_game_lines(self, season: int, week: int) -> pd.DataFrame:
        """
        Fetch Vegas lines for a specific week.
        
        Returns DataFrame with:
        - team, opponent
        - spread, over_under, moneyline
        - implied_team_total
        """
        # TODO: Integrate real Vegas lines from The Odds API or ESPN
        # This is a placeholder that generates example lines for demonstration
        
        teams = ['KC', 'BUF', 'PHI', 'SF', 'DAL', 'MIA', 'BAL', 'CIN',
                 'DET', 'JAX', 'LAC', 'NYJ', 'MIN', 'SEA', 'CLE', 'GB',
                 'PIT', 'TB', 'NO', 'LAR', 'DEN', 'LV', 'IND', 'ATL',
                 'TEN', 'CHI', 'NYG', 'WAS', 'ARI', 'CAR', 'NE', 'HOU']
        
        # Generate realistic lines based on team strength
        np.random.seed(season * 100 + week)
        
        records = []
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                break
            
            team1, team2 = teams[i], teams[i + 1]
            
            # Generate spread (-14 to +14)
            spread = np.random.uniform(-14, 14)
            
            # Generate over/under (35 to 55)
            over_under = np.random.uniform(38, 52)
            
            # Calculate implied team totals
            team1_implied = (over_under / 2) - (spread / 2)
            team2_implied = (over_under / 2) + (spread / 2)
            
            records.append({
                'season': season, 'week': week,
                'team': team1, 'opponent': team2,
                'spread': round(spread, 1),
                'over_under': round(over_under, 1),
                'implied_team_total': round(team1_implied, 1),
                'is_home': True
            })
            
            records.append({
                'season': season, 'week': week,
                'team': team2, 'opponent': team1,
                'spread': round(-spread, 1),
                'over_under': round(over_under, 1),
                'implied_team_total': round(team2_implied, 1),
                'is_home': False
            })
        
        return pd.DataFrame(records)
    
    def add_vegas_features(self, df: pd.DataFrame, 
                           team_column: str = 'team',
                           season_column: str = 'season',
                           week_column: str = 'week') -> pd.DataFrame:
        """
        Add Vegas-derived features to player DataFrame.
        
        Features added:
        - implied_team_total: Expected points for player's team
        - spread: Point spread (negative = favorite)
        - over_under: Total game points expected
        - game_script_score: Likelihood of pass-heavy game script
        """
        result = df.copy()
        
        # Get unique season/week combinations
        season_weeks = df[[season_column, week_column]].drop_duplicates()
        
        all_lines = []
        for _, row in season_weeks.iterrows():
            lines = self.fetch_game_lines(row[season_column], row[week_column])
            all_lines.append(lines)
        
        if all_lines:
            vegas_df = pd.concat(all_lines, ignore_index=True)
            
            # Merge with player data
            result = result.merge(
                vegas_df[['season', 'week', 'team', 'spread', 'over_under', 'implied_team_total']],
                left_on=[season_column, week_column, team_column],
                right_on=['season', 'week', 'team'],
                how='left'
            )
            
            # Calculate game script score (higher = more likely to pass)
            # Teams that are underdogs tend to pass more
            result['game_script_score'] = result['spread'].fillna(0) * -1 + result['over_under'].fillna(45) / 10
            
            # Fill missing values with league averages
            result['spread'] = result['spread'].fillna(0)
            result['over_under'] = result['over_under'].fillna(45)
            result['implied_team_total'] = result['implied_team_total'].fillna(22.5)
            result['game_script_score'] = result['game_script_score'].fillna(4.5)
        
        return result
    
    def get_player_prop_estimate(self, player_name: str, stat_type: str,
                                 season: int, week: int) -> Dict[str, float]:
        """
        Estimate player prop line based on historical performance and Vegas lines.
        
        This is a simplified estimation - in production, you'd use actual prop lines.
        """
        # Placeholder - would integrate with actual prop data
        return {
            'line': 0,
            'over_odds': -110,
            'under_odds': -110,
            'confidence': 0.5
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def create_rolling_features(df: pd.DataFrame, 
                           target_column: str = 'fantasy_points',
                           player_column: str = 'player_id',
                           windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """
    Create rolling average features for time-series modeling.
    
    Args:
        df: Player data sorted by time
        target_column: Column to create rolling features for
        player_column: Column to group by
        windows: List of window sizes
        
    Returns:
        DataFrame with rolling features added
    """
    result = df.copy()
    
    for window in windows:
        col_name = f'{target_column}_rolling_{window}'
        result[col_name] = result.groupby(player_column)[target_column].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        
        # Rolling std for consistency measure
        std_col = f'{target_column}_std_{window}'
        result[std_col] = result.groupby(player_column)[target_column].transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        )
    
    return result


def create_lag_features(df: pd.DataFrame,
                       target_column: str = 'fantasy_points',
                       player_column: str = 'player_id',
                       lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """
    Create lag features (previous game performance).
    """
    result = df.copy()
    
    for lag in lags:
        col_name = f'{target_column}_lag_{lag}'
        result[col_name] = result.groupby(player_column)[target_column].shift(lag)
    
    return result


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }

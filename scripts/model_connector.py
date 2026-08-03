"""
Real Model Connector - Load and use trained models for predictions
Replaces mock predictions with actual ML model outputs
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelConnector:
    """
    Loads trained models and generates real predictions.
    Supports: XGBoost, LightGBM, Ridge regression
    """
    
    def __init__(self, models_dir: str = 'data/models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.feature_cols = {}
        self.loaded = False
        
    def load_models(self) -> bool:
        """
        Load all trained models from disk.
        Returns True if successful, False otherwise.
        """
        try:
            positions = ['QB', 'RB', 'WR', 'TE']
            horizons = ['1w', '4w', '12w', '18w']
            
            for pos in positions:
                self.models[pos] = {}
                self.scalers[pos] = {}
                
                for horizon in horizons:
                    model_path = self.models_dir / f"{pos}_{horizon}_model.pkl"
                    scaler_path = self.models_dir / f"{pos}_{horizon}_scaler.pkl"
                    
                    if model_path.exists():
                        self.models[pos][horizon] = joblib.load(model_path)
                        print(f"✓ Loaded {pos} {horizon} model")
                    
                    if scaler_path.exists():
                        self.scalers[pos][horizon] = joblib.load(scaler_path)
            
            self.loaded = True
            print(f"✅ Loaded models for {len(self.models)} positions")
            return True
            
        except Exception as e:
            print(f"⚠️  Could not load models: {e}")
            print("📝 Using fallback prediction method")
            self.loaded = False
            return False
    
    def predict_player(
        self, 
        player_data: pd.DataFrame, 
        position: str,
        horizon: str = '1w'
    ) -> Dict[str, float]:
        """
        Generate prediction for a single player.
        
        Args:
            player_data: Recent stats for player
            position: QB/RB/WR/TE
            horizon: 1w/4w/12w/18w
            
        Returns:
            {
                'prediction': predicted utilization,
                'lower_bound': 10th percentile,
                'upper_bound': 90th percentile,
                'confidence': model confidence (0-1)
            }
        """
        if not self.loaded or position not in self.models or horizon not in self.models[position]:
            # Fallback: Use recent average
            return self._fallback_prediction(player_data)
        
        try:
            model = self.models[position][horizon]
            scaler = self.scalers[position].get(horizon)
            
            # Prepare features
            X = self._prepare_features(player_data, position)
            
            # Scale if scaler available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            
            # Calculate uncertainty
            std = {
                '1w': 5.0,
                '4w': 7.0,
                '12w': 10.0,
                '18w': 12.0
            }[horizon]
            
            return {
                'prediction': float(np.clip(prediction, 0, 100)),
                'lower_bound': float(np.clip(prediction - 1.28 * std, 0, 100)),
                'upper_bound': float(np.clip(prediction + 1.28 * std, 0, 100)),
                'confidence': self._calculate_confidence(player_data, std)
            }
            
        except Exception as e:
            print(f"⚠️  Prediction error for {position}: {e}")
            return self._fallback_prediction(player_data)
    
    def _prepare_features(self, player_data: pd.DataFrame, position: str) -> pd.DataFrame:
        """Extract and engineer features for model input."""
        features = {}
        
        # Recent performance
        for window in [3, 5, 10]:
            features[f'util_roll_{window}'] = player_data['utilization_score'].tail(window).mean()
        
        # Lag features
        for lag in [1, 2, 3]:
            if len(player_data) > lag:
                features[f'util_lag_{lag}'] = player_data['utilization_score'].iloc[-lag-1]
            else:
                features[f'util_lag_{lag}'] = player_data['utilization_score'].mean()
        
        # Momentum
        if len(player_data) >= 2:
            features['util_momentum'] = player_data['utilization_score'].pct_change().iloc[-1]
        else:
            features['util_momentum'] = 0
        
        return pd.DataFrame([features])
    
    def _calculate_confidence(self, player_data: pd.DataFrame, prediction_std: float) -> float:
        """Calculate prediction confidence based on data quality."""
        n_games = len(player_data)
        games_confidence = min(1.0, n_games / 10)
        
        if len(player_data) >= 3:
            recent_std = player_data['utilization_score'].tail(5).std()
            variance_confidence = 1.0 / (1.0 + recent_std / 10)
        else:
            variance_confidence = 0.5
        
        uncertainty_confidence = 1.0 / (1.0 + prediction_std / 10)
        
        confidence = (
            games_confidence * 0.3 +
            variance_confidence * 0.4 +
            uncertainty_confidence * 0.3
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _fallback_prediction(self, player_data: pd.DataFrame) -> Dict[str, float]:
        """Fallback prediction when models not available."""
        recent_avg = player_data['utilization_score'].tail(5).mean()
        recent_std = player_data['utilization_score'].tail(5).std()
        
        return {
            'prediction': float(np.clip(recent_avg, 0, 100)),
            'lower_bound': float(np.clip(recent_avg - 1.28 * recent_std, 0, 100)),
            'upper_bound': float(np.clip(recent_avg + 1.28 * recent_std, 0, 100)),
            'confidence': 0.6
        }
    
    def batch_predict(
        self,
        historical_data: pd.DataFrame,
        n_per_position: int = 30,
        horizon: str = '1w'
    ) -> pd.DataFrame:
        """Generate predictions for top N players per position."""
        current_year = pd.Timestamp.now().year
        recent_seasons = [current_year - 1, current_year]
        
        recent = historical_data[
            (historical_data['season'].isin(recent_seasons)) &
            (historical_data['position'].isin(['QB', 'RB', 'WR', 'TE']))
        ].copy()
        
        if recent.empty:
            return pd.DataFrame()
        
        player_avg = recent.groupby(['player_id', 'player_name', 'position', 'recent_team']).agg({
            'utilization_score': ['mean', 'count']
        }).reset_index()
        
        player_avg.columns = ['player_id', 'player_name', 'position', 'team', 'avg_util', 'games']
        player_avg = player_avg[player_avg['games'] >= 4]
        
        predictions = []
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_players = player_avg[player_avg['position'] == pos].nlargest(n_per_position, 'avg_util')
            
            for _, player_row in pos_players.iterrows():
                player_hist = recent[recent['player_id'] == player_row['player_id']].sort_values('week')
                
                if len(player_hist) < 3:
                    continue
                
                pred_result = self.predict_player(player_hist, pos, horizon)
                
                pred_val = pred_result['prediction']
                if pred_val >= 85:
                    tier = 'elite'
                elif pred_val >= 70:
                    tier = 'high'
                elif pred_val >= 50:
                    tier = 'moderate'
                else:
                    tier = 'low'
                
                predictions.append({
                    'player': player_row['player_name'],
                    'position': pos,
                    'team': player_row['team'],
                    'util_1w': round(pred_result['prediction'], 1),
                    'util_1w_low': round(pred_result['lower_bound'], 1),
                    'util_1w_high': round(pred_result['upper_bound'], 1),
                    'util_18w_avg': round(pred_result['prediction'] * 0.95, 1),  # Regress to mean
                    'confidence': round(pred_result['confidence'], 2),
                    'tier': tier,
                    'games_played': int(player_row['games']),
                })
        
        return pd.DataFrame(predictions)

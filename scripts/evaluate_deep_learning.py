"""
Evaluate Deep Learning vs Traditional ML Models

This script:
1. Tests LSTM model on multi-year data
2. Compares against Gradient Boosting baseline
3. Evaluates performance across different training window sizes
4. Documents results

Usage:
    python scripts/evaluate_deep_learning.py
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not installed. Install with: pip install tensorflow")


def load_data():
    """Load the main predictions data."""
    data_path = Path(__file__).parent.parent / "data" / "daily_predictions.parquet"
    df = pd.read_parquet(data_path)
    try:
        from src.utils.leakage import drop_leakage_columns
        df = drop_leakage_columns(df)
    except Exception:
        pass
    print(f"Loaded {len(df):,} records")
    print(f"Seasons: {int(df['season'].min())}-{int(df['season'].max())}")
    return df


def prepare_sequences(df, position, sequence_length=4):
    """
    Prepare sequential data for LSTM.
    
    Creates sequences of player performance for time-series prediction.
    """
    pos_df = df[df['position'] == position].copy()
    
    # Core features
    feature_cols = ['fantasy_points', 'rushing_yards', 'receiving_yards', 
                    'receptions', 'targets']
    feature_cols = [c for c in feature_cols if c in pos_df.columns]
    
    # Sort by player and time
    pos_df = pos_df.sort_values(['player_id', 'season', 'week'])
    
    # Create target: next week's fantasy points
    pos_df['next_fp'] = pos_df.groupby('player_id')['fantasy_points'].shift(-1)
    pos_df = pos_df.dropna(subset=['next_fp'] + feature_cols)
    
    # Scale features
    scaler = StandardScaler()
    pos_df[feature_cols] = scaler.fit_transform(pos_df[feature_cols])
    
    # Create sequences
    X_sequences = []
    y_targets = []
    seasons = []
    
    for player_id in pos_df['player_id'].unique():
        player_data = pos_df[pos_df['player_id'] == player_id]
        
        if len(player_data) < sequence_length + 1:
            continue
        
        for i in range(len(player_data) - sequence_length):
            seq = player_data.iloc[i:i+sequence_length][feature_cols].values
            target = player_data.iloc[i+sequence_length]['next_fp']
            season = player_data.iloc[i+sequence_length]['season']
            
            X_sequences.append(seq)
            y_targets.append(target)
            seasons.append(season)
    
    return np.array(X_sequences), np.array(y_targets), np.array(seasons), scaler


def prepare_flat_features(df, position):
    """Prepare flat features for traditional ML."""
    pos_df = df[df['position'] == position].copy()
    
    feature_cols = ['fantasy_points', 'rushing_yards', 'receiving_yards', 
                    'receptions', 'targets', 'fp_rolling_3']
    feature_cols = [c for c in feature_cols if c in pos_df.columns]
    
    pos_df = pos_df.sort_values(['player_id', 'season', 'week'])
    pos_df['next_fp'] = pos_df.groupby('player_id')['fantasy_points'].shift(-1)
    pos_df = pos_df.dropna(subset=['next_fp'] + feature_cols)
    
    X = pos_df[feature_cols].fillna(0).values
    y = pos_df['next_fp'].values
    seasons = pos_df['season'].values
    
    return X, y, seasons


def build_lstm_model(input_shape, units=64):
    """Build LSTM model."""
    model = Sequential([
        LSTM(units, activation='tanh', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(units // 2, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def evaluate_models(df, position, train_years_list=None, test_year=None):
    """
    Evaluate LSTM vs Gradient Boosting across different training windows.
    test_year defaults to config TRAINING_END_YEAR_DEFAULT (current NFL season).
    """
    if train_years_list is None:
        train_years_list = [2, 4, 6, 10]
    if test_year is None:
        from config.settings import TRAINING_END_YEAR_DEFAULT
        test_year = TRAINING_END_YEAR_DEFAULT
    results = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating models for {position}")
    print('='*60)
    
    for train_years in train_years_list:
        train_start = test_year - train_years
        
        print(f"\n--- Training window: {train_years} years ({train_start}-{test_year-1}) ---")
        
        # Gradient Boosting (baseline)
        X_flat, y_flat, seasons_flat = prepare_flat_features(df, position)
        
        train_mask = (seasons_flat >= train_start) & (seasons_flat < test_year)
        test_mask = seasons_flat == test_year
        
        X_train_gb, y_train_gb = X_flat[train_mask], y_flat[train_mask]
        X_test_gb, y_test_gb = X_flat[test_mask], y_flat[test_mask]
        
        if len(X_train_gb) < 50 or len(X_test_gb) < 20:
            print(f"  Skipping - insufficient data")
            continue
        
        # Train Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        gb_model.fit(X_train_gb, y_train_gb)
        gb_pred = gb_model.predict(X_test_gb)
        
        gb_rmse = np.sqrt(mean_squared_error(y_test_gb, gb_pred))
        gb_mae = mean_absolute_error(y_test_gb, gb_pred)
        gb_corr = np.corrcoef(gb_pred, y_test_gb)[0, 1]
        
        print(f"  Gradient Boosting: RMSE={gb_rmse:.2f}, MAE={gb_mae:.2f}, Corr={gb_corr:.3f}")
        
        result = {
            'position': position,
            'train_years': train_years,
            'train_start': train_start,
            'test_year': test_year,
            'n_train': len(X_train_gb),
            'n_test': len(X_test_gb),
            'gb_rmse': round(gb_rmse, 2),
            'gb_mae': round(gb_mae, 2),
            'gb_corr': round(gb_corr, 3),
        }
        
        # LSTM (if TensorFlow available)
        if HAS_TENSORFLOW:
            try:
                X_seq, y_seq, seasons_seq, _ = prepare_sequences(df, position, sequence_length=4)
                
                train_mask_seq = (seasons_seq >= train_start) & (seasons_seq < test_year)
                test_mask_seq = seasons_seq == test_year
                
                X_train_lstm = X_seq[train_mask_seq]
                y_train_lstm = y_seq[train_mask_seq]
                X_test_lstm = X_seq[test_mask_seq]
                y_test_lstm = y_seq[test_mask_seq]
                
                if len(X_train_lstm) >= 50 and len(X_test_lstm) >= 20:
                    # Build and train LSTM
                    lstm_model = build_lstm_model(
                        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                        units=64
                    )
                    
                    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
                    
                    lstm_model.fit(
                        X_train_lstm, y_train_lstm,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
                    
                    lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))
                    lstm_mae = mean_absolute_error(y_test_lstm, lstm_pred)
                    lstm_corr = np.corrcoef(lstm_pred, y_test_lstm)[0, 1]
                    
                    print(f"  LSTM:              RMSE={lstm_rmse:.2f}, MAE={lstm_mae:.2f}, Corr={lstm_corr:.3f}")
                    
                    result['lstm_rmse'] = round(lstm_rmse, 2)
                    result['lstm_mae'] = round(lstm_mae, 2)
                    result['lstm_corr'] = round(lstm_corr, 3)
                    
                    # Comparison
                    if lstm_corr > gb_corr:
                        result['winner'] = 'LSTM'
                        print(f"  ✅ LSTM wins by {lstm_corr - gb_corr:.3f}")
                    else:
                        result['winner'] = 'GradientBoosting'
                        print(f"  ✅ Gradient Boosting wins by {gb_corr - lstm_corr:.3f}")
                else:
                    print(f"  LSTM: Skipped - insufficient sequence data")
                    result['lstm_rmse'] = None
                    result['lstm_mae'] = None
                    result['lstm_corr'] = None
                    result['winner'] = 'GradientBoosting'
                    
            except Exception as e:
                print(f"  LSTM Error: {e}")
                result['lstm_rmse'] = None
                result['lstm_mae'] = None
                result['lstm_corr'] = None
                result['winner'] = 'GradientBoosting'
        else:
            result['lstm_rmse'] = None
            result['lstm_mae'] = None
            result['lstm_corr'] = None
            result['winner'] = 'GradientBoosting (LSTM not available)'
        
        results.append(result)
    
    return results


def run_evaluation():
    """Run full evaluation."""
    print("="*70)
    print("DEEP LEARNING VS TRADITIONAL ML EVALUATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = load_data()
    
    all_results = []
    
    from config.settings import CURRENT_NFL_SEASON
    test_year = CURRENT_NFL_SEASON
    for position in ['RB', 'WR', 'TE', 'QB']:
        results = evaluate_models(
            df, position, 
            train_years_list=[2, 4, 6, 10],
            test_year=test_year
        )
        all_results.extend(results)
    
    # Save results
    data_dir = Path(__file__).parent.parent / "data"
    
    results_df = pd.DataFrame(all_results)
    csv_path = data_dir / "deep_learning_evaluation.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Results saved to: {csv_path}")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    if HAS_TENSORFLOW and 'lstm_corr' in results_df.columns:
        # Compare average performance
        valid_results = results_df.dropna(subset=['lstm_corr'])
        
        if len(valid_results) > 0:
            avg_gb = valid_results['gb_corr'].mean()
            avg_lstm = valid_results['lstm_corr'].mean()
            
            print(f"\nAverage Test Correlation:")
            print(f"  Gradient Boosting: {avg_gb:.3f}")
            print(f"  LSTM:              {avg_lstm:.3f}")
            
            if avg_lstm > avg_gb:
                print(f"\n  ✅ LSTM outperforms by {avg_lstm - avg_gb:.3f}")
            else:
                print(f"\n  ✅ Gradient Boosting outperforms by {avg_gb - avg_lstm:.3f}")
            
            # By position
            print("\nBy Position:")
            for pos in ['RB', 'WR', 'TE', 'QB']:
                pos_results = valid_results[valid_results['position'] == pos]
                if len(pos_results) > 0:
                    gb_avg = pos_results['gb_corr'].mean()
                    lstm_avg = pos_results['lstm_corr'].mean()
                    winner = 'LSTM' if lstm_avg > gb_avg else 'GB'
                    print(f"  {pos}: GB={gb_avg:.3f}, LSTM={lstm_avg:.3f} → {winner}")
            
            # By training years
            print("\nBy Training Years:")
            for years in [2, 4, 6, 10]:
                year_results = valid_results[valid_results['train_years'] == years]
                if len(year_results) > 0:
                    gb_avg = year_results['gb_corr'].mean()
                    lstm_avg = year_results['lstm_corr'].mean()
                    winner = 'LSTM' if lstm_avg > gb_avg else 'GB'
                    print(f"  {years} years: GB={gb_avg:.3f}, LSTM={lstm_avg:.3f} → {winner}")
    else:
        print("\nLSTM results not available (TensorFlow not installed)")
        print("Gradient Boosting results:")
        print(results_df[['position', 'train_years', 'gb_corr']].to_string(index=False))
    
    # Save summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'tensorflow_available': HAS_TENSORFLOW,
        'test_year': test_year,
        'positions_tested': ['RB', 'WR', 'TE', 'QB'],
        'training_windows': [2, 4, 6, 10],
        'results': all_results
    }
    
    json_path = data_dir / "deep_learning_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📋 Summary saved to: {json_path}")
    
    return results_df


if __name__ == "__main__":
    results = run_evaluation()
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

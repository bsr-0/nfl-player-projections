"""
Optimize Training Years for All Models

This script:
1. Tests different training window sizes (1-20 years)
2. Evaluates each window using cross-validation
3. Finds optimal training years for each position
4. Documents results in CSV and JSON formats
5. Updates model configuration to use optimal years

Usage:
    python scripts/optimize_training_years.py
"""
import os
import sys
import ssl
import certifi
import json
from pathlib import Path
from datetime import datetime

# Fix SSL certificates
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


def load_data():
    """Load the main predictions data."""
    data_path = Path(__file__).parent.parent / "data" / "daily_predictions.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        try:
            from src.utils.leakage import drop_leakage_columns
            df = drop_leakage_columns(df)
        except Exception:
            pass
        print(f"Loaded {len(df):,} records")
        print(f"Seasons: {sorted(df['season'].unique())}")
        return df
    raise FileNotFoundError(f"Data file not found: {data_path}")


def prepare_features(df, position):
    """Prepare features for a given position."""
    pos_df = df[df['position'] == position].copy()
    
    # Core features available across all years
    feature_cols = []
    
    # Check which columns exist
    potential_features = [
        'fantasy_points', 'rushing_yards', 'receiving_yards', 'receptions',
        'targets', 'rushing_attempts', 'passing_yards', 'passing_attempts',
        'fp_rolling_3', 'fp_rolling_5', 'rushing_yards_rolling_3',
        'receiving_yards_rolling_3', 'targets_rolling_3'
    ]
    
    for col in potential_features:
        if col in pos_df.columns:
            feature_cols.append(col)
    
    if not feature_cols:
        return None, None, None
    
    # Create target: next week's fantasy points
    pos_df = pos_df.sort_values(['player_id', 'season', 'week'])
    pos_df['next_fp'] = pos_df.groupby('player_id')['fantasy_points'].shift(-1)
    
    # Drop rows without target
    pos_df = pos_df.dropna(subset=['next_fp'] + feature_cols)
    
    if len(pos_df) < 100:
        return None, None, None
    
    X = pos_df[feature_cols].fillna(0)
    y = pos_df['next_fp']
    seasons = pos_df['season']
    
    return X, y, seasons


def evaluate_training_window(df, position, train_years, test_year=None):
    """
    Evaluate a specific training window size.

    Args:
        df: Full dataset
        position: Position to evaluate (RB, WR, TE, QB)
        train_years: Number of years to use for training
        test_year: Year to use for testing (default: config TRAINING_END_YEAR_DEFAULT)

    Returns:
        dict with metrics (correlation, rmse, n_samples)
    """
    if test_year is None:
        from config.settings import TRAINING_END_YEAR_DEFAULT
        test_year = TRAINING_END_YEAR_DEFAULT
    X, y, seasons = prepare_features(df, position)
    
    if X is None:
        return None
    
    # Define train/test split
    train_start = test_year - train_years
    train_mask = (seasons >= train_start) & (seasons < test_year)
    test_mask = seasons == test_year
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) < 50 or len(X_test) < 20:
        return None
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    try:
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_corr = np.corrcoef(train_pred, y_train)[0, 1]
        test_corr = np.corrcoef(test_pred, y_test)[0, 1]
        
        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
        
        return {
            'train_years': train_years,
            'train_start': train_start,
            'train_end': test_year - 1,
            'test_year': test_year,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'train_correlation': round(train_corr, 4),
            'test_correlation': round(test_corr, 4),
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'overfit_ratio': round(train_corr / test_corr, 3) if test_corr > 0 else None
        }
    except Exception as e:
        print(f"  Error training model: {e}")
        return None


def find_optimal_training_years(df, position, max_years=20, test_year=None):
    """
    Find optimal number of training years for a position.
    test_year defaults to config TRAINING_END_YEAR_DEFAULT (current NFL season).
    """
    if test_year is None:
        from config.settings import TRAINING_END_YEAR_DEFAULT
        test_year = TRAINING_END_YEAR_DEFAULT
    print(f"\n{'='*60}")
    print(f"Optimizing training years for {position}")
    print('='*60)
    
    results = []
    
    # Get available years
    available_years = sorted(df['season'].unique())
    min_year = int(min(available_years))
    
    # Calculate max possible training years
    max_possible = test_year - min_year
    max_years = min(max_years, max_possible)
    
    print(f"Testing training windows: 1 to {max_years} years")
    print(f"Available data: {min_year}-{int(max(available_years))}")
    
    for train_years in range(1, max_years + 1):
        result = evaluate_training_window(df, position, train_years, test_year)
        
        if result:
            result['position'] = position
            results.append(result)
            
            status = "✓" if result['test_correlation'] > 0.5 else "○"
            print(f"  {status} {train_years:2d} years ({result['train_start']}-{result['train_end']}): "
                  f"test_r={result['test_correlation']:.3f}, "
                  f"train_r={result['train_correlation']:.3f}, "
                  f"n={result['n_train']}")
    
    if not results:
        print(f"  No valid results for {position}")
        return None, []
    
    # Find optimal (best test correlation)
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_correlation'].idxmax()
    optimal = results_df.loc[best_idx]
    
    print(f"\n  ✅ OPTIMAL: {int(optimal['train_years'])} years "
          f"(test_r={optimal['test_correlation']:.3f})")
    
    return optimal.to_dict(), results


def run_full_optimization():
    """Run optimization for all positions and save results."""
    
    print("="*70)
    print("TRAINING YEARS OPTIMIZATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    # Positions to optimize
    positions = ['RB', 'WR', 'TE', 'QB']
    
    all_results = []
    optimal_config = {}
    
    for position in positions:
        optimal, results = find_optimal_training_years(df, position, max_years=15)
        
        if optimal:
            optimal_config[position] = {
                'optimal_years': int(optimal['train_years']),
                'test_correlation': optimal['test_correlation'],
                'train_correlation': optimal['train_correlation'],
                'overfit_ratio': optimal['overfit_ratio']
            }
            all_results.extend(results)
    
    # Save detailed results to CSV
    data_dir = Path(__file__).parent.parent / "data"
    
    results_df = pd.DataFrame(all_results)
    csv_path = data_dir / "training_years_optimization.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Detailed results saved to: {csv_path}")
    
    # Save optimal config to JSON (test_year = config default used in optimization)
    from config.settings import TRAINING_END_YEAR_DEFAULT
    config = {
        'generated_at': datetime.now().isoformat(),
        'test_year': TRAINING_END_YEAR_DEFAULT,
        'positions': optimal_config,
        'recommendation': 'Use these training years for each position model'
    }
    
    json_path = data_dir / "optimal_training_config.json"
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📋 Optimal config saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print("\nOptimal Training Years by Position:")
    print("-"*50)
    
    for pos, cfg in optimal_config.items():
        print(f"  {pos}: {cfg['optimal_years']} years "
              f"(test_r={cfg['test_correlation']:.3f}, "
              f"overfit={cfg['overfit_ratio']:.2f}x)")
    
    # Overall recommendation
    avg_years = np.mean([cfg['optimal_years'] for cfg in optimal_config.values()])
    print(f"\n  Average optimal: {avg_years:.1f} years")
    
    if avg_years <= 3:
        print("  💡 Recommendation: Use RECENT data (1-3 years)")
    elif avg_years <= 6:
        print("  💡 Recommendation: Use MODERATE history (4-6 years)")
    else:
        print("  💡 Recommendation: Use EXTENDED history (7+ years)")
    
    return optimal_config, results_df


def update_model_config(optimal_config):
    """Update model configuration to use optimal training years."""
    
    config_path = Path(__file__).parent.parent / "data" / "model_config.json"
    
    # Load existing config or create new
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with optimal training years
    config['training_years'] = {
        pos: cfg['optimal_years'] 
        for pos, cfg in optimal_config.items()
    }
    config['last_optimized'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Model config updated: {config_path}")
    return config


if __name__ == "__main__":
    optimal_config, results_df = run_full_optimization()
    update_model_config(optimal_config)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

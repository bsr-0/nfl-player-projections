"""
Training script for advanced ML models (model comparison pipeline).

Trains and evaluates:
1. Stacked Ensemble (XGBoost + LightGBM + Ridge)
2. Proper walk-forward backtesting (no data leakage)
3. Utilization-based feature engineering
4. Uncertainty quantification

Optimized for speed - runs in ~1-2 minutes.

NOTE: This script's backtest_results use a separate pipeline (Ridge/GBM/XGB/LGB
with its own feature list and CV folds), not the production ensemble from
train.py. For production backtest metrics (same pipeline as serve), the app's
primary source is: (1) train.py (runs backtest after training and writes
advanced_model_results.json), or (2) python -m src.evaluation.backtester.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from config.settings import POSITIONS, DATA_DIR, MODELS_DIR
from src.utils.database import DatabaseManager
from src.features.utilization import engineer_all_features, UtilizationCalculator
from src.features.qb_features import add_qb_features, QB_FEATURE_PATTERNS
from src.models.robust_validation import RobustTimeSeriesCV, validate_no_leakage
from src.data.external_data import add_external_features
from src.features.multiweek_features import add_multiweek_features
from src.features.season_long_features import add_season_long_features

# Check available libraries
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


def load_and_prepare_data(positions: list = None, min_games: int = 4) -> pd.DataFrame:
    """Load player data and prepare features for training."""
    print("📊 Loading data from database...")
    start = time.time()
    
    db = DatabaseManager()
    df = db.get_all_players_for_training(min_games=min_games)
    
    if df.empty:
        raise ValueError("No data found in database. Run data loader first.")
    
    positions = positions or POSITIONS
    df = df[df['position'].isin(positions)]
    
    print(f"   Loaded {len(df)} records for positions: {positions}")
    print(f"   Seasons: {sorted(df['season'].unique())}")
    print(f"   Time: {time.time() - start:.1f}s")
    
    # Sort by player and time for sequential features
    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features using the utilization module."""
    print("\n🔧 Engineering features...")
    start = time.time()
    
    # Use the comprehensive feature engineering
    df = engineer_all_features(df)
    
    # Add QB-specific features
    df = add_qb_features(df)
    
    # Add external data features (injury, defense, weather, Vegas)
    print("\n📊 Adding external data features...")
    df = add_external_features(df)
    
    # Add multi-week features (schedule strength, aggregation, injury probability)
    print("\n📅 Adding multi-week features...")
    df = add_multiweek_features(df, horizons=[1, 5, 18])
    
    # Add season-long features (age curves, games projection, rookie projections, ADP)
    print("\n🏆 Adding season-long features...")
    df = add_season_long_features(df)
    
    # Additional game context features
    df['games_played_season'] = df.groupby(['player_id', 'season']).cumcount() + 1
    df['is_home'] = (df['home_away'] == 'home').astype(int)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"   Created {len(df.columns)} total features")
    print(f"   Time: {time.time() - start:.1f}s")
    
    return df


def get_feature_columns(df: pd.DataFrame, position: str = None) -> list:
    """Get list of feature columns for modeling.
    
    CRITICAL: Only use features that would be KNOWN at prediction time:
    - Historical/lagged stats (previous games)
    - Rolling averages (from past games)
    - Player characteristics that don't change week-to-week
    
    EXCLUDE all current-week stats (these are what we're predicting!)
    """
    # Only include these specific feature patterns (known before game)
    allowed_patterns = [
        '_lag_',      # Lagged features (previous weeks)
        '_rolling_',  # Rolling averages (historical)
        'rolling_',   # Rolling stats
        '_trend',     # Trend features (historical)
        '_avg',       # Season averages (historical, shifted)
        'games_played',  # Games played so far
        'is_home',    # Known before game
        'consistency_score',  # Calculated from history
        'weekly_volatility',  # Calculated from history
        'coefficient_of_variation',
        'boom_bust_range',
        'confidence_score',
        # External data features (known before game)
        'injury_score',      # Player injury status
        'is_injured',        # Binary injury flag
        'opp_defense_rank',  # Opponent defense ranking
        'opp_matchup_score', # Matchup difficulty score
        'opp_pts_allowed',   # Points opponent allows
        'is_dome',           # Dome stadium
        'is_outdoor',        # Outdoor game
        'weather_score',     # Weather impact score
        'implied_team_total', # Vegas implied points
        'game_total',        # Vegas over/under
        'spread',            # Point spread
        # Multi-week features (known before game)
        'sos_next_',         # Schedule strength for N weeks
        'sos_rank_next_',    # Schedule strength rank
        'favorable_matchups_next_', # Count of favorable matchups
        'expected_games_next_',  # Expected games to play
        'projection_',       # Multi-week projections
        'floor_',            # Multi-week floor
        'ceiling_',          # Multi-week ceiling
        'injury_prob_next_', # Injury probability
        'expected_missed_games_', # Expected missed games
        'injury_risk_score_', # Injury risk score
        'opp_defense_strength', # Opponent defense strength
        # Season-long features (known before game)
        'age',               # Player age
        'age_factor',        # Age-based performance factor
        'age_expected_games', # Expected games based on age
        'decline_rate',      # Year-over-year decline rate
        'years_from_peak',   # Distance from peak age
        'is_in_prime',       # In prime performance window
        'projected_games',   # Projected games for season
        'historical_gpg',    # Historical games played rate
        'is_rookie',         # Rookie indicator
        'rookie_projected',  # Rookie projections
        'rookie_weight',     # Rookie projection weight
        # NOTE: position_rank REMOVED - it's calculated from fantasy_points (data leakage)
        # Use projected_position_rank instead (based on rolling averages)
        'projected_position_rank', # Projected rank based on historical performance
        'season_position_rank', # Season-long position rank (from prior weeks)
        'estimated_adp',     # Estimated ADP
        'projected_adp',     # Projected ADP
        'adp_value',         # ADP value score
        'positional_scarcity', # Position scarcity factor
        # Playoff/Week features (known before game)
        'is_playoff_week',   # Binary: playoff game
        'is_wild_card',      # Wild card round
        'is_divisional',     # Divisional round
        'is_conference_championship', # Conference championship
        'is_super_bowl',     # Super Bowl
        'season_phase',      # 0=Early, 1=Mid, 2=Late, 3=Playoff
        'week_normalized',   # Week position (0-1)
        'is_late_season',    # Weeks 15-18
        'is_high_stakes',    # Late season + playoffs
    ]
    
    # Add QB-specific patterns if position is QB
    if position == 'QB':
        allowed_patterns.extend([
            'is_mobile_qb',  # Derived from history
        ])
        # Add QB rolling/lag features
        for qb_feat in QB_FEATURE_PATTERNS:
            allowed_patterns.append(f'{qb_feat}_rolling')
            allowed_patterns.append(f'{qb_feat}_lag')
    
    # These are the ONLY safe features to use
    feature_cols = []
    for col in df.columns:
        if df[col].dtype not in ['int64', 'float64']:
            continue
        
        # Check if column matches any allowed pattern
        is_allowed = any(pattern in col for pattern in allowed_patterns)
        
        if is_allowed:
            feature_cols.append(col)
    
    return feature_cols


def train_and_backtest(df: pd.DataFrame, position: str, fast_mode: bool = True) -> dict:
    """Train model with proper walk-forward backtesting."""
    print(f"\n🔄 Training & Backtesting for {position}...")
    start = time.time()
    
    pos_df = df[df['position'] == position].copy()
    
    if len(pos_df) < 100:
        print(f"   ⚠️ Not enough data for {position} ({len(pos_df)} samples)")
        return None
    
    # Get feature columns (only historical/lagged features)
    feature_cols = get_feature_columns(pos_df, position=position)
    print(f"   Using {len(feature_cols)} features")
    
    # Use season-based validation (train on past seasons, test on future)
    seasons = sorted(pos_df['season'].unique())
    
    if len(seasons) < 2:
        print(f"   ⚠️ Need at least 2 seasons for backtesting")
        return None
    
    # Use robust cross-validation with proper scaling
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Validate no data leakage
    leakage_check = validate_no_leakage(pos_df, feature_cols)
    if not leakage_check['passed']:
        print(f"   ⚠️ Data leakage detected: {leakage_check['errors']}")
    
    # Cross-validation across multiple test seasons
    n_folds = min(len(seasons) - 1, 3)  # Up to 3 folds
    test_seasons = seasons[-n_folds:]
    
    models_config = {
        'Ridge': (Ridge, {'alpha': 1.0}),
        'GBM': (GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
    }
    
    if HAS_XGBOOST:
        models_config['XGBoost'] = (xgb.XGBRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42})
    
    if HAS_LIGHTGBM:
        models_config['LightGBM'] = (lgb.LGBMRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42, 'verbose': -1})
    
    # Collect results across folds
    model_results = {name: {'rmse': [], 'mae': [], 'r2': []} for name in models_config}
    
    for test_season in test_seasons:
        train_seasons = [s for s in seasons if s < test_season]
        if not train_seasons:
            continue
        
        train_df = pos_df[pos_df['season'].isin(train_seasons)]
        test_df = pos_df[pos_df['season'] == test_season]
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['fantasy_points']
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['fantasy_points']
        
        # Proper scaling: fit on train, transform both
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for name, (model_class, params) in models_config.items():
            model = model_class(**params)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
            model_results[name]['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))
            model_results[name]['mae'].append(mean_absolute_error(y_test, preds))
            model_results[name]['r2'].append(r2_score(y_test, preds))
    
    # Average across folds
    comparison = []
    best_rmse = float('inf')
    best_name = None
    
    for name, metrics in model_results.items():
        if not metrics['rmse']:
            continue
        avg_rmse = np.mean(metrics['rmse'])
        avg_mae = np.mean(metrics['mae'])
        avg_r2 = np.mean(metrics['r2'])
        
        comparison.append({'model': name, 'rmse': avg_rmse, 'mae': avg_mae, 'r2': avg_r2})
        
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_name = name
            best_metrics = {'rmse': avg_rmse, 'mae': avg_mae, 'r2': avg_r2}
    
    results = {
        'position': position,
        'best_model': best_name,
        'n_samples': len(pos_df),
        'n_features': len(feature_cols),
        'train_size': len(train_df),
        'test_size': len(test_df),
        'rmse': best_metrics['rmse'],
        'mae': best_metrics['mae'],
        'r2': best_metrics['r2'],
        'comparison': comparison,
        'time_seconds': time.time() - start
    }
    
    print(f"   ✅ {position} Best: {best_name} - RMSE: {results['rmse']:.2f}, R²: {results['r2']:.3f}")
    print(f"      Time: {results['time_seconds']:.1f}s")
    
    return results


def train_final_model(df: pd.DataFrame, position: str) -> dict:
    """Train final production model on all data."""
    print(f"\n� Training final model for {position}...")
    
    pos_df = df[df['position'] == position].copy()
    feature_cols = get_feature_columns(pos_df)
    
    X = pos_df[feature_cols].fillna(0)
    y = pos_df['fantasy_points']
    
    # Use GradientBoosting as default (good balance of speed/accuracy)
    from sklearn.ensemble import GradientBoostingRegressor
    
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save model
    import joblib
    model_path = MODELS_DIR / f'model_{position.lower()}_production.joblib'
    joblib.dump({
        'model': model,
        'feature_cols': feature_cols,
        'position': position
    }, model_path)
    
    print(f"   ✅ Saved to {model_path}")
    
    return {'model_path': str(model_path), 'n_features': len(feature_cols)}


def main(fast_mode: bool = True):
    """Main training pipeline."""
    total_start = time.time()
    
    print("=" * 60)
    print("🏈 Advanced Model Training Pipeline")
    print("=" * 60)
    print(f"Mode: {'Fast' if fast_mode else 'Thorough'}")
    
    # Check available libraries
    print("\n📦 Available libraries:")
    print(f"   XGBoost: {'✅' if HAS_XGBOOST else '❌'}")
    print(f"   LightGBM: {'✅' if HAS_LIGHTGBM else '❌'}")
    
    # Load and prepare data
    df = load_and_prepare_data()
    df = engineer_features(df)
    
    # Results storage
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'backtest_results': {},
        'final_models': {}
    }
    
    # Train models for each position
    for position in POSITIONS:
        print(f"\n{'='*60}")
        print(f"Position: {position}")
        print(f"{'='*60}")
        
        # Backtest to find best model
        backtest_result = train_and_backtest(df, position, fast_mode=fast_mode)
        if backtest_result:
            all_results['backtest_results'][position] = backtest_result
        
        # Train final production model
        final_result = train_final_model(df, position)
        all_results['final_models'][position] = final_result
    
    # Save results (app-compatible format; overwrites advanced_model_results.json).
    # For production-ensemble backtest metrics, prefer train.py + backtest or
    # python -m src.evaluation.backtester.
    results_path = DATA_DIR / 'advanced_model_results.json'
    
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
    
    all_results = convert_numpy(all_results)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {results_path}")
    
    # Summary
    print("\n📊 Summary:")
    for position in POSITIONS:
        if position in all_results['backtest_results']:
            r = all_results['backtest_results'][position]
            print(f"   {position}: {r['best_model']} - RMSE: {r['rmse']:.2f}, R²: {r['r2']:.3f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train advanced fantasy football models')
    parser.add_argument('--thorough', action='store_true', 
                        help='Run thorough week-by-week backtesting (slower but more accurate)')
    args = parser.parse_args()
    
    main(fast_mode=not args.thorough)

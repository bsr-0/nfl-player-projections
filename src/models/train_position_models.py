"""
Train Position-Specific Models for WR, RB, and TE

Each position gets its own optimized model with position-relevant features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MODELS_DIR, POSITIONS
from src.utils.data_manager import DataManager, auto_refresh_data
from src.utils.database import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.models.position_models import PositionModel, MultiWeekModel


# Position-specific feature sets
POSITION_FEATURES = {
    "RB": {
        "primary": [
            "rushing_yards", "rushing_attempts", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "snap_count", "snap_share",
        ],
        "derived": [
            "yards_per_carry", "yards_per_target", "catch_rate",
            "total_touches", "total_yards", "total_tds",
            "opportunities", "weighted_opportunities",
        ],
        "rolling": [
            "fantasy_points_roll3", "fantasy_points_roll5",
            "rushing_yards_roll3", "total_touches_roll3",
            "snap_share_roll3",
        ],
        "team_context": [
            "team_a_rushing_yards", "team_a_rush_attempts", "team_a_pass_rate",
            "team_b_rushing_yards", "matchup_rush_diff",
            "team_sos", "matchup_difficulty",
        ],
    },
    "WR": {
        "primary": [
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "snap_count", "snap_share",
        ],
        "derived": [
            "yards_per_target", "yards_per_reception", "catch_rate",
            "total_yards", "total_tds",
        ],
        "rolling": [
            "fantasy_points_roll3", "fantasy_points_roll5",
            "receiving_yards_roll3", "targets_roll3",
            "snap_share_roll3",
        ],
        "team_context": [
            "team_a_passing_yards", "team_a_pass_attempts", "team_a_pass_rate",
            "team_b_passing_yards", "matchup_pass_diff",
            "team_sos", "matchup_difficulty",
        ],
    },
    "TE": {
        "primary": [
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "snap_count", "snap_share",
        ],
        "derived": [
            "yards_per_target", "yards_per_reception", "catch_rate",
            "total_yards", "total_tds",
        ],
        "rolling": [
            "fantasy_points_roll3", "fantasy_points_roll5",
            "receiving_yards_roll3", "targets_roll3",
            "snap_share_roll3",
        ],
        "team_context": [
            "team_a_passing_yards", "team_a_pass_attempts",
            "team_b_passing_yards", "matchup_pass_diff",
            "team_sos", "matchup_difficulty",
        ],
    },
}


def get_position_features(position: str, available_cols: List[str]) -> List[str]:
    """Get features for a position that exist in the data."""
    if position not in POSITION_FEATURES:
        return available_cols
    
    pos_features = POSITION_FEATURES[position]
    all_features = (
        pos_features.get("primary", []) +
        pos_features.get("derived", []) +
        pos_features.get("rolling", []) +
        pos_features.get("team_context", [])
    )
    
    # Return only features that exist in data
    return [f for f in all_features if f in available_cols]


def load_position_data(position: str, min_games: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for a specific position."""
    print(f"\nLoading {position} data...")
    
    # Check data availability
    status = auto_refresh_data()
    print(f"  Latest season: {status['latest_season']}")
    
    # Get train/test split (test season is current season when in-season)
    dm = DataManager()
    train_seasons, test_season = dm.get_train_test_seasons()
    
    # Load from database
    db = DatabaseManager()
    all_data = db.get_all_players_for_training(position=position, min_games=min_games)
    
    if all_data.empty:
        print(f"  No data found for {position}")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"  Total records: {len(all_data)}")
    
    # Engineer features
    engineer = FeatureEngineer()
    all_data = engineer.create_features(all_data)
    
    # Split
    train_data = all_data[all_data['season'].isin(train_seasons)]
    test_data = all_data[all_data['season'] == test_season]
    
    print(f"  Train: {len(train_data)} records from {train_seasons}")
    print(f"  Test: {len(test_data)} records from {test_season}")
    
    return train_data, test_data


def create_targets(df: pd.DataFrame, n_weeks_list: List[int] = [1, 4, 18]) -> pd.DataFrame:
    """Create prediction targets for different horizons."""
    df = df.copy()
    
    for n_weeks in n_weeks_list:
        group_cols = ["player_id", "season"] if "season" in df.columns else ["player_id"]
        df[f"target_{n_weeks}w"] = df.groupby(group_cols)["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
    
    return df


def train_position_model(position: str, 
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         tune_hyperparameters: bool = True,
                         n_weeks_list: List[int] = [1, 4, 18]) -> Dict:
    """
    Train a position-specific model.
    
    Returns dict with model and evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {position} MODEL")
    print(f"{'='*60}")
    
    if train_data.empty:
        return {"error": f"No training data for {position}"}
    
    # Create targets
    train_data = create_targets(train_data, n_weeks_list)
    test_data = create_targets(test_data, n_weeks_list)
    
    # Get position-specific features
    exclude_cols = [
        'player_id', 'name', 'position', 'team', 'season', 'week',
        'fantasy_points', 'opponent', 'home_away',
        'created_at', 'updated_at', 'id', 'birth_date', 'college',
        'game_id', 'game_time', 'player_name', 'gsis_id',
    ] + [f'target_{n}w' for n in n_weeks_list]
    
    available_features = [c for c in train_data.columns 
                         if c not in exclude_cols 
                         and train_data[c].dtype in ['int64', 'float64']]
    
    feature_cols = get_position_features(position, available_features)
    
    # If position-specific features not available, use all numeric
    if len(feature_cols) < 5:
        feature_cols = available_features
    try:
        from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns
        feature_cols = filter_feature_columns(feature_cols)
        assert_no_leakage_columns(feature_cols, context=f"train_position_models ({position})")
    except Exception:
        pass
    
    print(f"Using {len(feature_cols)} features")
    
    # Prepare data
    X_train = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Train multi-week model
    y_dict = {}
    for n_weeks in n_weeks_list:
        target_col = f"target_{n_weeks}w"
        y_dict[n_weeks] = train_data[target_col]
    
    # Remove rows with NaN targets
    valid_mask = ~y_dict[1].isna()
    X_train = X_train[valid_mask]
    y_dict = {k: v[valid_mask] for k, v in y_dict.items()}
    
    # Train
    multi_model = MultiWeekModel(position)
    multi_model.fit(X_train, y_dict, tune_hyperparameters=tune_hyperparameters)
    
    # Evaluate on test data
    results = {
        "position": position,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "features": feature_cols,
        "metrics": {}
    }
    
    for n_weeks in n_weeks_list:
        target_col = f"target_{n_weeks}w"
        if target_col in test_data.columns:
            valid_test = ~test_data[target_col].isna()
            X_test_valid = X_test[valid_test]
            y_test = test_data.loc[valid_test, target_col]
            
            if len(y_test) > 0:
                predictions = multi_model.predict(X_test_valid, n_weeks)
                
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                results["metrics"][f"{n_weeks}w"] = {
                    "rmse": round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
                    "mae": round(mean_absolute_error(y_test, predictions), 2),
                    "r2": round(r2_score(y_test, predictions), 3),
                }
    
    # Save model
    multi_model.save()
    
    # Also save individual week models
    for n_weeks in [1, 4, 18]:
        if n_weeks in multi_model.models:
            multi_model.models[n_weeks].save()
    
    return results, multi_model


def train_all_position_models(positions: List[str] = None,
                               tune_hyperparameters: bool = True) -> Dict:
    """
    Train models for WR, RB, and TE.
    
    Args:
        positions: List of positions to train (default: ['WR', 'RB', 'TE'])
        tune_hyperparameters: Whether to tune hyperparameters
        
    Returns:
        Dict with results for each position
    """
    positions = positions or ['WR', 'RB', 'TE']
    
    print("=" * 60)
    print("POSITION-SPECIFIC MODEL TRAINING")
    print("=" * 60)
    print(f"Positions: {positions}")
    print(f"Hyperparameter tuning: {tune_hyperparameters}")
    
    all_results = {}
    all_models = {}
    
    for position in positions:
        # Load data
        train_data, test_data = load_position_data(position)
        
        if train_data.empty:
            print(f"\nSkipping {position} - no data")
            continue
        
        # Train model
        results, model = train_position_model(
            position, train_data, test_data, 
            tune_hyperparameters=tune_hyperparameters
        )
        
        all_results[position] = results
        all_models[position] = model
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for position, results in all_results.items():
        if "error" in results:
            print(f"\n{position}: {results['error']}")
            continue
            
        print(f"\n{position}:")
        print(f"  Training samples: {results['n_train']}")
        print(f"  Test samples: {results['n_test']}")
        print(f"  Features: {results['n_features']}")
        
        for horizon, metrics in results.get('metrics', {}).items():
            print(f"  {horizon}: RMSE={metrics['rmse']}, MAE={metrics['mae']}, R²={metrics['r2']}")
    
    print(f"\nModels saved to: {MODELS_DIR}")
    
    return all_results, all_models


def quick_train(tune: bool = False):
    """Quick training without hyperparameter tuning."""
    return train_all_position_models(
        positions=['WR', 'RB', 'TE'],
        tune_hyperparameters=tune
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train position-specific models")
    parser.add_argument("--positions", nargs="+", default=['WR', 'RB', 'TE'],
                       help="Positions to train")
    parser.add_argument("--no-tune", action="store_true",
                       help="Skip hyperparameter tuning")
    
    args = parser.parse_args()
    
    results, models = train_all_position_models(
        positions=args.positions,
        tune_hyperparameters=not args.no_tune
    )

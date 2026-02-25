"""
Model Diagnostics and Improvement Module

Provides systematic tools to:
1. Diagnose why model performance is poor
2. Identify specific weak points (positions, player types, weeks)
3. Test improvement strategies
4. Compare model versions
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import POSITIONS, DATA_DIR


class ModelDiagnostics:
    """
    Diagnoses model weaknesses and identifies areas for improvement.
    """
    
    def __init__(self):
        self.diagnostics_dir = DATA_DIR / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_diagnostics(self, df: pd.DataFrame,
                              pred_col: str = 'predicted_points',
                              actual_col: str = 'fantasy_points') -> Dict:
        """
        Run comprehensive diagnostics to find model weaknesses.
        
        Returns detailed breakdown of where model fails.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": self._calculate_metrics(df[actual_col], df[pred_col]),
            "by_position": {},
            "by_week": {},
            "by_score_range": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # Analyze by position
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            if len(pos_df) >= 20:
                results["by_position"][pos] = self._calculate_metrics(
                    pos_df[actual_col], pos_df[pred_col]
                )
        
        # Analyze by week (early vs late season)
        for week in sorted(df['week'].unique()):
            week_df = df[df['week'] == week]
            if len(week_df) >= 10:
                results["by_week"][int(week)] = self._calculate_metrics(
                    week_df[actual_col], week_df[pred_col]
                )
        
        # Analyze by score range (low, medium, high scorers)
        results["by_score_range"] = self._analyze_by_score_range(df, actual_col, pred_col)
        
        # Error analysis
        results["error_analysis"] = self._analyze_errors(df, actual_col, pred_col)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        return results
    
    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate standard metrics."""
        mask = ~(actual.isna() | predicted.isna())
        actual, predicted = actual[mask], predicted[mask]
        
        if len(actual) < 2:
            return {"error": "Insufficient data"}
        
        return {
            "n": len(actual),
            "rmse": round(np.sqrt(mean_squared_error(actual, predicted)), 2),
            "mae": round(mean_absolute_error(actual, predicted), 2),
            "r2": round(r2_score(actual, predicted), 3),
            "correlation": round(actual.corr(predicted), 3),
            "mean_error": round((predicted - actual).mean(), 2),  # Bias
            "std_error": round((predicted - actual).std(), 2),
        }
    
    def _analyze_by_score_range(self, df: pd.DataFrame, 
                                 actual_col: str, pred_col: str) -> Dict:
        """Analyze accuracy for different scoring ranges."""
        df = df.copy()
        
        # Define score ranges
        df['score_range'] = pd.cut(
            df[actual_col], 
            bins=[0, 8, 15, 25, 100],
            labels=['Low (0-8)', 'Medium (8-15)', 'High (15-25)', 'Elite (25+)']
        )
        
        results = {}
        for range_name in df['score_range'].unique():
            if pd.isna(range_name):
                continue
            range_df = df[df['score_range'] == range_name]
            if len(range_df) >= 10:
                results[str(range_name)] = self._calculate_metrics(
                    range_df[actual_col], range_df[pred_col]
                )
        
        return results
    
    def _analyze_errors(self, df: pd.DataFrame,
                        actual_col: str, pred_col: str) -> Dict:
        """Deep analysis of prediction errors."""
        df = df.copy()
        df['error'] = df[pred_col] - df[actual_col]
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = df['error'] / df[actual_col].replace(0, np.nan) * 100
        
        return {
            "mean_error": round(df['error'].mean(), 2),
            "median_error": round(df['error'].median(), 2),
            "std_error": round(df['error'].std(), 2),
            "skewness": round(df['error'].skew(), 2),
            "over_predictions_pct": round((df['error'] > 0).mean() * 100, 1),
            "under_predictions_pct": round((df['error'] < 0).mean() * 100, 1),
            "large_misses_pct": round((df['abs_error'] > 10).mean() * 100, 1),
            "worst_over_prediction": round(df['error'].max(), 1),
            "worst_under_prediction": round(df['error'].min(), 1),
        }
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on diagnostics."""
        recommendations = []
        
        overall = results["overall_metrics"]
        
        # Check overall performance
        if overall.get("r2", 0) < 0.15:
            recommendations.append("CRITICAL: R² < 0.15 - Model has weak predictive power. Consider:")
            recommendations.append("  → Add more features (team stats, matchup data)")
            recommendations.append("  → Try different model architectures")
            recommendations.append("  → Check for data quality issues")
        
        if overall.get("r2", 0) < 0.3:
            recommendations.append("Model R² is moderate. Room for improvement.")
        
        # Check for bias
        error_analysis = results.get("error_analysis", {})
        mean_error = error_analysis.get("mean_error", 0)
        if abs(mean_error) > 2:
            direction = "over" if mean_error > 0 else "under"
            recommendations.append(f"BIAS DETECTED: Model tends to {direction}-predict by {abs(mean_error):.1f} pts")
            recommendations.append("  → Apply bias correction to predictions")
        
        # Check position-specific issues
        by_position = results.get("by_position", {})
        if by_position:
            worst_pos = min(by_position.items(), key=lambda x: x[1].get('r2', 0))
            best_pos = max(by_position.items(), key=lambda x: x[1].get('r2', 0))
            
            if worst_pos[1].get('r2', 0) < 0.1:
                recommendations.append(f"WEAK POSITION: {worst_pos[0]} has very low accuracy (R²={worst_pos[1]['r2']})")
                recommendations.append(f"  → Add {worst_pos[0]}-specific features")
                recommendations.append(f"  → Train separate model for {worst_pos[0]}")
        
        # Check score range issues
        by_range = results.get("by_score_range", {})
        for range_name, metrics in by_range.items():
            if metrics.get('r2', 0) < 0.1:
                recommendations.append(f"WEAK RANGE: Poor accuracy for {range_name} scorers")
        
        # Check for high variance
        if error_analysis.get("std_error", 0) > 8:
            recommendations.append("HIGH VARIANCE: Predictions are inconsistent")
            recommendations.append("  → Consider ensemble methods to reduce variance")
        
        # Check for large misses
        if error_analysis.get("large_misses_pct", 0) > 15:
            recommendations.append(f"TOO MANY LARGE MISSES: {error_analysis['large_misses_pct']}% of predictions off by >10 pts")
            recommendations.append("  → Add outlier detection")
            recommendations.append("  → Cap extreme predictions")
        
        if not recommendations:
            recommendations.append("Model performance looks good! Minor tuning may still help.")
        
        return recommendations
    
    def print_diagnostics(self, results: Dict):
        """Print diagnostics in readable format."""
        print("=" * 60)
        print("MODEL DIAGNOSTICS REPORT")
        print("=" * 60)
        
        # Overall
        o = results["overall_metrics"]
        print(f"\nOVERALL: R²={o['r2']}, RMSE={o['rmse']}, MAE={o['mae']}")
        print(f"  Bias: {o['mean_error']:+.2f} pts, Std: {o['std_error']:.2f}")
        
        # By position
        print("\nBY POSITION:")
        for pos, m in results["by_position"].items():
            status = "✓" if m['r2'] > 0.2 else "⚠" if m['r2'] > 0.1 else "✗"
            print(f"  {status} {pos}: R²={m['r2']}, RMSE={m['rmse']}")
        
        # By score range
        print("\nBY SCORE RANGE:")
        for range_name, m in results["by_score_range"].items():
            status = "✓" if m['r2'] > 0.2 else "⚠" if m['r2'] > 0.1 else "✗"
            print(f"  {status} {range_name}: R²={m['r2']}, n={m['n']}")
        
        # Recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS:")
        print("=" * 60)
        for rec in results["recommendations"]:
            print(rec)


class ModelImprover:
    """
    Systematic model improvement through various strategies.
    """
    
    IMPROVEMENT_STRATEGIES = [
        "feature_engineering",
        "hyperparameter_tuning",
        "ensemble_methods",
        "bias_correction",
        "position_specific_models",
        "outlier_handling",
        "cross_validation",
    ]
    
    def __init__(self):
        self.results_dir = DATA_DIR / "improvement_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.improvement_log = []
    
    def run_improvement_pipeline(self, train_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  feature_cols: List[str],
                                  target_col: str = 'fantasy_points') -> Dict:
        """
        Run through improvement strategies and track results.
        """
        results = {
            "baseline": None,
            "improvements": {},
            "best_strategy": None,
            "best_improvement": 0
        }
        
        # Establish baseline
        print("Establishing baseline...")
        baseline = self._evaluate_baseline(train_df, test_df, feature_cols, target_col)
        results["baseline"] = baseline
        print(f"  Baseline RMSE: {baseline['rmse']:.2f}")
        
        # Try each improvement strategy
        for strategy in self.IMPROVEMENT_STRATEGIES:
            print(f"\nTrying: {strategy}...")
            try:
                improvement = self._try_strategy(
                    strategy, train_df, test_df, feature_cols, target_col, baseline
                )
                results["improvements"][strategy] = improvement
                
                if improvement.get("rmse_improvement", 0) > results["best_improvement"]:
                    results["best_improvement"] = improvement["rmse_improvement"]
                    results["best_strategy"] = strategy
                    
                print(f"  RMSE: {improvement.get('rmse', 'N/A')}, Improvement: {improvement.get('rmse_improvement', 0):.1f}%")
            except Exception as e:
                print(f"  Failed: {e}")
                results["improvements"][strategy] = {"error": str(e)}
        
        # Summary
        print("\n" + "=" * 60)
        print("IMPROVEMENT SUMMARY")
        print("=" * 60)
        print(f"Baseline RMSE: {baseline['rmse']:.2f}")
        if results["best_strategy"]:
            print(f"Best Strategy: {results['best_strategy']}")
            print(f"Best Improvement: {results['best_improvement']:.1f}%")
        
        return results
    
    def _evaluate_baseline(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                           feature_cols: List[str], target_col: str) -> Dict:
        """Evaluate baseline model performance."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        # Simple gradient boosting baseline
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions)
        }
    
    def _try_strategy(self, strategy: str, train_df: pd.DataFrame,
                      test_df: pd.DataFrame, feature_cols: List[str],
                      target_col: str, baseline: Dict) -> Dict:
        """Try a specific improvement strategy."""
        
        if strategy == "feature_engineering":
            return self._try_feature_engineering(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "hyperparameter_tuning":
            return self._try_hyperparameter_tuning(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "ensemble_methods":
            return self._try_ensemble(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "bias_correction":
            return self._try_bias_correction(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "position_specific_models":
            return self._try_position_specific(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "outlier_handling":
            return self._try_outlier_handling(train_df, test_df, feature_cols, target_col, baseline)
        elif strategy == "cross_validation":
            return self._try_cross_validation(train_df, feature_cols, target_col, baseline)
        else:
            return {"error": f"Unknown strategy: {strategy}"}
    
    def _try_feature_engineering(self, train_df, test_df, feature_cols, target_col, baseline):
        """Try adding interaction features."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        train = train_df.copy()
        test = test_df.copy()
        
        # Add interaction features if base features exist
        new_features = []
        if 'rushing_yards' in train.columns and 'rushing_attempts' in train.columns:
            train['ypc'] = train['rushing_yards'] / train['rushing_attempts'].replace(0, 1)
            test['ypc'] = test['rushing_yards'] / test['rushing_attempts'].replace(0, 1)
            new_features.append('ypc')
        
        if 'receiving_yards' in train.columns and 'targets' in train.columns:
            train['ypt'] = train['receiving_yards'] / train['targets'].replace(0, 1)
            test['ypt'] = test['receiving_yards'] / test['targets'].replace(0, 1)
            new_features.append('ypt')
        
        all_features = feature_cols + new_features
        all_features = [f for f in all_features if f in train.columns]
        
        X_train = train[all_features].fillna(0)
        y_train = train[target_col]
        X_test = test[all_features].fillna(0)
        y_test = test[target_col]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return {
            "rmse": rmse,
            "r2": r2_score(y_test, predictions),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
            "new_features": new_features
        }
    
    def _try_hyperparameter_tuning(self, train_df, test_df, feature_cols, target_col, baseline):
        """Try different hyperparameters."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        # Try more trees and deeper
        model = GradientBoostingRegressor(
            n_estimators=200, 
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return {
            "rmse": rmse,
            "r2": r2_score(y_test, predictions),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
            "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}
        }
    
    def _try_ensemble(self, train_df, test_df, feature_cols, target_col, baseline):
        """Try ensemble of multiple models."""
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        # Train multiple models
        models = [
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            RandomForestRegressor(n_estimators=100, random_state=42),
            Ridge(alpha=1.0)
        ]
        
        predictions_list = []
        for model in models:
            model.fit(X_train, y_train)
            predictions_list.append(model.predict(X_test))
        
        # Average predictions
        ensemble_pred = np.mean(predictions_list, axis=0)
        
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        return {
            "rmse": rmse,
            "r2": r2_score(y_test, ensemble_pred),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
            "n_models": len(models)
        }
    
    def _try_bias_correction(self, train_df, test_df, feature_cols, target_col, baseline):
        """Apply bias correction to predictions."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get training predictions to estimate bias
        train_pred = model.predict(X_train)
        bias = (y_train - train_pred).mean()
        
        # Apply bias correction to test predictions
        predictions = model.predict(X_test) + bias
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return {
            "rmse": rmse,
            "r2": r2_score(y_test, predictions),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
            "bias_correction": bias
        }
    
    def _try_position_specific(self, train_df, test_df, feature_cols, target_col, baseline):
        """Train separate models per position."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        all_predictions = []
        all_actuals = []
        
        for pos in train_df['position'].unique():
            train_pos = train_df[train_df['position'] == pos]
            test_pos = test_df[test_df['position'] == pos]
            
            if len(train_pos) < 50 or len(test_pos) < 10:
                continue
            
            X_train = train_pos[feature_cols].fillna(0)
            y_train = train_pos[target_col]
            X_test = test_pos[feature_cols].fillna(0)
            y_test = test_pos[target_col]
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            all_predictions.extend(model.predict(X_test))
            all_actuals.extend(y_test)
        
        if not all_predictions:
            return {"error": "Insufficient data for position-specific models"}
        
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        return {
            "rmse": rmse,
            "r2": r2_score(all_actuals, all_predictions),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
        }
    
    def _try_outlier_handling(self, train_df, test_df, feature_cols, target_col, baseline):
        """Handle outliers in training data."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        train = train_df.copy()
        
        # Remove extreme outliers from training (>3 std from mean)
        mean_target = train[target_col].mean()
        std_target = train[target_col].std()
        train = train[np.abs(train[target_col] - mean_target) <= 3 * std_target]
        
        X_train = train[feature_cols].fillna(0)
        y_train = train[target_col]
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df[target_col]
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Cap extreme predictions
        predictions = np.clip(predictions, 0, y_train.quantile(0.99))
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return {
            "rmse": rmse,
            "r2": r2_score(y_test, predictions),
            "rmse_improvement": (baseline['rmse'] - rmse) / baseline['rmse'] * 100,
            "outliers_removed": len(train_df) - len(train)
        }
    
    def _try_cross_validation(self, train_df, feature_cols, target_col, baseline):
        """Use time-series cross-validation for better generalization."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        X = train_df[feature_cols].fillna(0)
        y = train_df[target_col]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=1)
        cv_rmse = -scores.mean()
        
        return {
            "cv_rmse": cv_rmse,
            "cv_std": scores.std(),
            "rmse_improvement": (baseline['rmse'] - cv_rmse) / baseline['rmse'] * 100 if cv_rmse < baseline['rmse'] else 0,
            "n_folds": 5
        }


def diagnose_and_improve(test_season: int = None):
    """
    One-command function to diagnose model issues and suggest improvements.
    """
    from src.utils.data_manager import DataManager
    from src.models.train import load_training_data, prepare_features
    
    print("=" * 60)
    print("MODEL DIAGNOSTICS & IMPROVEMENT")
    print("=" * 60)
    
    # Load data (test season is current season when in-season)
    dm = DataManager()
    train_seasons, actual_test_season = dm.get_train_test_seasons(test_season=test_season)
    
    print(f"\nLoading data...")
    print(f"  Train: {train_seasons}")
    print(f"  Test: {actual_test_season}")
    
    train_data, test_data, _, _ = load_training_data(test_season=actual_test_season)
    
    if train_data.empty or test_data.empty:
        print("Insufficient data for diagnostics")
        return None, None
    
    # Prepare features
    train_data = prepare_features(train_data)
    test_data = prepare_features(test_data)
    
    # Create simple predictions for diagnostics
    test_data['predicted_points'] = test_data.groupby('player_id')['fantasy_points'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    
    # Run diagnostics
    print("\n" + "=" * 60)
    diagnostics = ModelDiagnostics()
    diag_results = diagnostics.run_full_diagnostics(
        test_data.dropna(subset=['predicted_points']),
        'predicted_points',
        'fantasy_points'
    )
    diagnostics.print_diagnostics(diag_results)
    
    # Run improvement pipeline if performance is poor
    if diag_results["overall_metrics"].get("r2", 0) < 0.3:
        print("\n" + "=" * 60)
        print("RUNNING IMPROVEMENT PIPELINE...")
        print("=" * 60)
        
        # Get feature columns
        feature_cols = [c for c in train_data.columns if c not in 
                       ['player_id', 'name', 'position', 'team', 'season', 'week', 
                        'fantasy_points', 'opponent', 'home_away']]
        feature_cols = [c for c in feature_cols if train_data[c].dtype in ['int64', 'float64']]
        
        improver = ModelImprover()
        improve_results = improver.run_improvement_pipeline(
            train_data, test_data, feature_cols[:20], 'fantasy_points'  # Limit features for speed
        )
        
        return diag_results, improve_results
    
    return diag_results, None


if __name__ == "__main__":
    diag, improve = diagnose_and_improve()

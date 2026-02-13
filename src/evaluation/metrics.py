"""Model evaluation metrics and testing utilities."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    precision_score, recall_score,
)
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODEL_CONFIG, POSITIONS


def spearman_rank_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, top_n: Optional[int] = 50
) -> float:
    """Spearman rank correlation between predicted and actual (target rho > 0.65 for top-50)."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    if top_n is not None and len(y_true) > top_n:
        idx = np.argsort(y_pred)[-top_n:]
        y_true, y_pred = y_true[idx], y_pred[idx]
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if np.isfinite(r) else np.nan


def tier_classification_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray,
    tier_edges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    """Tier classification accuracy (Elite/Strong/Flex/Waiver). Target >75%."""
    if tier_edges is None:
        tier_edges = {
            "elite": (18, 100),
            "strong": (12, 18),
            "flex": (7, 12),
            "waiver": (0, 7),
        }
    def assign_tier(x, edges):
        for name, (lo, hi) in edges.items():
            if lo <= x < hi:
                return name
        return "waiver"
    pred_tiers = np.array([assign_tier(p, tier_edges) for p in y_pred])
    true_tiers = np.array([assign_tier(t, tier_edges) for t in y_true])
    return float(np.mean(pred_tiers == true_tiers))


def boom_bust_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
    boom_thresh: float = 20.0, bust_thresh: float = 5.0,
) -> Dict[str, float]:
    """Precision/recall for 20+ point weeks (boom) and <5 point weeks (bust)."""
    boom_true = (y_true >= boom_thresh).astype(int)
    bust_true = (y_true < bust_thresh).astype(int)
    boom_pred = (y_pred >= boom_thresh).astype(int)
    bust_pred = (y_pred < bust_thresh).astype(int)
    out = {}
    if boom_true.sum() > 0:
        out["boom_precision"] = float(precision_score(boom_true, boom_pred, zero_division=0))
        out["boom_recall"] = float(recall_score(boom_true, boom_pred, zero_division=0))
    if bust_true.sum() > 0:
        out["bust_precision"] = float(precision_score(bust_true, bust_pred, zero_division=0))
        out["bust_recall"] = float(recall_score(bust_true, bust_pred, zero_division=0))
    return out


def vor_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray,
    replacement_by_position: Optional[Dict[str, float]] = None,
    positions: Optional[np.ndarray] = None,
) -> float:
    """Rank correlation of predicted vs actual VOR (value over replacement)."""
    if replacement_by_position is None:
        replacement_by_position = {"QB": 12.0, "RB": 10.0, "WR": 10.0, "TE": 8.0}
    if positions is not None and len(positions) == len(y_true):
        vor_true = y_true - np.array([replacement_by_position.get(p, 10.0) for p in positions])
        vor_pred = y_pred - np.array([replacement_by_position.get(p, 10.0) for p in positions])
    else:
        rep = float(np.percentile(y_true, 25))
        vor_true = y_true - rep
        vor_pred = y_pred - rep
    if len(vor_true) < 2:
        return np.nan
    r, _ = spearmanr(vor_true, vor_pred)
    return float(r) if np.isfinite(r) else np.nan


class ModelEvaluator:
    """Comprehensive model evaluation for NFL predictions."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                       model_name: str = "model") -> Dict[str, float]:
        """
        Evaluate a single model with multiple metrics.
        
        Args:
            model: Trained model with predict method
            X: Feature DataFrame
            y: True target values
            model_name: Name for logging
            
        Returns:
            Dict of metric names to values
        """
        predictions = model.predict(X)
        
        metrics = {
            "mse": mean_squared_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "mae": mean_absolute_error(y, predictions),
            "r2": r2_score(y, predictions),
            "mape": self._safe_mape(y, predictions),
            "median_ae": np.median(np.abs(y - predictions)),
        }
        
        # Add percentile errors
        errors = np.abs(y - predictions)
        metrics["p90_error"] = np.percentile(errors, 90)
        metrics["p95_error"] = np.percentile(errors, 95)
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = None) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            n_splits: Number of CV splits
            
        Returns:
            Dict with CV metrics
        """
        n_splits = n_splits or MODEL_CONFIG["cv_folds"]
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Multiple scoring metrics
        mse_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        
        return {
            "cv_mse_mean": -mse_scores.mean(),
            "cv_mse_std": mse_scores.std(),
            "cv_rmse_mean": np.sqrt(-mse_scores.mean()),
            "cv_mae_mean": -mae_scores.mean(),
            "cv_mae_std": mae_scores.std(),
            "cv_r2_mean": r2_scores.mean(),
            "cv_r2_std": r2_scores.std(),
        }
    
    def evaluate_by_position(self, model, X: pd.DataFrame, y: pd.Series,
                             positions: pd.Series) -> Dict[str, Dict]:
        """Evaluate model performance by position."""
        results = {}
        
        for position in POSITIONS:
            mask = positions == position
            if mask.sum() == 0:
                continue
            
            X_pos = X[mask]
            y_pos = y[mask]
            
            predictions = model.predict(X_pos)
            
            results[position] = {
                "n_samples": mask.sum(),
                "rmse": np.sqrt(mean_squared_error(y_pos, predictions)),
                "mae": mean_absolute_error(y_pos, predictions),
                "r2": r2_score(y_pos, predictions),
            }
        
        return results
    
    def evaluate_by_utilization_tier(self, model, X: pd.DataFrame, y: pd.Series,
                                      utilization_scores: pd.Series) -> Dict[str, Dict]:
        """Evaluate model performance by utilization score tier."""
        tiers = {
            "elite (80+)": utilization_scores >= 80,
            "strong (70-79)": (utilization_scores >= 70) & (utilization_scores < 80),
            "average (60-69)": (utilization_scores >= 60) & (utilization_scores < 70),
            "below_avg (50-59)": (utilization_scores >= 50) & (utilization_scores < 60),
            "low (<50)": utilization_scores < 50,
        }
        
        results = {}
        
        for tier_name, mask in tiers.items():
            if mask.sum() == 0:
                continue
            
            X_tier = X[mask]
            y_tier = y[mask]
            
            predictions = model.predict(X_tier)
            
            results[tier_name] = {
                "n_samples": mask.sum(),
                "actual_ppg": y_tier.mean(),
                "predicted_ppg": predictions.mean(),
                "rmse": np.sqrt(mean_squared_error(y_tier, predictions)),
                "mae": mean_absolute_error(y_tier, predictions),
            }
        
        return results
    
    def evaluate_prediction_horizon(self, model, X: pd.DataFrame, 
                                     y_dict: Dict[int, pd.Series]) -> Dict[int, Dict]:
        """Evaluate model across different prediction horizons."""
        results = {}
        
        for n_weeks, y in y_dict.items():
            valid_mask = ~y.isna()
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            if len(y_valid) == 0:
                continue
            
            predictions = model.predict(X_valid, n_weeks) if hasattr(model, 'n_weeks') else model.predict(X_valid)
            
            results[n_weeks] = {
                "n_samples": len(y_valid),
                "rmse": np.sqrt(mean_squared_error(y_valid, predictions)),
                "mae": mean_absolute_error(y_valid, predictions),
                "r2": r2_score(y_valid, predictions),
                "rmse_per_week": np.sqrt(mean_squared_error(y_valid, predictions)) / n_weeks,
            }
        
        return results
    
    def backtest(self, model, data: pd.DataFrame, 
                 feature_cols: List[str],
                 target_col: str = "fantasy_points",
                 test_seasons: List[int] = None) -> Dict[str, Dict]:
        """
        Backtest model on historical seasons.
        
        Args:
            model: Trained model
            data: Full dataset with season column
            feature_cols: Feature column names
            target_col: Target column name
            test_seasons: Seasons to test on
            
        Returns:
            Dict of season to metrics
        """
        if test_seasons is None:
            seasons_in_data = sorted(data["season"].dropna().unique())
            test_seasons = seasons_in_data[-2:] if len(seasons_in_data) >= 2 else (seasons_in_data[-1:] if seasons_in_data else [])
        results = {}
        
        for season in test_seasons:
            season_data = data[data["season"] == season]
            
            if len(season_data) == 0:
                continue
            
            X = season_data[feature_cols]
            y = season_data[target_col]
            
            predictions = model.predict(X)
            
            results[season] = {
                "n_samples": len(y),
                "rmse": np.sqrt(mean_squared_error(y, predictions)),
                "mae": mean_absolute_error(y, predictions),
                "r2": r2_score(y, predictions),
                "correlation": np.corrcoef(y, predictions)[0, 1],
            }
        
        return results
    
    def compare_models(self, models: Dict[str, any], 
                       X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            models: Dict of model_name to model
            X: Features
            y: Target
            
        Returns:
            DataFrame comparing model performance
        """
        results = []
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X, y, name)
            metrics["model"] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index("model")
        
        # Add ranking
        for col in ["rmse", "mae", "mape"]:
            if col in df.columns:
                df[f"{col}_rank"] = df[col].rank()
        
        return df
    
    def _safe_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE, handling zeros."""
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def generate_report(self, model, X: pd.DataFrame, y: pd.Series,
                        positions: pd.Series = None,
                        utilization_scores: pd.Series = None) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics
        report.append("\n## Overall Performance")
        metrics = self.evaluate_model(model, X, y)
        for name, value in metrics.items():
            report.append(f"  {name}: {value:.4f}")
        
        # Cross-validation
        report.append("\n## Cross-Validation Results")
        cv_metrics = self.cross_validate(model, X, y)
        for name, value in cv_metrics.items():
            report.append(f"  {name}: {value:.4f}")
        
        # By position
        if positions is not None:
            report.append("\n## Performance by Position")
            pos_metrics = self.evaluate_by_position(model, X, y, positions)
            for pos, metrics in pos_metrics.items():
                report.append(f"\n  {pos}:")
                for name, value in metrics.items():
                    report.append(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")
        
        # By utilization tier
        if utilization_scores is not None:
            report.append("\n## Performance by Utilization Tier")
            tier_metrics = self.evaluate_by_utilization_tier(model, X, y, utilization_scores)
            for tier, metrics in tier_metrics.items():
                report.append(f"\n  {tier}:")
                for name, value in metrics.items():
                    report.append(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Position-specific benchmark targets (per requirements Section IV.A)
# 1-week RMSE targets: QB 6.0-7.5, RB 7.0-8.5, WR 6.5-8.0, TE 5.5-7.0
# Using upper bound as achievable target threshold
POSITION_BENCHMARKS = {
    "QB": {"rmse_target": 7.5, "mape_target": 25.0, "r2_target": 0.50},
    "RB": {"rmse_target": 8.5, "mape_target": 25.0, "r2_target": 0.50},
    "WR": {"rmse_target": 8.0, "mape_target": 25.0, "r2_target": 0.50},
    "TE": {"rmse_target": 7.0, "mape_target": 25.0, "r2_target": 0.50},
}

# 4-week horizon benchmarks (Section IV.A: RMSE 15-25% higher than 1w)
POSITION_BENCHMARKS_4W = {
    "QB": {"rmse_target": 10.0, "mape_target": 35.0, "r2_target": 0.40},
    "RB": {"rmse_target": 11.0, "mape_target": 35.0, "r2_target": 0.40},
    "WR": {"rmse_target": 10.0, "mape_target": 35.0, "r2_target": 0.40},
    "TE": {"rmse_target": 9.0, "mape_target": 35.0, "r2_target": 0.40},
}

# 18-week (season-long) horizon benchmarks (Section IV.A)
POSITION_BENCHMARKS_18W = {
    "QB": {"rmse_target": 15.0, "mape_target": 45.0, "r2_target": 0.30},
    "RB": {"rmse_target": 15.0, "mape_target": 45.0, "r2_target": 0.30},
    "WR": {"rmse_target": 15.0, "mape_target": 45.0, "r2_target": 0.30},
    "TE": {"rmse_target": 15.0, "mape_target": 45.0, "r2_target": 0.30},
}


def check_position_benchmarks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position: str,
) -> Dict[str, any]:
    """Check if model meets position-specific performance benchmarks.

    Returns dict with metric values and pass/fail for each benchmark.
    """
    benchmarks = POSITION_BENCHMARKS.get(position, POSITION_BENCHMARKS.get("RB", {}))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100) if non_zero.sum() > 0 else None

    result = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 3),
        "mape": round(mape, 1) if mape is not None else None,
        "rmse_target": benchmarks.get("rmse_target"),
        "mape_target": benchmarks.get("mape_target"),
        "r2_target": benchmarks.get("r2_target"),
        "rmse_pass": rmse <= benchmarks.get("rmse_target", 999),
        "mape_pass": (mape is not None and mape <= benchmarks.get("mape_target", 999)) if mape else None,
        "r2_pass": r2 >= benchmarks.get("r2_target", -999),
        "all_pass": (
            rmse <= benchmarks.get("rmse_target", 999)
            and r2 >= benchmarks.get("r2_target", -999)
            and (mape is None or mape <= benchmarks.get("mape_target", 999))
        ),
    }
    return result


def run_model_tests(model, X_test: pd.DataFrame, y_test: pd.Series,
                    positions: pd.Series = None) -> bool:
    """
    Run automated tests on a trained model.
    
    Returns True if all tests pass.
    """
    evaluator = ModelEvaluator()
    all_passed = True
    
    print("\nRunning Model Tests...")
    print("-" * 40)
    
    # Test 1: Model can make predictions
    try:
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test), "Prediction length mismatch"
        print("✓ Test 1: Model produces predictions")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        all_passed = False
    
    # Test 2: Predictions are reasonable (not NaN, not extreme)
    try:
        assert not np.any(np.isnan(predictions)), "Predictions contain NaN"
        assert np.all(predictions >= -10), "Predictions too negative"
        assert np.all(predictions <= 100), "Predictions too high"
        print("✓ Test 2: Predictions are reasonable values")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        all_passed = False
    
    # Test 3: Model beats naive baseline (predicting mean)
    try:
        naive_pred = np.full_like(predictions, y_test.mean())
        model_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        assert model_rmse < naive_rmse, f"Model RMSE ({model_rmse:.2f}) >= Naive RMSE ({naive_rmse:.2f})"
        print(f"✓ Test 3: Model beats naive baseline (RMSE: {model_rmse:.2f} vs {naive_rmse:.2f})")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        all_passed = False
    
    # Test 4: R² is positive
    try:
        r2 = r2_score(y_test, predictions)
        assert r2 > 0, f"R² is negative: {r2:.4f}"
        print(f"✓ Test 4: R² is positive ({r2:.4f})")
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        all_passed = False
    
    # Test 5: Correlation with actual values
    try:
        correlation = np.corrcoef(y_test, predictions)[0, 1]
        assert correlation > 0.3, f"Correlation too low: {correlation:.4f}"
        print(f"✓ Test 5: Correlation is acceptable ({correlation:.4f})")
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        all_passed = False
    
    # Test 6: Position-specific performance (if positions provided)
    if positions is not None:
        try:
            for pos in POSITIONS:
                mask = positions == pos
                if mask.sum() < 10:
                    continue
                pos_r2 = r2_score(y_test[mask], predictions[mask])
                assert pos_r2 > -0.5, f"{pos} R² too negative: {pos_r2:.4f}"
            print("✓ Test 6: Position-specific performance acceptable")
        except Exception as e:
            print(f"✗ Test 6 FAILED: {e}")
            all_passed = False
    
    print("-" * 40)
    print(f"Tests {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


def check_position_benchmarks_for_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position: str,
    horizon: str = "1w",
) -> Dict[str, any]:
    """Check position benchmarks for a specific prediction horizon.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        position: Position (QB, RB, WR, TE).
        horizon: One of '1w', '4w', '18w'.

    Returns:
        Dict with metric values and pass/fail per benchmark.
    """
    if horizon == "4w":
        benchmarks = POSITION_BENCHMARKS_4W.get(position, POSITION_BENCHMARKS_4W.get("RB", {}))
    elif horizon == "18w":
        benchmarks = POSITION_BENCHMARKS_18W.get(position, POSITION_BENCHMARKS_18W.get("RB", {}))
    else:
        benchmarks = POSITION_BENCHMARKS.get(position, POSITION_BENCHMARKS.get("RB", {}))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100) if non_zero.sum() > 0 else None

    return {
        "horizon": horizon,
        "position": position,
        "rmse": round(rmse, 2),
        "r2": round(r2, 3),
        "mape": round(mape, 1) if mape is not None else None,
        "rmse_target": benchmarks.get("rmse_target"),
        "rmse_pass": rmse <= benchmarks.get("rmse_target", 999),
        "mape_target": benchmarks.get("mape_target"),
        "mape_pass": (mape is not None and mape <= benchmarks.get("mape_target", 999)) if mape else None,
        "r2_target": benchmarks.get("r2_target"),
        "r2_pass": r2 >= benchmarks.get("r2_target", -999),
    }


def compare_to_expert_consensus(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_expert: np.ndarray,
    position: str = None,
) -> Dict[str, any]:
    """Compare model predictions against expert consensus projections.

    Per requirements: ML models should match or exceed expert projections by 5-15%.
    Position-specific targets: QB/WR 8-12%, RB 10-15%, TE 12-18%.

    Args:
        y_true: Actual values.
        y_pred_model: Model predictions.
        y_pred_expert: Expert consensus predictions (e.g. FantasyPros ECR).

    Returns:
        Dict with head-to-head comparison metrics.
    """
    from config.settings import SUCCESS_CRITERIA

    model_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_model)))
    expert_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_expert)))

    improvement_pct = ((expert_rmse - model_rmse) / expert_rmse * 100) if expert_rmse > 0 else 0.0

    # Position-specific beat targets
    target_key = f"beat_expert_pct_{position.lower()}" if position else "beat_expert_pct_rb"
    target_pct = SUCCESS_CRITERIA.get(target_key, 10.0)

    model_mae = float(mean_absolute_error(y_true, y_pred_model))
    expert_mae = float(mean_absolute_error(y_true, y_pred_expert))

    return {
        "model_rmse": round(model_rmse, 2),
        "expert_rmse": round(expert_rmse, 2),
        "rmse_improvement_pct": round(improvement_pct, 1),
        "model_mae": round(model_mae, 2),
        "expert_mae": round(expert_mae, 2),
        "beat_expert_target_pct": target_pct,
        "beats_expert": improvement_pct >= target_pct,
        "position": position,
    }

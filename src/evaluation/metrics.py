"""Model evaluation metrics and testing utilities."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
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


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman correlation without SciPy dependency.
    Returns NaN for insufficient/degenerate inputs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=float)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=float)
    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return np.nan
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def spearman_rank_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, top_n: Optional[int] = 50
) -> float:
    """Spearman rank correlation between predicted and actual (target rho > 0.65 for top-50)."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    if top_n is not None and len(y_true) > top_n:
        idx = np.argsort(y_pred)[-top_n:]
        y_true, y_pred = y_true[idx], y_pred[idx]
    return _safe_spearman(y_true, y_pred)


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
    """Precision/recall/F1 for 20+ point weeks (boom) and <5 point weeks (bust).
    
    Includes class prevalence rates for context on class imbalance.
    F1 score provides a balanced metric when boom/bust classes are imbalanced
    (booms are typically ~15-20% of weeks, busts ~25-30%).
    """
    boom_true = (y_true >= boom_thresh).astype(int)
    bust_true = (y_true < bust_thresh).astype(int)
    boom_pred = (y_pred >= boom_thresh).astype(int)
    bust_pred = (y_pred < bust_thresh).astype(int)
    out = {}
    n = len(y_true)
    # Class prevalence (for understanding imbalance)
    out["boom_prevalence"] = float(boom_true.mean()) if n > 0 else 0.0
    out["bust_prevalence"] = float(bust_true.mean()) if n > 0 else 0.0
    if boom_true.sum() > 0:
        out["boom_precision"] = float(precision_score(boom_true, boom_pred, zero_division=0))
        out["boom_recall"] = float(recall_score(boom_true, boom_pred, zero_division=0))
        p, r = out["boom_precision"], out["boom_recall"]
        out["boom_f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    if bust_true.sum() > 0:
        out["bust_precision"] = float(precision_score(bust_true, bust_pred, zero_division=0))
        out["bust_recall"] = float(recall_score(bust_true, bust_pred, zero_division=0))
        p, r = out["bust_precision"], out["bust_recall"]
        out["bust_f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
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
    return _safe_spearman(vor_true, vor_pred)


def confidence_interval_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_std: np.ndarray,
    nominal_levels: Optional[List[float]] = None,
) -> Dict:
    """
    Evaluate confidence interval calibration across multiple nominal coverage levels.

    A well-calibrated model's predicted intervals should contain the actual value
    at approximately the stated confidence level. For example, an 80% CI should
    contain ~80% of actuals.

    Args:
        y_true: Actual observed values.
        y_pred: Point predictions (interval centers).
        pred_std: Predicted standard deviations (used to construct intervals).
        nominal_levels: Coverage levels to evaluate (default: [0.50, 0.80, 0.90, 0.95]).

    Returns:
        Dict with per-level actual coverage, calibration error, and overall assessment.
        Keys: 'levels' (list of per-level dicts), 'mean_calibration_error',
              'max_calibration_error', 'is_calibrated' (max error < 10pp).
    """
    if nominal_levels is None:
        nominal_levels = [0.50, 0.80, 0.90, 0.95]

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)

    # Filter to valid rows (finite values, positive std)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(pred_std) & (pred_std > 0)
    if valid.sum() < 20:
        return {
            "levels": [],
            "mean_calibration_error": None,
            "max_calibration_error": None,
            "is_calibrated": None,
            "n_valid": int(valid.sum()),
            "error": "Insufficient valid samples for calibration check (need >= 20)",
        }

    y_t = y_true[valid]
    y_p = y_pred[valid]
    std = pred_std[valid]
    n = len(y_t)

    # z-scores for each nominal level (two-tailed)
    from scipy.stats import norm as _norm

    levels_result = []
    calibration_errors = []

    for nominal in nominal_levels:
        z = _norm.ppf(0.5 + nominal / 2.0)
        lower = y_p - z * std
        upper = y_p + z * std
        covered = ((y_t >= lower) & (y_t <= upper))
        actual_coverage = float(covered.mean())
        cal_error = abs(actual_coverage - nominal)
        calibration_errors.append(cal_error)

        # Sharpness: mean interval width (narrower is better if calibrated)
        mean_width = float(np.mean(upper - lower))

        levels_result.append({
            "nominal": nominal,
            "actual_coverage": round(actual_coverage, 4),
            "calibration_error": round(cal_error, 4),
            "mean_interval_width": round(mean_width, 2),
            "pass": cal_error < 0.10,  # Within 10 percentage points
        })

    mean_cal_error = float(np.mean(calibration_errors))
    max_cal_error = float(np.max(calibration_errors))

    return {
        "levels": levels_result,
        "mean_calibration_error": round(mean_cal_error, 4),
        "max_calibration_error": round(max_cal_error, 4),
        "is_calibrated": max_cal_error < 0.10,
        "n_valid": n,
    }


class ModelEvaluator:
    """Comprehensive model evaluation for NFL predictions."""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                       model_name: str = "model",
                       positions: pd.Series = None,
                       expert_predictions: Optional[np.ndarray] = None,
                       expert_rmse_by_position: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Evaluate a single model with multiple metrics including
        fantasy-specific metrics (Spearman, tier accuracy, boom/bust, VOR).
        
        Args:
            model: Trained model with predict method
            X: Feature DataFrame
            y: True target values
            model_name: Name for logging
            positions: Optional position labels for VOR calculation
            expert_predictions: Optional expert-consensus predictions aligned to y
            expert_rmse_by_position: Optional dict {position: expert_rmse}
            
        Returns:
            Dict of metric names to values
        """
        predictions = model.predict(X)
        y_arr = np.asarray(y, dtype=float)
        pred_arr = np.asarray(predictions, dtype=float)
        
        metrics = {
            "mse": mean_squared_error(y_arr, pred_arr),
            "rmse": np.sqrt(mean_squared_error(y_arr, pred_arr)),
            "mae": mean_absolute_error(y_arr, pred_arr),
            "r2": r2_score(y_arr, pred_arr),
            "mape": self._safe_mape(y_arr, pred_arr),
            "median_ae": float(np.median(np.abs(y_arr - pred_arr))),
        }
        
        # Add percentile errors
        errors = np.abs(y_arr - pred_arr)
        metrics["p90_error"] = float(np.percentile(errors, 90))
        metrics["p95_error"] = float(np.percentile(errors, 95))
        
        # Fantasy-specific metrics (per requirements Section V)
        # Spearman rank correlation
        metrics["spearman_rho"] = spearman_rank_correlation(y_arr, pred_arr)
        metrics["spearman_top50"] = spearman_rank_correlation(y_arr, pred_arr, top_n=50)
        
        # Tier classification accuracy (Elite/Strong/Flex/Waiver)
        metrics["tier_accuracy"] = tier_classification_accuracy(y_arr, pred_arr)
        
        # Boom/bust prediction
        boom = boom_bust_metrics(y_arr, pred_arr)
        metrics["boom_precision"] = boom.get("boom_precision", np.nan)
        metrics["boom_recall"] = boom.get("boom_recall", np.nan)
        metrics["bust_precision"] = boom.get("bust_precision", np.nan)
        metrics["bust_recall"] = boom.get("bust_recall", np.nan)
        
        # VOR accuracy (requires position labels)
        if positions is not None:
            metrics["vor_accuracy"] = vor_accuracy(y_arr, pred_arr, positions=np.asarray(positions))
        
        # Within-N accuracy (per requirements Section VII)
        metrics["within_7_pts_pct"] = float(np.mean(errors <= 7.0) * 100)
        metrics["within_10_pts_pct"] = float(np.mean(errors <= 10.0) * 100)
        
        # Beat naive baseline (season average)
        baseline_pred = np.full_like(pred_arr, np.mean(y_arr))
        baseline_rmse = np.sqrt(mean_squared_error(y_arr, baseline_pred))
        metrics["naive_baseline_rmse"] = float(baseline_rmse)
        if baseline_rmse > 0:
            metrics["improvement_over_baseline_pct"] = float(
                (1 - metrics["rmse"] / baseline_rmse) * 100
            )
        else:
            metrics["improvement_over_baseline_pct"] = 0.0

        # Expert consensus comparison (if expert projections are provided)
        if expert_predictions is not None and len(expert_predictions) == len(y_arr):
            exp_arr = np.asarray(expert_predictions, dtype=float)
            expert_rmse = float(np.sqrt(mean_squared_error(y_arr, exp_arr)))
            metrics["expert_rmse"] = expert_rmse
            if expert_rmse > 0:
                metrics["improvement_over_expert_pct"] = float(
                    (expert_rmse - metrics["rmse"]) / expert_rmse * 100
                )
            else:
                metrics["improvement_over_expert_pct"] = 0.0
            metrics["beats_expert_overall"] = float(metrics["improvement_over_expert_pct"] > 0.0)

        # Confidence interval calibration check (per requirements)
        # If model supports uncertainty estimation, evaluate CI calibration
        if hasattr(model, 'predict_with_uncertainty'):
            try:
                _, pred_std = model.predict_with_uncertainty(X)
                pred_std_arr = np.asarray(pred_std, dtype=float)
                ci_cal = confidence_interval_calibration(y_arr, pred_arr, pred_std_arr)
                metrics["ci_calibration"] = ci_cal
                metrics["ci_is_calibrated"] = ci_cal.get("is_calibrated")
                metrics["ci_mean_calibration_error"] = ci_cal.get("mean_calibration_error")
                metrics["ci_max_calibration_error"] = ci_cal.get("max_calibration_error")
                # Per-level coverage for quick reference
                for lvl in ci_cal.get("levels", []):
                    nom_key = f"ci_{int(lvl['nominal']*100)}_coverage"
                    metrics[nom_key] = lvl["actual_coverage"]
            except Exception:
                pass

        # Expert RMSE lookup comparison by position (no expert predictions required)
        if expert_rmse_by_position and positions is not None:
            pos_arr = np.asarray(positions)
            improvements = []
            for pos in sorted(set(pos_arr)):
                exp_rmse = expert_rmse_by_position.get(pos)
                if exp_rmse is None or exp_rmse <= 0:
                    continue
                mask = pos_arr == pos
                if mask.sum() == 0:
                    continue
                pos_rmse = float(np.sqrt(mean_squared_error(y_arr[mask], pred_arr[mask])))
                improvements.append((exp_rmse - pos_rmse) / exp_rmse * 100)
            if improvements:
                metrics["improvement_over_expert_by_pos_avg_pct"] = float(np.mean(improvements))
        
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
        mse_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error", n_jobs=1)
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
        r2_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2", n_jobs=1)
        
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

        def _predict_for_horizon(_model, _X: pd.DataFrame, _n_weeks: int) -> np.ndarray:
            """Predict for a horizon, preferring horizon-aware model signatures."""
            # Multi-horizon wrappers (e.g., MultiWeekModel) expose predict(X, n_weeks).
            # Single-horizon models expose predict(X). Try the horizon form first.
            try:
                return _model.predict(_X, _n_weeks)
            except TypeError:
                return _model.predict(_X)
            except AttributeError:
                return _model.predict(_X)
        
        for n_weeks, y in y_dict.items():
            valid_mask = ~y.isna()
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            if len(y_valid) == 0:
                continue
            
            predictions = _predict_for_horizon(model, X_valid, n_weeks)
            
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
                        utilization_scores: pd.Series = None,
                        expert_rmse: Dict[str, float] = None) -> str:
        """Generate a comprehensive evaluation report including fantasy-specific metrics.
        
        Args:
            model: Trained model with predict method.
            X: Feature DataFrame.
            y: True target values.
            positions: Position labels for per-position evaluation.
            utilization_scores: Utilization scores for tier evaluation.
            expert_rmse: Optional dict {position: expert_rmse} for expert
                consensus comparison (Section V.C of requirements).
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall metrics (now includes fantasy-specific)
        report.append("\n## Overall Performance")
        metrics = self.evaluate_model(
            model, X, y, positions=positions, expert_rmse_by_position=expert_rmse
        )
        for name, value in metrics.items():
            if isinstance(value, float):
                report.append(f"  {name}: {value:.4f}")
            else:
                report.append(f"  {name}: {value}")
        
        # Fantasy-specific summary
        report.append("\n## Fantasy-Specific Metrics (Requirements Section V)")
        report.append(f"  Spearman Rank Correlation (all): {metrics.get('spearman_rho', 'N/A'):.4f}" if isinstance(metrics.get('spearman_rho'), float) else f"  Spearman Rank Correlation: N/A")
        report.append(f"  Spearman Top-50: {metrics.get('spearman_top50', 'N/A'):.4f}" if isinstance(metrics.get('spearman_top50'), float) else f"  Spearman Top-50: N/A")
        if isinstance(metrics.get("tier_accuracy"), float):
            report.append(f"  Tier Classification Accuracy: {metrics.get('tier_accuracy', 0.0) * 100:.1f}%")
        else:
            report.append("  Tier Accuracy: N/A")
        report.append(f"  Boom Precision/Recall: {self._fmt_metric(metrics.get('boom_precision'))} / {self._fmt_metric(metrics.get('boom_recall'))}")
        report.append(f"  Bust Precision/Recall: {self._fmt_metric(metrics.get('bust_precision'))} / {self._fmt_metric(metrics.get('bust_recall'))}")
        report.append(f"  Within 7 pts: {metrics.get('within_7_pts_pct', 'N/A'):.1f}%  (target: 70%+)")
        report.append(f"  Within 10 pts: {metrics.get('within_10_pts_pct', 'N/A'):.1f}%  (target: 80%+)")
        report.append(f"  Improvement over naive baseline: {metrics.get('improvement_over_baseline_pct', 'N/A'):.1f}%  (target: >25%)")
        if isinstance(metrics.get("improvement_over_expert_by_pos_avg_pct"), float):
            report.append(
                f"  Avg improvement vs expert by position: {metrics['improvement_over_expert_by_pos_avg_pct']:+.1f}%"
            )
        
        # Cross-validation
        report.append("\n## Cross-Validation Results")
        try:
            cv_metrics = self.cross_validate(model, X, y)
            for name, value in cv_metrics.items():
                report.append(f"  {name}: {value:.4f}")
        except Exception as e:
            report.append(f"  Cross-validation failed: {e}")
        
        # By position (with benchmarks)
        if positions is not None:
            report.append("\n## Performance by Position")
            pos_metrics = self.evaluate_by_position(model, X, y, positions)
            for pos, pm in pos_metrics.items():
                report.append(f"\n  {pos}:")
                for name, value in pm.items():
                    report.append(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")
                # Check against benchmarks
                bench = POSITION_BENCHMARKS.get(pos, {})
                if bench and 'rmse' in pm:
                    rmse_target = bench.get('rmse_target', 99)
                    status = "PASS" if pm['rmse'] <= rmse_target else "FAIL"
                    report.append(f"    benchmark: RMSE {pm['rmse']:.2f} vs target {rmse_target} [{status}]")
        
        # Expert consensus comparison (per requirements Section V.C)
        # expert_rmse: dict like {"QB": 7.5, "RB": 8.5, ...} giving known expert RMSE per position
        if expert_rmse and positions is not None:
            report.append("\n## Expert Consensus Comparison (Section V.C)")
            predictions = model.predict(X)
            y_arr = np.asarray(y, dtype=float)
            pred_arr = np.asarray(predictions, dtype=float)
            pos_arr = np.asarray(positions)
            for pos in sorted(set(pos_arr)):
                mask = pos_arr == pos
                if mask.sum() == 0:
                    continue
                model_rmse_val = float(np.sqrt(mean_squared_error(y_arr[mask], pred_arr[mask])))
                exp_rmse_val = expert_rmse.get(pos)
                if exp_rmse_val is not None and exp_rmse_val > 0:
                    imp = (exp_rmse_val - model_rmse_val) / exp_rmse_val * 100
                    status = "PASS" if imp > 0 else "FAIL"
                    report.append(f"  {pos}: Model RMSE {model_rmse_val:.2f} vs Expert {exp_rmse_val:.2f}  ({imp:+.1f}%) [{status}]")
        
        # By utilization tier
        if utilization_scores is not None:
            report.append("\n## Performance by Utilization Tier")
            tier_metrics = self.evaluate_by_utilization_tier(model, X, y, utilization_scores)
            for tier, tm in tier_metrics.items():
                report.append(f"\n  {tier}:")
                for name, value in tm.items():
                    report.append(f"    {name}: {value:.4f}" if isinstance(value, float) else f"    {name}: {value}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

    @staticmethod
    def _fmt_metric(value: Optional[float]) -> str:
        """Format numeric metric values while handling NaN/None safely."""
        if value is None:
            return "N/A"
        try:
            if np.isnan(value):
                return "N/A"
        except TypeError:
            return "N/A"
        return f"{float(value):.3f}"


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

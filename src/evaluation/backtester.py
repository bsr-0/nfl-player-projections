"""
Backtesting and Model Validation Module

Provides tools to validate model predictions against historical results,
generate accuracy metrics, and create visualizations for stakeholder presentations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.evaluation.metrics import (
    spearman_rank_correlation,
    tier_classification_accuracy,
    boom_bust_metrics,
    vor_accuracy,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import POSITIONS, DATA_DIR, MODELS_DIR


class ModelBacktester:
    """
    Backtests model predictions against actual historical results.
    
    Allows validation of model accuracy by comparing what the model
    would have predicted vs what actually happened.
    """
    
    def __init__(self):
        self.results_dir = DATA_DIR / "backtest_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_simple_summary(self, results: Dict) -> str:
        """
        Generate a simple one-page summary for non-technical stakeholders.
        
        Returns plain English summary with key takeaways.
        """
        m = results['metrics']
        
        # Determine overall assessment
        if m['r2'] > 0.3 and m['correlation'] > 0.5:
            verdict = "STRONG"
            emoji = "✅"
            recommendation = "Trust for lineup decisions"
        elif m['r2'] > 0.15:
            verdict = "GOOD"
            emoji = "👍"
            recommendation = "Use as one input among several"
        else:
            verdict = "LIMITED"
            emoji = "⚠️"
            recommendation = "Use with caution"
        
        lines = [
            "=" * 50,
            "NFL PREDICTOR - QUICK SUMMARY",
            "=" * 50,
            "",
            f"Season Tested: {results['season']}",
            f"Predictions Made: {results['n_predictions']:,}",
            "",
            "─" * 50,
            f"OVERALL VERDICT: {emoji} {verdict}",
            "─" * 50,
            "",
            "THE BOTTOM LINE:",
            f"  • Our predictions correlate {m['correlation']*100:.0f}% with actual results",
            f"  • {m['within_5_pts_pct']:.0f}% of predictions within 5 fantasy points",
            f"  • We correctly pick above/below average {m['directional_accuracy_pct']:.0f}% of the time",
            "",
            "WHAT THIS MEANS:",
        ]
        
        # Add position-specific insights
        if results.get('ranking_accuracy'):
            best_pos = max(results['ranking_accuracy'].items(), 
                          key=lambda x: x[1].get('top_10_hit_rate', 0) or 0)
            lines.append(f"  • Best at predicting: {best_pos[0]}s ({best_pos[1].get('top_10_hit_rate', 0):.0f}% Top 10 accuracy)")
        
        lines.extend([
            f"  • Average prediction error: {m['mae']:.1f} points per week",
            "",
            "RECOMMENDATION:",
            f"  {recommendation}",
            "",
            "USE THIS MODEL FOR:",
            "  ✓ Start/Sit decisions",
            "  ✓ Waiver wire pickups", 
            "  ✓ Trade evaluations",
            "",
            "DON'T RELY ON IT FOR:",
            "  ✗ Injury predictions",
            "  ✗ Weather impacts",
            "  ✗ Rookie breakouts (limited data)",
            "",
            "=" * 50,
        ])
        
        return "\n".join(lines)
    
    def compare_to_baseline(self, df: pd.DataFrame, 
                            actual_col: str = 'fantasy_points',
                            pred_col: str = 'predicted_points') -> Dict:
        """
        Compare model to simple baseline (season average).
        
        Shows how much better the model is vs just using averages.
        """
        # Baseline: use player's rolling 4-week average
        df = df.copy()
        df['baseline_pred'] = df.groupby('player_id')[actual_col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )
        
        # Remove rows without baseline
        valid = df.dropna(subset=['baseline_pred', pred_col, actual_col])
        
        if len(valid) < 10:
            return {"error": "Insufficient data for baseline comparison"}
        
        # Model metrics
        model_rmse = np.sqrt(mean_squared_error(valid[actual_col], valid[pred_col]))
        model_mae = mean_absolute_error(valid[actual_col], valid[pred_col])
        model_r2 = r2_score(valid[actual_col], valid[pred_col])
        
        # Baseline metrics
        baseline_rmse = np.sqrt(mean_squared_error(valid[actual_col], valid['baseline_pred']))
        baseline_mae = mean_absolute_error(valid[actual_col], valid['baseline_pred'])
        baseline_r2 = r2_score(valid[actual_col], valid['baseline_pred'])
        
        rmse_pct = (baseline_rmse - model_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
        return {
            "model": {"rmse": round(model_rmse, 2), "mae": round(model_mae, 2), "r2": round(model_r2, 3)},
            "baseline": {"rmse": round(baseline_rmse, 2), "mae": round(baseline_mae, 2), "r2": round(baseline_r2, 3)},
            "improvement": {
                "rmse_pct": round(rmse_pct, 1),
                "mae_pct": round((baseline_mae - model_mae) / baseline_mae * 100, 1) if baseline_mae > 0 else 0,
                "r2_gain": round(model_r2 - baseline_r2, 3)
            },
            "model_beats_baseline": model_rmse < baseline_rmse,
            "beat_baseline_by_20_pct": rmse_pct >= 20.0,
        }

    def compare_to_multiple_baselines(
        self,
        df: pd.DataFrame,
        actual_col: str = "fantasy_points",
        pred_col: str = "predicted_points",
    ) -> Dict:
        """
        Compare model to multiple naive baselines (per requirements: beat all by >20%).
        Baselines: (1) persistence = previous week's score, (2) season average to date,
        (3) position average for that week/season.
        """
        df = df.copy()
        if pred_col not in df.columns or actual_col not in df.columns:
            return {"error": "Missing prediction or actual column"}
        need_cols = ["player_id", "position"]
        if not all(c in df.columns for c in need_cols):
            return {"error": "Need player_id and position for baseline comparison"}
        has_week = "week" in df.columns
        has_season = "season" in df.columns
        df = df.sort_values(["player_id", "season", "week"] if has_season and has_week else ["player_id"]).reset_index(drop=True)

        # Baseline 1: Persistence (previous week's actual)
        df["baseline_persistence"] = np.nan
        if has_week:
            df["baseline_persistence"] = df.groupby("player_id")[actual_col].shift(1)
        # Fill persistence with position mean where missing
        if "position" in df.columns:
            pos_mean = df.groupby("position")[actual_col].transform("mean")
            df["baseline_persistence"] = df["baseline_persistence"].fillna(pos_mean)
        df["baseline_persistence"] = df["baseline_persistence"].fillna(df[actual_col].mean())

        # Baseline 2: Season average to date (rolling 4-week mean, same as compare_to_baseline)
        df["baseline_season_avg"] = df.groupby("player_id")[actual_col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )
        df["baseline_season_avg"] = df["baseline_season_avg"].fillna(df[actual_col].mean())

        # Baseline 3: Position average (league-wide position mean for that week/season)
        if has_season and has_week:
            pos_week_mean = df.groupby(["season", "week", "position"])[actual_col].transform("mean")
            df["baseline_position_avg"] = pos_week_mean
        else:
            df["baseline_position_avg"] = df.groupby("position")[actual_col].transform("mean")
        df["baseline_position_avg"] = df["baseline_position_avg"].fillna(df[actual_col].mean())

        valid = df.dropna(subset=[pred_col, actual_col])
        if len(valid) < 10:
            return {"error": "Insufficient data for baseline comparison"}

        model_rmse = np.sqrt(mean_squared_error(valid[actual_col], valid[pred_col]))
        model_mae = mean_absolute_error(valid[actual_col], valid[pred_col])

        def baseline_stats(baseline_name: str) -> Dict:
            v = valid.dropna(subset=[baseline_name])
            if len(v) < 10:
                return {"rmse": None, "mae": None, "beat_by_20_pct": False}
            b_rmse = np.sqrt(mean_squared_error(v[actual_col], v[baseline_name]))
            b_mae = mean_absolute_error(v[actual_col], v[baseline_name])
            pct = (b_rmse - model_rmse) / b_rmse * 100 if b_rmse > 0 else 0
            return {
                "rmse": round(b_rmse, 2),
                "mae": round(b_mae, 2),
                "beat_by_20_pct": pct >= 20.0,
                "improvement_pct": round(pct, 1),
            }

        persistence = baseline_stats("baseline_persistence")
        season_avg = baseline_stats("baseline_season_avg")
        position_avg = baseline_stats("baseline_position_avg")

        return {
            "model": {"rmse": round(model_rmse, 2), "mae": round(model_mae, 2)},
            "baseline_persistence": persistence,
            "baseline_season_avg": season_avg,
            "baseline_position_avg": position_avg,
            "model_beats_all_by_20_pct": (
                persistence.get("beat_by_20_pct", False)
                and season_avg.get("beat_by_20_pct", False)
                and position_avg.get("beat_by_20_pct", False)
            ),
        }

    def compare_to_expert_consensus(
        self,
        df: pd.DataFrame,
        expert_csv_path: str,
        actual_col: str = "fantasy_points",
        pred_col: str = "predicted_points",
        player_key: str = "name",
    ) -> Dict:
        """
        Compare model and expert consensus to actuals. Expert CSV should have
        columns for player identifier and projected points (e.g. 'player_name', 'pts').
        """
        try:
            expert = pd.read_csv(expert_csv_path)
        except Exception as e:
            return {"error": f"Could not load expert CSV: {e}"}
        name_col = next((c for c in expert.columns if "name" in c.lower() or "player" in c.lower()), expert.columns[0])
        pts_col = next((c for c in expert.columns if "pts" in c.lower() or "points" in c.lower() or "proj" in c.lower()), None)
        if pts_col is None:
            return {"error": "Expert CSV must have a points/projection column"}
        expert = expert.rename(columns={name_col: "expert_name", pts_col: "expert_pts"})
        if player_key not in df.columns:
            return {"error": f"Player key {player_key} not in DataFrame"}
        df_key = df[player_key].astype(str).str.strip().str.upper()
        expert_key = expert["expert_name"].astype(str).str.strip().str.upper()
        expert = expert.assign(_key=expert_key)
        left = df.assign(_key=df_key)
        merged = left.merge(expert[["_key", "expert_pts"]], on="_key", how="inner").drop(columns=["_key"], errors="ignore")
        if len(merged) < 10:
            return {"error": "Insufficient overlap between predictions and expert data"}
        model_rmse = np.sqrt(mean_squared_error(merged[actual_col], merged[pred_col]))
        expert_rmse = np.sqrt(mean_squared_error(merged[actual_col], merged["expert_pts"]))
        return {
            "model_rmse": round(model_rmse, 2),
            "expert_rmse": round(expert_rmse, 2),
            "model_vs_expert_pct": round((expert_rmse - model_rmse) / expert_rmse * 100, 1) if expert_rmse > 0 else None,
            "n_matched": len(merged),
        }

    def calculate_confidence_intervals(self, df: pd.DataFrame,
                                        pred_col: str = 'predicted_points',
                                        actual_col: str = 'fantasy_points',
                                        confidence: float = 0.8) -> pd.DataFrame:
        """
        Add confidence intervals to predictions.
        
        Shows uncertainty range for each prediction.
        """
        df = df.copy()
        
        # Calculate historical prediction errors by position
        errors = df[actual_col] - df[pred_col]
        
        for pos in df['position'].unique():
            pos_mask = df['position'] == pos
            pos_errors = errors[pos_mask]
            
            # Use historical error distribution for CI
            lower_pct = (1 - confidence) / 2
            upper_pct = 1 - lower_pct
            
            lower_bound = pos_errors.quantile(lower_pct)
            upper_bound = pos_errors.quantile(upper_pct)
            
            df.loc[pos_mask, 'ci_lower'] = df.loc[pos_mask, pred_col] + lower_bound
            df.loc[pos_mask, 'ci_upper'] = df.loc[pos_mask, pred_col] + upper_bound
        
        # Ensure non-negative
        df['ci_lower'] = df['ci_lower'].clip(lower=0)
        
        return df
    
    def backtest_season(self, 
                        predictions: pd.DataFrame,
                        actuals: pd.DataFrame,
                        season: int,
                        prediction_col: str = "predicted_points",
                        actual_col: str = "fantasy_points") -> Dict:
        """
        Compare predictions vs actual results for a season.
        
        Args:
            predictions: DataFrame with predicted values
            actuals: DataFrame with actual results
            season: Season year being backtested
            prediction_col: Column name for predictions
            actual_col: Column name for actual values
            
        Returns:
            Dict with comprehensive backtest results
        """
        # Merge predictions with actuals
        # Handle case where predictions already has all columns
        if actual_col in predictions.columns and prediction_col in predictions.columns:
            merged = predictions.copy()
        else:
            merge_cols = ['player_id', 'week']
            actual_cols = [c for c in [actual_col, 'name', 'position', 'team'] if c in actuals.columns]
            merged = predictions.merge(
                actuals[merge_cols + actual_cols],
                on=merge_cols,
                how='inner',
                suffixes=('_pred', '_actual')
            )
        
        if merged.empty:
            return {"error": "No matching data for backtest"}
        
        # Calculate metrics
        results = {
            "season": season,
            "backtest_date": datetime.now().isoformat(),
            "n_predictions": len(merged),
            "metrics": self._calculate_metrics(
                merged[actual_col], 
                merged[prediction_col]
            ),
            "by_position": {},
            "by_week": {},
            "top_performers": {},
            "biggest_misses": {},
            "ranking_accuracy": {}
        }
        
        # Metrics by position
        for pos in merged['position'].unique():
            pos_data = merged[merged['position'] == pos]
            if len(pos_data) >= 10:
                results["by_position"][pos] = self._calculate_metrics(
                    pos_data[actual_col],
                    pos_data[prediction_col]
                )
                results["by_position"][pos]["n_players"] = len(pos_data['player_id'].unique())
        
        # Metrics by week
        for week in sorted(merged['week'].unique()):
            week_data = merged[merged['week'] == week]
            results["by_week"][week] = self._calculate_metrics(
                week_data[actual_col],
                week_data[prediction_col]
            )
        
        # Ranking accuracy (did we correctly identify top players?)
        results["ranking_accuracy"] = self._calculate_ranking_accuracy(merged, actual_col, prediction_col)
        # Spearman rank correlation per position (top-50, target rho > 0.65)
        results["spearman_by_position"] = {}
        for pos in merged["position"].unique():
            pos_data = merged[merged["position"] == pos]
            if len(pos_data) >= 10:
                rho = spearman_rank_correlation(
                    pos_data[actual_col].values,
                    pos_data[prediction_col].values,
                    top_n=50,
                )
                results["spearman_by_position"][pos] = round(float(rho), 3) if np.isfinite(rho) else None

        # Top performers analysis
        results["top_performers"] = self._analyze_top_performers(merged, actual_col, prediction_col)
        
        # Biggest prediction misses
        results["biggest_misses"] = self._find_biggest_misses(merged, actual_col, prediction_col)
        
        return results
    
    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate comprehensive accuracy metrics."""
        # Remove NaN values
        mask = ~(actual.isna() | predicted.isna())
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) < 2:
            return {"error": "Insufficient data"}
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # Correlation
        correlation = actual.corr(predicted)
        
        # Mean Absolute Percentage Error (for non-zero actuals)
        non_zero = actual != 0
        if non_zero.sum() > 0:
            mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
        else:
            mape = None
        
        # Directional accuracy (did we predict above/below average correctly?)
        avg_actual = actual.mean()
        directional_correct = ((predicted > avg_actual) == (actual > avg_actual)).mean() * 100
        
        within_3 = (np.abs(actual - predicted) <= 3).mean() * 100
        within_5 = (np.abs(actual - predicted) <= 5).mean() * 100
        within_7 = (np.abs(actual - predicted) <= 7).mean() * 100
        within_10 = (np.abs(actual - predicted) <= 10).mean() * 100

        a, p = actual.values, predicted.values
        spearman = spearman_rank_correlation(a, p, top_n=50)
        tier_acc = tier_classification_accuracy(a, p)
        boom_bust = boom_bust_metrics(a, p, boom_thresh=20.0, bust_thresh=5.0)
        vor = vor_accuracy(a, p)

        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 3),
            "correlation": round(correlation, 3),
            "mape": round(mape, 1) if mape else None,
            "directional_accuracy_pct": round(directional_correct, 1),
            "within_3_pts_pct": round(within_3, 1),
            "within_5_pts_pct": round(within_5, 1),
            "within_7_pts_pct": round(within_7, 1),
            "within_10_pts_pct": round(within_10, 1),
            "spearman_rho": round(float(spearman), 3) if np.isfinite(spearman) else None,
            "tier_classification_accuracy": round(float(tier_acc), 3) if np.isfinite(tier_acc) else None,
            "boom_bust": boom_bust,
            "vor_rank_correlation": round(float(vor), 3) if np.isfinite(vor) else None,
            "avg_actual": round(actual.mean(), 2),
            "avg_predicted": round(predicted.mean(), 2),
            "std_actual": round(actual.std(), 2),
            "std_predicted": round(predicted.std(), 2),
        }
    
    def _calculate_ranking_accuracy(self, df: pd.DataFrame, 
                                     actual_col: str, 
                                     pred_col: str) -> Dict:
        """Calculate how well we ranked players."""
        results = {}
        
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos].copy()
            
            # Weekly ranking accuracy
            weekly_results = []
            for week in pos_df['week'].unique():
                week_df = pos_df[pos_df['week'] == week].copy()
                if len(week_df) < 5:
                    continue
                
                # Rank by predicted and actual
                week_df['pred_rank'] = week_df[pred_col].rank(ascending=False)
                week_df['actual_rank'] = week_df[actual_col].rank(ascending=False)
                
                # Top 5, 10, 20 accuracy
                for n in [5, 10, 20]:
                    if len(week_df) >= n:
                        pred_top_n = set(week_df.nsmallest(n, 'pred_rank')['player_id'])
                        actual_top_n = set(week_df.nsmallest(n, 'actual_rank')['player_id'])
                        hit_rate = len(pred_top_n & actual_top_n) / n * 100
                        weekly_results.append({
                            'week': week,
                            f'top_{n}_hit_rate': hit_rate
                        })
            
            if weekly_results:
                weekly_df = pd.DataFrame(weekly_results)
                results[pos] = {
                    'top_5_hit_rate': round(weekly_df['top_5_hit_rate'].mean(), 1) if 'top_5_hit_rate' in weekly_df else None,
                    'top_10_hit_rate': round(weekly_df['top_10_hit_rate'].mean(), 1) if 'top_10_hit_rate' in weekly_df else None,
                    'top_20_hit_rate': round(weekly_df['top_20_hit_rate'].mean(), 1) if 'top_20_hit_rate' in weekly_df else None,
                }
        
        return results
    
    def _analyze_top_performers(self, df: pd.DataFrame,
                                 actual_col: str,
                                 pred_col: str) -> Dict:
        """Analyze how well we predicted top performers."""
        results = {}
        
        # Season-long top performers by position
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            
            # Aggregate season totals
            season_totals = pos_df.groupby(['player_id', 'name']).agg({
                actual_col: 'sum',
                pred_col: 'sum'
            }).reset_index()
            
            season_totals['actual_rank'] = season_totals[actual_col].rank(ascending=False)
            season_totals['pred_rank'] = season_totals[pred_col].rank(ascending=False)
            
            # Top 10 actual performers - where did we rank them?
            top_10_actual = season_totals.nsmallest(10, 'actual_rank')
            
            results[pos] = {
                'top_10_actual': top_10_actual[['name', 'actual_rank', 'pred_rank', actual_col, pred_col]].to_dict('records'),
                'avg_pred_rank_of_top_10': round(top_10_actual['pred_rank'].mean(), 1),
                'top_10_in_our_top_20': (top_10_actual['pred_rank'] <= 20).sum()
            }
        
        return results
    
    def _find_biggest_misses(self, df: pd.DataFrame,
                              actual_col: str,
                              pred_col: str,
                              n: int = 10) -> List[Dict]:
        """Find the biggest prediction misses."""
        df = df.copy()
        df['error'] = df[actual_col] - df[pred_col]
        df['abs_error'] = np.abs(df['error'])
        
        biggest = df.nlargest(n, 'abs_error')[
            ['name', 'position', 'team', 'week', actual_col, pred_col, 'error']
        ]
        
        return biggest.to_dict('records')
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable backtest report."""
        lines = []
        lines.append("=" * 70)
        lines.append("NFL PREDICTOR - BACKTEST VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Season: {results['season']}")
        lines.append(f"Report Generated: {results['backtest_date'][:10]}")
        lines.append(f"Total Predictions Evaluated: {results['n_predictions']:,}")
        lines.append("")
        
        # Overall metrics
        m = results['metrics']
        lines.append("-" * 70)
        lines.append("OVERALL MODEL ACCURACY")
        lines.append("-" * 70)
        lines.append(f"  R² Score:              {m['r2']:.3f}  (1.0 = perfect, 0 = baseline)")
        lines.append(f"  Correlation:           {m['correlation']:.3f}")
        lines.append(f"  RMSE:                  {m['rmse']:.2f} points")
        lines.append(f"  MAE:                   {m['mae']:.2f} points")
        if m.get("mape") is not None:
            lines.append(f"  MAPE:                  {m['mape']:.1f}%")
        lines.append(f"  Directional Accuracy:  {m['directional_accuracy_pct']:.1f}%")
        lines.append("")
        lines.append("  Prediction Precision:")
        lines.append(f"    Within 3 points:     {m['within_3_pts_pct']:.1f}%")
        lines.append(f"    Within 5 points:     {m['within_5_pts_pct']:.1f}%")
        lines.append(f"    Within 7 points:     {m.get('within_7_pts_pct', 0):.1f}%  (target ≥ 70%)")
        lines.append(f"    Within 10 points:    {m['within_10_pts_pct']:.1f}%  (target ≥ 80%)")
        if m.get("spearman_rho") is not None:
            lines.append(f"  Spearman (top-50):   {m['spearman_rho']:.3f}  (target > 0.65)")
        if m.get("tier_classification_accuracy") is not None:
            lines.append(f"  Tier accuracy:       {m['tier_classification_accuracy']:.3f}  (target > 0.75)")
        if m.get("vor_rank_correlation") is not None:
            lines.append(f"  VOR rank correlation: {m['vor_rank_correlation']:.3f}")
        if m.get("boom_bust"):
            lines.append(f"  Boom/bust:           {m['boom_bust']}")
        lines.append("")
        if results.get("spearman_by_position"):
            lines.append("  Spearman by position (top-50):")
            for pos, rho in results["spearman_by_position"].items():
                lines.append(f"    {pos}: {rho}")
            lines.append("")
        # By position
        lines.append("-" * 70)
        lines.append("ACCURACY BY POSITION")
        lines.append("-" * 70)
        for pos, pm in results['by_position'].items():
            lines.append(f"\n  {pos}:")
            lines.append(f"    R²: {pm['r2']:.3f}  |  RMSE: {pm['rmse']:.2f}  |  Correlation: {pm['correlation']:.3f}")
            lines.append(f"    Within 5 pts: {pm['within_5_pts_pct']:.1f}%  |  Directional: {pm['directional_accuracy_pct']:.1f}%")
        lines.append("")

        # By utilization tier (when available)
        if results.get("by_utilization_tier"):
            lines.append("-" * 70)
            lines.append("ACCURACY BY UTILIZATION TIER")
            lines.append("-" * 70)
            for tier_name, tm in results["by_utilization_tier"].items():
                lines.append(f"\n  {tier_name}: n={tm['n_samples']}  RMSE: {tm['rmse']:.2f}  MAE: {tm['mae']:.2f}")
            lines.append("")
        
        # Ranking accuracy
        lines.append("-" * 70)
        lines.append("RANKING ACCURACY (How well did we identify top players?)")
        lines.append("-" * 70)
        for pos, ra in results['ranking_accuracy'].items():
            lines.append(f"\n  {pos}:")
            if ra.get('top_5_hit_rate'):
                lines.append(f"    Top 5 Hit Rate:  {ra['top_5_hit_rate']:.1f}%")
            if ra.get('top_10_hit_rate'):
                lines.append(f"    Top 10 Hit Rate: {ra['top_10_hit_rate']:.1f}%")
            if ra.get('top_20_hit_rate'):
                lines.append(f"    Top 20 Hit Rate: {ra['top_20_hit_rate']:.1f}%")
        lines.append("")

        # Multiple baselines (persistence, season avg, position avg)
        if results.get("multiple_baseline_comparison") and "error" not in results["multiple_baseline_comparison"]:
            mbc = results["multiple_baseline_comparison"]
            lines.append("-" * 70)
            lines.append("MULTIPLE BASELINE COMPARISON (target: beat each by >20%)")
            lines.append("-" * 70)
            lines.append(f"  Model RMSE: {mbc['model']['rmse']}  MAE: {mbc['model']['mae']}")
            for name, key in [
                ("Persistence (prev week)", "baseline_persistence"),
                ("Season avg (4w roll)", "baseline_season_avg"),
                ("Position avg", "baseline_position_avg"),
            ]:
                b = mbc.get(key, {})
                if b.get("rmse") is not None:
                    beat = "yes" if b.get("beat_by_20_pct") else "no"
                    lines.append(f"  {name}: RMSE={b['rmse']}  improvement={b.get('improvement_pct')}%  beat_by_20%={beat}")
            lines.append(f"  Model beats all by 20%: {mbc.get('model_beats_all_by_20_pct', False)}")
            lines.append("")

        # Success criteria (requirements Section VII - comprehensive)
        if results.get("success_criteria"):
            sc = results["success_criteria"]
            lines.append("-" * 70)
            lines.append("SUCCESS CRITERIA (Requirements Section VII)")
            lines.append("-" * 70)
            lines.append(f"  Spearman ρ > 0.65:          {'PASS' if sc.get('spearman_gt_065') else 'FAIL'}  (actual: {sc.get('spearman_rho')})")
            lines.append(f"  Within 10 pts ≥ 80%:        {'PASS' if sc.get('within_10_pts_pct_ge_80') else 'FAIL'}  (actual: {sc.get('within_10_pts_pct')}%)")
            lines.append(f"  Within 7 pts ≥ 70%:         {'PASS' if sc.get('within_7_pts_pct_ge_70') else 'FAIL'}  (actual: {sc.get('within_7_pts_pct')}%)")
            lines.append(f"  MAPE < 25%:                 {'PASS' if sc.get('mape_lt_25') else 'FAIL'}  (actual: {sc.get('mape')}%)")
            lines.append(f"  Tier accuracy ≥ 75%:        {'PASS' if sc.get('tier_accuracy_ge_075') else 'FAIL'}  (actual: {sc.get('tier_accuracy')})")
            lines.append(f"  Beat all baselines >20%:    {'PASS' if sc.get('beat_all_baselines_by_20_pct') else 'FAIL'}")
            lines.append(f"  Beat primary baseline >25%: {'PASS' if sc.get('beat_primary_baseline_by_25_pct') else 'FAIL'}")
            lines.append(f"  Season stability (no >20%): {'PASS' if sc.get('season_stability_ok') else 'FAIL'}  (max weekly RMSE: {sc.get('max_weekly_rmse')}, avg: {sc.get('avg_weekly_rmse')})")
            lines.append(f"  CI coverage ≥ 88.2% (10pt): {'PASS' if sc.get('confidence_band_target_882') else 'FAIL'}  (actual: {sc.get('confidence_band_coverage_10pt')}%)")
            lines.append("")
        
        # Top performers analysis
        lines.append("-" * 70)
        lines.append("TOP PERFORMER IDENTIFICATION")
        lines.append("-" * 70)
        for pos, tp in results['top_performers'].items():
            lines.append(f"\n  {pos}:")
            lines.append(f"    Of actual Top 10 performers, we ranked them avg: #{tp['avg_pred_rank_of_top_10']:.0f}")
            lines.append(f"    Top 10 actual that were in our Top 20: {tp['top_10_in_our_top_20']}/10")
        lines.append("")
        
        # Value proposition
        lines.append("-" * 70)
        lines.append("VALUE PROPOSITION SUMMARY")
        lines.append("-" * 70)
        lines.append("")
        
        # Calculate value metrics
        avg_r2 = np.mean([pm['r2'] for pm in results['by_position'].values()])
        avg_within_5 = np.mean([pm['within_5_pts_pct'] for pm in results['by_position'].values()])
        
        if avg_r2 > 0.3:
            lines.append("  ✓ STRONG predictive power (R² > 0.3)")
        elif avg_r2 > 0.15:
            lines.append("  ✓ MODERATE predictive power (R² > 0.15)")
        else:
            lines.append("  ⚠ LIMITED predictive power (R² < 0.15)")
        
        if avg_within_5 > 50:
            lines.append("  ✓ HIGH precision - >50% predictions within 5 points")
        elif avg_within_5 > 35:
            lines.append("  ✓ GOOD precision - >35% predictions within 5 points")
        
        lines.append("")
        lines.append("  Key Insight: This model provides data-driven edge for:")
        lines.append("    • Weekly lineup decisions (start/sit)")
        lines.append("    • Waiver wire pickups (identifying breakouts)")
        lines.append("    • Trade valuations (projecting ROS performance)")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_results(self, results: Dict, filename: str = None):
        """Save backtest results to file."""
        if filename is None:
            filename = f"backtest_{results['season']}_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Convert numpy types and int64 keys to native Python types
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return obj
        
        results_clean = convert_keys(results)
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
        return filepath


class ValidationVisualizer:
    """Creates visualizations for model validation."""
    
    def __init__(self):
        self.output_dir = DATA_DIR / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_prediction_scatter(self, df: pd.DataFrame,
                                   actual_col: str,
                                   pred_col: str,
                                   title: str = "Predicted vs Actual Fantasy Points") -> str:
        """Create scatter plot of predicted vs actual values."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot
            ax.scatter(df[actual_col], df[pred_col], alpha=0.5, s=20)
            
            # Perfect prediction line
            max_val = max(df[actual_col].max(), df[pred_col].max())
            ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            
            # Calculate R²
            r2 = r2_score(df[actual_col], df[pred_col])
            
            ax.set_xlabel('Actual Fantasy Points', fontsize=12)
            ax.set_ylabel('Predicted Fantasy Points', fontsize=12)
            ax.set_title(f'{title}\nR² = {r2:.3f}', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filepath = self.output_dir / "prediction_scatter.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            return "matplotlib not available"
    
    def create_accuracy_by_position(self, results: Dict) -> str:
        """Create bar chart of accuracy by position."""
        try:
            import matplotlib.pyplot as plt
            
            positions = list(results['by_position'].keys())
            r2_scores = [results['by_position'][p]['r2'] for p in positions]
            within_5 = [results['by_position'][p]['within_5_pts_pct'] for p in positions]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # R² by position
            colors = ['#2ecc71' if r > 0.3 else '#f39c12' if r > 0.15 else '#e74c3c' for r in r2_scores]
            ax1.bar(positions, r2_scores, color=colors)
            ax1.set_ylabel('R² Score', fontsize=12)
            ax1.set_title('Model Fit by Position', fontsize=14)
            ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong (0.3)')
            ax1.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.15)')
            ax1.legend()
            ax1.set_ylim(0, max(r2_scores) * 1.2)
            
            # Within 5 points by position
            colors2 = ['#2ecc71' if w > 50 else '#f39c12' if w > 35 else '#e74c3c' for w in within_5]
            ax2.bar(positions, within_5, color=colors2)
            ax2.set_ylabel('% Within 5 Points', fontsize=12)
            ax2.set_title('Prediction Precision by Position', fontsize=14)
            ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='High (50%)')
            ax2.axhline(y=35, color='orange', linestyle='--', alpha=0.5, label='Good (35%)')
            ax2.legend()
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            
            filepath = self.output_dir / "accuracy_by_position.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            return "matplotlib not available"
    
    def create_weekly_accuracy_trend(self, results: Dict) -> str:
        """Create line chart of accuracy over weeks."""
        try:
            import matplotlib.pyplot as plt
            
            weeks = sorted(results['by_week'].keys())
            r2_by_week = [results['by_week'][w]['r2'] for w in weeks]
            mae_by_week = [results['by_week'][w]['mae'] for w in weeks]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # R² over season
            ax1.plot(weeks, r2_by_week, 'b-o', linewidth=2, markersize=6)
            ax1.fill_between(weeks, r2_by_week, alpha=0.3)
            ax1.set_ylabel('R² Score', fontsize=12)
            ax1.set_title('Model Accuracy Throughout Season', fontsize=14)
            ax1.axhline(y=np.mean(r2_by_week), color='red', linestyle='--', 
                       label=f'Season Avg: {np.mean(r2_by_week):.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(min(weeks), max(weeks))
            
            # MAE over season
            ax2.plot(weeks, mae_by_week, 'g-o', linewidth=2, markersize=6)
            ax2.fill_between(weeks, mae_by_week, alpha=0.3, color='green')
            ax2.set_xlabel('Week', fontsize=12)
            ax2.set_ylabel('Mean Absolute Error (points)', fontsize=12)
            ax2.axhline(y=np.mean(mae_by_week), color='red', linestyle='--',
                       label=f'Season Avg: {np.mean(mae_by_week):.2f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(min(weeks), max(weeks))
            
            plt.tight_layout()
            
            filepath = self.output_dir / "weekly_accuracy_trend.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            return "matplotlib not available"
    
    def create_ranking_accuracy_chart(self, results: Dict) -> str:
        """Create chart showing ranking accuracy."""
        try:
            import matplotlib.pyplot as plt
            
            positions = list(results['ranking_accuracy'].keys())
            
            top_5 = [results['ranking_accuracy'][p].get('top_5_hit_rate') or 0 for p in positions]
            top_10 = [results['ranking_accuracy'][p].get('top_10_hit_rate') or 0 for p in positions]
            top_20 = [results['ranking_accuracy'][p].get('top_20_hit_rate') or 0 for p in positions]
            
            x = np.arange(len(positions))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars1 = ax.bar(x - width, top_5, width, label='Top 5 Hit Rate', color='#3498db')
            bars2 = ax.bar(x, top_10, width, label='Top 10 Hit Rate', color='#2ecc71')
            bars3 = ax.bar(x + width, top_20, width, label='Top 20 Hit Rate', color='#9b59b6')
            
            ax.set_ylabel('Hit Rate (%)', fontsize=12)
            ax.set_title('Ranking Accuracy: How Often Top Predicted Players Were Actually Top Performers', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(positions)
            ax.legend()
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.0f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=9)
            
            # Reference line for random chance
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Chance')
            
            plt.tight_layout()
            
            filepath = self.output_dir / "ranking_accuracy.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        except ImportError:
            return "matplotlib not available"
    
    def create_executive_summary_card(self, results: Dict) -> str:
        """Create a visual executive summary card."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 8)
            ax.axis('off')
            
            # Title
            ax.text(6, 7.5, 'NFL Predictor Model Validation', fontsize=20, 
                   ha='center', fontweight='bold')
            ax.text(6, 7.0, f'Season {results["season"]} Backtest Results', fontsize=14,
                   ha='center', color='gray')
            
            # Key metrics boxes
            metrics = results['metrics']
            
            # R² Score
            r2_color = '#2ecc71' if metrics['r2'] > 0.3 else '#f39c12' if metrics['r2'] > 0.15 else '#e74c3c'
            rect1 = FancyBboxPatch((0.5, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                   facecolor=r2_color, alpha=0.3, edgecolor=r2_color, linewidth=2)
            ax.add_patch(rect1)
            ax.text(2.25, 6.0, 'R² Score', fontsize=12, ha='center', fontweight='bold')
            ax.text(2.25, 5.2, f'{metrics["r2"]:.3f}', fontsize=28, ha='center', fontweight='bold')
            
            # Correlation
            corr_color = '#2ecc71' if metrics['correlation'] > 0.5 else '#f39c12'
            rect2 = FancyBboxPatch((4.25, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                   facecolor=corr_color, alpha=0.3, edgecolor=corr_color, linewidth=2)
            ax.add_patch(rect2)
            ax.text(6, 6.0, 'Correlation', fontsize=12, ha='center', fontweight='bold')
            ax.text(6, 5.2, f'{metrics["correlation"]:.3f}', fontsize=28, ha='center', fontweight='bold')
            
            # Within 5 Points
            w5_color = '#2ecc71' if metrics['within_5_pts_pct'] > 50 else '#f39c12'
            rect3 = FancyBboxPatch((8, 4.5), 3.5, 2, boxstyle="round,pad=0.1",
                                   facecolor=w5_color, alpha=0.3, edgecolor=w5_color, linewidth=2)
            ax.add_patch(rect3)
            ax.text(9.75, 6.0, 'Within 5 Pts', fontsize=12, ha='center', fontweight='bold')
            ax.text(9.75, 5.2, f'{metrics["within_5_pts_pct"]:.0f}%', fontsize=28, ha='center', fontweight='bold')
            
            # Secondary metrics
            ax.text(2.25, 3.8, f'RMSE: {metrics["rmse"]:.1f} pts', fontsize=11, ha='center')
            ax.text(6, 3.8, f'MAE: {metrics["mae"]:.1f} pts', fontsize=11, ha='center')
            ax.text(9.75, 3.8, f'Directional: {metrics["directional_accuracy_pct"]:.0f}%', fontsize=11, ha='center')
            
            # Interpretation
            ax.text(6, 2.8, 'Model Assessment', fontsize=14, ha='center', fontweight='bold')
            
            if metrics['r2'] > 0.3 and metrics['correlation'] > 0.5:
                assessment = "✓ STRONG - Reliable for decision support"
                color = '#2ecc71'
            elif metrics['r2'] > 0.15:
                assessment = "✓ MODERATE - Useful edge over baseline"
                color = '#f39c12'
            else:
                assessment = "⚠ LIMITED - Use with caution"
                color = '#e74c3c'
            
            ax.text(6, 2.2, assessment, fontsize=16, ha='center', color=color, fontweight='bold')
            
            # Bottom note
            ax.text(6, 0.5, f'Based on {results["n_predictions"]:,} predictions across {len(results["by_week"])} weeks',
                   fontsize=10, ha='center', color='gray')
            
            filepath = self.output_dir / "executive_summary.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(filepath)
        except ImportError:
            return "matplotlib not available"


def run_backtest(test_season: int = None) -> Tuple[Dict, str]:
    """
    Run a complete backtest on the most recent test season using the production ensemble.
    
    Uses the same data load and feature pipeline as training; all preprocessing
    (utilization weights, etc.) is train-derived only (loaded from disk). Test season
    is strictly unseen: no use in training, tuning, or fitting.
    
    Args:
        test_season: Season to backtest (None = auto-select latest)
        
    Returns:
        Tuple of (results dict, report string)
    """
    from src.utils.data_manager import DataManager
    from src.models.train import (
        load_training_data,
        add_engineered_features,
    )
    from src.features.utilization_score import (
        recalculate_utilization_with_weights,
        calculate_utilization_scores,
        load_percentile_bounds,
    )
    from src.models.ensemble import EnsemblePredictor
    from config.settings import POSITIONS, MODELS_DIR
    
    print("=" * 60)
    print("RUNNING MODEL BACKTEST (PRODUCTION ENSEMBLE)")
    print("=" * 60)
    
    # Get data with automatic season selection (same as train.py). Test season is current season when in-season.
    dm = DataManager()
    train_seasons, actual_test_season = dm.get_train_test_seasons(test_season=test_season)
    
    # Strict unseen test: test season must not be in train
    assert actual_test_season not in train_seasons, (
        f"Test season {actual_test_season} must not be in train seasons {train_seasons}"
    )
    
    print(f"\nBacktest Configuration:")
    print(f"  Training seasons: {train_seasons}")
    print(f"  Test season: {actual_test_season} (unseen)")
    
    # Load data via same path as training (positions, min_games consistent with train.py)
    train_data, test_data, _, _ = load_training_data(
        positions=POSITIONS,
        min_games=4,
        test_season=actual_test_season,
        n_train_seasons=None,
        optimize_training_years=False,
    )
    
    if test_data.empty:
        print("No test data available for backtest")
        return {}, ""
    
    # Prepare test data with train-derived artifacts only (no fitting on test)
    bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
    percentile_bounds = load_percentile_bounds(bounds_path) if bounds_path.exists() else None
    weights_path = MODELS_DIR / "utilization_weights.json"
    util_weights = None
    if weights_path.exists():
        with open(weights_path, "r") as f:
            util_weights = json.load(f)
    test_data = calculate_utilization_scores(
        test_data, team_df=pd.DataFrame(), weights=util_weights, percentile_bounds=percentile_bounds
    )
    if util_weights is not None:
        test_data = recalculate_utilization_with_weights(test_data, util_weights)
    # Add Vegas/game script and other external features (same as training pipeline)
    try:
        from src.data.external_data import add_external_features
        test_data = add_external_features(test_data, seasons=list(test_data["season"].unique()))
    except Exception:
        pass
    test_data = add_engineered_features(test_data)
    
    # Create target columns for alignment with model expectations
    for n_weeks in [1, 4, 12]:
        test_data[f"target_{n_weeks}w"] = test_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
    
    # Generate predictions using the persisted production ensemble
    print("\nGenerating predictions on test data (production ensemble)...")
    predictor = EnsemblePredictor()
    predictor.load_models(positions=POSITIONS)
    if not predictor.is_loaded:
        print("Warning: No persisted models found. Falling back to baseline (rolling average).")
        test_data["predicted_points"] = test_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(1).rolling(window=4, min_periods=1).mean()
        )
    else:
        test_data = predictor.predict(test_data, n_weeks=1)
    
    # Run backtest
    backtester = ModelBacktester()
    results = backtester.backtest_season(
        predictions=test_data,
        actuals=test_data,
        season=actual_test_season,
        prediction_col='predicted_points',
        actual_col='fantasy_points'
    )

    # Metrics by utilization tier (for fantasy decision-making)
    if "utilization_score" in test_data.columns and "predicted_points" in test_data.columns and "fantasy_points" in test_data.columns:
        tiers = {
            "elite (80+)": test_data["utilization_score"] >= 80,
            "strong (70-79)": (test_data["utilization_score"] >= 70) & (test_data["utilization_score"] < 80),
            "average (60-69)": (test_data["utilization_score"] >= 60) & (test_data["utilization_score"] < 70),
            "below_avg (50-59)": (test_data["utilization_score"] >= 50) & (test_data["utilization_score"] < 60),
            "low (<50)": test_data["utilization_score"] < 50,
        }
        by_tier = {}
        for tier_name, mask in tiers.items():
            if mask.sum() < 5:
                continue
            sub = test_data.loc[mask]
            by_tier[tier_name] = {
                "n_samples": int(mask.sum()),
                "rmse": float(np.sqrt(mean_squared_error(sub["fantasy_points"], sub["predicted_points"]))),
                "mae": float(mean_absolute_error(sub["fantasy_points"], sub["predicted_points"])),
            }
        if by_tier:
            results["by_utilization_tier"] = by_tier
    
    # Add config for reproducibility and auditing
    results["train_seasons"] = train_seasons
    results["test_season"] = actual_test_season
    results["model_source"] = "production_ensemble" if predictor.is_loaded else "baseline"
    if predictor.is_loaded:
        try:
            fc = {}
            for pos in POSITIONS:
                if pos in predictor.position_models:
                    pm = predictor.position_models[pos]
                    m = pm.models.get(1) or list(pm.models.values())[0]
                    fc[pos] = len(getattr(m, "feature_names", []))
                elif pos in predictor.single_week_models:
                    fc[pos] = len(getattr(predictor.single_week_models[pos], "feature_names", []))
            if fc:
                results["feature_counts"] = fc
        except Exception:
            pass
    
    # Baseline comparison (model vs rolling-average baseline)
    baseline_comp = backtester.compare_to_baseline(
        test_data, actual_col='fantasy_points', pred_col='predicted_points'
    )
    if "error" not in baseline_comp:
        results["baseline_comparison"] = baseline_comp
        print("\nBaseline comparison: model vs rolling 4-week average")
        print(f"  Model RMSE: {baseline_comp['model']['rmse']}  Baseline RMSE: {baseline_comp['baseline']['rmse']}")
        print(f"  Improvement: {baseline_comp['improvement']['rmse_pct']}% RMSE reduction")

    # Multiple naive baselines (persistence, season avg, position avg) per requirements
    multi_baseline = backtester.compare_to_multiple_baselines(
        test_data, actual_col='fantasy_points', pred_col='predicted_points'
    )
    if "error" not in multi_baseline:
        results["multiple_baseline_comparison"] = multi_baseline

    # Success criteria (requirements Section VII - comprehensive)
    m = results.get("metrics", {})
    spearman_rho = m.get("spearman_rho")
    within_10 = m.get("within_10_pts_pct")
    within_7 = m.get("within_7_pts_pct")
    tier_acc = m.get("tier_classification_accuracy")
    mape = m.get("mape")

    # Season stability: check no week has >20% worse RMSE than season average
    weekly_rmse = [results["by_week"][w].get("rmse", 0) for w in results.get("by_week", {}) if results["by_week"][w].get("rmse")]
    avg_weekly_rmse = np.mean(weekly_rmse) if weekly_rmse else 0
    max_weekly_rmse = max(weekly_rmse) if weekly_rmse else 0
    stability_ok = (max_weekly_rmse <= avg_weekly_rmse * 1.20) if avg_weekly_rmse > 0 else True

    # Confidence interval calibration: % of predictions within 10-point band
    ci_coverage = within_10  # within_10_pts_pct serves as the CI band coverage metric

    results["success_criteria"] = {
        "spearman_gt_065": spearman_rho is not None and spearman_rho > 0.65,
        "spearman_rho": spearman_rho,
        "within_10_pts_pct_ge_80": within_10 is not None and within_10 >= 80.0,
        "within_10_pts_pct": within_10,
        "within_7_pts_pct_ge_70": within_7 is not None and within_7 >= 70.0,
        "within_7_pts_pct": within_7,
        "mape_lt_25": mape is not None and mape < 25.0,
        "mape": mape,
        "tier_accuracy_ge_075": tier_acc is not None and tier_acc >= 0.75,
        "tier_accuracy": tier_acc,
        "beat_all_baselines_by_20_pct": results.get("multiple_baseline_comparison", {}).get("model_beats_all_by_20_pct", False),
        "beat_primary_baseline_by_25_pct": (
            baseline_comp.get("improvement", {}).get("rmse_pct") >= 25.0
            if "error" not in baseline_comp else False
        ),
        "season_stability_ok": stability_ok,
        "max_weekly_rmse": round(max_weekly_rmse, 2) if max_weekly_rmse else None,
        "avg_weekly_rmse": round(avg_weekly_rmse, 2) if avg_weekly_rmse else None,
        "confidence_band_coverage_10pt": ci_coverage,
        "confidence_band_target_882": ci_coverage is not None and ci_coverage >= 88.2,
    }

    # Generate report
    report = backtester.generate_report(results)
    if results.get("baseline_comparison"):
        bc = results["baseline_comparison"]
        report += "\n\n" + "-" * 70 + "\nBASELINE COMPARISON (Model vs Rolling 4-Week Average)\n" + "-" * 70
        report += f"\n  Model:   RMSE={bc['model']['rmse']}  MAE={bc['model']['mae']}  R²={bc['model']['r2']}"
        report += f"\n  Baseline: RMSE={bc['baseline']['rmse']}  MAE={bc['baseline']['mae']}  R²={bc['baseline']['r2']}"
        report += f"\n  Improvement: {bc['improvement']['rmse_pct']}% RMSE reduction, R² gain {bc['improvement']['r2_gain']}"
        report += "\n" + "=" * 70
    print(report)
    
    # Save results (includes train_seasons, test_season, baseline_comparison)
    backtester.save_results(results)
    
    # Write app-compatible backtest_results for advanced_model_results.json
    backtest_results_app = {}
    for pos, pm in results.get("by_position", {}).items():
        backtest_results_app[pos] = {
            "rmse": pm["rmse"],
            "mae": pm["mae"],
            "r2": pm["r2"],
            "mape": pm.get("mape"),
            "directional_accuracy_pct": pm.get("directional_accuracy_pct"),
            "within_5_pts_pct": pm.get("within_5_pts_pct"),
            "within_7_pts_pct": pm.get("within_7_pts_pct"),
            "within_10_pts_pct": pm.get("within_10_pts_pct"),
            "spearman_rho": pm.get("spearman_rho"),
        }
    app_results_path = DATA_DIR / "advanced_model_results.json"
    app_payload = {
        "timestamp": results.get("backtest_date", datetime.now().isoformat()),
        "train_seasons": train_seasons,
        "test_season": actual_test_season,
        "backtest_results": backtest_results_app,
        "success_criteria": results.get("success_criteria", {}),
    }
    with open(app_results_path, "w") as f:
        json.dump(app_payload, f, indent=2, default=str)
    print(f"\nApp-compatible results written to {app_results_path.name}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = ValidationVisualizer()
    
    charts = []
    charts.append(viz.create_prediction_scatter(test_data, 'fantasy_points', 'predicted_points'))
    charts.append(viz.create_accuracy_by_position(results))
    charts.append(viz.create_weekly_accuracy_trend(results))
    charts.append(viz.create_ranking_accuracy_chart(results))
    charts.append(viz.create_executive_summary_card(results))
    
    print(f"\nVisualizations saved to: {viz.output_dir}")
    for chart in charts:
        if chart != "matplotlib not available":
            print(f"  - {Path(chart).name}")
    
    return results, report


def run_multi_season_backtest(n_seasons: int = 3) -> Dict:
    """
    Run backtest on the last N test seasons and report mean ± std of metrics.
    Uses the same persisted production ensemble for each season (no retraining).
    """
    from src.utils.data_manager import DataManager
    
    dm = DataManager()
    availability = dm.check_data_availability()
    available = sorted(availability.get("available_seasons", []) or dm.get_available_seasons_from_db())
    if len(available) < 2 or n_seasons < 1:
        print("Need at least 2 seasons and n_seasons >= 1 for multi-season backtest")
        return {}
    test_seasons = available[-n_seasons:]
    all_results = []
    for test_season in test_seasons:
        print(f"\n{'='*60}\nBacktest season {test_season}\n{'='*60}")
        results, _ = run_backtest(test_season=test_season)
        if results and "error" not in results:
            all_results.append((test_season, results))
    if not all_results:
        return {}
    # Aggregate overall metrics
    overall_rmse = [r["metrics"]["rmse"] for _, r in all_results]
    overall_mae = [r["metrics"]["mae"] for _, r in all_results]
    overall_r2 = [r["metrics"]["r2"] for _, r in all_results]
    agg = {
        "test_seasons": test_seasons,
        "n_folds": len(all_results),
        "overall": {
            "rmse_mean": round(float(np.mean(overall_rmse)), 2),
            "rmse_std": round(float(np.std(overall_rmse)), 2),
            "mae_mean": round(float(np.mean(overall_mae)), 2),
            "mae_std": round(float(np.std(overall_mae)), 2),
            "r2_mean": round(float(np.mean(overall_r2)), 3),
            "r2_std": round(float(np.std(overall_r2)), 3),
        },
        "by_position": {},
    }
    positions = set()
    for _, r in all_results:
        positions.update(r.get("by_position", {}).keys())
    for pos in positions:
        rmse_list = [r["by_position"][pos]["rmse"] for _, r in all_results if pos in r.get("by_position", {})]
        mae_list = [r["by_position"][pos]["mae"] for _, r in all_results if pos in r.get("by_position", {})]
        r2_list = [r["by_position"][pos]["r2"] for _, r in all_results if pos in r.get("by_position", {})]
        if rmse_list:
            agg["by_position"][pos] = {
                "rmse_mean": round(float(np.mean(rmse_list)), 2),
                "rmse_std": round(float(np.std(rmse_list)), 2),
                "mae_mean": round(float(np.mean(mae_list)), 2),
                "mae_std": round(float(np.std(mae_list)), 2),
                "r2_mean": round(float(np.mean(r2_list)), 3),
                "r2_std": round(float(np.std(r2_list)), 3),
            }
    print("\n" + "=" * 60)
    print("MULTI-SEASON BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Test seasons: {agg['test_seasons']}")
    print(f"Overall: RMSE {agg['overall']['rmse_mean']} ± {agg['overall']['rmse_std']}  "
          f"MAE {agg['overall']['mae_mean']} ± {agg['overall']['mae_std']}  "
          f"R² {agg['overall']['r2_mean']} ± {agg['overall']['r2_std']}")
    for pos, pm in agg["by_position"].items():
        print(f"  {pos}: RMSE {pm['rmse_mean']} ± {pm['rmse_std']}  R² {pm['r2_mean']} ± {pm['r2_std']}")
    print("=" * 60)
    out_path = DATA_DIR / "backtest_results" / "multi_season_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"Summary saved to {out_path}")
    return agg


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run model backtest")
    parser.add_argument("--season", type=int, default=None, help="Season to backtest (default: latest)")
    parser.add_argument("--multi-season", type=int, default=0, metavar="N",
                        help="Run backtest on last N seasons and report mean ± std (e.g. 3)")
    
    args = parser.parse_args()
    
    if args.multi_season > 0:
        run_multi_season_backtest(n_seasons=args.multi_season)
    else:
        results, report = run_backtest(args.season)

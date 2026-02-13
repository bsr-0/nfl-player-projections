"""
Performance Tracking System
Tracks model accuracy week-over-week and displays trending performance.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PerformanceTracker:
    """
    Tracks prediction accuracy by comparing predictions vs actuals.
    Maintains historical performance metrics.
    """
    
    def __init__(self, db_path: str = "../data/nfl_data.db"):
        self.db_path = Path(db_path)
        self.performance_table = "model_performance"
        self._initialize_performance_table()
    
    def _initialize_performance_table(self):
        """Create performance tracking table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.performance_table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position TEXT NOT NULL,
                horizon TEXT NOT NULL,
                season INTEGER NOT NULL,
                week INTEGER NOT NULL,
                mae REAL,
                rmse REAL,
                r2_score REAL,
                accuracy_pct REAL,
                n_predictions INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(position, horizon, season, week)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def calculate_metrics(
        self, 
        predictions: pd.Series, 
        actuals: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Remove NaN values
        mask = ~(predictions.isna() | actuals.isna())
        preds = predictions[mask]
        acts = actuals[mask]
        
        if len(preds) == 0:
            return {
                'mae': np.nan,
                'rmse': np.nan,
                'r2_score': np.nan,
                'accuracy_pct': np.nan,
                'n': 0
            }
        
        # Mean Absolute Error
        mae = np.mean(np.abs(preds - acts))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((preds - acts) ** 2))
        
        # R² Score
        ss_res = np.sum((acts - preds) ** 2)
        ss_tot = np.sum((acts - acts.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Accuracy (within 10 points)
        within_10 = np.sum(np.abs(preds - acts) <= 10)
        accuracy_pct = (within_10 / len(preds)) * 100
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2_score': round(r2, 3),
            'accuracy_pct': round(accuracy_pct, 1),
            'n': len(preds)
        }
    
    def get_current_accuracy_dashboard(self) -> pd.DataFrame:
        """
        Get current accuracy metrics for all positions/horizons.
        Last 4 weeks average.
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT 
            position,
            horizon,
            AVG(mae) as avg_mae,
            AVG(rmse) as avg_rmse,
            AVG(r2_score) as avg_r2,
            AVG(accuracy_pct) as avg_accuracy,
            SUM(n_predictions) as total_predictions,
            COUNT(*) as weeks_tracked
        FROM {self.performance_table}
        WHERE season >= (SELECT MAX(season) FROM {self.performance_table})
            AND week >= (SELECT MAX(week) FROM {self.performance_table}) - 4
        GROUP BY position, horizon
        ORDER BY position, horizon
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Round for display
        for col in ['avg_mae', 'avg_rmse', 'avg_r2', 'avg_accuracy']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df
    
    def generate_performance_report(self) -> str:
        """Generate a formatted performance report."""
        dashboard = self.get_current_accuracy_dashboard()
        
        report = ["="*60]
        report.append("MODEL PERFORMANCE REPORT (Last 4 Weeks)")
        report.append("="*60)
        
        if dashboard.empty:
            report.append("\n⚠️  No performance data available yet.")
            report.append("Run predictions and actuals comparison first.")
            return "\n".join(report)
        
        for _, row in dashboard.iterrows():
            pos = row['position']
            hor = row['horizon']
            
            report.append(f"\n{pos} ({hor}):")
            report.append(f"  • R² Score: {row['avg_r2']:.3f}")
            report.append(f"  • MAE: {row['avg_mae']:.1f} points")
            report.append(f"  • Accuracy (±10): {row['avg_accuracy']:.1f}%")
            report.append(f"  • Predictions: {row['total_predictions']}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    tracker = PerformanceTracker()
    print(tracker.generate_performance_report())

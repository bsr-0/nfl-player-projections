"""
Utilization Score Weight Optimizer - delegates to utilization_weight_optimizer.

This module provides a backward-compatible UtilizationWeightOptimizer interface
that delegates to the unified utilization_weight_optimizer (train-only temporal
split, optional alpha CV, target_util_1w or target_1w). Weights are persisted to
config MODELS_DIR/utilization_weights.json for train/serve consistency.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import UTILIZATION_WEIGHTS, MODELS_DIR

from src.features.utilization_weight_optimizer import (
    fit_utilization_weights,
    get_utilization_weights,
    _get_default_weights,
)


class UtilizationWeightOptimizer:
    """
    Backward-compatible wrapper that delegates to utilization_weight_optimizer.
    Uses train-only temporal split; no hardcoded year splits.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or (MODELS_DIR / "utilization_weights.json")
        self.optimized_weights = {}
        self.optimization_stats = {}

    def optimize_weights(self, df, position: str, horizon: int = 1) -> Dict[str, float]:
        """
        Optimize utilization weights for one position using unified pipeline.
        Delegates to fit_utilization_weights (train-only; target_util_1w or target_1w).
        If df lacks utilization components or target, returns defaults.
        """
        target_col = "target_util_1w" if "target_util_1w" in df.columns else "target_1w"
        if target_col not in df.columns:
            pos_weights = self._get_default_weights(position)
            self.optimized_weights[position] = pos_weights
            return pos_weights
        all_weights = fit_utilization_weights(
            df, target_col=target_col, min_samples=100, tune_alpha_cv=True
        )
        pos_weights = all_weights.get(position)
        if pos_weights is None:
            pos_weights = self._get_default_weights(position)
        self.optimized_weights[position] = pos_weights
        self.optimization_stats[position] = {
            "horizon": horizon,
            "target_col": target_col,
            "optimized_at": datetime.now().isoformat(),
        }
        return pos_weights

    def _get_default_weights(self, position: str) -> Dict[str, float]:
        """Return config defaults (matches utilization_score key names)."""
        return UTILIZATION_WEIGHTS.get(position, UTILIZATION_WEIGHTS["RB"]).copy()

    def save_weights(self):
        """Save optimized weights to cache file."""
        data = {
            'weights': self.optimized_weights,
            'stats': self.optimization_stats,
            'saved_at': datetime.now().isoformat()
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_weights(self) -> bool:
        """Load cached weights from MODELS_DIR or legacy cache path."""
        for path in [self.cache_path, MODELS_DIR / "utilization_weights.json"]:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    if "weights" in data:
                        self.optimized_weights = data.get("weights", {})
                        self.optimization_stats = data.get("stats", {})
                    else:
                        self.optimized_weights = {k: v for k, v in data.items() if isinstance(v, dict) and k in ["QB", "RB", "WR", "TE"]}
                        self.optimization_stats = {}
                    return bool(self.optimized_weights)
                except Exception:
                    pass
        return False
    
    def get_weights_for_display(self, position: str) -> Dict:
        """Get weights and stats formatted for display."""
        weights = self.optimized_weights.get(position, self._get_default_weights(position))
        stats = self.optimization_stats.get(position, {})
        
        return {
            'weights': weights,
            'stats': stats,
            'is_optimized': position in self.optimized_weights
        }


def optimize_all_positions(df: pd.DataFrame, horizon: int = 4) -> Dict[str, Dict]:
    """
    Convenience function to optimize weights for all positions.
    
    Args:
        df: DataFrame with player stats
        horizon: Forecast horizon (default 4 for rolling 4-week - most predictive)
        
    Returns:
        Dict with optimized weights for each position
    """
    optimizer = UtilizationWeightOptimizer()
    
    results = {}
    for position in ['RB', 'WR', 'TE']:
        weights = optimizer.optimize_weights(df, position, horizon)
        results[position] = {
            'weights': weights,
            'stats': optimizer.optimization_stats.get(position, {})
        }
    
    # Save to cache
    optimizer.save_weights()
    
    return results

"""
Data-driven optimization of Utilization Score component weights.

Fits optimal weights from TRAINING data only (temporal split; no test data).
Objective options:
  - Option A: Maximize correlation with future fantasy points (target_1w).
  - Option B: Maximize correlation with future utilization (target_util_1w). Default.

Uses non-negative least squares (NNLS) with L2 regularization. Weights sum to 1
per position. Optional inner time-series CV to tune alpha.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import UTILIZATION_WEIGHTS, MODELS_DIR
from src.utils.helpers import safe_divide


# Component columns per position (must match UtilizationScoreCalculator output)
UTIL_COMPONENTS = {
    "RB": ["snap_share_pct", "rush_share_pct", "target_share_pct", "redzone_share_pct", "touch_share_pct", "high_value_touch_rate"],
    "WR": ["target_share_pct", "air_yards_share_pct", "snap_share_pct", "redzone_targets_pct", "route_participation_pct", "high_value_touch_rate"],
    "TE": ["target_share_pct", "snap_share_pct", "redzone_targets_pct", "air_yards_share_pct", "inline_rate_pct", "high_value_touch_rate"],
    "QB": ["dropback_rate_pct", "rush_share_pct", "redzone_opp_pct", "play_volume_pct"],
}

KEY_MAP = {
    "RB": {"snap_share_pct": "snap_share", "rush_share_pct": "rush_share",
           "target_share_pct": "target_share", "redzone_share_pct": "redzone_share",
           "touch_share_pct": "touch_share", "high_value_touch_rate": "high_value_touch"},
    "WR": {"target_share_pct": "target_share", "air_yards_share_pct": "air_yards_share",
           "snap_share_pct": "snap_share", "redzone_targets_pct": "redzone_targets",
           "route_participation_pct": "route_participation", "high_value_touch_rate": "high_value_touch"},
    "TE": {"target_share_pct": "target_share", "snap_share_pct": "snap_share",
           "redzone_targets_pct": "redzone_targets", "air_yards_share_pct": "air_yards_share",
           "inline_rate_pct": "inline_rate", "high_value_touch_rate": "high_value_touch"},
    "QB": {"dropback_rate_pct": "dropback_rate", "rush_share_pct": "rush_attempt_share",
           "redzone_opp_pct": "redzone_opportunity", "play_volume_pct": "play_volume"},
}


def _get_default_weights() -> Dict:
    """Return config defaults as fallback."""
    return {k: v.copy() if isinstance(v, dict) else v for k, v in UTILIZATION_WEIGHTS.items()}


def _fit_one_position(
    pos_df: pd.DataFrame,
    position: str,
    available: List[str],
    target_col: str,
    alpha: float,
) -> Optional[Dict[str, float]]:
    """Fit weights for one position. Returns dict of component -> weight or None."""
    from scipy.optimize import nnls

    X = pos_df[available].fillna(0).replace([np.inf, -np.inf], 0)
    y = pos_df[target_col]
    valid = y.notna() & (np.isfinite(y)) & (y >= 0)
    if valid.sum() < 100:
        return None
    X = X[valid].values
    y = y[valid].values
    X_scaled = np.clip(X, 0, 100)
    XtX = X_scaled.T @ X_scaled + alpha * np.eye(len(available))
    Xty = X_scaled.T @ y
    try:
        weights_raw, _ = nnls(XtX, Xty)
    except Exception:
        return None
    if weights_raw.sum() <= 0:
        return None
    weights_norm = weights_raw / weights_raw.sum()
    pos_keys = KEY_MAP.get(position, {})
    return {pos_keys.get(col, col): float(weights_norm[i]) for i, col in enumerate(available)}


def _tune_alpha_time_series_cv(
    pos_df: pd.DataFrame,
    position: str,
    available: List[str],
    target_col: str,
    alphas: List[float],
    n_splits: int = 3,
) -> float:
    """Tune alpha via time-series CV (by season). Returns best alpha."""
    from sklearn.metrics import mean_squared_error

    seasons = sorted(pos_df["season"].unique())
    if len(seasons) < n_splits + 1:
        return 1.0
    test_seasons = seasons[-n_splits:]
    best_alpha, best_rmse = 1.0, np.inf
    for alpha in alphas:
        rmses = []
        for test_season in test_seasons:
            train_df = pos_df[pos_df["season"] < test_season]
            test_df = pos_df[pos_df["season"] == test_season]
            if len(train_df) < 100 or len(test_df) < 20:
                continue
            w = _fit_one_position(train_df, position, available, target_col, alpha)
            if w is None:
                continue
            # Predict test: composite = sum(weight * component) for available cols
            comp_cols = [c for c in available if c in test_df.columns]
            pred = np.zeros(len(test_df))
            for i, col in enumerate(comp_cols):
                key = KEY_MAP.get(position, {}).get(col, col)
                if key in w:
                    pred += w[key] * test_df[col].fillna(0).values
            pred = np.clip(pred, 0, 100)
            y_true = np.asarray(test_df[target_col].values, dtype=np.float64)
            # Drop NaN/inf so sklearn does not raise
            finite = np.isfinite(y_true) & np.isfinite(pred)
            if np.sum(finite) < 10:
                continue
            rmses.append(np.sqrt(mean_squared_error(y_true[finite], pred[finite])))
        if rmses and np.mean(rmses) < best_rmse:
            best_rmse = np.mean(rmses)
            best_alpha = alpha
    return best_alpha


def fit_utilization_weights(
    train_data: pd.DataFrame,
    target_col: str = "target_util_1w",
    min_samples: int = 200,
    alpha: float = 1.0,
    tune_alpha_cv: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Fit optimal utilization component weights from training data only (no test data).

    Objective: Option B by default - maximize correlation with future utilization
    (target_util_1w). Set target_col="target_1w" for Option A (future fantasy points).

    Uses NNLS with L2 regularization. Weights sum to 1 per position. Fit only on
    train_data (temporal split is caller's responsibility).

    Args:
        train_data: Training DataFrame (train seasons only)
        target_col: Target column: "target_util_1w" (future util) or "target_1w" (future FP)
        min_samples: Minimum samples per position to fit
        alpha: Ridge regularization strength
        tune_alpha_cv: If True, tune alpha via inner time-series CV

    Returns:
        Dict mapping position -> dict of component -> weight
    """
    result = _get_default_weights()

    # Prefer utilization target if not specified
    if target_col not in train_data.columns and "target_util_1w" in train_data.columns:
        target_col = "target_util_1w"
    elif target_col not in train_data.columns and "target_1w" in train_data.columns:
        target_col = "target_1w"
    if target_col not in train_data.columns:
        return result

    for position in ["RB", "WR", "TE", "QB"]:
        components = UTIL_COMPONENTS.get(position, [])
        if not components:
            continue

        pos_df = train_data[train_data["position"] == position].copy()
        if len(pos_df) < min_samples:
            continue

        available = [c for c in components if c in pos_df.columns]
        if len(available) < 2:
            continue

        if tune_alpha_cv and "season" in pos_df.columns:
            alpha_use = _tune_alpha_time_series_cv(
                pos_df, position, available, target_col,
                alphas=[0.1, 0.5, 1.0, 2.0, 5.0],
                n_splits=3,
            )
        else:
            alpha_use = alpha

        weights_dict = _fit_one_position(pos_df, position, available, target_col, alpha_use)
        if weights_dict is not None:
            result[position] = weights_dict

    return result


def get_utilization_weights(
    train_data: Optional[pd.DataFrame] = None,
    use_data_driven: bool = True,
    target_col: str = "target_util_1w",
) -> Dict[str, Dict[str, float]]:
    """
    Get utilization weights - data-driven if available, else config defaults.

    Uses same train-only temporal discipline; no test data in fitting.

    Args:
        train_data: Optional training data for fitting
        use_data_driven: If True and train_data provided, fit weights from data
        target_col: Target for weight regression (target_util_1w or target_1w)

    Returns:
        Position -> component -> weight dict
    """
    if use_data_driven and train_data is not None and len(train_data) > 500:
        try:
            weights = fit_utilization_weights(train_data, target_col=target_col)
            if any(
                sum(w.values()) > 0.5 for pos, w in weights.items() if isinstance(w, dict)
            ):
                return weights
        except Exception:
            pass
    return _get_default_weights()

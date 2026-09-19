"""Minimal fixed-grid hyperparameter search for the game-outcome/margin models.

Not Optuna (see config.settings.GAME_OUTCOME_MODEL_CONFIG's phase-1/2
fixed-hyperparameter rationale) -- a small, explicit grid is enough to check
whether tuning moves the needle at all before investing in a real search.

Leakage-safety approach: tuned ONCE on data strictly before the earliest
walk-forward test season, then the winning hyperparameters are reused FIXED
across every outer walk-forward fold -- never re-tuned per fold. This is a
deliberate simplification versus full per-fold re-tuning (which would cost
several times more training for a "minimal" check without changing the
leakage story: both approaches only ever tune on data strictly before
whichever season is ultimately being scored, since the earliest test season
precedes every other test season in the walk-forward by construction).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.models.position_models import SeasonAwareTimeSeriesSplit

# Small, explicit grids -- kept deliberately short (a handful of combos per
# model) so a "minimal" tuning pass runs in seconds on a few thousand rows,
# not to find a global optimum.
CLASSIFIER_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "logistic": {"C": [0.1, 1.0, 10.0]},
    "xgboost": {"max_depth": [3, 4], "n_estimators": [200, 400]},
    "random_forest": {"max_depth": [4, 6], "min_samples_leaf": [10, 30]},
}

REGRESSOR_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "ridge": {"alpha": [1.0, 10.0, 30.0, 100.0]},
    "xgboost": {"max_depth": [3, 4], "n_estimators": [200, 400]},
    "random_forest": {"max_depth": [4, 6], "min_samples_leaf": [10, 30]},
}


def _inner_cv_score(
    build_model: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_splits: int,
) -> float:
    """Mean scorer value across an inner walk-forward CV (gap_seasons=0,
    same rationale as the outer harness: features are already lagged
    within-season, so no purge gap is needed here either)."""
    cv = SeasonAwareTimeSeriesSplit(n_splits=n_splits, seasons=seasons, gap_seasons=0)
    scores = []
    for train_idx, test_idx in cv.split(X):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores.append(scorer(model, X.iloc[test_idx], y[test_idx]))
    return float(np.mean(scores)) if scores else float("inf")


def grid_search(
    param_grid: Dict[str, Sequence],
    build_model: Callable[..., object],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_splits: int = 3,
) -> Dict:
    """Try every combination in `param_grid` (name -> candidate values) via
    inner walk-forward CV; lower `scorer` output is better. Returns the best
    params/score plus every combination tried, for a printed audit trail."""
    keys = list(param_grid)
    combos = list(itertools.product(*param_grid.values()))
    all_results: List[Dict] = []
    best_params: Dict = {}
    best_score = float("inf")
    for combo in combos:
        params = dict(zip(keys, combo))
        score = _inner_cv_score(lambda p=params: build_model(**p), X, y, seasons, scorer, n_splits)
        all_results.append({"params": params, "score": score})
        if score < best_score:
            best_score, best_params = score, params
    return {"best_params": best_params, "best_score": best_score, "all_results": all_results}


def tune_all_arms(
    grids: Dict[str, Dict[str, Sequence]],
    arm_classes: Dict[str, Callable[..., object]],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_splits: int = 3,
) -> Dict[str, dict]:
    """Run `grid_search` for every arm named in both `grids` and `arm_classes`.

    Returns {arm_name: best_params}, directly usable as the `tuned_params`
    argument to either backtester's `run_walk_forward_backtest`.
    """
    tuned: Dict[str, dict] = {}
    for name, grid in grids.items():
        if name not in arm_classes:
            continue
        result = grid_search(grid, arm_classes[name], X, y, seasons, scorer, n_splits)
        tuned[name] = result["best_params"]
        print(f"  tuned {name}: {result['best_params']} (inner-CV score {result['best_score']:.4f})")
    return tuned

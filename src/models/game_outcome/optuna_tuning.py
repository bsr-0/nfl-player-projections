"""Optuna-based hyperparameter tuning for the game-outcome/margin models --
the real search deferred at phase-1 planning time in favor of
`tuning.py`'s minimal fixed grid ("no Optuna tuning in phase 1... a natural
fast-follow using the existing `_tune_*`/`objective(trial)` pattern already
used for the fantasy regressors").

Mirrors src/models/single_week_ppr/tuning.py's conventions: TPESampler with
a fixed seed for reproducibility, optuna's own logging silenced to WARNING
so a multi-hundred-trial run doesn't spam stdout, one inner walk-forward CV
(via the same SeasonAwareTimeSeriesSplit gap_seasons=0 rationale as
tuning.py's `_inner_cv_score`) scored per trial.

Same leakage discipline as tuning.py: the caller is responsible for handing
this ONLY data strictly before the outer walk-forward's earliest test
season (see train_game_outcome_model.py/train_game_margin_model.py's
`_tune`/`_tune_optuna`) -- tuned once, then the winning params are reused
FIXED across every outer fold, never re-tuned per fold.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler

from src.models.position_models import SeasonAwareTimeSeriesSplit

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Wider, continuous search spaces than tuning.py's fixed grid -- that's the
# whole point of using Optuna instead of a handful of hand-picked combos.
# xgboost/random_forest spaces are shared between classification and
# regression (same underlying capacity knobs); only `logistic`/`ridge`
# differ since they're different estimators entirely.
_XGB_SPACE: Callable[["optuna.Trial"], dict] = lambda trial: {
    "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
    "max_depth": trial.suggest_int("max_depth", 2, 6),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
}
_RF_SPACE: Callable[["optuna.Trial"], dict] = lambda trial: {
    "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
    "max_depth": trial.suggest_int("max_depth", 2, 10),
    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
}

CLASSIFIER_SEARCH_SPACES: Dict[str, Callable] = {
    "logistic": lambda trial: {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)},
    "xgboost": _XGB_SPACE,
    "random_forest": _RF_SPACE,
}

REGRESSOR_SEARCH_SPACES: Dict[str, Callable] = {
    "ridge": lambda trial: {"alpha": trial.suggest_float("alpha", 0.1, 300.0, log=True)},
    "xgboost": _XGB_SPACE,
    "random_forest": _RF_SPACE,
}


def _inner_cv_score(
    build_model: Callable[[], object],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_splits: int,
) -> float:
    """Same inner walk-forward CV as tuning.py's `_inner_cv_score` -- kept
    as a separate (small, duplicated) copy rather than importing from
    tuning.py, since the two tuners have no other shared state and
    importing one from the other would just add an arbitrary direction to
    an otherwise-independent dependency."""
    cv = SeasonAwareTimeSeriesSplit(n_splits=n_splits, seasons=seasons, gap_seasons=0)
    scores = []
    for train_idx, test_idx in cv.split(X):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = build_model()
        model.fit(X.iloc[train_idx], y[train_idx])
        scores.append(scorer(model, X.iloc[test_idx], y[test_idx]))
    return float(np.mean(scores)) if scores else float("inf")


def tune_arm_optuna(
    search_space_fn: Callable[["optuna.Trial"], dict],
    arm_cls: Callable[..., object],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_trials: int,
    n_splits: int,
    seed: int = 42,
) -> Dict:
    """One arm's Optuna study. Lower `scorer` output is better (log-loss or
    MAE, matching tuning.py's convention). Returns best_params/best_score
    plus the full trials dataframe for an audit trail."""

    def objective(trial: "optuna.Trial") -> float:
        params = search_space_fn(trial)
        return _inner_cv_score(lambda p=params: arm_cls(**p), X, y, seasons, scorer, n_splits)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": study.best_params, "best_score": study.best_value, "n_trials": len(study.trials)}


def tune_all_arms_optuna(
    search_spaces: Dict[str, Callable],
    arm_classes: Dict[str, Callable[..., object]],
    X: pd.DataFrame,
    y: np.ndarray,
    seasons: np.ndarray,
    scorer: Callable[[object, pd.DataFrame, np.ndarray], float],
    n_trials: int = 40,
    n_splits: int = 3,
) -> Dict[str, dict]:
    """Run `tune_arm_optuna` for every arm named in both `search_spaces` and
    `arm_classes`. Returns {arm_name: best_params}, directly usable as the
    `tuned_params` argument to either backtester's `run_walk_forward_backtest`
    -- same shape tuning.py's `tune_all_arms` returns, so callers can swap
    between the two tuners without other code changes."""
    tuned: Dict[str, dict] = {}
    for name, space_fn in search_spaces.items():
        if name not in arm_classes:
            continue
        result = tune_arm_optuna(space_fn, arm_classes[name], X, y, seasons, scorer, n_trials, n_splits)
        tuned[name] = result["best_params"]
        print(
            f"  [optuna, {n_trials} trials] tuned {name}: {result['best_params']} "
            f"(inner-CV score {result['best_score']:.4f})"
        )
    return tuned

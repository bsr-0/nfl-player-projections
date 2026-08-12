"""Phase 5 (next_focus.md) hyperparameter tuning for the FINAL_CONFIG
architectures chosen in Phases 2-3.

Deliberately does NOT reuse src/models/position_models.py's
_subsample_for_tuning trick (60% recent + 40% random-older row cap) —
that's already flagged as a methodology concern in GAPS.md §7.4, and
Optuna trials here are cheap refits on already-loaded data (no repeated
feature engineering), so there's no need to subsample for speed.

Search space mirrors position_models.py:_tune_lightgbm (lines 974-1020)
since that's the established, already-tuned range for this problem — not
reinvented from scratch.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

from config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_walk_forward_folds(
    train_seasons: Sequence[int], n_folds: int = 3
) -> List[Tuple[List[int], int]]:
    """Expanding-window inner folds within train_seasons only — the outer
    test season (from run_fold) is never touched by this function.

    Each fold: (inner_train_seasons, inner_val_season), with a purge gap
    (MODEL_CONFIG["cv_gap_seasons"]) of seasons dropped from the end of
    inner_train immediately before inner_val, mirroring the existing
    SeasonAwareTimeSeriesSplit gap-purge idea (position_models.py).
    Degrades gracefully (fewer folds) when train_seasons is short.
    """
    seasons = sorted(train_seasons)
    gap = MODEL_CONFIG.get("cv_gap_seasons", 1)
    folds: List[Tuple[List[int], int]] = []
    # Candidate validation seasons: the last n_folds seasons, each needing
    # at least one season of inner-train history before the gap.
    candidates = seasons[-n_folds:] if len(seasons) > n_folds else seasons[1:]
    for val_season in candidates:
        inner_train = [s for s in seasons if s <= val_season - 1 - gap]
        if not inner_train:
            continue
        folds.append((inner_train, val_season))
    return folds


def _gbm_search_space(trial: "optuna.Trial", tune_huber_alpha: bool) -> dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    if tune_huber_alpha:
        params["alpha"] = trial.suggest_float("alpha", 0.5, 0.95)
    return params


def _build_model(architecture_name: str, params: dict):
    from src.models.single_week_ppr.architectures import GBMRegressor, YeoJohnsonHuber

    if architecture_name == "C_gbm_mae":
        return GBMRegressor(objective="regression_l1", **params)
    if architecture_name == "B_gbm_huber":
        return GBMRegressor(objective="huber", **params)
    if architecture_name == "F_yeojohnson_huber":
        return YeoJohnsonHuber(**params)
    raise ValueError(f"Tuning not implemented for architecture: {architecture_name!r}")


def tune_hyperparameters_for_fold(
    architecture_name: str,
    X_train,
    y_train,
    seasons_train,
    sample_weight: Optional[np.ndarray] = None,
    n_trials: int = 100,
    n_inner_folds: int = 3,
) -> Dict:
    """Returns the best hyperparameter dict found via nested walk-forward CV
    within the fold's training seasons only. Falls back to {} (== defaults)
    if there isn't enough history for even one inner fold.
    """
    inner_folds = inner_walk_forward_folds(sorted(set(seasons_train)), n_folds=n_inner_folds)
    if not inner_folds:
        logger.warning("No inner folds available for %s (train_seasons=%s) — using defaults",
                        architecture_name, sorted(set(seasons_train)))
        return {}

    tune_huber_alpha = architecture_name in ("F_yeojohnson_huber", "B_gbm_huber")
    seasons_arr = np.asarray(seasons_train)

    def objective(trial: "optuna.Trial") -> float:
        params = _gbm_search_space(trial, tune_huber_alpha)
        fold_mae = []
        for inner_train_seasons, val_season in inner_folds:
            train_mask = np.isin(seasons_arr, inner_train_seasons)
            val_mask = seasons_arr == val_season
            if train_mask.sum() < 20 or val_mask.sum() < 5:
                continue
            model = _build_model(architecture_name, params)
            w = sample_weight[train_mask] if sample_weight is not None else None
            model.fit(X_train[train_mask], y_train[train_mask], sample_weight=w)
            preds = model.predict(X_train[val_mask])
            fold_mae.append(mean_absolute_error(y_train[val_mask], preds))
        if not fold_mae:
            return float("inf")
        return float(np.mean(fold_mae))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

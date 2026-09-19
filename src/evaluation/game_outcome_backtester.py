"""Walk-forward evaluation for the game-outcome model.

Separate from src/evaluation/backtester.py (ModelBacktester) because that
harness is regression-only throughout (RMSE/MAE/R2, no accuracy/log-loss/
ROC-AUC/Brier anywhere) and is 2600 lines of fantasy-points-specific report
generation -- a new sibling module is cleaner than bending it to classification.

Reuses `SeasonAwareTimeSeriesSplit` (src/models/position_models.py) directly:
it is already model-agnostic, and is the house standard for expanding-window,
whole-season-fold CV that never splits a season across train/test.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from config.settings import GAME_OUTCOME_MODEL_CONFIG
from src.models.game_outcome.baseline import HomeFieldBaseline, VegasFavoriteBaseline
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.models.game_outcome.models import GameOutcomeLogisticModel, GameOutcomeRFModel, GameOutcomeXGBModel
from src.models.position_models import SeasonAwareTimeSeriesSplit

# Arms whose hyperparameters can be overridden via `tuned_params` in
# run_walk_forward_backtest -- the two baselines are never tuned.
TUNABLE_ARM_CLASSES: Dict[str, Callable[..., object]] = {
    "logistic": GameOutcomeLogisticModel,
    "xgboost": GameOutcomeXGBModel,
    "random_forest": GameOutcomeRFModel,
}

ARM_FACTORIES: Dict[str, Callable[[], object]] = {
    **TUNABLE_ARM_CLASSES,
    "vegas_favorite": VegasFavoriteBaseline,
    "home_field": HomeFieldBaseline,
}


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
    # log_loss/roc_auc are undefined with a single class present in y_true --
    # a real possibility on a small fold; report NaN rather than crash.
    if len(np.unique(y_true)) > 1:
        metrics["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["log_loss"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def _calibration_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 5) -> List[Dict[str, float]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(proba, bins[1:-1])
    table = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        table.append(
            {
                "bin_range": (float(bins[b]), float(bins[b + 1])),
                "n": int(mask.sum()),
                "mean_predicted": float(proba[mask].mean()),
                "actual_win_rate": float(y_true[mask].mean()),
            }
        )
    return table


def run_walk_forward_backtest(
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    tuned_params: Optional[Dict[str, dict]] = None,
) -> Dict:
    """Train each arm fresh per fold on strictly-prior seasons, score on the held-out season(s).

    `tuned_params`, if given, is a dict like {"xgboost": {"max_depth": 4}} --
    only affects TUNABLE_ARM_CLASSES; the two baselines are never tuned. The
    caller (scripts/train_game_outcome_model.py's --tune flag) is responsible
    for having produced these params from data strictly before every season
    scored here.

    Returns per-fold metrics for every arm plus pooled (all-folds-concatenated)
    metrics and a calibration table per arm.
    """
    tuned_params = tuned_params or {}
    arm_factories: Dict[str, Callable[[], object]] = {
        **{name: (lambda cls=cls, name=name: cls(**tuned_params.get(name, {}))) for name, cls in TUNABLE_ARM_CLASSES.items()},
        "vegas_favorite": VegasFavoriteBaseline,
        "home_field": HomeFieldBaseline,
    }

    df = build_game_outcome_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    X = df[feat_cols]
    y = df["home_win"].to_numpy()
    season_arr = df["season"].to_numpy()

    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    gap = GAME_OUTCOME_MODEL_CONFIG["cv_gap_seasons"]
    splitter = SeasonAwareTimeSeriesSplit(n_splits=n_splits, seasons=season_arr, gap_seasons=gap)

    fold_reports: List[Dict] = []
    pooled_preds: Dict[str, List[np.ndarray]] = {name: [] for name in arm_factories}
    pooled_true: List[np.ndarray] = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X)):
        train_seasons = sorted(set(season_arr[train_idx].tolist()))
        test_seasons = sorted(set(season_arr[test_idx].tolist()))
        if train_seasons and test_seasons and max(train_seasons) >= min(test_seasons):
            raise AssertionError(
                f"Walk-forward violated in fold {fold_i}: train seasons "
                f"{train_seasons} must all be strictly before test seasons {test_seasons}"
            )

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        fold_report = {
            "fold": fold_i,
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "arms": {},
        }
        for name, factory in arm_factories.items():
            model = factory().fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            fold_report["arms"][name] = _classification_metrics(y_test, proba)
            pooled_preds[name].append(proba)
        pooled_true.append(y_test)
        fold_reports.append(fold_report)

    y_all = np.concatenate(pooled_true) if pooled_true else np.array([])
    pooled = {}
    calibration = {}
    for name in arm_factories:
        proba_all = np.concatenate(pooled_preds[name]) if pooled_preds[name] else np.array([])
        if len(proba_all) == 0:
            continue
        pooled[name] = _classification_metrics(y_all, proba_all)
        calibration[name] = _calibration_table(y_all, proba_all)

    return {
        "n_folds": len(fold_reports),
        "n_games_total": int(len(df)),
        "folds": fold_reports,
        "pooled": pooled,
        "calibration": calibration,
    }

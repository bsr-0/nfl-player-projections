"""Walk-forward evaluation for the phase-2 margin/total regression models.

Sibling to src/evaluation/game_outcome_backtester.py (the phase-1 classifier
harness) rather than a shared module -- regression metrics (MAE/RMSE/ATS-or-
O/U pick accuracy against the closing line) are different enough from
classification metrics (accuracy/log-loss/ROC-AUC/Brier) that combining them
would just add branching, not shared logic. Reuses the same
`SeasonAwareTimeSeriesSplit` walk-forward harness and season range.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import GAME_OUTCOME_MODEL_CONFIG
from src.models.game_outcome.baseline import MarketLineBaseline
from src.models.game_outcome.features import build_margin_total_rows, feature_columns
from src.models.game_outcome.models import (
    GameMarginRidgeModel,
    GameRegressionRFModel,
    GameRegressionXGBModel,
)
from src.models.position_models import SeasonAwareTimeSeriesSplit

# label_col: the regression target. line_col: the market's own forecast for
# that target, used both as a baseline prediction and as the number a
# "pick" (ATS / over-under) is made against. pick_name: label for the
# derived accuracy metric.
TARGET_CONFIG: Dict[str, Dict[str, str]] = {
    "margin": {"label_col": "home_margin", "line_col": "spread_line", "pick_name": "ats_accuracy"},
    "total": {"label_col": "game_total", "line_col": "total_line", "pick_name": "ou_accuracy"},
}

# Arms whose hyperparameters can be overridden via `tuned_params` in
# run_walk_forward_backtest -- market_line is never tuned (it has none).
TUNABLE_ARM_CLASSES: Dict[str, Callable[..., object]] = {
    "ridge": GameMarginRidgeModel,
    "xgboost": GameRegressionXGBModel,
    "random_forest": GameRegressionRFModel,
}


def _arm_factories(line_col: str, tuned_params: Dict[str, dict]) -> Dict[str, Callable[[], object]]:
    return {
        **{
            name: (lambda cls=cls, name=name: cls(**tuned_params.get(name, {})))
            for name, cls in TUNABLE_ARM_CLASSES.items()
        },
        "market_line": lambda: MarketLineBaseline(line_col),
    }


def _regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, line: np.ndarray, pick_name: str, n_features: int
) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    n = len(y_true)
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 else float("nan")
    # Standard adjusted-R^2 penalty for predictor count: 1 - (1-R^2)*(n-1)/(n-p-1).
    # `n_features` is 0 for market_line (it isn't fit on X at all -- it just
    # echoes the line), so its adjusted R^2 equals its plain R^2. Undefined
    # (NaN, not 0 or a divide-by-zero) whenever n - p - 1 <= 0 -- a fold with
    # more predictors than (test rows - 1) can't support the penalty term.
    denom = n - n_features - 1
    adj_r2 = float(1 - (1 - r2) * (n - 1) / denom) if (denom > 0 and r2 == r2) else float("nan")

    # A "pick" is which side of the line a prediction lands on. The
    # MarketLineBaseline predicts the line exactly (pred - line == 0 always),
    # so it makes no pick by construction -- excluded from its own accuracy
    # denominator rather than special-cased, same treatment a real model's
    # occasional exact-tie prediction gets.
    pred_sign = np.sign(y_pred - line)
    actual_sign = np.sign(y_true - line)
    made_a_pick = pred_sign != 0
    push = actual_sign == 0  # actual result landed exactly on the line
    denom_mask = made_a_pick & ~push

    accuracy = float((pred_sign[denom_mask] == actual_sign[denom_mask]).mean()) if denom_mask.any() else float("nan")

    return {
        "n": int(n),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        pick_name: accuracy,
        "n_picks": int(denom_mask.sum()),
    }


def run_walk_forward_backtest(
    target: str,
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    tuned_params: Optional[Dict[str, dict]] = None,
) -> Dict:
    """Train each arm fresh per fold on strictly-prior seasons, score on the held-out season(s).

    `target` is "margin" or "total" (see TARGET_CONFIG). `tuned_params`, if
    given, is a dict like {"xgboost": {"max_depth": 4}} -- only affects
    TUNABLE_ARM_CLASSES; market_line is never tuned. The caller
    (scripts/train_game_margin_model.py's --tune flag) is responsible for
    having produced these params from data strictly before every season
    scored here. Returns per-fold metrics for every arm plus pooled
    (all-folds-concatenated) metrics.
    """
    if target not in TARGET_CONFIG:
        raise ValueError(f"target must be one of {list(TARGET_CONFIG)}, got {target!r}")
    cfg = TARGET_CONFIG[target]
    tuned_params = tuned_params or {}

    df = build_margin_total_rows(seasons=seasons)
    feat_cols = feature_columns(df)
    X = df[feat_cols]
    y = df[cfg["label_col"]].to_numpy()
    line = df[cfg["line_col"]].to_numpy()
    season_arr = df["season"].to_numpy()

    n_splits = n_test_seasons or GAME_OUTCOME_MODEL_CONFIG["n_walk_forward_test_seasons"]
    gap = GAME_OUTCOME_MODEL_CONFIG["cv_gap_seasons"]
    splitter = SeasonAwareTimeSeriesSplit(n_splits=n_splits, seasons=season_arr, gap_seasons=gap)
    arms = _arm_factories(cfg["line_col"], tuned_params)

    fold_reports: List[Dict] = []
    pooled_preds: Dict[str, List[np.ndarray]] = {name: [] for name in arms}
    pooled_true: List[np.ndarray] = []
    pooled_line: List[np.ndarray] = []

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
        line_test = line[test_idx]

        fold_report = {
            "fold": fold_i,
            "train_seasons": train_seasons,
            "test_seasons": test_seasons,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "arms": {},
        }
        for name, factory in arms.items():
            model = factory().fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=float)
            n_features = 0 if name == "market_line" else len(feat_cols)
            fold_report["arms"][name] = _regression_metrics(y_test, pred, line_test, cfg["pick_name"], n_features)
            pooled_preds[name].append(pred)
        pooled_true.append(y_test)
        pooled_line.append(line_test)
        fold_reports.append(fold_report)

    y_all = np.concatenate(pooled_true) if pooled_true else np.array([])
    line_all = np.concatenate(pooled_line) if pooled_line else np.array([])
    pooled = {}
    for name in arms:
        preds_all = np.concatenate(pooled_preds[name]) if pooled_preds[name] else np.array([])
        if len(preds_all) == 0:
            continue
        n_features = 0 if name == "market_line" else len(feat_cols)
        pooled[name] = _regression_metrics(y_all, preds_all, line_all, cfg["pick_name"], n_features)

    return {
        "target": target,
        "n_folds": len(fold_reports),
        "n_games_total": int(len(df)),
        "folds": fold_reports,
        "pooled": pooled,
    }

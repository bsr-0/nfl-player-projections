"""Walk-forward evaluation for Plan A's team-allocation share models.

Sibling to src/evaluation/game_margin_backtester.py -- same walk-forward
harness (SeasonAwareTimeSeriesSplit, strict=True since these numbers are
Plan A's acceptance-criteria evidence, not inner-CV tuning; see
src/models/position_models.py), same train-fresh-per-fold discipline,
applied to a [0,1]-bounded share target instead of margin/total.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.baseline import RollingShareBaseline
from src.models.team_allocation.features import (
    ROLL_WINDOW,
    VOLUME_COLS,
    feature_columns,
    filter_population,
    load_share_rows,
)
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel

TUNABLE_ARM_CLASSES: Dict[str, Callable[..., object]] = {
    "ridge": ShareRidgeModel,
    "xgboost": ShareXGBModel,
}


def _arm_factories(roll3_col: str, tuned_params: Dict[str, dict]) -> Dict[str, Callable[[], object]]:
    return {
        **{
            name: (lambda cls=cls, name=name: cls(**tuned_params.get(name, {})))
            for name, cls in TUNABLE_ARM_CLASSES.items()
        },
        "rolling3": lambda: RollingShareBaseline(roll3_col),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    n = len(y_true)
    r2 = float(r2_score(y_true, y_pred)) if n >= 2 else float("nan")
    # n_features=0 for rolling3 (it isn't fit on X at all), matching the
    # adjusted-R^2 convention in game_margin_backtester.py.
    denom = n - n_features - 1
    adj_r2 = float(1 - (1 - r2) * (n - 1) / denom) if (denom > 0 and r2 == r2) else float("nan")
    return {"n": int(n), "mae": mae, "rmse": rmse, "r2": r2, "adj_r2": adj_r2}


def run_walk_forward_backtest(
    target: str,
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    tuned_params: Optional[Dict[str, dict]] = None,
) -> Dict:
    """Train each arm fresh per fold on strictly-prior seasons, score on the held-out season(s).

    `target` is one of VOLUME_COLS (e.g. "targets" -> label column
    `share_of_team_targets`). `tuned_params`, if given, only affects
    TUNABLE_ARM_CLASSES; `rolling3` is never tuned (it has none).
    """
    if target not in VOLUME_COLS:
        raise ValueError(f"target must be one of {VOLUME_COLS}, got {target!r}")
    tuned_params = tuned_params or {}
    label_col = f"share_of_team_{target}"
    roll3_col = f"{label_col}_roll{ROLL_WINDOW}"

    df = load_share_rows(seasons=seasons)
    df = filter_population(df, target)
    feat_cols = feature_columns(df)
    X = df[feat_cols]
    y = df[label_col].to_numpy()
    season_arr = df["season"].to_numpy()

    n_splits = n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"]
    gap = TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"]
    splitter = SeasonAwareTimeSeriesSplit(n_splits=n_splits, seasons=season_arr, gap_seasons=gap, strict=True)
    arms = _arm_factories(roll3_col, tuned_params)

    fold_reports: List[Dict] = []
    pooled_preds: Dict[str, List[np.ndarray]] = {name: [] for name in arms}
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
        for name, factory in arms.items():
            model = factory().fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=float)
            n_features = 0 if name == "rolling3" else len(feat_cols)
            fold_report["arms"][name] = _regression_metrics(y_test, pred, n_features)
            pooled_preds[name].append(pred)
        pooled_true.append(y_test)
        fold_reports.append(fold_report)

    y_all = np.concatenate(pooled_true) if pooled_true else np.array([])
    pooled = {}
    for name in arms:
        preds_all = np.concatenate(pooled_preds[name]) if pooled_preds[name] else np.array([])
        if len(preds_all) == 0:
            continue
        n_features = 0 if name == "rolling3" else len(feat_cols)
        pooled[name] = _regression_metrics(y_all, preds_all, n_features)

    return {
        "target": target,
        "label_col": label_col,
        "n_folds": len(fold_reports),
        "n_rows_total": int(len(df)),
        "folds": fold_reports,
        "pooled": pooled,
    }

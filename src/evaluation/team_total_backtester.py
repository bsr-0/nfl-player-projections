"""Leakage-safe walk-forward forecasts of team-level PPR component totals."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
from src.evaluation.team_reconstruction_candidates import _team_feature_columns
from src.evaluation.team_share_backtester import bootstrap_mae_delta
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.features import ALL_VOLUME_COLS, ROLL_WINDOW, load_share_rows
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel


def _model(kind: str):
    if kind == "ridge":
        return ShareRidgeModel()
    if kind == "xgb":
        return ShareXGBModel()
    raise ValueError(kind)


def _metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mean_actual": float(np.mean(actual)),
        "mean_predicted": float(np.mean(pred)),
    }


def _team_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, List[str]]:
    key = ["team", "season", "week"]
    total = f"team_{target}"
    lag = f"team_{target}_roll{ROLL_WINDOW}"
    features = _team_feature_columns(df, target)
    cols = list(dict.fromkeys(key + features + [total, lag]))
    team = df[cols].sort_values(key).drop_duplicates(key).reset_index(drop=True)
    return team, features


def _choose_blend(actual: np.ndarray, ridge: np.ndarray, xgb: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(ridge) & np.isfinite(xgb)
    if valid.sum() < 20:
        return 0.5
    return float(min(
        np.arange(0.0, 1.001, 0.05),
        key=lambda a: mean_absolute_error(actual[valid], a * ridge[valid] + (1 - a) * xgb[valid]),
    ))


def run_team_total_backtest(
    target: str,
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
) -> Dict:
    if target not in ALL_VOLUME_COLS:
        raise ValueError(f"target must be one of {ALL_VOLUME_COLS}, got {target!r}")
    df = load_share_rows(seasons=seasons)
    team, feature_cols = _team_frame(df, target)
    total_col = f"team_{target}"
    lag_col = f"team_{target}_roll{ROLL_WINDOW}"
    seasons_arr = team.season.to_numpy()
    X = team[feature_cols]
    splitter = SeasonAwareTimeSeriesSplit(
        n_splits=n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"],
        seasons=seasons_arr,
        gap_seasons=TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"],
        strict=True,
    )
    arms = ("rolling3", "ridge", "xgb", "blend")
    preds = {arm: [] for arm in arms}
    actuals = []
    folds = []
    prediction_rows = []
    for fold, (tr, te) in enumerate(splitter.split(X)):
        train, test = team.iloc[tr], team.iloc[te]
        train_seasons = sorted(set(train.season.tolist()))
        test_seasons = sorted(set(test.season.tolist()))
        if max(train_seasons) >= min(test_seasons):
            raise AssertionError("team-total walk-forward fold includes future training data")
        y_train = train[total_col].to_numpy(float)
        y_test = test[total_col].to_numpy(float)
        base = test[lag_col].to_numpy(float)
        fitted = {}
        for kind in ("ridge", "xgb"):
            model = _model(kind).fit(train[feature_cols], y_train)
            fitted[kind] = np.clip(np.asarray(model.predict(test[feature_cols]), float), 0, None)
        last = max(train_seasons)
        cal_train = train[train.season < last]
        cal = train[train.season == last]
        if len(cal_train) >= 20 and len(cal) >= 10:
            cal_models = {kind: _model(kind).fit(cal_train[feature_cols], cal_train[total_col].to_numpy(float)) for kind in ("ridge", "xgb")}
            alpha = _choose_blend(
                cal[total_col].to_numpy(float),
                np.clip(cal_models["ridge"].predict(cal[feature_cols]), 0, None),
                np.clip(cal_models["xgb"].predict(cal[feature_cols]), 0, None),
            )
        else:
            alpha = 0.5
        preds["rolling3"].append(base)
        preds["ridge"].append(fitted["ridge"])
        preds["xgb"].append(fitted["xgb"])
        preds["blend"].append(alpha * fitted["ridge"] + (1 - alpha) * fitted["xgb"])
        actuals.append(y_test)
        folds.append({"fold": fold, "train_seasons": train_seasons, "test_seasons": test_seasons, "blend_alpha": alpha, "n_test": int(len(te))})
        fold_predictions = test[["team", "season", "week"]].copy()
        fold_predictions["fold"] = fold
        fold_predictions["actual"] = y_test
        fold_predictions["rolling3"] = base
        fold_predictions["ridge"] = fitted["ridge"]
        fold_predictions["xgb"] = fitted["xgb"]
        fold_predictions["blend"] = alpha * fitted["ridge"] + (1 - alpha) * fitted["xgb"]
        prediction_rows.append(fold_predictions)

    actual = np.concatenate(actuals) if actuals else np.array([])
    pooled = {}
    base_pred = np.concatenate(preds["rolling3"])
    common_valid = np.isfinite(actual) & np.isfinite(base_pred)
    for arm in arms:
        pred = np.concatenate(preds[arm]) if preds[arm] else np.array([])
        # Acceptance comparisons use the same rows as rolling-3. A learned
        # model may produce a cold-start estimate, but that is a separate
        # coverage experiment and must not change the primary population.
        valid = common_valid & np.isfinite(pred)
        pooled[arm] = _metrics(actual[valid], pred[valid]) if valid.any() else {"n": 0, "mae": float("nan"), "rmse": float("nan")}
        pooled[arm]["n_rows_excluded_cold_start"] = int((~valid).sum())
    for arm in ("ridge", "xgb", "blend"):
        candidate = np.concatenate(preds[arm])
        valid = np.isfinite(actual) & np.isfinite(base_pred) & np.isfinite(candidate)
        pooled[arm]["vs_rolling3_bootstrap"] = bootstrap_mae_delta(actual[valid], candidate[valid], base_pred[valid])
    return {
        "target": target,
        "feature_columns": feature_cols,
        "n_team_weeks_total": int(len(team)),
        "folds": folds,
        "pooled": pooled,
        "predictions": pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(),
    }

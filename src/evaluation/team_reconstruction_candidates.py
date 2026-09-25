"""Reconstruction-aware Plan A candidate evaluation.

The original share backtester optimizes unweighted share MAE.  That is not
the same objective as the yardage reconstruction: a one-point share error on
a team expected to produce 450 yards matters more than the same error on a
team expected to produce 250 yards.  This module keeps the same strict
walk-forward folds, but adds two leakage-safe candidates:

* volume-weighted regressors, using only the lagged team-total baseline as
  training weights; and
* fold-local blends whose alpha is selected on absolute reconstructed-volume
  error in the latest training season, rather than share error.

The returned OOF rows are consumed by the existing reconstruction code, so
all arms are compared on identical player-week rows and the same lagged team
total.  These are research candidates only; no production artifacts are
written here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.baseline import RollingShareBaseline
from src.models.team_allocation.features import (
    ROLL_WINDOW,
    ALL_VOLUME_COLS,
    VOLUME_COLS,
    feature_columns,
    filter_population,
    load_share_rows,
)
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel


MODEL_KINDS = {"ridge": ShareRidgeModel, "xgb": ShareXGBModel}
RECON_ARMS = (
    "rolling3",
    "ridge",
    "xgb",
    "ridge_volume_weighted",
    "xgb_volume_weighted",
    "ridge_reconstruction_blend",
    "xgb_reconstruction_blend",
    "rolling3_teamtotal_ridge",
    "rolling3_teamtotal_xgb",
    "ridge_teamtotal_ridge",
    "ridge_teamtotal_xgb",
    "xgb_teamtotal_ridge",
    "xgb_teamtotal_xgb",
    "ridge_volume_weighted_teamtotal_ridge",
    "ridge_volume_weighted_teamtotal_xgb",
    "xgb_volume_weighted_teamtotal_ridge",
    "xgb_volume_weighted_teamtotal_xgb",
    "ridge_reconstruction_blend_teamtotal_ridge",
    "ridge_reconstruction_blend_teamtotal_xgb",
    "xgb_reconstruction_blend_teamtotal_ridge",
    "xgb_reconstruction_blend_teamtotal_xgb",
    "rolling3_teamtotal_blend",
    "ridge_teamtotal_blend",
    "xgb_teamtotal_blend",
    "ridge_volume_weighted_teamtotal_blend",
    "xgb_volume_weighted_teamtotal_blend",
    "ridge_reconstruction_blend_teamtotal_blend",
    "xgb_reconstruction_blend_teamtotal_blend",
)


def _safe_volume_weights(team_total: np.ndarray, power: float) -> np.ndarray:
    """Finite, bounded weights derived only from lagged team totals."""
    total = np.asarray(team_total, dtype=float)
    finite = np.isfinite(total) & (total > 0)
    if not finite.any():
        return np.ones(len(total), dtype=float)
    reference = float(np.nanmedian(total[finite]))
    weights = np.ones(len(total), dtype=float)
    weights[finite] = np.power(np.clip(total[finite] / reference, 0.25, 4.0), power)
    # Missing/cold-start team totals are not used to invent a signal and get
    # neutral weight; those rows are excluded from the reconstruction score.
    return weights


def _fit(kind: str, X: pd.DataFrame, y: np.ndarray, sample_weight=None):
    model = MODEL_KINDS[kind]()
    return model.fit(X, y, sample_weight=sample_weight)


def _best_volume_alpha(y_volume: np.ndarray, model_share: np.ndarray,
                       baseline_share: np.ndarray, team_total: np.ndarray) -> float:
    valid = np.isfinite(y_volume) & np.isfinite(model_share) & np.isfinite(baseline_share) & np.isfinite(team_total)
    valid &= team_total > 0
    if valid.sum() < 20:
        return 0.0
    yv = y_volume[valid]
    mv = model_share[valid] * team_total[valid]
    bv = baseline_share[valid] * team_total[valid]
    return float(min(
        np.arange(0.0, 1.001, 0.05),
        key=lambda alpha: mean_absolute_error(yv, alpha * mv + (1.0 - alpha) * bv),
    ))


def _team_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Only lagged team-context columns for the team-total forecaster."""
    current = {f"team_{name}" for name in ALL_VOLUME_COLS}
    cols = [
        c for c in df.columns
        if c.startswith("team_") and c not in current and c != f"team_{target}"
    ]
    if not cols:
        raise ValueError("team-week table has no lagged team features")
    return cols


def _team_total_forecasts(
    df: pd.DataFrame,
    target: str,
    train_seasons: List[int],
    test_seasons: List[int],
) -> Dict[str, np.ndarray]:
    """Fit leakage-safe team-total models and map predictions to player rows."""
    key = ["season", "week", "team"]
    total_col = f"team_{target}"
    lag_col = f"team_{target}_roll{ROLL_WINDOW}"
    team_cols = _team_feature_columns(df, target)
    select_cols = list(dict.fromkeys(key + team_cols + [total_col, lag_col]))
    team = (
        df[select_cols]
        .sort_values(key)
        .drop_duplicates(key)
        .reset_index(drop=True)
    )
    train = team[team.season.isin(train_seasons)]
    test = team[team.season.isin(test_seasons)]
    out = {"ridge": {}, "xgb": {}}
    for kind in ("ridge", "xgb"):
        model = _fit(kind, train[team_cols], train[total_col].to_numpy(dtype=float))
        pred = np.clip(np.asarray(model.predict(test[team_cols]), dtype=float), 0.0, None)
        pred_map = test[key].copy()
        pred_map["_pred"] = pred
        out[kind] = pred_map
    mapped = {}
    player_keys = df[key]
    for kind, pred_map in out.items():
        mapped[kind] = player_keys.merge(pred_map, on=key, how="left")["_pred"].to_numpy(dtype=float)
    # Team-total model is allowed to be used only where the baseline had a
    # known lagged total, so all reconstruction arms retain identical rows.
    valid_baseline = player_keys.merge(team[key + [lag_col]], on=key, how="left")[lag_col].to_numpy(dtype=float)
    for kind in mapped:
        mapped[kind] = np.where(np.isfinite(valid_baseline), mapped[kind], np.nan)
    # Select the Ridge/XGBoost mixture using only the latest training season.
    # This makes the team-total choice a fold-local rule rather than a
    # post-hoc choice based on the held-out season.
    last_train = max(train_seasons)
    cal_train = team[team.season < last_train]
    cal_test = team[team.season == last_train]
    if len(cal_train) >= 40 and len(cal_test) >= 20:
        cal_preds = {}
        for kind in ("ridge", "xgb"):
            cal_model = _fit(kind, cal_train[team_cols], cal_train[total_col].to_numpy(dtype=float))
            cal_preds[kind] = np.clip(np.asarray(cal_model.predict(cal_test[team_cols]), dtype=float), 0.0, None)
        actual = cal_test[total_col].to_numpy(dtype=float)
        alpha = min(
            np.arange(0.0, 1.001, 0.05),
            key=lambda a: mean_absolute_error(actual, a * cal_preds["ridge"] + (1.0 - a) * cal_preds["xgb"]),
        )
    else:
        alpha = 0.5
    mapped["blend"] = alpha * mapped["ridge"] + (1.0 - alpha) * mapped["xgb"]
    return mapped


def walk_forward_reconstruction_oof(
    target: str,
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    weight_power: float = 1.0,
) -> pd.DataFrame:
    """Return strict walk-forward OOF predictions for reconstruction arms."""
    if target not in VOLUME_COLS:
        raise ValueError(f"target must be one of {VOLUME_COLS}, got {target!r}")
    label = f"share_of_team_{target}"
    roll = f"{label}_roll{ROLL_WINDOW}"
    team_total_col = f"team_{target}_roll{ROLL_WINDOW}"
    df = filter_population(load_share_rows(seasons=seasons), target).reset_index(drop=True)
    cols = feature_columns(df)
    X, y = df[cols], df[label].to_numpy(dtype=float)
    actual_volume = df[target].to_numpy(dtype=float)
    team_total = df[team_total_col].to_numpy(dtype=float)
    seasons_arr = df["season"].to_numpy()
    splitter = SeasonAwareTimeSeriesSplit(
        n_splits=n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"],
        seasons=seasons_arr,
        gap_seasons=TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"],
        strict=True,
    )
    id_cols = ["player_id", "season", "week", "team", "position"]
    rows: List[pd.DataFrame] = []
    for fold, (tr, te) in enumerate(splitter.split(X)):
        train, test = X.iloc[tr], X.iloc[te]
        yt, yv = y[tr], y[te]
        base = RollingShareBaseline(roll).fit(train, yt).predict(test)
        train_seasons = sorted(set(seasons_arr[tr].tolist()))
        test_seasons = sorted(set(seasons_arr[te].tolist()))
        team_totals = _team_total_forecasts(df, target, train_seasons, test_seasons)
        # A calibration model is trained strictly before the latest training
        # season; alpha therefore never sees the held-out season.
        last_train = max(seasons_arr[tr])
        cal = seasons_arr[tr] == last_train
        share_predictions: Dict[str, np.ndarray] = {}
        for kind in ("ridge", "xgb"):
            raw_model = _fit(kind, train, yt)
            raw = np.clip(np.asarray(raw_model.predict(test), dtype=float), 0.0, 1.0)
            weighted = _fit(
                kind,
                train,
                yt,
                sample_weight=_safe_volume_weights(team_total[tr], weight_power),
            )
            weighted_pred = np.clip(np.asarray(weighted.predict(test), dtype=float), 0.0, 1.0)
            if cal.any():
                cal_model = _fit(kind, train.loc[~cal], yt[~cal])
                cal_pred = np.clip(np.asarray(cal_model.predict(train.loc[cal]), dtype=float), 0.0, 1.0)
                alpha = _best_volume_alpha(
                    yt[cal] * team_total[tr][cal], cal_pred,
                    train[roll].fillna(0).to_numpy(dtype=float)[cal],
                    team_total[tr][cal],
                )
            else:
                alpha = 0.0
            blended = np.clip(alpha * raw + (1.0 - alpha) * base, 0.0, 1.0)
            for arm, pred in (
                (kind, raw),
                (f"{kind}_volume_weighted", weighted_pred),
                (f"{kind}_reconstruction_blend", blended),
            ):
                share_predictions[arm] = pred
                part = df.iloc[te][id_cols].copy()
                part["fold"] = fold
                part["arm"] = arm
                part["predicted_share"] = pred
                part["actual_share"] = yv
                part["actual_volume"] = actual_volume[te]
                part["team_total_roll3"] = team_total[te]
                part["blend_alpha"] = alpha
                rows.append(part)
        share_predictions["rolling3"] = base
        part = df.iloc[te][id_cols].copy()
        part["fold"] = fold
        part["arm"] = "rolling3"
        part["predicted_share"] = base
        part["actual_share"] = yv
        part["actual_volume"] = actual_volume[te]
        part["team_total_roll3"] = team_total[te]
        part["blend_alpha"] = 0.0
        rows.append(part)
        # Evaluate allocation and total-volume improvements separately and in
        # combination.  The same share prediction is multiplied by a
        # walk-forward team-total forecast; no current-week team total is
        # used as a feature or scoring input.
        for share_arm, share_pred in share_predictions.items():
            for total_kind, predicted_total in team_totals.items():
                arm = f"{share_arm}_teamtotal_{total_kind}"
                part = df.iloc[te][id_cols].copy()
                part["fold"] = fold
                part["arm"] = arm
                part["predicted_share"] = share_pred
                part["actual_share"] = yv
                part["actual_volume"] = actual_volume[te]
                part["team_total_roll3"] = predicted_total[te]
                part["blend_alpha"] = 0.0
                rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_reconstruction_candidate_backtest(
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    weight_power: float = 1.0,
) -> Dict:
    """Evaluate candidate arm combinations on reconstructed yardage points."""
    from src.evaluation.team_reconstruction_backtester import _reconstruct_for_arm, _scored_rows
    from src.evaluation.team_share_backtester import _regression_metrics, _segment_metrics, bootstrap_mae_delta

    oof_by_target = {
        target: walk_forward_reconstruction_oof(target, seasons, n_test_seasons, weight_power)
        for target in ("rushing_yards", "receiving_yards")
    }
    arms = [arm for arm in RECON_ARMS if arm in set(oof_by_target["rushing_yards"].arm) and arm in set(oof_by_target["receiving_yards"].arm)]
    per_arm = {arm: _reconstruct_for_arm(oof_by_target, arm) for arm in arms}
    scored = {arm: _scored_rows(frame) for arm, frame in per_arm.items()}
    pooled: Dict[str, Dict] = {}
    for arm, frame in scored.items():
        y_true = frame["actual_partial_points"].to_numpy()
        y_pred = frame["predicted_partial_points"].to_numpy()
        pooled[arm] = {
            **_regression_metrics(y_true, y_pred, 0 if arm == "rolling3" else 1),
            "by_position": _segment_metrics(y_true, y_pred, frame["position"].to_numpy(), 0 if arm == "rolling3" else 1),
            "by_season": {
                str(season): _regression_metrics(
                    group["actual_partial_points"].to_numpy(),
                    group["predicted_partial_points"].to_numpy(),
                    0 if arm == "rolling3" else 1,
                )
                for season, group in frame.groupby("season", sort=True)
            },
            "n_rows_excluded_cold_start": int(len(per_arm[arm]) - len(frame)),
        }
    baseline = scored.get("rolling3")
    if baseline is not None:
        for arm, frame in scored.items():
            if arm == "rolling3" or len(frame) != len(baseline):
                continue
            pooled[arm]["vs_rolling3_bootstrap"] = bootstrap_mae_delta(
                frame["actual_partial_points"].to_numpy(),
                frame["predicted_partial_points"].to_numpy(),
                baseline["predicted_partial_points"].to_numpy(),
            )
    comparable = {arm: metrics["mae"] for arm, metrics in pooled.items() if arm != "rolling3"}
    winner = min(comparable, key=comparable.get) if comparable else None
    return {
        "targets_reconstructed": ["rushing_yards", "receiving_yards"],
        "weight_power": weight_power,
        "arms": arms,
        "n_rows_total": int(len(next(iter(per_arm.values())))) if per_arm else 0,
        "best_candidate_by_pooled_mae": winner,
        "pooled": pooled,
    }

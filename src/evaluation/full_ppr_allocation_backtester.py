"""Walk-forward player allocation models for full-PPR component shares."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
from src.evaluation.team_share_backtester import bootstrap_mae_delta
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.baseline import RollingShareBaseline
from src.models.team_allocation.features import ALL_VOLUME_COLS, ROLL_WINDOW, feature_columns, filter_population, load_share_rows
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel
from src.models.team_allocation.reconstruct import renormalize_shares
from src.models.team_allocation.hierarchical import (
    SPARSE_EVENT_TARGETS,
    add_role_tier,
    hierarchical_feature_columns,
    predict_conditional_hierarchical,
    predict_residual_over_opportunity,
    predict_poisson_count_allocation,
    predict_multinomial_count_allocation,
)



def _fit(kind: str, X: pd.DataFrame, y: np.ndarray):
    model = ShareRidgeModel() if kind == "ridge" else ShareXGBModel()
    return model.fit(X, y)


def _two_stage(train_X, test_X, y_train, positions_train, positions_test, kind="xgb"):
    pred = np.zeros(len(test_X), dtype=float)
    for position in np.unique(positions_test):
        tr = positions_train == position
        te = positions_test == position
        if tr.sum() < 40 or te.sum() == 0:
            continue
        active = y_train[tr] > 0
        if active.sum() < 20 or (~active).sum() < 10:
            model = _fit(kind, train_X.loc[tr], y_train[tr])
            pred[te] = np.clip(model.predict(test_X.loc[te]), 0, 1)
            continue
        classifier = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        classifier.fit(train_X.loc[tr], active.astype(int))
        probability = classifier.predict_proba(test_X.loc[te])[:, 1]
        positive = _fit(kind, train_X.loc[tr].loc[active], y_train[tr][active])
        pred[te] = probability * np.clip(positive.predict(test_X.loc[te]), 0, 1)
    return pred


def run_allocation_backtest(
    target: str,
    seasons: Optional[List[int]] = None,
    n_test_seasons: Optional[int] = None,
    *,
    sparse_feature_families: Iterable[str] | None = None,
) -> Dict:
    if target not in ALL_VOLUME_COLS:
        raise ValueError(f"target must be one of {ALL_VOLUME_COLS}, got {target!r}")
    label = f"share_of_team_{target}"
    roll = f"{label}_roll{ROLL_WINDOW}"
    df = filter_population(load_share_rows(seasons=seasons), target).reset_index(drop=True)
    cols = feature_columns(df, include_full_ppr=True)
    X, y = df[cols], df[label].to_numpy(float)
    season_arr = df.season.to_numpy()
    positions = df.position.to_numpy()
    splitter = SeasonAwareTimeSeriesSplit(
        n_splits=n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"],
        seasons=season_arr,
        gap_seasons=TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"],
        strict=True,
    )
    family_label = "all" if sparse_feature_families is None else "+".join(sorted(set(sparse_feature_families)))
    arm_suffix = "" if sparse_feature_families is None else f"_{family_label}"
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X)):
        train, test = X.iloc[tr], X.iloc[te]
        yt, yv = y[tr], y[te]
        base = RollingShareBaseline(roll).fit(train, yt).predict(test)
        base_renorm = renormalize_shares(base, df.iloc[te][["team", "season", "week"]].reset_index(drop=True))
        for kind in ("ridge", "xgb"):
            model = _fit(kind, train, yt)
            raw = np.clip(np.asarray(model.predict(test), float), 0, 1)
            last = max(season_arr[tr])
            cal = season_arr[tr] == last
            if cal.sum() >= 20 and (~cal).sum() >= 20:
                cal_model = _fit(kind, train.loc[~cal], yt[~cal])
                alpha = min(
                    np.arange(0, 1.01, .05),
                    key=lambda a: mean_absolute_error(yt[cal], a * cal_model.predict(train.loc[cal]) + (1 - a) * train.loc[cal][roll].fillna(0)),
                )
            else:
                alpha = 0.5
            blend = np.clip(alpha * raw + (1 - alpha) * base, 0, 1)
            renorm = renormalize_shares(
                blend,
                df.iloc[te][["team", "season", "week"]].reset_index(drop=True),
                zero_fallback="zeros" if target in SPARSE_EVENT_TARGETS else "equal",
            )
            candidates = {f"{kind}_blend_renorm": renorm}
            if target in {"receiving_tds", "rushing_tds", "passing_tds", "interceptions"} and kind == "xgb":
                staged = _two_stage(train, test, yt, positions[tr], positions[te], kind="xgb")
                candidates["xgb_two_stage_renorm"] = renormalize_shares(
                    staged,
                    df.iloc[te][["team", "season", "week"]].reset_index(drop=True),
                    zero_fallback="zeros",
                )
            for arm, pred in candidates.items():
                part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
                part["fold"], part["arm"] = fold, arm
                part["actual_share"], part["predicted_share"] = yv, pred
                rows.append(part)
        if target in SPARSE_EVENT_TARGETS:
            sparse_cols = hierarchical_feature_columns(
                df.iloc[tr], target, families=sparse_feature_families
            )
            hierarchical = predict_conditional_hierarchical(
                df.iloc[tr].copy(),
                df.iloc[te].copy(),
                target,
                feature_cols=sparse_cols,
            )
            # Sparse shares are *unconditional*: their team sum is the
            # calibrated probability/mass of the event, not always one.
            # Renormalizing a nonzero team-week here would erase the gate
            # calibration (and turn, say, a 0.25 TD probability into 1.0).
            part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
            part["fold"], part["arm"] = fold, f"hierarchical_conditional{arm_suffix}_event_mass"
            part["actual_share"], part["predicted_share"] = yv, hierarchical
            rows.append(part)
            residual = predict_residual_over_opportunity(
                df.iloc[tr].copy(),
                df.iloc[te].copy(),
                target,
                feature_cols=sparse_cols,
            )
            part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
            part["fold"], part["arm"] = fold, f"residual_opportunity{arm_suffix}_event_mass"
            part["actual_share"], part["predicted_share"] = yv, residual
            rows.append(part)
            count_pred = predict_poisson_count_allocation(
                df.iloc[tr].copy(),
                df.iloc[te].copy(),
                target,
                feature_cols=sparse_cols,
            )
            part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
            part["fold"], part["arm"] = fold, f"poisson_count{arm_suffix}_renorm"
            part["actual_share"], part["predicted_share"] = yv, count_pred
            rows.append(part)
            multinomial_pred = predict_multinomial_count_allocation(
                df.iloc[tr].copy(),
                df.iloc[te].copy(),
                target,
                feature_cols=sparse_cols,
            )
            part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
            part["fold"], part["arm"] = fold, f"multinomial_count{arm_suffix}_event_mass"
            part["actual_share"], part["predicted_share"] = yv, multinomial_pred
            rows.append(part)
        part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
        part["fold"], part["arm"] = fold, "rolling3"
        part["actual_share"], part["predicted_share"] = yv, base
        rows.append(part)
        part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
        part["fold"], part["arm"] = fold, "rolling3_renorm"
        part["actual_share"], part["predicted_share"] = yv, base_renorm
        rows.append(part)

    out = pd.concat(rows, ignore_index=True)
    role_frame = add_role_tier(
        df[["player_id", "season", "week", "team", "position", "snap_share_roll3", "share_of_team_targets_roll3"]]
    )[["player_id", "season", "week", "team", "role_tier"]]
    out = out.merge(
        role_frame,
        on=["player_id", "season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    metrics = {}
    baseline = out[out.arm == "rolling3"].sort_values(["fold", "player_id"])
    baseline_renorm = out[out.arm == "rolling3_renorm"].sort_values(["fold", "player_id"])
    for arm, group in out.groupby("arm"):
        team_group = group.groupby(["season", "week", "team"], sort=False)
        predicted_team_sum = team_group.predicted_share.sum()
        actual_team_sum = team_group.actual_share.sum()
        metrics[arm] = {
            "n": int(len(group)),
            "mae": float(mean_absolute_error(group.actual_share, group.predicted_share)),
            "mean_abs_team_sum_error": float(group.groupby(["season", "week", "team"]).predicted_share.sum().sub(1).abs().mean()),
            "mean_predicted_team_sum": float(predicted_team_sum.mean()),
            "mean_actual_team_sum": float(actual_team_sum.mean()),
            "team_sum_mae": float(mean_absolute_error(actual_team_sum, predicted_team_sum)),
        }
        metrics[arm]["segment_mae"] = {
            segment: {
                str(value): float(mean_absolute_error(part.actual_share, part.predicted_share))
                for value, part in group.groupby(segment, dropna=False)
            }
            for segment in ("position", "role_tier", "is_cold_start")
        }
        if arm not in {"rolling3", "rolling3_renorm"}:
            candidate = group.sort_values(["fold", "player_id"])
            metrics[arm]["vs_rolling3_bootstrap"] = bootstrap_mae_delta(
                baseline.actual_share.to_numpy(), candidate.predicted_share.to_numpy(), baseline.predicted_share.to_numpy()
            )
            if arm != "rolling3_renorm":
                metrics[arm]["vs_rolling3_renorm_bootstrap"] = bootstrap_mae_delta(
                    baseline_renorm.actual_share.to_numpy(), candidate.predicted_share.to_numpy(), baseline_renorm.predicted_share.to_numpy()
                )
    return {
        "target": target,
        "feature_columns": cols,
        "sparse_feature_families": family_label,
        "n_rows": int(len(out)),
        "metrics": metrics,
        "predictions": out,
    }

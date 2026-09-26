"""Past-season-only evaluation of Plan B's four volume-share baselines.

This evaluates the represented roster population, not full PPR or a joint
team allocation. No fitting or preprocessing uses held-out labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation.paired_ppr_comparison import paired_week_interval
from src.models.team_allocation.features import VOLUME_COLS, filter_population
from src.models.team_hierarchical.features import feature_columns
from src.models.team_hierarchical.models import MixedEffectsFitError, MixedEffectsShareModel

KEYS = ["player_id", "season", "week", "team", "position"]
ARMS = ["rolling3", "fixed_ridge", "mixed_effects"]


def evaluation_features(target: str) -> list[str]:
    """A predeclared small feature set; no sweep on the evaluation seasons."""
    if target not in VOLUME_COLS:
        raise ValueError(f"unsupported Plan B volume target: {target}")
    return ["slot", "depth_chart_rank", "roster_snap_share_s2d", "is_cold_start",
            f"share_of_team_{target}_roll3", f"share_of_team_{target}_s2d",
            f"team_{target}_roll3", f"team_{target}_s2d"]


def validate_panel(panel: pd.DataFrame, target: str, test_seasons: list[int]) -> dict:
    features = evaluation_features(target)
    label = f"share_of_team_{target}"
    required = set(KEYS + features + [label])
    if panel.columns.duplicated().any() or required - set(panel):
        raise ValueError(f"invalid panel columns; missing: {sorted(required - set(panel))}")
    if panel.empty or panel[KEYS].isna().any().any() or panel.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("panel must have unique, nonnull player-week keys")
    if not test_seasons or test_seasons != sorted(set(test_seasons)):
        raise ValueError("test seasons must be a nonempty increasing unique list")
    for col in ("season", "week"):
        values = pd.to_numeric(panel[col], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values != values.astype(int)).any() or (values < 1).any():
            raise ValueError(f"invalid {col}")
    if len(filter_population(panel, target)) != len(panel):
        raise ValueError("panel contains positions excluded from this target's population")
    MixedEffectsShareModel._check_identity(panel[["player_id", "slot"]])
    safe = feature_columns(panel)  # audit the entire schema before the compact selection
    if set(features) - set(safe):
        raise ValueError("evaluation features did not pass the feature availability audit")
    truth = pd.to_numeric(panel[label], errors="raise").to_numpy(float)
    if not np.isfinite(truth).all() or ((truth < 0) | (truth > 1)).any():
        raise ValueError("actual shares must be finite and in [0, 1]")
    numeric = panel[features[1:]].apply(pd.to_numeric, errors="raise")
    if np.isinf(numeric.to_numpy(float)).any():
        raise ValueError("panel features contain infinity")
    for suffix in ("roll3", "s2d"):
        share = numeric[f"share_of_team_{target}_{suffix}"]
        if ((share.dropna() < 0) | (share.dropna() > 1)).any():
            raise ValueError("lagged shares must be in [0, 1] or missing")
    folds = []
    for season in test_seasons:
        train, test = panel[panel.season < season], panel[panel.season == season]
        if train.empty or test.empty:
            raise ValueError(f"season {season} lacks training or held-out rows")
        folds.append({"test_season": int(season), "train_seasons": sorted(map(int, train.season.unique())),
                      "n_train": len(train), "n_test": len(test)})
    return {"target": target, "metric": "player_week_share_mae", "features": features,
            "n_panel": len(panel), "folds": folds,
            "missing_feature_cells": int(numeric.isna().sum().sum()),
            "rolling3_null_to_zero_rows": int(panel.loc[panel.season.isin(test_seasons), features[4]].isna().sum())}


def _fixed_control(features):
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    preprocess = ColumnTransformer([
        ("slot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["slot"]),
        ("numeric", numeric, features[1:]),
    ])
    return make_pipeline(preprocess, Ridge(alpha=1.0, solver="svd"))


def _metrics(rows):
    return {arm: {"n": len(rows), "mae": float(np.abs(rows[arm] - rows.actual_share).mean())}
            for arm in ARMS}


def run_backtest(panel: pd.DataFrame, target: str, test_seasons: list[int], *,
                 maxiter: int = 200, n_bootstrap: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    report = validate_panel(panel, target, test_seasons)
    if type(n_bootstrap) is not int or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer >= 100")
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    features, label = report["features"], f"share_of_team_{target}"
    outputs = []
    for fold in report["folds"]:
        season = fold["test_season"]
        train, test = panel[panel.season < season], panel[panel.season == season]
        X, test_X = train[["player_id"] + features], test[["player_id"] + features]
        model = MixedEffectsShareModel(maxiter=maxiter, failure_policy="raise")
        try:
            model.fit(X, train[label].to_numpy(float))
            mixed = model.predict(test_X)
        except MixedEffectsFitError as exc:
            raise MixedEffectsFitError(f"Plan B {target}, held-out {season}: {exc}; diagnostics={model.fit_diagnostics_}") from exc
        control = _fixed_control(features).fit(train[features], train[label])
        fixed = control.predict(test[features])
        out = test[KEYS + ["slot"]].copy()
        out["train_end_season"] = int(train.season.max())
        out["unseen_player"] = ~test.player_id.isin(train.player_id)
        out["actual_share"] = test[label].to_numpy(float)
        out["rolling3"] = test[features[4]].fillna(0).to_numpy(float)
        for arm, prediction in (("mixed_effects", mixed), ("fixed_ridge", fixed)):
            if len(prediction) != len(test) or not np.isfinite(prediction).all():
                raise ValueError(f"{arm}: missing or nonfinite predictions in {season}")
            out[arm + "_raw"] = prediction
            out[arm] = np.clip(prediction, 0, 1)
        fold["fit"] = model.fit_diagnostics_
        fold["prediction"] = model.prediction_diagnostics_
        fold["clipped_rows"] = {arm: int((out[arm] != out[arm + "_raw"]).sum()) for arm in ARMS[1:]}
        fold["rolling3_null_to_zero_rows"] = int(test[features[4]].isna().sum())
        outputs.append(out)
    rows = pd.concat(outputs, ignore_index=True)
    expected = panel.loc[panel.season.isin(test_seasons), KEYS].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    report["pooled"] = _metrics(rows)
    for column in ("season", "position", "unseen_player"):
        report[f"by_{column}"] = {str(value): _metrics(group) for value, group in rows.groupby(column, sort=True)}
    report["comparisons"] = {}
    for baseline in ("rolling3", "fixed_ridge"):
        paired = rows[["season", "week"]].copy()
        paired["delta_abs_error"] = np.abs(rows.mixed_effects - rows.actual_share) - np.abs(rows[baseline] - rows.actual_share)
        report["comparisons"][f"mixed_effects_minus_{baseline}"] = paired_week_interval(paired, n_bootstrap, seed)
    sums = rows.groupby(["season", "week", "team"])[ARMS + ["actual_share"]].sum()
    report["represented_team_mass"] = {
        "n_team_weeks": len(sums), "mean_actual_share_sum": float(sums.actual_share.mean()),
        "arms": {arm: {"mean_predicted_share_sum": float(sums[arm].mean()),
                       "teams_above_one": int((sums[arm] > 1 + 1e-9).sum()),
                       "mass_mae": float(np.abs(sums[arm] - sums.actual_share).mean())} for arm in ARMS}}
    report.update(status="complete", ridge_alpha=1.0, maxiter=maxiter, reml=True,
                  prediction_bounds=[0, 1], team_renormalization=False,
                  limitations=["Share accuracy on capped roster rows; not PPR accuracy or full-panel accuracy.",
                               "Player random intercept baseline; no joint team constraint or serving integration.",
                               "Retrospective seasonal evaluation; lagged in-season observed features are available each week.",
                               "Week-block intervals do not capture serial dependence across weeks."])
    return rows, report

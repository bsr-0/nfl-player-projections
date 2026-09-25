"""Artifact-backed Plan A share prediction path.

This module is deliberately small and side-effect free: training code writes a
serialized artifact, while serving loads that artifact and applies its stored
feature schema and team/week renormalization contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel
from src.models.team_allocation.reconstruct import renormalize_shares

SERVING_SCHEMA_VERSION = "plan-a-serving-v1"


class PlanAShareArtifact:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def fit(cls, train: pd.DataFrame, target: str, model_kind: str = "xgb"):
        from src.models.team_allocation.features import feature_columns, filter_population

        train = filter_population(train, target).copy()
        label = f"share_of_team_{target}"
        columns = feature_columns(train)
        model = ShareXGBModel() if model_kind == "xgb" else ShareRidgeModel()
        model.fit(train[columns], train[label].to_numpy(float))
        return cls({
            "schema_version": SERVING_SCHEMA_VERSION,
            "target": target,
            "model_kind": model_kind,
            "feature_columns": columns,
            "model": model,
        })

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        expected = self.payload["feature_columns"]
        missing = sorted(set(expected) - set(rows.columns))
        if missing:
            raise ValueError(f"Plan A serving rows missing features: {missing}")
        raw = np.clip(np.asarray(self.payload["model"].predict(rows[expected]), float), 0, None)
        keys = rows[["team", "season", "week"]].reset_index(drop=True)
        return renormalize_shares(raw, keys)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.payload, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = joblib.load(path)
        if payload.get("schema_version") != SERVING_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Plan A serving schema: {payload.get('schema_version')!r}")
        return cls(payload)


TARGET_ARMS = {
    "targets": "xgb_volume_blend_renorm",
    "receiving_yards": "xgb_volume_blend_renorm",
    "rushing_yards": "ridge_volume_blend_renorm",
    "rushing_attempts": "xgb_two_stage_rushing",
}


def _alpha(y, pred, base):
    from sklearn.metrics import mean_absolute_error
    return min(np.arange(0, 1.01, .05), key=lambda a: mean_absolute_error(y, a * pred + (1 - a) * base))


def _tiers(values, q1, q2):
    values = np.asarray(values, float)
    return np.where(values <= q1, "low", np.where(values <= q2, "mid", "high"))


def fit_validated_artifact(train: pd.DataFrame, target: str, test_season: int):
    """Fit the frozen target-specific Plan A winner on pre-test data."""
    from src.models.team_allocation.features import feature_columns, filter_population
    from src.models.team_allocation.baseline import RollingShareBaseline

    train = filter_population(train[train.season < test_season].copy(), target)
    label = f"share_of_team_{target}"; roll = f"{label}_roll3"
    cols = feature_columns(train); X, y = train[cols], train[label].to_numpy(float)
    arm = TARGET_ARMS[target]; payload = {
        "schema_version": SERVING_SCHEMA_VERSION, "target": target, "arm": arm,
        "test_season": int(test_season), "feature_columns": cols, "roll_column": roll,
    }
    q1, q2 = np.quantile(train[roll].fillna(0), [1/3, 2/3])
    payload["tier_q1"], payload["tier_q2"], payload["alpha_shrinkage_k"] = float(q1), float(q2), 25.0
    if target != "rushing_attempts":
        kind = "ridge" if target == "rushing_yards" else "xgb"
        model = ShareRidgeModel().fit(X, y) if kind == "ridge" else ShareXGBModel().fit(X, y)
        last = int(train.season.max()); cal = train.season.to_numpy() == last
        alpha_map = {"all": 0.5}
        if (~cal).sum() >= 20:
            cal_model = ShareRidgeModel().fit(X.iloc[~cal], y[~cal]) if kind == "ridge" else ShareXGBModel().fit(X.iloc[~cal], y[~cal])
            alpha_map = {}
            cal_pred = np.asarray(cal_model.predict(X.iloc[cal]), float)
            cal_base = train.iloc[cal][roll].fillna(0).to_numpy()
            for tier in ("low", "mid", "high"):
                mask = _tiers(train.iloc[cal][roll].fillna(0), q1, q2) == tier
                alpha_map[tier] = float(_alpha(y[cal][mask], cal_pred[mask], cal_base[mask])) if mask.sum() >= 20 else 0.5
            alpha_map["all"] = float(_alpha(y[cal], cal_pred, cal_base))
        payload.update({"model": model, "alpha_map": alpha_map})
    else:
        generic = ShareXGBModel().fit(X, y)
        positions = train.position.to_numpy(); models = {}; classifiers = {}
        for pos in np.unique(positions):
            mask = positions == pos; active = y[mask] > 0
            if mask.sum() < 40 or active.sum() < 20 or active.all(): continue
            classifiers[pos] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
            classifiers[pos].fit(X.loc[mask], active.astype(int))
            models[pos] = ShareXGBModel().fit(X.loc[mask].loc[active], y[mask][active])
        last = int(train.season.max()); cal = train.season.to_numpy() == last
        cal_pred = np.asarray(generic.predict(X.iloc[cal]), float); cal_base = train.iloc[cal][roll].fillna(0).to_numpy()
        cal_pos = train.iloc[cal].position.to_numpy(); cal_tier = _tiers(cal_base, q1, q2); alpha_map = {"all": float(_alpha(y[cal], cal_pred, cal_base))}
        for pos in np.unique(cal_pos):
            for tier in ("low", "mid", "high"):
                mask = (cal_pos == pos) & (cal_tier == tier)
                if mask.sum() >= 20:
                    raw = _alpha(y[cal][mask], cal_pred[mask], cal_base[mask]); w = mask.sum() / (mask.sum() + 25.0)
                    alpha_map[f"{pos}|{tier}"] = float(w * raw + (1 - w) * alpha_map["all"])
        payload.update({"generic_model": generic, "classifiers": classifiers, "positive_models": models, "alpha_map": alpha_map})
    return PlanAShareArtifact(payload)


def predict_validated_artifact(artifact: PlanAShareArtifact, rows: pd.DataFrame) -> np.ndarray:
    p = artifact.payload; X = rows[p["feature_columns"]]; base = rows[p["roll_column"]].fillna(0).to_numpy(float)
    tiers = _tiers(base, p["tier_q1"], p["tier_q2"]); target = p["target"]
    if target != "rushing_attempts":
        raw = np.asarray(p["model"].predict(X), float)
        pred = np.array([p["alpha_map"].get(t, p["alpha_map"].get("all", .5)) * r + (1 - p["alpha_map"].get(t, p["alpha_map"].get("all", .5))) * b for r, b, t in zip(raw, base, tiers)])
    else:
        pred = np.zeros(len(rows)); positions = rows.position.to_numpy()
        for pos in np.unique(positions):
            mask = positions == pos
            if pos not in p["classifiers"]: pred[mask] = base[mask]; continue
            prob = p["classifiers"][pos].predict_proba(X.loc[mask])[:, 1]
            positive = np.clip(p["positive_models"][pos].predict(X.loc[mask]), 0, 1)
            alpha = np.array([p["alpha_map"].get(f"{pos}|{t}", p["alpha_map"]["all"]) for t in tiers[mask]])
            pred[mask] = alpha * prob * positive + (1 - alpha) * base[mask]
    return renormalize_shares(np.clip(pred, 0, None), rows[["team", "season", "week"]].reset_index(drop=True))

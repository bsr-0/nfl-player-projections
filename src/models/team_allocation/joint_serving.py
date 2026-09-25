"""Artifact-backed shadow serving for the guarded full-PPR Plan A candidate.

This module is deliberately separate from the UI and the legacy yardage-only
Plan A serializer.  It is the single inference path used to create a shadow
export: train/freeze writes one immutable artifact, and export reloads that
artifact before producing component and PPR predictions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS
from src.evaluation.team_reconstruction_candidates import _team_feature_columns
from src.models.team_allocation.features import ROLL_WINDOW, feature_columns, filter_population
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel
from src.models.team_allocation.hierarchical import SPARSE_EVENT_TARGETS
from src.models.team_allocation.reconstruct import renormalize_shares
from src.utils.helpers import calculate_fantasy_points_df


JOINT_SERVING_SCHEMA_VERSION = "plan-a-joint-serving-v1"
KEY = ["player_id", "season", "week", "team", "position"]
TEAM_KEY = ["team", "season", "week"]


def _model(kind: str):
    if kind == "ridge":
        return ShareRidgeModel()
    if kind == "xgb":
        return ShareXGBModel()
    raise ValueError(f"unsupported model kind {kind!r}")


def _share_blend_alpha(train: pd.DataFrame, cols: list[str], label: str, roll: str, kind: str) -> float:
    """Select a blend weight on the final training season only."""
    last = int(train.season.max())
    cal = train.season.to_numpy(int) == last
    if cal.sum() < 20 or (~cal).sum() < 20:
        return 0.5
    model = _model(kind).fit(train.loc[~cal, cols], train.loc[~cal, label].to_numpy(float))
    raw = np.asarray(model.predict(train.loc[cal, cols]), float)
    base = train.loc[cal, roll].fillna(0.0).to_numpy(float)
    actual = train.loc[cal, label].to_numpy(float)
    return float(min(
        np.arange(0.0, 1.001, 0.05),
        key=lambda alpha: mean_absolute_error(actual, alpha * raw + (1.0 - alpha) * base),
    ))


def _fit_direct_share_arm(train_rows: pd.DataFrame, target: str, arm: str) -> dict[str, Any]:
    train = filter_population(train_rows, target).copy()
    if train.empty:
        raise ValueError(f"no training rows for {target}")
    label = f"share_of_team_{target}"
    roll = f"{label}_roll{ROLL_WINDOW}"
    if roll not in train:
        raise ValueError(f"missing rolling baseline column {roll!r}")
    payload: dict[str, Any] = {"target": target, "arm": arm, "roll_column": roll}
    if arm in {"rolling3", "rolling3_renorm"}:
        payload["kind"] = "rolling3"
        payload["renormalize"] = arm.endswith("_renorm")
        payload["zero_fallback"] = "equal"
        return payload
    if not arm.endswith("_blend_renorm"):
        raise ValueError(f"unsupported direct allocation arm {arm!r}")
    kind = arm.removesuffix("_blend_renorm")
    if kind not in {"ridge", "xgb"}:
        raise ValueError(f"unsupported allocation arm {arm!r}")
    cols = feature_columns(train, include_full_ppr=True)
    payload.update({
        "kind": kind,
        "feature_columns": cols,
        "model": _model(kind).fit(train[cols], train[label].to_numpy(float)),
        "blend_alpha": _share_blend_alpha(train, cols, label, roll, kind),
        "renormalize": True,
        "zero_fallback": "zeros" if target in SPARSE_EVENT_TARGETS else "equal",
    })
    return payload


def _fit_share_arm(train_rows: pd.DataFrame, target: str, arm: str) -> dict[str, Any]:
    """Persist the exact direct or outer rolling-3 convex allocation arm."""
    if not arm.startswith("blend:"):
        return _fit_direct_share_arm(train_rows, target, arm)
    _, candidate_arm, weight_text = arm.rsplit(":", 2)
    weight = float(weight_text)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"invalid outer blend weight {weight}")
    return {
        "target": target,
        "arm": arm,
        "kind": "outer_rolling3_blend",
        "weight": weight,
        "roll_column": f"share_of_team_{target}_roll{ROLL_WINDOW}",
        "candidate": _fit_direct_share_arm(train_rows, target, candidate_arm),
    }


def _predict_share_arm(payload: Mapping[str, Any], rows: pd.DataFrame) -> np.ndarray:
    kind = payload["kind"]
    base = rows[payload["roll_column"]].fillna(0.0).to_numpy(float)
    if kind == "rolling3":
        pred = base
    elif kind == "outer_rolling3_blend":
        candidate = _predict_share_arm(payload["candidate"], rows)
        pred = (1.0 - float(payload["weight"])) * base + float(payload["weight"]) * candidate
        # The selected outer blend is evaluated as a convex mixture of the
        # already-renormalized candidate and the raw rolling-3 incumbent.
        return np.clip(pred, 0.0, 1.0)
    else:
        expected = payload["feature_columns"]
        missing = sorted(set(expected).difference(rows.columns))
        if missing:
            raise ValueError(f"shadow rows missing allocation features: {missing}")
        raw = np.asarray(payload["model"].predict(rows[expected]), float)
        alpha = float(payload["blend_alpha"])
        pred = alpha * raw + (1.0 - alpha) * base
    pred = np.clip(pred, 0.0, None)
    if payload.get("renormalize", False):
        # Rolling3-renorm uses the backtester's equal fallback. Learned
        # sparse-event arms use zeros, so an all-zero TD/INT team-week does
        # not invent an event share at serving time.
        fallback = payload.get("zero_fallback")
        if fallback is None:  # Older v1 artifacts did not persist this field.
            fallback = ("zeros" if payload["target"] in SPARSE_EVENT_TARGETS and kind != "rolling3"
                        else "equal")
        pred = renormalize_shares(pred, rows[TEAM_KEY].reset_index(drop=True), zero_fallback=fallback)
    return np.clip(pred, 0.0, 1.0)


def _fit_team_arm(train_rows: pd.DataFrame, target: str, arm: str) -> dict[str, Any]:
    total_col = f"team_{target}"
    if arm == "rolling3":
        roll = f"{total_col}_roll{ROLL_WINDOW}"
        if roll not in train_rows:
            raise ValueError(f"missing rolling team-total column {roll!r}")
        return {"target": target, "arm": arm, "feature_columns": [], "roll_column": roll}
    cols = _team_feature_columns(train_rows, target)
    all_cols = list(dict.fromkeys(TEAM_KEY + cols + [total_col]))
    # Match the backtester's deterministic team-week representative exactly;
    # player rows carry repeated team fields and an unsorted duplicate drop
    # can otherwise select a different representative than validation did.
    team = train_rows[all_cols].sort_values(TEAM_KEY).drop_duplicates(TEAM_KEY).reset_index(drop=True)
    if team.empty:
        raise ValueError(f"no team rows for {target}")
    models = {kind: _model(kind).fit(team[cols], team[total_col].to_numpy(float)) for kind in ("ridge", "xgb")}
    payload: dict[str, Any] = {"target": target, "arm": arm, "feature_columns": cols, "models": models}
    if arm in {"ridge", "xgb"}:
        return payload
    if arm != "blend":
        raise ValueError(f"unsupported team-total arm {arm!r}")
    last = int(team.season.max())
    cal = team.season.to_numpy(int) == last
    if cal.sum() >= 10 and (~cal).sum() >= 20:
        prior_models = {
            kind: _model(kind).fit(team.loc[~cal, cols], team.loc[~cal, total_col].to_numpy(float))
            for kind in ("ridge", "xgb")
        }
        ridge = np.clip(np.asarray(prior_models["ridge"].predict(team.loc[cal, cols]), float), 0.0, None)
        xgb = np.clip(np.asarray(prior_models["xgb"].predict(team.loc[cal, cols]), float), 0.0, None)
        actual = team.loc[cal, total_col].to_numpy(float)
        alpha = min(
            np.arange(0.0, 1.001, 0.05),
            key=lambda a: mean_absolute_error(actual, a * ridge + (1.0 - a) * xgb),
        )
    else:
        alpha = 0.5
    payload["blend_alpha"] = float(alpha)
    return payload


def _predict_team_arm(payload: Mapping[str, Any], rows: pd.DataFrame) -> np.ndarray:
    if payload["arm"] == "rolling3":
        roll = payload["roll_column"]
        if roll not in rows:
            raise ValueError(f"shadow rows missing rolling team-total column {roll!r}")
        return rows[roll].to_numpy(float)
    cols = payload["feature_columns"]
    missing = sorted(set(cols).difference(rows.columns))
    if missing:
        raise ValueError(f"shadow rows missing team-total features: {missing}")
    team = rows[TEAM_KEY + cols].drop_duplicates(TEAM_KEY).copy()
    ridge = np.clip(np.asarray(payload["models"]["ridge"].predict(team[cols]), float), 0.0, None)
    xgb = np.clip(np.asarray(payload["models"]["xgb"].predict(team[cols]), float), 0.0, None)
    if payload["arm"] == "ridge":
        pred = ridge
    elif payload["arm"] == "xgb":
        pred = xgb
    else:
        alpha = float(payload["blend_alpha"])
        pred = alpha * ridge + (1.0 - alpha) * xgb
    team["_prediction"] = pred
    return rows[TEAM_KEY].merge(team[TEAM_KEY + ["_prediction"]], on=TEAM_KEY, how="left", validate="many_to_one")["_prediction"].to_numpy(float)


class JointPlanAArtifact:
    """A frozen full-PPR Plan A candidate with explicit arm provenance."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    @classmethod
    def fit(
        cls,
        rows: pd.DataFrame,
        *,
        allocation_arms: Mapping[str, str],
        team_total_arms: Mapping[str, str],
        train_through: int,
        train_from: int | None = None,
        version: str,
        source_selector: str,
    ) -> "JointPlanAArtifact":
        train = rows[rows.season <= train_through].copy()
        if train_from is not None:
            train = train[train.season >= train_from].copy()
        if train.empty or int(train.season.max()) != int(train_through):
            raise ValueError(f"expected training rows through {train_through}")
        targets = list(PPR_RECONSTRUCTION_TARGETS)
        if set(targets) != set(allocation_arms) or set(targets) != set(team_total_arms):
            raise ValueError("artifact requires exactly the eight selected full-PPR component arms")
        return cls({
            "schema_version": JOINT_SERVING_SCHEMA_VERSION,
            "model_version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "train_through": int(train_through),
            "train_from": int(train_from) if train_from is not None else None,
            "source_selector": source_selector,
            "targets": targets,
            "allocation_arms": dict(allocation_arms),
            "team_total_arms": dict(team_total_arms),
            "allocation": {target: _fit_share_arm(train, target, allocation_arms[target]) for target in targets},
            "team_totals": {target: _fit_team_arm(train, target, team_total_arms[target]) for target in targets},
        })

    def predict(self, rows: pd.DataFrame, *, include_actual: bool = True) -> pd.DataFrame:
        required = set(KEY + TEAM_KEY)
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise ValueError(f"shadow rows missing identity columns: {missing}")
        out = rows[KEY].drop_duplicates(KEY).copy()
        for target in self.payload["targets"]:
            population = filter_population(rows, target).copy()
            if population.empty:
                out[f"predicted_share_{target}"] = 0.0
                out[f"predicted_{target}"] = 0.0
                continue
            component = population[KEY].copy()
            component[f"predicted_share_{target}"] = _predict_share_arm(self.payload["allocation"][target], population)
            component[f"predicted_team_{target}"] = _predict_team_arm(self.payload["team_totals"][target], population)
            component[f"predicted_{target}"] = component[f"predicted_share_{target}"] * component[f"predicted_team_{target}"]
            out = out.merge(component, on=KEY, how="left", validate="one_to_one")
        for target in self.payload["targets"]:
            out[f"predicted_share_{target}"] = out[f"predicted_share_{target}"].fillna(0.0)
            out[f"predicted_{target}"] = out[f"predicted_{target}"].fillna(0.0)
        predicted_components = out[[f"predicted_{target}" for target in self.payload["targets"]]].rename(
            columns={f"predicted_{target}": target for target in self.payload["targets"]}
        )
        out["predicted_ppr"] = calculate_fantasy_points_df(predicted_components)
        # The explicit source frame is retained for historical shadow
        # evaluation but intentionally omitted in live exports.
        if include_actual and set(self.payload["targets"]).issubset(rows.columns):
            actual = rows[KEY + self.payload["targets"]].drop_duplicates(KEY)
            out = out.merge(actual, on=KEY, how="left", validate="one_to_one", suffixes=("", "_actual"))
            out["actual_ppr"] = calculate_fantasy_points_df(out[self.payload["targets"]])
        out["model_version"] = self.payload["model_version"]
        out["artifact_schema_version"] = self.payload["schema_version"]
        return out

    def metadata(self) -> dict[str, Any]:
        """Serializable, model-free release record for review and auditing."""
        return {
            key: self.payload[key]
            for key in (
                "schema_version", "model_version", "created_at", "train_from", "train_through", "source_selector",
                "targets", "allocation_arms", "team_total_arms",
            )
        } | {
            "allocation_details": {
                target: {k: v for k, v in payload.items() if k not in {"model", "candidate"}}
                for target, payload in self.payload["allocation"].items()
            },
            "team_total_details": {
                target: {k: v for k, v in payload.items() if k != "models"}
                for target, payload in self.payload["team_totals"].items()
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "JointPlanAArtifact":
        payload = joblib.load(path)
        if payload.get("schema_version") != JOINT_SERVING_SCHEMA_VERSION:
            raise ValueError(f"unsupported joint Plan A schema {payload.get('schema_version')!r}")
        return cls(payload)

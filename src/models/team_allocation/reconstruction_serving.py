"""Serialized serving path for the frozen reconstruction-aware candidate.

This is intentionally separate from the existing four-target Plan A serving
artifacts: it serves the yardage-slice candidate with a fold-local share blend
and a fold-local Ridge/XGBoost team-total ensemble. It is research-only until
the acceptance report promotes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from src.evaluation.team_reconstruction_candidates import (
    _best_volume_alpha,
    _fit,
    _team_feature_columns,
)
from src.models.team_allocation.features import ROLL_WINDOW, feature_columns, filter_population
from src.models.team_allocation.reconstruct import renormalize_shares

SCHEMA_VERSION = "plan-a-reconstruction-serving-v1"


class ReconstructionArtifact:
    def __init__(self, payload: dict):
        self.payload = payload

    @classmethod
    def fit(cls, rows: pd.DataFrame, target: str, test_season: int):
        train = filter_population(rows[rows.season < test_season].copy(), target)
        label = f"share_of_team_{target}"
        roll = f"{label}_roll{ROLL_WINDOW}"
        total_col = f"team_{target}"
        total_roll = f"team_{target}_roll{ROLL_WINDOW}"
        share_cols = feature_columns(train)
        share_model = _fit("xgb", train[share_cols], train[label].to_numpy(float))
        last = int(train.season.max())
        cal = train.season.to_numpy() == last
        if (~cal).sum() >= 20 and cal.sum() >= 20:
            cal_model = _fit("xgb", train.loc[~cal, share_cols], train.loc[~cal, label].to_numpy(float))
            alpha = _best_volume_alpha(
                train.loc[cal, label].to_numpy(float) * train.loc[cal, total_roll].to_numpy(float),
                np.asarray(cal_model.predict(train.loc[cal, share_cols]), float),
                train.loc[cal, roll].fillna(0).to_numpy(float),
                train.loc[cal, total_roll].to_numpy(float),
            )
        else:
            alpha = 0.0

        team_cols = _team_feature_columns(train, target)
        key = ["season", "week", "team"]
        team_cols_all = list(dict.fromkeys(key + team_cols + [total_col, total_roll]))
        team = train[team_cols_all].drop_duplicates(key).reset_index(drop=True)
        team_models = {
            kind: _fit(kind, team[team_cols], team[total_col].to_numpy(float))
            for kind in ("ridge", "xgb")
        }
        if (~cal).sum() >= 20 and cal.sum() >= 20:
            team_train = team[team.season < last]
            team_cal = team[team.season == last]
            if len(team_train) >= 20 and len(team_cal) >= 10:
                cal_team_models = {
                    kind: _fit(kind, team_train[team_cols], team_train[total_col].to_numpy(float))
                    for kind in ("ridge", "xgb")
                }
                cal_pred = {
                    kind: np.clip(np.asarray(model.predict(team_cal[team_cols]), float), 0, None)
                    for kind, model in cal_team_models.items()
                }
                actual = team_cal[total_col].to_numpy(float)
                team_alpha = min(
                    np.arange(0, 1.01, .05),
                    key=lambda a: np.mean(np.abs(actual - (a * cal_pred["ridge"] + (1 - a) * cal_pred["xgb"]))),
                )
            else:
                team_alpha = 0.5
        else:
            team_alpha = 0.5
        return cls({
            "schema_version": SCHEMA_VERSION,
            "target": target,
            "test_season": int(test_season),
            "share_columns": share_cols,
            "roll_column": roll,
            "team_columns": team_cols,
            "team_total_roll_column": total_roll,
            "share_model": share_model,
            "share_alpha": float(alpha),
            "team_models": team_models,
            "team_alpha": float(team_alpha),
        })

    def predict(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.payload
        base = rows[p["roll_column"]].fillna(0).to_numpy(float)
        raw = np.asarray(p["share_model"].predict(rows[p["share_columns"]]), float)
        blended = np.clip(p["share_alpha"] * raw + (1 - p["share_alpha"]) * base, 0, None)
        share = renormalize_shares(blended, rows[["team", "season", "week"]].reset_index(drop=True))
        key = ["season", "week", "team"]
        team_rows = rows[key + p["team_columns"]].drop_duplicates(key).copy()
        ridge = np.clip(np.asarray(p["team_models"]["ridge"].predict(team_rows[p["team_columns"]]), float), 0, None)
        xgb = np.clip(np.asarray(p["team_models"]["xgb"].predict(team_rows[p["team_columns"]]), float), 0, None)
        team_rows["_pred_total"] = p["team_alpha"] * ridge + (1 - p["team_alpha"]) * xgb
        total = rows[key].merge(team_rows[key + ["_pred_total"]], on=key, how="left")["_pred_total"].to_numpy(float)
        valid = np.isfinite(rows[p["team_total_roll_column"]].to_numpy(float))
        total = np.where(valid, total, np.nan)
        return share, total

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.payload, path)

    @classmethod
    def load(cls, path: str | Path):
        payload = joblib.load(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema: {payload.get('schema_version')!r}")
        return cls(payload)

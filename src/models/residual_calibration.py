"""Leakage-safe empirical residual calibration for player simulations.

The artifact models residual *shape*, not a new point forecast. Every draw is
centered on the supplied production prediction. Actual activity is retained as
a diagnostic because it is known only after a game and must never select a
future player's distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.models.oof_capture import ZERO_FLOOR


SCHEMA_VERSION = 1
MIN_STRATUM_ROWS = 50


def _activity_from_prediction(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.where(np.asarray(values, dtype=float) > ZERO_FLOOR, "predicted_active", "predicted_near_zero")


def _experience(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.where(np.asarray(values, dtype=bool), "cold_start", "returning")


def _key(position: str | None = None, activity: str | None = None,
         experience: str | None = None) -> str:
    return "|".join(value if value is not None else "*" for value in (position, activity, experience))


@dataclass(frozen=True)
class EmpiricalResidualCalibration:
    """Serializable residual pools with causal backoff selection."""

    residual_pools: dict[str, tuple[float, ...]]
    min_stratum_rows: int
    diagnostics: dict

    def select_pool(self, *, position: str, predicted_points: float,
                    is_cold_start: bool = False) -> tuple[float, ...]:
        if not np.isfinite(predicted_points):
            raise ValueError("predicted_points must be finite")
        pos = str(position).upper()
        activity = _activity_from_prediction(np.asarray([predicted_points]))[0]
        experience = _experience(np.asarray([is_cold_start]))[0]
        candidates = (
            _key(pos, activity, experience), _key(pos, activity, None),
            _key(pos, None, None), _key(None, None, None),
        )
        for candidate in candidates:
            pool = self.residual_pools.get(candidate)
            if pool:
                return pool
        raise ValueError(f"calibration artifact has no residual pool for position {pos}")

    def draw_residuals(self, *, position: str, predicted_points: float,
                       is_cold_start: bool, n_draws: int,
                       rng: np.random.Generator) -> np.ndarray:
        if n_draws < 2:
            raise ValueError("n_draws must be at least two")
        pool = np.asarray(self.select_pool(
            position=position, predicted_points=predicted_points,
            is_cold_start=is_cold_start), dtype=float)
        if len(pool) < 2 or not np.isfinite(pool).all():
            raise ValueError("selected residual pool is invalid")
        # Centering is deliberate: this calibrates uncertainty without
        # replacing the served point forecast with an in-sample residual bias.
        centered = pool - pool.mean()
        return rng.choice(centered, size=n_draws, replace=True)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "min_stratum_rows": self.min_stratum_rows,
            "residual_pools": {key: list(values) for key, values in self.residual_pools.items()},
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmpiricalResidualCalibration":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported residual-calibration schema")
        pools = {str(key): tuple(float(value) for value in values)
                 for key, values in payload.get("residual_pools", {}).items()}
        if _key(None, None, None) not in pools:
            raise ValueError("calibration artifact lacks global residual pool")
        return cls(pools, int(payload["min_stratum_rows"]), dict(payload.get("diagnostics", {})))


def fit_empirical_residual_calibration(panel: pd.DataFrame,
                                       *, min_stratum_rows: int = MIN_STRATUM_ROWS) -> EmpiricalResidualCalibration:
    """Fit causal selection pools and postgame activity diagnostics from OOF rows."""
    required = {"position", "predicted_points", "actual_points", "residual", "is_cold_start"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"OOF panel lacks calibration columns: {sorted(missing)}")
    if min_stratum_rows < 2:
        raise ValueError("min_stratum_rows must be at least two")
    frame = panel.copy()
    for column in ("predicted_points", "actual_points", "residual"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"OOF panel has null or nonfinite {column}")
    if not np.allclose(frame["residual"].to_numpy(float),
                       frame["predicted_points"].to_numpy(float) - frame["actual_points"].to_numpy(float),
                       atol=1e-12, rtol=0):
        raise ValueError("OOF residuals are not predicted_points - actual_points")
    frame["position"] = frame["position"].astype(str).str.upper()
    frame["predicted_activity"] = _activity_from_prediction(frame["predicted_points"])
    frame["experience"] = _experience(frame["is_cold_start"])
    frame["actual_activity"] = np.where(frame["actual_points"] > ZERO_FLOOR,
                                         "nonzero_actual", "near_zero_actual")

    pools: dict[str, tuple[float, ...]] = {}
    groupings = [
        (["position", "predicted_activity", "experience"], lambda row: _key(*row)),
        (["position", "predicted_activity"], lambda row: _key(row[0], row[1], None)),
        (["position"], lambda row: _key(row[0], None, None)),
    ]
    for columns, make_key in groupings:
        for values, group in frame.groupby(columns, dropna=False):
            values = values if isinstance(values, tuple) else (values,)
            if len(group) >= min_stratum_rows:
                pools[make_key(tuple(str(value) for value in values))] = tuple(group["residual"].astype(float))
    pools[_key(None, None, None)] = tuple(frame["residual"].astype(float))
    diagnostics = {
        "n_rows": int(len(frame)),
        "seasons": sorted(int(value) for value in frame["season"].unique()) if "season" in frame else [],
        "selection_strata": ["position", "predicted_activity", "experience"],
        "actual_activity_is_diagnostic_only": True,
        # Persist counts rather than infer them later from the serialized
        # residual values.  This makes a fitted artifact auditable without
        # allowing any postgame label to affect prospective pool selection.
        "pool_counts": {key: int(len(values)) for key, values in sorted(pools.items())},
        "by_position_actual_activity": [
            {"position": str(position), "actual_activity": str(activity), "n": int(len(group)),
             "residual_mean": float(group["residual"].mean()),
             "residual_sd": float(group["residual"].std(ddof=1)) if len(group) > 1 else 0.0}
            for (position, activity), group in frame.groupby(["position", "actual_activity"], dropna=False)
        ],
    }
    return EmpiricalResidualCalibration(pools, min_stratum_rows, diagnostics)


def save_calibration(artifact: EmpiricalResidualCalibration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_calibration(path: str | Path) -> EmpiricalResidualCalibration:
    return EmpiricalResidualCalibration.from_dict(json.loads(Path(path).read_text()))


def independent_residual_matrix(rows: pd.DataFrame, artifact: EmpiricalResidualCalibration,
                                *, n_draws: int, seed: int) -> np.ndarray:
    required = {"position", "predicted_points", "is_cold_start"}
    if missing := required - set(rows.columns):
        raise ValueError(f"simulation rows lack calibration columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    return np.column_stack([
        artifact.draw_residuals(position=row.position, predicted_points=float(row.predicted_points),
                                is_cold_start=bool(row.is_cold_start), n_draws=n_draws, rng=rng)
        for row in rows.itertuples(index=False)
    ])


def role_keys_from_panel_game(game_rows: pd.DataFrame) -> tuple[str, ...]:
    """Stable home/away role keys using only prediction-time row fields."""
    required = {"team", "home_team", "away_team", "position", "predicted_points", "player_id"}
    if missing := required - set(game_rows.columns):
        raise ValueError(f"game rows lack role columns: {sorted(missing)}")
    roles: dict[object, str] = {}
    for (team, position), group in game_rows.groupby(["team", "position"], sort=False):
        side = "home" if group["home_team"].iloc[0] == team else "away"
        if not ((group["home_team"] == team) | (group["away_team"] == team)).all():
            raise ValueError("player team is not scheduled in its game")
        ordered = group.sort_values(["predicted_points", "player_id"], ascending=[False, True])
        for rank, index in enumerate(ordered.index, start=1):
            roles[index] = f"{side}_{str(position).upper()}{rank}"
    return tuple(roles[index] for index in game_rows.index)


def induce_role_rank_dependence(residuals: np.ndarray, *, role_keys: tuple[str, ...],
                                correlation_model, seed: int) -> np.ndarray:
    """Reorder empirical draws by correlated ranks, preserving each marginal."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(role_keys) or not np.isfinite(values).all():
        raise ValueError("residual matrix must be finite draws by role-aligned players")
    scores = correlation_model.sample_scaled_residuals(
        role_keys, np.ones(len(role_keys)), values.shape[0], seed)
    dependent = np.empty_like(values)
    for column in range(values.shape[1]):
        ranks = np.argsort(np.argsort(scores[:, column], kind="mergesort"), kind="mergesort")
        dependent[:, column] = np.sort(values[:, column])[ranks]
    return dependent

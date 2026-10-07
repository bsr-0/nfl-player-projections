"""Leakage-safe empirical residual calibration for player simulations.

The artifact models residual *shape*, not a new point forecast. Every draw is
centered on the supplied production prediction. Actual activity is retained as
a diagnostic because it is known only after a game and must never select a
future player's distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

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
    _deviation_cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def _pool_key(self, *, position: str, predicted_points: float, is_cold_start: bool) -> str:
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
            if self.residual_pools.get(candidate):
                return candidate
        raise ValueError(f"calibration artifact has no residual pool for position {pos}")

    def select_pool(self, *, position: str, predicted_points: float,
                    is_cold_start: bool = False) -> tuple[float, ...]:
        return self.residual_pools[self._pool_key(
            position=position, predicted_points=predicted_points, is_cold_start=is_cold_start)]

    def _deviations(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._deviation_cache:
            pool = np.asarray(self.residual_pools[key], dtype=float)
            if len(pool) < 2 or not np.isfinite(pool).all():
                raise ValueError("selected residual pool is invalid")
            # Centering is deliberate: this calibrates uncertainty without
            # replacing the served point forecast with an in-sample residual
            # bias. Pools store oof_capture's residual (predicted - actual);
            # negate so deviations are actual-minus-prediction and callers can
            # use predicted + draw. Adding raw residuals mirrors the error
            # shape: right-skewed boom games become a heavy lower tail.
            deviations = pool.mean() - pool
            ordered = np.sort(deviations)
            # Callers receive the cached arrays themselves; read-only stops
            # an in-place edit from silently corrupting every later draw.
            deviations.setflags(write=False)
            ordered.setflags(write=False)
            self._deviation_cache[key] = (deviations, ordered)
        return self._deviation_cache[key]

    def support_deviations(self, *, position: str, predicted_points: float,
                           is_cold_start: bool = False) -> np.ndarray:
        """Every equally weighted actual-minus-prediction outcome this row can draw."""
        return self._deviations(self._pool_key(
            position=position, predicted_points=predicted_points, is_cold_start=is_cold_start))[0]

    def sorted_support_deviations(self, *, position: str, predicted_points: float,
                                  is_cold_start: bool = False) -> np.ndarray:
        return self._deviations(self._pool_key(
            position=position, predicted_points=predicted_points, is_cold_start=is_cold_start))[1]

    def draw_residuals(self, *, position: str, predicted_points: float,
                       is_cold_start: bool, n_draws: int,
                       rng: np.random.Generator) -> np.ndarray:
        if n_draws < 2:
            raise ValueError("n_draws must be at least two")
        deviations = self.support_deviations(
            position=position, predicted_points=predicted_points, is_cold_start=is_cold_start)
        return rng.choice(deviations, size=n_draws, replace=True)

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


ANALOG_SCHEMA_VERSION = 1
ANALOG_KIND = "prediction_analog"
ANALOG_FAMILIES = ("residual", "outcome")
ANALOG_GLOBAL_KEY = "*"


@dataclass(frozen=True)
class PredictionAnalogCalibration:
    """Residual shape from the k historical rows with the nearest prediction.

    Heteroscedasticity and shape both change continuously with the size of
    the served projection (a 4-point WR is zero-inflated and right-skewed; a
    20-point WR is wider and more symmetric), which fixed strata cannot
    follow. Donors are the k same-position OOF rows whose prediction is
    closest; a position with fewer than k rows backs off to all positions.

    Families (both centred, so every draw keeps the served mean):
    - ``residual``: deviation = donor (actual - predicted);
    - ``outcome``: deviation = donor actual - mean donor actual, i.e. the
      donors' realised outcomes shifted once to the served mean. This keeps
      the donors' zero mass and lower bound instead of re-imposing each
      donor's own miss on a different prediction.
    ``is_cold_start`` is accepted for interface parity and ignored.
    """

    family: str
    k: int
    predicted: dict[str, tuple[float, ...]]
    actual: dict[str, tuple[float, ...]]
    diagnostics: dict
    _arrays: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.family not in ANALOG_FAMILIES:
            raise ValueError(f"family must be one of {ANALOG_FAMILIES}")
        if int(self.k) < 2:
            raise ValueError("k must be at least two")
        if set(self.predicted) != set(self.actual) or ANALOG_GLOBAL_KEY not in self.predicted:
            raise ValueError("analog calibration needs aligned per-position and global arrays")
        for key in self.predicted:
            predicted = np.asarray(self.predicted[key], dtype=float)
            actual = np.asarray(self.actual[key], dtype=float)
            if len(predicted) != len(actual) or len(predicted) < 2:
                raise ValueError(f"analog arrays for {key!r} are misaligned or too short")
            if not (np.isfinite(predicted).all() and np.isfinite(actual).all()):
                raise ValueError(f"analog arrays for {key!r} are not finite")
            if (np.diff(predicted) < 0).any():
                raise ValueError(f"analog predictions for {key!r} are not sorted")
            predicted.setflags(write=False)
            actual.setflags(write=False)
            self._arrays[key] = (predicted, actual)

    def donors(self, *, position: str, predicted_points: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(predicted_points):
            raise ValueError("predicted_points must be finite")
        key = str(position).upper()
        if key not in self._arrays or len(self._arrays[key][0]) < self.k:
            key = ANALOG_GLOBAL_KEY
        predicted, actual = self._arrays[key]
        n = len(predicted)
        if n <= self.k:
            return predicted, actual
        # The k nearest values in a sorted array form a contiguous block that
        # lies inside [i - k, i + k), so only that window needs ranking.
        # Ties in distance break toward the lower index: deterministic.
        centre = int(np.searchsorted(predicted, predicted_points))
        window = np.arange(max(0, centre - self.k), min(n, centre + self.k))
        distance = np.abs(predicted[window] - predicted_points)
        chosen = np.sort(window[np.lexsort((window, distance))[:self.k]])
        return predicted[chosen], actual[chosen]

    def support_deviations(self, *, position: str, predicted_points: float,
                           is_cold_start: bool = False) -> np.ndarray:
        """Every equally weighted actual-minus-prediction outcome this row can draw."""
        donor_predicted, donor_actual = self.donors(
            position=position, predicted_points=predicted_points)
        deviations = donor_actual - donor_predicted if self.family == "residual" else donor_actual
        return deviations - deviations.mean()

    def sorted_support_deviations(self, *, position: str, predicted_points: float,
                                  is_cold_start: bool = False) -> np.ndarray:
        return np.sort(self.support_deviations(
            position=position, predicted_points=predicted_points, is_cold_start=is_cold_start))

    def draw_residuals(self, *, position: str, predicted_points: float,
                       is_cold_start: bool, n_draws: int,
                       rng: np.random.Generator) -> np.ndarray:
        if n_draws < 2:
            raise ValueError("n_draws must be at least two")
        return rng.choice(self.support_deviations(
            position=position, predicted_points=predicted_points), size=n_draws, replace=True)

    def to_dict(self) -> dict:
        return {
            "kind": ANALOG_KIND, "schema_version": ANALOG_SCHEMA_VERSION,
            "family": self.family, "k": int(self.k),
            "predicted": {key: list(values) for key, values in self.predicted.items()},
            "actual": {key: list(values) for key, values in self.actual.items()},
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PredictionAnalogCalibration":
        if payload.get("kind") != ANALOG_KIND or payload.get("schema_version") != ANALOG_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-analog calibration schema")
        return cls(
            str(payload["family"]), int(payload["k"]),
            {str(key): tuple(float(v) for v in values) for key, values in payload["predicted"].items()},
            {str(key): tuple(float(v) for v in values) for key, values in payload["actual"].items()},
            dict(payload.get("diagnostics", {})))


def fit_prediction_analog_calibration(panel: pd.DataFrame, *, family: str,
                                      k: int) -> PredictionAnalogCalibration:
    """Fit donor arrays from OOF rows; callers pass only rows before the target week."""
    required = {"position", "predicted_points", "actual_points"}
    if missing := required - set(panel.columns):
        raise ValueError(f"OOF panel lacks analog-calibration columns: {sorted(missing)}")
    frame = panel[list(required)].copy()
    for column in ("predicted_points", "actual_points"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"OOF panel has null or nonfinite {column}")
        frame[column] = values.astype(float)
    frame["position"] = frame["position"].astype(str).str.upper()

    def arrays(part: pd.DataFrame) -> tuple[tuple[float, ...], tuple[float, ...]]:
        predicted = part["predicted_points"].to_numpy(float)
        actual = part["actual_points"].to_numpy(float)
        order = np.lexsort((actual, predicted))
        return tuple(predicted[order]), tuple(actual[order])

    if len(frame) < 2:
        raise ValueError("analog calibration needs at least two OOF rows")
    predicted_by_key, actual_by_key = {}, {}
    counts = frame["position"].value_counts().sort_index()
    for position, part in frame.groupby("position", sort=True):
        # A position with fewer than k rows is never used on its own (it
        # backs off to all positions), so it needs no array of its own.
        if len(part) >= int(k):
            predicted_by_key[position], actual_by_key[position] = arrays(part)
    predicted_by_key[ANALOG_GLOBAL_KEY], actual_by_key[ANALOG_GLOBAL_KEY] = arrays(frame)
    diagnostics = {
        "n_rows": int(len(frame)), "family": family, "k": int(k),
        "rows_by_position": {pos: int(n) for pos, n in counts.items()},
        "positions_backing_off_to_global": sorted(pos for pos, n in counts.items() if n < int(k)),
    }
    return PredictionAnalogCalibration(family, int(k), predicted_by_key, actual_by_key, diagnostics)


def normal_scores(rows: pd.DataFrame, calibration) -> np.ndarray:
    """Phi^-1 of each realised outcome's mid-rank PIT within its own marginal.

    This is the Gaussian-copula scale that `induce_role_rank_dependence`
    samples on, so dependence fitted on these scores is the dependence the
    simulator reproduces (Pearson on raw, skewed residuals is not). The PIT
    is clipped to [0.5/n, 1 - 0.5/n] so outcomes beyond the support stay finite.
    """
    required = {"position", "predicted_points", "actual_points", "is_cold_start"}
    if missing := required - set(rows.columns):
        raise ValueError(f"rows lack normal-score columns: {sorted(missing)}")
    scores = np.empty(len(rows), dtype=float)
    for index, row in enumerate(rows.itertuples(index=False)):
        support = calibration.sorted_support_deviations(
            position=row.position, predicted_points=float(row.predicted_points),
            is_cold_start=bool(row.is_cold_start))
        observed = float(row.actual_points) - float(row.predicted_points)
        below = np.searchsorted(support, observed, side="left")
        through = np.searchsorted(support, observed, side="right")
        n = len(support)
        pit = (below + .5 * (through - below)) / n
        scores[index] = norm.ppf(np.clip(pit, .5 / n, 1 - .5 / n))
    return scores


def save_calibration(artifact, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_calibration(path: str | Path):
    payload = json.loads(Path(path).read_text())
    if payload.get("kind") == ANALOG_KIND:
        return PredictionAnalogCalibration.from_dict(payload)
    return EmpiricalResidualCalibration.from_dict(payload)


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

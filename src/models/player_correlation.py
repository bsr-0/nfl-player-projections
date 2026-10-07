"""Residual-dependence models for player simulations.

Everything is keyed by game role (home_WR1, away_RB2, ...), never player id:
the actual players change every game. ``FactorCopulaModel`` is the served
dependence; ``RoleResidualCorrelationModel`` survives only as the backtest's
legacy comparison arm.
"""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def _correlation(covariance: np.ndarray) -> np.ndarray:
    variance = np.clip(np.diag(covariance), 1e-12, None)
    scale = np.sqrt(variance)
    correlation = covariance / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    return correlation

def _sample_scaled(covariance: np.ndarray, marginal_sds: np.ndarray,
                   n_draws: int, seed: int) -> np.ndarray:
    scales = np.asarray(marginal_sds, dtype=float)
    if scales.ndim != 1 or scales.shape[0] != covariance.shape[0]:
        raise ValueError("marginal_sds must align with fitted correlation keys")
    if n_draws < 1 or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("draws must be positive and marginal_sds finite/positive")
    scaled_covariance = _correlation(covariance) * np.outer(scales, scales)
    return np.random.default_rng(seed).multivariate_normal(
        np.zeros(len(scales)), scaled_covariance, size=n_draws, check_valid="raise")

@dataclass
class RoleResidualCorrelationModel:
    """Role-keyed residual correlation model reusable across player lineups."""
    role_keys: tuple[str, ...]
    means: np.ndarray
    covariance: np.ndarray
    shrinkage: float

    def covers(self, role: str) -> bool:
        return role in self.role_keys

    def sample_scaled_residuals(self, requested_roles: tuple[str, ...],
                                marginal_sds: np.ndarray, n_draws: int,
                                seed: int = 42) -> np.ndarray:
        missing = set(requested_roles) - set(self.role_keys)
        if missing:
            raise ValueError(f"correlation artifact lacks requested roles: {sorted(missing)}")
        indices = [self.role_keys.index(role) for role in requested_roles]
        subset = self.covariance[np.ix_(indices, indices)]
        return _sample_scaled(subset, marginal_sds, n_draws, seed)

def _nearest_psd(matrix, floor=1e-8):
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T

def fit_sparse_role_residual_correlation(rows: np.ndarray, role_keys: list[str],
                                         shrinkage: float = 0.25,
                                         min_pair_rows: int = 20) -> RoleResidualCorrelationModel:
    """Fit a role artifact from incomplete historical game vectors.

    NFL game lineups do not contain every role in every game. Pairwise-complete
    covariance lets sparse roles retain a diagonal variance while unseen role
    pairs default to zero covariance; shrinkage plus PSD projection makes the
    result safe to sample for any later lineup.
    """
    residuals = np.asarray(rows, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(role_keys):
        raise ValueError("rows must be games by correlation keys")
    if residuals.shape[0] < 2 or len(set(role_keys)) != len(role_keys):
        raise ValueError("need at least two games and unique role keys")
    if not 0 <= shrinkage <= 1 or min_pair_rows < 2:
        raise ValueError("invalid shrinkage or min_pair_rows")
    means = np.nanmean(residuals, axis=0)
    if not np.isfinite(means).all():
        raise ValueError("every role must have at least one finite residual")
    covariance = np.zeros((len(role_keys), len(role_keys)), dtype=float)
    for i in range(len(role_keys)):
        valid_i = np.isfinite(residuals[:, i])
        if valid_i.sum() < 2:
            raise ValueError(f"role {role_keys[i]!r} has fewer than two residuals")
        covariance[i, i] = float(np.var(residuals[valid_i, i], ddof=1))
        for j in range(i + 1, len(role_keys)):
            valid = valid_i & np.isfinite(residuals[:, j])
            if valid.sum() >= min_pair_rows:
                value = float(np.cov(residuals[valid, i], residuals[valid, j], ddof=1)[0, 1])
                covariance[i, j] = covariance[j, i] = value
    diagonal = np.diag(np.diag(covariance))
    return RoleResidualCorrelationModel(
        tuple(role_keys), means,
        _nearest_psd((1 - shrinkage) * covariance + shrinkage * diagonal), shrinkage)


# ---------------------------------------------------------------------------
# Structured (factor) dependence on the Gaussian-copula scale.
# ---------------------------------------------------------------------------

MAX_ROLE_RANK = 3
ROLE_KEY_PATTERN = re.compile(r"^(home|away)_([A-Z]+)(\d+)$")
FACTOR_NAMES = ("game", "team_volume", "script")
FACTOR_STRUCTURES = ("team", "role")
FACTOR_KIND = "factor_copula"
FACTOR_SCHEMA_VERSION = 1
# Every role keeps at least this idiosyncratic variance on the copula scale.
UNIQUENESS_FLOOR = 1e-3
# Role loadings shrink toward the shared (team-structure) loading with the
# weight of this many observed pairs: negligible for starters with thousands
# of pairs, decisive for rare depth roles whose loadings the data cannot pin.
LOADING_PRIOR_PAIRS = 50.0
MIN_CLASS_PAIRS = 3


def parse_role_key(role: str) -> tuple[str, str]:
    """``home_WR5`` -> (``home``, ``WRx``): side plus depth-capped canonical role."""
    match = ROLE_KEY_PATTERN.match(str(role))
    if not match:
        raise ValueError(f"not a side_POSrank role key: {role!r}")
    side, position, rank = match.group(1), match.group(2), int(match.group(3))
    if rank < 1:
        raise ValueError(f"role rank must be >= 1: {role!r}")
    return side, f"{position}{rank}" if rank <= MAX_ROLE_RANK else f"{position}x"


def pooled_pair_correlations(scores: np.ndarray, role_keys, game_ids) -> pd.DataFrame:
    """Pooled same-game correlation of normal scores by (role pair, relation).

    Rows are player-games aligned with ``scores``. Every within-game pair
    contributes once to its class; relation is ``same_team`` when both
    players are on the same side. Pairs of one canonical role (e.g. two
    ``WRx`` teammates) enter in both orders so the class correlation is
    symmetric, while ``n`` still counts distinct pairs.
    """
    scores = np.asarray(scores, dtype=float)
    roles, games = list(role_keys), np.asarray(game_ids)
    if not (len(scores) == len(roles) == len(games)):
        raise ValueError("scores, role_keys and game_ids must align")
    if not np.isfinite(scores).all():
        raise ValueError("normal scores must be finite")
    parsed = [parse_role_key(role) for role in roles]
    columns = ["role_a", "role_b", "relation", "n", "correlation"]
    if len(scores) < 2:
        return pd.DataFrame(columns=columns)
    names = sorted({canonical for _, canonical in parsed})
    code = np.array([names.index(canonical) for _, canonical in parsed])
    home = np.array([side == "home" for side, _ in parsed])
    order = np.argsort(games, kind="mergesort")
    boundaries = np.flatnonzero(games[order][1:] != games[order][:-1]) + 1
    left, right = [], []
    for block in np.split(order, boundaries):
        if len(block) >= 2:
            first, second = np.triu_indices(len(block), k=1)
            left.append(block[first])
            right.append(block[second])
    if not left:
        return pd.DataFrame(columns=columns)
    i, j = np.concatenate(left), np.concatenate(right)
    # Orient every pair as (lower role code, higher role code) so a class is
    # one unordered role pair.
    swap = code[i] > code[j]
    i, j = np.where(swap, j, i), np.where(swap, i, j)
    same_role = code[i] == code[j]
    # Pairs of one canonical role enter in both orders at half weight.
    a = np.concatenate([scores[i], scores[j][same_role]])
    b = np.concatenate([scores[j], scores[i][same_role]])
    first_code = np.concatenate([code[i], code[i][same_role]])
    second_code = np.concatenate([code[j], code[j][same_role]])
    same_team = np.concatenate([home[i] == home[j], (home[i] == home[j])[same_role]])
    weight = np.concatenate([np.where(same_role, .5, 1.), np.full(same_role.sum(), .5)])
    frame = pd.DataFrame({"first": first_code, "second": second_code, "same": same_team,
                          "a": a, "b": b, "w": weight})
    output = []
    for (first, second, same), part in frame.groupby(["first", "second", "same"], sort=True):
        x, y = part["a"].to_numpy(float), part["b"].to_numpy(float)
        x, y = x - x.mean(), y - y.mean()
        denominator = np.sqrt((x * x).sum() * (y * y).sum())
        output.append({"role_a": names[first], "role_b": names[second],
                       "relation": "same_team" if same else "opponent",
                       "n": int(round(float(part["w"].sum()))),
                       "correlation": float((x * y).sum() / denominator) if denominator > 0 else float("nan")})
    return pd.DataFrame(output, columns=columns).sort_values(
        ["role_a", "role_b", "relation"], kind="mergesort").reset_index(drop=True)


def _implied(first: np.ndarray, second: np.ndarray, relation: str) -> float:
    game = first[0] * second[0]
    script = first[2] * second[2]
    if relation == "same_team":
        return float(game + first[1] * second[1] + script)
    if relation == "opponent":
        return float(game - script)
    raise ValueError(f"unknown relation {relation!r}")


@dataclass(frozen=True)
class FactorCopulaModel:
    """Gaussian-copula dependence from three latent game factors.

    z_i = g_i G + v_i T_team(i) + s_i (+1 home / -1 away) S + e_i, with G a
    shared game-environment factor, T one independent volume factor per
    team, S an antisymmetric game-script factor (what raises one side's
    passing lowers the other's), and Var(z_i) = 1. Implied correlations:
    same team g_i g_j + v_i v_j + s_i s_j; opponents g_i g_j - s_i s_j.
    Being an explicit factor model, every lineup's matrix is PSD by
    construction -- no projection step distorts the fitted values.

    ``structure="team"`` shares one loading across roles (two free moments:
    pooled same-team and opponent correlation); ``structure="role"`` fits a
    loading per canonical role. Roles absent from training fall back to
    ``default_loading`` (the shared loading for "team", zero for "role").
    """

    structure: str
    loadings: dict[str, tuple[float, float, float]]
    default_loading: tuple[float, float, float]
    diagnostics: dict

    def __post_init__(self):
        if self.structure not in FACTOR_STRUCTURES:
            raise ValueError(f"structure must be one of {FACTOR_STRUCTURES}")
        for role, loading in {**self.loadings, "<default>": self.default_loading}.items():
            values = np.asarray(loading, dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"loading for {role!r} must be three finite values")
            if float(values @ values) > 1 - UNIQUENESS_FLOOR + 1e-9:
                raise ValueError(f"loading for {role!r} leaves no idiosyncratic variance")

    def covers(self, role: str) -> bool:
        try:
            parse_role_key(role)
        except ValueError:
            return False
        return True

    def loading(self, canonical_role: str) -> np.ndarray:
        return np.asarray(self.loadings.get(canonical_role, self.default_loading), dtype=float)

    def implied_correlation(self, role_a: str, role_b: str, relation: str) -> float:
        return _implied(self.loading(role_a), self.loading(role_b), relation)

    def _loading_matrix(self, requested_roles) -> np.ndarray:
        rows = []
        for role in requested_roles:
            side, canonical = parse_role_key(role)
            game, volume, script = self.loading(canonical)
            home = side == "home"
            rows.append([game, volume if home else 0., 0. if home else volume,
                         script if home else -script])
        return np.asarray(rows, dtype=float).reshape(len(rows), 4)

    def correlation_matrix(self, requested_roles) -> np.ndarray:
        loadings = self._loading_matrix(requested_roles)
        matrix = loadings @ loadings.T
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def sample_scaled_residuals(self, requested_roles: tuple[str, ...],
                                marginal_sds: np.ndarray, n_draws: int,
                                seed: int = 42) -> np.ndarray:
        scales = np.asarray(marginal_sds, dtype=float)
        if scales.ndim != 1 or scales.shape[0] != len(requested_roles):
            raise ValueError("marginal_sds must align with requested roles")
        if n_draws < 1 or not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("draws must be positive and marginal_sds finite/positive")
        loadings = self._loading_matrix(requested_roles)
        uniqueness = 1.0 - (loadings ** 2).sum(axis=1)
        rng = np.random.default_rng(seed)
        factors = rng.standard_normal((n_draws, loadings.shape[1]))
        noise = rng.standard_normal((n_draws, len(requested_roles)))
        return (factors @ loadings.T + noise * np.sqrt(uniqueness)) * scales

    def to_dict(self) -> dict:
        return {"kind": FACTOR_KIND, "schema_version": FACTOR_SCHEMA_VERSION,
                "structure": self.structure,
                "loadings": {role: list(map(float, values)) for role, values in sorted(self.loadings.items())},
                "default_loading": list(map(float, self.default_loading)),
                "diagnostics": self.diagnostics}

    @classmethod
    def from_dict(cls, payload: dict) -> "FactorCopulaModel":
        if payload.get("kind") != FACTOR_KIND or payload.get("schema_version") != FACTOR_SCHEMA_VERSION:
            raise ValueError("unsupported factor-copula schema")
        return cls(str(payload["structure"]),
                   {str(role): tuple(float(v) for v in values) for role, values in payload["loadings"].items()},
                   tuple(float(v) for v in payload["default_loading"]),
                   dict(payload.get("diagnostics", {})))


def _usable_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"role_a", "role_b", "relation", "n", "correlation"}
    if missing := required - set(pairs.columns):
        raise ValueError(f"pair table lacks columns: {sorted(missing)}")
    usable = pairs.loc[(pairs["n"] >= MIN_CLASS_PAIRS) & np.isfinite(pairs["correlation"].astype(float))]
    return usable.reset_index(drop=True)


def _shared_loading(pairs: pd.DataFrame) -> tuple[tuple[float, float, float], dict]:
    """Loading reproducing the pooled same-team and opponent correlations."""
    moments = {}
    for relation in ("same_team", "opponent"):
        part = pairs.loc[pairs["relation"] == relation]
        moments[relation] = (float(np.average(part["correlation"], weights=part["n"]))
                             if len(part) else 0.0)
    same, opponent = moments["same_team"], moments["opponent"]
    game_sq, script_sq = max(opponent, 0.0), max(-opponent, 0.0)
    volume_sq = max(same - game_sq - script_sq, 0.0)
    total = game_sq + volume_sq + script_sq
    cap = 1 - UNIQUENESS_FLOOR
    scale = min(1.0, cap / total) if total > 0 else 1.0
    loading = tuple(float(np.sqrt(value * scale)) for value in (game_sq, volume_sq, script_sq))
    diagnostics = {"pooled_same_team": same, "pooled_opponent": opponent,
                   "representable": bool(same >= abs(opponent) and total <= cap)}
    return loading, diagnostics


def _to_loadings(raw: np.ndarray) -> np.ndarray:
    """Unconstrained rows -> loadings with squared norm strictly below the cap."""
    cap = np.sqrt(1 - UNIQUENESS_FLOOR)
    return raw * cap / np.sqrt(1 + (raw ** 2).sum(axis=1, keepdims=True))


def _from_loadings(loadings: np.ndarray) -> np.ndarray:
    cap_sq = 1 - UNIQUENESS_FLOOR
    norm_sq = np.minimum((loadings ** 2).sum(axis=1, keepdims=True), cap_sq * (1 - 1e-6))
    return loadings / np.sqrt(cap_sq - norm_sq)


def _normalise_signs(loadings: np.ndarray, roles: list[str]) -> np.ndarray:
    """Factor signs are unidentified; fix them so fitted output is reproducible."""
    result = loadings.copy()
    for column in range(result.shape[1]):
        anchor = result[roles.index("QB1"), column] if "QB1" in roles else result[:, column].sum()
        if anchor < 0 or (anchor == 0 and result[:, column].sum() < 0):
            result[:, column] *= -1
    return result


def fit_factor_copula(pairs: pd.DataFrame, *, structure: str) -> FactorCopulaModel:
    """Fit loadings by n-weighted least squares to pooled pair correlations."""
    if structure not in FACTOR_STRUCTURES:
        raise ValueError(f"structure must be one of {FACTOR_STRUCTURES}")
    usable = _usable_pairs(pairs)
    if usable.empty:
        return FactorCopulaModel(structure, {}, (0.0, 0.0, 0.0),
                                 {"status": "no_usable_pairs", "n_classes": 0, "n_pairs": 0})
    shared, shared_diagnostics = _shared_loading(usable)
    base = {"n_classes": int(len(usable)), "n_pairs": int(usable["n"].sum()), **shared_diagnostics}
    if structure == "team":
        return FactorCopulaModel("team", {}, shared, {"status": "fitted", **base})

    roles = sorted(set(usable["role_a"]) | set(usable["role_b"]))
    index = {role: i for i, role in enumerate(roles)}
    first = usable["role_a"].map(index).to_numpy()
    second = usable["role_b"].map(index).to_numpy()
    same = (usable["relation"] == "same_team").to_numpy()
    target = usable["correlation"].to_numpy(float)
    weight = np.sqrt(usable["n"].to_numpy(float))
    shared_array = np.asarray(shared, dtype=float)
    prior = np.sqrt(LOADING_PRIOR_PAIRS)

    def residuals(flat: np.ndarray) -> np.ndarray:
        loadings = _to_loadings(flat.reshape(len(roles), 3))
        a, b = loadings[first], loadings[second]
        implied = a[:, 0] * b[:, 0] + np.where(same, a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2],
                                                -a[:, 2] * b[:, 2])
        return np.concatenate([weight * (implied - target),
                               prior * (loadings - shared_array).reshape(-1)])

    start = _from_loadings(np.tile(shared_array, (len(roles), 1)))
    best = None
    for seed in range(4):
        # The script factor is invisible at a symmetric start; deterministic
        # perturbations let the optimiser find it, and the best of several
        # starts guards against a poor local minimum.
        x0 = (start + np.random.default_rng(seed).normal(0, .3, start.shape)).reshape(-1)
        result = least_squares(residuals, x0, method="trf")
        if best is None or result.cost < best.cost:
            best = result
    loadings = _normalise_signs(_to_loadings(best.x.reshape(len(roles), 3)), roles)
    fitted = {role: tuple(float(v) for v in loadings[i]) for role, i in index.items()}
    model = FactorCopulaModel("role", fitted, (0.0, 0.0, 0.0), {})
    implied = np.array([model.implied_correlation(a, b, r) for a, b, r in
                        usable[["role_a", "role_b", "relation"]].itertuples(index=False)])
    diagnostics = {"status": "fitted" if best.success else "optimizer_not_converged",
                   "optimizer_cost": float(best.cost), **base,
                   "weighted_rmse": float(np.sqrt(np.average((implied - target) ** 2, weights=usable["n"])))}
    return FactorCopulaModel("role", fitted, (0.0, 0.0, 0.0), diagnostics)

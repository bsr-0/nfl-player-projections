"""Residual-correlation primitives for player simulations.

Player-keyed models are useful only for a fixed, already-aligned lineup.
Role-keyed models are the production-oriented interface because the actual
players change every game; examples are home_WR1 and away_RB1.
"""
from dataclasses import dataclass
import numpy as np

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
class ResidualCorrelationModel:
    """Fixed-player correlation model; do not use across changing lineups."""
    player_keys: tuple[str, ...]
    means: np.ndarray
    covariance: np.ndarray
    shrinkage: float

    def sample_residuals(self, n_draws: int, seed: int = 42) -> np.ndarray:
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        return np.random.default_rng(seed).multivariate_normal(
            self.means, self.covariance, size=n_draws, check_valid="raise")

    def sample_scaled_residuals(self, marginal_sds: np.ndarray,
                                n_draws: int, seed: int = 42) -> np.ndarray:
        return _sample_scaled(self.covariance, marginal_sds, n_draws, seed)

@dataclass
class RoleResidualCorrelationModel:
    """Role-keyed residual correlation model reusable across player lineups."""
    role_keys: tuple[str, ...]
    means: np.ndarray
    covariance: np.ndarray
    shrinkage: float

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

def _fit(rows: np.ndarray, keys: list[str], shrinkage: float):
    residuals = np.asarray(rows, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(keys):
        raise ValueError("rows must be games by correlation keys")
    if residuals.shape[0] < 2 or not np.isfinite(residuals).all():
        raise ValueError("need at least two finite games")
    if len(set(keys)) != len(keys):
        raise ValueError("correlation keys must be unique")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    return residuals.mean(axis=0), _nearest_psd(
        (1 - shrinkage) * covariance + shrinkage * diagonal)

def fit_residual_correlation(rows: np.ndarray, player_keys: list[str],
                             shrinkage: float = 0.25) -> ResidualCorrelationModel:
    means, covariance = _fit(rows, player_keys, shrinkage)
    return ResidualCorrelationModel(tuple(player_keys), means, covariance, shrinkage)

def fit_role_residual_correlation(rows: np.ndarray, role_keys: list[str],
                                  shrinkage: float = 0.25) -> RoleResidualCorrelationModel:
    means, covariance = _fit(rows, role_keys, shrinkage)
    return RoleResidualCorrelationModel(tuple(role_keys), means, covariance, shrinkage)

def apply_shared_residuals(point_predictions: np.ndarray,
                           model: ResidualCorrelationModel, seed: int = 42) -> np.ndarray:
    point = np.asarray(point_predictions, dtype=float)
    if point.ndim != 1 or point.shape[0] != len(model.player_keys):
        raise ValueError("point predictions must align with fitted keys")
    return point + model.sample_residuals(1, seed)[0]

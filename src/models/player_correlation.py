"""Shared-residual correlation layer for player simulations."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ResidualCorrelationModel:
    player_keys: tuple[str, ...]
    means: np.ndarray
    covariance: np.ndarray
    shrinkage: float

    def sample_residuals(self, n_draws: int, seed: int = 42) -> np.ndarray:
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        return np.random.default_rng(seed).multivariate_normal(
            self.means, self.covariance, size=n_draws, check_valid="raise")

def _nearest_psd(matrix, floor=1e-8):
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, floor)) @ vectors.T

def fit_residual_correlation(rows: np.ndarray, player_keys: list[str],
                             shrinkage: float = 0.25) -> ResidualCorrelationModel:
    residuals = np.asarray(rows, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(player_keys):
        raise ValueError("rows must be games by players")
    if residuals.shape[0] < 2 or not np.isfinite(residuals).all():
        raise ValueError("need at least two finite games")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(covariance))
    covariance = _nearest_psd((1 - shrinkage) * covariance + shrinkage * diagonal)
    return ResidualCorrelationModel(tuple(player_keys), residuals.mean(axis=0), covariance, shrinkage)

def apply_shared_residuals(point_predictions: np.ndarray,
                           model: ResidualCorrelationModel, seed: int = 42) -> np.ndarray:
    point = np.asarray(point_predictions, dtype=float)
    if point.ndim != 1 or point.shape[0] != len(model.player_keys):
        raise ValueError("point predictions must align with fitted keys")
    return point + model.sample_residuals(1, seed)[0]

"""Shared-residual correlation layer for player simulations."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ResidualCorrelationModel:
    player_keys: tuple[str, ...]
    means: np.ndarray
    covariance: np.ndarray
    shrinkage: float

    def _correlation(self) -> np.ndarray:
        variance = np.clip(np.diag(self.covariance), 1e-12, None)
        scale = np.sqrt(variance)
        correlation = self.covariance / np.outer(scale, scale)
        correlation = (correlation + correlation.T) / 2.0
        np.fill_diagonal(correlation, 1.0)
        return correlation

    def sample_residuals(self, n_draws: int, seed: int = 42) -> np.ndarray:
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        return np.random.default_rng(seed).multivariate_normal(
            self.means, self.covariance, size=n_draws, check_valid="raise")

    def sample_scaled_residuals(self, marginal_sds: np.ndarray,
                                n_draws: int, seed: int = 42) -> np.ndarray:
        """Draw correlated residuals with the caller's calibrated marginal SDs.

        Player intervals describe total player uncertainty. We preserve those
        marginal scales while importing only the fitted teammate correlation,
        avoiding double-counting residual variance.
        """
        scales = np.asarray(marginal_sds, dtype=float)
        if scales.ndim != 1 or scales.shape[0] != len(self.player_keys):
            raise ValueError("marginal_sds must align with fitted player keys")
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("marginal_sds must be finite and positive")
        covariance = self._correlation() * np.outer(scales, scales)
        return np.random.default_rng(seed).multivariate_normal(
            np.zeros(len(scales)), covariance, size=n_draws, check_valid="raise")

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
    if len(set(player_keys)) != len(player_keys):
        raise ValueError("player_keys must be unique")
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

"""Conservative team-opportunity allocation for simulation draws."""
from __future__ import annotations

import numpy as np

def draw_usage_shares(shares: np.ndarray, *, concentration: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Draw player shares while conserving the modeled team's opportunity pool.

    The unmodeled remainder receives its own Dirichlet bucket, so a partial
    player list never falsely claims every target/carry belongs to it.
    """
    shares = np.asarray(shares, dtype=float)
    if shares.ndim != 1 or len(shares) == 0:
        raise ValueError("shares must be a non-empty vector")
    if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 1).any():
        raise ValueError("shares must be finite probabilities in [0, 1]")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be positive")
    total = float(shares.sum())
    if total > 1.0 + 1e-9:
        raise ValueError("player usage shares cannot exceed one team opportunity pool")
    remainder = max(0.0, 1.0 - total)
    alpha = np.maximum(np.append(shares, remainder) * concentration, 1e-6)
    return rng.dirichlet(alpha)[:len(shares)]

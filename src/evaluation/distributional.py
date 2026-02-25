"""Distributional predictions: boom/bust probabilities from uncertainty estimates.

Given a model's point predictions and standard deviations, this module
computes per-player probabilities of:
- **Boom**: exceeding a position-specific high threshold (top-12 weekly finish).
- **Bust**: falling below a position-specific low threshold (replacement level).

These probabilities enable downstream consumers (lineup optimizers, DFS stacks)
to make risk-aware decisions rather than relying solely on point estimates.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Position-specific thresholds (PPR scoring, approximate weekly values)
# Boom = top-12 weekly finish; Bust = below replacement level
BOOM_THRESHOLDS: Dict[str, float] = {
    "QB": 25.0,
    "RB": 18.0,
    "WR": 18.0,
    "TE": 14.0,
}

BUST_THRESHOLDS: Dict[str, float] = {
    "QB": 10.0,
    "RB": 5.0,
    "WR": 5.0,
    "TE": 3.0,
}


def boom_bust_probabilities(
    predictions: np.ndarray,
    std_devs: np.ndarray,
    positions: np.ndarray,
    boom_thresholds: Optional[Dict[str, float]] = None,
    bust_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute per-player boom and bust probabilities.

    Assumes predictions follow a Gaussian distribution centered on the
    point prediction with the given standard deviation.  This is a
    reasonable first-order approximation when the model's uncertainty
    estimates are calibrated (see C5 conformal recalibration).

    Args:
        predictions: Point predictions (fantasy points).
        std_devs: Predicted standard deviations.
        positions: Array of position labels (QB, RB, WR, TE).
        boom_thresholds: Override position-specific boom thresholds.
        bust_thresholds: Override position-specific bust thresholds.

    Returns:
        Dict with keys: boom_prob, bust_prob, safe_floor (10th percentile),
        upside_ceiling (90th percentile).
    """
    boom_t = boom_thresholds or BOOM_THRESHOLDS
    bust_t = bust_thresholds or BUST_THRESHOLDS

    n = len(predictions)
    boom_prob = np.zeros(n)
    bust_prob = np.zeros(n)
    safe_floor = np.zeros(n)
    upside_ceiling = np.zeros(n)

    for i in range(n):
        pred = predictions[i]
        std = max(std_devs[i], 1e-6)
        pos = positions[i] if i < len(positions) else "WR"

        boom_threshold = boom_t.get(pos, 18.0)
        bust_threshold = bust_t.get(pos, 5.0)

        # P(X > boom_threshold) = 1 - CDF(boom_threshold)
        boom_prob[i] = 1.0 - norm.cdf(boom_threshold, loc=pred, scale=std)
        # P(X < bust_threshold) = CDF(bust_threshold)
        bust_prob[i] = norm.cdf(bust_threshold, loc=pred, scale=std)
        # 10th and 90th percentile
        safe_floor[i] = norm.ppf(0.10, loc=pred, scale=std)
        upside_ceiling[i] = norm.ppf(0.90, loc=pred, scale=std)

    return {
        "boom_prob": boom_prob,
        "bust_prob": bust_prob,
        "safe_floor": safe_floor,
        "upside_ceiling": upside_ceiling,
    }


def classify_risk_tier(
    boom_prob: np.ndarray,
    bust_prob: np.ndarray,
) -> np.ndarray:
    """Classify each player into a risk tier.

    Returns:
        Array of strings: "high_upside", "safe_floor", "volatile", "risky".
    """
    n = len(boom_prob)
    tiers = np.empty(n, dtype=object)

    for i in range(n):
        bp = boom_prob[i]
        busp = bust_prob[i]

        if bp > 0.30 and busp < 0.15:
            tiers[i] = "high_upside"
        elif bp < 0.10 and busp < 0.10:
            tiers[i] = "safe_floor"
        elif bp > 0.20 and busp > 0.20:
            tiers[i] = "volatile"
        else:
            tiers[i] = "moderate"

    return tiers


def format_distributional_summary(
    predictions: np.ndarray,
    std_devs: np.ndarray,
    positions: np.ndarray,
    names: Optional[np.ndarray] = None,
    top_n: int = 10,
) -> str:
    """Format a human-readable summary of boom/bust probabilities."""
    dist = boom_bust_probabilities(predictions, std_devs, positions)
    tiers = classify_risk_tier(dist["boom_prob"], dist["bust_prob"])

    lines = [
        "=" * 70,
        "DISTRIBUTIONAL PREDICTION SUMMARY",
        "=" * 70,
        "",
        f"{'Player':<25} {'Pred':>6} {'Boom%':>6} {'Bust%':>6} {'Floor':>6} {'Ceil':>6} {'Tier':<12}",
        "-" * 70,
    ]

    # Sort by prediction descending
    order = np.argsort(-predictions)[:top_n]
    for idx in order:
        name = names[idx] if names is not None else f"Player_{idx}"
        lines.append(
            f"{str(name):<25} {predictions[idx]:>6.1f} "
            f"{dist['boom_prob'][idx]*100:>5.1f}% "
            f"{dist['bust_prob'][idx]*100:>5.1f}% "
            f"{dist['safe_floor'][idx]:>6.1f} "
            f"{dist['upside_ceiling'][idx]:>6.1f} "
            f"{tiers[idx]:<12}"
        )

    return "\n".join(lines)

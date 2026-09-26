"""Multiple-comparison correction for a family of per-cell p-values.

A segment table with M cells (e.g. position x cold-start x week-bucket) runs
M independent tests. At a nominal 5% false-positive rate per cell, the
chance at least one cell looks "significant" with NO real effect anywhere is
1 - 0.95**M -- 56% at M=16, not 5%. Highlighting whichever cell has the
smallest p-value from an uncorrected table is exactly this failure mode.

Two standard corrections, no scipy/statsmodels dependency (both are a few
lines of sorted-array arithmetic):

  holm: controls the family-wise error rate (P(>=1 false positive) <= alpha)
        exactly, and is uniformly more powerful than plain Bonferroni. Use
        this when a false positive is costly to act on (a segment finding
        that leads to a real code/architecture decision).
  bh:   controls the false discovery rate (expected proportion of false
        positives AMONG rejections <= alpha), which is less conservative
        than Holm once M is large. Use this for exploratory screening across
        many segments where some false positives are an acceptable cost of
        not missing real ones.

NaN p-values (a cell with too few clusters to bootstrap, see
oof_capture.cluster_bootstrap_ci) are excluded from the family -- they were
never testable, so they must not consume part of the alpha budget or shrink
the multiplicity penalty for the cells that were.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def holm_bonferroni(p_values: Sequence[Optional[float]], alpha: float = 0.05) -> List[bool]:
    """Step-down Holm-Bonferroni. Returns one bool per input, same order.

    NaN entries return False (untestable, not "not significant") and do not
    occupy a slot in the m used to compute other cells' thresholds.
    """
    return _step_procedure(p_values, alpha, mode="holm")


def benjamini_hochberg(p_values: Sequence[Optional[float]], alpha: float = 0.05) -> List[bool]:
    """Step-up Benjamini-Hochberg (false discovery rate control).

    NaN entries return False and are excluded from m, as in `holm_bonferroni`.
    """
    return _step_procedure(p_values, alpha, mode="bh")


def _step_procedure(p_values: Sequence[Optional[float]], alpha: float, *, mode: str) -> List[bool]:
    testable = [(i, p) for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    result = [False] * len(p_values)
    if not testable:
        return result

    m = len(testable)
    ordered = sorted(testable, key=lambda t: t[1])  # ascending by p-value

    if mode == "holm":
        # Reject p(1)..p(k) where k is the largest index such that
        # p(i) <= alpha / (m - i + 1) for every i <= k (1-indexed). Step
        # DOWN from the smallest p-value; stop at the first failure.
        reject_through = 0
        for rank, (_, p) in enumerate(ordered, start=1):
            threshold = alpha / (m - rank + 1)
            if p <= threshold:
                reject_through = rank
            else:
                break
        for rank in range(reject_through):
            original_index = ordered[rank][0]
            result[original_index] = True

    elif mode == "bh":
        # Reject p(1)..p(k) where k is the LARGEST index such that
        # p(i) <= (i/m) * alpha. Step UP from the largest p-value looking
        # for the first (largest-index) one that clears its threshold.
        reject_through = 0
        for rank, (_, p) in enumerate(ordered, start=1):
            threshold = (rank / m) * alpha
            if p <= threshold:
                reject_through = rank
        for rank in range(reject_through):
            original_index = ordered[rank][0]
            result[original_index] = True

    else:
        raise ValueError(f"unknown correction mode: {mode!r}")

    return result


def bootstrap_two_sided_p_value(draws: np.ndarray) -> float:
    """A p-value from bootstrap draws of a statistic, consistent with a CI
    built from the SAME draws (see `cluster_bootstrap_distribution`).

    p = 2 * min(P(draw <= 0), P(draw >= 0)), i.e. how much of the bootstrap
    distribution's mass is on the opposite side of zero from where it's
    centred -- doubled for a two-sided test. A (add-one) continuity
    correction avoids ever reporting exactly p=0 from a finite number of
    draws, which would overstate confidence past what n_boot draws can
    actually resolve.
    """
    if draws.size == 0:
        return float("nan")
    n = draws.size
    p_low = (np.sum(draws <= 0) + 1) / (n + 1)
    p_high = (np.sum(draws >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(p_low, p_high)))

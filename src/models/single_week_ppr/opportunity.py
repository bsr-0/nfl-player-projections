"""Opportunity modelling: predict snaps, then production conditional on them.

The production model retains only 51-68% of the true dynamic range across
realized-snap buckets, over-predicting low-snap games and under-predicting
high-snap ones (GAPS.md 2026-08-20). Realized snaps are not knowable at
forecast time, so that gap measures opportunity uncertainty, not production
error -- which is the entire motivation for splitting the two.

    E[PPR] = E[snaps] x E[PPR per snap]

Both factors are estimated from the SAME leakage-safe feature set the
baseline uses; no realized-week column appears in CAUSAL_FEATURES at any
position, so the set is as safe for a snaps target as for a points target.

Two things this form assumes, stated rather than buried:

1. Snaps and per-snap efficiency are conditionally independent given
   features. E[XY] = E[X]E[Y] only holds under independence. This is the
   assumption under test, not a derivation.
2. Both components must be MEAN-oriented -- a product of medians does not
   approximate a mean -- so both use MSE objectives even though the
   baseline's per-position architectures are mostly median/Huber.

The per-snap model is fitted with `sample_weight = snap_count x recency`.
Without it a 1-snap, 6-point game contributes a 6.0 per-snap rate with the
same influence as a 60-snap starter, and the fitted rate stops being the
quantity that reconstructs totals.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.models.single_week_ppr.architectures import GBMRegressor

# Snap coverage begins in 2013 (see population.SNAP_LABEL_MIN_SEASON).
OPPORTUNITY_MIN_SEASON = 2013
OPPORTUNITY_TARGET = "snap_count"

# Clip predicted snaps to a plausible range. Negative snaps are meaningless
# and a GBM extrapolating past the observed maximum adds nothing; the cap is
# well above any real single-game offensive snap count.
MIN_PREDICTED_SNAPS = 0.0
MAX_PREDICTED_SNAPS = 90.0


def opportunity_training_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting the snaps model: snap-era, snaps observed."""
    season = pd.to_numeric(train_df["season"], errors="coerce")
    snaps = pd.to_numeric(train_df[OPPORTUNITY_TARGET], errors="coerce")
    return train_df[(season >= OPPORTUNITY_MIN_SEASON) & snaps.notna() & (snaps > 0)]


def fit_opportunity_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    sample_weight: Optional[np.ndarray] = None,
) -> GBMRegressor:
    """E[snaps | pre-game information]. MSE, because the product needs a mean."""
    model = GBMRegressor(objective="regression")
    model.fit(train_df[list(feature_cols)],
              pd.to_numeric(train_df[OPPORTUNITY_TARGET], errors="coerce"),
              sample_weight=sample_weight)
    return model


def predict_opportunity(model: GBMRegressor, test_df: pd.DataFrame,
                        feature_cols: Sequence[str]) -> pd.Series:
    raw = model.predict(test_df[list(feature_cols)])
    return pd.Series(np.clip(raw, MIN_PREDICTED_SNAPS, MAX_PREDICTED_SNAPS),
                     index=test_df.index)


def fit_rate_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    recency_weight: Optional[np.ndarray] = None,
) -> GBMRegressor:
    """E[PPR per snap]. Snap-weighted, so the rate reconstructs totals."""
    snaps = pd.to_numeric(train_df[OPPORTUNITY_TARGET], errors="coerce")
    rate = pd.to_numeric(train_df["fantasy_points"], errors="coerce") / snaps
    weight = snaps.to_numpy(dtype=float)
    if recency_weight is not None:
        weight = weight * np.asarray(recency_weight, dtype=float)
    model = GBMRegressor(objective="regression")
    model.fit(train_df[list(feature_cols)], rate, sample_weight=weight)
    return model


def combine(expected_snaps: pd.Series, expected_rate: pd.Series) -> pd.Series:
    """E[PPR] = E[snaps] x E[PPR per snap].

    Not clipped at zero: PPR is genuinely negative sometimes (fumbles,
    interceptions), and flooring it would bias the mean upward.
    """
    return expected_snaps * expected_rate


def snap_bucket_slope(actual: pd.Series, predicted: pd.Series,
                      snaps: pd.Series, buckets, labels) -> float:
    """Slope of mean-predicted against mean-actual across snap buckets.

    1.0 is calibrated; the baseline sits at 0.52-0.69. This is the
    mechanism check -- an MAE win without movement here has not addressed
    the problem the split was built for.
    """
    b = pd.cut(pd.to_numeric(snaps, errors="coerce"), bins=buckets, labels=labels, right=True)
    frame = pd.DataFrame({"a": actual, "p": predicted, "b": b}).dropna(subset=["b"])
    grouped = frame.groupby("b", observed=True).agg(a=("a", "mean"), p=("p", "mean")).dropna()
    if len(grouped) < 2:
        return float("nan")
    return float(np.polyfit(grouped["a"], grouped["p"], 1)[0])

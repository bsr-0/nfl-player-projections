"""The opportunity layer: predict snaps, then production per snap.

These pin the choices that make the multiplicative form mean the thing it
claims (GAPS.md pre-registration, 2026-08-20).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.single_week_ppr.opportunity import (
    MAX_PREDICTED_SNAPS, OPPORTUNITY_MIN_SEASON, combine, fit_opportunity_model,
    fit_rate_model, opportunity_training_rows, predict_opportunity, snap_bucket_slope,
)


def _frame(n=400, seed=0):
    rng = np.random.RandomState(seed)
    snaps = rng.randint(1, 70, n).astype(float)
    return pd.DataFrame({
        "season": rng.choice([2018, 2019, 2020], n),
        "snap_count": snaps,
        "x": snaps + rng.normal(0, 3, n),           # informative
        "noise": rng.normal(0, 1, n),
        "fantasy_points": snaps * 0.2 + rng.normal(0, 1, n),
    })


def test_training_rows_require_snap_era_and_observed_snaps():
    df = pd.DataFrame({
        "season": [2010, 2018, 2018, 2018],
        "snap_count": [40.0, 40.0, 0.0, np.nan],
        "fantasy_points": [5.0, 5.0, 0.0, 0.0],
    })
    out = opportunity_training_rows(df)
    assert len(out) == 1
    assert out["season"].iloc[0] == 2018
    assert OPPORTUNITY_MIN_SEASON == 2013


def test_opportunity_model_recovers_a_real_snap_signal():
    df = _frame()
    m = fit_opportunity_model(df, ["x", "noise"])
    pred = predict_opportunity(m, df, ["x", "noise"])
    assert pred.corr(df["snap_count"]) > 0.9


def test_predicted_snaps_are_clipped_to_a_plausible_range():
    df = _frame()
    m = fit_opportunity_model(df, ["x", "noise"])
    wild = df.copy()
    wild["x"] = 10_000.0
    pred = predict_opportunity(m, wild, ["x", "noise"])
    assert (pred >= 0).all() and (pred <= MAX_PREDICTED_SNAPS).all()


def test_rate_model_is_snap_weighted_not_row_weighted():
    """A 1-snap 6-point fluke must not carry the same weight as a starter.

    Unweighted, the fitted rate is dragged toward the low-snap outliers;
    snap-weighting keeps it near the rate that reconstructs totals.
    """
    rng = np.random.RandomState(1)
    n = 300
    starters = pd.DataFrame({
        "season": 2018, "snap_count": rng.randint(50, 65, n).astype(float),
        "x": 1.0, "fantasy_points": 0.0})
    starters["fantasy_points"] = starters["snap_count"] * 0.2
    flukes = pd.DataFrame({
        "season": 2018, "snap_count": np.ones(n),
        "x": 1.0, "fantasy_points": 6.0})          # 6.0 PPR per snap
    df = pd.concat([starters, flukes], ignore_index=True)

    weighted = fit_rate_model(df, ["x"]).predict(df[["x"]]).mean()
    from src.models.single_week_ppr.architectures import GBMRegressor
    unweighted = GBMRegressor(objective="regression").fit(
        df[["x"]], df["fantasy_points"] / df["snap_count"]).predict(df[["x"]]).mean()
    assert weighted < unweighted, "snap weighting must pull the rate toward starters"
    assert weighted < 1.0


def test_combine_is_the_stated_product():
    snaps = pd.Series([10.0, 50.0])
    rate = pd.Series([0.2, 0.3])
    assert combine(snaps, rate).tolist() == [2.0, 15.0]


def test_combine_does_not_floor_negative_points():
    """PPR goes negative on fumbles and interceptions; clipping at zero
    would bias the mean upward."""
    assert combine(pd.Series([20.0]), pd.Series([-0.1])).iloc[0] == pytest.approx(-2.0)


def test_snap_bucket_slope_is_one_when_calibrated():
    n = 500
    rng = np.random.RandomState(2)
    snaps = pd.Series(rng.randint(1, 70, n).astype(float))
    actual = snaps * 0.2
    slope = snap_bucket_slope(actual, actual.copy(), snaps,
                              [0, 5, 15, 30, 50, np.inf],
                              ["0-5", "5-15", "15-30", "30-50", "50+"])
    assert slope == pytest.approx(1.0, abs=0.01)


def test_snap_bucket_slope_detects_compression():
    """A prediction shrunk toward its mean must score below 1.0 -- this is
    the diagnostic the whole experiment is judged on."""
    n = 500
    rng = np.random.RandomState(3)
    snaps = pd.Series(rng.randint(1, 70, n).astype(float))
    actual = snaps * 0.2
    compressed = actual.mean() + 0.5 * (actual - actual.mean())
    slope = snap_bucket_slope(actual, compressed, snaps,
                              [0, 5, 15, 30, 50, np.inf],
                              ["0-5", "5-15", "15-30", "30-50", "50+"])
    assert slope == pytest.approx(0.5, abs=0.05)

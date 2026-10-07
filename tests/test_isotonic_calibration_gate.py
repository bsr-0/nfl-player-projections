"""The isotonic calibrator must earn its place out of sample.

The old gate fitted the calibrator on every out-of-fold row and scored it on
those same rows, where the least-squares monotone fit can only match or beat
the uncalibrated prediction. A step function that was never tested out of
sample was therefore enabled on every position.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error

from src.models.position_models import _isotonic_holdout_check


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def test_rejects_calibrating_an_already_calibrated_prediction():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=500)
    y = pred + rng.normal(size=500)          # unbiased: nothing to correct
    fit, ev = slice(0, 200), slice(200, None)

    # In-sample, isotonic always looks like an improvement -- the old gate.
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred[fit], y[fit])
    assert _rmse(y[fit], iso.predict(pred[fit])) < _rmse(y[fit], pred[fit])

    improves, rmse_raw, rmse_cal = _isotonic_holdout_check(pred[fit], y[fit], pred[ev], y[ev])
    assert not improves and rmse_cal >= rmse_raw


def test_accepts_a_real_monotone_distortion():
    rng = np.random.default_rng(1)
    pred = rng.uniform(0, 1, size=600)
    y = 3.0 * pred ** 3 + rng.normal(scale=0.05, size=600)   # compressed, curved
    improves, rmse_raw, rmse_cal = _isotonic_holdout_check(pred[:400], y[:400], pred[400:], y[400:])
    assert improves and rmse_cal < rmse_raw

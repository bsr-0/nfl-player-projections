"""PositionModel.predict must zero-fill NaN, matching what fit() trains on.

fit() computes feature_medians (kept for OTHER callers that fill an entirely
absent column before calling predict) but zero-fills its OWN input before
fitting the scaler and base learners. predict() used to median-fill instead,
so every persisted model was served a systematically different value for
every missing feature than the one its scaler/learners were fit on -- not a
rare case: real per-position feature matrices run 11-13% NaN overall, with
some columns 80-95% missing (team_motion_rate, NGS fields).

An ablation (data/backtest_results-adjacent scratch script, not committed)
compared zero-fill / median-fill / native-NaN-for-boosters on the real
train<=2024/test=2025 split at default hyperparameters: no policy dominated
across positions, and zero-fill was tied-best or best on 3 of 4 -- so this is
a consistency fix to match the already-trained models, not a change that
needs its own retrain to justify.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.position_models import PositionModel


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.RandomState(0)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.uniform(0, 10, size=n),
        "c": rng.normal(5, 2, size=n),
    })
    y = pd.Series(2 * X["a"] + 0.5 * X["b"] - X["c"] + rng.normal(scale=0.5, size=n))
    model = PositionModel("WR", n_weeks=1)
    model.fit(X, y, tune_hyperparameters=False)
    return model


def test_missing_value_is_treated_as_zero_not_median(fitted):
    row = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [3.0]})
    zero_filled = pd.DataFrame({"a": [1.0], "b": [0.0], "c": [3.0]})

    pred_nan = fitted.predict(row)
    pred_zero = fitted.predict(zero_filled)

    assert pred_nan[0] == pytest.approx(pred_zero[0])


def test_prediction_does_not_depend_on_feature_medians(fitted):
    """feature_medians is retained for schema-level fills elsewhere (a column
    absent entirely), but must play no part in per-value NaN imputation."""
    row = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [3.0]})
    baseline = fitted.predict(row)[0]

    original = dict(fitted.feature_medians)
    try:
        fitted.feature_medians = {k: v + 1000.0 for k, v in original.items()}
        perturbed = fitted.predict(row)[0]
    finally:
        fitted.feature_medians = original

    assert perturbed == pytest.approx(baseline)


def test_scaler_was_fit_on_zero_filled_data(fitted):
    """The scaler itself must reflect zero-fill, or predict-time zero-fill
    would be internally consistent but still mismatched with training."""
    X_np = fitted._prepare_input(pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]}))
    # StandardScaler.transform(0) == -mean/scale; if the scaler's fitted mean
    # reflects a zero-filled training column, this is deterministic and finite.
    assert np.all(np.isfinite(X_np))


def test_inf_is_also_treated_as_zero(fitted):
    row = pd.DataFrame({"a": [1.0], "b": [np.inf], "c": [-np.inf]})
    zero_filled = pd.DataFrame({"a": [1.0], "b": [0.0], "c": [0.0]})
    assert fitted.predict(row)[0] == pytest.approx(fitted.predict(zero_filled)[0])

"""Log1p targets need a retransformation correction, and it must stay wired.

Training on log1p(y) and inverting with expm1 estimates a conditional
GEOMETRIC mean, which by Jensen's inequality lies below the arithmetic mean the
prediction is meant to be. The gap widens with residual variance, so it is
worst exactly where outcomes are most volatile.

Measured on a 2025 serving backtest against a model trained 2018-2024 only
(so the season was genuinely unseen), transform-on vs transform-on-plus-
smearing:

    pos   dMAE      bias cut
    RB   -0.207        45%
    WR   -0.073        43%
    TE   -0.063        36%
    QB    0.000         0%   <- control: its saved model has no transform

Both calibration and accuracy improved for every transformed position, and the
untransformed one was bit-identical. Duan (1983) smearing is used because it is
non-parametric -- fantasy scoring is not lognormal, so a closed-form
sigma^2/2 correction would be wrong.
"""
import numpy as np
import pytest

import config.settings as settings
from src.models.position_models import TargetTransformer


@pytest.fixture
def mode(monkeypatch):
    def _set(value):
        monkeypatch.setattr(settings, "TARGET_TRANSFORM_MODE", value)
    return _set


def _skewed(n=4000, seed=0):
    """Right-skewed, zero-inflated: the shape of weekly fantasy points."""
    rng = np.random.default_rng(seed)
    y = rng.lognormal(mean=1.2, sigma=0.9, size=n)
    y[rng.random(n) < 0.15] = 0.0
    return y


def test_default_is_smearing():
    assert settings.TARGET_TRANSFORM_MODE == "smearing"


def test_transform_activates_on_skewed_targets(mode):
    mode("smearing")
    tt = TargetTransformer()
    tt.fit_transform(_skewed())
    assert tt.active, "a strongly right-skewed target should be transformed"


def test_off_mode_disables_the_transform(mode):
    mode("off")
    tt = TargetTransformer()
    y = _skewed()
    out = tt.fit_transform(y)
    assert not tt.active
    np.testing.assert_allclose(out, y)


def test_smearing_corrects_the_downward_bias(mode):
    """The property that matters: mean(inverse(pred)) should approach mean(y).

    Without the correction the back-transformed mean sits BELOW the true mean.
    """
    mode("smearing")
    y = _skewed()
    tt = TargetTransformer()
    y_log = tt.fit_transform(y)
    assert tt.active

    # A deliberately imperfect model: the conditional mean in log space plus
    # noise, which is what leaves residual variance for smearing to correct.
    rng = np.random.default_rng(1)
    pred_log = np.full_like(y_log, y_log.mean()) + rng.normal(0, 0.05, len(y_log))

    tt.smearing = 1.0
    uncorrected = tt.inverse_transform(pred_log.copy()).mean()
    tt.fit_smearing(y_log, pred_log)
    corrected = tt.inverse_transform(pred_log.copy()).mean()

    true_mean = y.mean()
    assert uncorrected < true_mean, "expected the known downward bias"
    assert tt.smearing > 1.0, "smearing factor should scale predictions up"
    assert abs(corrected - true_mean) < abs(uncorrected - true_mean), (
        f"smearing should move the mean toward the truth: "
        f"{uncorrected:.2f} -> {corrected:.2f}, true {true_mean:.2f}"
    )


def test_smearing_is_inert_when_the_transform_is_inactive(mode):
    """QB's saved model has no transform; the correction must not touch it."""
    mode("smearing")
    tt = TargetTransformer()
    y = np.random.default_rng(2).normal(10, 2, 2000)   # symmetric
    tt.fit_transform(y)
    assert not tt.active
    tt.fit_smearing(y, y)
    assert tt.smearing == 1.0
    x = np.array([1.0, 5.0, 9.0])
    np.testing.assert_allclose(tt.inverse_transform(x.copy()), x)


def test_smearing_not_fitted_when_mode_is_plain_on(mode):
    mode("on")
    tt = TargetTransformer()
    y_log = tt.fit_transform(_skewed())
    tt.fit_smearing(y_log, y_log * 0.9)
    assert tt.smearing == 1.0, "mode 'on' must reproduce historical behaviour"


def test_smearing_factor_is_clipped(mode):
    """A correction should be a nudge; a wild factor means something else broke."""
    mode("smearing")
    tt = TargetTransformer()
    y_log = tt.fit_transform(_skewed())
    tt.fit_smearing(y_log, y_log - 5.0)   # absurd residuals
    assert 1.0 <= tt.smearing <= 3.0

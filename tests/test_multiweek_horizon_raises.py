"""MultiWeekModel.predict must fail loudly on an untrained horizon.

It used to silently substitute the closest trained model with no scaling
(a request for n_weeks=18 -- never trained; see config.settings.
TRAINING_HORIZONS -- got served the 4-week model's raw output under the
"18w" label). Callers then divided that by 18 for a per-game rate, so the
result was wrong by roughly the ratio of the two horizons. Raising instead
lets callers (scripts/generate_app_data.py) only request horizons that
actually have a model.
"""
import pandas as pd
import pytest

from src.models.position_models import MultiWeekModel


class _FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


@pytest.fixture
def multi_week_model():
    m = MultiWeekModel("WR")
    # Mirror what fit() does: representative models cover their whole
    # horizon group (short=1-3, medium=4-8); "long" (9-18) is untrained.
    short = _FakeModel(1.0)
    medium = _FakeModel(4.0)
    for week in m.horizon_groups["short"]:
        m.models[week] = short
    for week in m.horizon_groups["medium"]:
        m.models[week] = medium
    return m


def test_trained_horizon_predicts(multi_week_model):
    X = pd.DataFrame({"a": [1, 2, 3]})
    assert list(multi_week_model.predict(X, n_weeks=1)) == [1.0, 1.0, 1.0]
    assert list(multi_week_model.predict(X, n_weeks=4)) == [4.0, 4.0, 4.0]
    # Weeks 5-8 share the medium representative model by design.
    assert list(multi_week_model.predict(X, n_weeks=8)) == [4.0, 4.0, 4.0]


def test_untrained_long_horizon_raises(multi_week_model):
    X = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="n_weeks=18"):
        multi_week_model.predict(X, n_weeks=18)
    with pytest.raises(ValueError, match="n_weeks=9"):
        multi_week_model.predict(X, n_weeks=9)

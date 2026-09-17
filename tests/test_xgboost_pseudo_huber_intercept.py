"""XGBoost's pseudo-Huber member must actually fit targets far from 0.5.

xgboost 2.0.x starts reg:pseudohubererror boosting from base_score=0.5 and
never estimates an intercept. The loss's hessian vanishes as slope^3/|r|^3,
so for a target centred around 46 (the QB utilization score) no split ever
clears min_child_weight and the model collapses to a single constant. The
2026-09-16 retrain showed this as 100 identical Optuna trials at MSE 9828.

Synthetic data at the real target scale; no DB.
"""
import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from config.settings import HUBER_DELTA
from src.models.position_models import PositionModel, _pseudo_huber_base_score


def _util_scale_data(seed=0, n=3000, d=40):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 46 + 8 * X[:, 0] + 4 * X[:, 1] + rng.normal(scale=12, size=n)
    return X, y


def test_base_score_is_target_mean_weighted_or_not():
    y = np.array([10.0, 20.0, 60.0])
    assert _pseudo_huber_base_score(y) == pytest.approx(30.0)
    assert _pseudo_huber_base_score(y, np.array([1.0, 1.0, 2.0])) == pytest.approx(37.5)
    assert _pseudo_huber_base_score(y, np.array([1.0])) == pytest.approx(30.0)  # length mismatch ignored


def test_default_intercept_collapses_at_utilization_scale():
    """Documents the failure mode so the fix below is demonstrably load-bearing."""
    X, y = _util_scale_data()
    m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=HUBER_DELTA,
                         n_estimators=100, max_depth=4, learning_rate=0.05,
                         min_child_weight=5, tree_method="hist", n_jobs=1).fit(X, y)
    assert m.predict(X).std() < 0.01, "xgboost now estimates an intercept; test is stale"


def test_train_xgboost_fits_utilization_scale_target():
    X, y = _util_scale_data()
    cut = 2200
    model = PositionModel("QB")
    model.best_params = {"xgboost": dict(n_estimators=100, max_depth=4, learning_rate=0.05,
                                         min_child_weight=5, colsample_bytree=0.8)}
    fitted = model._train_xgboost(X[:cut], y[:cut], X[cut:], y[cut:])
    pred = fitted.predict(X[cut:])
    baseline = np.mean((y[cut:] - y[:cut].mean()) ** 2)
    mse = np.mean((y[cut:] - pred) ** 2)
    assert pred.std() > 1.0, "still predicting a constant"
    assert mse < 0.8 * baseline, f"XGB not beating the mean: {mse:.1f} vs {baseline:.1f}"


def test_tuning_objective_varies_across_trials(monkeypatch):
    """The real symptom: every trial returned the same value to 12 decimals."""
    pytest.importorskip("optuna")
    X, y = _util_scale_data()
    model = PositionModel("QB")
    monkeypatch.setattr(PositionModel, "_subsample_for_tuning",
                        staticmethod(lambda X, y, seasons=None, max_samples=8000: (X, y, seasons)))
    seen = []
    orig = xgb.XGBRegressor.fit

    def spy(self, Xf, yf, *a, **k):
        r = orig(self, Xf, yf, *a, **k)
        seen.append(float(np.mean((yf - self.predict(Xf)) ** 2)))
        return r
    monkeypatch.setattr(xgb.XGBRegressor, "fit", spy)
    model._tune_xgboost(X, y, n_trials=3, seasons=np.repeat(np.arange(2018, 2024), 500))
    assert len(set(round(v, 6) for v in seen)) > 1, f"all fits identical: {seen[:3]}"
    assert max(seen) < np.var(y), "fits are worse than predicting the mean"

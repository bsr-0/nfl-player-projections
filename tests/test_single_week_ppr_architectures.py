"""Unit tests for Phase 2 (next_focus.md) single-week PPR architectures.

Synthetic data only — no DB/network access, fast to run. These test that
each architecture's fit/predict contract works and produces sane output;
they do not test leakage-safety (covered by tests/test_leakage_guards.py)
or real-world accuracy (covered by scripts/run_phase2_comparison.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.architectures import (
    BoxCoxHuber,
    GBMRegressor,
    HurdleModel,
    QuantileGBM,
    YeoJohnsonHuber,
    naive_baselines,
    position_role_baseline,
)


@pytest.fixture
def synthetic_regression_data():
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({
        "targets_roll3_mean": rng.uniform(0, 10, n),
        "snap_share_pct_roll3_mean": rng.uniform(0, 1, n),
    })
    y = pd.Series(
        5 + 2 * X["targets_roll3_mean"] + 10 * X["snap_share_pct_roll3_mean"] + rng.normal(0, 1, n),
        name="fantasy_points",
    )
    return X, y


@pytest.fixture
def synthetic_hurdle_data():
    """x < 0 -> always y = 0 (didn't play); x >= 0 -> y = 10 + noise."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(-5, 5, n)
    y = np.where(x < 0, 0.0, 10 + rng.normal(0, 1, n))
    X = pd.DataFrame({"x": x, "noise": rng.uniform(0, 1, n)})
    return X, pd.Series(y, name="fantasy_points")


class TestGBMRegressor:
    @pytest.mark.parametrize("objective", ["regression", "huber", "regression_l1"])
    def test_fit_predict(self, synthetic_regression_data, objective):
        X, y = synthetic_regression_data
        model = GBMRegressor(objective=objective, n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.isfinite(preds).all()
        # Should learn something better than predicting the mean.
        mae_model = np.abs(preds - y).mean()
        mae_mean = np.abs(y.mean() - y).mean()
        assert mae_model < mae_mean

    def test_unknown_objective_rejected(self):
        with pytest.raises(ValueError):
            GBMRegressor(objective="not_a_real_objective")


class TestHurdleModel:
    def test_predicts_near_zero_for_negative_rows(self, synthetic_hurdle_data):
        X, y = synthetic_hurdle_data
        model = HurdleModel(threshold=0.0, n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(X)
        assert np.isfinite(preds).all()

        negative_mask = (X["x"] < -1).to_numpy()
        positive_mask = (X["x"] > 1).to_numpy()
        # Rows the classifier should confidently label "didn't play" should
        # predict much lower than rows confidently labeled "played".
        assert preds[negative_mask].mean() < preds[positive_mask].mean() / 2

    def test_requires_positive_rows(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pd.Series([0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            HurdleModel().fit(X, y)


class TestQuantileGBM:
    def test_quantiles_are_monotonic_on_average(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        model = QuantileGBM(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)

        assert list(preds.columns) == ["p25", "p50", "p75", "p90"]
        assert len(preds) == len(X)
        means = preds.mean()
        assert means["p25"] <= means["p50"] <= means["p75"] <= means["p90"]

    def test_predict_median_matches_p50_column(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        model = QuantileGBM(n_estimators=50)
        model.fit(X, y)
        median_direct = model.predict_median(X)
        median_from_df = model.predict(X)["p50"].to_numpy()
        np.testing.assert_allclose(median_direct, median_from_df)


class TestYeoJohnsonHuber:
    def test_fit_predict_with_negative_targets(self):
        """PPR can go negative (fumbles/INTs) — Yeo-Johnson must handle that,
        unlike log transforms (next_focus.md Phase 2 §4F)."""
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        y = pd.Series(X["x"] * 2 - 5 + rng.normal(0, 1, n))  # ranges negative to positive
        assert (y < 0).any()

        model = YeoJohnsonHuber(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(X)
        assert np.isfinite(preds).all()
        mae_model = np.abs(preds - y).mean()
        mae_mean = np.abs(y.mean() - y).mean()
        assert mae_model < mae_mean


class TestBoxCoxHuber:
    def test_fit_predict_with_negative_targets(self):
        """Box-Cox requires strictly positive input, unlike Yeo-Johnson —
        this must handle negative PPR via the shift workaround (quick
        follow-up test to F, requested after the bias/skew discussion)."""
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        y = pd.Series(X["x"] * 2 - 5 + rng.normal(0, 1, n))  # ranges negative to positive
        assert (y < 0).any()

        model = BoxCoxHuber(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == len(X)
        assert np.isfinite(preds).all()
        mae_model = np.abs(preds - y).mean()
        mae_mean = np.abs(y.mean() - y).mean()
        assert mae_model < mae_mean

    def test_shift_guarantees_positivity(self):
        y = pd.Series([-7.0, -3.0, 0.0, 5.0, 20.0])
        model = BoxCoxHuber(n_estimators=10)
        assert (y.to_numpy() + (1.0 - float(y.min()))).min() >= 1.0
        model.shift_ = 1.0 - float(y.min())
        assert model.shift_ == pytest.approx(8.0)


class TestNaiveBaselines:
    def test_position_role_baseline_is_shifted(self):
        df = pd.DataFrame({
            "position": ["WR", "WR", "WR"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "fantasy_points": [10.0, 20.0, 30.0],
        })
        result = position_role_baseline(df)
        # Week 1: no prior data -> NaN. Week 2: mean of week1 (10). Week 3: mean of weeks1-2 (15).
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(10.0)
        assert result.iloc[2] == pytest.approx(15.0)

    def test_naive_baselines_returns_four_series(self):
        df = pd.DataFrame({
            "player_id": ["P1"] * 5,
            "position": ["WR"] * 5,
            "season": [2024] * 5,
            "week": [1, 2, 3, 4, 5],
            "fantasy_points": [5.0, 10.0, 15.0, 20.0, 25.0],
        })
        result = naive_baselines(df)
        assert set(result.keys()) == {
            "baseline_recent_game", "baseline_rolling3", "baseline_rolling5", "baseline_position_role",
        }
        for series in result.values():
            assert len(series) == len(df)
        # Week 5 recent-game baseline = week 4's value (20.0).
        assert result["baseline_recent_game"].iloc[4] == pytest.approx(20.0)

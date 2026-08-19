"""Fold-safety of the variant-B snap imputation.

The imputation statistic must come from the training fold alone. A median
computed over full history would leak future seasons into earlier folds --
small in effect, fatal to the experiment's credibility.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import ts_backtester as bt


@pytest.fixture(autouse=True)
def restore_mode():
    original = bt.SNAP_IMPUTATION_MODE
    yield
    bt.SNAP_IMPUTATION_MODE = original


def _frames():
    train = pd.DataFrame({
        "season": [2016, 2016, 2017, 2017],
        bt.SNAP_ROLL3_COL: [10.0, 20.0, 30.0, np.nan],
        bt.SNAP_KNOWN_COL: [1.0, 1.0, 1.0, 0.0],
    })
    test = pd.DataFrame({
        "season": [2018, 2018],
        # a wildly different distribution -- if it influences the fill value,
        # test data has leaked into the statistic
        bt.SNAP_ROLL3_COL: [np.nan, 900.0],
        bt.SNAP_KNOWN_COL: [0.0, 1.0],
    })
    return train, test


def test_mode_zero_is_a_passthrough(mod=None):
    """Variant A must be the untouched production path, not a re-derivation."""
    bt.SNAP_IMPUTATION_MODE = "zero"
    train, test = _frames()
    out_tr, out_te, cols = bt.apply_snap_imputation(train, test, ["x"])

    assert out_tr[bt.SNAP_ROLL3_COL].isna().sum() == 1
    assert out_te[bt.SNAP_ROLL3_COL].isna().sum() == 1
    assert cols == ["x"]


def test_fill_value_ignores_test_rows_entirely():
    """The decisive leakage check: a 900.0 in test must not move the fill."""
    bt.SNAP_IMPUTATION_MODE = "median"
    train, test = _frames()
    _, out_te, _ = bt.apply_snap_imputation(train, test, ["x"])

    filled = out_te[bt.SNAP_ROLL3_COL].iloc[0]
    train_median = np.median([10.0, 20.0, 30.0])
    assert filled == pytest.approx(train_median)
    assert filled != pytest.approx(900.0)


def test_fill_is_unchanged_when_test_values_change():
    """Stronger form: perturbing test data must not alter any fill value."""
    bt.SNAP_IMPUTATION_MODE = "median"
    train, test_a = _frames()
    _, out_a, _ = bt.apply_snap_imputation(train, test_a, ["x"])

    train, test_b = _frames()
    test_b[bt.SNAP_ROLL3_COL] = [np.nan, -5000.0]
    _, out_b, _ = bt.apply_snap_imputation(train, test_b, ["x"])

    assert out_a[bt.SNAP_ROLL3_COL].iloc[0] == out_b[bt.SNAP_ROLL3_COL].iloc[0]


def test_era_median_is_used_when_the_era_exists_in_train():
    bt.SNAP_IMPUTATION_MODE = "median"
    train = pd.DataFrame({
        "season": [2016, 2016, 2019, 2019],
        bt.SNAP_ROLL3_COL: [10.0, 10.0, 80.0, 80.0],
        bt.SNAP_KNOWN_COL: [1.0, 1.0, 1.0, 1.0],
    })
    test = pd.DataFrame({
        "season": [2017, 2020],
        bt.SNAP_ROLL3_COL: [np.nan, np.nan],
        bt.SNAP_KNOWN_COL: [0.0, 0.0],
    })
    _, out_te, _ = bt.apply_snap_imputation(train, test, ["x"])

    assert out_te[bt.SNAP_ROLL3_COL].iloc[0] == pytest.approx(10.0)  # pre2018
    assert out_te[bt.SNAP_ROLL3_COL].iloc[1] == pytest.approx(80.0)  # post2018


def test_era_boundary_is_2018():
    assert bt._era(pd.Series([2017]))[0] == "pre2018"
    assert bt._era(pd.Series([2018]))[0] == "post2018"


def test_missing_era_falls_back_to_overall_train_median():
    """A fold whose training data has no post-2018 rows still must fill."""
    bt.SNAP_IMPUTATION_MODE = "median"
    train = pd.DataFrame({
        "season": [2015, 2016],
        bt.SNAP_ROLL3_COL: [40.0, 60.0],
        bt.SNAP_KNOWN_COL: [1.0, 1.0],
    })
    test = pd.DataFrame({
        "season": [2019], bt.SNAP_ROLL3_COL: [np.nan], bt.SNAP_KNOWN_COL: [0.0],
    })
    _, out_te, _ = bt.apply_snap_imputation(train, test, ["x"])

    assert out_te[bt.SNAP_ROLL3_COL].iloc[0] == pytest.approx(50.0)
    assert out_te[bt.SNAP_ROLL3_COL].notna().all()


def test_known_indicator_joins_the_feature_list_only_in_variant_b():
    train, test = _frames()

    bt.SNAP_IMPUTATION_MODE = "zero"
    _, _, cols_a = bt.apply_snap_imputation(train, test, ["x"])
    bt.SNAP_IMPUTATION_MODE = "median"
    _, _, cols_b = bt.apply_snap_imputation(train, test, ["x"])

    assert bt.SNAP_KNOWN_COL not in cols_a
    assert bt.SNAP_KNOWN_COL in cols_b


def test_no_nan_survives_variant_b():
    """Otherwise the blanket fillna(0) downstream reintroduces the very zero
    this variant exists to avoid."""
    bt.SNAP_IMPUTATION_MODE = "median"
    train, test = _frames()
    out_tr, out_te, _ = bt.apply_snap_imputation(train, test, ["x"])

    assert out_tr[bt.SNAP_ROLL3_COL].notna().all()
    assert out_te[bt.SNAP_ROLL3_COL].notna().all()

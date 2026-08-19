"""The snap-imputation artifact boundary.

The statistical calculation is the easy part. These test the thing that
actually goes wrong in production: an artifact fitted on one training set
being silently reused against another, or an imputation value that turns
out to depend on the inference population.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.features.utilization_score import (
    SNAP_ROLL3_COL,
    apply_snap_imputation,
    fit_snap_imputation,
    load_snap_imputation,
    save_snap_imputation,
    snap_era,
    validate_snap_imputation_meta,
)


def _rows(seasons, positions, values):
    return pd.DataFrame({
        "season": seasons, "position": positions, SNAP_ROLL3_COL: values,
    })


# --- fitting -------------------------------------------------------------

def test_fit_is_position_by_era():
    train = _rows([2016, 2016, 2019, 2019], ["WR"] * 4, [10.0, 20.0, 80.0, 90.0])
    values = fit_snap_imputation(train)

    assert values[("WR", "pre2018")] == pytest.approx(15.0)
    assert values[("WR", "post2018")] == pytest.approx(85.0)


def test_positions_do_not_pool():
    train = _rows([2016] * 4, ["WR", "WR", "RB", "RB"], [80.0, 90.0, 10.0, 20.0])
    values = fit_snap_imputation(train)

    assert values[("WR", "pre2018")] == pytest.approx(85.0)
    assert values[("RB", "pre2018")] == pytest.approx(15.0)


def test_era_boundary_is_2018():
    assert snap_era(pd.Series([2017])).iloc[0] == "pre2018"
    assert snap_era(pd.Series([2018])).iloc[0] == "post2018"


def test_unknown_rows_do_not_contribute_to_the_median():
    train = _rows([2016] * 3, ["WR"] * 3, [10.0, 20.0, np.nan])
    assert fit_snap_imputation(train)[("WR", "pre2018")] == pytest.approx(15.0)


# --- artifact round trip -------------------------------------------------

def test_roundtrip_preserves_values(tmp_path):
    train = _rows([2016, 2019], ["WR", "WR"], [10.0, 80.0])
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path, metadata={"train_seasons": [2016, 2019]})

    values, meta = load_snap_imputation(path, return_meta=True)
    assert values[("WR", "pre2018")] == pytest.approx(10.0)
    assert meta["train_seasons"] == [2016, 2019]


def test_missing_artifact_loads_empty_rather_than_guessing(tmp_path):
    assert load_snap_imputation(tmp_path / "nope.json") == {}


def test_empty_values_leave_the_frame_untouched():
    """A missing artifact must fail visibly upstream, not quietly impute."""
    df = _rows([2016], ["WR"], [np.nan])
    out = apply_snap_imputation(df, {})
    assert out[SNAP_ROLL3_COL].isna().all()


# --- the failure mode this whole exercise exists to prevent ---------------

def test_imputation_values_are_immune_to_test_rows(tmp_path):
    """Fit on season A, persist, then add arbitrary test-season rows and
    reload. The artifact must be unchanged."""
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path,
                         metadata={"train_seasons": [2016]})
    before = load_snap_imputation(path)

    # a wildly different test population appears
    test = _rows([2024] * 3, ["WR"] * 3, [900.0, 950.0, np.nan])
    apply_snap_imputation(test, before)
    after = load_snap_imputation(path)

    assert before == after
    assert after[("WR", "pre2018")] == pytest.approx(15.0)


def test_applying_to_test_rows_uses_persisted_values_only():
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    values = fit_snap_imputation(train)
    test = _rows([2016, 2016], ["WR", "WR"], [np.nan, 900.0])

    out = apply_snap_imputation(test, values)
    assert out[SNAP_ROLL3_COL].iloc[0] == pytest.approx(15.0)


def test_artifact_from_different_seasons_is_refused():
    """Season set A artifact against a pipeline expecting A+B must fail."""
    meta = {"train_seasons": [2016, 2017]}
    assert validate_snap_imputation_meta(meta, [2016, 2017]) is True
    assert validate_snap_imputation_meta(meta, [2016, 2017, 2018]) is False


def test_absent_or_empty_metadata_is_refused():
    assert validate_snap_imputation_meta({}, [2016]) is False
    assert validate_snap_imputation_meta({"train_seasons": []}, [2016]) is False


# --- fallbacks -----------------------------------------------------------

def test_missing_era_falls_back_to_the_position_median():
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    values = fit_snap_imputation(train)
    test = _rows([2024], ["WR"], [np.nan])          # no post2018 WR in train

    assert apply_snap_imputation(test, values)[SNAP_ROLL3_COL].iloc[0] == pytest.approx(15.0)


def test_unseen_position_falls_back_to_global():
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    values = fit_snap_imputation(train)
    test = _rows([2016], ["TE"], [np.nan])

    assert apply_snap_imputation(test, values)[SNAP_ROLL3_COL].iloc[0] == pytest.approx(15.0)


def test_known_values_are_never_overwritten():
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    values = fit_snap_imputation(train)
    test = _rows([2016, 2016], ["WR", "WR"], [77.0, np.nan])

    out = apply_snap_imputation(test, values)
    assert out[SNAP_ROLL3_COL].iloc[0] == pytest.approx(77.0)
    assert out[SNAP_ROLL3_COL].iloc[1] == pytest.approx(15.0)


def test_no_nan_survives_when_an_artifact_exists():
    """Otherwise the blanket fillna(0) downstream reintroduces the zero."""
    train = _rows([2016, 2016], ["WR", "WR"], [10.0, 20.0])
    values = fit_snap_imputation(train)
    test = _rows([2016, 2024], ["WR", "TE"], [np.nan, np.nan])

    assert apply_snap_imputation(test, values)[SNAP_ROLL3_COL].notna().all()


# --- one transformation, two callers -------------------------------------

def test_backtest_and_production_paths_agree(tmp_path):
    """The architectural guarantee: given the same training data, the
    backtester's fold-local fit and production's persisted artifact must
    produce identical imputed values. They previously had separate
    implementations, which is how a divergence goes unnoticed."""
    from src.evaluation import ts_backtester as bt

    train = _rows([2016, 2016, 2019, 2019], ["WR"] * 4, [10.0, 20.0, 80.0, 90.0])
    test = _rows([2016, 2019], ["WR", "WR"], [np.nan, np.nan])

    # production: fit -> persist -> reload -> apply
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path,
                         metadata={"train_seasons": [2016, 2019]})
    production = apply_snap_imputation(test, load_snap_imputation(path))

    # backtest: fit in-fold -> apply
    original = bt.SNAP_IMPUTATION_MODE
    try:
        bt.SNAP_IMPUTATION_MODE = "median"
        _, backtest, _ = bt.apply_snap_imputation(train, test, ["x"])
    finally:
        bt.SNAP_IMPUTATION_MODE = original

    pd.testing.assert_series_equal(
        production[SNAP_ROLL3_COL], backtest[SNAP_ROLL3_COL], check_names=False)


def test_persisted_values_survive_a_reload_unchanged(tmp_path):
    """Model artifacts must be loadable independently of the training run."""
    train = _rows([2016, 2019], ["WR", "TE"], [10.0, 80.0])
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path,
                         metadata={"train_seasons": [2016, 2019]})

    first = load_snap_imputation(path)
    raw = json.loads(path.read_text())          # a fresh process would see this
    second = load_snap_imputation(path)

    assert first == second
    assert "__meta__" in raw
    assert any("|" in k for k in raw if k != "__meta__")

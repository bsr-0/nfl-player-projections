"""The 2015/2016 "Probable" rule change must not cross a training window silently.

`Probable` scores 0.85 and was abolished after 2015, so injury_score reads
mildly unavailable for ~2,700 player-weeks a season before 2016 and for none
after. Deliberately not remapped -- see the docstring on the guard.
"""
import warnings

import pandas as pd
import pytest

from src.features.feature_engineering import (
    PROBABLE_ABOLISHED_AFTER_SEASON,
    _warn_on_probable_era_span,
)


def _injuries(rows):
    return pd.DataFrame(rows, columns=["season", "report_status"])


def test_window_spanning_the_boundary_warns():
    df = _injuries([(2015, "Probable"), (2015, "Out"), (2019, "Questionable")])

    with pytest.warns(RuntimeWarning, match="Probable"):
        _warn_on_probable_era_span(df)


def test_modern_only_window_is_silent():
    df = _injuries([(2019, "Questionable"), (2021, "Out")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_on_probable_era_span(df)


def test_legacy_only_window_is_silent():
    """Entirely pre-2016 is internally consistent -- every season uses the
    same reporting rules."""
    df = _injuries([(2013, "Probable"), (2014, "Out")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_on_probable_era_span(df)


def test_spanning_window_without_probable_rows_is_silent():
    """The artifact is the status, not the year range."""
    df = _injuries([(2015, "Out"), (2019, "Questionable")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_on_probable_era_span(df)


def test_boundary_is_2015():
    assert PROBABLE_ABOLISHED_AFTER_SEASON == 2015

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_on_probable_era_span(_injuries([(2016, "Questionable"), (2019, "Out")]))


def test_warning_quantifies_the_exposure():
    df = _injuries([(2014, "Probable")] * 7 + [(2019, "Out")])

    with pytest.warns(RuntimeWarning, match=r"7 Probable rows"):
        _warn_on_probable_era_span(df)


def test_malformed_input_does_not_raise():
    for df in (pd.DataFrame(), _injuries([]), pd.DataFrame({"season": [2014]})):
        _warn_on_probable_era_span(df)


def test_default_training_window_crosses_the_boundary_with_warning_intact():
    """2026-09-16: reconsidered, not reverted.

    TRAINING_START_YEAR_DEFAULT moved to MIN_HISTORICAL_YEAR (2006) after a
    real backtest -- which already had injury_score in the feature set --
    showed a net improvement from the wider window despite this exact
    comparability gap. The empirical result already reflects whatever cost
    the gap imposes; it wasn't enough to erase the gain. What must NOT
    happen is losing the signal silently: this used to assert the default
    stayed clear of the boundary, which stopped being true when the window
    changed. The guard itself (_warn_on_probable_era_span) is not
    weakened -- confirm it still fires for the real default window,
    so the tradeoff stays visible to whoever revisits this next rather
    than disappearing along with the test that used to assert against it.
    """
    from config.settings import MIN_HISTORICAL_YEAR, TRAINING_START_YEAR_DEFAULT

    assert TRAINING_START_YEAR_DEFAULT <= PROBABLE_ABOLISHED_AFTER_SEASON, (
        "this test's premise is that the default crosses the boundary -- "
        "if it no longer does, restore test_default_training_window_avoids_the_boundary instead"
    )
    assert TRAINING_START_YEAR_DEFAULT == MIN_HISTORICAL_YEAR

    with pytest.warns(RuntimeWarning, match="Probable"):
        _warn_on_probable_era_span(_injuries(
            [(TRAINING_START_YEAR_DEFAULT, "Probable"), (2019, "Questionable")]
        ))

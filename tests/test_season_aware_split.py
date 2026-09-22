"""SeasonAwareTimeSeriesSplit must yield exactly the folds it promises, and
the tuning subsample must carry its own rows' season labels.

The 2026-09-09 retrain died in WR Ridge tuning with sklearn's
"cv.split and cv.get_n_splits returned inconsistent results. Expected 3
splits, got 2": with 4 distinct seasons, n_splits=3 and gap_seasons=1 the
first fold had no training seasons and was silently skipped. WR reached
that count because the tuners took `seasons[-len(X_tune):]` -- the last N
rows of the full frame -- as the labels for a subsample drawn from
elsewhere.
"""
import numpy as np
import pytest
from sklearn.linear_model import RidgeCV

from src.models.position_models import PositionModel, SeasonAwareTimeSeriesSplit


def _data(n_seasons, per_season=200, seed=0):
    rng = np.random.RandomState(seed)
    seasons = np.repeat(np.arange(2018, 2018 + n_seasons), per_season)
    X = rng.normal(size=(len(seasons), 4))
    y = X[:, 0] + rng.normal(size=len(seasons))
    return X, y, seasons


@pytest.mark.parametrize("n_seasons", [2, 3, 4, 5, 8])
def test_get_n_splits_matches_yielded_folds(n_seasons):
    X, y, seasons = _data(n_seasons)
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1)
    folds = list(cv.split(X, y))
    assert len(folds) == cv.get_n_splits(X) == 3


def test_four_seasons_with_gap_no_longer_breaks_ridgecv():
    X, y, seasons = _data(4)
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1)
    RidgeCV(alphas=np.logspace(-2, 2, 5), cv=cv).fit(X, y)   # raised ValueError before


def test_season_aware_folds_are_temporal_with_purge_gap():
    X, y, seasons = _data(6)
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1)
    for train_idx, test_idx in cv.split(X, y):
        test_season = np.unique(seasons[test_idx])
        assert len(test_season) == 1
        # gap of one full season between the last training season and the test season
        assert seasons[train_idx].max() <= test_season[0] - 2


def test_tuning_subsample_keeps_season_labels_aligned():
    seasons = np.repeat(np.arange(2010, 2026), 1000)
    rng = np.random.RandomState(1)
    X = np.column_stack([seasons.astype(float), rng.normal(size=len(seasons))])
    y = rng.normal(size=len(seasons))

    X_sub, y_sub, seasons_sub = PositionModel._subsample_for_tuning(X, y, seasons)

    assert len(X_sub) == len(y_sub) == len(seasons_sub) == 8000
    assert np.all(X_sub[:, 0] == seasons_sub)            # label travels with its row
    assert len(set(seasons_sub)) == 16                    # older seasons are still represented
    assert len(set(seasons[-8000:])) < 16                 # the old tail-slice would not have been


def test_tuning_subsample_passthrough_below_cap():
    X, y, seasons = _data(3, per_season=10)
    X_sub, y_sub, seasons_sub = PositionModel._subsample_for_tuning(X, y, seasons)
    assert X_sub is X and y_sub is y and seasons_sub is seasons


def test_strict_raises_instead_of_falling_back_to_season_unaware_split():
    """strict=True must never silently degrade to the season-unaware plain
    TimeSeriesSplit fallback: too few seasons for the requested fold count
    should raise, not produce folds that don't actually respect season
    boundaries (see game_outcome_backtester.py / game_margin_backtester.py,
    whose walk-forward numbers are shown to users as honest, held-out
    accuracy -- a silent fallback there would misrepresent that)."""
    X, y, seasons = _data(4)  # only 4 seasons; n_splits=3 + gap=1 needs >= 5
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1, strict=True)
    with pytest.raises(ValueError, match="strict=True"):
        list(cv.split(X, y))


def test_strict_still_works_when_enough_seasons_are_available():
    X, y, seasons = _data(6)
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1, strict=True)
    folds = list(cv.split(X, y))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert seasons[train_idx].max() < seasons[test_idx].min()


def test_non_strict_default_still_falls_back_as_before():
    """Default behavior (strict=False) is unchanged -- tuning callers that
    rely on the season-unaware fallback for a fixed fold count keep it."""
    X, y, seasons = _data(4)
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=seasons, gap_seasons=1)
    folds = list(cv.split(X, y))
    assert len(folds) == cv.get_n_splits(X) == 3

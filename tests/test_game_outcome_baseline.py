"""Baseline correctness: well-formed probabilities, graceful degradation on
missing spread_line, and the spread-sign convention locked in against
hand-checked historical results (not just a docstring assumption).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.game_outcome.baseline import HomeFieldBaseline, VegasFavoriteBaseline

# Hand-checked real games (season, week, home, away, spread_line, home_won).
# spread_line convention: positive = home favored. Verified 2026-09-18 against
# data/nfl_data.db: corr(spread_line, home_margin) = +0.44 over 3,028 games
# (2015-2025), home favorites (spread_line > 0) won 66.7% of the time.
KNOWN_GAMES = [
    (2023, 10, "DAL", "NYG", 17.5, True),   # DAL favored by 17.5, won by 32
    (2023, 6, "BUF", "NYG", 15.5, True),    # BUF favored by 15.5, won by 5
    (2023, 17, "WAS", "SF", -14.0, False),  # WAS underdog by 14, lost
]


def test_probabilities_are_well_formed():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"spread_line": rng.uniform(-14, 14, size=200)})
    y = (X["spread_line"] + rng.normal(scale=5, size=200) > 0).astype(int)

    model = VegasFavoriteBaseline().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_degrades_to_half_on_null_spread_line():
    X_train = pd.DataFrame({"spread_line": [3.0, -3.0, 7.0, -7.0, 1.0, -1.0]})
    y_train = pd.Series([1, 0, 1, 0, 1, 0])
    model = VegasFavoriteBaseline().fit(X_train, y_train)

    X_test = pd.DataFrame({"spread_line": [np.nan]})
    proba = model.predict_proba(X_test)
    assert proba[0, 1] == pytest.approx(0.5)


def test_degenerate_fold_does_not_crash():
    # Every label the same class -- must fall back gracefully, not raise.
    X = pd.DataFrame({"spread_line": [3.0, -3.0, 7.0]})
    y = pd.Series([1, 1, 1])
    model = VegasFavoriteBaseline().fit(X, y)
    proba = model.predict_proba(X)
    assert np.allclose(proba[:, 1], 0.5)


def test_spread_sign_convention_matches_known_historical_games():
    X_train = pd.DataFrame({"spread_line": [s for *_, s, _ in KNOWN_GAMES] * 20})
    y_train = pd.Series([int(w) for *_, w in KNOWN_GAMES] * 20)
    model = VegasFavoriteBaseline().fit(X_train, y_train)

    for _, _, _, _, spread, won in KNOWN_GAMES:
        proba_home_win = model.predict_proba(pd.DataFrame({"spread_line": [spread]}))[0, 1]
        if spread > 0:
            assert proba_home_win > 0.5, "a home favorite must be predicted more likely to win"
        elif spread < 0:
            assert proba_home_win < 0.5, "a home underdog must be predicted less likely to win"


def test_home_field_baseline_always_predicts_home_win():
    X = pd.DataFrame({"spread_line": [3.0, -3.0, np.nan]})
    model = HomeFieldBaseline(win_rate=0.57)
    proba = model.predict_proba(X)
    assert np.allclose(proba[:, 1], 0.57)
    assert (model.predict(X) == 1).all()


def test_home_field_baseline_accuracy_matches_hand_computed_expectation():
    # 7 home wins, 3 home losses -> accuracy of "home always wins" is 0.7.
    y_true = np.array([1] * 7 + [0] * 3)
    model = HomeFieldBaseline()
    preds = model.predict(pd.DataFrame({"spread_line": np.zeros(10)}))
    accuracy = (preds == y_true).mean()
    assert accuracy == pytest.approx(0.7)

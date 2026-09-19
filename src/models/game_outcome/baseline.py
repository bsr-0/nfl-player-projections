"""Baselines the phase-1 models must clear.

Both expose the same `fit(X, y)` / `predict_proba(X)` interface as the real
models (src/models/game_outcome/models.py) so the backtester can score all
four arms identically.

Sign convention for `spread_line` (verified empirically against 2015-2025
results, 2026-09-18): positive = home team favored. corr(spread_line,
home_margin) = +0.44 over 3,028 games; home teams favored by the market
(spread_line > 0) won 66.7% of the time, home underdogs (spread_line < 0)
won 34.9% of the time. This is nflverse's own convention, not the
traditional US-sportsbook "negative = favorite" convention -- do not assume
the opposite sign without re-checking against real results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Historical home-field win rate floor. Not re-derived from data at fit time
# -- it is a fixed sanity floor every real model and the Vegas baseline must
# clear, not itself a competitive baseline.
HOME_FIELD_WIN_RATE = 0.57


class VegasFavoriteBaseline:
    """Predicts from `spread_line` alone via a 1-feature logistic curve.

    Fit once per walk-forward fold on that fold's training data only (this
    class has no season awareness itself -- the caller is responsible for
    only ever calling `fit` with rows from strictly-prior seasons).
    """

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y) -> "VegasFavoriteBaseline":
        s = X["spread_line"]
        y = pd.Series(y).reset_index(drop=True)
        mask = s.notna().reset_index(drop=True)
        s_valid = s.reset_index(drop=True)[mask]
        y_valid = y[mask]
        if len(y_valid) < 2 or y_valid.nunique() < 2:
            # Degenerate fold (too few rows, or every label the same class):
            # fall back to the trivial constant rather than crash.
            self.model = None
            self.fitted = False
            return self
        self.model = LogisticRegression()
        self.model.fit(s_valid.values.reshape(-1, 1), y_valid.values)
        self.fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = X["spread_line"].values
        proba = np.full(len(X), 0.5)
        if self.fitted and self.model is not None:
            mask = ~pd.isna(s)
            if mask.any():
                proba[mask] = self.model.predict_proba(s[mask].reshape(-1, 1))[:, 1]
        return np.column_stack([1 - proba, proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class HomeFieldBaseline:
    """Trivial floor: predict the home team always wins.

    Every real model and the Vegas baseline must clear this to be worth
    anything -- it costs nothing to compute and requires no fitting.
    """

    def __init__(self, win_rate: float = HOME_FIELD_WIN_RATE) -> None:
        self.win_rate = win_rate

    def fit(self, X: pd.DataFrame, y) -> "HomeFieldBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.full(len(X), self.win_rate)
        return np.column_stack([1 - proba, proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


class MarketLineBaseline:
    """Predicts the closing line itself -- e.g. `predicted_margin = spread_line`
    or `predicted_total = total_line`.

    Since `spread_line` already represents the market's implied home margin
    (positive = home favored by roughly that many points, see the sign-
    convention note above) and `total_line` already represents the market's
    implied combined score, this is not a trivial floor like `HomeFieldBaseline`
    -- it IS the market's own forecast, and the real bar for the margin/total
    regressors to clear (see src/evaluation/game_margin_backtester.py).
    Requires no fitting.
    """

    def __init__(self, line_column: str) -> None:
        self.line_column = line_column

    def fit(self, X: pd.DataFrame, y) -> "MarketLineBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.line_column].to_numpy()

"""Naive baseline for Plan A's share models.

`RollingShareBaseline` plays the same role `MarketLineBaseline` plays in
src/models/game_outcome/baseline.py: the bar every real model must clear.
It requires no fitting -- it just echoes the player's own lagged
trailing-3-game share, i.e. "my forecast for this week is what happened the
last 3 weeks."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RollingShareBaseline:
    def __init__(self, roll_col: str) -> None:
        self.roll_col = roll_col

    def fit(self, X: pd.DataFrame, y) -> "RollingShareBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.roll_col].fillna(0.0).to_numpy()

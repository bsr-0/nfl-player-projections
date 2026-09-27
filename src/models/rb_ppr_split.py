"""Pregame RB receiving fraction; never changes the supplied PPR center.

A small, prespecified history estimator. Population priors use earlier seasons;
player histories use only completed prior weeks, never the forecast week's stats.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd

KEY = ["player_id", "season", "week"]
COMPONENTS = ["receptions", "receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds"]


def component_history(history: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY + COMPONENTS)
    if missing := required - set(history):
        raise ValueError(f"RB history missing {sorted(missing)}")
    frame = history.copy()
    if "position" in frame and not frame.position.eq("RB").all():
        raise ValueError("split fitting history must contain only RBs")
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("RB history must have unique, nonnull player-week keys")
    values = frame[["season", "week"] + COMPONENTS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("RB history contains unknown/nonfinite stats")
    frame["receiving_component"] = (
        frame.receptions + .1 * frame.receiving_yards + 6 * frame.receiving_tds).clip(lower=0)
    frame["rushing_component"] = (.1 * frame.rushing_yards + 6 * frame.rushing_tds).clip(lower=0)
    return frame.sort_values(KEY, kind="stable")


@dataclass(frozen=True)
class RBReceivingFractionModel:
    train_through_season: int
    receiving_prior: float
    rushing_prior: float
    training_rows: int
    lookback_games: int = 8
    prior_games: float = 4.0

    def __post_init__(self):
        if (self.lookback_games < 1 or self.prior_games <= 0 or self.training_rows < 1
            or not np.isfinite([self.receiving_prior, self.rushing_prior, self.prior_games]).all()
            or min(self.receiving_prior, self.rushing_prior) < 0
            or self.receiving_prior + self.rushing_prior <= 0):
            raise ValueError("invalid RB split model parameters")

    def to_dict(self) -> dict:
        return asdict(self)

    def predict(self, rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        """Return fractions in input order, with explicit per-row evidence cutoffs."""
        if not set(KEY) <= set(rows) or rows[KEY].isna().any().any() or rows.duplicated(KEY).any():
            raise ValueError("forecast rows require unique, nonnull player-week keys")
        if "position" in rows and not rows.position.eq("RB").all():
            raise ValueError("receiving fraction only applies to RBs")
        if not (rows.season > self.train_through_season).all():
            raise ValueError("fitted prior must end before every forecast season")
        data = component_history(history)
        groups = {str(pid): group for pid, group in data.groupby("player_id", sort=False)}
        empty = data.iloc[:0]
        output = []
        for row in rows.itertuples(index=False):
            player = groups.get(str(row.player_id), empty)
            prior = player.loc[(player.season < row.season) | (
                (player.season == row.season) & (player.week < row.week))].tail(self.lookback_games)
            receiving = float(prior.receiving_component.sum()) + self.prior_games * self.receiving_prior
            rushing = float(prior.rushing_component.sum()) + self.prior_games * self.rushing_prior
            fraction = receiving / (receiving + rushing)
            output.append({"player_id": row.player_id, "season": row.season, "week": row.week,
                           "receiving_points_fraction": fraction, "history_games": len(prior),
                           "latest_history_season": int(prior.season.iloc[-1]) if len(prior) else None,
                           "latest_history_week": int(prior.week.iloc[-1]) if len(prior) else None,
                           "receiving_role": "low" if fraction < .25 else "mixed" if fraction < .5 else "high"})
        return pd.DataFrame(output, columns=KEY + ["receiving_points_fraction", "history_games",
                            "latest_history_season", "latest_history_week", "receiving_role"])


def fit_rb_receiving_fraction(history: pd.DataFrame, *, train_through_season: int,
                             lookback_games: int = 8, prior_games: float = 4.0) -> RBReceivingFractionModel:
    # Filter before any fitted summaries. Later rows cannot influence the prior.
    train = component_history(history.loc[history.season <= train_through_season])
    if train.empty:
        raise ValueError("no earlier-season RB training data")
    return RBReceivingFractionModel(train_through_season, float(train.receiving_component.mean()),
                                    float(train.rushing_component.mean()), len(train),
                                    lookback_games, prior_games)

"""Expected games played, for the season projection's opportunity term.

    season PPR = E[games played] x E[PPR per game | played]

The weekly model supplies the second factor and is NOT touched by anything
here. That separation is the point: the weekly model answers "how good is
this player when he plays", and this module answers "how often should I
expect him to play". Asking one estimator to do both is what the
2026-08-20 investigation removed.

Not to be confused with `availability.py`, which holds per-WEEK P(plays)
estimators built for the synthetic-week architecture that has since been
abandoned. Those are used only by two experiment scripts. This module is
season-level and is the one wired into projections.

### What this estimator does and does not claim

Validated (GAPS.md 2026-08-20): shrunk historical availability predicts
games played better than a constant in **12/12 position-folds**, by
0.49-1.07 games, and cuts mean |season PPR bias| from 15.02 to 4.42.

It improves season PPR MAE at QB and RB (3/3 folds each) and **not**
consistently at TE/WR (1/3 each). It is adopted for the games component on
the strength of the games-played and bias results, not on a general
season-MAE claim.

**Boundary, which the experiment could not cross:** the evaluation
population is players with >= 1 observed game, because a player with zero
games has no rows to evaluate against. So this answers *"conditional on
this player having a season, how many games does he play"* -- NOT *"will
this player play at all"*. It has not been shown to identify players who
miss a roster, start the season injured, retire, are cut, or are rookies
with no NFL history. Do not read the fallback below as a solution to that.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.models.single_week_ppr.population import (
    RECEIVING_CHARTING_MIN_SEASON, RECEIVING_DEPENDENT_POSITIONS, label_participation,
)
from config.settings import regular_season_max_week, regular_season_week_sql

# Weight on the player's own history is n_prior_games / (n_prior_games + K).
# At K=16 (roughly one season), a player with one prior season gets ~50%
# weight and a five-year veteran ~84%. The raw per-player rate over-corrects
# (WR season bias +12.36 vs +7.85 shrunk), which is why this exists.
SHRINKAGE_K = 16.0


def load_player_seasons(db_path: Optional[str] = None) -> pd.DataFrame:
    """Observed games and PPR per player-season, participation contract applied.

    Every row is an OBSERVED participation. Nothing is fabricated: a player
    with no rows in a season simply does not appear, and no zero-week is
    manufactured to fill the gap.
    """
    import sqlite3
    from config.settings import DB_PATH, MIN_HISTORICAL_YEAR, POSITIONS

    conn = sqlite3.connect(str(db_path or DB_PATH))
    df = pd.read_sql(f"""
        SELECT pws.player_id, p.position, pws.season, pws.week, pws.team,
               pws.snap_count, pws.fantasy_points, pws.data_source
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE p.position IN ({','.join('?' * len(POSITIONS))})
          AND {regular_season_week_sql('pws.week', 'pws.season')}
          AND pws.season >= {MIN_HISTORICAL_YEAR}
    """, conn, params=list(POSITIONS))
    sched = pd.read_sql(
        f"SELECT season, week, home_team AS team FROM schedule "
        f"WHERE {regular_season_week_sql()} UNION ALL "
        f"SELECT season, week, away_team AS team FROM schedule "
        f"WHERE {regular_season_week_sql()}", conn)
    conn.close()

    df["participation_quality"] = label_participation(df)
    df = df[df["participation_quality"] >= 1]
    recv = df["position"].isin(RECEIVING_DEPENDENT_POSITIONS)
    df = df[~(recv & (df["season"] < RECEIVING_CHARTING_MIN_SEASON))]

    played = df.groupby(["player_id", "position", "season"]).agg(
        games_played=("week", "nunique"),
        ppr=("fantasy_points", "sum"),
        team=("team", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
    ).reset_index()
    possible = (sched.groupby(["season", "team"])["week"].nunique()
                .rename("possible_games").reset_index())
    played = played.merge(possible, on=["season", "team"], how="left")
    # Fallback for a team with no schedule rows at all. Was a flat 18, wrong
    # twice: 18 is a WEEK count and this column counts GAMES (one bye), and it
    # ignored the era. Games = max regular-season week - 1, i.e. 16 through
    # 2020 and 17 from 2021.
    fallback = played["season"].map(lambda s: regular_season_max_week(s) - 1)
    played["possible_games"] = played["possible_games"].fillna(fallback)
    played["rate"] = played["games_played"] / played["possible_games"]
    played["ppr_per_game"] = played["ppr"] / played["games_played"]
    return played


class SeasonAvailabilityEstimator:
    """E[games played] from prior seasons only, shrunk toward the position mean.

    Causality is structural rather than conventional: `fit` requires the
    target season and drops anything at or after it, so a caller cannot
    accidentally train on the season being projected.
    """

    def __init__(self, shrinkage_k: float = SHRINKAGE_K):
        self.shrinkage_k = float(shrinkage_k)
        self.before_season: Optional[int] = None
        self.position_rate_: Optional[pd.Series] = None
        self.player_history_: Optional[pd.DataFrame] = None

    def fit(self, player_seasons: pd.DataFrame, before_season: int) -> "SeasonAvailabilityEstimator":
        required = {"player_id", "position", "season", "games_played", "possible_games"}
        missing = required - set(player_seasons.columns)
        if missing:
            raise ValueError(f"player_seasons missing columns: {sorted(missing)}")

        hist = player_seasons[pd.to_numeric(player_seasons["season"], errors="coerce")
                              < int(before_season)]
        if hist.empty:
            raise ValueError(f"no seasons before {before_season} to fit on")

        self.before_season = int(before_season)
        self.position_rate_ = (hist["games_played"] / hist["possible_games"]).groupby(
            hist["position"]).mean()
        self.player_history_ = hist.groupby("player_id").agg(
            prior_games=("games_played", "sum"),
            prior_possible=("possible_games", "sum")).reset_index()
        return self

    def predict_rate(self, players: pd.DataFrame) -> pd.Series:
        """Expected participation rate per row.

        Fallback for a player with no prior history: the position mean rate,
        with zero weight on player history (`w = 0 / (0 + K)`). This is a
        documented degradation, not an estimate of whether an unknown player
        will play at all -- see the module docstring.
        """
        if self.position_rate_ is None:
            raise RuntimeError("call fit() before predict_rate()")

        merged = players.merge(self.player_history_, on="player_id", how="left")
        prior_games = merged["prior_games"].fillna(0.0)
        player_rate = merged["prior_games"] / merged["prior_possible"]

        position_rate = merged["position"].map(self.position_rate_)
        if position_rate.isna().any():
            position_rate = position_rate.fillna(self.position_rate_.mean())

        w = prior_games / (prior_games + self.shrinkage_k)
        rate = w * player_rate.fillna(position_rate) + (1.0 - w) * position_rate
        return pd.Series(np.clip(rate.to_numpy(), 0.0, 1.0), index=players.index)

    def predict_games(self, players: pd.DataFrame,
                      possible_games: Optional[pd.Series] = None) -> pd.Series:
        if possible_games is None:
            possible_games = players["possible_games"]
        return self.predict_rate(players) * np.asarray(possible_games, dtype=float)

    def has_history(self, players: pd.DataFrame) -> pd.Series:
        known = set(self.player_history_["player_id"]) if self.player_history_ is not None else set()
        return players["player_id"].isin(known)


def project_season_ppr(expected_games: pd.Series, expected_ppr_per_game: pd.Series) -> pd.Series:
    """season PPR = E[games] x E[PPR per game].

    Deliberately the whole calculation. The season-bias improvement comes
    from this product; adding a separate post-hoc bias correction on top
    would double-count the same effect.
    """
    return pd.Series(np.asarray(expected_games, dtype=float)
                     * np.asarray(expected_ppr_per_game, dtype=float),
                     index=getattr(expected_games, "index", None))

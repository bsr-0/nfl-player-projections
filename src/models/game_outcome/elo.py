"""Sequential Elo power-rating system for the game-outcome model.

Unlike the `_lagged_form` aggregates in features.py (which lag a rolling
STATISTIC), Elo is inherently sequential state: each team carries one
running rating that updates game-by-game. The "pre-game" rating attached to
a given (team, season, week) is, by construction, computed only from games
strictly before it in chronological (season, week) order -- there is no
separate shift/lag step needed the way there is for the rolling-mean
features, because the update loop itself never looks ahead.

Ties ARE included when fitting (unlike the win/loss and margin/total label
populations, which drop the ~0.1% of games that end in a tie) -- a tie is a
real result that should move both teams' ratings toward each other (via
actual_home=0.5), and this module is a general-purpose sequential rating,
not tied to the classifier's label population.
"""
from __future__ import annotations

import math
import sqlite3
from typing import Dict, Optional

import pandas as pd


class EloRatingSystem:
    """Sequential Elo ratings for NFL teams.

    `fit(games)` processes `games` in the (season, week) order given (caller
    is responsible for sorting -- see `_team_elo_pre` in features.py) and
    records, for every game, the PRE-game rating of both teams before that
    game's own result is folded in. `current_elo(team, as_of_season)` gives
    the latest known rating for a team as of some future season, applying
    the same season-boundary regression the sequential loop would have
    applied had that team actually played in `as_of_season` already --
    used only for not-yet-played games (see `build_prediction_rows`).
    """

    def __init__(
        self,
        initial: float = 1500.0,
        k: float = 20.0,
        home_advantage: float = 55.0,
        season_regression: float = 0.75,
        mov_multiplier: bool = True,
    ):
        self.initial = initial
        self.k = k
        self.home_advantage = home_advantage
        self.season_regression = season_regression
        self.mov_multiplier = mov_multiplier
        self.ratings_: Dict[str, float] = {}
        self.last_season_: Dict[str, int] = {}
        self.history_: pd.DataFrame = pd.DataFrame(
            columns=["season", "week", "home_team", "away_team", "home_elo_pre", "away_elo_pre"]
        )

    def _regressed(self, rating: float) -> float:
        return self.initial + self.season_regression * (rating - self.initial)

    def _pregame_rating(self, team: str, season: int) -> float:
        """Current rating for `team`, regressing toward `initial` exactly
        once if `team`'s last played season is strictly before `season`
        (a new-season reset, applied lazily the first time the team is
        seen in the new season -- mirrors how a real power-rating carries
        over across the offseason rather than resetting to a blank slate)."""
        if team not in self.ratings_:
            self.ratings_[team] = self.initial
            return self.initial
        rating = self.ratings_[team]
        if self.last_season_.get(team, season) < season:
            rating = self._regressed(rating)
            self.ratings_[team] = rating
        return rating

    def fit(self, games: pd.DataFrame) -> "EloRatingSystem":
        """`games` must have season, week, home_team, away_team, home_score,
        away_score, already sorted chronologically (season, then week) --
        this method does not sort, since callers may have additional
        same-week tiebreak ordering (e.g. game_time) that matters more than
        this module should assume."""
        history_rows = []
        for row in games.itertuples(index=False):
            home, away, season, week = row.home_team, row.away_team, row.season, row.week
            home_pre = self._pregame_rating(home, season)
            away_pre = self._pregame_rating(away, season)

            elo_diff = (home_pre + self.home_advantage) - away_pre
            expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

            if row.home_score > row.away_score:
                actual_home = 1.0
            elif row.home_score < row.away_score:
                actual_home = 0.0
            else:
                actual_home = 0.5

            margin = abs(row.home_score - row.away_score)
            mov_mult = math.log(margin + 1) if self.mov_multiplier else 1.0
            update = self.k * mov_mult * (actual_home - expected_home)

            history_rows.append(
                {
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "home_elo_pre": home_pre,
                    "away_elo_pre": away_pre,
                }
            )
            self.ratings_[home] = home_pre + update
            self.ratings_[away] = away_pre - update
            self.last_season_[home] = season
            self.last_season_[away] = season

        self.history_ = pd.DataFrame(history_rows)
        return self

    def current_elo(self, team: str, as_of_season: int) -> float:
        """Latest known rating for `team`, as it would be seen ahead of
        `as_of_season` if `team` has not yet played that season -- applies
        the same season-boundary regression `_pregame_rating` would, without
        mutating state (used for not-yet-played games only)."""
        if team not in self.ratings_:
            return self.initial
        rating = self.ratings_[team]
        if self.last_season_.get(team, as_of_season) < as_of_season:
            rating = self._regressed(rating)
        return rating

    def pregame_long_frame(self) -> pd.DataFrame:
        """One row per (team, season, week) the team appeared in a PLAYED
        game, with that game's pre-game rating -- home/away sides stacked
        into the same long format `_lagged_form`'s callers already use."""
        home = self.history_[["season", "week", "home_team", "home_elo_pre"]].rename(
            columns={"home_team": "team", "home_elo_pre": "elo_pre"}
        )
        away = self.history_[["season", "week", "away_team", "away_elo_pre"]].rename(
            columns={"away_team": "team", "away_elo_pre": "elo_pre"}
        )
        return pd.concat([home, away], ignore_index=True)


def load_all_completed_games(con: sqlite3.Connection) -> pd.DataFrame:
    """All completed games (ties included -- see module docstring), full
    history, sorted chronologically. Deliberately not the label-building
    query in labels.py: that one drops ties and never returns scores past
    its own module. This one is Elo-internal and never merged into any
    feature frame directly (only `EloRatingSystem`'s pre-game ratings are)."""
    df = pd.read_sql(
        """
        SELECT season, week, home_team, away_team, home_score, away_score
        FROM schedule
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL AND week != 0
        ORDER BY season, week
        """,
        con,
    )
    return df.reset_index(drop=True)

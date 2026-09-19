"""Label construction for the game-outcome model.

home_win = 1 if home_score > away_score else 0, computed only from completed,
non-tie games. Ties are dropped entirely rather than encoded as a third class
or arbitrarily broken -- they are ~0.1% of games and a binary classifier has
no principled way to place them.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Optional

import pandas as pd

from config.settings import DB_PATH

logger = logging.getLogger(__name__)


def _load_completed_nontie_games_with_scores(
    seasons: Optional[Iterable[int]],
    con: Optional[sqlite3.Connection],
) -> pd.DataFrame:
    """Shared query behind both label builders below: completed, non-tie
    games with home_score/away_score still attached. Callers derive their
    own label(s) from the scores and must drop the score columns before
    returning -- never expose raw scores past this module (see
    src/utils/leakage.py:sanitize_schedule_df, the second line of defense
    if a caller merges a returned frame with a raw schedule pull)."""
    owns_con = con is None
    conn = sqlite3.connect(str(DB_PATH)) if owns_con else con
    try:
        params: list = []
        season_filter = ""
        if seasons is not None:
            # Cast to plain Python int -- a caller passing numpy.int64 values
            # (e.g. from a DataFrame's `.unique()`, as scripts/tune_elo_params.py
            # does) causes sqlite3 to silently bind them as something that
            # matches NO row rather than raising, so this returns an
            # empty-but-valid-looking DataFrame instead of an error. Found
            # 2026-09-18 while tuning Elo hyperparameters.
            seasons = [int(s) for s in seasons]
            placeholders = ",".join("?" * len(seasons))
            season_filter = f"AND season IN ({placeholders})"
            params = seasons
        df = pd.read_sql(
            f"""
            SELECT season, week, home_team, away_team,
                   home_score, away_score, spread_line, total_line
            FROM schedule
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            {season_filter}
            ORDER BY season, week
            """,
            conn,
            params=params,
        )
    finally:
        if owns_con:
            conn.close()

    n_before = len(df)
    df = df[df["home_score"] != df["away_score"]].copy()
    n_ties = n_before - len(df)
    if n_ties:
        logger.info("Dropped %d tied game(s) from game-outcome training data", n_ties)
    return df.reset_index(drop=True)


def load_completed_games(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """One row per completed, non-tie game with its home_win label attached.

    Returns: season, week, home_team, away_team, spread_line, total_line, home_win.
    """
    df = _load_completed_nontie_games_with_scores(seasons, con)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    return df.drop(columns=["home_score", "away_score"])


def load_margin_total_labels(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """One row per completed, non-tie game with home_margin/game_total labels.

    Returns: season, week, home_team, away_team, spread_line, total_line,
    home_margin, game_total. Same completed-non-tie game population as
    `load_completed_games` (dropping the same rare ties keeps both targets'
    populations identical for apples-to-apples comparison), but the labels
    themselves are continuous: `home_margin = home_score - away_score`,
    `game_total = home_score + away_score`.
    """
    df = _load_completed_nontie_games_with_scores(seasons, con)
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["game_total"] = df["home_score"] + df["away_score"]
    return df.drop(columns=["home_score", "away_score"])


def load_upcoming_games(
    season: int,
    week: Optional[int] = None,
    con: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """One row per scheduled-but-not-yet-played game for `season` (optionally
    restricted to a single `week`). No label -- the game hasn't happened yet.

    Returns: season, week, home_team, away_team, spread_line, total_line.
    `spread_line`/`total_line` are NaN for games too far out for the market
    to have posted a line yet (see src/models/game_outcome/predict.py).
    """
    owns_con = con is None
    conn = sqlite3.connect(str(DB_PATH)) if owns_con else con
    try:
        params: list = [int(season)]  # see numpy.int64 SQL-binding note above
        week_filter = ""
        if week is not None:
            week_filter = "AND week = ?"
            params.append(int(week))
        df = pd.read_sql(
            f"""
            SELECT season, week, home_team, away_team, spread_line, total_line
            FROM schedule
            WHERE season = ? {week_filter}
              AND (home_score IS NULL OR away_score IS NULL)
            ORDER BY week
            """,
            conn,
            params=params,
        )
    finally:
        if owns_con:
            conn.close()
    return df.reset_index(drop=True)

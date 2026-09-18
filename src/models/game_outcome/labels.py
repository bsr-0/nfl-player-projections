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


def load_completed_games(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """One row per completed, non-tie game with its home_win label attached.

    Returns: season, week, home_team, away_team, spread_line, total_line, home_win.

    home_score/away_score are deliberately NOT returned -- they are consumed
    here only to derive the label, and must never reappear in a frame that is
    later used as model input (see src/utils/leakage.py:sanitize_schedule_df,
    which is the second line of defense if a caller merges this frame with a
    raw schedule pull).
    """
    owns_con = con is None
    conn = sqlite3.connect(str(DB_PATH)) if owns_con else con
    try:
        params: list = []
        season_filter = ""
        if seasons is not None:
            seasons = list(seasons)
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

    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df = df.drop(columns=["home_score", "away_score"])
    return df.reset_index(drop=True)

"""Player age, derived from birth date rather than assumed from position.

Exists because `season_long_features.add_age_features` used to fall back to
`avg_ages = {'QB': 28, 'RB': 25, 'WR': 26, 'TE': 27}` whenever neither
`age` nor `years_exp` was in the frame -- and neither ever was, because
`get_all_players_for_training()` returns neither. So every training row got
its position's constant, which made `age_curve` (a declared CAUSAL_FEATURE
at all four positions) a zero-variance column, along with `age_factor`,
`age_expected_games`, `decline_rate`, `years_from_peak` and `is_in_prime`.
See GAPS.md 2026-08-19.

Age is taken as of Sept 1 of the season, matching
`preseason_projector.PreseasonProjector._season_start`, so a single player
has one age for the whole season rather than one that ticks over mid-year.
Birth date is static and known years in advance, so this introduces no
look-ahead.

The position-constant fallback survives as a last resort, but it now warns
loudly above `FALLBACK_WARN_THRESHOLD` -- the guardrail whose absence let
the original bug run silently. `players.birth_date` covers ~99% of players
after scripts/backfill_player_birth_dates.py, so a high fallback rate now
means something is wrong rather than something is normal.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Same values the old silent fallback used, kept so the last-resort path is
# unchanged in behaviour -- what changed is that it is now the last resort
# instead of the only path.
POSITION_FALLBACK_AGES: Dict[str, float] = {"QB": 28.0, "RB": 25.0, "WR": 26.0, "TE": 27.0}
DEFAULT_FALLBACK_AGE = 26.0

# Above this share of rows on the position constant, warn. Coverage is ~100%
# when the birth-date backfill has been run, so anything near this threshold
# is a regression, not a data limitation.
FALLBACK_WARN_THRESHOLD = 0.10

DAYS_PER_YEAR = 365.25


def season_start(season: int) -> datetime:
    return datetime(int(season), 9, 1)


@lru_cache(maxsize=4)
def birth_date_map(db_path: Optional[str] = None) -> Dict[str, str]:
    """player_id -> birth_date, cached (static data, read once per process)."""
    if db_path is None:
        from config.settings import DB_PATH
        db_path = str(DB_PATH)
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT player_id, birth_date FROM players "
            "WHERE birth_date IS NOT NULL AND birth_date != ''"
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        logger.warning("Could not read birth dates from %s: %s", db_path, e)
        return {}
    return {pid: bd for pid, bd in rows if bd}


def age_from_birth_date(birth_date, season) -> float:
    if birth_date is None or birth_date == "" or pd.isna(birth_date):
        return np.nan
    try:
        born = pd.to_datetime(birth_date)
    except (ValueError, TypeError):
        return np.nan
    if pd.isna(born) or pd.isna(season):
        return np.nan
    return (season_start(int(season)) - born.to_pydatetime()).days / DAYS_PER_YEAR


def derive_age(df: pd.DataFrame, db_path: Optional[str] = None) -> pd.Series:
    """Age in years per row, best available source first.

        1. a `birth_date` column on the frame
        2. `players.birth_date`, joined on player_id
        3. an `age` column already on the frame
        4. 22 + years_exp
        5. the position constant

    Steps 4-5 are the degraded paths the original bug lived in; the
    fallback rate is logged so they cannot be silent again.

    Birth dates deliberately outrank an existing `age` column (they were
    ranked below it until FEATURE_VERSION 34). `season_long_features`
    populates `age` with a per-POSITION CONSTANT before this runs, and since
    the birth-date fill only touches NaNs, consulting the column first meant
    real birth dates were never read at all -- `age_curve` was effectively
    constant even for the 98.9% of players whose birth date is known. A real
    date always beats a placeholder; the column survives as a fallback for
    rows with no birth date.
    """
    if "season" not in df.columns:
        raise ValueError("derive_age needs a 'season' column")

    season = pd.to_numeric(df["season"], errors="coerce")
    age = pd.Series(np.nan, index=df.index, dtype=float)

    def _fill_from_birth_dates(current: pd.Series, births: pd.Series) -> pd.Series:
        missing = current.isna() & births.notna() & season.notna()
        if not missing.any():
            return current
        computed = [
            age_from_birth_date(b, s)
            for b, s in zip(births[missing], season[missing])
        ]
        current.loc[missing] = computed
        return current

    if "birth_date" in df.columns:
        age = _fill_from_birth_dates(age, df["birth_date"])

    if age.isna().any() and "player_id" in df.columns:
        age = _fill_from_birth_dates(age, df["player_id"].map(birth_date_map(db_path)))

    # Only now consult a pre-existing `age` column -- see the docstring for why
    # it must not take priority over a real birth date.
    if age.isna().any() and "age" in df.columns:
        age = age.fillna(pd.to_numeric(df["age"], errors="coerce"))

    n_before_degraded = int(age.isna().sum())

    if age.isna().any() and "years_exp" in df.columns:
        exp = pd.to_numeric(df["years_exp"], errors="coerce")
        missing = age.isna() & exp.notna()
        age.loc[missing] = 22.0 + exp[missing]

    n_constant = 0
    if age.isna().any():
        position = df["position"] if "position" in df.columns else pd.Series(index=df.index, dtype=object)
        constant = position.map(POSITION_FALLBACK_AGES).fillna(DEFAULT_FALLBACK_AGE)
        missing = age.isna()
        n_constant = int(missing.sum())
        age.loc[missing] = constant[missing]

    total = len(df)
    if total and n_constant / total > FALLBACK_WARN_THRESHOLD:
        logger.warning(
            "age fell back to the position constant on %.1f%% of rows (>%.0f%% threshold). "
            "This makes age_curve and every age-derived feature near-constant. "
            "Run scripts/backfill_player_birth_dates.py.",
            100.0 * n_constant / total, 100.0 * FALLBACK_WARN_THRESHOLD,
        )
    elif n_before_degraded:
        logger.info("age: %d/%d rows had no birth date (%d ended on the position constant)",
                    n_before_degraded, total, n_constant)

    return age

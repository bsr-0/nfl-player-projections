"""Leakage-safe feature construction for the game-outcome model.

One row per completed game, oriented home-vs-away: team-performance features
are `home_minus_away_*` differences of pre-game-known quantities, rather than
two independent home/away rows. Two rows per game would not be independent
samples (a CV fold could split a game's two "perspectives" across train/test),
so the single-row-diff framing structurally avoids that leakage class.

All team-form aggregates are lagged to strictly before the target game via
`shift(1)` ahead of `.expanding()`/`.rolling()` -- the same discipline
documented in src/utils/leakage.py's FEATURE_AVAILABILITY registry for the
player models, extended here for the `home_minus_away_` column family.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH, GAME_OUTCOME_MODEL_CONFIG
from src.models.game_outcome.labels import load_completed_games
from src.utils.leakage import audit_feature_availability, sanitize_schedule_df

logger = logging.getLogger(__name__)

# team_stats aggregates to build home/away form differences from. Confirmed
# against the live DB 2026-09-18: total_plays/third_down_conv are 100%
# populated every season 2006-2025; drive_success_rate/avg_drive_epa/
# points_per_drive/neutral_pass_rate_oe are 94-100% every season (89.2% in
# 2025 specifically) -- good enough to use across the full history without
# restricting the earliest training season.
TEAM_STAT_COLS = [
    "points_scored",
    "points_allowed",
    "total_yards",
    "passing_yards",
    "rushing_yards",
    "turnovers",
    "third_down_conv",
    "drive_success_rate",
    "avg_drive_epa",
    "points_per_drive",
    "neutral_pass_rate_oe",
]

# team_stats has two known-junk all-null team codes for the 2025 season only
# (LAR, JAC) alongside the real, populated LA/JAX rows that `schedule` and
# every other table actually use -- confirmed 2026-09-18 by comparing team
# codes across schedule/team_stats/game_weather (schedule and game_weather
# agree exactly; team_stats has 34 codes to their 32, the two extras being
# these all-null rows). Filtering to points_scored IS NOT NULL drops them
# without needing an explicit alias map, and is a general guard against any
# future dead-code-path row of the same shape.
_TEAM_STATS_SQL = """
    SELECT team, season, week, points_scored, points_allowed, total_yards,
           passing_yards, rushing_yards, turnovers, third_down_conv,
           drive_success_rate, avg_drive_epa, points_per_drive,
           neutral_pass_rate_oe
    FROM team_stats
    WHERE points_scored IS NOT NULL
"""


def _connect(con: Optional[sqlite3.Connection]) -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH)) if con is None else con


def _load_team_week_stats(con: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql(_TEAM_STATS_SQL, con)
    return df.sort_values(["team", "season", "week"]).reset_index(drop=True)


def _lagged_form(
    team_week: pd.DataFrame,
    stat_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """Per-team, per-week, season-to-date and last-N-game means of PRIOR games.

    `shift(1)` is applied before `.expanding()`/`.rolling()` so the value
    attached to (team, season, week) uses only games strictly before that
    week within the same season -- never the target game itself.
    """
    grp = team_week.groupby(["team", "season"], group_keys=False)
    out = team_week[["team", "season", "week"]].copy()
    out["n_prior_games"] = grp.cumcount()

    for col in stat_cols:
        out[f"{col}_s2d"] = grp[col].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        out[f"{col}_roll{window}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return out


def _prior_season_means(team_week: pd.DataFrame, stat_cols: list[str]) -> pd.DataFrame:
    """Full-season mean per (team, season), shifted forward one season.

    Used as the cold-start fallback for a team's early-season weeks, where
    strictly-prior-game data within the current season is too thin to trust.
    """
    means = team_week.groupby(["team", "season"])[stat_cols].mean().reset_index()
    means["season"] = means["season"] + 1
    return means.rename(columns={c: f"{c}_prior_season" for c in stat_cols})


def _team_form_table(con: sqlite3.Connection, window: int, min_prior_games: int) -> pd.DataFrame:
    """One row per (team, season, week): lagged form + cold-start fallback + flag."""
    team_week = _load_team_week_stats(con)
    lagged = _lagged_form(team_week, TEAM_STAT_COLS, window)
    prior = _prior_season_means(team_week, TEAM_STAT_COLS)
    lagged = lagged.merge(prior, on=["team", "season"], how="left")

    lagged["is_cold_start"] = (lagged["n_prior_games"] < min_prior_games).astype(int)
    for col in TEAM_STAT_COLS:
        cold = lagged["is_cold_start"].astype(bool)
        # Cold start: prefer the prior season's full-season mean; if that is
        # also missing (team's first tracked season -- 2006, or a relocated
        # franchise), leave NaN for the model's own imputer to handle rather
        # than silently zero-filling.
        lagged.loc[cold, f"{col}_s2d"] = lagged.loc[cold, f"{col}_s2d"].where(
            lagged.loc[cold, f"{col}_prior_season"].isna(), lagged.loc[cold, f"{col}_prior_season"]
        )
        lagged.loc[cold, f"{col}_roll{window}"] = lagged.loc[cold, f"{col}_roll{window}"].where(
            lagged.loc[cold, f"{col}_prior_season"].isna(), lagged.loc[cold, f"{col}_prior_season"]
        )
    drop_cols = [f"{c}_prior_season" for c in TEAM_STAT_COLS] + ["n_prior_games"]
    return lagged.drop(columns=drop_cols)


def _team_win_form(con: sqlite3.Connection, window: int, min_prior_games: int) -> pd.DataFrame:
    """Same lagged-form treatment as _team_form_table, applied to win/loss.

    Built from `schedule` results directly (not team_stats) -- cheap, no new
    machinery, and a real predictive signal on its own (recent form).
    """
    games = load_completed_games(con=con)
    home = games[["season", "week", "home_team", "home_win"]].rename(
        columns={"home_team": "team", "home_win": "win"}
    )
    away = games[["season", "week", "away_team", "home_win"]].rename(
        columns={"away_team": "team"}
    )
    away["win"] = 1 - games["home_win"]
    team_results = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "season", "week"]
    )

    grp = team_results.groupby(["team", "season"], group_keys=False)
    out = team_results[["team", "season", "week"]].copy()
    out["n_prior_games"] = grp.cumcount()
    out["win_pct_s2d"] = grp["win"].transform(lambda s: s.shift(1).expanding().mean())
    out[f"win_pct_roll{window}"] = grp["win"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )

    prior_season = team_results.groupby(["team", "season"])["win"].mean().reset_index()
    prior_season["season"] = prior_season["season"] + 1
    prior_season = prior_season.rename(columns={"win": "prior_season_win_pct"})
    out = out.merge(prior_season, on=["team", "season"], how="left")

    cold = out["n_prior_games"] < min_prior_games
    out.loc[cold, "win_pct_s2d"] = out.loc[cold, "win_pct_s2d"].where(
        out.loc[cold, "prior_season_win_pct"].isna(), out.loc[cold, "prior_season_win_pct"]
    )
    out.loc[cold, f"win_pct_roll{window}"] = out.loc[cold, f"win_pct_roll{window}"].where(
        out.loc[cold, "prior_season_win_pct"].isna(), out.loc[cold, "prior_season_win_pct"]
    )
    return out.drop(columns=["n_prior_games"])


def _load_weather(con: sqlite3.Connection, seasons: Optional[list[int]]) -> pd.DataFrame:
    params: list = []
    season_filter = ""
    if seasons is not None:
        placeholders = ",".join("?" * len(seasons))
        season_filter = f"WHERE season IN ({placeholders})"
        params = seasons
    return pd.read_sql(
        f"""
        SELECT season, week, home_team, away_team, is_dome, temp_f, wind_mph, precip_mm
        FROM game_weather
        {season_filter}
        """,
        con,
        params=params,
    )


def build_game_outcome_rows(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
    include_weather: bool = True,
) -> pd.DataFrame:
    """One row per completed game: id columns + home_win label + feature columns.

    `seasons`, if given, restricts the returned games -- but team-form tables
    are always built from the FULL team_stats/schedule history so that early
    games in the requested range still get correctly lagged prior-game data
    rather than an artificially cold start.
    """
    owns_con = con is None
    conn = _connect(con)
    try:
        window = GAME_OUTCOME_MODEL_CONFIG["rolling_window_games"]
        min_prior = GAME_OUTCOME_MODEL_CONFIG["min_prior_games_for_form"]

        games = load_completed_games(seasons=seasons, con=conn)
        games = sanitize_schedule_df(games)  # defense in depth; label already extracted

        team_form = _team_form_table(conn, window, min_prior)
        win_form = _team_win_form(conn, window, min_prior)

        form_cols = [c for c in team_form.columns if c not in ("team", "season", "week")]
        win_cols = [c for c in win_form.columns if c not in ("team", "season", "week")]

        home_form = team_form.rename(
            columns={"team": "home_team", **{c: f"home_{c}" for c in form_cols}}
        )
        away_form = team_form.rename(
            columns={"team": "away_team", **{c: f"away_{c}" for c in form_cols}}
        )
        home_win_form = win_form.rename(
            columns={"team": "home_team", **{c: f"home_{c}" for c in win_cols}}
        )
        away_win_form = win_form.rename(
            columns={"team": "away_team", **{c: f"away_{c}" for c in win_cols}}
        )

        rows = games.merge(home_form, on=["season", "week", "home_team"], how="left")
        rows = rows.merge(away_form, on=["season", "week", "away_team"], how="left")
        rows = rows.merge(home_win_form, on=["season", "week", "home_team"], how="left")
        rows = rows.merge(away_win_form, on=["season", "week", "away_team"], how="left")

        # is_cold_start is a per-team flag, not a stat to difference -- handled
        # separately below as a combined home-or-away flag.
        diff_cols = [c for c in form_cols if c != "is_cold_start"] + win_cols
        for col in diff_cols:
            rows[f"home_minus_away_{col}"] = rows[f"home_{col}"] - rows[f"away_{col}"]

        rows["is_cold_start"] = (
            (rows["home_is_cold_start"] == 1) | (rows["away_is_cold_start"] == 1)
        ).astype(int)
        rows["game_week"] = rows["week"].astype(int)

        # Drop only the intermediate per-side columns we just merged in and
        # differenced -- NOT the "home_minus_away_*" diff columns themselves
        # or "home_win" (both also happen to start with "home_").
        intermediate_cols = [f"home_{c}" for c in form_cols] + [f"away_{c}" for c in form_cols]
        intermediate_cols += [f"home_{c}" for c in win_cols] + [f"away_{c}" for c in win_cols]
        rows = rows.drop(columns=intermediate_cols)

        if include_weather:
            weather = _load_weather(conn, list(seasons) if seasons is not None else None)
            rows = rows.merge(
                weather, on=["season", "week", "home_team", "away_team"], how="left"
            )
    finally:
        if owns_con:
            conn.close()

    id_cols = ["season", "week", "home_team", "away_team"]
    label_col = "home_win"
    feature_cols = [c for c in rows.columns if c not in id_cols + [label_col]]

    unclassified = audit_feature_availability(feature_cols)
    if unclassified:
        raise ValueError(
            "Unclassified game-outcome feature columns (add to "
            f"src/utils/leakage.py's FEATURE_AVAILABILITY): {sorted(unclassified)}"
        )

    return rows[id_cols + [label_col] + feature_cols].reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    id_and_label = {"season", "week", "home_team", "away_team", "home_win"}
    return [c for c in df.columns if c not in id_and_label]

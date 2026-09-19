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

IMPORTANT data-quality finding (2026-09-18): `team_stats.points_scored`,
`points_allowed`, `turnovers`, and `third_down_conv` are placeholder zeros
for every row 2006-2022 (the schema's `DEFAULT 0`, never actually populated
by the upstream merge for those seasons -- confirmed by querying the live
DB: all four columns are exactly 0 for 100% of 2006-2022 rows, only
becoming real-valued from 2023 onward). `total_yards`/`passing_yards`/
`rushing_yards` and the drive-based columns (`drive_success_rate`,
`avg_drive_epa`, `points_per_drive`, `neutral_pass_rate_oe`) are genuinely
populated the whole history -- they come from a different (PBP-direct)
aggregation path than the broken schedule-merge one. This is a pre-existing
defect in `team_stats`, not something introduced here, and it is out of
scope to fix the upstream loader in this phase -- fixing it here: points
are instead derived directly from `schedule.home_score`/`away_score`
(reliable back to 2006, see labels.py), and turnovers/third_down_conv are
dropped from phase 1's feature set entirely (no reliable full-history
substitute without new PBP aggregation work -- a real phase-2 item).
Mixing a real 2023 value against a phantom zero from 2022 in a cold-start
prior-season fallback is exactly the kind of silent defect this would have
caused (see GAPS.md-style incidents elsewhere in this repo): it produced a
-188-point season-average diff in an early walk-forward backtest run before
this fix, which is what surfaced the underlying defect.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH, GAME_OUTCOME_MODEL_CONFIG
from src.models.game_outcome.elo import EloRatingSystem, load_all_completed_games
from src.models.game_outcome.labels import (
    load_completed_games,
    load_margin_total_labels,
    load_upcoming_games,
)
from src.models.game_outcome.market_odds import load_market_odds_features
from src.utils.leakage import audit_feature_availability, sanitize_schedule_df

logger = logging.getLogger(__name__)

# Reliable team_stats aggregates only -- see the module docstring for why
# points_scored/points_allowed/turnovers/third_down_conv are excluded here
# and rebuilt from `schedule` instead.
TEAM_STAT_COLS = [
    "total_yards",
    "passing_yards",
    "rushing_yards",
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
# these all-null rows). Filtering to total_yards IS NOT NULL drops them
# without needing an explicit alias map, and is a general guard against any
# future dead-code-path row of the same shape.
#
# `week != 0` is defense-in-depth against a second, separate junk-row
# pattern that DID exist in team_stats (31 rows/season, 2020-2025, all
# total_plays=0 placeholders -- confirmed pre-existing, unrelated to any
# real game; `schedule` has never had a week=0 row). Root cause traced
# 2026-09-18 to src/utils/database.py's `week - 1` prior-week join, which
# had already been patched there with the same `ts.week >= 1` guard back
# on 2026-08-29 (see that file) but never had the underlying rows deleted.
# The 186 rows were deleted from the DB directly on 2026-09-18
# (scripts/backfill_team_stats_points_turnovers.py's investigation) --
# this filter is now redundant against the current DB but kept in case a
# future ingest run recreates rows of the same shape before the write-path
# root cause is fixed.

_TEAM_STATS_SQL = """
    SELECT team, season, week, total_yards, passing_yards, rushing_yards,
           drive_success_rate, avg_drive_epa, points_per_drive,
           neutral_pass_rate_oe
    FROM team_stats
    WHERE total_yards IS NOT NULL AND week != 0
"""


def _connect(con: Optional[sqlite3.Connection]) -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH)) if con is None else con


def _lagged_form(
    long_df: pd.DataFrame,
    value_cols: list[str],
    window: int,
    min_prior_games: int,
) -> pd.DataFrame:
    """Per-(team, season, week): season-to-date + last-N-game means of PRIOR
    games, with a prior-season-mean cold-start fallback and an `is_cold_start`
    flag.

    `long_df` must have columns `team`, `season`, `week`, plus each of
    `value_cols`, one row per (team, season, week). `shift(1)` is applied
    before `.expanding()`/`.rolling()` so the value attached to a given
    (team, season, week) uses only games strictly before that week within
    the same season -- never the target game itself.
    """
    long_df = long_df.sort_values(["team", "season", "week"]).reset_index(drop=True)
    grp = long_df.groupby(["team", "season"], group_keys=False)
    out = long_df[["team", "season", "week"]].copy()
    out["n_prior_games"] = grp.cumcount()

    for col in value_cols:
        out[f"{col}_s2d"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
        out[f"{col}_roll{window}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    prior_season = long_df.groupby(["team", "season"])[value_cols].mean().reset_index()
    prior_season["season"] = prior_season["season"] + 1
    prior_season = prior_season.rename(columns={c: f"{c}_prior_season" for c in value_cols})
    out = out.merge(prior_season, on=["team", "season"], how="left")

    out["is_cold_start"] = (out["n_prior_games"] < min_prior_games).astype(int)
    cold = out["is_cold_start"].astype(bool)
    for col in value_cols:
        prior_col = f"{col}_prior_season"
        # Cold start: prefer the prior season's full-season mean; if that is
        # also missing (team's first tracked season -- 2006, or a relocated
        # franchise), leave NaN for the model's own imputer to handle rather
        # than silently zero-filling.
        out.loc[cold, f"{col}_s2d"] = out.loc[cold, f"{col}_s2d"].where(
            out.loc[cold, prior_col].isna(), out.loc[cold, prior_col]
        )
        out.loc[cold, f"{col}_roll{window}"] = out.loc[cold, f"{col}_roll{window}"].where(
            out.loc[cold, prior_col].isna(), out.loc[cold, prior_col]
        )

    drop_cols = [f"{c}_prior_season" for c in value_cols] + ["n_prior_games"]
    return out.drop(columns=drop_cols)


def _append_placeholder_rows(
    long_df: pd.DataFrame,
    team_season_weeks: set[tuple[str, int, int]],
    value_cols: list[str],
) -> pd.DataFrame:
    """Append one all-NaN row per (team, season, week) not already present.

    Used only for `build_prediction_rows`: gives `_lagged_form`'s shift(1)
    a row to compute FROM for a not-yet-played game, without ever
    contributing a real value to the aggregate itself (NaN is dropped by
    `.expanding()`/`.rolling()` before it would be included, and shift(1)
    already excludes the row's own value regardless).
    """
    existing = set(zip(long_df["team"], long_df["season"], long_df["week"]))
    new_rows = [
        {"team": t, "season": s, "week": w, **{c: np.nan for c in value_cols}}
        for (t, s, w) in team_season_weeks
        if (t, s, w) not in existing
    ]
    if not new_rows:
        return long_df
    return pd.concat([long_df, pd.DataFrame(new_rows)], ignore_index=True)


def _team_week_stats(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(_TEAM_STATS_SQL, con)


def _load_games_with_scores(con: sqlite3.Connection) -> pd.DataFrame:
    """Internal-only: completed, non-tie games WITH home_score/away_score kept.

    Never returned or merged directly into the public `rows` frame -- used
    only to derive the points_scored/points_allowed team-week table below.
    labels.load_completed_games() deliberately strips these two columns as a
    leakage guard, so this is a separate, narrowly-scoped query rather than a
    variant of that function.
    """
    df = pd.read_sql(
        """
        SELECT season, week, home_team, away_team, home_score, away_score
        FROM schedule
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_score != away_score
        """,
        con,
    )
    return df


def _team_week_scores(con: sqlite3.Connection) -> pd.DataFrame:
    """Points scored/allowed per (team, season, week), derived from `schedule`
    (not `team_stats` -- see module docstring)."""
    games = _load_games_with_scores(con)
    home = games[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    home["points_scored"] = games["home_score"]
    home["points_allowed"] = games["away_score"]
    away = games[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    away["points_scored"] = games["away_score"]
    away["points_allowed"] = games["home_score"]
    return pd.concat([home, away], ignore_index=True)


def _team_week_wins(all_seasons_games: pd.DataFrame) -> pd.DataFrame:
    home = all_seasons_games[["season", "week", "home_team", "home_win"]].rename(
        columns={"home_team": "team", "home_win": "win"}
    )
    away = all_seasons_games[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    away["win"] = 1 - all_seasons_games["home_win"]
    return pd.concat([home, away], ignore_index=True)


_REST_DAYS_CAP = 20  # bye weeks top out around 14-17; beyond that it's a season
                     # boundary (offseason gap), which isn't a "rest" signal --
                     # clip so it doesn't dominate scaling as a huge outlier.


def _team_rest_days(con: sqlite3.Connection) -> pd.DataFrame:
    """Days since each team's immediately-prior game (any season), per (team, season, week).

    Not run through `_lagged_form`'s shift/rolling machinery -- this is a
    single date difference against the schedule, which is fixed in advance
    and therefore legitimately known pre-kickoff without any lag needed
    (same availability class as `spread_line`/`total_line`).
    """
    games = pd.read_sql(
        "SELECT season, week, home_team, away_team, game_time FROM schedule "
        "WHERE game_time IS NOT NULL AND game_time != ''",
        con,
    )
    games["game_date"] = pd.to_datetime(games["game_time"], errors="coerce")
    games = games.dropna(subset=["game_date"])
    home = games[["season", "week", "home_team", "game_date"]].rename(columns={"home_team": "team"})
    away = games[["season", "week", "away_team", "game_date"]].rename(columns={"away_team": "team"})
    team_week = pd.concat([home, away], ignore_index=True).sort_values(["team", "game_date"])
    team_week["rest_days"] = team_week.groupby("team")["game_date"].diff().dt.days.clip(upper=_REST_DAYS_CAP)
    return team_week[["team", "season", "week", "rest_days"]].reset_index(drop=True)


def _team_elo_pre(
    con: sqlite3.Connection,
    games_override: Optional[pd.DataFrame],
    elo_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Pre-game Elo rating per (team, season, week) -- see elo.py. Always
    fit on the FULL played-game history (ties included, same rationale as
    every other team-form table here: early games in a restricted `seasons`
    range still need correctly-computed prior state).

    For `games_override` (not-yet-played games), the exact (season, week)
    row can't come from `pregame_long_frame()` -- that only has rows for
    games Elo actually processed. Use `current_elo()` instead, which is the
    same rating a played game in that season would have gotten as its
    pre-game value (same season-boundary regression logic), without
    mutating the fitted system's state.

    `elo_params`, if given, overrides `GAME_OUTCOME_MODEL_CONFIG["elo_params"]`
    -- used only by scripts/tune_elo_params.py to rebuild the feature frame
    under different candidate Elo hyperparameters without touching the
    committed default.
    """
    all_games = load_all_completed_games(con)
    elo = EloRatingSystem(**(elo_params or GAME_OUTCOME_MODEL_CONFIG["elo_params"])).fit(all_games)
    long_df = elo.pregame_long_frame()

    if games_override is not None:
        rows = []
        for _, r in games_override.iterrows():
            for side in ("home_team", "away_team"):
                team = r[side]
                rows.append(
                    {
                        "team": team,
                        "season": r["season"],
                        "week": r["week"],
                        "elo_pre": elo.current_elo(team, r["season"]),
                    }
                )
        long_df = pd.concat([long_df, pd.DataFrame(rows)], ignore_index=True)

    return long_df


def _load_weather(con: sqlite3.Connection, seasons: Optional[list[int]]) -> pd.DataFrame:
    params: list = []
    season_filter = ""
    if seasons is not None:
        # See the numpy.int64 SQL-binding note in labels.py -- same fix.
        seasons = [int(s) for s in seasons]
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


def _build_feature_frame(
    seasons: Optional[Iterable[int]],
    con: Optional[sqlite3.Connection],
    include_weather: bool,
    games_override: Optional[pd.DataFrame] = None,
    elo_params: Optional[dict] = None,
    include_market_odds: bool = False,
) -> pd.DataFrame:
    """id columns + feature columns only, NO label -- shared by every target
    (home_win for phase 1, home_margin/game_total for phase 2) AND by
    `build_prediction_rows` (future, not-yet-played games). Callers merge
    their own label column(s) on afterward (or none, for predictions);
    keeping label construction out of this function is what lets multiple
    targets/callers reuse one feature pipeline without risking a label
    leaking into another's feature set.

    `seasons`, if given, restricts the returned games -- but team-form tables
    are always built from the FULL team_stats/schedule history so that early
    games in the requested range still get correctly lagged prior-game data
    rather than an artificially cold start. Ignored if `games_override` is
    given (the caller has already picked its own games, e.g. one week's
    worth of not-yet-played matchups).

    `games_override`, if given, replaces the default "all completed, non-tie
    games" base frame -- used by `build_prediction_rows` to run the exact
    same feature pipeline against SCHEDULED-BUT-UNPLAYED games instead. Must
    have columns season/week/home_team/away_team/spread_line/total_line and
    no score/label columns (a future game has neither).
    """
    owns_con = con is None
    conn = _connect(con)
    try:
        window = GAME_OUTCOME_MODEL_CONFIG["rolling_window_games"]
        min_prior = GAME_OUTCOME_MODEL_CONFIG["min_prior_games_for_form"]

        if games_override is not None:
            games = sanitize_schedule_df(games_override.copy())  # defense in depth
        else:
            # home_win is a LABEL for one specific target, not a feature --
            # drop it immediately so it can never leak into a different
            # target's (e.g. margin/total regression's) feature set. Each
            # public builder below merges its own label(s) back on after
            # this shared frame is built.
            games = load_completed_games(seasons=seasons, con=conn).drop(columns=["home_win"])
            games = sanitize_schedule_df(games)  # defense in depth

        # Full history for lagging (not just the requested `seasons`), same
        # rationale as before: early games in a restricted range still need
        # correctly lagged prior-game data.
        all_games = load_completed_games(con=conn)

        team_week = _team_week_stats(conn)
        score_long = _team_week_scores(conn)
        win_long = _team_week_wins(all_games)

        if games_override is not None:
            # `team_week`/`score_long`/`win_long` only have rows for games
            # that were actually PLAYED -- joining an upcoming game to them
            # by exact (season, week) finds nothing (found 2026-09-18 while
            # wiring up predictions for not-yet-played 2026 games: every
            # home_minus_away_* feature came back NaN for a future week, and
            # the is_cold_start combination below silently read that as "not
            # cold start" instead of correctly flagging it). Appending one
            # placeholder row per team per upcoming game, with NaN stat
            # values, fixes this for free: `_lagged_form`'s shift(1) already
            # excludes the row's own value before aggregating, so the
            # placeholder's NaN never enters the mean -- the placeholder just
            # gives shift(1) a position to compute FROM, correctly yielding
            # the expanding/rolling mean of every real prior game.
            pairs = set()
            for _, r in games.iterrows():
                pairs.add((r["home_team"], r["season"], r["week"]))
                pairs.add((r["away_team"], r["season"], r["week"]))
            team_week = _append_placeholder_rows(team_week, pairs, TEAM_STAT_COLS)
            score_long = _append_placeholder_rows(score_long, pairs, ["points_scored", "points_allowed"])
            win_long = _append_placeholder_rows(win_long, pairs, ["win"])

        team_form = _lagged_form(team_week, TEAM_STAT_COLS, window, min_prior)
        score_form = _lagged_form(score_long, ["points_scored", "points_allowed"], window, min_prior)
        win_form = _lagged_form(win_long, ["win"], window, min_prior).rename(
            columns={
                "win_s2d": "win_pct_s2d",
                f"win_roll{window}": f"win_pct_roll{window}",
            }
        )

        # `_lagged_form` always adds its own "is_cold_start" column, so
        # team_form/score_form/win_form each independently have one. Merging
        # all three without renaming it first would collide three times over
        # (pandas silently suffixes to _x/_y, leaving orphaned junk columns
        # behind rather than raising -- found 2026-09-18 while adding the
        # margin/total target: `is_cold_start` matches as a *substring* of
        # its own mangled name, e.g. "home_is_cold_start_x", so
        # audit_feature_availability let the leftover junk column through
        # silently instead of catching it). Rename to a source-specific name
        # per table before merging so there is never a collision, then
        # combine explicitly below.
        team_form = team_form.rename(columns={"is_cold_start": "is_cold_start__team"})
        score_form = score_form.rename(columns={"is_cold_start": "is_cold_start__score"})
        win_form = win_form.rename(columns={"is_cold_start": "is_cold_start__win"})

        form_cols = [c for c in team_form.columns if c not in ("team", "season", "week")]
        score_cols = [c for c in score_form.columns if c not in ("team", "season", "week")]
        win_cols = [c for c in win_form.columns if c not in ("team", "season", "week")]
        cold_start_cols = ["is_cold_start__team", "is_cold_start__score", "is_cold_start__win"]

        def _side(df: pd.DataFrame, cols: list[str], side: str) -> pd.DataFrame:
            team_col = f"{side}_team"
            return df.rename(columns={"team": team_col, **{c: f"{side}_{c}" for c in cols}})

        rest_form = _team_rest_days(conn)
        rest_cols = ["rest_days"]

        elo_form = _team_elo_pre(conn, games_override, elo_params)
        elo_cols = ["elo_pre"]

        rows = games
        for df, cols in (
            (team_form, form_cols),
            (score_form, score_cols),
            (win_form, win_cols),
            (rest_form, rest_cols),
            (elo_form, elo_cols),
        ):
            rows = rows.merge(_side(df, cols, "home"), on=["season", "week", "home_team"], how="left")
            rows = rows.merge(_side(df, cols, "away"), on=["season", "week", "away_team"], how="left")

        # Combine into one flag rather than differencing three separate 0/1
        # columns -- never treat cold-start as a stat to difference.
        diff_cols = [
            c for c in (form_cols + score_cols + win_cols + rest_cols + elo_cols) if c not in cold_start_cols
        ]

        for col in diff_cols:
            rows[f"home_minus_away_{col}"] = rows[f"home_{col}"] - rows[f"away_{col}"]

        home_cold_cols = [f"home_{c}" for c in cold_start_cols]
        away_cold_cols = [f"away_{c}" for c in cold_start_cols]

        def _any_cold_or_missing(cols: list[str]) -> pd.Series:
            sub = rows[cols]
            # A missing cold-start flag (no matching row at all -- shouldn't
            # happen after the placeholder-row fix above, but this is the
            # bug that fix was patching: NaN silently read as "not cold
            # start" instead of the safer "treat as cold start") counts the
            # same as an explicit 1.
            return (sub == 1).any(axis=1) | sub.isna().any(axis=1)

        rows["is_cold_start"] = (_any_cold_or_missing(home_cold_cols) | _any_cold_or_missing(away_cold_cols)).astype(int)
        rows["game_week"] = rows["week"].astype(int)

        # Drop only the intermediate per-side columns we just merged in and
        # differenced -- NOT the "home_minus_away_*" diff columns themselves
        # or "home_win" (both also happen to start with "home_").
        all_side_cols = list(dict.fromkeys(form_cols + score_cols + win_cols + rest_cols + elo_cols))
        intermediate_cols = [f"home_{c}" for c in all_side_cols] + [f"away_{c}" for c in all_side_cols]
        rows = rows.drop(columns=intermediate_cols)

        if include_weather:
            weather = _load_weather(conn, list(seasons) if seasons is not None else None)
            rows = rows.merge(
                weather, on=["season", "week", "home_team", "away_team"], how="left"
            )

        if include_market_odds:
            # Opt-in only (default False) -- game_odds only covers 2020+,
            # see market_odds.py's module docstring. Using this means
            # accepting a much shorter usable training history than the
            # committed 2006+ models; never enabled by default.
            market_odds = load_market_odds_features(conn)
            rows = rows.merge(
                market_odds, on=["season", "week", "home_team", "away_team"], how="left"
            )
    finally:
        if owns_con:
            conn.close()

    id_cols = ["season", "week", "home_team", "away_team"]
    feature_cols = [c for c in rows.columns if c not in id_cols]

    unclassified = audit_feature_availability(feature_cols)
    if unclassified:
        raise ValueError(
            "Unclassified game-outcome feature columns (add to "
            f"src/utils/leakage.py's FEATURE_AVAILABILITY): {sorted(unclassified)}"
        )

    return rows[id_cols + feature_cols].reset_index(drop=True)


def build_game_outcome_rows(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
    include_weather: bool = True,
    elo_params: Optional[dict] = None,
    include_market_odds: bool = False,
) -> pd.DataFrame:
    """One row per completed, non-tie game: id columns + home_win label + feature columns.

    `elo_params`, if given, overrides the committed default Elo hyperparameters
    -- see `_team_elo_pre`; used by scripts/tune_elo_params.py only.

    `include_market_odds`, if True, adds multi-bookmaker consensus/
    disagreement features from `game_odds` (2020+ only -- see market_odds.py).
    Defaults to False so the committed 2006+ models are unaffected; only
    scripts/train_market_odds_model.py turns this on.
    """
    frame = _build_feature_frame(
        seasons, con, include_weather, elo_params=elo_params, include_market_odds=include_market_odds
    )

    owns_con = con is None
    conn = _connect(con)
    try:
        labels = load_completed_games(seasons=seasons, con=conn)[
            ["season", "week", "home_team", "away_team", "home_win"]
        ]
    finally:
        if owns_con:
            conn.close()

    rows = frame.merge(labels, on=["season", "week", "home_team", "away_team"], how="inner")
    id_cols = ["season", "week", "home_team", "away_team"]
    feature_cols = [c for c in rows.columns if c not in id_cols + ["home_win"]]
    return rows[id_cols + ["home_win"] + feature_cols].reset_index(drop=True)


def build_margin_total_rows(
    seasons: Optional[Iterable[int]] = None,
    con: Optional[sqlite3.Connection] = None,
    include_weather: bool = True,
    elo_params: Optional[dict] = None,
    include_market_odds: bool = False,
) -> pd.DataFrame:
    """One row per completed, non-tie game (same population as
    `build_game_outcome_rows` -- the underlying feature frame already
    excludes ties, so keeping them out here too keeps the two targets'
    game populations identical for apples-to-apples comparison): id
    columns + `home_margin`/`game_total` labels + the same feature columns
    as `build_game_outcome_rows`.

    `elo_params`, if given, overrides the committed default Elo hyperparameters.
    `include_market_odds`, if True, adds `game_odds`-derived features (2020+
    only -- see market_odds.py and build_game_outcome_rows's docstring).
    """
    frame = _build_feature_frame(
        seasons, con, include_weather, elo_params=elo_params, include_market_odds=include_market_odds
    )

    owns_con = con is None
    conn = _connect(con)
    try:
        labels = load_margin_total_labels(seasons=seasons, con=conn)[
            ["season", "week", "home_team", "away_team", "home_margin", "game_total"]
        ]
    finally:
        if owns_con:
            conn.close()

    rows = frame.merge(labels, on=["season", "week", "home_team", "away_team"], how="inner")
    id_cols = ["season", "week", "home_team", "away_team"]
    label_cols = ["home_margin", "game_total"]
    feature_cols = [c for c in rows.columns if c not in id_cols + label_cols]
    return rows[id_cols + label_cols + feature_cols].reset_index(drop=True)


def build_prediction_rows(
    season: int,
    week: Optional[int] = None,
    con: Optional[sqlite3.Connection] = None,
    include_weather: bool = True,
    include_market_odds: bool = False,
) -> pd.DataFrame:
    """One row per SCHEDULED-BUT-NOT-YET-PLAYED game: id columns + the same
    feature columns as `build_game_outcome_rows`/`build_margin_total_rows`,
    but no label (the game hasn't happened).

    Runs the identical team-form/lag pipeline as the training builders --
    "prior games" for a future week just means "every completed game so far,"
    which `_lagged_form` already handles correctly since it only ever looks
    at games strictly before the target (season, week), never at the target
    row itself. A model trained on `build_game_outcome_rows`'s output can be
    called directly on this frame's `feature_columns(...)` subset -- same
    column names, same dtypes, same construction.
    """
    owns_con = con is None
    conn = _connect(con)
    try:
        upcoming = load_upcoming_games(season, week=week, con=conn)
        if upcoming.empty:
            return upcoming
        return _build_feature_frame(
            seasons=None,
            con=conn,
            include_weather=include_weather,
            games_override=upcoming,
            include_market_odds=include_market_odds,
        )
    finally:
        if owns_con:
            conn.close()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Works for either target's frame -- id/label columns from both
    `build_game_outcome_rows` (home_win) and `build_margin_total_rows`
    (home_margin, game_total) are excluded regardless of which one produced
    `df`, since each frame only ever has its own label columns present."""
    id_and_label = {"season", "week", "home_team", "away_team", "home_win", "home_margin", "game_total"}
    return [c for c in df.columns if c not in id_and_label]

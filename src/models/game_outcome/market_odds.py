"""Multi-bookmaker market-odds features from `game_odds` -- explicitly
deferred at phase-1 planning ("Not `game_odds` -- that table is 2020+ only
and would truncate ~14 seasons of usable training history for a
richer-but-unused multi-bookmaker signal. Defer `game_odds` to a later
phase."). Picked up here as an OPT-IN feature set (`include_market_odds`
in features.py), never the default, so the long-history 2006+ models are
untouched.

`game_odds` has two data-quality wrinkles worth knowing before reading the
code below:

1. `season`/`week` are frequently NULL or wrong on the raw rows (~1% of
   rows have NULL season, and week is NULL for ~2.7% of rows) -- these
   columns are NOT used for anything here. Every game is instead
   re-identified by matching `game_odds.event_id` to `schedule` via
   (home_team, away_team, commence_time vs. game_time), which is reliable.

2. The same (home_team, away_team) pair can appear under TWO different
   `event_id`s within a season -- one is the real game, the other is an
   early placeholder listing from the odds API with a commence_time that
   can be MONTHS off from the actual scheduled kickoff (confirmed
   2026-09-19: a CLE @ NYJ 2023-08-04 placeholder event alongside the real
   2023-12-29 event for what was actually a Week 17 game). Resolved by
   matching each schedule game to whichever event_id's commence_time is
   closest to schedule.game_time, within a tolerance -- the placeholder is
   many weeks off and gets rejected by the tolerance check.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

# Games are always played on the same calendar day as scheduled, but
# commence_time is UTC and game_time is a US-local date, so a Sunday
# evening kickoff can show up as UTC "Monday" -- 1 day of slack covers
# every real timezone case; the known placeholder-event bug (~4 months off)
# is rejected by an enormous margin either way.
_EVENT_MATCH_TOLERANCE_DAYS = 1.5


def _match_events_to_schedule(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per schedule game with its matched `game_odds.event_id`
    (NaN if no odds event was found within tolerance) -- see module
    docstring for why this can't just be a (home_team, away_team) join."""
    schedule = pd.read_sql(
        "SELECT season, week, home_team, away_team, game_time FROM schedule "
        "WHERE game_time IS NOT NULL AND game_time != ''",
        con,
    )
    events = pd.read_sql(
        "SELECT DISTINCT event_id, home_team, away_team, commence_time FROM game_odds "
        "WHERE event_id IS NOT NULL AND commence_time IS NOT NULL",
        con,
    )
    if schedule.empty or events.empty:
        return pd.DataFrame(columns=["season", "week", "home_team", "away_team", "event_id"])

    schedule["game_dt"] = pd.to_datetime(schedule["game_time"], errors="coerce", utc=True)
    events["commence_dt"] = pd.to_datetime(events["commence_time"], errors="coerce", utc=True)

    merged = schedule.merge(events, on=["home_team", "away_team"], how="inner")
    merged["day_diff"] = (merged["commence_dt"] - merged["game_dt"]).abs().dt.total_seconds() / 86400.0
    merged = merged[merged["day_diff"] <= _EVENT_MATCH_TOLERANCE_DAYS]
    # Multiple odds events can still pass the tolerance for a rare same-week
    # doubleheader-adjacent case -- keep only the closest match per game.
    merged = merged.sort_values("day_diff").drop_duplicates(
        subset=["season", "week", "home_team", "away_team"], keep="first"
    )
    return merged[["season", "week", "home_team", "away_team", "event_id"]].reset_index(drop=True)


def _closing_odds(con: sqlite3.Connection) -> pd.DataFrame:
    """Per (event_id, bookmaker, market): the LAST fetch strictly before
    that event's commence_time -- i.e. the closing line. `fetched_at >=
    commence_time` rows are dropped entirely rather than risked: a fetch
    taken during or after the game could reflect in-game line movement
    (or a postgame settlement snapshot), which would leak the outcome."""
    df = pd.read_sql(
        "SELECT event_id, commence_time, bookmaker, market, home_price, away_price, "
        "home_point, fetched_at FROM game_odds",
        con,
    )
    df["commence_dt"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    df["fetched_dt"] = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
    df = df[df["fetched_dt"] < df["commence_dt"]]
    df = df.sort_values("fetched_dt").drop_duplicates(subset=["event_id", "bookmaker", "market"], keep="last")
    return df


def _american_odds_to_implied_prob(price: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(price < 0, -price / (-price + 100.0), 100.0 / (price + 100.0))


def _event_market_features(closing: pd.DataFrame) -> pd.DataFrame:
    """Per event_id: cross-bookmaker median (consensus) and std (disagreement)
    for the spread and total, plus a vig-removed moneyline implied
    home-win probability. `market_n_books_*` is a coverage/liquidity proxy
    (deep, well-covered games get more books; std is meaningless/NaN off a
    single book, which the model's own imputer handles like any other
    missing value -- not silently zero-filled).

    SIGN CONVENTION (verified 2026-09-19): `game_odds.home_point` for the
    "spreads" market uses the standard American-odds convention -- NEGATIVE
    means the home team is favored (e.g. home_point=-7.0 means home
    favored by 7). This is the OPPOSITE of `schedule.spread_line`, whose
    convention (positive = home favored) was verified earlier in this
    project. Confirmed by correlating the two: -0.976 (not exactly -1
    since they're different bookmakers'/consensus quotes of the same
    underlying market, just opposite-signed). Negated below so
    `market_spread_median` matches `schedule.spread_line`'s convention --
    getting this wrong would have silently taught a model the reverse of
    what "the market favors the home team by N" means.
    """
    spreads = closing[(closing["market"] == "spreads") & closing["home_point"].notna()].copy()
    spreads["home_point"] = -spreads["home_point"]
    totals = closing[(closing["market"] == "totals") & closing["home_point"].notna()]
    h2h = closing[(closing["market"] == "h2h") & closing["home_price"].notna() & closing["away_price"].notna()].copy()

    spread_agg = spreads.groupby("event_id")["home_point"].agg(
        market_spread_median="median", market_spread_std="std", market_n_books_spread="count"
    )
    total_agg = totals.groupby("event_id")["home_point"].agg(
        market_total_median="median", market_total_std="std", market_n_books_total="count"
    )

    if h2h.empty:
        h2h_agg = pd.DataFrame(columns=["market_moneyline_median", "market_moneyline_std", "market_n_books_moneyline"])
    else:
        home_implied = _american_odds_to_implied_prob(h2h["home_price"].to_numpy())
        away_implied = _american_odds_to_implied_prob(h2h["away_price"].to_numpy())
        # Remove the vig (both sides' implied probabilities sum to > 1 by the
        # book's overround) by renormalizing -- otherwise "home is favored"
        # is conflated with "this book just has a bigger overround."
        h2h["home_implied_novig"] = home_implied / (home_implied + away_implied)
        h2h_agg = h2h.groupby("event_id")["home_implied_novig"].agg(
            market_moneyline_median="median", market_moneyline_std="std", market_n_books_moneyline="count"
        )

    out = spread_agg.join(total_agg, how="outer").join(h2h_agg, how="outer")
    return out.reset_index()


def load_market_odds_features(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per (season, week, home_team, away_team) with consensus/
    disagreement market-odds features -- covers every `schedule` game
    (played or not) that has a matched `game_odds` event, 2020+ only (see
    module docstring). Callers merge this onto their feature frame the
    same way `_load_weather` is merged in features.py -- a plain
    (season, week, home_team, away_team) left join, no home/away diffing
    needed since these are already consensus values, not per-team stats.
    """
    matched = _match_events_to_schedule(con)
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "season", "week", "home_team", "away_team",
                "market_spread_median", "market_spread_std", "market_n_books_spread",
                "market_total_median", "market_total_std", "market_n_books_total",
                "market_moneyline_median", "market_moneyline_std", "market_n_books_moneyline",
            ]
        )
    closing = _closing_odds(con)
    event_features = _event_market_features(closing)
    return matched.merge(event_features, on="event_id", how="left").drop(columns=["event_id"])

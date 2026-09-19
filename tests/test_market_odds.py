"""Tests for src/models/game_outcome/market_odds.py.

Covers the two real data-quality wrinkles found in `game_odds` (see that
module's docstring): the sign-convention flip needed to match
`schedule.spread_line`, and the placeholder-event matching bug, plus the
closing-line (pre-kickoff-only) filter and its wiring into features.py.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.models.game_outcome.market_odds import load_market_odds_features
from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "market_odds_test.db")


def _insert_schedule(db, season, week, home, away, game_time):
    db.insert_schedule(
        {
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "game_time": game_time,
            "home_score": 24,
            "away_score": 20,
        }
    )


def _insert_odds(con, event_id, home, away, commence_time, bookmaker, market, home_price=None,
                  away_price=None, home_point=None, fetched_at="2024-01-01T00:00:00Z"):
    con.execute(
        "INSERT INTO game_odds (event_id, home_team, away_team, commence_time, bookmaker, market, "
        "home_price, away_price, home_point, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (event_id, home, away, commence_time, bookmaker, market, home_price, away_price, home_point, fetched_at),
    )


def test_sign_convention_flipped_to_match_spread_line(db):
    """game_odds.home_point uses the American-odds convention (negative =
    home favored); market_spread_median must be POSITIVE for a favored
    home team to match schedule.spread_line's convention."""
    _insert_schedule(db, 2024, 1, "AAA", "BBB", "2024-09-08")
    con = sqlite3.connect(str(db.db_path))
    _insert_odds(
        con, "evt1", "AAA", "BBB", "2024-09-08T17:00:00Z", "draftkings", "spreads",
        home_price=-110, away_price=-110, home_point=-7.0,  # home favored by 7 (American convention)
        fetched_at="2024-09-07T00:00:00Z",
    )
    con.commit()
    features = load_market_odds_features(con)
    row = features[(features["season"] == 2024) & (features["home_team"] == "AAA")].iloc[0]
    assert row["market_spread_median"] == pytest.approx(7.0)
    con.close()


def test_placeholder_event_rejected_by_date_tolerance(db):
    """A second event_id for the same team pair with a commence_time far
    from the actual schedule date (the real bug found 2026-09-19) must be
    rejected -- only the near-date event's odds should be used."""
    _insert_schedule(db, 2023, 17, "CLE", "NYJ", "2023-12-28")
    con = sqlite3.connect(str(db.db_path))
    # Placeholder: months off, should be rejected.
    _insert_odds(con, "placeholder", "CLE", "NYJ", "2023-08-04T00:01:00Z", "draftkings", "spreads",
                 home_point=-99.0, fetched_at="2023-08-01T00:00:00Z")
    # Real event: 1 day off due to UTC/local timezone, should be matched.
    _insert_odds(con, "real", "CLE", "NYJ", "2023-12-29T01:15:00Z", "draftkings", "spreads",
                 home_point=-3.0, fetched_at="2023-12-28T12:00:00Z")
    con.commit()
    features = load_market_odds_features(con)
    row = features[(features["season"] == 2023) & (features["week"] == 17)].iloc[0]
    assert row["market_spread_median"] == pytest.approx(3.0)  # from "real", not -99 -> 99 from "placeholder"
    con.close()


def test_in_game_or_postgame_fetch_excluded_from_closing_line(db):
    """A fetch taken AT OR AFTER commence_time must never be used -- it
    could reflect in-game line movement or a settlement snapshot, which
    would leak game-state information into a 'pre-kickoff' feature."""
    _insert_schedule(db, 2024, 1, "AAA", "BBB", "2024-09-08")
    con = sqlite3.connect(str(db.db_path))
    _insert_odds(con, "evt1", "AAA", "BBB", "2024-09-08T17:00:00Z", "draftkings", "spreads",
                 home_point=-3.0, fetched_at="2024-09-08T16:00:00Z")  # legitimate pre-kickoff fetch
    _insert_odds(con, "evt1", "AAA", "BBB", "2024-09-08T17:00:00Z", "draftkings", "spreads",
                 home_point=-50.0, fetched_at="2024-09-08T18:00:00Z")  # AFTER kickoff -- must be dropped
    con.commit()
    features = load_market_odds_features(con)
    row = features[(features["season"] == 2024) & (features["home_team"] == "AAA")].iloc[0]
    assert row["market_spread_median"] == pytest.approx(3.0)
    con.close()


def test_multiple_bookmakers_produce_median_and_std(db):
    _insert_schedule(db, 2024, 1, "AAA", "BBB", "2024-09-08")
    con = sqlite3.connect(str(db.db_path))
    for book, point in (("draftkings", -3.0), ("fanduel", -4.0), ("betmgm", -2.0)):
        _insert_odds(con, "evt1", "AAA", "BBB", "2024-09-08T17:00:00Z", book, "spreads",
                     home_point=point, fetched_at="2024-09-07T00:00:00Z")
    con.commit()
    features = load_market_odds_features(con)
    row = features[(features["season"] == 2024) & (features["home_team"] == "AAA")].iloc[0]
    assert row["market_n_books_spread"] == 3
    assert row["market_spread_median"] == pytest.approx(3.0)  # median of [3,4,2]
    assert row["market_spread_std"] > 0
    con.close()


def test_no_odds_match_returns_nan_not_zero(db):
    """A game with no matching game_odds event should get NaN market_*
    features (for the model's own imputer to handle), never a silent 0 --
    0 would look like 'pick'em, no disagreement' rather than 'no data'."""
    _insert_schedule(db, 2024, 1, "AAA", "BBB", "2024-09-08")
    con = sqlite3.connect(str(db.db_path))
    features = load_market_odds_features(con)
    assert features.empty or features["market_spread_median"].isna().all()
    con.close()


def test_wired_into_build_game_outcome_rows_when_opted_in(db):
    from src.models.game_outcome.features import build_game_outcome_rows, feature_columns

    _insert_schedule(db, 2024, 1, "AAA", "BBB", "2024-09-08")
    con = sqlite3.connect(str(db.db_path))
    con.execute(
        "UPDATE schedule SET spread_line=-3.0, total_line=44.0 WHERE season=2024 AND home_team='AAA'"
    )
    _insert_odds(con, "evt1", "AAA", "BBB", "2024-09-08T17:00:00Z", "draftkings", "spreads",
                 home_point=-3.0, fetched_at="2024-09-07T00:00:00Z")
    con.commit()

    default_rows = build_game_outcome_rows(seasons=[2024], con=con, include_weather=False)
    assert not any(c.startswith("market_") for c in default_rows.columns)

    opted_in = build_game_outcome_rows(
        seasons=[2024], con=con, include_weather=False, include_market_odds=True
    )
    assert "market_spread_median" in feature_columns(opted_in)
    con.close()

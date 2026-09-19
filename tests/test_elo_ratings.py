"""Tests for src/models/game_outcome/elo.py -- the sequential Elo power
rating, and its wiring into features.py as `home_minus_away_elo_pre`.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.models.game_outcome.elo import EloRatingSystem


def _games(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team", "home_score", "away_score"])


def test_first_game_pre_rating_is_initial_for_both_teams():
    games = _games([(2020, 1, "AAA", "BBB", 20, 10)])
    elo = EloRatingSystem(initial=1500.0).fit(games)
    row = elo.history_.iloc[0]
    assert row["home_elo_pre"] == 1500.0
    assert row["away_elo_pre"] == 1500.0


def test_winner_rating_increases_and_loser_decreases():
    games = _games([(2020, 1, "AAA", "BBB", 20, 10), (2020, 2, "AAA", "CCC", 10, 10)])
    elo = EloRatingSystem(initial=1500.0, k=20.0, home_advantage=0.0, mov_multiplier=False).fit(games)
    # AAA won game 1 as home team -> its rating going into game 2 (pre) is above initial.
    game2_pre = elo.history_.iloc[1]["home_elo_pre"]
    assert game2_pre > 1500.0


def test_pregame_rating_never_depends_on_that_games_own_score():
    """Leakage sentinel: change game 2's own score and confirm its own
    pre-game ratings are unaffected (they're fixed before that game's result
    is known -- only LATER games should ever see a different rating)."""
    base = _games([(2020, 1, "AAA", "BBB", 20, 10), (2020, 2, "AAA", "CCC", 14, 14)])
    mutated = _games([(2020, 1, "AAA", "BBB", 20, 10), (2020, 2, "AAA", "CCC", 45, 0)])

    elo_base = EloRatingSystem().fit(base)
    elo_mutated = EloRatingSystem().fit(mutated)

    pre_base = elo_base.history_.iloc[1][["home_elo_pre", "away_elo_pre"]]
    pre_mutated = elo_mutated.history_.iloc[1][["home_elo_pre", "away_elo_pre"]]
    pd.testing.assert_series_equal(pre_base, pre_mutated)


def test_tie_moves_both_ratings_toward_each_other():
    games = _games([(2020, 1, "AAA", "BBB", 14, 14), (2020, 2, "AAA", "CCC", 10, 10)])
    elo = EloRatingSystem(home_advantage=0.0).fit(games)
    game2_home_pre = elo.history_.iloc[1]["home_elo_pre"]
    # AAA tied as home favorite-by-home-field-only (equal pre-ratings, small
    # home_advantage=0 here) -- a tie against an equal opponent should leave
    # the rating essentially unchanged, not push it decisively either way.
    assert game2_home_pre == pytest.approx(1500.0, abs=1e-9)


def test_season_boundary_regresses_toward_initial():
    season_1_only = _games([(2020, 1, "AAA", "BBB", 40, 0)])
    post_2020 = EloRatingSystem(initial=1500.0, season_regression=0.5).fit(season_1_only).ratings_["AAA"]

    games = _games(
        [
            (2020, 1, "AAA", "BBB", 40, 0),  # AAA wins big, rating rises well above 1500
            (2021, 1, "AAA", "CCC", 20, 20),  # first 2021 game -- pre-rating should be regressed
        ]
    )
    elo = EloRatingSystem(initial=1500.0, season_regression=0.5).fit(games)
    pre_2021 = elo.history_.iloc[1]["home_elo_pre"]
    assert post_2020 > 1500.0
    assert pre_2021 == pytest.approx(1500.0 + 0.5 * (post_2020 - 1500.0))
    assert 1500.0 < pre_2021 < post_2020


def test_current_elo_matches_regression_for_future_season_without_mutating_state():
    games = _games([(2020, 1, "AAA", "BBB", 40, 0)])
    elo = EloRatingSystem(initial=1500.0, season_regression=0.5).fit(games)
    rating_before = elo.ratings_["AAA"]
    future = elo.current_elo("AAA", as_of_season=2021)
    assert future == pytest.approx(1500.0 + 0.5 * (rating_before - 1500.0))
    # current_elo must not mutate state -- calling it twice gives the same answer.
    assert elo.current_elo("AAA", as_of_season=2021) == future
    assert elo.ratings_["AAA"] == rating_before


def test_current_elo_unseen_team_returns_initial():
    elo = EloRatingSystem(initial=1500.0).fit(_games([(2020, 1, "AAA", "BBB", 10, 7)]))
    assert elo.current_elo("ZZZ", as_of_season=2020) == 1500.0


def test_pregame_long_frame_has_one_row_per_team_per_played_game():
    games = _games([(2020, 1, "AAA", "BBB", 20, 10), (2020, 2, "AAA", "CCC", 14, 21)])
    elo = EloRatingSystem().fit(games)
    long_df = elo.pregame_long_frame()
    assert len(long_df) == 4
    assert set(long_df.columns) == {"season", "week", "team", "elo_pre"}
    assert set(long_df["team"]) == {"AAA", "BBB", "CCC"}


def test_home_minus_away_elo_feature_builds_and_is_leakage_classified(tmp_path):
    """End-to-end through features.py against a tiny synthetic DB -- mirrors
    the fixture pattern in test_game_outcome_leakage.py."""
    from src.models.game_outcome import features as features_mod

    db_path = tmp_path / "elo_test.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        "CREATE TABLE schedule (season INT, week INT, home_team TEXT, away_team TEXT, "
        "home_score INT, away_score INT, spread_line REAL, total_line REAL, game_time TEXT)"
    )
    con.execute(
        "CREATE TABLE team_stats (team TEXT, season INT, week INT, total_yards REAL, "
        "passing_yards REAL, rushing_yards REAL, drive_success_rate REAL, avg_drive_epa REAL, "
        "points_per_drive REAL, neutral_pass_rate_oe REAL)"
    )
    con.execute("CREATE TABLE game_weather (season INT, week INT, home_team TEXT, away_team TEXT, "
                "is_dome INT, temp_f REAL, wind_mph REAL, precip_mm REAL)")
    games = [
        (2020, 1, "AAA", "BBB", 20, 10, -3.0, 44.0, "2020-09-10"),
        (2020, 2, "AAA", "CCC", 14, 21, 1.0, 41.0, "2020-09-17"),
        (2020, 3, "BBB", "CCC", 17, 17, 2.0, 40.0, "2020-09-24"),
    ]
    con.executemany(
        "INSERT INTO schedule VALUES (?,?,?,?,?,?,?,?,?)", games
    )
    for season, week, home, away, *_ in games:
        for team in (home, away):
            con.execute(
                "INSERT INTO team_stats VALUES (?,?,?,?,?,?,?,?,?,?)",
                (team, season, week, 300.0, 200.0, 100.0, 0.4, 0.05, 2.0, 0.0),
            )
    con.commit()

    df = features_mod.build_game_outcome_rows(con=con, include_weather=True)
    assert "home_minus_away_elo_pre" in df.columns
    assert df["home_minus_away_elo_pre"].notna().all()
    con.close()

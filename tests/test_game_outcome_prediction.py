"""Serving-path invariants: predicting a scheduled-but-unplayed game must
produce the same feature schema as training, with real (non-NaN)
team-form values for a team with completed prior games -- not the
silent-NaN join failure found 2026-09-18 (a future week has no row in
team_stats/score_long/win_long to join against by exact week, so every
home_minus_away_* feature came back NaN and is_cold_start read that as
"not cold start" instead of correctly flagging it).
"""
import sqlite3

from src.models.game_outcome.features import (
    build_game_outcome_rows,
    build_prediction_rows,
    feature_columns,
)
from src.utils.database import DatabaseManager
from src.utils.leakage import audit_feature_availability
from tests.test_game_outcome_leakage import SEASON, _seed_db


def _seed_with_upcoming_week(db: DatabaseManager, n_played_weeks: int, upcoming_week: int):
    _seed_db(db, n_weeks=n_played_weeks, sentinel_week=-1)  # -1: no sentinel needed here
    db.insert_schedule(
        {
            "season": SEASON,
            "week": upcoming_week,
            "home_team": "AAA",
            "away_team": "BBB",
            "spread_line": 2.0,
            "total_line": 44.0,
            # home_score/away_score omitted -> NULL -> "not yet played"
        }
    )


def test_prediction_frame_has_same_feature_schema_as_training(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_with_upcoming_week(db, n_played_weeks=4, upcoming_week=5)

    con = sqlite3.connect(str(db.db_path))
    try:
        train = build_game_outcome_rows(seasons=[SEASON], con=con, include_weather=False)
        upcoming = build_prediction_rows(SEASON, week=5, con=con, include_weather=False)
    finally:
        con.close()

    assert set(feature_columns(train)) == set(feature_columns(upcoming))


def test_prediction_row_has_no_label_column(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_with_upcoming_week(db, n_played_weeks=4, upcoming_week=5)

    con = sqlite3.connect(str(db.db_path))
    try:
        upcoming = build_prediction_rows(SEASON, week=5, con=con, include_weather=False)
    finally:
        con.close()

    assert "home_win" not in upcoming.columns
    assert {"home_score", "away_score"}.isdisjoint(upcoming.columns)


def test_team_form_features_are_not_nan_for_a_team_with_played_games(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_with_upcoming_week(db, n_played_weeks=4, upcoming_week=5)

    con = sqlite3.connect(str(db.db_path))
    try:
        upcoming = build_prediction_rows(SEASON, week=5, con=con, include_weather=False)
    finally:
        con.close()

    assert len(upcoming) == 1
    row = upcoming.iloc[0]
    # AAA/BBB both have 4 played games by week 5 -- team-form features must
    # reflect that, not silently be NaN because week 5 has no team_stats row.
    assert row["home_minus_away_total_yards_s2d"] == row["home_minus_away_total_yards_s2d"]  # not NaN
    assert row["is_cold_start"] == 0  # 4 prior games clears min_prior_games_for_form=3


def test_all_prediction_feature_columns_are_classified(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_with_upcoming_week(db, n_played_weeks=4, upcoming_week=5)

    con = sqlite3.connect(str(db.db_path))
    try:
        upcoming = build_prediction_rows(SEASON, week=5, con=con, include_weather=False)
    finally:
        con.close()

    assert audit_feature_availability(feature_columns(upcoming)) == []


def test_no_upcoming_games_returns_empty_frame_not_an_error(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_db(db, n_weeks=4, sentinel_week=-1)  # every seeded game is already "played"

    con = sqlite3.connect(str(db.db_path))
    try:
        upcoming = build_prediction_rows(SEASON, week=99, con=con, include_weather=False)
    finally:
        con.close()

    assert upcoming.empty

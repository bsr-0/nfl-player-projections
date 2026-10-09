"""Opponent-strength lag-1 features must exist for a game not yet played.

`opp_fpts_allowed_s2d_lag1` and `opp_fpts_allowed_dvoa_adjusted_lag1` were
joined on (opponent, season, week) in tables that hold a row only after the
game. Live serving therefore defaulted both to 0.0 on every row, while
training and replays (target week already played) had real values.
"""
import numpy as np
import pandas as pd
import pytest

import src.utils.database as db_mod
from src.features.feature_engineering import FeatureEngineer, _with_target_week_rows
from src.utils.database import DatabaseManager

TEAMS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    return DatabaseManager(db_path=path)


def _seed(db, weeks, byes=()):
    """Defense allows 10*week to every position; offense produces 5; AAA always plays BBB."""
    with db._get_connection() as conn:
        for week in weeks:
            for team, opp in (("AAA", "BBB"), ("BBB", "AAA"), ("CCC", "AAA")):
                if (team, week) in byes:
                    continue
                conn.execute("INSERT INTO team_defense_stats (team, season, week, fantasy_points_allowed_qb, "
                             "fantasy_points_allowed_rb, fantasy_points_allowed_wr, fantasy_points_allowed_te) "
                             "VALUES (?,?,?,?,?,?,?)", (team, 2026, week, *([10.0 * week] * 4)))
                conn.execute("INSERT INTO team_offense_stats (team, season, week, fantasy_points_produced_qb, "
                             "fantasy_points_produced_rb, fantasy_points_produced_wr, fantasy_points_produced_te) "
                             "VALUES (?,?,?,?,?,?,?)", (team, 2026, week, *([5.0] * 4)))
                conn.execute("INSERT INTO team_stats (team, season, week, opponent) VALUES (?,?,?,?)",
                             (team, 2026, week, opp))
        conn.commit()


def _rows(week, opponent="BBB"):
    return pd.DataFrame({"team": ["AAA"], "opponent": [opponent], "season": [2026], "week": [week],
                         "position": ["WR"]})


def _s2d(df):
    return FeatureEngineer()._add_opp_fpts_allowed_s2d_lag1(df).opp_fpts_allowed_s2d_lag1.iloc[0]


def _dvoa(df):
    return FeatureEngineer()._add_opp_fpts_allowed_dvoa_adjusted_lag1(df).opp_fpts_allowed_dvoa_adjusted_lag1.iloc[0]


def test_an_unplayed_week_gets_the_mean_of_every_earlier_week(db):
    _seed(db, [1, 2, 3, 4])
    assert _s2d(_rows(5)) == pytest.approx(25.0)       # mean(10, 20, 30, 40)


def test_an_unplayed_week_gets_the_mean_of_earlier_residuals(db):
    _seed(db, [1, 2, 3, 4])
    # BBB allowed 10w; its opponent AAA's expected output is 5 from week 2 on
    # (week 1 has no history, so its residual is NaN and is skipped).
    assert _dvoa(_rows(5)) == pytest.approx(np.mean([20 - 5, 30 - 5, 40 - 5]))


def test_a_played_week_is_unchanged_by_the_fix(db):
    _seed(db, [1, 2, 3, 4, 5])
    assert _s2d(_rows(5)) == pytest.approx(25.0)        # row exists: mean(10..40), not week 5's own 50
    assert _s2d(_rows(3)) == pytest.approx(15.0)
    assert _dvoa(_rows(5)) == pytest.approx(np.mean([15, 25, 35]))


def test_the_value_does_not_depend_on_the_games_that_follow(db):
    _seed(db, [1, 2, 3, 4])
    before = _s2d(_rows(5))
    _seed(db, [5, 6])
    assert _s2d(_rows(5)) == pytest.approx(before)


def test_a_bye_in_the_previous_week_still_resolves(db):
    _seed(db, [1, 2, 3, 4], byes={("BBB", 4)})
    assert _s2d(_rows(5)) == pytest.approx(20.0)       # mean(10, 20, 30)


def test_a_table_more_than_one_week_stale_still_defaults(db):
    _seed(db, [1, 2, 3])
    assert _s2d(_rows(6)) == 0.0
    assert _dvoa(_rows(6)) == 0.0


def test_a_future_week_is_never_filled_from_a_different_season(db):
    _seed(db, [1, 2, 3, 4])
    df = _rows(5).assign(season=2027)
    assert _s2d(df) == 0.0


def test_padding_only_adds_the_missing_next_week():
    frame = pd.DataFrame({"team": ["A", "A", "B", "B"], "season": [2026] * 4, "week": [1, 2, 1, 2]})
    games = pd.DataFrame({"opponent": ["A", "B", "A"], "season": [2026] * 3, "week": [3, 2, 9]})
    out = _with_target_week_rows(frame, games)
    assert sorted(map(tuple, out[["team", "week"]].to_numpy())) == [
        ("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)]   # A week 9 is too far; B week 2 exists
    assert _with_target_week_rows(frame, games.iloc[0:0]).equals(frame)

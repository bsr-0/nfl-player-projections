"""Train/test season selection across the season boundary.

Two defects, both hit for real on 2026-09-03 when the calendar rolled into
NFL week 1 while zero 2026 games had been played:

1. `current_season_has_weeks_played()` is a CALENDAR check (week_num >= 1). It
   says nothing about whether a game has been played, so between the nominal
   start of week 1 and actual kickoff it returned True while no result existed
   anywhere. get_train_test_seasons() then demanded database rows that could
   not exist and raised, blocking ALL training.

2. Once that raise was removed, selection fell through to in-season handling
   and chose the last COMPLETED season as the test set -- training on
   2018-2024 and silently discarding 2025 from production models.

The season a model should project is the one that has not been played yet,
whatever the calendar calls today.
"""
import pytest

import config.settings as settings
from src.utils import nfl_calendar
from src.utils.data_manager import DataManager
from src.utils.database import DatabaseManager

# These tests used to read the LIVE database and assert "2026 has 272
# scheduled games and no scores" -- true until the 2026 opener was played on
# 2026-09-09, at which point all three failed although the code was right
# (AUDIT_REPORT.md #19/#21: tests must not depend on DB_PATH state). They now
# build the state they assert about.


@pytest.fixture
def schedule_db(tmp_path, monkeypatch):
    """A private schedule table; season_has_completed_games reads DB_PATH."""
    db = DatabaseManager(db_path=tmp_path / "test.db")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "test.db")
    for week in range(1, 4):
        db.insert_schedule({"season": 2025, "week": week, "home_team": "AAA", "away_team": "BBB",
                            "home_score": 20, "away_score": 17})
        db.insert_schedule({"season": 2026, "week": week, "home_team": "AAA", "away_team": "BBB"})
    return db


def test_completed_games_check_reads_scores_not_the_calendar(schedule_db):
    dm = DataManager()
    assert dm._season_has_completed_games(2025) is True, "2025 has scored games"
    assert dm._season_has_completed_games(2026) is False, (
        "2026 has scheduled games and no scores; a season is in progress "
        "when games have been PLAYED, not when a date passes week 1")

    schedule_db.insert_schedule({"season": 2026, "week": 1, "home_team": "AAA", "away_team": "BBB",
                                 "home_score": 13, "away_score": 10})
    assert dm._season_has_completed_games(2026) is True, "one played game is enough"


def test_unplayed_future_season_is_never_treated_as_completed(schedule_db):
    assert DataManager()._season_has_completed_games(2099) is False


@pytest.fixture
def calendar_week_one_before_kickoff(monkeypatch):
    """Week 1 of 2026 per the calendar, no 2026 game played, 2018-2025 in the DB."""
    monkeypatch.setattr(DataManager, "get_available_seasons_from_db", lambda self: list(range(2018, 2026)))
    monkeypatch.setattr(DataManager, "_season_has_completed_games", staticmethod(lambda season: False))
    monkeypatch.setattr(nfl_calendar, "get_current_nfl_season", lambda today=None: 2026)
    monkeypatch.setattr(nfl_calendar, "current_season_has_weeks_played", lambda today=None: True)
    monkeypatch.setattr(nfl_calendar, "is_draft_prep_window", lambda today=None: False)
    monkeypatch.setattr(nfl_calendar, "get_projection_season", lambda today=None: 2026)


def test_selection_trains_on_all_completed_history_before_kickoff(calendar_week_one_before_kickoff):
    """The projection season is the test set; nothing completed is discarded."""
    train, test = DataManager().get_train_test_seasons()
    assert test == 2026
    assert test not in train, f"test season {test} leaked into train"
    assert max(train) == test - 1, (
        f"train ends {max(train)} but test is {test}; a completed season is "
        f"being discarded from training")


def test_selection_does_not_raise_before_kickoff(calendar_week_one_before_kickoff):
    """It previously raised, which blocked training entirely."""
    DataManager().get_train_test_seasons()


def test_data_loading_guard_uses_the_same_completed_games_rule():
    """The third guard, in load_training_data, had the same calendar bug.

    Fixing get_train_test_seasons only moved the failure: selection produced
    train 2018-2025 / test 2026 correctly, then load_training_data raised
    "Current season is in progress but test set is empty" because its own
    `in_season` was still pure calendar arithmetic. An empty test set is the
    EXPECTED state before kickoff, not an error.
    """
    import inspect
    from src.models import data_loading

    src = inspect.getsource(data_loading.load_training_data)
    guard = src[src.index("in_season = "):src.index("caller_set_season = ")]
    assert "season_has_completed_games" in guard, (
        "load_training_data must gate on a played game, not the calendar")


def test_one_implementation_of_the_completed_games_check(schedule_db):
    """Both call sites resolve to the calendar module, so they cannot drift."""
    from src.utils.nfl_calendar import season_has_completed_games

    for season in (2025, 2026, 2099):
        assert DataManager()._season_has_completed_games(season) == season_has_completed_games(season)

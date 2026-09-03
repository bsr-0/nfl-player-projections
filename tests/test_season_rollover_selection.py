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

from config.settings import DB_PATH
from src.utils.data_manager import DataManager

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="needs the local database")


def test_completed_games_check_reads_scores_not_the_calendar():
    dm = DataManager()
    assert dm._season_has_completed_games(2025) is True, "2025 is fully played"
    assert dm._season_has_completed_games(2026) is False, (
        "2026 has 272 scheduled games and no scores; a season is in progress "
        "when games have been PLAYED, not when a date passes week 1")


def test_unplayed_future_season_is_never_treated_as_completed():
    assert DataManager()._season_has_completed_games(2099) is False


def test_selection_trains_on_all_completed_history_before_kickoff():
    """The projection season is the test set; nothing completed is discarded."""
    train, test = DataManager().get_train_test_seasons()
    assert test not in train, f"test season {test} leaked into train"
    assert max(train) == test - 1, (
        f"train ends {max(train)} but test is {test}; a completed season is "
        f"being discarded from training")


def test_selection_does_not_raise_before_kickoff():
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


def test_one_implementation_of_the_completed_games_check():
    """Both call sites resolve to the calendar module, so they cannot drift."""
    from src.utils.nfl_calendar import season_has_completed_games

    assert DataManager()._season_has_completed_games(2026) == season_has_completed_games(2026)

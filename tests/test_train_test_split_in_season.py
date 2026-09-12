"""In-season split handling when the current season's stats are not loadable.

2026-09-10: one 2026 game had been played (so the season counts as in
progress, correctly -- the check reads the schedule, not the calendar), but
nflverse had not published 2026 weekly stats (404) and the play-by-play
fallback failed. Two things then blocked every retrain:

  * get_train_test_seasons raised "current season ... not in the database"
    even for a caller that passed an explicit test_season (the split every
    served model has used), and
  * auto_refresh_data() computed a default split purely to report a status,
    and let that raise propagate out of load_training_data.
"""
import pytest

from src.utils import data_manager as dm_mod
from src.utils.data_manager import DataManager, auto_refresh_data
from src.utils import nfl_calendar

AVAILABLE = list(range(2018, 2026))


@pytest.fixture
def in_season_without_current_stats(monkeypatch):
    """2026 has a completed game, but no 2026 rows exist in the DB."""
    monkeypatch.setattr(DataManager, "get_available_seasons_from_db", lambda self: list(AVAILABLE))
    monkeypatch.setattr(DataManager, "_season_has_completed_games", staticmethod(lambda season: True))
    monkeypatch.setattr(nfl_calendar, "get_current_nfl_season", lambda today=None: 2026)
    monkeypatch.setattr(nfl_calendar, "current_season_has_weeks_played", lambda today=None: True)
    monkeypatch.setattr(nfl_calendar, "is_draft_prep_window", lambda today=None: False)
    monkeypatch.setattr(nfl_calendar, "get_projection_season", lambda today=None: 2026)


def test_explicit_test_season_is_not_blocked(in_season_without_current_stats):
    train, test = DataManager().get_train_test_seasons(test_season=2025)
    assert test == 2025
    assert train == list(range(2018, 2025))


def test_default_split_still_refuses_to_guess(in_season_without_current_stats):
    with pytest.raises(ValueError, match="not in the database"):
        DataManager().get_train_test_seasons()


def test_auto_refresh_status_does_not_block_on_default_split(monkeypatch):
    from src.data import auto_refresh

    monkeypatch.setattr(auto_refresh.NFLDataRefresher, "refresh", lambda self, force=False: {})
    monkeypatch.setattr(DataManager, "check_data_availability",
                        lambda self, force=False: {"available_seasons": list(AVAILABLE), "latest_season": 2025})
    monkeypatch.setattr(DataManager, "ensure_schedule_loaded", lambda self, season: None)
    monkeypatch.setattr(DataManager, "get_prediction_season", lambda self: 2026)

    def _raise(self, *a, **k):
        raise ValueError("Current season 2026 has completed games but is not in the database.")
    monkeypatch.setattr(DataManager, "get_train_test_seasons", _raise)

    with pytest.warns(UserWarning, match="Default train/test split unavailable"):
        status = auto_refresh_data()

    assert status["train_test_split"] is None
    assert status["prediction_season"] == 2026

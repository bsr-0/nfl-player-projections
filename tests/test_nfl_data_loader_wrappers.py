"""Tests for retry-wrapped nfl_data_py fetch helpers in nfl_data_loader."""

import pandas as pd

from src.data import nfl_data_loader


def test_retry_wrappers_call_expected_nfl_data_py_methods_once(monkeypatch):
    """Each retry wrapper should call the matching nfl_data_py import function exactly once."""
    seasons = [2024]
    weekly_df = pd.DataFrame({"player_id": ["p1"], "season": [2024], "week": [1]})
    snap_df = pd.DataFrame({"player_id": ["p1"], "season": [2024], "week": [1]})
    schedule_df = pd.DataFrame({"season": [2024], "week": [1]})
    roster_df = pd.DataFrame({"player_id": ["p1"], "season": [2024]})

    calls = {"weekly": 0, "snap": 0, "schedule": 0, "roster": 0}

    def fake_import_weekly_data(arg):
        calls["weekly"] += 1
        assert arg == seasons
        return weekly_df

    def fake_import_snap_counts(arg):
        calls["snap"] += 1
        assert arg == seasons
        return snap_df

    def fake_import_schedules(arg):
        calls["schedule"] += 1
        assert arg == seasons
        return schedule_df

    def fake_import_seasonal_rosters(arg):
        calls["roster"] += 1
        assert arg == seasons
        return roster_df

    class FakeNfl:
        import_weekly_data = staticmethod(fake_import_weekly_data)
        import_snap_counts = staticmethod(fake_import_snap_counts)
        import_schedules = staticmethod(fake_import_schedules)
        import_seasonal_rosters = staticmethod(fake_import_seasonal_rosters)

    monkeypatch.setattr(nfl_data_loader, "_get_nfl", lambda: FakeNfl)

    assert nfl_data_loader._fetch_weekly_data(seasons).equals(weekly_df)
    assert nfl_data_loader._fetch_snap_counts(seasons).equals(snap_df)
    assert nfl_data_loader._fetch_schedules(seasons).equals(schedule_df)
    assert nfl_data_loader._fetch_rosters(seasons).equals(roster_df)

    assert calls == {"weekly": 1, "snap": 1, "schedule": 1, "roster": 1}


def test_load_weekly_data_continues_using_fetch_wrapper(monkeypatch):
    """NFLDataLoader.load_weekly_data should call _fetch_weekly_data wrapper."""
    seasons = [2024]
    wrapper_calls = {"count": 0}

    weekly_df = pd.DataFrame(
        {
            "player_id": ["p1"],
            "player_name": ["Player One"],
            "season": [2024],
            "week": [1],
            "position": ["QB"],
            "team": ["KC"],
            "completions": [10],
            "attempts": [20],
            "passing_yards": [100],
            "passing_tds": [1],
            "interceptions": [0],
            "carries": [0],
            "rushing_yards": [0],
            "rushing_tds": [0],
            "receptions": [0],
            "targets": [0],
            "receiving_yards": [0],
            "receiving_tds": [0],
            "fumbles_lost": [0],
        }
    )

    def fake_fetch_weekly(arg):
        wrapper_calls["count"] += 1
        assert arg == seasons
        return weekly_df.copy()

    monkeypatch.setattr(nfl_data_loader, "_fetch_weekly_data", fake_fetch_weekly)
    monkeypatch.setattr(nfl_data_loader, "get_current_nfl_season", lambda: 2025)
    monkeypatch.setattr(nfl_data_loader, "current_season_has_weeks_played", lambda: False)

    loader = nfl_data_loader.NFLDataLoader()
    result = loader.load_weekly_data(seasons, store_in_db=False, use_pbp_fallback=False)

    assert not result.empty
    assert wrapper_calls["count"] == 1

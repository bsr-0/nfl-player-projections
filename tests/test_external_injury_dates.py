"""Injury seasons with and without date_modified must load together.

nflverse publishes 2025-26 injury reports without date_modified while earlier
seasons keep it (tz-aware). Concatenated, pandas' to_datetime(utc=True) used
to raise a bare AssertionError, and every caller caught it and skipped the
whole external-feature block (Vegas, weather, injury) in training and serving.
"""
import pandas as pd

from src.data import external_data as ed


def _season(season, with_dates):
    df = pd.DataFrame({"gsis_id": ["a", "b"], "season": season, "week": 3, "team": "KC",
                       "report_status": ["Out", "Questionable"], "full_name": ["A", "B"],
                       "position": "WR"})
    if with_dates:
        # The old feed: tz-aware, microsecond resolution; b's report came after kickoff.
        df["date_modified"] = pd.to_datetime(["2024-09-20 18:00", "2024-09-23 02:00"]).tz_localize("UTC") \
            .astype("datetime64[us, UTC]")
    return df


def test_seasons_with_and_without_dates_load_and_keep_the_leak_guard(monkeypatch):
    frames = {2024: _season(2024, True), 2025: _season(2025, False)}
    monkeypatch.setattr(ed, "_cached_import", lambda kind, seasons, fetch: frames[seasons[0]].copy())
    loader = ed.InjuryDataLoader()
    kick = pd.DataFrame({"season": [2024, 2025], "week": 3, "team": "KC",
                         "kickoff": pd.to_datetime(["2024-09-22 17:00", "2025-09-21 17:00"], utc=True)})
    monkeypatch.setattr(loader, "_load_kickoff_times", lambda seasons: kick)

    injuries = loader.load_injuries([2024, 2025])
    assert str(injuries["date_modified"].dtype).startswith("datetime64") and len(injuries) == 4
    out = loader.get_player_injury_status(injuries)          # raised AssertionError before the fix
    got = set(zip(out["season"], out["player_id"]))
    assert (2024, "a") in got and (2024, "b") not in got     # post-kickoff report still dropped
    assert {(2025, "a"), (2025, "b")} <= got                  # no timestamp: kept

"""A veteran with a thin or missed prior season still gets a Step 8 row.

Until 2026-09-17 the anchor row was the target-1 season aggregate, and
aggregates under MIN_GAMES=6 were dropped -- so a non-rookie who played 3
games last year (Nabers, Hill, Murray entering 2026) or none at all
(Watson) had no row, no season projection, "pending" on the draft board
and no pace to blend toward on the weekly page: 178 of 628 week-1 players.

The anchor is now the most recent season in the lookback window; lags stay
by calendar year, so a missed season is a NaN lag rather than a shifted
history. Synthetic history; the DB is stubbed out.
"""
import numpy as np
import pandas as pd
import pytest

import src.models.preseason_features as pf


def _weeks(player_id, name, season, n_games, ppg, team="DET"):
    return [{"player_id": player_id, "player_name": name, "position": "WR",
             "birth_date": None, "team": team, "season": season, "week": w,
             "fantasy_points": ppg, "passing_yards": 0, "passing_tds": 0,
             "interceptions": 0, "rushing_yards": 0, "rushing_attempts": 0,
             "targets": 8, "receptions": 5, "receiving_yards": 60, "air_yards": 80,
             "passing_attempts": 0, "passing_completions": 0,
             "snap_share": 0.8, "target_share": 0.25, "rush_share": 0.0}
            for w in range(1, n_games + 1)]


@pytest.fixture
def history():
    rows = []
    # full-time starter every year -- the previously covered population
    for s in (2022, 2023, 2024, 2025):
        rows += _weeks("full", "Full Timer", s, 16, 15.0)
    # 3-game 2025 after two full seasons (a Nabers)
    rows += _weeks("thin", "Thin Prior", 2023, 15, 18.0)
    rows += _weeks("thin", "Thin Prior", 2024, 16, 19.0)
    rows += _weeks("thin", "Thin Prior", 2025, 3, 20.0)
    # missed 2025 entirely after a full 2024 (a Watson)
    rows += _weeks("missed", "Missed Year", 2023, 16, 14.0)
    rows += _weeks("missed", "Missed Year", 2024, 16, 16.0)
    # last played 2021: outside the 3-season window, must NOT get a row
    rows += _weeks("gone", "Long Gone", 2021, 16, 12.0)
    # 2026 rookie
    rows += _weeks("rook", "Rookie", 2026, 1, 9.0)
    return pd.DataFrame(rows)


@pytest.fixture
def pairs(monkeypatch, history):
    monkeypatch.setattr(pf, "_load_full_history", lambda db: history)
    monkeypatch.setattr(pf, "_inference_season_teams",
                        lambda season: pd.DataFrame(columns=["player_id", "season", "team"]))
    monkeypatch.setattr(pf, "_cold_start_rows_incoming",
                        lambda db, target, hist: pd.DataFrame())
    monkeypatch.setattr(pf, "career_static_by_player",
                        lambda db: pd.DataFrame(columns=["player_id"]))
    out = pf.build_multiyear_season_pairs(db=None, seasons=[2022, 2023, 2024, 2025],
                                          inference_season=2026)
    return out[out.target_season == 2026].set_index("player_id")


def test_thin_and_missed_prior_seasons_get_a_row(pairs):
    assert {"full", "thin", "missed"} <= set(pairs.index)
    assert "gone" not in pairs.index, "outside the lookback window"


def test_lags_stay_by_calendar_year(pairs):
    thin = pairs.loc["thin"]
    assert thin["games_played_y1"] == 3 and thin["ppg_y1"] == pytest.approx(20.0)
    assert thin["games_played_y2"] == 16 and thin["ppg_y2"] == pytest.approx(19.0)
    missed = pairs.loc["missed"]
    assert np.isnan(missed["games_played_y1"]), "a missed season is a NaN lag, not last year's numbers"
    assert missed["games_played_y2"] == 16 and missed["ppg_y2"] == pytest.approx(16.0)
    assert missed["years_of_history"] == 2


def test_years_exp_counts_every_prior_season_from_raw_history(pairs):
    assert pairs.loc["full", "years_exp"] == 4
    assert pairs.loc["thin", "years_exp"] == 3
    assert pairs.loc["missed", "years_exp"] == 2


def test_veterans_are_not_cold_start(pairs):
    assert int(pairs.loc["thin", "is_cold_start"]) == 0
    assert int(pairs.loc["missed", "is_cold_start"]) == 0

"""prev_season_ppg must be the prior season's PPG, constant within a season.

It used to be a within-season expanding mean shifted one row per player:
equal to prior-season PPG only at week 1 (minus the final game), NaN at
week 2 for 95.6% of rows (measured 2022-2025), and the CURRENT season's
lagged mean from week 3 on -- while leakage.py declared it "prior season".
bayesian_prior_ppg and career_year_flag are built on it.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


def _frame():
    # P1: 2023 = 10, 20, 30 (ppg 20); 2024 = 40, 50 (ppg 45); 2025 stub week.
    # P2: only 2024 (ppg 5) -> 2025 sees it, 2024 has no prior.
    return pd.DataFrame({
        "player_id": ["P1"] * 6 + ["P2"] * 3,
        "season":    [2023, 2023, 2023, 2024, 2024, 2025, 2024, 2024, 2025],
        "week":      [1, 2, 3, 1, 2, 1, 1, 2, 1],
        "fantasy_points": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan, 5.0, 5.0, 7.0],
    })


@pytest.fixture
def fe():
    return FeatureEngineer.__new__(FeatureEngineer)


def test_is_prior_season_mean_and_constant_within_season(fe):
    out = fe._add_prev_season_ppg(_frame())
    p1 = out[out.player_id == "P1"].set_index(["season", "week"])["prev_season_ppg"]
    assert p1[(2024, 1)] == 20.0
    assert p1[(2024, 2)] == 20.0          # week 2 used to be NaN
    assert p1[(2025, 1)] == 45.0          # stub row (no game yet) still resolves
    assert out.groupby(["player_id", "season"])["prev_season_ppg"].nunique(dropna=False).max() == 1


def test_no_prior_season_is_nan_not_current_season(fe):
    out = fe._add_prev_season_ppg(_frame())
    p2 = out[out.player_id == "P2"].set_index(["season", "week"])["prev_season_ppg"]
    assert pd.isna(p2[(2024, 1)]) and pd.isna(p2[(2024, 2)])
    assert p2[(2025, 1)] == 5.0


def test_row_order_does_not_matter(fe):
    df = _frame()
    shuffled = df.sample(frac=1.0, random_state=3)
    a = fe._add_prev_season_ppg(df.copy()).set_index(["player_id", "season", "week"])["prev_season_ppg"]
    b = fe._add_prev_season_ppg(shuffled.copy()).set_index(["player_id", "season", "week"])["prev_season_ppg"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_season_without_games_is_skipped_not_carried_as_nan(fe):
    df = pd.DataFrame({
        "player_id": ["P1"] * 4,
        "season": [2022, 2023, 2023, 2024],
        "week": [1, 1, 2, 1],
        "fantasy_points": [12.0, np.nan, np.nan, 3.0],   # missed all of 2023
    })
    out = fe._add_prev_season_ppg(df)
    assert out.loc[out.season == 2024, "prev_season_ppg"].iloc[0] == 12.0


def test_career_year_flag_is_season_level(fe):
    # Prior season (2024, ppg 45) is >30% above the career baseline before it (2023, ppg 20).
    df = fe._add_prev_season_ppg(_frame())
    out = fe._add_career_year_flag(df)
    p1 = out[out.player_id == "P1"].set_index(["season", "week"])["career_year_flag"]
    assert p1[(2025, 1)] == 1
    assert p1[(2024, 1)] == 0 and p1[(2024, 2)] == 0   # no career baseline before 2023
    assert p1[(2023, 1)] == 0


def test_rookie_prior_uses_frame_independent_debut(fe, monkeypatch):
    """A veteran whose first in-frame season is the window start must not get a rookie prior."""
    monkeypatch.setattr(FeatureEngineer, "_load_rookie_priors", classmethod(lambda cls: {"WR": {"rd1": 9.9}}))
    monkeypatch.setattr(FeatureEngineer, "_load_draft_rounds", lambda self: {"VET": 1, "ROOK": 1})
    df = pd.DataFrame({
        "player_id": ["VET", "VET", "ROOK", "ROOK"],
        "season": [2024, 2024, 2024, 2024],
        "week": [1, 2, 1, 2],
        "fantasy_points": [8.0, 9.0, 4.0, 6.0],
        "position": ["WR"] * 4,
        "first_nfl_season": [2015, 2015, 2024, 2024],
    })
    out = fe._add_prev_season_ppg(df)
    assert out.loc[out.player_id == "ROOK", "prev_season_ppg"].eq(9.9).all()
    assert out.loc[out.player_id == "VET", "prev_season_ppg"].isna().all()

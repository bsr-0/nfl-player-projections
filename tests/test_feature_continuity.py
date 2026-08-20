"""Features must not lurch between adjacent seasons, and unknown must stay unknown.

Every expensive defect found on 2026-08-20 was a train/test discontinuity in
a declared CAUSAL_FEATURE, and no existing gate saw any of them (GAPS.md).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.quality_gates import check_feature_season_continuity


def _frame(seasons_values: dict, col="f"):
    rows = []
    for season, vals in seasons_values.items():
        for v in vals:
            rows.append({"season": season, col: v})
    return pd.DataFrame(rows)


def test_step_change_between_adjacent_seasons_fails():
    """The is_dome shape: flat for years, then a jump."""
    rng = np.random.RandomState(0)
    df = _frame({2022: rng.normal(0.03, 0.01, 200), 2023: rng.normal(0.03, 0.01, 200),
                 2024: rng.normal(0.03, 0.01, 200), 2025: rng.normal(0.36, 0.01, 200)})
    out = check_feature_season_continuity(df, ["f"])
    assert out["passed"] is False
    assert out["examples"][0]["from_season"] == 2024
    assert out["examples"][0]["to_season"] == 2025


def test_gradual_league_trend_passes():
    """A real trend moves a little each year; an ingestion break moves once."""
    rng = np.random.RandomState(1)
    df = _frame({y: rng.normal(0.30 + 0.01 * (y - 2020), 0.05, 300)
                 for y in range(2020, 2026)})
    assert check_feature_season_continuity(df, ["f"])["passed"]


def test_missingness_jump_fails_even_when_the_mean_holds():
    """A column going fully populated -> fully NaN keeps its mean but is
    still a break."""
    df = _frame({2023: [1.0] * 200, 2024: [1.0] * 200, 2025: [np.nan] * 200})
    out = check_feature_season_continuity(df, ["f"])
    assert out["passed"] is False
    assert out["examples"][0]["missingness_jump"] == pytest.approx(1.0)


def test_constant_feature_does_not_divide_by_zero():
    df = _frame({2023: [0.5] * 100, 2024: [0.5] * 100, 2025: [0.5] * 100})
    assert check_feature_season_continuity(df, ["f"])["passed"]


def test_non_numeric_and_absent_columns_are_skipped():
    df = _frame({2024: [1.0] * 50, 2025: [1.0] * 50})
    df["label"] = "x"
    out = check_feature_season_continuity(df, ["f", "label", "not_here"])
    assert out["passed"]
    assert out["checked_features"] == 2


def test_scheme_tendencies_are_nan_when_unknown(monkeypatch):
    """Pre-2022 has no FTN charting; a constant 0.5 made the column a
    literal constant for 17 seasons."""
    from src.features.feature_engineering import PositionFeatureEngineer
    fe = PositionFeatureEngineer("RB")
    df = pd.DataFrame({"team": ["SF", "SF"], "season": [2010, 2011]})
    out = fe._add_scheme_tendencies(df.copy())
    assert out["team_motion_rate"].isna().all()
    assert out["team_play_action_rate"].isna().all()


def test_bounded_scaler_preserves_nan():
    """The filler that made marking a feature unknown pointless."""
    from src.models.feature_preparation import _apply_bounded_scaling
    from pathlib import Path
    import tempfile
    train = pd.DataFrame({"team_motion_rate": [0.2, 0.4, np.nan, 0.8],
                          "fantasy_points": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"team_motion_rate": [np.nan, 0.6],
                         "fantasy_points": [1.0, 2.0]})
    with tempfile.TemporaryDirectory() as tmp:
        _apply_bounded_scaling(train, test, Path(tmp) / "s.joblib")
    assert train["team_motion_rate"].isna().sum() == 1, "NaN must survive scaling"
    assert test["team_motion_rate"].isna().sum() == 1
    observed = train["team_motion_rate"].dropna()
    assert observed.min() == pytest.approx(0.0) and observed.max() == pytest.approx(1.0), \
        "observed values must still span the scaled range"


def test_known_missingness_boundary_is_exempt():
    """A documented source limit must not fail the gate forever -- a gate
    that always fails is a gate nobody reads."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "season": np.repeat([2021, 2022, 2023, 2024], 100),
        "team_motion_rate": np.concatenate([[np.nan] * 200, rng.normal(0.36, 0.02, 200)]),
        "undocumented": np.concatenate([[np.nan] * 200, rng.normal(0.36, 0.02, 200)]),
    })
    out = check_feature_season_continuity(df, ["team_motion_rate", "undocumented"])
    assert [e["feature"] for e in out["examples"]] == ["undocumented"]


def test_exempt_column_still_checked_for_level_shifts():
    """Exemption covers the NaN boundary only, not a jump between observed
    seasons."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({
        "season": np.repeat([2023, 2024, 2025], 200),
        "team_motion_rate": np.concatenate([
            rng.normal(0.36, 0.01, 400), rng.normal(0.90, 0.01, 200)]),
    })
    out = check_feature_season_continuity(df, ["team_motion_rate"])
    assert out["passed"] is False
    assert out["examples"][0]["from_season"] == 2024

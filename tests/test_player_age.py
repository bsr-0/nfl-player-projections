"""Age must come from birth dates, not from the player's position.

The bug these pin: `season_long_features.add_age_features` fell back to
`{'QB': 28, 'RB': 25, 'WR': 26, 'TE': 27}` whenever `age` and `years_exp`
were both absent -- which was always, in the training frame. Every row got
its position's constant, so `age_curve` (a declared CAUSAL_FEATURE at all
four positions) had zero variance, and so did age_factor,
age_expected_games, decline_rate, years_from_peak and is_in_prime.
"""
import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.features.player_age import (
    DEFAULT_FALLBACK_AGE, POSITION_FALLBACK_AGES, age_from_birth_date,
    birth_date_map, derive_age, season_start,
)


@pytest.fixture
def db(tmp_path):
    """A players table with two known birth dates and one missing."""
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (player_id TEXT, birth_date TEXT)")
    conn.executemany("INSERT INTO players VALUES (?, ?)", [
        ("00-0000001", "1990-01-01"),
        ("00-0000002", "2000-06-15"),
        ("00-0000003", ""),
    ])
    conn.commit()
    conn.close()
    birth_date_map.cache_clear()
    yield str(path)
    birth_date_map.cache_clear()


def test_age_is_measured_at_season_start():
    """One age per season, not one that ticks over mid-year -- so week 1 and
    week 17 of the same season agree."""
    assert season_start(2020) == pd.Timestamp("2020-09-01").to_pydatetime()
    age = age_from_birth_date("1990-09-01", 2020)
    assert age == pytest.approx(30.0, abs=0.01)


def test_age_varies_within_position(db):
    """The core regression: two QBs born ten years apart must not share an
    age."""
    df = pd.DataFrame({
        "player_id": ["00-0000001", "00-0000002"],
        "position": ["QB", "QB"],
        "season": [2020, 2020],
    })
    age = derive_age(df, db_path=db)
    assert age.nunique() == 2
    assert age.iloc[0] == pytest.approx(30.67, abs=0.05)
    assert age.iloc[1] == pytest.approx(20.21, abs=0.05)
    assert not (age == POSITION_FALLBACK_AGES["QB"]).any()


def test_age_advances_with_season(db):
    """Same player, three seasons: age must increase by one per season."""
    df = pd.DataFrame({
        "player_id": ["00-0000001"] * 3,
        "position": ["QB"] * 3,
        "season": [2018, 2019, 2020],
    })
    age = derive_age(df, db_path=db)
    diffs = np.diff(age.to_numpy())
    assert np.allclose(diffs, 1.0, atol=0.01)


def test_birth_date_column_on_the_frame_wins_over_db(db):
    df = pd.DataFrame({
        "player_id": ["00-0000001"],
        "birth_date": ["1980-09-01"],
        "position": ["QB"],
        "season": [2020],
    })
    assert derive_age(df, db_path=db).iloc[0] == pytest.approx(40.0, abs=0.05)


def test_falls_back_to_years_exp_then_position_constant(db):
    """Both degraded paths still work -- they are last resorts now, not the
    only path."""
    df = pd.DataFrame({
        "player_id": ["00-0000003", "00-0000004"],
        "position": ["RB", "WR"],
        "season": [2020, 2020],
        "years_exp": [5.0, np.nan],
    })
    age = derive_age(df, db_path=db)
    assert age.iloc[0] == 27.0                              # 22 + years_exp
    assert age.iloc[1] == POSITION_FALLBACK_AGES["WR"]      # position constant


def test_unknown_position_gets_the_generic_default(db):
    df = pd.DataFrame({"player_id": ["x"], "position": ["K"], "season": [2020]})
    assert derive_age(df, db_path=db).iloc[0] == DEFAULT_FALLBACK_AGE


def test_heavy_constant_fallback_warns(db, caplog):
    """The guardrail whose absence let the original bug run silently."""
    df = pd.DataFrame({
        "player_id": ["unknown"] * 10,
        "position": ["QB"] * 10,
        "season": [2020] * 10,
    })
    with caplog.at_level(logging.WARNING):
        derive_age(df, db_path=db)
    assert any("fell back to the position constant" in r.message for r in caplog.records)


def test_no_warning_when_birth_dates_cover_the_frame(db, caplog):
    df = pd.DataFrame({
        "player_id": ["00-0000001", "00-0000002"],
        "position": ["QB", "RB"],
        "season": [2020, 2020],
    })
    with caplog.at_level(logging.WARNING):
        derive_age(df, db_path=db)
    assert not any("fell back" in r.message for r in caplog.records)


def test_malformed_birth_date_degrades_instead_of_raising(db):
    df = pd.DataFrame({
        "player_id": ["00-0000001"],
        "birth_date": ["not-a-date"],
        "position": ["QB"],
        "season": [2020],
    })
    # Falls through to the DB value for that player, which is valid.
    assert derive_age(df, db_path=db).iloc[0] == pytest.approx(30.67, abs=0.05)


def test_missing_season_column_is_an_error_not_a_silent_constant(db):
    with pytest.raises(ValueError, match="season"):
        derive_age(pd.DataFrame({"player_id": ["00-0000001"], "position": ["QB"]}), db_path=db)


def test_age_uses_no_future_information(db):
    """Birth date is static, so a 2018 row must not shift when 2020 rows are
    present in the same frame."""
    alone = derive_age(pd.DataFrame({
        "player_id": ["00-0000001"], "position": ["QB"], "season": [2018]}), db_path=db)
    with_future = derive_age(pd.DataFrame({
        "player_id": ["00-0000001"] * 2, "position": ["QB"] * 2,
        "season": [2018, 2020]}), db_path=db)
    assert alone.iloc[0] == pytest.approx(with_future.iloc[0])


@pytest.fixture
def patched_lookup(monkeypatch):
    """The end-to-end paths call derive_age with no db_path, so they would
    otherwise read the production DB. Patch the lookup, not the DB."""
    import src.features.player_age as pa
    monkeypatch.setattr(pa, "birth_date_map",
                        lambda db_path=None: {"00-0000001": "1990-01-01",
                                              "00-0000002": "2000-06-15"})


def test_age_curve_is_not_constant_once_age_varies(patched_lookup):
    """End-to-end on the feature that was declared causal but frozen."""
    from src.features.feature_engineering import PositionFeatureEngineer
    df = pd.DataFrame({
        "player_id": ["00-0000001", "00-0000002"],
        "position": ["QB", "QB"],
        "season": [2020, 2020],
    })
    out = PositionFeatureEngineer("QB")._add_age_curve_feature(df.copy())
    assert out["age_curve"].nunique() > 1, "age_curve must vary once ages differ"


def test_age_features_vary_end_to_end(patched_lookup):
    """Every downstream age-derived feature, not just age itself."""
    from src.features.season_long_features import AgeCurveModel
    df = pd.DataFrame({
        "player_id": ["00-0000001", "00-0000002"],
        "position": ["RB", "RB"],
        "season": [2020, 2020],
    })
    out = AgeCurveModel().add_age_features(df.copy())
    for col in ["age", "age_factor", "age_expected_games", "decline_rate",
                "years_from_peak", "is_in_prime"]:
        assert col in out.columns, f"{col} missing"
    assert out["age"].nunique() == 2
    assert out["years_from_peak"].nunique() == 2

"""The participation contract: which player-weeks are production observations.

Each case here is a way the contract could silently rot back into "a stats
row happened to exist" -- the conflation the snap-based target definition
exists to remove.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.single_week_ppr.population import (
    QUALITY_COLUMN, QUALITY_INFERRED, QUALITY_SNAP_CONFIRMED, QUALITY_UNOBSERVED,
    apply_regime, evaluation_population, label_participation, regime_feature_columns,
    snap_bucket, tenure_bucket,
)


def _rows(records):
    return pd.DataFrame(records)


def test_played_and_scored_zero_is_a_production_observation():
    """The row the whole refactor exists to protect: 37 snaps, 0 PPR."""
    df = _rows([{"season": 2022, "week": 3, "snap_count": 37,
                 "fantasy_points": 0.0, "data_source": "inferred_snap_verified_zero"}])
    assert label_participation(df).iloc[0] == QUALITY_SNAP_CONFIRMED
    assert len(apply_regime(df, "A_clean_modern")) == 1


def test_zero_snaps_is_not_a_production_observation():
    """Active but never took the field: an availability fact, not a
    production one."""
    df = _rows([{"season": 2022, "week": 3, "snap_count": 0,
                 "fantasy_points": 0.0, "data_source": "nflverse_stats"}])
    assert label_participation(df).iloc[0] == QUALITY_UNOBSERVED
    assert apply_regime(df, "B_extended").empty


def test_snap_era_null_snaps_is_unobserved_not_inferred():
    """An unmatched snap record must not fall back to the PPR proxy inside
    the regime that claims snap-confirmed labels."""
    df = _rows([{"season": 2016, "week": 5, "snap_count": np.nan,
                 "fantasy_points": 14.2, "data_source": "nflverse_stats"}])
    assert label_participation(df).iloc[0] == QUALITY_UNOBSERVED


def test_pre_2013_positive_ppr_is_inferred_participation():
    df = _rows([{"season": 2010, "week": 5, "snap_count": 0,
                 "fantasy_points": 9.4, "data_source": "nflverse_stats"}])
    assert label_participation(df).iloc[0] == QUALITY_INFERRED


def test_pre_2013_zero_ppr_is_excluded_even_with_a_stats_row():
    """Pre-2013 snap_count is a placeholder 0, not a measurement, so a
    zero-point row there is ambiguous -- exclude rather than contaminate."""
    df = _rows([{"season": 2010, "week": 5, "snap_count": 0,
                 "fantasy_points": 0.0, "data_source": "nflverse_stats"}])
    assert label_participation(df).iloc[0] == QUALITY_UNOBSERVED


def test_pre_2013_pbp_confirmed_zero_survives():
    """Direct play-by-play evidence of participation outranks the PPR proxy."""
    df = _rows([{"season": 2010, "week": 5, "snap_count": 0,
                 "fantasy_points": 0.0, "data_source": "inferred_pbp_confirmed_zero"}])
    assert label_participation(df).iloc[0] == QUALITY_INFERRED


def test_regime_a_excludes_the_historical_era_entirely():
    df = _rows([
        {"season": 2010, "week": 1, "snap_count": 0, "fantasy_points": 12.0,
         "data_source": "nflverse_stats"},
        {"season": 2020, "week": 1, "snap_count": 44, "fantasy_points": 12.0,
         "data_source": "nflverse_stats"},
    ])
    assert apply_regime(df, "A_clean_modern")["season"].tolist() == [2020]
    assert apply_regime(df, "B_extended")["season"].tolist() == [2010, 2020]


def test_regimes_b_and_c_select_identical_rows():
    """B and C differ only in what the model is shown, never in population."""
    df = _rows([
        {"season": 2010, "week": 1, "snap_count": 0, "fantasy_points": 12.0,
         "data_source": "nflverse_stats"},
        {"season": 2020, "week": 1, "snap_count": 44, "fantasy_points": 0.0,
         "data_source": "inferred_snap_verified_zero"},
    ])
    b = apply_regime(df, "B_extended")
    c = apply_regime(df, "C_extended_flagged")
    pd.testing.assert_frame_equal(b, c)


def test_only_regime_c_exposes_the_quality_flag():
    cols = ["a", "b"]
    assert regime_feature_columns(cols, "A_clean_modern") == cols
    assert regime_feature_columns(cols, "B_extended") == cols
    assert regime_feature_columns(cols, "C_extended_flagged") == cols + [QUALITY_COLUMN]


def test_evaluation_population_is_snap_confirmed_only():
    """All arms must be scored on identical rows or the contrast is
    confounded by which weeks each arm was allowed to be graded on."""
    df = _rows([
        {"season": 2024, "week": 1, "snap_count": 44, "fantasy_points": 8.0,
         "data_source": "nflverse_stats"},
        {"season": 2024, "week": 2, "snap_count": np.nan, "fantasy_points": 8.0,
         "data_source": "nflverse_stats"},
    ])
    assert evaluation_population(df)["week"].tolist() == [1]


def test_unknown_regime_raises():
    with pytest.raises(ValueError, match="Unknown regime"):
        apply_regime(_rows([{"season": 2020, "snap_count": 1,
                             "fantasy_points": 1.0, "data_source": "x"}]), "D")


def test_snap_buckets_land_where_their_names_read():
    out = snap_bucket(pd.Series([1, 5, 6, 15, 30, 50, 51]))
    assert out.astype(str).tolist() == ["0-5", "0-5", "5-15", "5-15", "15-30", "30-50", "50+"]


def test_tenure_bucket_without_first_season_is_unclassified_not_rookie():
    df = _rows([{"season": 2020, "first_season": np.nan},
                {"season": 2020, "first_season": 2020}])
    out = tenure_bucket(df)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == "rookie"

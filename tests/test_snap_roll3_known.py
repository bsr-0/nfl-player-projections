"""`snap_share_pct_roll3_known` -- how much of the rolling window was measured.

The failure this prevents: roll3 = 0.42 looks equally trustworthy whether it
averaged three known games or one known game and two unknowns.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


def _frame(shares):
    return pd.DataFrame({
        "player_id": ["p1"] * len(shares),
        "season": [2015] * len(shares),
        "week": list(range(1, len(shares) + 1)),
        "snap_share_pct": shares,
    })


def _known(shares):
    fe = FeatureEngineer.__new__(FeatureEngineer)
    out = fe._add_snap_roll3_known(_frame(shares), window=3)
    return out["snap_share_pct_roll3_known"].tolist()


def test_fully_known_history_scores_one():
    # row 4 looks back at weeks 1-3, all measured
    assert _known([50.0, 60.0, 70.0, 80.0])[3] == pytest.approx(1.0)


def test_partially_known_history_is_penalised():
    """One known of three available -> 1/3, not 1.0."""
    shares = [50.0, np.nan, np.nan, 80.0]
    assert _known(shares)[3] == pytest.approx(1 / 3)


def test_two_of_three_known():
    shares = [50.0, 60.0, np.nan, 80.0]
    assert _known(shares)[3] == pytest.approx(2 / 3)


def test_no_known_history_scores_zero():
    shares = [np.nan, np.nan, np.nan, 80.0]
    assert _known(shares)[3] == pytest.approx(0.0)


def test_short_history_is_not_penalised_for_being_short():
    """A player's 2nd game has one prior game. If it is known, his history is
    as complete as it can be -- 1.0, not 1/3. Dividing by the window size
    would conflate 'new player' with 'missing data'."""
    assert _known([50.0, 60.0])[1] == pytest.approx(1.0)


def test_short_history_with_unknown_prior_scores_zero():
    assert _known([np.nan, 60.0])[1] == pytest.approx(0.0)


def test_first_row_has_no_history():
    """No prior games at all -> 0.0, and never NaN."""
    out = _known([50.0, 60.0])
    assert out[0] == 0.0
    assert not np.isnan(out[0])


def test_column_is_never_nan():
    """It must survive the blanket fillna(0) in calculate_all_scores, which
    means it has to carry information as a real number, not a NaN."""
    out = _known([np.nan, np.nan, np.nan, np.nan])
    assert not any(np.isnan(v) for v in out)


def test_known_is_independent_per_player():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    df = pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2"],
        "snap_share_pct": [50.0, 60.0, np.nan, 70.0],
    })
    out = fe._add_snap_roll3_known(df, window=3)["snap_share_pct_roll3_known"]

    assert out.iloc[1] == pytest.approx(1.0)   # p1's prior game known
    assert out.iloc[3] == pytest.approx(0.0)   # p2's prior game unknown


def test_missing_source_column_is_a_noop():
    fe = FeatureEngineer.__new__(FeatureEngineer)
    df = pd.DataFrame({"player_id": ["p1"], "week": [1]})
    assert "snap_share_pct_roll3_known" not in fe._add_snap_roll3_known(df, 3).columns

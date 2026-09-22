"""src/models/team_allocation/reconstruct.py -- renormalization and the
partial fantasy-points reconstruction."""
import numpy as np
import pandas as pd
import pytest

from config.settings import SCORING
from src.models.team_allocation.reconstruct import (
    POINTS_ELIGIBLE_VOLUME_COLS,
    reconstruct_partial_fantasy_points,
    reconstruct_volume,
    renormalize_shares,
)


def _keys(rows):
    return pd.DataFrame(rows, columns=["team", "season", "week"])


def test_renormalize_shares_sums_to_one_within_each_group():
    keys = _keys([
        ("AAA", 2021, 1), ("AAA", 2021, 1),
        ("BBB", 2021, 1), ("BBB", 2021, 1), ("BBB", 2021, 1),
    ])
    pred = np.array([0.3, 0.5, 0.1, 0.1, 0.1])
    out = renormalize_shares(pred, keys)

    aaa = out[:2]
    bbb = out[2:]
    assert aaa.sum() == pytest.approx(1.0)
    assert bbb.sum() == pytest.approx(1.0)
    # relative proportions within a group are preserved
    assert aaa[1] / aaa[0] == pytest.approx(0.5 / 0.3)


def test_renormalize_shares_falls_back_to_equal_split_when_degenerate():
    keys = _keys([("AAA", 2021, 1)] * 4)
    pred = np.zeros(4)
    out = renormalize_shares(pred, keys)
    assert out == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_renormalize_shares_clips_negative_predictions_before_summing():
    keys = _keys([("AAA", 2021, 1), ("AAA", 2021, 1)])
    pred = np.array([-0.5, 0.5])
    out = renormalize_shares(pred, keys)
    # the negative prediction is clipped to 0 before the sum, so all weight
    # goes to the other player rather than the negative one distorting it
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)


def test_renormalize_shares_rejects_mismatched_lengths():
    keys = _keys([("AAA", 2021, 1)])
    with pytest.raises(ValueError, match="matching length"):
        renormalize_shares(np.array([0.1, 0.2]), keys)


def test_reconstruct_volume_multiplies_elementwise():
    share = np.array([0.5, 0.25])
    team_total = np.array([10.0, 40.0])
    out = reconstruct_volume(share, team_total)
    assert out == pytest.approx([5.0, 10.0])


def test_reconstruct_volume_propagates_nan_team_total():
    share = np.array([0.5])
    team_total = np.array([np.nan])
    out = reconstruct_volume(share, team_total)
    assert np.isnan(out[0])


def test_reconstruct_partial_fantasy_points_matches_scoring_constants():
    volumes = {"rushing_yards": np.array([50.0, 0.0]), "receiving_yards": np.array([30.0, 100.0])}
    points = reconstruct_partial_fantasy_points(volumes)
    expected = (
        volumes["rushing_yards"] * SCORING["rushing_yards"]
        + volumes["receiving_yards"] * SCORING["receiving_yards"]
    )
    np.testing.assert_allclose(points.to_numpy(), expected)


def test_reconstruct_partial_fantasy_points_ignores_non_eligible_columns():
    """targets/rushing_attempts have no direct scoring entry and must not
    silently contribute points even if passed in."""
    volumes = {"rushing_yards": np.array([50.0]), "targets": np.array([1000.0])}
    points = reconstruct_partial_fantasy_points(volumes)
    assert points.iloc[0] == pytest.approx(50.0 * SCORING["rushing_yards"])


def test_reconstruct_partial_fantasy_points_raises_without_eligible_columns():
    with pytest.raises(ValueError, match="reconstruct_partial_fantasy_points needs"):
        reconstruct_partial_fantasy_points({"targets": np.array([5.0])})


def test_points_eligible_cols_are_a_subset_of_volume_cols():
    from src.models.team_allocation.features import VOLUME_COLS
    assert set(POINTS_ELIGIBLE_VOLUME_COLS).issubset(set(VOLUME_COLS))

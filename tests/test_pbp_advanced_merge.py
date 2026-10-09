"""PBP-derived columns must reach the weekly stats of the current season.

The weekly release has none of them and standardization zero-fills them first,
so the merge must replace a default 0, not only a missing value. 2026 rows had
recv_epa non-zero on 0.1% of rows (80% in 2024/2025) while the PBP cache held
the real values.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.nfl_data_loader import NFLDataLoader

KEY = ["player_id", "season", "week"]


@pytest.fixture
def loader():
    return NFLDataLoader.__new__(NFLDataLoader)


def _weekly(**cols):
    base = {"player_id": ["a", "b", "c"], "season": [2026] * 3, "week": [1, 1, 1]}
    return pd.DataFrame({**base, **cols})


def _pbp(**cols):
    base = {"player_id": ["a", "b", "z"], "season": [2026] * 3, "week": [1, 1, 1]}
    return pd.DataFrame({**base, **cols})


def test_a_default_zero_is_replaced_by_the_pbp_value(loader):
    out = loader._merge_advanced_pbp_features(_weekly(recv_epa=[0.0, 0.0, 0.0]), _pbp(recv_epa=[1.5, -0.4, 9.0]))
    assert out.set_index("player_id").recv_epa.to_dict() == {"a": 1.5, "b": -0.4, "c": 0.0}


def test_a_missing_value_is_filled(loader):
    out = loader._merge_advanced_pbp_features(_weekly(recv_epa=[np.nan, np.nan, np.nan]), _pbp(recv_epa=[1.5, -0.4, 9.0]))
    assert out.set_index("player_id").recv_epa.loc[["a", "b"]].tolist() == [1.5, -0.4]
    assert np.isnan(out.set_index("player_id").recv_epa["c"])


def test_a_stored_nonzero_value_is_never_overwritten(loader):
    out = loader._merge_advanced_pbp_features(_weekly(recv_epa=[2.0, 0.0, 0.0]), _pbp(recv_epa=[1.5, -0.4, 9.0]))
    assert out.set_index("player_id").recv_epa.to_dict() == {"a": 2.0, "b": -0.4, "c": 0.0}


def test_a_missing_pbp_value_does_not_overwrite_a_zero(loader):
    out = loader._merge_advanced_pbp_features(_weekly(recv_epa=[0.0, 0.0, 0.0]), _pbp(recv_epa=[np.nan, 0.7, 9.0]))
    assert out.set_index("player_id").recv_epa.to_dict() == {"a": 0.0, "b": 0.7, "c": 0.0}


def test_a_column_the_weekly_frame_lacks_is_added(loader):
    out = loader._merge_advanced_pbp_features(_weekly(), _pbp(redzone_targets=[2, 0, 5]))
    assert out.set_index("player_id").redzone_targets.loc[["a", "b"]].tolist() == [2, 0]


def test_rows_and_columns_outside_the_advanced_set_are_untouched(loader):
    w = _weekly(targets=[7, 3, 1], recv_epa=[0.0, 0.0, 0.0])
    out = loader._merge_advanced_pbp_features(w, _pbp(recv_epa=[1.5, -0.4, 9.0], targets=[99, 99, 99]))
    assert out.targets.tolist() == [7, 3, 1] and len(out) == 3

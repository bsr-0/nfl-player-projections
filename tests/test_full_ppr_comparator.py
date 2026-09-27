"""The full-PPR comparison must refuse silent population changes."""
import pandas as pd
import pytest

from scripts.compare_full_ppr_head_to_head import (
    _assert_exact_keys, _expected_production_keys,
)


def _keys():
    return pd.DataFrame({"player_id": ["00-001", "00-002"], "season": [2025, 2025],
                         "week": [2, 3], "team": ["AAA", "BBB"],
                         "position": ["QB", "WR"]})


def test_exact_comparison_requires_all_and_only_frozen_target_games():
    expected = _keys()
    _assert_exact_keys(expected.iloc[::-1], expected, "same keys")
    with pytest.raises(ValueError, match="exact target-game keys differ"):
        _assert_exact_keys(expected.iloc[:1], expected, "missing")
    with pytest.raises(ValueError, match="duplicate"):
        _assert_exact_keys(pd.concat([expected, expected.iloc[:1]]), expected, "duplicate")
    altered = expected.copy()
    altered.loc[0, "position"] = "RB"
    with pytest.raises(ValueError, match="exact target-game keys differ"):
        _assert_exact_keys(altered, expected, "position mismatch")


def test_target_map_keeps_next_observed_game_and_leading_zero_id():
    mapping = pd.DataFrame({
        "player_id": ["00-001", "00-001", "00-002"],
        "target_season": [2025.0, float("nan"), 2025.0],
        "target_week": [4.0, float("nan"), 7.0],
        "target_team": ["AAA", None, "BBB"],
        "target_position": ["QB", None, "WR"],
    })
    result = _expected_production_keys(mapping)
    assert len(result) == 2
    assert result.player_id.tolist() == ["00-001", "00-002"]
    assert result.week.tolist() == [4, 7]
    duplicate = pd.concat([mapping, mapping.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _expected_production_keys(duplicate)

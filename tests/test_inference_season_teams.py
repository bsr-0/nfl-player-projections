"""Where an unplayed season's team assignments come from.

`_team_for_season` reads player_weekly_stats, which by definition has no rows
for a season nobody has played. So at inference `dest_team` came back NaN for
every player, `team_changed` was 0 for all of them, and the destination
profiles -- which are multiplied by `team_changed` -- zero-filled. The model
trains with those populated. This is the same train/serve skew class GAPS
records for PreseasonProjector's feature query.
"""
import pandas as pd
import pytest

from src.models.preseason_features import (
    _destination_team_profiles, _inference_season_teams,
)


def _history():
    """Two seasons of one team's receivers, plus a season to be predicted."""
    rows = []
    for season, targets in ((2023, 100), (2024, 120), (2025, 140)):
        for week in range(1, 11):
            rows.append({"team": "DET", "position": "WR", "season": season,
                         "week": week, "targets": targets / 10,
                         "rushing_attempts": 1.0})
    return pd.DataFrame(rows)


def test_the_unplayed_season_gets_a_destination_profile():
    """Without it the merge finds nothing and both columns zero-fill."""
    profiles = _destination_team_profiles(_history(), inference_season=2026)

    row = profiles[(profiles.season == 2026) & (profiles.team == "DET")]
    assert len(row) == 1
    # The mean of the last three played seasons, which is what the shift(1)
    # rolling window would have produced for 2026.
    assert row.iloc[0]["dest_hist_tgt_pg"] == pytest.approx(12.0)


def test_no_inference_season_leaves_the_frame_alone():
    profiles = _destination_team_profiles(_history())

    assert 2026 not in set(profiles.season)


def test_a_played_season_is_not_overwritten():
    """If the season has rows of its own they win; nothing synthetic is added."""
    profiles = _destination_team_profiles(_history(), inference_season=2025)

    assert (profiles.season == 2025).sum() == 1


def test_missing_rosters_are_empty_rather_than_an_error():
    """A season nobody has ingested yet must not break the pair builder."""
    out = _inference_season_teams(2099)

    assert out.empty
    assert list(out.columns) == ["player_id", "season", "team"]


@pytest.mark.parametrize("season", [2026])
def test_the_roster_snapshot_supplies_teams_for_the_upcoming_season(season):
    out = _inference_season_teams(season)
    if out.empty:
        pytest.skip("no rosters ingested for this season")

    assert out.player_id.is_unique
    assert out.team.notna().all()

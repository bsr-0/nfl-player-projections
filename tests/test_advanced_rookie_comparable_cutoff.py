"""Rookie comparables must be from completed seasons before each feature row."""
import pandas as pd

from src.features.advanced_rookie_injury import AdvancedRookieProjector


def _comparables():
    return pd.DataFrame({
        "name": ["past", "future"], "position": ["WR", "WR"],
        "season": [2024, 2025], "draft_round": [2, 2],
        "draft_pick": [40, 40], "fantasy_points_avg": [5.0, 100.0],
        "games": [16, 16], "fantasy_points": [80.0, 1600.0],
    })


def test_historical_frame_respects_cutoff_even_when_broader():
    projector = AdvancedRookieProjector()
    result = projector.get_comparable_projection(
        position="WR", draft_round=2, draft_pick=40,
        historical_df=_comparables(), max_season=2024)
    assert [row["season"] for row in result["comparable_players"]] == [2024]
    assert result["ceiling_from_comps"] == 5.0


def test_projected_2025_rookie_cannot_use_2025_comparable(monkeypatch):
    projector = AdvancedRookieProjector()
    monkeypatch.setattr(projector, "_load_historical_rookies",
                        lambda min_season, max_season: _comparables())
    profile = projector.project_rookie(
        player_id="R", name="rookie", position="WR", draft_round=2,
        draft_pick=40, as_of_season=2025, as_of_week=1)
    assert profile.comparable_players == ["past"]


def test_feature_builder_passes_its_row_season_to_projection(monkeypatch):
    projector = AdvancedRookieProjector()
    monkeypatch.setattr(projector, "_load_historical_rookies",
                        lambda min_season, max_season: _comparables())
    seen = []
    original = projector.project_rookie

    def checked_projection(*args, **kwargs):
        seen.append(kwargs["as_of_season"])
        return original(*args, **kwargs)

    monkeypatch.setattr(projector, "project_rookie", checked_projection)
    row = pd.DataFrame([{
        "player_id": "R", "name": "rookie", "season": 2025, "week": 1,
        "team": "AAA", "position": "WR", "fantasy_points": 0.0,
        "first_nfl_season": 2025, "draft_season": 2025,
        "draft_round": 2, "draft_pick": 40, "is_undrafted": 0,
    }])
    output = projector.add_advanced_rookie_features(row)
    assert seen == [2025]
    assert output.rookie_ceiling_ppg.iloc[0] < 50

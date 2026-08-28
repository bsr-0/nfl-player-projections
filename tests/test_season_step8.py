"""Guards for the extracted Step 8 production model.

Step 8 lived only inside `_step8_arm`, an EVALUATION harness that required
target-season actuals, so it could not project an unplayed season. These tests
pin the two properties that extraction had to preserve or add.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.season_step8 import Step8SeasonModel, possible_games_for_players


class TestInferenceNeedsNoTarget:
    """The whole point of the extraction. `attach_conditional_target` inner-joins
    the panel for both the target AND possible_games, and a panel row only
    exists once a season has been played -- so on an upcoming season the join
    drops every row."""

    def test_predict_rejects_missing_possible_games_loudly(self):
        m = Step8SeasonModel()
        m.production_model, m.availability, m.features = object(), object(), []
        with pytest.raises(ValueError, match="possible_games"):
            m.predict(pd.DataFrame({"player_id": ["a"]}))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            Step8SeasonModel().predict(pd.DataFrame({"player_id": ["a"]}))

    def test_missing_fitted_feature_is_fatal_not_silent(self):
        """A feature present at fit and absent at predict means the two frames
        were built by different paths. Silently dropping it is how a model ends
        up quietly weaker -- the same failure class as the is_power5 bug."""
        m = Step8SeasonModel()
        m.production_model, m.availability = object(), object()
        m.features = ["a", "b", "c"]
        with pytest.raises(ValueError, match="missing at predict"):
            m.predict(pd.DataFrame({"a": [1.0], "possible_games": [17.0]}))


class TestPossibleGamesIsEraAware:
    """Must not re-introduce the wild-card week that a flat 18-week cap
    admitted into every pre-2021 season total."""

    @pytest.mark.parametrize("season,expected", [(2015, 16), (2020, 16), (2021, 17), (2024, 17)])
    def test_games_per_season_by_era(self, season, expected):
        players = pd.DataFrame({"dest_team": ["KC", "BUF", "PHI"]})
        got = possible_games_for_players(players, season)
        assert set(got.unique()) <= {float(expected)}, (season, got.unique())

    def test_unknown_team_falls_back_rather_than_dropping(self):
        players = pd.DataFrame({"dest_team": ["KC", "NOT_A_TEAM", None]})
        got = possible_games_for_players(players, 2024)
        assert got.notna().all()
        assert len(got) == 3

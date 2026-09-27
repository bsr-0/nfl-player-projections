"""Training must stop when an external feature join breaks row identity."""
import pandas as pd
import pytest

from src.models.feature_preparation import _prepare_training_data
from src.utils.database import DatabaseManager
import src.data.external_data as external_data


def test_training_preparation_propagates_external_identity_error(monkeypatch):
    for name in ("ensure_team_defense_stats", "ensure_team_offense_stats",
                 "ensure_team_personnel_stats"):
        monkeypatch.setattr(DatabaseManager, name, lambda self: None)

    def broken_join(frame, seasons=None):
        raise ValueError("external feature joins changed player-row count")

    monkeypatch.setattr(external_data, "add_external_features", broken_join)
    train = pd.DataFrame([{"player_id": "P", "season": 2024, "week": 1,
                           "team": "AAA", "position": "WR"}])
    test = train.assign(season=2025)
    with pytest.raises(ValueError, match="changed player-row count"):
        _prepare_training_data(train, test, ["WR"], tune_hyperparameters=False,
                               n_trials=1, fit_models=False)

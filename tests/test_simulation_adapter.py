import numpy as np
import pandas as pd
import pytest

from src.models.game_simulation import simulate_game_scripts
from src.models.simulation_adapter import game_inputs_from_predictions, player_inputs_from_predictions, simulation_inputs_from_predictions

GAMES = pd.DataFrame([{
    "season": 2026, "week": 2, "home_team": "H", "away_team": "A",
    "home_win_prob_logistic": .62, "predicted_margin_ridge": 3.5,
    "predicted_total_ridge": 45.0,
}])
PLAYERS = pd.DataFrame([
    {"player_id": "h_qb", "team": "H", "opponent": "A", "position": "QB",
     "predicted_points": 18.0, "prediction_ci80_lower": 12.0, "prediction_ci80_upper": 24.0},
    {"player_id": "a_wr", "team": "A", "opponent": "H", "position": "WR",
     "predicted_points": 14.0, "prediction_ci80_lower": 8.0, "prediction_ci80_upper": 20.0},
    {"player_id": "bye", "team": "X", "opponent": "Y", "position": "RB", "predicted_points": 10.0},
])

def test_adapters_match_current_serving_columns():
    games = game_inputs_from_predictions(GAMES)
    assert list(games) == ["2026_2_H_A"]
    players = player_inputs_from_predictions(PLAYERS, games)
    assert [p.player_id for p in players["2026_2_H_A"]] == ["h_qb", "a_wr"]
    assert players["2026_2_H_A"][0].sd_fantasy_points == pytest.approx(12 / (2 * 1.281551565545))

def test_adapter_consumes_optional_usage_and_zero_participation():
    # PLAYERS[0] is the QB row -- QBs aren't targeted, so their share field
    # is pass_share (share of the team's own dropbacks), not target_share
    # (see share_fields in simulation_adapter.py and the KEEP comment in
    # scripts/generate_weekly_data.py).
    players = PLAYERS.iloc[:1].copy()
    players["pass_share"] = .30
    players["participation_prob"] = 0.0
    game_players = player_inputs_from_predictions(players, game_inputs_from_predictions(GAMES))
    player = game_players["2026_2_H_A"][0]
    assert player.usage_share == .30
    assert player.participation_prob == 0.0

def test_complete_adapter_returns_simulation_ready_game():
    inputs = simulation_inputs_from_predictions(GAMES, PLAYERS)
    game, players = inputs["2026_2_H_A"]
    draws = simulate_game_scripts(game, 3, seed=11)
    assert len(players) == 2 and len(draws) == 3

def test_adapter_rejects_missing_or_nonfinite_game_prediction():
    with pytest.raises(ValueError, match="missing"):
        game_inputs_from_predictions(GAMES.drop(columns=["predicted_total_ridge"]))
    bad = GAMES.copy()
    bad.loc[0, "predicted_margin_ridge"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        game_inputs_from_predictions(bad)

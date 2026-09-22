import numpy as np
import pytest

from src.models.game_simulation import GameScriptInput, PlayerSimulationInput, simulate_players
from src.models.simulation_schema import (
    SIMULATION_SCHEMA_VERSION, build_simulation_payload, summarize_player_draws,
    validate_game_draws, validate_player_draws,
)

def _rows():
    game = GameScriptInput("g1", "H", "A", .6, 3, 44)
    players = [PlayerSimulationInput("p1", "H", "QB", 18)]
    rows = simulate_players(game, players, 4, 9)
    game_rows = [{
        "game_id": "g1", "draw": row.draw, "home_score": row.home_score,
        "away_score": row.away_score, "home_plays": row.home_plays,
        "away_plays": row.away_plays, "home_pass_attempts": row.home_pass_attempts,
        "away_pass_attempts": row.away_pass_attempts,
    } for row in __import__("src.models.game_simulation", fromlist=["simulate_game_scripts"]).simulate_game_scripts(game, 4, 9)]
    return game_rows, rows

def test_simulator_rows_fit_versioned_payload():
    games, players = _rows()
    payload = build_simulation_payload(games, players, seed=9, model_config={"outcome": "logistic"})
    assert payload["schema_version"] == SIMULATION_SCHEMA_VERSION
    assert payload["game_count"] == 1
    assert len(payload["player_summary"]) == 1
    assert payload["player_summary"][0]["draw_count"] == 4

def test_summary_has_distribution_fields():
    rows = [{"game_id": "g", "draw": i, "player_id": "p", "team": "H",
             "position": "WR", "active": True, "fantasy_points": float(i)}
            for i in range(10)]
    summary = summarize_player_draws(rows)[0]
    assert summary["p10"] == pytest.approx(0.9)
    assert summary["p90"] == pytest.approx(8.1)

def test_schema_rejects_bad_bounds_and_orphans():
    with pytest.raises(ValueError, match="pass attempts"):
        validate_game_draws([{"game_id": "g", "draw": 0, "home_score": 1,
                              "away_score": 1, "home_plays": 2, "away_plays": 2,
                              "home_pass_attempts": 3, "away_pass_attempts": 1}])
    with pytest.raises(ValueError, match="absent"):
        build_simulation_payload([{"game_id": "g", "draw": 0, "home_score": 1,
                                   "away_score": 1, "home_plays": 2, "away_plays": 2,
                                   "home_pass_attempts": 1, "away_pass_attempts": 1}],
                                  [{"game_id": "x", "draw": 0, "player_id": "p",
                                    "team": "H", "position": "RB", "active": True,
                                    "fantasy_points": 1}], seed=1)

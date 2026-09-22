import numpy as np
import pytest

from src.models.game_simulation import (
    GameScriptInput, PlayerSimulationInput, TeamVolumeBaseline,
    _split_score, role_keys_for_players, simulate_game_scripts, simulate_players,
)
from src.models.player_correlation import (
    fit_residual_correlation,
    fit_role_residual_correlation,
)

def test_scripts_are_deterministic_and_bounded():
    game = GameScriptInput("g1", "H", "A", .6, 3, 44, TeamVolumeBaseline(64, .60), TeamVolumeBaseline(63, .57))
    assert simulate_game_scripts(game, 20, 7) == simulate_game_scripts(game, 20, 7)
    assert all(x.home_pass_attempts <= x.home_plays and x.away_pass_attempts <= x.away_plays
               for x in simulate_game_scripts(game, 20, 7))

def test_score_split_conserves_total_at_extreme_margins():
    assert _split_score(10.0, 25.0) == (10.0, 0.0)
    assert _split_score(10.0, -25.0) == (0.0, 10.0)
    home, away = _split_score(44.0, 3.0)
    assert home + away == pytest.approx(44.0)

def test_game_draws_honor_win_probability_and_expected_margin():
    game = GameScriptInput("calibration", "H", "A", .75, 3.0, 100.0,
                           margin_sd=10.0, total_sd=0.0)
    draws = simulate_game_scripts(game, 5000, 19)
    wins = np.mean([draw.home_won for draw in draws])
    margins = np.mean([draw.simulated_margin for draw in draws])
    assert wins == pytest.approx(.75, abs=.02)
    assert margins == pytest.approx(3.0, abs=.35)
    assert all(draw.home_score + draw.away_score == pytest.approx(draw.simulated_total)
               for draw in draws)

def test_players_share_game_script():
    game = GameScriptInput("g1", "H", "A", .5, 0, 42)
    rows = simulate_players(game, [PlayerSimulationInput("qb", "H", "QB", 18, 4, .7)], 10, 2)
    assert len(rows) == 10 and all(r["plays"] >= r["pass_attempts"] for r in rows)

def test_correlated_residual_model_is_used_by_player_simulation():
    model = fit_residual_correlation(
        np.array([[-2, -2], [-1, -1], [1, 1], [2, 2]], dtype=float),
        ["p1", "p2"], shrinkage=0.0)
    sampled = model.sample_scaled_residuals(np.array([8.0, 8.0]), 1000, seed=3)
    assert np.corrcoef(sampled.T)[0, 1] > .9
    game = GameScriptInput("g1", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("p1", "H", "WR", 20, 8),
               PlayerSimulationInput("p2", "H", "WR", 20, 8)]
    rows = simulate_players(game, players, 25, 2, correlation_model=model)
    assert len(rows) == 50
    assert {row["position"] for row in rows} == {"WR"}

def test_supplied_usage_shares_are_allocated_by_team_pool():
    game = GameScriptInput("usage", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("rb1", "H", "RB", 15, 3, .60),
               PlayerSimulationInput("rb2", "H", "RB", 10, 3, .30)]
    rows = simulate_players(game, players, 20, 7)
    assert len(rows) == 40

def test_mixed_usage_share_pool_is_rejected():
    game = GameScriptInput("usage_mixed", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("rb1", "H", "RB", 15, 3, .60),
               PlayerSimulationInput("rb2", "H", "RB", 10, 3, None)]
    with pytest.raises(ValueError, match="usage shares"):
        simulate_players(game, players, 2, 7)

def test_role_correlation_applies_to_changed_player_lineup():
    model = fit_role_residual_correlation(
        np.array([[-2, -2], [-1, -1], [1, 1], [2, 2]], dtype=float),
        ["home_WR1", "home_WR2"], shrinkage=0.0)
    game = GameScriptInput("g2", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("new_p1", "H", "WR", 20, 8, .30),
               PlayerSimulationInput("new_p2", "H", "WR", 15, 8, .20)]
    assert role_keys_for_players(game, players) == ("home_WR1", "home_WR2")
    rows = simulate_players(game, players, 10, 2, correlation_model=model)
    assert len(rows) == 20

def test_correlation_keys_must_match_players():
    model = fit_residual_correlation(np.ones((2, 2)), ["a", "b"])
    game = GameScriptInput("g1", "H", "A", .5, 0, 42)
    with pytest.raises(ValueError, match="keys"):
        simulate_players(game, [PlayerSimulationInput("a", "H", "WR", 10)], 2, 1,
                         correlation_model=model)

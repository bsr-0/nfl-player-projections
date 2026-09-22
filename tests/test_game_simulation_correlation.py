import numpy as np
import pytest
from src.models.game_simulation import GameScriptInput, PlayerSimulationInput, TeamVolumeBaseline, simulate_game_scripts, simulate_players
from src.models.player_correlation import apply_shared_residuals, fit_residual_correlation

def test_scripts_are_deterministic_and_bounded():
    game = GameScriptInput("g1", "H", "A", .6, 3, 44, TeamVolumeBaseline(64, .60), TeamVolumeBaseline(63, .57))
    assert simulate_game_scripts(game, 20, 7) == simulate_game_scripts(game, 20, 7)
    assert all(x.home_pass_attempts <= x.home_plays and x.away_pass_attempts <= x.away_plays for x in simulate_game_scripts(game, 20, 7))

def test_players_share_game_script():
    game = GameScriptInput("g1", "H", "A", .5, 0, 42)
    rows = simulate_players(game, [PlayerSimulationInput("qb", "H", "QB", 18, 4, .7)], 10, 2)
    assert len(rows) == 10 and all(r["plays"] >= r["pass_attempts"] for r in rows)

def test_correlation_is_psd_and_reproducible():
    model = fit_residual_correlation(np.array([[1,2,0],[2,4,1],[-1,-2,0],[0,1,-1.]], float), ["a","b","c"], .2)
    assert np.all(np.linalg.eigvalsh(model.covariance) > 0)
    assert np.array_equal(model.sample_residuals(5, 3), model.sample_residuals(5, 3))

def test_invalid_residuals_rejected():
    with pytest.raises(ValueError):
        fit_residual_correlation(np.array([[1,np.nan],[2,3]]), ["a","b"])
    with pytest.raises(ValueError):
        apply_shared_residuals(np.ones(3), fit_residual_correlation(np.ones((2,2)), ["a","b"]))

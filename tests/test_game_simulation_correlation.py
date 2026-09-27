import numpy as np
import pytest

from src.models.game_simulation import (
    GameScriptInput, PlayerSimulationInput, TeamVolumeBaseline,
    MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD, _split_score,
    implied_win_probability_from_margin, margin_win_probability_disagreement,
    role_assignment_diagnostics, role_keys_for_players,
    simulate_game_scripts, simulate_players,
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

def test_mixed_usage_share_pool_falls_back_for_unsupplied_players():
    game = GameScriptInput("usage_mixed", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("rb1", "H", "RB", 15, 3, .60),
               PlayerSimulationInput("rb2", "H", "RB", 10, 3, None)]
    rows = simulate_players(game, players, 20, 7)
    assert len(rows) == 40

def test_qb_and_receiver_shares_are_not_pooled_together():
    # A starting QB's own dropback share (~1.0) plus receivers' target
    # shares would exceed one team's opportunity pool if wrongly pooled
    # together -- this must not raise.
    game = GameScriptInput("usage_qb", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("qb1", "H", "QB", 20, 4, .98),
               PlayerSimulationInput("wr1", "H", "WR", 15, 3, .28),
               PlayerSimulationInput("wr2", "H", "WR", 10, 3, .22)]
    rows = simulate_players(game, players, 20, 7)
    assert len(rows) == 60

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

def test_role_keys_for_players_ignores_usage_share():
    """Ranking must match role_keys_from_panel_game's fitting-time
    convention (points/player_id only) -- usage_share isn't in the OOF panel
    the correlation model is fit from, so using it here would let the same
    real player get a different role at serve time than at fit time."""
    game = GameScriptInput("g5", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("low_share_high_points", "H", "RB", 15, 3, .10),
               PlayerSimulationInput("high_share_low_points", "H", "RB", 8, 3, .60)]
    assert role_keys_for_players(game, players) == ("home_RB1", "home_RB2")

def test_role_assignment_diagnostics_flags_only_slots_with_a_genuine_points_tie():
    game = GameScriptInput("g3", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("rb_starter", "H", "RB", 15, 3),
               PlayerSimulationInput("rb_backup_a", "H", "RB", 4, 3),
               PlayerSimulationInput("rb_backup_b", "H", "RB", 4, 3)]
    diagnostics = role_assignment_diagnostics(game, players)
    assert diagnostics == {"home_RB1": False, "home_RB2": True, "home_RB3": True}

def test_role_assignment_diagnostics_matches_role_keys_for_players_grouping():
    """Must never drift from the actual role assignment it's diagnosing --
    same players, same (side, position) groups, same rank order."""
    game = GameScriptInput("g4", "H", "A", .5, 0, 42)
    players = [PlayerSimulationInput("wr1", "H", "WR", 18, 4),
               PlayerSimulationInput("wr2", "H", "WR", 12, 4),
               PlayerSimulationInput("wr3", "A", "WR", 9, 4)]
    keys = role_keys_for_players(game, players)
    diagnostics = role_assignment_diagnostics(game, players)
    assert set(diagnostics.keys()) == set(keys)
    assert diagnostics[keys[0]] is False  # 18 is unique within home_WR
    assert diagnostics[keys[1]] is False  # 12 is unique within home_WR
    assert diagnostics[keys[2]] is False  # lone away_WR1, no group peer to tie

def test_implied_win_probability_from_margin_known_values():
    assert implied_win_probability_from_margin(0.0, 10.0) == pytest.approx(0.5)
    assert implied_win_probability_from_margin(5.0, 0.0) == 1.0
    assert implied_win_probability_from_margin(-5.0, 0.0) == 0.0
    assert implied_win_probability_from_margin(0.0, 0.0) == 0.5
    # margin == margin_sd -> Phi(1), the standard normal CDF at one sigma
    assert implied_win_probability_from_margin(10.0, 10.0) == pytest.approx(0.8413, abs=1e-3)

def test_margin_win_probability_disagreement_flags_a_real_mismatch():
    consistent = GameScriptInput("c1", "H", "A", .618, 3.0, 44.0, margin_sd=10.0)
    assert margin_win_probability_disagreement(consistent) < MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD
    inconsistent = GameScriptInput("c2", "H", "A", .01, 10.0, 44.0, margin_sd=10.0)
    assert margin_win_probability_disagreement(inconsistent) > MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD

def test_simulate_game_scripts_warns_on_large_disagreement(caplog):
    game = GameScriptInput("mismatch", "H", "A", .01, 10.0, 44.0, margin_sd=10.0)
    with caplog.at_level("WARNING", logger="src.models.game_simulation"):
        simulate_game_scripts(game, 5, 1)
    assert any("disagrees" in record.message for record in caplog.records)

def test_simulate_game_scripts_does_not_warn_when_consistent(caplog):
    game = GameScriptInput("consistent", "H", "A", .618, 3.0, 44.0, margin_sd=10.0)
    with caplog.at_level("WARNING", logger="src.models.game_simulation"):
        simulate_game_scripts(game, 5, 1)
    assert not any("disagrees" in record.message for record in caplog.records)

def test_correlation_keys_must_match_players():
    model = fit_residual_correlation(np.ones((2, 2)), ["a", "b"])
    game = GameScriptInput("g1", "H", "A", .5, 0, 42)
    with pytest.raises(ValueError, match="keys"):
        simulate_players(game, [PlayerSimulationInput("a", "H", "WR", 10)], 2, 1,
                         correlation_model=model)


def test_empirical_residual_draws_preserve_point_center_without_game_script():
    game = GameScriptInput("empirical", "H", "A", .5, 0, 42)
    draws = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    rows = simulate_players(
        game, [PlayerSimulationInput("p", "H", "WR", 10.0, participation_prob=1.0)],
        n_draws=4, seed=9, residual_draws=draws, apply_game_script=False)
    assert [row["fantasy_points"] for row in rows] == [8.0, 9.0, 11.0, 12.0]

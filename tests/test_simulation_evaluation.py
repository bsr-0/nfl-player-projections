import numpy as np
import pandas as pd
import pytest

from src.models.simulation_evaluation import (
    energy_score, evaluate_joint_player_draws, game_draw_calibration,
    marginal_calibration, variogram_score,
)

def test_energy_score_prefers_accurate_draws():
    truth = np.array([10., 20.])
    close = np.array([[10., 20.], [11., 19.], [9., 21.]])
    far = np.array([[30., 40.], [31., 39.], [29., 41.]])
    assert energy_score(close, truth) < energy_score(far, truth)

def test_variogram_score_prefers_matching_dependence_not_location():
    """Variogram score (Scheuerer & Hamill 2015) only compares within-draw
    pairwise |x_i - x_j|^p against the observation's own |y_i - y_j|^p --
    it never looks at each dimension's absolute level, so it is
    translation-invariant by construction. `far` below is `close` shifted
    +20 in both dimensions with identical internal spread, so it scores
    identically to `close`; that is the metric doing its documented job of
    isolating dependence-structure fidelity (energy_score and marginal
    CRPS separately cover location/marginal accuracy). What the variogram
    score IS sensitive to is a mismatched pairwise spread, e.g. an ensemble
    whose two dimensions move in lockstep when the truth's dimensions are
    10 apart.
    """
    truth = np.array([10., 20.])
    close = np.array([[10., 20.], [11., 19.], [9., 21.]])
    mismatched_dependence = np.array([[10., 10.], [11., 11.], [9., 9.]])
    assert variogram_score(close, truth) < variogram_score(mismatched_dependence, truth)
    far_same_dependence = np.array([[30., 40.], [31., 39.], [29., 41.]])
    assert variogram_score(close, truth) == pytest.approx(variogram_score(far_same_dependence, truth))

def test_joint_and_marginal_evaluation():
    draws = pd.DataFrame([
        {"game_id": "g", "draw": d, "player_id": p, "fantasy_points": value}
        for d, pair in enumerate([(10., 20.), (11., 19.), (9., 21.)])
        for p, value in zip(("a", "b"), pair)
    ])
    actuals = pd.DataFrame([
        {"game_id": "g", "player_id": "a", "fantasy_points": 10.},
        {"game_id": "g", "player_id": "b", "fantasy_points": 20.},
    ])
    result = evaluate_joint_player_draws(draws, actuals)
    assert result["games_scored"] == 1
    summary = pd.DataFrame([
        {"game_id": "g", "player_id": "a", "mean": 10., "p10": 9., "p25": 9.5, "p75": 10.5, "p90": 11.},
        {"game_id": "g", "player_id": "b", "mean": 20., "p10": 19., "p25": 19.5, "p75": 20.5, "p90": 21.},
    ])
    assert marginal_calibration(summary, actuals)["coverage_80"] == 1.

def test_evaluator_rejects_missing_simulated_player_actual():
    draws = pd.DataFrame([
        {"game_id": "g", "draw": d, "player_id": p, "fantasy_points": float(d)}
        for d in range(2) for p in ("a", "b")
    ])
    actuals = pd.DataFrame([{"game_id": "g", "player_id": "a", "fantasy_points": 1.}])
    with pytest.raises(ValueError, match="lacks actual outcomes"):
        evaluate_joint_player_draws(draws, actuals)

def test_game_draw_calibration():
    draws = pd.DataFrame([
        {"game_id": "g", "draw": 0, "home_won": True, "simulated_total": 45., "simulated_margin": 3.},
        {"game_id": "g", "draw": 1, "home_won": True, "simulated_total": 47., "simulated_margin": 5.},
    ])
    actuals = pd.DataFrame([{"game_id": "g", "home_score": 27., "away_score": 20.}])
    result = game_draw_calibration(draws, actuals)
    assert result["n"] == 1 and result["total_mae"] == pytest.approx(1.)

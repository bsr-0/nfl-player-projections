import numpy as np
import pandas as pd
import pytest

from src.models.player_correlation import (
    FactorCopulaModel,
    fit_factor_copula,
    parse_role_key,
    pooled_pair_correlations,
)
from tests.simulation_fixtures import DEPENDENT_LOADINGS, ROSTER, factor_scores


def _scores(n_games, loadings, seed=0):
    rng = np.random.default_rng(seed)
    canonical = [f"{position}{rank}" for position, rank, _ in ROSTER]
    scores, roles, games = [], [], []
    for game in range(n_games):
        sides = ["home"] * len(canonical) + ["away"] * len(canonical)
        z = factor_scores(canonical * 2, sides, rng, loadings)
        scores.extend(z)
        roles.extend(f"{side}_{role}" for side, role in zip(sides, canonical * 2))
        games.extend([game] * len(sides))
    return np.asarray(scores), roles, np.asarray(games)


def _true(role_a, role_b, relation):
    return FactorCopulaModel("role", DEPENDENT_LOADINGS, (0., 0., 0.), {}).implied_correlation(
        role_a, role_b, relation)


def test_parse_role_key_caps_depth():
    assert parse_role_key("home_WR2") == ("home", "WR2")
    assert parse_role_key("away_WR7") == ("away", "WRx")
    with pytest.raises(ValueError):
        parse_role_key("WR1")


def test_pooled_pairs_count_each_within_game_pair_once():
    scores = np.array([1., 2., 3., -1., 0., 5.])
    roles = ["home_WR4", "home_WR5", "away_QB1", "home_WR4", "home_WR5", "away_QB1"]
    pairs = pooled_pair_correlations(scores, roles, np.array([0, 0, 0, 1, 1, 1]))
    counts = {(r.role_a, r.role_b, r.relation): r.n for r in pairs.itertuples()}
    # WR4 and WR5 both canonicalise to WRx; each game has one such pair.
    assert counts == {("QB1", "WRx", "opponent"): 4, ("WRx", "WRx", "same_team"): 2}


def test_role_structure_recovers_known_game_dependence():
    scores, roles, games = _scores(3000, DEPENDENT_LOADINGS, seed=1)
    model = fit_factor_copula(pooled_pair_correlations(scores, roles, games), structure="role")
    assert model.diagnostics["status"] == "fitted"
    errors = [abs(model.implied_correlation(a, b, rel) - _true(a, b, rel))
              for a in DEPENDENT_LOADINGS for b in DEPENDENT_LOADINGS
              for rel in ("same_team", "opponent") if not (a == b and rel == "same_team")]
    assert max(errors) < 0.06
    # The mechanism the script factor exists for: RB1 and WR1 compete.
    assert model.implied_correlation("RB1", "WR1", "same_team") < -0.05
    assert model.implied_correlation("QB1", "WR1", "same_team") > 0.3


def test_team_structure_reproduces_pooled_moments_with_two_parameters():
    scores, roles, games = _scores(1500, DEPENDENT_LOADINGS, seed=2)
    pairs = pooled_pair_correlations(scores, roles, games)
    model = fit_factor_copula(pairs, structure="team")
    assert model.loadings == {}
    same = pairs[pairs.relation == "same_team"]
    opponent = pairs[pairs.relation == "opponent"]
    expected_same = np.average(same.correlation, weights=same.n)
    expected_opponent = np.average(opponent.correlation, weights=opponent.n)
    assert model.implied_correlation("QB1", "TE1", "same_team") == pytest.approx(expected_same, abs=1e-9)
    assert model.implied_correlation("RB2", "WR3", "opponent") == pytest.approx(expected_opponent, abs=1e-9)
    assert model.diagnostics["representable"] is True


def test_independent_scores_fit_to_near_zero_dependence():
    scores, roles, games = _scores(1500, None, seed=3)
    model = fit_factor_copula(pooled_pair_correlations(scores, roles, games), structure="role")
    for a in DEPENDENT_LOADINGS:
        for b in DEPENDENT_LOADINGS:
            assert abs(model.implied_correlation(a, b, "opponent")) < 0.08


def test_lineup_matrices_are_valid_symmetric_and_sampling_matches():
    model = FactorCopulaModel("role", DEPENDENT_LOADINGS, (0., 0., 0.), {})
    lineup = ("home_QB1", "home_WR1", "home_RB1", "away_QB1", "away_WR1", "away_RB1", "away_K1")
    matrix = model.correlation_matrix(lineup)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0
    # Unseen canonical role (kicker) is independent of everyone.
    assert np.allclose(matrix[-1, :-1], 0.0)
    # Home/away symmetry.
    assert matrix[0, 1] == pytest.approx(matrix[3, 4])
    assert matrix[0, 4] == pytest.approx(matrix[3, 1])
    draws = model.sample_scaled_residuals(lineup, np.ones(len(lineup)), 200_000, seed=4)
    assert np.abs(np.corrcoef(draws, rowvar=False) - matrix).max() < 0.02
    again = model.sample_scaled_residuals(lineup, np.ones(len(lineup)), 1000, seed=4)
    assert np.array_equal(again, model.sample_scaled_residuals(lineup, np.ones(len(lineup)), 1000, seed=4))


def test_factor_model_round_trips_and_rejects_invalid_loadings():
    model = FactorCopulaModel("role", DEPENDENT_LOADINGS, (0., 0., 0.), {"status": "fitted"})
    assert FactorCopulaModel.from_dict(model.to_dict()) == model
    with pytest.raises(ValueError, match="idiosyncratic"):
        FactorCopulaModel("role", {"QB1": (0.8, 0.8, 0.0)}, (0., 0., 0.), {})
    assert model.covers("home_WR9") and not model.covers("garbage")


def test_empty_pair_table_gives_independent_model():
    empty = pd.DataFrame(columns=["role_a", "role_b", "relation", "n", "correlation"])
    model = fit_factor_copula(empty, structure="role")
    assert model.diagnostics["status"] == "no_usable_pairs"
    assert np.allclose(model.correlation_matrix(("home_QB1", "away_QB1")), np.eye(2))

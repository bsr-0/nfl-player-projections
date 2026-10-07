import numpy as np
import pandas as pd
import pytest

from src.models.residual_calibration import (
    PredictionAnalogCalibration,
    fit_empirical_residual_calibration,
    fit_prediction_analog_calibration,
    independent_residual_matrix,
    induce_role_rank_dependence,
    load_calibration,
    normal_scores,
    role_keys_from_panel_game,
    save_calibration,
)
from src.models.player_correlation import fit_sparse_role_residual_correlation


def _panel():
    rows = []
    for season in (2023, 2024, 2025):
        for player, team, position, predicted, actual in (
            ("h_qb", "H", "QB", 18.0, 17.0 + (season - 2023)),
            ("a_wr", "A", "WR", 12.0, 10.0 + (season - 2023)),
        ):
            rows.append({
                "player_id": player, "season": season, "week": 1,
                "game_id": f"{season}_1_H_A", "team": team,
                "opponent": "A" if team == "H" else "H", "home_team": "H", "away_team": "A",
                "position": position, "predicted_points": predicted, "actual_points": actual,
                "residual": predicted - actual, "is_cold_start": False,
            })
    return pd.DataFrame(rows)


def test_calibration_centers_empirical_draws_on_production_prediction():
    panel = _panel()
    artifact = fit_empirical_residual_calibration(panel, min_stratum_rows=2)
    draws = independent_residual_matrix(panel.iloc[:2], artifact, n_draws=2000, seed=3)
    assert np.abs(draws.mean(axis=0)).max() < 0.15
    assert artifact.diagnostics["actual_activity_is_diagnostic_only"] is True


def test_draws_are_actual_minus_prediction_deviations_not_mirrored_residuals():
    # Right-skewed outcomes around a fixed prediction: mostly small misses,
    # occasional boom games. predicted + draw must reproduce that shape.
    actuals = np.array([8.0] * 90 + [30.0] * 10)
    panel = pd.DataFrame({
        "position": "WR", "predicted_points": 10.0, "actual_points": actuals,
        "residual": 10.0 - actuals, "is_cold_start": False,
    })
    artifact = fit_empirical_residual_calibration(panel, min_stratum_rows=2)
    simulated = 10.0 + independent_residual_matrix(
        panel.iloc[:1], artifact, n_draws=20_000, seed=11)[:, 0]
    expected = 10.0 + (actuals - actuals.mean())
    assert simulated.max() == pytest.approx(expected.max())
    assert simulated.min() == pytest.approx(expected.min())
    assert np.quantile(simulated, .95) > np.quantile(simulated, .50) + 10.0


def test_role_rank_dependence_preserves_each_empirical_marginal():
    residuals = np.array([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])
    model = fit_sparse_role_residual_correlation(
        np.array([[-2., -2.], [-1., -1.], [1., 1.], [2., 2.]]),
        ["home_WR1", "away_WR1"], shrinkage=0.0, min_pair_rows=2)
    dependent = induce_role_rank_dependence(
        residuals, role_keys=("home_WR1", "away_WR1"), correlation_model=model, seed=4)
    assert np.array_equal(np.sort(dependent[:, 0]), np.sort(residuals[:, 0]))
    assert np.array_equal(np.sort(dependent[:, 1]), np.sort(residuals[:, 1]))


def test_role_keys_use_pregame_prediction_order_only():
    game = _panel().query("season == 2023").copy()
    assert role_keys_from_panel_game(game) == ("home_QB1", "away_WR1")







def _heteroscedastic_panel(n=6000, seed=0):
    """Zero-inflated, right-skewed WR outcomes whose spread grows with the projection."""
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(0.5, 20.0, size=n)
    played = rng.random(n) > 0.25 * np.exp(-predicted / 5.0)
    shape = 2.0
    actual = np.where(played, rng.gamma(shape, 1.0, size=n), 0.0)
    # Scale so E[actual | predicted] == predicted exactly.
    p_play = 1 - 0.25 * np.exp(-predicted / 5.0)
    actual = actual * predicted / (shape * p_play)
    return pd.DataFrame({
        "position": "WR", "predicted_points": predicted, "actual_points": actual,
        "residual": predicted - actual, "is_cold_start": False,
    })


def test_analog_donors_are_k_nearest_within_position_and_deterministic():
    panel = pd.DataFrame({
        "position": ["WR"] * 6 + ["TE"] * 2,
        "predicted_points": [1., 2., 3., 4., 5., 6., 3., 9.],
        "actual_points": [0., 1., 2., 3., 4., 5., 6., 7.],
    })
    artifact = fit_prediction_analog_calibration(panel, family="outcome", k=3)
    predicted, actual = artifact.donors(position="wr", predicted_points=3.4)
    assert predicted.tolist() == [2., 3., 4.]
    assert actual.tolist() == [1., 2., 3.]
    # Equidistant candidates (2 and 4 around 3.0) tie-break toward the lower index.
    predicted, _ = artifact.donors(position="WR", predicted_points=3.0)
    assert predicted.tolist() == [2., 3., 4.]
    # TE has only 2 rows < k, so it backs off to all positions.
    predicted, _ = artifact.donors(position="TE", predicted_points=8.8)
    assert predicted.tolist() == [5., 6., 9.]
    assert artifact.diagnostics["positions_backing_off_to_global"] == ["TE"]


def test_analog_support_is_centred_on_the_served_prediction():
    artifact = fit_prediction_analog_calibration(_heteroscedastic_panel(), family="residual", k=200)
    for predicted in (1.0, 8.0, 19.0):
        support = artifact.support_deviations(position="WR", predicted_points=predicted)
        assert len(support) == 200
        assert abs(support.mean()) < 1e-9


def test_outcome_analog_respects_support_where_pooled_residuals_do_not():
    train = _heteroscedastic_panel(seed=1)
    legacy = fit_empirical_residual_calibration(train, min_stratum_rows=50)
    analog = fit_prediction_analog_calibration(train, family="outcome", k=250)
    rng = np.random.default_rng(2)
    low = 1.0
    legacy_draws = low + legacy.draw_residuals(position="WR", predicted_points=low,
                                               is_cold_start=False, n_draws=5000, rng=rng)
    analog_draws = low + analog.draw_residuals(position="WR", predicted_points=low,
                                               is_cold_start=False, n_draws=5000, rng=rng)
    # Fantasy points here are never negative; a pooled residual imposes
    # 20-point players' misses on a 1-point player.
    assert (legacy_draws < -1.0).mean() > 0.10
    assert (analog_draws < -1.0).mean() < 0.01


def test_analog_coverage_is_near_nominal_across_projection_levels():
    train, test = _heteroscedastic_panel(seed=3), _heteroscedastic_panel(n=4000, seed=4)
    legacy = fit_empirical_residual_calibration(train, min_stratum_rows=50)
    analog = fit_prediction_analog_calibration(train, family="residual", k=250)

    def coverage(artifact, rows):
        hits = []
        for row in rows.itertuples(index=False):
            support = row.predicted_points + artifact.support_deviations(
                position="WR", predicted_points=row.predicted_points, is_cold_start=False)
            low, high = np.quantile(support, [.1, .9])
            hits.append(low <= row.actual_points <= high)
        return float(np.mean(hits))

    low_rows = test[test["predicted_points"] < 5]
    high_rows = test[test["predicted_points"] > 15]
    assert abs(coverage(analog, low_rows) - .8) < .05
    assert abs(coverage(analog, high_rows) - .8) < .05
    # One pooled distribution is too wide for small projections and too
    # narrow for large ones.
    assert coverage(legacy, low_rows) > .9
    assert coverage(legacy, high_rows) < .7


def test_normal_scores_are_monotone_finite_and_standardised():
    train = _heteroscedastic_panel(seed=5)
    analog = fit_prediction_analog_calibration(train, family="residual", k=250)
    probe = pd.DataFrame({"position": "WR", "predicted_points": 10.0,
                          "actual_points": [-50.0, 0.0, 5.0, 10.0, 20.0, 500.0],
                          "is_cold_start": False})
    scores = normal_scores(probe, analog)
    assert np.isfinite(scores).all()
    assert (np.diff(scores) > 0).all()
    held_out = _heteroscedastic_panel(n=3000, seed=6)
    z = normal_scores(held_out, analog)
    assert abs(z.mean()) < .1
    assert abs(z.std() - 1) < .1


def test_calibration_artifacts_round_trip_and_dispatch_by_kind(tmp_path):
    train = _heteroscedastic_panel(n=500, seed=7)
    analog = fit_prediction_analog_calibration(train, family="outcome", k=50)
    legacy = fit_empirical_residual_calibration(train, min_stratum_rows=50)
    loaded_analog = load_calibration(save_calibration(analog, tmp_path / "analog.json"))
    loaded_legacy = load_calibration(save_calibration(legacy, tmp_path / "legacy.json"))
    assert isinstance(loaded_analog, PredictionAnalogCalibration)
    assert loaded_analog == analog
    assert loaded_legacy == legacy
    for original, loaded in ((analog, loaded_analog), (legacy, loaded_legacy)):
        assert np.array_equal(
            original.support_deviations(position="WR", predicted_points=7.0),
            loaded.support_deviations(position="WR", predicted_points=7.0))
    with pytest.raises(ValueError, match="family"):
        fit_prediction_analog_calibration(train, family="bogus", k=50)


def test_cached_supports_cannot_be_mutated_by_callers():
    train = _heteroscedastic_panel(n=500, seed=8)
    legacy = fit_empirical_residual_calibration(train, min_stratum_rows=50)
    with pytest.raises(ValueError):
        legacy.support_deviations(position="WR", predicted_points=5.0)[0] = 0.0
    with pytest.raises(ValueError):
        legacy.sorted_support_deviations(position="WR", predicted_points=5.0)[0] = 0.0
    analog = fit_prediction_analog_calibration(train, family="outcome", k=50)
    before = analog.support_deviations(position="WR", predicted_points=5.0).copy()
    returned = analog.support_deviations(position="WR", predicted_points=5.0)
    returned += 100.0  # a fresh array: editing it must not leak into the artifact
    assert np.array_equal(analog.support_deviations(position="WR", predicted_points=5.0), before)

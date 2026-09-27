import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_calibrated_simulation import _write_run, run, select_candidate
from scripts.verify_calibrated_simulation import verify as verify_calibrated_run
from src.models.residual_calibration import (
    fit_empirical_residual_calibration,
    independent_residual_matrix,
    induce_role_rank_dependence,
    role_keys_from_panel_game,
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


def test_rolling_comparison_scores_three_modes_on_identical_keys():
    report, draws, actuals = run(_panel(), draws=40, seed=7)
    assert set(report["modes"]) == {
        "production_marginal", "calibrated_independent", "calibrated_role_correlation"}
    assert report["scoring_population"]["rows"] == len(actuals)
    for mode, frame in draws.groupby("mode"):
        assert set(map(tuple, frame[["game_id", "player_id"]].drop_duplicates().to_numpy())) == set(
            map(tuple, actuals[["game_id", "player_id"]].to_numpy()))
        assert report["modes"][mode]["rows"] == len(actuals)


def test_future_labels_cannot_change_earlier_outer_fold_selection():
    panel = _panel()
    baseline = select_candidate(panel, outer_season=2025, n_draws=20, seed=7, candidates=(2, 3))
    changed = panel.copy()
    future = changed["season"] == 2025
    changed.loc[future, "actual_points"] += 100.0
    changed.loc[future, "residual"] = (
        changed.loc[future, "predicted_points"] - changed.loc[future, "actual_points"])
    assert select_candidate(changed, outer_season=2025, n_draws=20, seed=7, candidates=(2, 3)) == baseline


def test_first_outer_fold_with_one_prior_season_uses_explicit_default():
    decision = select_candidate(_panel(), outer_season=2024, n_draws=20, seed=7, candidates=(2, 3))
    assert decision["selected_min_stratum_rows"] == 50
    assert decision["selection_reason"] == "default_insufficient_inner_history"


def test_saved_calibration_run_recomputes_and_verifies(tmp_path):
    panel = _panel()
    report, draws, actuals, marginal, games = run(
        panel, draws=20, seed=7, candidates=(2, 3), n_bootstrap=100, return_details=True)
    output = tmp_path / "calibrated"
    _write_run(
        output, report=report, draws=draws, actuals=actuals, marginal=marginal,
        joint_by_game=games, oof_run_dir=tmp_path / "oof", panel=panel,
        oof_verification={"panel_sha256": "test-panel"}, seed=7, n_bootstrap=100)
    assert verify_calibrated_run(output)["status"] == "verified"


def test_saved_calibration_run_rejects_a_tampered_draw_file(tmp_path):
    panel = _panel()
    report, draws, actuals, marginal, games = run(
        panel, draws=20, seed=7, candidates=(2, 3), n_bootstrap=100, return_details=True)
    output = tmp_path / "calibrated"
    _write_run(
        output, report=report, draws=draws, actuals=actuals, marginal=marginal,
        joint_by_game=games, oof_run_dir=tmp_path / "oof", panel=panel,
        oof_verification={"panel_sha256": "test-panel"}, seed=7, n_bootstrap=100)
    altered = pd.read_parquet(output / "player_draws.parquet")
    altered.loc[0, "fantasy_points"] += 1.0
    altered.to_parquet(output / "player_draws.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_calibrated_run(output)

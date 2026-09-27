import json

import numpy as np
import pandas as pd
import pytest

import scripts.generate_simulation_data as generate
from scripts.evaluate_calibrated_simulation import ROLE_FACTOR, minimum_bootstrap_for_holm, run
from src.models.calibrated_simulation import (
    Candidate,
    attach_roles,
    candidate_grid,
    fit_candidate,
    fit_role_factor,
    fit_simulation_artifacts,
    load_simulation_artifacts,
    save_simulation_artifacts,
    simulate_game,
    week_ids,
)
from tests.simulation_fixtures import DEPENDENT_LOADINGS, ROSTER, synthetic_panel

GRID = candidate_grid((50,), (60,))


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(seasons=(2022, 2023, 2024), weeks=3, n_teams=6, loadings=DEPENDENT_LOADINGS, seed=5)


@pytest.fixture(scope="module")
def artifacts(panel):
    return fit_simulation_artifacts(panel, candidates=GRID, seed=7)


def test_served_draws_equal_backtest_role_factor_draws(panel):
    result = run(panel, draws=30, seed=7, candidates=GRID, n_bootstrap=minimum_bootstrap_for_holm(),
                 selection_bootstrap=100)
    entry = result.report["selection"][-1]
    target = entry["season"] * 100 + entry["week"]
    keyed = panel.reset_index(drop=True)
    keyed["role_key"] = attach_roles(keyed)
    history = keyed.loc[week_ids(keyed) < target]
    calibration = fit_candidate(Candidate.parse(entry["full"]["candidate"]), history)
    dependence = fit_role_factor(history, calibration)
    rows = result.draws.rows
    in_week = (week_ids(rows) == target)
    assert rows.loc[in_week, "game_id"].nunique() == 3
    for game_id in rows.loc[in_week, "game_id"].unique():
        columns = np.flatnonzero((rows["game_id"] == game_id).to_numpy())
        served = simulate_game(rows.iloc[columns], calibration=calibration, dependence=dependence,
                               n_draws=30, seed=7)
        assert np.array_equal(served, result.draws.matrices[ROLE_FACTOR][:, columns])


def test_artifacts_round_trip_and_fail_closed(tmp_path, artifacts):
    assert artifacts.fitted_through == (2024, 3)
    assert artifacts.selection["target_week_id"] == 202404
    run_dir = save_simulation_artifacts(artifacts, tmp_path, provenance={"test": True})
    loaded = load_simulation_artifacts(tmp_path)  # resolves latest.json
    assert loaded.manifest["run_id"] == run_dir.name
    assert loaded.dependence == artifacts.dependence
    assert np.array_equal(loaded.calibration.support_deviations(position="WR", predicted_points=10.0),
                          artifacts.calibration.support_deviations(position="WR", predicted_points=10.0))
    assert loaded.known_player_ids == artifacts.known_player_ids
    calibration = run_dir / "calibration.json"
    calibration.write_text(calibration.read_text().replace("1", "2", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_simulation_artifacts(run_dir)
    with pytest.raises(FileNotFoundError):
        load_simulation_artifacts(tmp_path / "nothing_here")


def _serving_week(season=2026, week=5):
    games = pd.DataFrame([{"home_team": "T0", "away_team": "T1", "home_win_prob_logistic": .6,
                           "predicted_margin_ridge": 3.0, "predicted_total_ridge": 45.0},
                          {"home_team": "T2", "away_team": "T3", "home_win_prob_logistic": .4,
                           "predicted_margin_ridge": -2.0, "predicted_total_ridge": 41.0}])
    rows = []
    for home, away in (("T0", "T1"), ("T2", "T3")):
        for team, opponent in ((home, away), (away, home)):
            for position, rank, mean in ROSTER:
                rows.append({"player_id": f"{team}_{position}{rank}", "team": team, "opponent": opponent,
                             "position": position, "predicted_points": mean, "actual_points": 99.0})
    rows.append({"player_id": "T0_K1", "team": "T0", "opponent": "T1", "position": "K",
                 "predicted_points": 8.0, "actual_points": 99.0})
    rows.append({"player_id": "T9_WR1", "team": "T9", "opponent": "T8", "position": "WR",
                 "predicted_points": 9.0, "actual_points": 99.0})
    return games, pd.DataFrame(rows)


def test_week_inputs_drop_outcomes_and_report_exclusions(artifacts):
    games, players = _serving_week()
    _, rows, report = generate.build_week_inputs(games, players, season=2026, week=5,
                                                 known_player_ids=artifacts.known_player_ids)
    assert report["dropped_outcome_columns"] == ["actual_points"]
    assert report["excluded_unsupported_position"] == 1
    assert report["excluded_team_not_scheduled"] == 1
    assert report["simulated"] == 28 and "actual_points" not in rows
    wrong = players.copy()
    wrong.loc[0, "opponent"] = "T3"
    with pytest.raises(ValueError, match="opponent"):
        generate.build_week_inputs(games, wrong, season=2026, week=5, known_player_ids=frozenset())


def test_served_week_ignores_actuals_and_stays_centred(artifacts):
    games, players = _serving_week()

    def payload_for(frame):
        g, rows, _ = generate.build_week_inputs(games, frame, season=2026, week=5,
                                                known_player_ids=artifacts.known_player_ids)
        return generate.simulate_week(g, rows, artifacts, draws=3000, seed=11, model_config={})

    first = payload_for(players)
    second = payload_for(players.assign(actual_points=-5.0))
    for game_id in first.draws:
        assert np.array_equal(first.draws[game_id], second.draws[game_id])
    for game_id, matrix in first.draws.items():
        served = first.game_players(game_id)["served_prediction"].to_numpy()
        assert np.abs(matrix.mean(axis=0) - served).max() < 0.6


def test_generate_week_writes_v2_and_refuses_weeks_already_fitted(tmp_path, monkeypatch, artifacts):
    games, players = _serving_week()
    monkeypatch.setattr(generate, "DOCS_DATA", tmp_path)
    (tmp_path / "game_predictions_2026_wk5.json").write_text(games.to_json(orient="records"))
    (tmp_path / "weekly_2026_wk5.json").write_text(players.to_json(orient="records"))
    output = generate.generate_week(2026, 5, draws=200, seed=3, write_parquet=False, artifacts=artifacts)
    site = json.loads(output.read_text())
    assert site["schema_version"] == "game-sim-site-v2"
    assert site["model_config"]["simulation_status"] == "calibrated_copula"
    assert site["game_count"] == 2 and len(site["player_summary"]) == 28
    (tmp_path / "game_predictions_2024_wk3.json").write_text(games.to_json(orient="records"))
    (tmp_path / "weekly_2024_wk3.json").write_text(players.to_json(orient="records"))
    assert generate.generate_week(2024, 3, draws=200, seed=3, write_parquet=False, artifacts=artifacts) is None

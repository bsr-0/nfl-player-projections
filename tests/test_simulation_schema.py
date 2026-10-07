import json

import numpy as np
import pandas as pd
import pytest

from src.models.simulation_io import (
    SITE_SCHEMA_VERSION, build_site_payload, write_simulation_parquet, write_simulation_site_payload,
)
from src.models.simulation_schema import (
    SimulationPayload, draws_frame, game_summary, player_summary, summarize_values, validate_payload,
)


def _payload(**overrides):
    games = pd.DataFrame([{"game_id": "g", "season": 2026, "week": 5, "home_team": "H", "away_team": "A",
                           "served_home_win_prob": .6, "served_margin": 3.0, "served_total": 44.0}])
    players = pd.DataFrame([
        {"game_id": "g", "player_id": pid, "team": team, "position": pos, "served_prediction": pred,
         "role_key": role, "is_cold_start": False}
        for pid, team, pos, pred, role in (("qb", "H", "QB", 18., "home_QB1"), ("wr1", "H", "WR", 14., "home_WR1"),
                                           ("te", "H", "TE", 8., "home_TE1"), ("rb", "A", "RB", 12., "away_RB1"))])
    rng = np.random.default_rng(0)
    draws = {"g": players["served_prediction"].to_numpy() + rng.normal(0, 3, size=(50, 4))}
    fields = dict(seed=7, model_config={"simulation_status": "calibrated_copula"},
                  games=games, players=players, draws=draws)
    fields.update(overrides)
    return SimulationPayload(**fields)


def test_summaries_are_recomputed_from_draws():
    payload = _payload()
    summary = {row["player_id"]: row for row in player_summary(payload)}
    assert summary["qb"]["mean"] == pytest.approx(payload.draws["g"][:, 0].mean())
    assert summary["qb"]["served_prediction"] == 18.0
    values = np.arange(10.0)
    assert summarize_values(values)["p10"] == pytest.approx(0.9)
    assert summarize_values(values)["p90"] == pytest.approx(8.1)


def test_game_summary_sums_player_draws_and_passes_served_game_fields_through():
    payload = _payload()
    game = game_summary(payload)[0]
    assert game["served_margin"] == 3.0
    sums = {(s["kind"], s["side"]): s for s in game["fantasy_sums"]}
    assert sums[("stack", "home")]["player_ids"] == ["qb", "wr1", "te"]
    assert sums[("game_total", "both")]["mean"] == pytest.approx(payload.draws["g"].sum(axis=1).mean())
    assert ("stack", "away") not in sums


@pytest.mark.parametrize("change, message", [
    (lambda p: p.players.assign(game_id="missing"), "draws must cover|absent"),
    (lambda p: p.players.assign(team=["H", "H", "H", "Z"]), "not playing"),
    (lambda p: p.players.assign(position=["QB", "WR", "TE", "K"]), "position"),
])
def test_validation_rejects_inconsistent_players(change, message):
    payload = _payload()
    with pytest.raises(ValueError, match=message):
        validate_payload(_payload(players=change(payload)))


def test_validation_rejects_bad_draw_matrices():
    payload = _payload()
    with pytest.raises(ValueError, match="does not match"):
        validate_payload(_payload(draws={"g": payload.draws["g"][:, :3]}))
    bad = payload.draws["g"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        validate_payload(_payload(draws={"g": bad}))


def test_site_json_has_summaries_only_and_parquet_has_all_draws(tmp_path):
    payload = _payload()
    path = write_simulation_site_payload(payload, json_path=tmp_path / "site.json")
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == SITE_SCHEMA_VERSION == build_site_payload(payload)["schema_version"]
    assert "draws" not in saved and saved["draws_per_player"] == 50
    written = write_simulation_parquet(payload, parquet_dir=tmp_path / "pq", stem="sim")
    assert sorted(p.name for p in written) == ["sim_games.parquet", "sim_player_draws.parquet",
                                               "sim_players.parquet"]
    long = pd.read_parquet(tmp_path / "pq" / "sim_player_draws.parquet")
    assert len(long) == 50 * 4
    assert long.equals(draws_frame(payload))


def test_published_week_scores_against_actuals():
    from scripts.evaluate_simulation import evaluate
    payload = _payload()
    actuals = payload.players[["game_id", "player_id"]].assign(fantasy_points=[20.0, 10.0, 8.0, 12.0])
    result = evaluate(draws_frame(payload), payload.players, actuals)
    assert result["marginal"]["n"] == 4
    assert result["joint_player"]["games_scored"] == 1
    assert "mean_variogram_score_p05_scaled" in result["joint_player"]
    with pytest.raises(ValueError, match="without actual outcomes"):
        evaluate(draws_frame(payload), payload.players, actuals.iloc[:3])

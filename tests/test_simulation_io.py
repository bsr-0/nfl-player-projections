import json
import pytest

from src.models.simulation_io import (
    write_simulation_parquet,
    write_simulation_payload,
    write_simulation_site_payload,
)

def _payload():
    return {
        "schema_version": "game-sim-v1", "seed": 42, "game_count": 1,
        "game_ids": ["g"], "model_config": {},
        "game_draws": [{"game_id": "g", "draw": 0, "home_score": 1.0,
                        "away_score": 0.0, "home_plays": 60.0, "away_plays": 60.0,
                        "home_pass_attempts": 35.0, "away_pass_attempts": 35.0,
                        "home_won": True, "simulated_margin": 1.0,
                        "simulated_total": 1.0}],
        "player_draws": [{"game_id": "g", "draw": 0, "player_id": "p",
                          "team": "H", "position": "QB", "active": True,
                          "fantasy_points": 10.0}],
        "player_summary": [{
            "game_id": "g", "player_id": "p", "team": "H", "position": "QB",
            "draw_count": 1, "mean": 10.0, "median": 10.0, "p10": 10.0,
            "p25": 10.0, "p75": 10.0, "p90": 10.0, "std": 0.0,
            "prob_zero": 0.0, "prob_active": 1.0, "max": 10.0,
        }],
    }

def test_research_writer_is_valid_and_does_not_emit_nan(tmp_path):
    path = tmp_path / "simulation.json"
    written = write_simulation_payload(_payload(), json_path=path)
    assert written == [path]
    assert json.loads(path.read_text())["schema_version"] == "game-sim-v1"

def test_site_writer_excludes_raw_monte_carlo_rows(tmp_path):
    path = tmp_path / "site.json"
    write_simulation_site_payload(_payload(), json_path=path)
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == "game-sim-site-v1"
    assert "player_draws" not in saved and "game_draws" not in saved
    assert saved["player_summary"][0]["mean"] == 10.0

def test_writer_rejects_stale_summary_before_writing(tmp_path):
    payload = _payload()
    payload["player_summary"][0]["mean"] = 999.0
    with pytest.raises(ValueError, match="player_summary"):
        write_simulation_payload(payload, json_path=tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()

def test_parquet_writer_requires_engine_or_writes_tables(tmp_path):
    try:
        written = write_simulation_parquet(_payload(), parquet_dir=tmp_path / "parquet",
                                           stem="simulation")
    except RuntimeError as exc:
        assert "Parquet engine" in str(exc)
    else:
        assert len(written) == 3
        assert all(path.exists() for path in written)

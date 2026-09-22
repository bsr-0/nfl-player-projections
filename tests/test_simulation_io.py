import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.simulation_io import write_simulation_payload

def _payload():
    return {
        "schema_version": "game-sim-v1", "seed": 42, "game_count": 1,
        "game_ids": ["g"], "model_config": {},
        "game_draws": [{"game_id": "g", "draw": 0, "home_score": 1.0,
                        "away_score": 0.0, "home_plays": 60.0, "away_plays": 60.0,
                        "home_pass_attempts": 35.0, "away_pass_attempts": 35.0}],
        "player_draws": [{"game_id": "g", "draw": 0, "player_id": "p",
                          "team": "H", "position": "QB", "active": True,
                          "fantasy_points": 10.0}],
        "player_summary": [{"game_id": "g", "player_id": "p", "team": "H",
                            "position": "QB", "draw_count": 1, "mean": 10.0}],
    }

def test_json_writer_is_valid_and_does_not_emit_nan(tmp_path):
    path = tmp_path / "simulation.json"
    written = write_simulation_payload(_payload(), json_path=path)
    assert written == [path]
    assert json.loads(path.read_text())["schema_version"] == "game-sim-v1"

def test_parquet_writer_requires_engine_or_writes_tables(tmp_path):
    try:
        written = write_simulation_payload(_payload(), json_path=tmp_path / "x.json",
                                           parquet_dir=tmp_path / "parquet")
    except RuntimeError as exc:
        assert "Parquet engine" in str(exc)
    else:
        assert len(written) == 4
        assert all(path.exists() for path in written)

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "generate_draft_data",
    PROJECT_ROOT / "scripts" / "generate_draft_data.py",
)
gdd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gdd
spec.loader.exec_module(gdd)


def test_generate_model_performance_applies_authoritative_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(gdd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gdd, "BACKTEST_DIR", tmp_path / "backtest_results")
    monkeypatch.setattr(gdd, "_load_latest_backtest_json", lambda season=None: None)
    monkeypatch.setattr(gdd, "_load_authoritative_position_map", lambda: {"p1": "TE"})
    monkeypatch.setattr(
        gdd,
        "_load_ts_backtest_predictions",
        lambda season=None: pd.DataFrame([
            {
                "player_id": "p1",
                "name": "T.McBride",
                "position": "WR",
                "team": "ARI",
                "predicted": 10.0,
                "actual": 12.0,
            },
            {
                "player_id": "p1",
                "name": "T.McBride",
                "position": "WR",
                "team": "ARI",
                "predicted": 11.0,
                "actual": 13.0,
            },
        ]),
    )

    gdd.generate_model_performance()

    payload = json.loads((tmp_path / "model_performance.json").read_text())
    assert payload["per_player_season_totals"][0]["position"] == "TE"


def test_load_season_data_applies_authoritative_positions(tmp_path, monkeypatch):
    parquet = tmp_path / "daily_predictions.parquet"
    pd.DataFrame([
        {
            "player_id": "p1",
            "name": "Pat Freiermuth",
            "position": "WR",
            "season": 2025,
            "week": 1,
        }
    ]).to_parquet(parquet, index=False)

    monkeypatch.setattr(gdd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gdd, "_load_authoritative_position_map", lambda: {"p1": "TE"})

    df = gdd.load_season_data(2025)

    assert len(df) == 1
    assert df.iloc[0]["position"] == "TE"

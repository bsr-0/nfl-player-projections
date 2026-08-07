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


class TestFloorCeiling:
    """GAPS.md §11.2.C follow-up: asymmetric floor/ceiling replaced the
    symmetric spread formula after real backtest residuals were confirmed
    right-skewed (not classically bimodal, but not symmetric either)."""

    def test_floor_never_exceeds_point_estimate(self):
        for total in [5.0, 50.0, 150.0, 300.0]:
            for position in ["QB", "RB", "WR", "TE"]:
                floor, ceiling = gdd._floor_ceiling(total, 0.7, position)
                assert floor <= total, f"floor {floor} > total {total} for {position}"

    def test_ceiling_never_below_point_estimate(self):
        for total in [5.0, 50.0, 150.0, 300.0]:
            for position in ["QB", "RB", "WR", "TE"]:
                floor, ceiling = gdd._floor_ceiling(total, 0.7, position)
                assert ceiling >= total, f"ceiling {ceiling} < total {total} for {position}"

    def test_floor_never_negative(self):
        for total in [1.0, 5.0, 50.0]:
            floor, _ = gdd._floor_ceiling(total, 0.3, "RB")
            assert floor >= 0.0

    def test_missing_confidence_falls_back_to_default(self):
        floor_a, ceiling_a = gdd._floor_ceiling(100.0, None, "WR")
        floor_b, ceiling_b = gdd._floor_ceiling(100.0, gdd.FLOOR_CEILING_DEFAULT_CONFIDENCE, "WR")
        assert floor_a == floor_b
        assert ceiling_a == ceiling_b

    def test_spread_is_asymmetric_around_point_estimate(self):
        """The whole point of this fix: floor-distance and ceiling-distance
        should generally differ, unlike the old symmetric formula."""
        floor, ceiling = gdd._floor_ceiling(120.0, 0.7, "RB")
        floor_dist = 120.0 - floor
        ceiling_dist = ceiling - 120.0
        assert floor_dist != ceiling_dist

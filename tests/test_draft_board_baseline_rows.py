"""The draft board's prior-season baseline must be built from played games.

daily_predictions.parquet carries the upcoming week's prediction-target rows
(no fantasy_points yet) alongside history. When 2026 gained 880 such stubs,
load_season_data(2026) stopped being empty, the "no data, use last season"
fallback never fired, and the board built every baseline out of stubs --
NaN volatility, and compute_risk_scores' int cast crashed (2026-09-10).
"""
import numpy as np
import pandas as pd

from scripts import generate_draft_data as gdd


def _parquet(tmp_path, monkeypatch, frame):
    path = tmp_path / "daily_predictions.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(gdd, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gdd, "_load_authoritative_position_map", lambda: {})


def test_stub_rows_do_not_count_as_season_data(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "player_id": ["P1", "P1", "P1"],
        "name": ["A.Player"] * 3,
        "season": [2025, 2025, 2026],
        "week": [1, 2, 1],
        "team": ["DEN"] * 3,
        "position": ["WR"] * 3,
        "fantasy_points": [10.0, 20.0, np.nan],      # 2026 wk1 is an unplayed stub
        "utilization_score": [50.0, 60.0, np.nan],
    })
    _parquet(tmp_path, monkeypatch, frame)

    assert gdd.load_season_data(2026).empty
    played = gdd.load_season_data(2025)
    assert len(played) == 2


def test_risk_scores_survive_missing_volatility():
    agg = pd.DataFrame({
        "player_id": ["P1", "P2"], "position": ["RB", "RB"],
        "games_played": [17, 1],
        "vol_mean": [3.0, np.nan], "cv_mean": [0.4, np.nan], "consistency_mean": [0.8, np.nan],
    })
    out = gdd.compute_risk_scores(agg)
    assert out["risk_score"].notna().all()
    assert out["risk_score"].dtype.kind == "i"
    assert out.loc[1, "risk_score"] > out.loc[0, "risk_score"]   # one game is riskier than a full season

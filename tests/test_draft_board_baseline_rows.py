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


def _db_with_weeks(tmp_path, monkeypatch, max_week_by_season):
    """A DB whose player_weekly_stats has rows up to max_week for each season."""
    import sqlite3
    from config import settings
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE player_weekly_stats (player_id TEXT, season INT, week INT)")
    conn.executemany("INSERT INTO player_weekly_stats VALUES (?,?,?)",
                     [("P1", s, w) for s, mw in max_week_by_season.items() for w in range(1, mw + 1)])
    conn.commit(); conn.close()
    monkeypatch.setattr(settings, "DB_PATH", db)


def test_prior_season_is_the_last_completed_one(tmp_path, monkeypatch):
    """Third time this bit: with one week of the new season ingested, the
    board and the Model Performance page both treated 2026 as the prior
    season -- J.Allen's 'season' became one game. A season only qualifies
    once its regular season is fully in the DB."""
    _db_with_weeks(tmp_path, monkeypatch, {2024: 22, 2025: 22, 2026: 1})
    assert gdd._latest_completed_season() == 2025


def test_incomplete_season_never_qualifies_even_if_latest(tmp_path, monkeypatch):
    # 2025 stopped at week 17 (18 needed from 2021 on); 2024 is complete.
    _db_with_weeks(tmp_path, monkeypatch, {2024: 18, 2025: 17})
    assert gdd._latest_completed_season() == 2024


def test_pre_2021_seasons_complete_at_week_17(tmp_path, monkeypatch):
    _db_with_weeks(tmp_path, monkeypatch, {2020: 17, 2021: 17})
    assert gdd._latest_completed_season() == 2020

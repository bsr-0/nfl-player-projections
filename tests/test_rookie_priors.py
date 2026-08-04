"""Tests for Phase 4C rookie-prior fill (scripts/compute_rookie_priors.py
+ FeatureEngineer._apply_rookie_prior).

The rookie prior is supposed to replace the default 0-fill on
``prev_season_ppg`` for rows where the player is in their first
season.  Non-rookie NaN cases (gaps, missed seasons) must NOT be
touched — those still get the 0-fill from ``_impute_missing``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def fe():
    return FeatureEngineer("causal")


def _row(player_id, position, season, week, fp, **kw):
    base = {
        "player_id": player_id,
        "position": position,
        "team": kw.get("team", "X"),
        "opponent": kw.get("opponent", "Y"),
        "season": season,
        "week": week,
        "fantasy_points": fp,
        "home_away": "home",
    }
    base.update(kw)
    return base


def test_rookie_prior_fills_week1_week2(fe, monkeypatch):
    # Stub the priors + draft_round loaders so the test is hermetic.
    monkeypatch.setattr(
        FeatureEngineer, "_ROOKIE_PRIORS_CACHE",
        {"WR": {"rd1": 9.5, "rd2_3": 6.5, "rd4_7": 4.0, "UDFA": 3.5}},
    )
    monkeypatch.setattr(
        FeatureEngineer, "_DRAFT_ROUND_CACHE",
        {"R1_WR": 1, "UDFA_WR": None},
    )

    # Two rookies in 2024: R1_WR (pick-1 WR) and UDFA_WR.
    df = pd.DataFrame([
        _row("R1_WR", "WR", 2024, 1, 5.0),
        _row("R1_WR", "WR", 2024, 2, 12.0),
        _row("R1_WR", "WR", 2024, 3, 8.0),
        _row("UDFA_WR", "WR", 2024, 1, 2.0),
        _row("UDFA_WR", "WR", 2024, 2, 4.0),
    ])
    out = fe._add_prev_season_ppg(df.copy())
    r1 = out[out["player_id"] == "R1_WR"].reset_index(drop=True)
    udfa = out[out["player_id"] == "UDFA_WR"].reset_index(drop=True)
    # Week 1 and 2 are NaN from the shift(1).shift(1) pattern → filled.
    assert r1.loc[0, "prev_season_ppg"] == 9.5
    assert r1.loc[1, "prev_season_ppg"] == 9.5
    # Week 3 of rookie year is the expanding mean from W1→W2 of season
    # one row ago, i.e. FP of W1 = 5.0.  NOT filled by the prior
    # (already non-NaN).
    assert r1.loc[2, "prev_season_ppg"] == 5.0
    # UDFA bucket
    assert udfa.loc[0, "prev_season_ppg"] == 3.5
    assert udfa.loc[1, "prev_season_ppg"] == 3.5


def test_non_rookie_nan_not_touched(fe, monkeypatch):
    monkeypatch.setattr(
        FeatureEngineer, "_ROOKIE_PRIORS_CACHE",
        {"RB": {"rd1": 13.0, "rd2_3": 7.5, "rd4_7": 4.5, "UDFA": 5.0}},
    )
    monkeypatch.setattr(FeatureEngineer, "_DRAFT_ROUND_CACHE", {"VET_RB": 2})

    # Veteran RB: 2023 rookie season, plays 2024 too.  Only earliest
    # season's rows should be flagged as rookie by the helper.
    df = pd.DataFrame([
        _row("VET_RB", "RB", 2023, 1, 10.0),
        _row("VET_RB", "RB", 2023, 2, 12.0),
        _row("VET_RB", "RB", 2024, 1, 14.0),
        _row("VET_RB", "RB", 2024, 2, 16.0),
    ])
    out = fe._add_prev_season_ppg(df.copy())
    out_2024 = out[out["season"] == 2024].reset_index(drop=True)
    # Week 1 of 2024: the shift(1) of season_expanding_ppg lands on
    # the last row of 2023 (week 2), whose expanding mean through
    # shift-1 = FP of W1 of 2023 = 10.0.  Not NaN, not the rookie
    # prior — the player already has prior-season data.
    assert out_2024.loc[0, "prev_season_ppg"] == 10.0
    # Rookie-year rows (2023) still get the rd2_3 prior at W1/W2
    out_2023 = out[out["season"] == 2023].reset_index(drop=True)
    assert out_2023.loc[0, "prev_season_ppg"] == 7.5
    assert out_2023.loc[1, "prev_season_ppg"] == 7.5


def test_missing_priors_file_is_noop(fe, monkeypatch):
    # No priors available → _apply_rookie_prior is a no-op and leaves
    # NaN for downstream _impute_missing to handle.
    monkeypatch.setattr(FeatureEngineer, "_ROOKIE_PRIORS_CACHE", None)
    def _no_priors(cls):
        return None
    monkeypatch.setattr(FeatureEngineer, "_load_rookie_priors", classmethod(_no_priors))
    df = pd.DataFrame([
        _row("X", "WR", 2024, 1, 5.0),
        _row("X", "WR", 2024, 2, 10.0),
    ])
    out = fe._add_prev_season_ppg(df.copy())
    # W1 still NaN (no prior, no fill).
    assert pd.isna(out.loc[0, "prev_season_ppg"])


def test_round_bucket_classification():
    assert FeatureEngineer._round_bucket_for(1) == "rd1"
    assert FeatureEngineer._round_bucket_for(2) == "rd2_3"
    assert FeatureEngineer._round_bucket_for(3) == "rd2_3"
    assert FeatureEngineer._round_bucket_for(4) == "rd4_7"
    assert FeatureEngineer._round_bucket_for(7) == "rd4_7"
    assert FeatureEngineer._round_bucket_for(None) == "UDFA"


def test_priors_json_shape():
    p = Path(__file__).resolve().parents[1] / "data" / "rookie_priors.json"
    if not p.exists():
        pytest.skip("rookie_priors.json not fitted yet")
    raw = json.loads(p.read_text())
    for pos in ("QB", "RB", "WR", "TE"):
        assert pos in raw
        for bucket in ("rd1", "rd2_3", "rd4_7", "UDFA"):
            assert bucket in raw[pos]
            assert isinstance(raw[pos][bucket], (int, float))
            assert raw[pos][bucket] > 0

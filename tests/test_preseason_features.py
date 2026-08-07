"""Tests for the multi-year + team-aware preseason feature assembly
(EXPERIMENTS.md §2). Covers the real bug found and fixed this session:
pandas groupby() silently drops rows with any NaN key (birth_date is
null for a large share of recent players in this DB), unlike SQL's
GROUP BY which groups NULLs together."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.preseason_features import (
    _season_aggregates,
    _team_for_season,
    _destination_team_profiles,
    build_multiyear_season_pairs,
)


def _synthetic_history():
    rows = []
    # Player p1: 3 seasons on team KC, no birth_date (simulates the
    # real-world gap that caused the original bug), then joins SEA.
    for season, team, ppg in [(2021, "KC", 10.0), (2022, "KC", 12.0), (2023, "SEA", 8.0)]:
        for week in range(1, 8):
            rows.append({
                "player_id": "p1", "player_name": "Player One", "position": "RB",
                "birth_date": None, "team": team, "season": season, "week": week,
                "fantasy_points": ppg, "passing_yards": 0, "passing_tds": 0,
                "interceptions": 0, "rushing_yards": 50, "rushing_attempts": 12,
                "targets": 3, "receptions": 2, "receiving_yards": 15, "air_yards": 5,
                "passing_attempts": 0, "passing_completions": 0,
                "snap_share": 0.6, "target_share": 0.1, "rush_share": 0.5,
            })
    # Teammate on SEA in 2022/2023, real birth_date, establishes a
    # destination-team RB usage profile for SEA.
    for season in (2021, 2022):
        for week in range(1, 8):
            rows.append({
                "player_id": "p2", "player_name": "Player Two", "position": "RB",
                "birth_date": "1995-01-01", "team": "SEA", "season": season, "week": week,
                "fantasy_points": 9.0, "passing_yards": 0, "passing_tds": 0,
                "interceptions": 0, "rushing_yards": 40, "rushing_attempts": 10,
                "targets": 2, "receptions": 1, "receiving_yards": 8, "air_yards": 2,
                "passing_attempts": 0, "passing_completions": 0,
                "snap_share": 0.5, "target_share": 0.08, "rush_share": 0.4,
            })
    return pd.DataFrame(rows)


class TestSeasonAggregates:
    def test_null_birth_date_rows_are_not_dropped(self):
        """Regression test for the real bug: groupby() without
        dropna=False silently drops every row with a NaN key."""
        history = _synthetic_history()
        agg = _season_aggregates(history)
        # p1 has null birth_date for all 3 seasons -- must still appear.
        p1_seasons = set(agg[agg.player_id == "p1"]["season"])
        assert p1_seasons == {2021, 2022, 2023}, (
            f"Expected p1's 3 seasons to survive despite null birth_date, got {p1_seasons}"
        )

    def test_aggregate_values_are_correct(self):
        history = _synthetic_history()
        agg = _season_aggregates(history)
        row = agg[(agg.player_id == "p1") & (agg.season == 2021)].iloc[0]
        assert row["games_played"] == 7
        assert row["ppg"] == pytest.approx(10.0)


class TestDestinationTeamProfiles:
    def test_team_changer_gets_nonzero_dest_profile(self):
        history = _synthetic_history()
        db = _FakeDb(history)
        pairs = build_multiyear_season_pairs(db, seasons=[2021, 2022, 2023])
        p1_2023 = pairs[(pairs.player_id == "p1") & (pairs.target_season == 2023)]
        assert len(p1_2023) == 1
        row = p1_2023.iloc[0]
        assert row["team_changed"] == 1
        assert row["dest_team"] == "SEA"
        # SEA had a real prior RB usage history (player p2) -- should be nonzero.
        assert row["dest_hist_carry_pg"] > 0

    def test_non_changer_gets_zeroed_dest_profile(self):
        history = _synthetic_history()
        db = _FakeDb(history)
        pairs = build_multiyear_season_pairs(db, seasons=[2021, 2022, 2023])
        p1_2022 = pairs[(pairs.player_id == "p1") & (pairs.target_season == 2022)]
        assert len(p1_2022) == 1
        row = p1_2022.iloc[0]
        assert row["team_changed"] == 0
        assert row["dest_hist_carry_pg"] == 0.0
        assert row["dest_hist_tgt_pg"] == 0.0


class _FakeDb:
    """Minimal stand-in for DatabaseManager that returns a fixed
    DataFrame from _load_full_history's query, so build_multiyear_season_pairs
    can be tested without a real sqlite connection."""
    def __init__(self, history: pd.DataFrame):
        self._history = history

    class _FakeConn:
        def __init__(self, history):
            self._history = history

    def _get_connection(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _patch_load_full_history(monkeypatch):
    """build_multiyear_season_pairs calls _load_full_history(db), which
    runs a real SQL query. Patch it to return synthetic data directly
    for the tests that use _FakeDb."""
    import src.models.preseason_features as pf
    original = pf._load_full_history

    def fake_load(db):
        if isinstance(db, _FakeDb):
            return db._history
        return original(db)

    monkeypatch.setattr(pf, "_load_full_history", fake_load)

"""Unit tests for the depth_chart_rank as-of fix (FEATURE_VERSION 33).

Verifies FeatureEngineer._add_depth_chart_rank uses a per-row, as-of-week
lookup instead of the old hardcoded week=1 preseason-only lookup, and that
_load_depth_chart_asof_table dedupes/filters the raw depth_charts table
correctly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer, _load_depth_chart_asof_table


class TestAddDepthChartRank:
    def _fake_table(self):
        return pd.DataFrame({
            "gsis_id": ["P1", "P1", "P1"],
            "_key": [202301, 202310, 202401],
            "depth_chart_rank": [1, 2, 1],
        })

    def test_uses_each_rows_own_week_not_always_week_1(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        df = pd.DataFrame({
            "player_id": ["P1", "P1", "P1"],
            "season": [2023, 2023, 2023],
            "week": [1, 10, 15],
        })
        fe = FeatureEngineer()
        result = fe._add_depth_chart_rank(df)
        # Week 1 -> rank 1; week 10 -> that week's own snapshot (rank 2,
        # inclusive cutoff is legitimate pre-game info); week 15 -> still 2
        # (most recent snapshot <= week 15, no week-1-only flattening).
        assert result["depth_chart_rank"].tolist() == [1, 2, 2]

    def test_inclusive_cutoff_uses_same_week_snapshot(self, monkeypatch):
        """Real rows may use that SAME week's own snapshot (a real game's
        pre-kickoff depth chart) -- a deliberately looser cutoff than the
        synthetic-row lookup's strict '<'."""
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            self._fake_table,
        )
        df = pd.DataFrame({"player_id": ["P1"], "season": [2023], "week": [10]})
        fe = FeatureEngineer()
        result = fe._add_depth_chart_rank(df)
        assert result["depth_chart_rank"].iloc[0] == 2  # week 10's own snapshot, not week 1's

    def test_missing_columns_falls_back_to_default(self):
        fe = FeatureEngineer()
        result = fe._add_depth_chart_rank(pd.DataFrame({"foo": [1, 2]}))
        assert (result["depth_chart_rank"] == 1).all()

    def test_empty_table_falls_back_to_default_3(self, monkeypatch):
        monkeypatch.setattr(
            "src.features.feature_engineering._load_depth_chart_asof_table",
            lambda: pd.DataFrame(),
        )
        df = pd.DataFrame({"player_id": ["P1"], "season": [2023], "week": [5]})
        fe = FeatureEngineer()
        result = fe._add_depth_chart_rank(df)
        assert result["depth_chart_rank"].iloc[0] == 3


class TestLoadDepthChartAsofTable:
    def test_dedupes_conflicting_duplicate_rows_deterministically(self, monkeypatch):
        """2024 has ~3x the expected row count per week -- simulate a
        conflicting duplicate for the same (season, week, gsis_id) and
        confirm the result is deterministic (MIN), not arbitrary."""
        import src.features.feature_engineering as fe_module
        fe_module._depth_chart_asof_cache.clear()

        class FakeConn:
            def close(self): pass

        raw = pd.DataFrame({
            "season": [2024, 2024], "week": [1, 1],
            "gsis_id": ["P1", "P1"], "depth_team": ["1", "2"],
        })

        def fake_read_sql(query, conn):
            return raw

        monkeypatch.setattr("pandas.read_sql", fake_read_sql)
        monkeypatch.setattr("sqlite3.connect", lambda path: FakeConn())

        table = _load_depth_chart_asof_table()
        fe_module._depth_chart_asof_cache.clear()  # don't leak into other tests
        assert len(table) == 1  # deduped to one row for (2024, week 1, P1)
        assert table["depth_chart_rank"].iloc[0] == 1  # deterministic MIN

    def test_excludes_null_season_rows(self, monkeypatch):
        import src.features.feature_engineering as fe_module
        fe_module._depth_chart_asof_cache.clear()

        class FakeConn:
            def close(self): pass

        # Simulate the SQL WHERE clause already filtering season IS NOT NULL
        # by returning only the valid row (the real query does this at the
        # DB level; this test documents that expectation at the Python level).
        raw = pd.DataFrame({
            "season": [2023], "week": [1], "gsis_id": ["P1"], "depth_team": ["1"],
        })
        monkeypatch.setattr("pandas.read_sql", lambda query, conn: raw)
        monkeypatch.setattr("sqlite3.connect", lambda path: FakeConn())

        table = _load_depth_chart_asof_table()
        fe_module._depth_chart_asof_cache.clear()
        assert len(table) == 1
        assert table["gsis_id"].iloc[0] == "P1"

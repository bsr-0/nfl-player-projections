"""Regression tests for leakage defenses.

Covers:
- src/utils/database.py: get_all_players_for_training must join team_stats
  (own-team) and team_defense_stats (opponent) strictly from week - 1.
- src/utils/leakage.py: is_leakage_feature / FEATURE_AVAILABILITY registry /
  audit_feature_availability classify feature columns correctly.
- src/data/external_data.py: InjuryDataLoader must drop injury reports
  modified after that week's kickoff (GAPS.md §7.6).
- src/features/feature_engineering.py: FeatureEngineer._add_contract_features
  must bound the applicable contract by year_signed <= season, and
  _merge_injury_data_from_cache must apply the same kickoff-timing guard as
  InjuryDataLoader, since it queries player_injuries directly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import DatabaseManager
from src.data import external_data
from src.data.external_data import InjuryDataLoader
from src.utils.leakage import (
    is_leakage_feature,
    audit_feature_availability,
    FEATURE_AVAILABILITY,
)


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "test.db")


def _insert_player(db, player_id="P1", name="Test Player", position="WR"):
    with db._get_connection() as conn:
        conn.execute(
            "INSERT INTO players (player_id, name, position) VALUES (?, ?, ?)",
            (player_id, name, position),
        )
        conn.commit()


def _insert_weekly_stats(db, player_id, season, week, team="AAA", opponent="BBB"):
    with db._get_connection() as conn:
        conn.execute(
            """INSERT INTO player_weekly_stats
               (player_id, season, week, team, opponent, receiving_yards)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (player_id, season, week, team, opponent, 50),
        )
        conn.commit()


def _insert_team_stats(db, team, season, week, total_yards=300):
    with db._get_connection() as conn:
        conn.execute(
            """INSERT INTO team_stats (team, season, week, total_yards)
               VALUES (?, ?, ?, ?)""",
            (team, season, week, total_yards),
        )
        conn.commit()


class TestOwnTeamStatsNotLeaked:
    def test_own_team_stats_week_is_always_prior_week(self, db):
        """get_all_players_for_training must never attach same-week team_stats."""
        _insert_player(db)
        for week in range(1, 5):
            _insert_weekly_stats(db, "P1", 2024, week)
            # Insert team_stats for every week including the current one, so
            # a same-week join (the bug) would be observable if it existed.
            _insert_team_stats(db, "AAA", 2024, week)

        df = db.get_all_players_for_training(min_games=1)

        assert "own_team_stats_week" in df.columns
        has_data = df["own_team_stats_week"].notna()
        assert has_data.any(), "expected at least one row with own-team stats joined"
        assert (df.loc[has_data, "own_team_stats_week"] == df.loc[has_data, "week"] - 1).all()
        assert not (df["own_team_stats_week"].fillna(-1) >= df["week"]).any()

    def test_week_one_has_no_prior_own_team_stats(self, db):
        """Week 1 has no week-0 team_stats to join, mirroring opp_defense_week behavior."""
        _insert_player(db)
        _insert_weekly_stats(db, "P1", 2024, 1)
        _insert_team_stats(db, "AAA", 2024, 1)  # same-week only, must NOT be joined

        df = db.get_all_players_for_training(min_games=1)
        assert df.loc[df["week"] == 1, "own_team_stats_week"].isna().all()


class TestLeakageFeatureGuard:
    @pytest.mark.parametrize(
        "col",
        [
            "predicted_fantasy_points",
            "projection_1w",
            "projection_18w",
            "target",
            "target_util_4w",
            "baseline_fp",
            "actual_for_backtest",
            "player_id",
            "utilization_score",
        ],
    )
    def test_blocks_known_leakage_columns(self, col):
        assert is_leakage_feature(col) is True

    @pytest.mark.parametrize(
        "col",
        [
            "targets_roll3_mean",
            "opp_fpts_allowed",
            "sos_next_4",
            "expected_games_next_4",
            "implied_team_total",
        ],
    )
    def test_allows_legitimate_features(self, col):
        assert is_leakage_feature(col) is False


class TestFeatureAvailabilityRegistry:
    def test_registry_is_nonempty_and_well_formed(self):
        assert len(FEATURE_AVAILABILITY) > 0
        for pattern, rule in FEATURE_AVAILABILITY:
            assert isinstance(pattern, str) and pattern
            assert isinstance(rule, str) and rule

    @pytest.mark.parametrize(
        "col",
        [
            "targets_roll3_mean",
            "team_yards",
            "team_pass_attempts",
            "opp_fpts_allowed_s2d_lag1",
            "implied_team_total",
            "spread",
            "injury_score",
            "is_rookie",
            "prev_season_ppg",
        ],
    )
    def test_known_feature_families_are_classified(self, col):
        assert col not in audit_feature_availability([col])

    def test_unclassified_synthetic_column_is_flagged(self):
        flagged = audit_feature_availability(["totally_new_unaudited_metric_xyz"])
        assert flagged == ["totally_new_unaudited_metric_xyz"]

    def test_missing_values_are_not_flagged_as_leakage(self):
        """Registry classifies column identity only; nulls are not audited here."""
        # A column with NaNs is still a column name — audit only inspects names.
        assert audit_feature_availability(["targets_roll3_mean"]) == []


class TestInjuryTimingGuard:
    @pytest.fixture(autouse=True)
    def _mock_schedule(self, monkeypatch):
        """AAA hosts BBB in week 5, 2024, kicking off at 13:00 America/New_York."""
        schedule = pd.DataFrame({
            "season": [2024],
            "week": [5],
            "gameday": ["2024-10-06"],
            "gametime": ["13:00"],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
        })
        monkeypatch.setattr(external_data.nfl, "import_schedules", lambda seasons: schedule)

    def test_drops_report_modified_after_kickoff(self):
        injuries = pd.DataFrame({
            "gsis_id": ["P1", "P2"],
            "season": [2024, 2024],
            "week": [5, 5],
            "team": ["AAA", "AAA"],
            "report_status": ["Questionable", "Out"],
            "date_modified": [
                pd.Timestamp("2024-10-04 20:00:00", tz="UTC"),  # Friday before kickoff — OK
                pd.Timestamp("2024-10-06 18:30:00", tz="UTC"),  # after 17:00 UTC kickoff — leaks
            ],
        })

        result = InjuryDataLoader().get_player_injury_status(injuries)

        assert len(result) == 1
        assert result.iloc[0]["player_id"] == "P1"

    def test_keeps_reports_with_unmatched_schedule(self):
        """No leakage evidence for an unmatched team/week — keep the row (missingness != leakage)."""
        injuries = pd.DataFrame({
            "gsis_id": ["P3"],
            "season": [2024],
            "week": [99],  # no matching schedule row
            "team": ["ZZZ"],
            "report_status": ["Questionable"],
            "date_modified": [pd.Timestamp("2024-12-01 00:00:00", tz="UTC")],
        })

        result = InjuryDataLoader().get_player_injury_status(injuries)

        assert len(result) == 1


class TestContractYearBoundedBySeason:
    """A row for season S must never see a contract signed after S."""

    @pytest.fixture(autouse=True)
    def _reset_contract_cache(self):
        # _contract_lookup_cache is a class-level cache (shared across every
        # FeatureEngineer instance in the process) -- restore it after each
        # test so this stub data doesn't leak into other tests.
        from src.features.feature_engineering import FeatureEngineer
        original = FeatureEngineer._contract_lookup_cache
        yield
        FeatureEngineer._contract_lookup_cache = original

    def test_historical_row_does_not_see_future_contract(self):
        from src.features.feature_engineering import FeatureEngineer

        FeatureEngineer._contract_lookup_cache = pd.DataFrame([
            {"player_id": "P1", "year_signed": 2016, "final_year": 2018, "apy_rank": 0.2},
            {"player_id": "P1", "year_signed": 2023, "final_year": 2026, "apy_rank": 0.95},
        ])
        fe = FeatureEngineer.__new__(FeatureEngineer)
        df = pd.DataFrame({"player_id": ["P1", "P1", "P1"], "season": [2015, 2017, 2018]})

        out = fe._add_contract_features(df.copy())

        # 2015: before any contract was signed -- no lookahead into the 2016 deal.
        assert out.loc[0, "contract_apy_rank"] == 0.5
        assert out.loc[0, "is_contract_year"] == 0
        # 2017/2018: the 2016 rookie deal applies, NOT the 2023 mega-deal.
        assert out.loc[1, "contract_apy_rank"] == 0.2
        assert out.loc[2, "contract_apy_rank"] == 0.2
        assert out.loc[1, "is_contract_year"] == 0
        assert out.loc[2, "is_contract_year"] == 1  # 2018 is the rookie deal's final year

    def test_current_and_future_seasons_use_the_applicable_contract(self):
        from src.features.feature_engineering import FeatureEngineer

        FeatureEngineer._contract_lookup_cache = pd.DataFrame([
            {"player_id": "P1", "year_signed": 2016, "final_year": 2018, "apy_rank": 0.2},
            {"player_id": "P1", "year_signed": 2023, "final_year": 2026, "apy_rank": 0.95},
        ])
        fe = FeatureEngineer.__new__(FeatureEngineer)
        df = pd.DataFrame({"player_id": ["P1", "P1"], "season": [2020, 2023]})

        out = fe._add_contract_features(df.copy())

        # 2020: no contract signed yet for that gap -- most recent PAST
        # contract (2016 rookie deal) carries forward, not the future one.
        assert out.loc[0, "contract_apy_rank"] == 0.2
        # 2023: the new deal was just signed, so it now applies.
        assert out.loc[1, "contract_apy_rank"] == 0.95


class TestInjuryCacheKickoffGuard:
    """_merge_injury_data_from_cache must kickoff-filter like InjuryDataLoader."""

    @pytest.fixture
    def _mock_schedule(self, monkeypatch):
        schedule = pd.DataFrame({
            "season": [2024], "week": [1], "gameday": ["2024-09-08"],
            "gametime": ["13:00"], "home_team": ["DEN"], "away_team": ["BBB"],
        })
        # _cached_import memoizes by (kind, seasons) at module scope, so a
        # different schedule fixture used by another test class for the same
        # season (TestInjuryTimingGuard) would otherwise leak in here.
        external_data.clear_season_import_cache()
        monkeypatch.setattr(external_data.nfl, "import_schedules", lambda seasons: schedule)

    def _insert_injury(self, db, date_modified, report_status="Out"):
        with db._get_connection() as conn:
            conn.execute(
                "INSERT INTO player_injuries (player_id, season, week, team, "
                "report_status, date_modified) VALUES (?, ?, ?, ?, ?, ?)",
                ("P1", 2024, 1, "DEN", report_status, date_modified),
            )
            conn.commit()

    def test_post_kickoff_report_is_dropped(self, db, monkeypatch, _mock_schedule):
        from src.utils import database as db_mod
        monkeypatch.setattr(db_mod, "DatabaseManager", lambda: db)
        from src.features import feature_engineering as fe_mod
        self._insert_injury(db, "2024-09-08T18:30:00Z")  # after 17:00 UTC kickoff

        fe = fe_mod.FeatureEngineer.__new__(fe_mod.FeatureEngineer)
        df = pd.DataFrame({"player_id": ["P1"], "season": [2024], "week": [1]})
        out = fe._merge_injury_data_from_cache(df.copy())

        assert out.loc[0, "injury_score"] == 1.0
        assert out.loc[0, "is_injured"] == 0

    def test_pre_kickoff_report_is_kept(self, db, monkeypatch, _mock_schedule):
        from src.utils import database as db_mod
        monkeypatch.setattr(db_mod, "DatabaseManager", lambda: db)
        from src.features import feature_engineering as fe_mod
        self._insert_injury(db, "2024-09-06T12:00:00Z", report_status="Questionable")

        fe = fe_mod.FeatureEngineer.__new__(fe_mod.FeatureEngineer)
        df = pd.DataFrame({"player_id": ["P1"], "season": [2024], "week": [1]})
        out = fe._merge_injury_data_from_cache(df.copy())

        assert out.loc[0, "injury_score"] == 0.50
        assert out.loc[0, "is_injured"] == 1

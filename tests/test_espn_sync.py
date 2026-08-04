"""Tests for ESPN sync + normalization layer."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

from src.integrations.espn_sync import (
    ESPNSyncService,
    LeagueState,
    LeagueSettings,
    PlayerState,
    OpponentState,
    SLOT_MAP,
)


@pytest.fixture
def mock_connector():
    """Mock ESPNFantasyConnector with realistic data."""
    connector = MagicMock()
    connector.connected = True
    connector.year = 2025
    connector.league = MagicMock()

    connector.get_league_info.return_value = {
        "name": "Test League",
        "year": 2025,
        "num_teams": 10,
        "current_week": 3,
    }

    connector.get_my_team.return_value = {
        "team_name": "My Team",
        "team_id": 1,
        "wins": 2,
        "losses": 0,
        "roster": [
            {"name": "Patrick Mahomes", "position": "QB", "team": "KC",
             "projected_points": 22.5, "slot": "QB", "injury_status": "Active"},
            {"name": "Saquon Barkley", "position": "RB", "team": "PHI",
             "projected_points": 18.2, "slot": "RB", "injury_status": "Active"},
            {"name": "Derrick Henry", "position": "RB", "team": "BAL",
             "projected_points": 15.8, "slot": "RB", "injury_status": "Active"},
            {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN",
             "projected_points": 19.1, "slot": "WR", "injury_status": "Active"},
            {"name": "CeeDee Lamb", "position": "WR", "team": "DAL",
             "projected_points": 17.4, "slot": "WR", "injury_status": "Questionable"},
            {"name": "Travis Kelce", "position": "TE", "team": "KC",
             "projected_points": 14.2, "slot": "TE", "injury_status": "Active"},
            {"name": "Bijan Robinson", "position": "RB", "team": "ATL",
             "projected_points": 16.5, "slot": "RB/WR/TE", "injury_status": "Active"},
            {"name": "Chris Olave", "position": "WR", "team": "NO",
             "projected_points": 12.3, "slot": "BE", "injury_status": "Active"},
            {"name": "Tyler Bass", "position": "K", "team": "BUF",
             "projected_points": 8.0, "slot": "K", "injury_status": "Active"},
        ],
    }

    connector.get_free_agents.return_value = [
        {"name": "Tank Dell", "position": "WR", "team": "HOU",
         "projected_points": 11.0, "percent_owned": 45.0},
        {"name": "Jaylen Warren", "position": "RB", "team": "PIT",
         "projected_points": 9.5, "percent_owned": 30.0},
    ]

    return connector


@pytest.fixture
def mock_predictor():
    """Mock NFLPredictor returning sample predictions."""
    predictor = MagicMock()
    predictor.predict_next_week.return_value = pd.DataFrame([
        {"player_id": "00-0036355", "name": "Patrick Mahomes", "position": "QB",
         "predicted_ppg": 23.1, "prediction_ci80_lower": 16.0, "prediction_ci80_upper": 30.0},
        {"player_id": "00-0036252", "name": "Saquon Barkley", "position": "RB",
         "predicted_ppg": 17.5, "prediction_ci80_lower": 10.0, "prediction_ci80_upper": 25.0},
        {"player_id": "00-0039337", "name": "Ja'Marr Chase", "position": "WR",
         "predicted_ppg": 20.2, "prediction_ci80_lower": 13.0, "prediction_ci80_upper": 27.0},
    ])
    return predictor


@pytest.fixture
def mock_db_players():
    """Sample players DataFrame from DB."""
    return pd.DataFrame([
        {"player_id": "00-0036355", "name": "Patrick Mahomes", "team": "KC", "position": "QB"},
        {"player_id": "00-0036252", "name": "Saquon Barkley", "team": "PHI", "position": "RB"},
        {"player_id": "00-0033293", "name": "Derrick Henry", "team": "BAL", "position": "RB"},
        {"player_id": "00-0039337", "name": "Ja'Marr Chase", "team": "CIN", "position": "WR"},
        {"player_id": "00-0039338", "name": "CeeDee Lamb", "team": "DAL", "position": "WR"},
        {"player_id": "00-0033118", "name": "Travis Kelce", "team": "KC", "position": "TE"},
        {"player_id": "00-0039220", "name": "Bijan Robinson", "team": "ATL", "position": "RB"},
        {"player_id": "00-0037675", "name": "Chris Olave", "team": "NO", "position": "WR"},
        {"player_id": "00-0039900", "name": "Tank Dell", "team": "HOU", "position": "WR"},
    ])


class TestPlayerState:
    def test_is_resolved_with_id(self):
        p = PlayerState(
            player_id="00-0036355", name="Test", team="KC", position="QB",
            slot="QB", proj_points=20.0, espn_proj_points=18.0,
            confidence=0.7, volatility=4.0, injury_status="Active",
            percent_owned=0.0, edge=2.0,
        )
        assert p.is_resolved()

    def test_is_resolved_without_id(self):
        p = PlayerState(
            player_id="", name="Test", team="KC", position="QB",
            slot="QB", proj_points=20.0, espn_proj_points=18.0,
            confidence=0.5, volatility=5.0, injury_status="Active",
            percent_owned=0.0, edge=2.0,
        )
        assert not p.is_resolved()


class TestSlotMapping:
    def test_flex_variants(self):
        assert SLOT_MAP["RB/WR/TE"] == "FLEX"
        assert SLOT_MAP["RB/WR"] == "FLEX"
        assert SLOT_MAP["WR/TE"] == "FLEX"

    def test_bench(self):
        assert SLOT_MAP["BE"] == "BENCH"

    def test_standard_positions(self):
        assert SLOT_MAP["QB"] == "QB"
        assert SLOT_MAP["RB"] == "RB"
        assert SLOT_MAP["WR"] == "WR"
        assert SLOT_MAP["TE"] == "TE"


class TestESPNSyncService:
    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_sync_produces_league_state(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        assert isinstance(state, LeagueState)
        assert state.week == 3
        assert state.season == 2025
        assert state.sync_timestamp != ""

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_starters_vs_bench_split(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        # K is filtered out, so 8 players total (7 starters + 1 bench)
        assert len(state.roster) == 8
        assert len(state.starters) == 7
        assert len(state.bench) == 1
        assert state.bench[0].name == "Chris Olave"

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_kicker_filtered_out(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        names = [p.name for p in state.roster]
        assert "Tyler Bass" not in names

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_player_resolution(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        mahomes = next(p for p in state.roster if "Mahomes" in p.name)
        assert mahomes.player_id == "00-0036355"
        assert mahomes.is_resolved()

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_model_projection_attached(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        mahomes = next(p for p in state.roster if "Mahomes" in p.name)
        # Model says 23.1, ESPN says 22.5
        assert mahomes.proj_points == 23.1
        assert mahomes.espn_proj_points == 22.5
        assert mahomes.edge == pytest.approx(0.6, abs=0.01)

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_espn_fallback_when_no_prediction(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        # Henry is in roster but not in mock predictions
        henry = next(p for p in state.roster if "Henry" in p.name)
        # Falls back to ESPN projection
        assert henry.proj_points == 15.8

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_no_predictor_uses_espn_projections(self, mock_db_cls, mock_connector, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=None)
        state = service.sync(team_id=1)

        mahomes = next(p for p in state.roster if "Mahomes" in p.name)
        assert mahomes.proj_points == 22.5  # ESPN fallback

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_settings_extraction(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        assert state.settings.num_teams == 10
        assert state.settings.league_name == "Test League"
        assert state.settings.scoring_format in ("ppr", "half_ppr", "standard")

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_free_agents_synced(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        # Free agents fetched (4 positions * return of 2 each = 8, but mock returns same 2)
        assert len(state.free_agents) > 0

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_resolution_rate(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        # All players in mock_db_players should resolve
        assert state.resolution_rate == 1.0

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_unresolved_player_tracked(self, mock_db_cls, mock_connector, mock_predictor):
        # DB missing some players
        incomplete_db = pd.DataFrame([
            {"player_id": "00-0036355", "name": "Patrick Mahomes", "team": "KC", "position": "QB"},
        ])
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = incomplete_db
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        assert len(state.unresolved_players) > 0
        assert state.resolution_rate < 1.0

    def test_not_connected_raises(self, mock_connector):
        mock_connector.connected = False
        service = ESPNSyncService(mock_connector)
        with pytest.raises(RuntimeError, match="not connected"):
            service.sync(team_id=1)

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_injury_status_preserved(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        lamb = next(p for p in state.roster if "Lamb" in p.name)
        assert lamb.injury_status == "Questionable"

    @patch("src.integrations.espn_sync.DatabaseManager")
    def test_flex_slot_mapped(self, mock_db_cls, mock_connector, mock_predictor, mock_db_players):
        mock_db_instance = MagicMock()
        mock_db_instance.get_all_players.return_value = mock_db_players
        mock_db_cls.return_value = mock_db_instance

        service = ESPNSyncService(mock_connector, predictor=mock_predictor)
        state = service.sync(team_id=1)

        bijan = next(p for p in state.roster if "Robinson" in p.name)
        assert bijan.slot == "FLEX"


class TestLeagueState:
    def test_roster_projected_total(self):
        starters = [
            PlayerState("id1", "P1", "KC", "QB", "QB", 20.0, 18.0, 0.7, 4.0, "Active", 0, 2.0),
            PlayerState("id2", "P2", "PHI", "RB", "RB", 15.0, 14.0, 0.6, 5.0, "Active", 0, 1.0),
        ]
        state = LeagueState(
            roster=starters, starters=starters, bench=[], free_agents=[],
            opponent=None, settings=LeagueSettings("ppr", {}, 10, "Test"),
            week=1, season=2025, sync_timestamp="2025-01-01T00:00:00Z",
        )
        assert state.roster_projected_total == 35.0

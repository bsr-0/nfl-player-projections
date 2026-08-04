"""Targeted tests for training/prediction pipeline integration updates."""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import train as train_module
from src.predict import NFLPredictor
from src.data.external_data import ExternalDataIntegrator, WeatherDataLoader


def test_train_helper_applies_advanced_features(monkeypatch):
    """`add_advanced_features` should enrich data when module is available."""
    base = pd.DataFrame({"player_id": ["p1"], "position": ["RB"]})

    def _fake_add(df):
        out = df.copy()
        out["advanced_flag"] = 1
        return out

    monkeypatch.setattr(
        "src.features.advanced_rookie_injury.add_advanced_rookie_injury_features",
        _fake_add,
    )
    out = train_module.add_advanced_features(base)
    assert "advanced_flag" in out.columns
    assert out["advanced_flag"].iloc[0] == 1


def test_predict_prepare_features_keeps_advanced_parity(monkeypatch):
    """Prediction feature preparation should include advanced rookie/injury columns."""
    predictor = NFLPredictor()
    predictor.utilization_calculator.calculate_all_scores = lambda df, _: df.assign(utilization_score=55.0)
    predictor.feature_engineer.create_features = lambda df, include_target=False: df.assign(base_feature=1.0)

    def _fake_add(df):
        out = df.copy()
        out["injury_prob_advanced"] = 0.05
        return out

    monkeypatch.setattr(
        "src.features.advanced_rookie_injury.add_advanced_rookie_injury_features",
        _fake_add,
    )

    inp = pd.DataFrame({"player_id": ["p1"], "name": ["Player One"], "position": ["RB"], "team": ["KC"]})
    out = predictor._prepare_features(inp)
    assert "injury_prob_advanced" in out.columns


def test_train_bounded_scaler_artifact_roundtrip(tmp_path):
    train_df = pd.DataFrame({
        "snap_share_pct": [10.0, 20.0, 40.0, 60.0],
        "target_rate": [0.1, 0.2, 0.3, 0.4],
        "unbounded_feature": [1000, 950, 1100, 1050],
    })
    test_df = pd.DataFrame({
        "snap_share_pct": [30.0, 50.0],
        "target_rate": [0.15, 0.35],
        "unbounded_feature": [980, 1020],
    })
    artifact_path = tmp_path / "feature_scaler_bounded.joblib"

    artifact = train_module._apply_bounded_scaling(train_df, test_df, artifact_path)

    assert artifact_path.exists()
    assert "snap_share_pct" in artifact["columns"]
    assert "target_rate" in artifact["columns"]
    assert "unbounded_feature" not in artifact["columns"]
    assert float(train_df["snap_share_pct"].min()) == 0.0
    assert float(train_df["snap_share_pct"].max()) == 1.0


def test_predict_prepare_features_applies_bounded_scaler(monkeypatch):
    predictor = NFLPredictor()
    predictor.utilization_calculator.calculate_all_scores = lambda df, _: df.assign(utilization_score=55.0)
    predictor.feature_engineer.create_features = lambda df, include_target=False: df.assign(base_feature=1.0)
    predictor.bounded_scaler_artifact = {
        "columns": ["snap_share_pct"],
        "scaler": type(
            "_FakeScaler",
            (),
            {"transform": staticmethod(lambda arr: arr / 100.0)},
        )(),
    }

    monkeypatch.setattr(
        "src.features.advanced_rookie_injury.add_advanced_rookie_injury_features",
        lambda df: df,
    )
    monkeypatch.setattr("src.data.external_data.add_external_features", lambda df, seasons=None: df)
    monkeypatch.setattr("src.features.season_long_features.add_season_long_features", lambda df: df)

    inp = pd.DataFrame({
        "player_id": ["p1"],
        "name": ["Player One"],
        "position": ["RB"],
        "team": ["KC"],
        "season": [2024],
        "snap_share_pct": [80.0],
    })
    out = predictor._prepare_features(inp)
    assert float(out["snap_share_pct"].iloc[0]) == 0.8


def test_external_injury_fallback_by_name(monkeypatch):
    """When player IDs mismatch, injury merge should fallback to normalized name."""
    integrator = ExternalDataIntegrator()
    base = pd.DataFrame({
        "player_id": ["db_id_1"],
        "name": ["John Doe"],
        "position": ["RB"],
        "team": ["KC"],
        "season": [2024],
        "week": [5],
        "opponent": ["SF"],
        "home_away": ["home"],
    })

    monkeypatch.setattr(integrator.injury_loader, "load_injuries", lambda seasons: pd.DataFrame({"dummy": [1]}))
    monkeypatch.setattr(
        integrator.injury_loader,
        "get_player_injury_status",
        lambda _: pd.DataFrame({
            "player_id": ["api_id_9"],
            "player_name": ["John Doe"],
            "season": [2024],
            "week": [5],
            "injury_score": [0.5],
            "is_injured": [1],
        }),
    )
    monkeypatch.setattr(integrator.defense_loader, "get_opponent_matchup_features", lambda df: df)
    monkeypatch.setattr(integrator.weather_loader, "load_weather_data", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(integrator.weather_loader, "get_weather_features", lambda df, schedules: df.assign(is_dome=0, is_outdoor=1, weather_score=0.9))
    monkeypatch.setattr(integrator.vegas_loader, "load_vegas_lines", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(integrator.vegas_loader, "get_vegas_features", lambda df, lines: df.assign(implied_team_total=23.0, game_total=46.0, spread=0.0))

    out = integrator.add_all_external_features(base, seasons=[2024])
    assert float(out["injury_score"].iloc[0]) == 0.5
    assert int(out["is_injured"].iloc[0]) == 1


def test_weather_loader_uses_real_schedule_columns():
    """Weather loader should consume schedule temp/wind/weather when present."""
    loader = WeatherDataLoader()
    players = pd.DataFrame({
        "season": [2024],
        "week": [3],
        "team": ["KC"],
        "opponent": ["SF"],
        "home_away": ["home"],
    })
    schedules = pd.DataFrame({
        "season": [2024],
        "week": [3],
        "home_team": ["KC"],
        "away_team": ["SF"],
        "temp": [28],
        "wind": [19],
        "weather": ["rain showers"],
    })
    out = loader.get_weather_features(players, schedules=schedules)
    assert "weather_temp_f" in out.columns
    assert "weather_wind_mph" in out.columns
    assert "is_cold_game" in out.columns
    assert int(out["is_rain_game"].iloc[0]) == 1

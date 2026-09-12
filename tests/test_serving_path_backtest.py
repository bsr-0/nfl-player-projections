"""backtester.run_backtest walks the SERVING path and scores raw outcomes.

End-to-end with a private DB and a fake predictor: the walk-forward must call
predict(as_of=(season, week)) once per played week, score each prediction
against THAT week's raw points (a prediction made as_of week w is for week w),
use the prior season's rows only as baseline history, and write a trusted,
model-identified artifact through the same save/gate path as training.
"""
import json

import pandas as pd
import pytest

import src.evaluation.backtester as bt
import src.utils.database as db_mod
from src.utils.data_manager import DataManager
from src.utils.database import DatabaseManager

# 40 players x 3 weeks = 120 scored predictions: the publish gate refuses an
# artifact with fewer than 100 as a tiny sample (see assess_artifact_trust).
PLAYERS = [(f"P{i:02d}", ["WR", "RB", "QB", "TE"][i % 4]) for i in range(40)]
WEEKS = [1, 2, 3]


@pytest.fixture
def private_db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)          # DatabaseManager() default
    monkeypatch.setattr(bt, "DATA_DIR", tmp_path)          # artifacts + app payload + charts
    monkeypatch.setattr(bt, "MODELS_DIR", tmp_path)        # no model_metadata.json here
    db = DatabaseManager(db_path=path)
    for pid, pos in PLAYERS:
        db.insert_player({"player_id": pid, "name": f"Player {pid}", "position": pos})
    for season in (2024, 2025):
        for week in WEEKS:
            db.insert_schedule({"season": season, "week": week, "home_team": "AAA", "away_team": "BBB",
                                "spread_line": 3.0, "total_line": 45.0, "home_score": 24, "away_score": 20})
            for i, (pid, _) in enumerate(PLAYERS):
                db.insert_player_weekly_stats({
                    "player_id": pid, "season": season, "week": week, "team": "AAA", "opponent": "BBB",
                    "fantasy_points": 10.0 + i + week + (5 if season == 2025 else 0),
                })
    return db


class _FakeEnsemble:
    position_models = {}
    component_predictors = {}


class _FakePredictor:
    """Serving-path stand-in: predicts week w's actual + 1 for every player."""
    calls = []

    def __init__(self):
        self.predictor = _FakeEnsemble()

    def initialize(self):
        return True

    def predict(self, n_weeks=1, top_n=100, as_of=None, **_):
        season, week = as_of
        _FakePredictor.calls.append(as_of)
        with DatabaseManager()._get_connection() as con:
            rows = pd.read_sql_query(
                "SELECT player_id, fantasy_points FROM player_weekly_stats WHERE season=? AND week=?",
                con, params=[season, week])
        return pd.DataFrame({"player_id": rows.player_id, "predicted_points": rows.fantasy_points + 1.0,
                             "position": "WR", "name": rows.player_id})


@pytest.fixture
def fake_serving(monkeypatch):
    import src.predict as predict_mod
    _FakePredictor.calls = []
    monkeypatch.setattr(predict_mod, "NFLPredictor", _FakePredictor)
    monkeypatch.setattr(DataManager, "get_train_test_seasons",
                        lambda self, test_season=None, **k: (list(range(2018, 2025)), 2025))


def test_evaluation_frame_has_prior_season_history_and_vegas_columns(private_db):
    frame = bt._evaluation_frame(2025)
    assert set(frame.season.unique()) == {2024, 2025}
    assert len(frame) == len(PLAYERS) * len(WEEKS) * 2
    home = frame[frame.team == "AAA"].iloc[0]
    assert home.implied_team_total == (45.0 + 3.0) / 2     # home favoured by 3 -> 24
    assert home.spread == 3.0 and home.game_total == 45.0


def test_walk_forward_scores_each_week_against_that_weeks_raw_points(private_db, fake_serving):
    results, report = bt.run_backtest(test_season=2025)

    assert _FakePredictor.calls == [(2025, 1), (2025, 2), (2025, 3)]
    assert results["n_predictions"] == len(PLAYERS) * len(WEEKS)        # 2024 rows are history only
    assert results["metrics"]["rmse"] == pytest.approx(1.0, abs=0.01)   # pred = actual + 1, same week
    assert results["backtest_path"] == "serving_path_as_of_walk_forward"
    assert results["weeks_evaluated"] == WEEKS
    assert results["partial_season"] is False
    assert results["model_source"] == "production_ensemble"
    assert results["model_type"] == "per_position_stacked_ensemble"
    assert results["trust"]["trusted"] is True
    assert "multiple_baseline_comparison" in results and "strong_baseline_comparison" in results

    artifacts = list((private_db.db_path.parent / "backtest_results").glob("backtest_2025_*.json"))
    assert len(artifacts) == 1 and "UNTRUSTED" not in artifacts[0].name
    saved = json.loads(artifacts[0].read_text())
    assert saved["trust"]["trusted"] is True and saved["backtest_path"] == results["backtest_path"]
    assert (private_db.db_path.parent / "advanced_model_results.json").exists()


def test_weeks_argument_limits_the_walk(private_db, fake_serving):
    results, _ = bt.run_backtest(test_season=2025, weeks=[2])
    assert _FakePredictor.calls == [(2025, 2)]
    assert results["n_predictions"] == len(PLAYERS)
    assert results["weeks_evaluated"] == [2]
    # A quick check is marked so it can never become the season headline.
    assert results["partial_season"] is True
    artifacts = list((private_db.db_path.parent / "backtest_results").glob("backtest_2025_*.json"))
    assert len(artifacts) == 1 and artifacts[0].name.endswith("_PARTIAL.json")

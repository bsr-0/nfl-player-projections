"""Walk-forward correctness for the game-outcome model.

Mirrors tests/test_season_aware_split.py's invariant (train seasons strictly
precede test seasons) applied to the game-outcome frame's shape, and checks
the eval script itself -- not just the splitter -- never lets a held-out
season influence its own training fold.
"""
import sqlite3

import numpy as np
import pytest

from src.evaluation.game_outcome_backtester import run_walk_forward_backtest
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.utils.database import DatabaseManager

SEASONS = [2018, 2019, 2020, 2021, 2022]
WEEKS = [1, 2, 3, 4]


def _seed_multi_season_db(db: DatabaseManager):
    for season in SEASONS:
        for week in WEEKS:
            home_score = 24 if (season + week) % 2 == 0 else 17
            away_score = 20
            db.insert_schedule(
                {
                    "season": season,
                    "week": week,
                    "home_team": "AAA",
                    "away_team": "BBB",
                    "spread_line": 2.5,
                    "total_line": 44.0,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
            for team in ("AAA", "BBB"):
                db.insert_team_stats(
                    {
                        "team": team,
                        "season": season,
                        "week": week,
                        "opponent": "BBB" if team == "AAA" else "AAA",
                        "home_away": "home" if team == "AAA" else "away",
                        "points_scored": 20.0 + week,
                        "points_allowed": 17.0,
                        "total_yards": 350.0,
                        "passing_yards": 220.0,
                        "rushing_yards": 130.0,
                        "turnovers": 1.0,
                        "third_down_conv": 0.4,
                        "drive_success_rate": 0.5,
                        "avg_drive_epa": 0.1,
                        "points_per_drive": 2.0,
                        "neutral_pass_rate_oe": 0.0,
                    }
                )


@pytest.fixture
def multi_season_db(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "test.db")
    _seed_multi_season_db(db)
    return db


def _rows(db):
    con = sqlite3.connect(str(db.db_path))
    try:
        return build_game_outcome_rows(seasons=SEASONS, con=con, include_weather=False)
    finally:
        con.close()


def test_splitter_folds_never_let_test_season_leak_into_training(multi_season_db):
    df = _rows(multi_season_db)
    season_arr = df["season"].to_numpy()
    cv = SeasonAwareTimeSeriesSplit(n_splits=3, seasons=season_arr, gap_seasons=0)
    for train_idx, test_idx in cv.split(df):
        test_seasons = np.unique(season_arr[test_idx])
        assert len(test_seasons) == 1
        assert season_arr[train_idx].max() < test_seasons[0]


def test_backtest_script_trains_only_on_strictly_prior_seasons(multi_season_db, monkeypatch):
    import src.evaluation.game_outcome_backtester as go_bt

    monkeypatch.setattr(
        go_bt,
        "build_game_outcome_rows",
        lambda seasons=None: _rows(multi_season_db),
    )
    report = run_walk_forward_backtest(seasons=SEASONS, n_test_seasons=2)

    assert report["n_folds"] == 2
    for fold in report["folds"]:
        assert max(fold["train_seasons"]) < min(fold["test_seasons"])
        assert set(fold["train_seasons"]).isdisjoint(fold["test_seasons"])

    for arm_metrics in report["pooled"].values():
        assert 0.0 <= arm_metrics["accuracy"] <= 1.0


def test_backtest_raises_rather_than_silently_using_season_unaware_split(multi_season_db, monkeypatch):
    """Only 5 seasons are seeded; n_test_seasons=5 with gap_seasons=0 needs
    6 seasons of season-aware headroom. The backtester must fail loudly
    here (strict=True) rather than silently falling back to a row-order
    TimeSeriesSplit and reporting the result as an honest held-out
    walk-forward -- see src/models/position_models.py's SeasonAwareTimeSeriesSplit."""
    import src.evaluation.game_outcome_backtester as go_bt

    monkeypatch.setattr(
        go_bt,
        "build_game_outcome_rows",
        lambda seasons=None: _rows(multi_season_db),
    )
    with pytest.raises(ValueError, match="strict=True"):
        run_walk_forward_backtest(seasons=SEASONS, n_test_seasons=5)

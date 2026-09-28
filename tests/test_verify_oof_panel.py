import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_oof_panel import verify
from src.models.oof_capture import ZERO_FLOOR, build_panel, capture_fold_rows, fold_coverage, segment_report, write_run_panel


def _frame(season, players):
    """One scored row per player plus its season-final game, which has no
    next game and therefore no actual (an expected, explained drop)."""
    n = len(players)
    scored = pd.DataFrame({
        "player_id": players,
        "season": season,
        "week": list(range(1, n + 1)),
        "team": ["A"] * n,
        "opponent": ["B"] * n,
        "position": ["QB"] * n,
        "predicted_points": np.arange(n, dtype=float) + 1,
        "actual_for_backtest": np.arange(n, dtype=float) + 2,
    })
    final = scored.assign(week=scored["week"] + 10, actual_for_backtest=np.nan)
    return pd.concat([scored, final], ignore_index=True)


def _write_complete_run(tmp_path: Path, *, frames=None, context=None):
    first, second = frames or (_frame(2023, ["a", "b", "c"]), _frame(2024, ["a", "b", "c"]))
    captured = [
        capture_fold_rows(first, train_seasons=[2022], test_season=2023),
        capture_fold_rows(second, train_seasons=[2022, 2023], test_season=2024),
    ]
    panel = build_panel(captured)
    extra = {}
    if context is not None:
        panel = context(panel)
        extra = {"game_context_source": "test schedule"}
    coverage = pd.concat([
        fold_coverage(first, captured[0], test_season=2023),
        fold_coverage(second, captured[1], test_season=2024),
    ], ignore_index=True)
    written = write_run_panel(
        panel, tmp_path, coverage=coverage, label="test",
        metadata={"folds": [
            {"test_season": 2023, "train_seasons": [2022]},
            {"test_season": 2024, "train_seasons": [2022, 2023]},
        ], **extra},
    )
    run_dir = Path(written["run_dir"])
    panel_for_segments = panel.copy()
    panel_for_segments["all_rows"] = "all"
    panel_for_segments["actual_activity"] = np.where(panel_for_segments["actual_points"] > ZERO_FLOOR, "nonzero_actual", "near_zero_actual")
    segment_report(panel_for_segments, by=("all_rows",), cluster="player_id").to_csv(run_dir / "total_segment_report.csv", index=False)
    segment_report(panel, by=("season",), cluster="player_id").to_csv(run_dir / "segment_report.csv", index=False)
    segment_report(panel, by=("position",), cluster="player_id").to_csv(run_dir / "position_segment_report.csv", index=False)
    segment_report(panel, by=("position", "is_cold_start", "week_bucket"), cluster="player_id").to_csv(run_dir / "experience_segment_report.csv", index=False)
    segment_report(panel, by=("position", "season"), cluster="player_id").to_csv(run_dir / "position_season_segment_report.csv", index=False)
    segment_report(panel_for_segments, by=("actual_activity",), cluster="player_id").to_csv(run_dir / "actual_activity_segment_report.csv", index=False)
    return run_dir


def test_verifier_writes_completion_marker_only_for_valid_run(tmp_path):
    run_dir = _write_complete_run(tmp_path)
    result = verify(run_dir)
    assert result["status"] == "verified"
    assert result["unexplained_drops"] == 0


def test_verifier_rejects_changed_panel_without_marker(tmp_path):
    run_dir = _write_complete_run(tmp_path)
    panel_path = run_dir / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel.loc[0, "residual"] = 999.0
    panel.to_parquet(panel_path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        verify(run_dir)
    assert not (run_dir / "verification.json").exists()


def test_verifier_rejects_unexplained_coverage_loss(tmp_path):
    run_dir = _write_complete_run(tmp_path)
    coverage_path = run_dir / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage[0]["n_missing_prediction_or_actual"] = 1
    coverage_path.write_text(json.dumps(coverage))
    with pytest.raises(ValueError, match="unexplained"):
        verify(run_dir)


def test_verifier_requires_game_context_of_the_target_game(tmp_path):
    """actual_points is the next observed game's outcome; context attached to
    the forecast-origin game pairs residuals from different games."""
    def frame(season):
        return pd.DataFrame({
            "player_id": ["a", "a", "b", "b"], "season": season, "week": [1, 3, 1, 3],
            "team": ["A", "A", "B", "B"], "opponent": ["B", "C", "A", "D"],
            "position": "QB", "predicted_points": 5.0,
            "actual_for_backtest": [7.0, np.nan, 8.0, np.nan],
        })

    def target_context(panel):
        return panel.assign(game_id=panel["season"].astype(str) + panel["target_team"],
                            home_team=panel["target_team"], away_team=panel["target_opponent"])

    def origin_context(panel):
        return panel.assign(game_id=panel["season"].astype(str) + "AB",
                            home_team=panel["team"], away_team=panel["opponent"])

    good = _write_complete_run(tmp_path / "good", frames=(frame(2023), frame(2024)),
                               context=target_context)
    assert verify(good)["status"] == "verified"
    bad = _write_complete_run(tmp_path / "bad", frames=(frame(2023), frame(2024)),
                              context=origin_context)
    with pytest.raises(ValueError, match="target team/opponent"):
        verify(bad)


def test_verifier_rejects_an_actual_without_a_next_game(tmp_path):
    """Every captured actual is a next-game outcome; one on a season-final row
    is not what the panel claims to hold."""
    def frame(season):
        return pd.DataFrame({"player_id": ["a"], "season": season, "week": [1], "team": "A",
                             "opponent": "B", "position": "QB", "predicted_points": 1.0,
                             "actual_for_backtest": 2.0})
    run_dir = _write_complete_run(tmp_path, frames=(frame(2023), frame(2024)))
    with pytest.raises(ValueError, match="negative unexplained"):
        verify(run_dir)

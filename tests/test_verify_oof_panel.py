import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_oof_panel import verify
from src.models.oof_capture import ZERO_FLOOR, build_panel, capture_fold_rows, fold_coverage, segment_report, write_run_panel


def _frame(season, players):
    n = len(players)
    return pd.DataFrame({
        "player_id": players,
        "season": season,
        "week": list(range(1, n + 1)),
        "team": ["A"] * n,
        "opponent": ["B"] * n,
        "position": ["QB"] * n,
        "predicted_points": np.arange(n, dtype=float) + 1,
        "actual_for_backtest": np.arange(n, dtype=float) + 2,
    })


def _write_complete_run(tmp_path: Path):
    first, second = _frame(2023, ["a", "b", "c"]), _frame(2024, ["a", "b", "c"])
    captured = [
        capture_fold_rows(first, train_seasons=[2022], test_season=2023),
        capture_fold_rows(second, train_seasons=[2022, 2023], test_season=2024),
    ]
    panel = build_panel(captured)
    coverage = pd.concat([
        fold_coverage(first, captured[0], test_season=2023),
        fold_coverage(second, captured[1], test_season=2024),
    ], ignore_index=True)
    written = write_run_panel(
        panel, tmp_path, coverage=coverage, label="test",
        metadata={"folds": [
            {"test_season": 2023, "train_seasons": [2022]},
            {"test_season": 2024, "train_seasons": [2022, 2023]},
        ]},
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

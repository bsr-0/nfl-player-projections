#!/usr/bin/env python3
"""Independently verify one immutable production OOF panel run.

The command writes ``verification.json`` only after every check succeeds.
It never repairs a panel or fills missing values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.oof_capture import (  # noqa: E402
    IDENTITY_COLUMNS,
    PREDICTION_COLUMN,
    ZERO_FLOOR,
    segment_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _same_report(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        return False
    n_index = actual.columns.get_loc("n") if "n" in actual.columns else 0
    sort_columns = list(actual.columns[:n_index])
    if not sort_columns:
        sort_columns = list(actual.columns)
    a = actual.sort_values(sort_columns).reset_index(drop=True)
    e = expected.sort_values(sort_columns).reset_index(drop=True)
    if list(a.columns) != list(e.columns):
        return False
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]):
            if not np.allclose(a[column].to_numpy(dtype=float), e[column].to_numpy(dtype=float), equal_nan=True, atol=1e-12, rtol=1e-12):
                return False
        elif not a[column].astype(object).equals(e[column].astype(object)):
            return False
    return True


def verify(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    panel_path = run_dir / "panel.parquet"
    coverage_path = run_dir / "coverage.json"
    if not manifest_path.exists() or not panel_path.exists():
        _fail("run directory must contain manifest.json and panel.parquet")
    manifest = json.loads(manifest_path.read_text())
    panel = pd.read_parquet(panel_path)

    required = set(IDENTITY_COLUMNS) | {PREDICTION_COLUMN, "actual_points", "residual",
                                        "train_seasons", "n_train_seasons"}
    missing = sorted(required - set(panel.columns))
    if missing:
        _fail(f"panel missing required columns: {missing}")
    if manifest.get("panel_sha256") != _sha256(panel_path):
        _fail("panel SHA-256 does not match manifest")
    if int(manifest.get("n_rows", -1)) != len(panel):
        _fail("manifest row count does not match panel")
    if int(manifest.get("n_players", -1)) != panel["player_id"].nunique():
        _fail("manifest player count does not match panel")
    if manifest.get("game_context_source"):
        context_columns = {"game_id", "home_team", "away_team"}
        if not context_columns <= set(panel.columns):
            _fail("manifest declares game context but panel lacks game context columns")
        if panel[list(context_columns)].isna().any().any():
            _fail("panel game context contains nulls")
        valid_side = ((panel["team"] == panel["home_team"]) & (panel["opponent"] == panel["away_team"])) | (
            (panel["team"] == panel["away_team"]) & (panel["opponent"] == panel["home_team"]))
        if not valid_side.all():
            _fail("panel game context does not match player team/opponent")

    keys = ["player_id", "season", "week"]
    if panel.duplicated(keys).any():
        _fail("panel contains duplicate player-season-week rows")
    for column in ("season", "week", "n_train_seasons", PREDICTION_COLUMN, "actual_points", "residual"):
        values = pd.to_numeric(panel[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            _fail(f"panel column {column!r} contains null or nonfinite values")
    if not np.allclose(
        panel["residual"].to_numpy(dtype=float),
        panel[PREDICTION_COLUMN].to_numpy(dtype=float) - panel["actual_points"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    ):
        _fail("residual is not predicted_points - actual_points")

    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        _fail("manifest must declare nonempty folds")
    fold_by_test = {int(row["test_season"]): row for row in folds}
    panel_seasons = set(panel["season"].astype(int))
    if panel_seasons != set(fold_by_test):
        _fail(f"panel seasons {sorted(panel_seasons)} differ from manifest folds {sorted(fold_by_test)}")
    for test_season, row in fold_by_test.items():
        train_seasons = {int(s) for s in row.get("train_seasons", [])}
        if test_season in train_seasons or any(s >= test_season for s in train_seasons):
            _fail(f"fold {test_season} has non-prior training season {sorted(train_seasons)}")
        fold_rows = panel.loc[panel["season"].eq(test_season)]
        declared = set(str(s) for s in fold_rows["train_seasons"].unique())
        expected = ",".join(str(s) for s in sorted(train_seasons))
        if declared != {expected}:
            _fail(f"panel train-season declaration mismatch for fold {test_season}")

    if not coverage_path.exists():
        _fail("coverage.json is required for a verified run")
    coverage = pd.DataFrame(json.loads(coverage_path.read_text()))
    coverage_required = {"test_season", "position", "n_offered", "n_captured", "n_dropped",
                         "n_intentionally_skipped", "n_missing_prediction_or_actual"}
    if not coverage_required <= set(coverage.columns):
        _fail(f"coverage missing required columns: {sorted(coverage_required - set(coverage.columns))}")
    if (coverage["n_dropped"] != coverage["n_offered"] - coverage["n_captured"]).any():
        _fail("coverage dropped counts do not equal offered minus captured")
    if (coverage["n_missing_prediction_or_actual"] < 0).any():
        _fail("coverage has negative unexplained drop counts")
    if coverage["n_missing_prediction_or_actual"].sum() != 0:
        _fail("coverage contains unexplained missing prediction/actual rows")
    actual_counts = panel.groupby(["season", "position"]).size().rename("actual").reset_index()
    captured_counts = coverage.rename(columns={"test_season": "season", "n_captured": "actual"})
    joined = actual_counts.merge(captured_counts[["season", "position", "actual"]], on=["season", "position"], how="outer", suffixes=("_panel", "_coverage")).fillna(0)
    if (joined["actual_panel"] != joined["actual_coverage"]).any():
        _fail("panel counts do not match coverage captured counts")

    panel_for_segments = panel.copy()
    panel_for_segments["all_rows"] = "all"
    panel_for_segments["actual_activity"] = np.where(
        panel_for_segments["actual_points"] > ZERO_FLOOR,
        "nonzero_actual", "near_zero_actual")
    report_specs = [
        ("total_segment_report.csv", ("all_rows",), panel_for_segments),
        ("segment_report.csv", ("season",)),
        ("position_segment_report.csv", ("position",), panel),
        ("position_season_segment_report.csv", ("position", "season"), panel),
        ("experience_segment_report.csv", ("position", "is_cold_start", "week_bucket"), panel),
        ("actual_activity_segment_report.csv", ("actual_activity",), panel_for_segments),
    ]
    for item in report_specs:
        filename, grouping, *frame = item
        path = run_dir / filename
        if not path.exists():
            _fail(f"missing required segment report {filename}")
        report_frame = frame[0] if frame else panel
        if not set(grouping) <= set(report_frame.columns):
            _fail(f"panel lacks grouping columns for {filename}: {grouping}")
        expected = segment_report(report_frame, by=grouping, cluster="player_id")
        actual = pd.read_csv(path)
        if not _same_report(actual, expected):
            _fail(f"saved report {filename} does not recompute from panel")

    career_path = run_dir / "career_segment_report.csv"
    career_available = "is_cold_start_career" in panel.columns and panel["is_cold_start_career"].notna().any()
    if career_available:
        if not career_path.exists():
            _fail("career labels are present but career_segment_report.csv is missing")
        expected = segment_report(panel, by=("position", "is_cold_start_career"), cluster="player_id")
        if not _same_report(pd.read_csv(career_path), expected):
            _fail("career segment report does not recompute from panel")

    return {
        "status": "verified",
        "run_dir": str(run_dir.resolve()),
        "panel_sha256": _sha256(panel_path),
        "n_rows": int(len(panel)),
        "n_players": int(panel["player_id"].nunique()),
        "seasons": sorted(int(s) for s in panel["season"].unique()),
        "positions": sorted(str(p) for p in panel["position"].unique()),
        "coverage_rows": int(len(coverage)),
        "unexplained_drops": int(coverage["n_missing_prediction_or_actual"].sum()),
        "career_segments_available": bool(career_available),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    output = args.run_dir / "verification.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Verified OOF panel: {args.run_dir}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

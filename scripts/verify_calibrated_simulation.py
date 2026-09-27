#!/usr/bin/env python3
"""Read-only verification for an immutable calibrated-simulation research run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_calibrated_simulation import (  # noqa: E402
    IDENTITY_COLUMNS,
    _marginal_player_metrics,
    _marginal_segments,
)
from src.models.residual_calibration import load_calibration  # noqa: E402
from src.models.simulation_evaluation import evaluate_joint_player_draws  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return False
    columns = list(expected.columns)
    left, right = actual[columns].sort_values(columns[:3]).reset_index(drop=True), expected.sort_values(columns[:3]).reset_index(drop=True)
    for column in columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(pd.to_numeric(left[column], errors="coerce"), right[column].to_numpy(float),
                               equal_nan=True, atol=1e-12, rtol=1e-12):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def verify(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("run directory lacks manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported calibrated-simulation manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest lacks output file hashes")
    for relative, expected_hash in files.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"output hash mismatch: {relative}")

    draws = pd.read_parquet(run_dir / "player_draws.parquet")
    actuals = pd.read_parquet(run_dir / "actuals.parquet")
    required_draws = {"mode", "draw", "fantasy_points", *IDENTITY_COLUMNS}
    required_actuals = {"actual_points", *IDENTITY_COLUMNS}
    if missing := required_draws - set(draws.columns):
        raise ValueError(f"draws lack required columns: {sorted(missing)}")
    if missing := required_actuals - set(actuals.columns):
        raise ValueError(f"actuals lack required columns: {sorted(missing)}")
    if actuals.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("actuals contain duplicate player-game rows")
    if not np.isfinite(draws["fantasy_points"].to_numpy(float)).all():
        raise ValueError("draws contain nonfinite fantasy_points")
    if not np.isfinite(actuals["actual_points"].to_numpy(float)).all():
        raise ValueError("actuals contain nonfinite actual_points")
    expected_keys = set(map(tuple, actuals[list(IDENTITY_COLUMNS)].to_numpy()))
    for mode, part in draws.groupby("mode", sort=True):
        keys = set(map(tuple, part[list(IDENTITY_COLUMNS)].drop_duplicates().to_numpy()))
        if keys != expected_keys:
            raise ValueError(f"{mode} does not match the exact actual player-game keys")
        draws_per_key = part.groupby(list(IDENTITY_COLUMNS))["draw"].nunique()
        if not (draws_per_key == int(manifest["draws_per_player"])).all():
            raise ValueError(f"{mode} has incomplete simulation draws")

    report = json.loads((run_dir / "report.json").read_text())
    if int(report["scoring_population"]["rows"]) != len(actuals):
        raise ValueError("report row count does not match actuals")
    if int(manifest["n_rows"]) != len(actuals):
        raise ValueError("manifest row count does not match actuals")
    metric_frames, game_frames = [], []
    for mode, part in draws.groupby("mode", sort=True):
        metrics = _marginal_player_metrics(part, actuals)
        metrics.insert(0, "mode", mode)
        metric_frames.append(metrics)
        joint = evaluate_joint_player_draws(part, actuals.rename(columns={"actual_points": "fantasy_points"}))
        by_game = pd.DataFrame(joint["by_game"])
        by_game.insert(0, "mode", mode)
        game_frames.append(by_game)
    marginal = pd.concat(metric_frames, ignore_index=True)
    games = pd.concat(game_frames, ignore_index=True)
    if not _same_frame(pd.read_csv(run_dir / "marginal_player_metrics.csv"), marginal):
        raise ValueError("marginal player metrics do not recompute from saved draws")
    if not _same_frame(pd.read_csv(run_dir / "joint_game_metrics.csv"), games):
        raise ValueError("joint game metrics do not recompute from saved draws")
    if not _same_frame(pd.read_csv(run_dir / "marginal_segment_report.csv"), _marginal_segments(marginal)):
        raise ValueError("marginal segment report does not recompute")

    selections = json.loads((run_dir / "calibration_selection.json").read_text())
    if selections != report.get("outer_season_selection"):
        raise ValueError("selection report differs from saved report")
    for selection in selections:
        path = run_dir / "outer_folds" / str(selection["outer_season"]) / "residual_calibration.json"
        artifact = load_calibration(path)
        if artifact.min_stratum_rows != int(selection["selected_min_stratum_rows"]):
            raise ValueError(f"artifact threshold mismatch for outer season {selection['outer_season']}")
        if artifact.diagnostics.get("pool_counts") != selection.get("artifact_pool_counts"):
            raise ValueError(f"artifact pool counts mismatch for outer season {selection['outer_season']}")
    return {"status": "verified", "run_dir": str(run_dir.resolve()),
            "rows": int(len(actuals)), "games": int(actuals["game_id"].nunique()),
            "modes": sorted(str(mode) for mode in draws["mode"].unique())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir)
    (args.run_dir / "verification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

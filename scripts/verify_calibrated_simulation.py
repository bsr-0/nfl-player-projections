#!/usr/bin/env python3
"""Read-only verification for an immutable calibrated-simulation research run.

Recomputes every row-level metric from the saved draws through an independent
path (long frames and ``evaluate_joint_player_draws``) and requires it to
match what the run wrote.
"""
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
    MODES,
    _marginal_player_metrics,
    _marginal_segments,
    _sum_rows,
)
from src.models.player_correlation import FactorCopulaModel  # noqa: E402
from src.models.residual_calibration import (  # noqa: E402
    EmpiricalResidualCalibration,
    PredictionAnalogCalibration,
    load_calibration,
)
from src.models.simulation_evaluation import evaluate_joint_player_draws  # noqa: E402

MARGINAL_KEYS = ["mode", *IDENTITY_COLUMNS]
JOINT_KEYS = ["mode", "game_id"]
SUM_KEYS = ["mode", "game_id", "sum_kind", "side"]
SEGMENT_KEYS = ["segment", "value", "mode"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> bool:
    """Equal up to row order, matched on the frame's full unique key."""
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        return False
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        return False
    columns = list(expected.columns)
    string_keys = {key: str for key in keys}
    left = actual[columns].astype(string_keys).sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.astype(string_keys).sort_values(keys, kind="mergesort").reset_index(drop=True)
    for column in columns:
        if pd.api.types.is_numeric_dtype(right[column]) and not pd.api.types.is_bool_dtype(right[column]):
            if not np.allclose(pd.to_numeric(left[column], errors="coerce"), right[column].to_numpy(float),
                               equal_nan=True, atol=1e-12, rtol=1e-12):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def _check_calibration(path: Path, candidate_id: str) -> None:
    artifact = load_calibration(path)
    family, size = candidate_id.split(":")
    if family == "legacy":
        ok = isinstance(artifact, EmpiricalResidualCalibration) and artifact.min_stratum_rows == int(size)
    else:
        ok = (isinstance(artifact, PredictionAnalogCalibration) and artifact.family == family
              and artifact.k == int(size))
    if not ok:
        raise ValueError(f"saved calibration {path.name} does not match selected candidate {candidate_id}")


def verify(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("run directory lacks manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported calibrated-simulation manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest lacks output file hashes")
    for relative, expected_hash in files.items():
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"output hash mismatch: {relative}")
    if tuple(manifest.get("modes", ())) != MODES:
        raise ValueError(f"manifest modes {manifest.get('modes')} differ from {list(MODES)}")

    actuals = pd.read_parquet(run_dir / "actuals.parquet")
    if actuals.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("actuals contain duplicate player-game rows")
    if not np.isfinite(actuals["actual_points"].to_numpy(float)).all():
        raise ValueError("actuals contain nonfinite actual_points")
    report = json.loads((run_dir / "report.json").read_text())
    if int(report["scoring_population"]["rows"]) != len(actuals) or int(manifest["n_rows"]) != len(actuals):
        raise ValueError("report/manifest row count does not match actuals")
    expected_keys = set(map(tuple, actuals[list(IDENTITY_COLUMNS)].astype(str).to_numpy()))
    actual_games = {game_id: rows for game_id, rows in actuals.groupby("game_id", sort=True)}

    marginal_frames, joint_frames, sum_frames = [], [], []
    for mode in MODES:
        relative = f"player_draws/{mode}.parquet"
        if relative not in files:
            raise ValueError(f"manifest lacks draws for {mode}")
        part = pd.read_parquet(run_dir / relative)
        if set(part["mode"].unique()) != {mode}:
            raise ValueError(f"{relative} contains other modes")
        if not np.isfinite(part["fantasy_points"].to_numpy(float)).all():
            raise ValueError(f"{mode} draws contain nonfinite fantasy_points")
        keys = set(map(tuple, part[list(IDENTITY_COLUMNS)].astype(str).drop_duplicates().to_numpy()))
        if keys != expected_keys:
            raise ValueError(f"{mode} does not match the exact actual player-game keys")
        if not (part.groupby(list(IDENTITY_COLUMNS))["draw"].nunique() == int(manifest["draws_per_player"])).all():
            raise ValueError(f"{mode} has incomplete simulation draws")
        metrics = _marginal_player_metrics(part, actuals)
        metrics.insert(0, "mode", mode)
        marginal_frames.append(metrics)
        joint = evaluate_joint_player_draws(part, actuals.rename(columns={"actual_points": "fantasy_points"}),
                                            scale_column="predicted_points")
        by_game = pd.DataFrame(joint["by_game"])
        by_game.insert(0, "mode", mode)
        joint_frames.append(by_game)
        sums = []
        for game_id, game_draws in part.groupby("game_id", sort=True):
            rows = actual_games[game_id].sort_values("player_id", kind="mergesort")
            matrix = (game_draws.pivot(index="draw", columns="player_id", values="fantasy_points")
                      .sort_index().reindex(columns=rows["player_id"]).to_numpy(float))
            sums.extend(_sum_rows(rows, matrix))
        sum_frame = pd.DataFrame(sums)
        sum_frame.insert(0, "mode", mode)
        sum_frames.append(sum_frame)
        del part
    marginal = pd.concat(marginal_frames, ignore_index=True)
    if not _same_frame(pd.read_csv(run_dir / "marginal_player_metrics.csv"), marginal, MARGINAL_KEYS):
        raise ValueError("marginal player metrics do not recompute from saved draws")
    if not _same_frame(pd.read_csv(run_dir / "joint_game_metrics.csv"),
                       pd.concat(joint_frames, ignore_index=True), JOINT_KEYS):
        raise ValueError("joint game metrics do not recompute from saved draws")
    if not _same_frame(pd.read_csv(run_dir / "joint_sum_metrics.csv"),
                       pd.concat(sum_frames, ignore_index=True), SUM_KEYS):
        raise ValueError("joint sum metrics do not recompute from saved draws")
    if not _same_frame(pd.read_csv(run_dir / "marginal_segment_report.csv"),
                       _marginal_segments(marginal), SEGMENT_KEYS):
        raise ValueError("marginal segment report does not recompute")

    selections = json.loads((run_dir / "selection_log.json").read_text())
    if selections != report.get("selection"):
        raise ValueError("selection log differs from saved report")
    last_by_season = {}
    for entry in selections:
        last_by_season[int(entry["season"])] = entry
    for season, entry in last_by_season.items():
        folder = run_dir / "outer_folds" / str(season)
        _check_calibration(folder / f"calibration_full_week{int(entry['week']):02d}.json",
                           entry["full"]["candidate"])
        _check_calibration(folder / f"calibration_legacy_week{int(entry['week']):02d}.json",
                           entry["legacy"]["candidate"])
        factor_weeks = json.loads((folder / "factor_models.json").read_text())
        for week_models in factor_weeks.values():
            for payload in week_models.values():
                FactorCopulaModel.from_dict(payload)
    return {"status": "verified", "run_dir": str(run_dir.resolve()),
            "rows": int(len(actuals)), "games": int(actuals["game_id"].nunique()),
            "modes": list(MODES)}


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

#!/usr/bin/env python3
"""Train one isolated served-model fold and export target-game full-PPR rows.

`prepare` freezes the raw train/test frames from the normal data loader.
`run` consumes exactly those frames, writes trained weights outside the live
models directory, and verifies every origin with a next observed game has a
prediction. It is deliberately a single fold so 2025 can be evaluated before
spending hours on earlier seasons.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import MODEL_CONFIG, MODELS_DIR, POSITIONS
from src.evaluation.full_ppr_head_to_head import (
    FULL_SCORING_DEFINITION, checked_production_fold, origin_to_target_map,
    scoring_weights,
)
from src.evaluation.paired_ppr_comparison import KEY, check_keys, file_sha256, read_rows
from src.models.data_loading import load_training_data
from src.models.train import _run_one_fold
from src.utils.models_dir import redirect_models_dir


def _live_hashes() -> dict[str, str]:
    patterns = ("model_*.joblib", "multiweek_*.joblib", "utilization_weights.json",
                "utilization_percentile_bounds.json", "model_metadata.json")
    paths = sorted({path for pattern in patterns for path in MODELS_DIR.glob(pattern)})
    return {str(path.resolve()): file_sha256(path) for path in paths}


def _source_hashes() -> dict[str, str]:
    """Hash code on the served-fold execution path, excluding other research arms."""
    relative = (
        "config/settings.py",
        "src/data/external_data.py", "src/data/entity_resolver.py",
        "src/evaluation/full_ppr_head_to_head.py",
        "src/evaluation/paired_ppr_comparison.py", "src/evaluation/ppr_truth.py",
        "src/features/advanced_rookie_injury.py", "src/features/college_conference.py",
        "src/features/feature_engineering.py", "src/features/season_long_features.py",
        "src/features/utilization_score.py",
        "src/models/advanced_techniques.py", "src/models/data_loading.py",
        "src/models/ensemble.py", "src/models/feature_preparation.py",
        "src/models/oof_capture.py", "src/models/position_models.py",
        "src/models/train.py", "src/models/utilization_to_fp.py",
        "src/utils/database.py", "src/utils/data_manager.py", "src/utils/helpers.py",
        "src/utils/models_dir.py",
    )
    paths = [ROOT / path for path in relative]
    paths.append(Path(__file__).resolve())
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"served-fold lineage source missing: {missing}")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def prepare(test_season: int, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    train, test, train_seasons, actual_test = load_training_data(
        positions=list(POSITIONS), test_season=test_season,
        optimize_training_years=False, strict_requirements=True)
    if actual_test != test_season or not train_seasons or max(train_seasons) >= test_season:
        raise ValueError("loaded training seasons reach the held-out season")
    if sorted(map(int, train.season.unique())) != sorted(map(int, train_seasons)):
        raise ValueError("loaded training rows differ from declared training seasons")
    if test.empty or not test.season.eq(test_season).all():
        raise ValueError("missing or mixed held-out season")
    check_keys(test, "raw production held-out fold")
    mapping = origin_to_target_map(test)
    if not mapping.target_week.notna().any():
        raise ValueError("held-out fold has no next-game targets")
    output_dir.mkdir(parents=True)
    train_path = output_dir / "raw_train.parquet"
    test_path = output_dir / "raw_test.parquet"
    map_path = output_dir / "target_map.csv"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    mapping.to_csv(map_path, index=False, float_format="%.17g")
    manifest = {
        "status": "prepared", "test_season": int(test_season),
        "train_seasons": list(map(int, train_seasons)),
        "rows": {"train": len(train), "test": len(test),
                 "next_game_origins": int(mapping.target_week.notna().sum())},
        "inputs": {name: {"file": path.name, "sha256": file_sha256(path)}
                   for name, path in (("train", train_path), ("test", test_path),
                                      ("mapping", map_path))},
        "source_code_sha256": _source_hashes(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run(input_dir: Path, output_dir: Path, *, n_trials: int | None = None) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    meta_path = input_dir / "manifest.json"
    meta_hash = file_sha256(meta_path)
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "prepared":
        raise ValueError("input directory is not a completed preflight")
    if _source_hashes() != meta.get("source_code_sha256"):
        raise ValueError("training source changed since the frozen preflight")
    paths = {name: input_dir / entry["file"] for name, entry in meta["inputs"].items()}
    for name, path in paths.items():
        if file_sha256(path) != meta["inputs"][name]["sha256"]:
            raise ValueError(f"frozen {name} input hash changed")
    train = pd.read_parquet(paths["train"])
    test = pd.read_parquet(paths["test"])
    mapping = read_rows(paths["mapping"])
    test_season = int(meta["test_season"])
    train_seasons = list(map(int, meta["train_seasons"]))
    if len(train) != meta["rows"]["train"] or len(test) != meta["rows"]["test"]:
        raise ValueError("frozen train/test row count changed")
    if max(train_seasons) >= test_season or not test.season.eq(test_season).all():
        raise ValueError("fold input reaches held-out season")
    if MODEL_CONFIG.get("position_target_type") != {pos: "fp" for pos in POSITIONS}:
        raise ValueError("served model does not predict fantasy-point units for all positions")
    same_map = origin_to_target_map(test)
    pd.testing.assert_frame_equal(mapping.reset_index(drop=True), same_map.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    trials = int(n_trials if n_trials is not None else MODEL_CONFIG["n_optuna_trials"])
    if trials < 1:
        raise ValueError("n_trials must be positive")
    before_live = _live_hashes()
    output_dir.mkdir(parents=True)
    collector: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    with redirect_models_dir(output_dir / "models"):
        _, aggregate = _run_one_fold(
            train, test, train_seasons, test_season, list(POSITIONS),
            tune_hyperparameters=True, n_trials=trials,
            oof_collector=collector, oof_coverage_collector=coverage)
    if _source_hashes() != meta["source_code_sha256"]:
        raise RuntimeError("training source changed during the isolated fold")
    if _live_hashes() != before_live:
        raise RuntimeError("live served model artifacts changed during isolated fold")
    if len(collector) != 1 or len(coverage) != 1:
        raise RuntimeError("held-out fold produced no single complete row-level capture")
    captured = collector[0]
    eligible = mapping.loc[mapping.target_week.notna(),
                           ["player_id", "origin_season", "origin_week", "origin_team", "origin_position"]]
    got = captured[KEY].rename(columns={"season": "origin_season", "week": "origin_week",
                                        "team": "origin_team", "position": "origin_position"})
    coverage_keys = eligible.merge(got, on=list(eligible), how="outer", validate="one_to_one", indicator=True)
    if not coverage_keys._merge.eq("both").all():
        sample = coverage_keys.loc[coverage_keys._merge.ne("both")].head(3).to_dict("records")
        raise ValueError(f"production OOF capture differs from next-game origins: {sample}")
    predictions = checked_production_fold(captured, mapping)
    if not predictions.season.eq(test_season).all():
        raise ValueError("target-game output crosses the held-out season")
    path = output_dir / "predictions.csv"
    predictions.to_csv(path, index=False, float_format="%.17g")
    saved = read_rows(path)
    mae = float(np.abs(saved.predicted_ppr - saved.actual_ppr).mean())
    if len(saved) != len(predictions) or not np.isfinite(mae):
        raise ValueError("saved production fold is incomplete")
    model_hashes = {str(path.relative_to(output_dir)): file_sha256(path)
                    for path in sorted((output_dir / "models").rglob("*.joblib"))}
    if not model_hashes:
        raise RuntimeError("isolated fold did not persist model weights")
    manifest = {
        "schema_version": 1, "label": "served_model_seasonal_fold",
        "key_semantics": "target_game", "rows_file": path.name,
        "rows_sha256": file_sha256(path), "prediction_column": "predicted_ppr",
        "actual_column": "actual_ppr", "scoring_definition": FULL_SCORING_DEFINITION,
        "scoring_weights": scoring_weights(),
        "folds": [{"test_season": test_season,
                   "trained_through_season": max(train_seasons),
                   "selection_through_season": None}],
        "provenance": {"source_preflight": str(meta_path.resolve()),
                       "source_preflight_sha256": meta_hash,
                       "target_identity_source": "next observed raw test row within player-season",
                       "prediction_target": "served fp-mode one-week prediction",
                       "fit_n_trials": trials, "model_artifact_sha256": model_hashes,
                       "source_code_sha256": meta["source_code_sha256"]},
    }
    report = {"status": "complete", "rows": len(saved), "test_season": test_season,
              "train_end_season": max(train_seasons), "saved_row_mae": mae,
              "coverage": coverage[0].to_dict("records"),
              "aggregate_fold_metrics": aggregate.get("by_position", {}) if aggregate else {},
              "prediction_sha256": manifest["rows_sha256"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("--test-season", type=int, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    fit = sub.add_parser("run")
    fit.add_argument("--input-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--n-trials", type=int, help="Default is production MODEL_CONFIG n_optuna_trials")
    args = parser.parse_args()
    try:
        result = (prepare(args.test_season, args.output_dir) if args.command == "prepare"
                  else run(args.input_dir, args.output_dir, n_trials=args.n_trials))
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f"served full-PPR fold stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

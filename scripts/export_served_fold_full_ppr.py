#!/usr/bin/env python3
"""Train one isolated served-model fold and export target-game full-PPR rows.

`prepare` freezes the raw train/test frames from the normal data loader.
`run` consumes exactly those frames, writes trained weights outside the live
models directory, persists the row-level capture, and verifies every origin
with a next observed game has a prediction. If a post-training check fails,
`finalize` re-validates the persisted capture without refitting. It is deliberately a single fold so 2025 can be evaluated before
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

CAPTURE_FILE = "raw_capture.parquet"
COVERAGE_FILE = "coverage.parquet"
FIT_STATE_FILE = "fit_state.json"


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


def _describe_hash_diff(before: dict[str, str], after: dict[str, str]) -> str:
    """Name which paths changed/were added/were removed between two hash snapshots."""
    changed = sorted(p for p in before.keys() & after.keys() if before[p] != after[p])
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    parts = []
    if changed:
        parts.append(f"changed: {changed}")
    if added:
        parts.append(f"added: {added}")
    if removed:
        parts.append(f"removed: {removed}")
    return "; ".join(parts) if parts else "no path-level difference detected"


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
            oof_collector=collector, oof_coverage_collector=coverage,
            # A capture failure must stop with its own message, not a
            # logged warning followed by a generic "no capture" error.
            oof_strict=True)
    after_source = _source_hashes()
    if after_source != meta["source_code_sha256"]:
        raise RuntimeError(
            "training source changed during the isolated fold "
            f"({_describe_hash_diff(meta['source_code_sha256'], after_source)})")
    after_live = _live_hashes()
    if after_live != before_live:
        raise RuntimeError(
            "live served model artifacts changed during isolated fold "
            f"({_describe_hash_diff(before_live, after_live)})")
    if len(collector) != 1 or len(coverage) != 1:
        raise RuntimeError("held-out fold produced no single complete row-level capture")
    # Persist the fit's outputs BEFORE any post-training validation, so a check
    # failure can be fixed and re-run with `finalize` instead of a new fit.
    collector[0].to_parquet(output_dir / CAPTURE_FILE, index=False)
    coverage[0].to_parquet(output_dir / COVERAGE_FILE, index=False)
    fit_state = {
        "status": "fitted", "source_preflight": str(meta_path.resolve()),
        "source_preflight_sha256": meta_hash, "fit_n_trials": trials,
        "fit_source_code_sha256": meta["source_code_sha256"],
        "capture_sha256": file_sha256(output_dir / CAPTURE_FILE),
        "coverage_sha256": file_sha256(output_dir / COVERAGE_FILE),
        "model_artifact_sha256": _model_hashes(output_dir),
        "aggregate_fold_metrics": aggregate.get("by_position", {}) if aggregate else {},
    }
    (output_dir / FIT_STATE_FILE).write_text(
        json.dumps(fit_state, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    return finalize(input_dir, output_dir)


def _model_hashes(output_dir: Path) -> dict[str, str]:
    hashes = {str(path.relative_to(output_dir)): file_sha256(path)
              for path in sorted((output_dir / "models").rglob("*.joblib"))}
    if not hashes:
        raise RuntimeError("isolated fold did not persist model weights")
    return hashes


def finalize(input_dir: Path, output_dir: Path) -> dict:
    """Validate a persisted fit's capture and publish predictions/manifest.

    Re-runnable after a post-training check failure: it reads only the frozen
    preflight and the fit's saved capture, verifying both against the hashes
    recorded when the fit finished. Validation code may have changed since the
    fit; the manifest records the fit-time source hashes as the training
    lineage and any finalize-time difference separately.
    """
    state_path = output_dir / FIT_STATE_FILE
    if not state_path.exists():
        raise ValueError(f"no completed fit to finalize in {output_dir}")
    if (output_dir / "manifest.json").exists():
        raise ValueError(f"fold already finalized: {output_dir}")
    state = json.loads(state_path.read_text())
    meta_path = input_dir / "manifest.json"
    meta_hash = file_sha256(meta_path)
    if (str(meta_path.resolve()) != state["source_preflight"]
            or meta_hash != state["source_preflight_sha256"]):
        raise ValueError("finalize input is not the preflight this fit used")
    meta = json.loads(meta_path.read_text())
    paths = {name: input_dir / entry["file"] for name, entry in meta["inputs"].items()}
    for name, path in paths.items():
        if file_sha256(path) != meta["inputs"][name]["sha256"]:
            raise ValueError(f"frozen {name} input hash changed")
    for name, file in (("capture", CAPTURE_FILE), ("coverage", COVERAGE_FILE)):
        if file_sha256(output_dir / file) != state[f"{name}_sha256"]:
            raise ValueError(f"persisted {name} changed since the fit")
    model_hashes = _model_hashes(output_dir)
    if model_hashes != state["model_artifact_sha256"]:
        raise ValueError("persisted model weights changed since the fit")
    mapping = read_rows(paths["mapping"])
    test_season = int(meta["test_season"])
    train_seasons = list(map(int, meta["train_seasons"]))
    captured = pd.read_parquet(output_dir / CAPTURE_FILE)
    coverage = pd.read_parquet(output_dir / COVERAGE_FILE)
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
    finalize_source = _source_hashes()
    provenance = {"source_preflight": state["source_preflight"],
                  "source_preflight_sha256": meta_hash,
                  "target_identity_source": "next observed raw test row within player-season",
                  "prediction_target": "served fp-mode one-week prediction",
                  "fit_n_trials": state["fit_n_trials"], "model_artifact_sha256": model_hashes,
                  "source_code_sha256": state["fit_source_code_sha256"]}
    if finalize_source != state["fit_source_code_sha256"]:
        provenance["finalize_source_code_sha256"] = finalize_source
        provenance["finalize_source_change"] = _describe_hash_diff(
            state["fit_source_code_sha256"], finalize_source)
    manifest = {
        "schema_version": 1, "label": "served_model_seasonal_fold",
        "key_semantics": "target_game", "rows_file": path.name,
        "rows_sha256": file_sha256(path), "prediction_column": "predicted_ppr",
        "actual_column": "actual_ppr", "scoring_definition": FULL_SCORING_DEFINITION,
        "scoring_weights": scoring_weights(),
        "folds": [{"test_season": test_season,
                   "trained_through_season": max(train_seasons),
                   "selection_through_season": None}],
        "provenance": provenance,
    }
    report = {"status": "complete", "rows": len(saved), "test_season": test_season,
              "train_end_season": max(train_seasons), "saved_row_mae": mae,
              "coverage": coverage.to_dict("records"),
              "aggregate_fold_metrics": state["aggregate_fold_metrics"],
              "prediction_sha256": manifest["rows_sha256"]}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=float) + "\n")
    # manifest.json last: it is the completion marker finalize/compare key on.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
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
    fin = sub.add_parser("finalize", help="Re-validate and publish a fit whose post-training checks failed")
    fin.add_argument("--input-dir", type=Path, required=True)
    fin.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.test_season, args.output_dir)
        elif args.command == "run":
            result = run(args.input_dir, args.output_dir, n_trials=args.n_trials)
        else:
            result = finalize(args.input_dir, args.output_dir)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f"served full-PPR fold stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

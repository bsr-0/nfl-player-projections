"""Smoke test for the served-fold export workflow (testing only, not evidence).

Runs the REAL `export_served_fold_full_ppr.run` end to end -- real feature
preparation, real test-target winsorization, real `capture_fold_rows`, real
target-game mapping, `checked_production_fold`, then predeclare and compare -- on a few training seasons
with minimal tuning, so a defect in the post-training checks surfaces in
minutes instead of after a multi-hour fit. The 2026-10-07 "captured shifted
actual does not match mapped target game" failure happened only after the full
retrain (test targets were winsorized; they are raw since then); this script
reproduces that class of failure quickly because it shares the code path, and
it now requires captured actuals to equal raw next-game points exactly. It
also re-runs `finalize` on a copy to prove a failed fit can be recovered.

Nothing it writes is usable as a result: the model is a few-season, one-trial
fit. It reads a frozen preflight read-only, writes only under --output-dir,
and redirects MODELS_DIR (run() does) so served artifacts are untouched. This
file is deliberately outside the hashed served-fold source set.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shrink_config() -> None:
    from config.settings import MODEL_CONFIG
    MODEL_CONFIG.update({
        "n_optuna_trials": 1, "cv_folds": 2, "stability_n_bootstrap": 2,
        "lstm_optuna_trials": 0, "deep_optuna_trials": 0, "lstm_epochs": 1,
        "deep_epochs": 1, "enable_shap_pdp": False,
    })


def build_smoke_preflight(source: Path, out: Path, exporter, n_train_seasons: int) -> Path:
    """Copy a frozen preflight, keeping only the last few training seasons."""
    meta = json.loads((source / "manifest.json").read_text())
    train = pd.read_parquet(source / "raw_train.parquet")
    keep = sorted(map(int, train.season.unique()))[-n_train_seasons:]
    train = train[train.season.isin(keep)].reset_index(drop=True)
    out.mkdir(parents=True)
    train.to_parquet(out / "raw_train.parquet", index=False)
    shutil.copy(source / "raw_test.parquet", out / "raw_test.parquet")
    shutil.copy(source / "target_map.csv", out / "target_map.csv")
    test_rows = meta["rows"]["test"]
    manifest = {
        "status": "prepared", "test_season": meta["test_season"], "train_seasons": keep,
        "rows": {"train": len(train), "test": test_rows,
                 "next_game_origins": meta["rows"]["next_game_origins"]},
        "inputs": {name: {"file": path.name, "sha256": exporter.file_sha256(path)}
                   for name, path in (("train", out / "raw_train.parquet"),
                                      ("test", out / "raw_test.parquet"),
                                      ("mapping", out / "target_map.csv"))},
        "source_code_sha256": exporter._source_hashes(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def capture_vs_raw(fold_dir: Path, preflight: Path) -> dict:
    """Captured actuals versus raw next-game points (must be identical)."""
    from src.evaluation.full_ppr_head_to_head import origin_to_target_map
    captured = pd.read_parquet(fold_dir / "raw_capture.parquet")
    mapping = origin_to_target_map(pd.read_parquet(preflight / "raw_test.parquet"))
    key = ["player_id", "season", "week", "team", "position"]
    renamed = mapping.rename(columns={"origin_season": "season", "origin_week": "week",
                                      "origin_team": "team", "origin_position": "position"})
    own = [c for c in captured.columns if c in renamed.columns and c.startswith("target_")]
    joined = captured.drop(columns=own).merge(renamed, on=key, how="left")
    gap = np.abs(joined.actual_points.to_numpy(float)
                 - joined.target_fantasy_points.to_numpy(float))
    return {"rows": len(joined), "rows_differing_from_raw": int((gap > 0.05).sum()),
            "max_gap": float(gap.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-dir", type=Path, required=True,
                        help="Frozen served-fold preflight (read only).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seasons", type=int, default=3)
    parser.add_argument("--plan-a-manifest", type=Path, default=ROOT / (
        "data/experiments/full_ppr_head_to_head_20260925/inputs/plan_a_full.manifest.json"),
        help="Exercise predeclare/compare against this Plan A manifest.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output exists: {args.output_dir}")
    started = time.time()
    exporter = _load_script("export_served_fold_full_ppr")
    _shrink_config()
    pre = build_smoke_preflight(args.preflight_dir, args.output_dir / "inputs",
                                exporter, args.train_seasons)
    fold = args.output_dir / "fold"
    result = exporter.run(pre, fold, n_trials=1)
    check = capture_vs_raw(fold, pre)
    if check["rows_differing_from_raw"]:
        raise SystemExit(f"captured actuals differ from raw next-game points: {check}")
    # Recovery path: strip the published outputs from a copy and finalize it.
    redo = args.output_dir / "fold_refinalized"
    shutil.copytree(fold, redo)
    for name in ("manifest.json", "report.json", "predictions.csv"):
        (redo / name).unlink()
    state = json.loads((redo / exporter.FIT_STATE_FILE).read_text())
    if state["model_artifact_sha256"] != exporter._model_hashes(redo):
        raise SystemExit("copied fit does not reproduce its model hashes")
    try:
        exporter.finalize(pre, redo)
    except ValueError as exc:
        raise SystemExit(f"finalize on a persisted fit failed: {exc}")
    if (redo / "predictions.csv").read_bytes() != (fold / "predictions.csv").read_bytes():
        raise SystemExit("finalize did not reproduce the run's predictions")
    # The comparator refuses a fold from a different preflight, so the smoke
    # run declares its own population from its own preflight.
    comparer = _load_script("compare_full_ppr_head_to_head")
    population = args.output_dir / "population"
    comparer.predeclare(args.plan_a_manifest, pre / "manifest.json", population)
    comparer.compare(population / "manifest.json", fold / "manifest.json",
                     args.output_dir / "comparison")
    summary = {"smoke_only": True, "seconds": round(time.time() - started, 1),
               "predictions": result["rows"], "saved_row_mae": result["saved_row_mae"],
               "comparison_report": str(args.output_dir / "comparison" / "report.json"),
               "capture_vs_raw": check, "refinalize_reproduced_predictions": True}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

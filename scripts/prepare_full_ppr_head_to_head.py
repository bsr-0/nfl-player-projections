#!/usr/bin/env python3
"""Freeze ten-component full-PPR truth and Plan A's zero-extra forecast."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.evaluation.full_ppr_head_to_head import (
    FULL_SCORING_DEFINITION, checked_full_truth, load_context, scoring_weights,
)
from src.evaluation.paired_ppr_comparison import (
    KEY, file_sha256, load_predictions, read_rows,
)


def prepare(plan_a_manifest: Path, raw_truth_path: Path, db_path: Path,
            output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    plan_a, source_manifest, source = load_predictions(plan_a_manifest)
    raw_hash = file_sha256(raw_truth_path)
    raw = read_rows(raw_truth_path)
    seasons = sorted(map(int, raw.season.unique()))
    stats, canonical, history, draft = load_context(db_path, seasons)
    truth = checked_full_truth(raw, stats, canonical, history, draft)
    if file_sha256(raw_truth_path) != raw_hash:
        raise ValueError("raw truth changed during preparation")
    paired = truth[KEY + ["actual_ppr", "actual_full_ppr"]].merge(
        plan_a, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all() or len(paired) != len(plan_a):
        raise ValueError("Plan A predictions and ten-component truth have different keys")
    if not np.allclose(paired.recorded_actual, paired.actual_ppr, rtol=0, atol=1e-9):
        raise ValueError("Plan A recorded eight-component labels disagree with checked truth")
    prediction = truth[KEY].copy()
    prediction["predicted_ppr"] = paired.prediction.to_numpy(float)
    prediction["actual_ppr"] = paired.actual_full_ppr.to_numpy(float)
    # No observed fumbles or two-point conversions enter predicted_ppr.
    if not np.isfinite(prediction[["predicted_ppr", "actual_ppr"]].to_numpy(float)).all():
        raise ValueError("nonfinite full-PPR prediction or label")
    output_dir.mkdir(parents=True)
    truth_path = output_dir / "full_truth.csv"
    prediction_path = output_dir / "plan_a_full_predictions.csv"
    truth.to_csv(truth_path, index=False, float_format="%.17g")
    prediction.to_csv(prediction_path, index=False, float_format="%.17g")
    saved_truth = read_rows(truth_path)
    saved_prediction = read_rows(prediction_path)
    if len(saved_truth) != len(truth) or len(saved_prediction) != len(prediction):
        raise ValueError("saved full-PPR rows differ in count")
    mae = float(np.abs(saved_prediction.predicted_ppr - saved_prediction.actual_ppr).mean())
    db_stat = db_path.stat()
    manifest = {
        "schema_version": 1, "label": "plan_a_zero_extra_components",
        "key_semantics": "target_game", "rows_file": prediction_path.name,
        "rows_sha256": file_sha256(prediction_path),
        "prediction_column": "predicted_ppr", "actual_column": "actual_ppr",
        "scoring_definition": FULL_SCORING_DEFINITION,
        "scoring_weights": scoring_weights(),
        "prediction_extension": {"fumbles_lost": 0, "two_point_conversions": 0},
        "folds": source_manifest["folds"],
        "provenance": {"source_plan_a_manifest": str(plan_a_manifest.resolve()),
                       "source_plan_a_manifest_sha256": file_sha256(plan_a_manifest),
                       "source_plan_a_rows_sha256": source["rows_sha256"],
                       "source_raw_truth": str(raw_truth_path.resolve()),
                       "source_raw_truth_sha256": raw_hash,
                       "read_only_db": str(db_path.resolve()), "db_size": db_stat.st_size,
                       "db_mtime_ns": db_stat.st_mtime_ns,
                       "full_truth_sha256": file_sha256(truth_path)},
    }
    audit = {
        "status": "complete", "n_rows": len(saved_prediction),
        "fold_rows": {str(s): int(n) for s, n in saved_truth.groupby("season").size().items()},
        "changed_full_vs_eight_labels": int(saved_truth.extra_component_points.ne(0).sum()),
        "mean_absolute_extra_points": float(saved_truth.extra_component_points.abs().mean()),
        "plan_a_full_ppr_mae": mae,
        "cohorts": {
            "draft_rookie": int(saved_truth.draft_rookie.sum()),
            "returning_player": int(saved_truth.returning_player.sum()),
            "first_observed_season": int(saved_truth.first_observed_season.sum()),
            "snap_segment": {str(k): int(v) for k, v in saved_truth.snap_segment.value_counts().items()},
        },
        "scoring_definition": FULL_SCORING_DEFINITION,
        "full_truth_sha256": file_sha256(truth_path),
        "prediction_sha256": manifest["rows_sha256"],
        "saved_mae_recomputed": True,
    }
    (output_dir / "plan_a_full.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-a-manifest", type=Path, required=True)
    parser.add_argument("--raw-truth", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit = prepare(args.plan_a_manifest, args.raw_truth, args.db, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"full-PPR preparation stopped: {exc}\n")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

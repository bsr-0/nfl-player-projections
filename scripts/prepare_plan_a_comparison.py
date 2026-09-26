#!/usr/bin/env python3
"""Freeze verified Plan A truth and manifests for the paired comparison, without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH, SCORING
from src.evaluation.ppr_truth import KEY as OOF_KEY, TARGETS, checked_truth
from src.evaluation.paired_ppr_comparison import (
    KEY, SCORING_DEFINITION, check_keys, file_sha256, finite_columns, read_rows,
)


def prepare(selector_dir: Path, database: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    report_path, rows_path = selector_dir / "joint_selector.json", selector_dir / "ppr_oof_rows.csv"
    report_hash, rows_hash = file_sha256(report_path), file_sha256(rows_path)
    report = json.loads(report_path.read_text())
    if report.get("scoring_definition") != SCORING_DEFINITION:
        raise ValueError("selector report is not the corrected raw eight-component run")
    input_hashes = report.get("input_sha256", {})
    if len(input_hashes) != 16:
        raise ValueError("selector must identify all 8 allocation and 8 team-total CSV hashes")
    for source, digest in input_hashes.items():
        if file_sha256(ROOT / source) != digest:
            raise ValueError(f"saved selector input has changed: {source}")
    rows = read_rows(rows_path)
    check_keys(rows, "Plan A OOF")
    finite_columns(rows, ["fold", "actual_ppr", "predicted_ppr_selected", "predicted_ppr_baseline"], "Plan A OOF")
    if not set(OOF_KEY).issubset(rows):
        raise ValueError("Plan A OOF fold identity is missing")
    seasons = sorted(int(s) for s in rows.season.unique())
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        raw = pd.read_sql_query(
            f"SELECT {', '.join(KEY + TARGETS)} FROM team_week_player_shares "
            f"WHERE season IN ({','.join('?' for _ in seasons)})", connection, params=seasons)
    check_keys(raw, "current raw share panel")
    truth = checked_truth(raw, rows[OOF_KEY])
    if not np.allclose(truth.actual_ppr, rows.actual_ppr, atol=1e-9, rtol=0):
        raise ValueError("saved Plan A labels differ from current raw components")
    ordered = truth.sort_values(["season", "week", "team", "player_id"])
    truth_hash = hashlib.sha256(pd.util.hash_pandas_object(ordered[TARGETS + KEY], index=False).to_numpy().tobytes()).hexdigest()
    if truth_hash != report.get("raw_truth_sha256"):
        raise ValueError("current raw truth hash differs from the corrected selector report")
    if len(rows) != report["comparison"]["pooled"]["n"]:
        raise ValueError("saved Plan A row count differs from report")
    for fold, part in rows.groupby("fold", sort=True):
        audited = report["input_audit"]["folds"].get(str(int(fold)))
        if (not audited or part.season.nunique() != 1 or len(part) != audited["rows"]
                or int(part.season.iloc[0]) != audited["season"]):
            raise ValueError(f"Plan A fold {fold} differs from saved audit")
        for name, column in (("selected", "predicted_ppr_selected"), ("baseline", "predicted_ppr_baseline")):
            mae = float((part[column] - part.actual_ppr).abs().mean())
            saved_mae = report["comparison"]["folds"][str(int(fold))][f"{name}_mae"]
            if not np.isclose(mae, saved_mae, atol=1e-12, rtol=0):
                raise ValueError(f"Plan A {fold}/{name} saved row MAE differs from report")
    if file_sha256(report_path) != report_hash or file_sha256(rows_path) != rows_hash:
        raise ValueError("selector artifacts changed during preparation")
    if any(file_sha256(ROOT / source) != digest for source, digest in input_hashes.items()):
        raise ValueError("saved selector inputs changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=False)
    truth.to_csv(output_dir / "raw_truth.csv", index=False)
    rows[KEY].to_csv(output_dir / "population_keys.csv", index=False)
    for name, column in (("plan_a", "predicted_ppr_selected"), ("rolling3", "predicted_ppr_baseline")):
        manifest = {
            "schema_version": 1, "label": name, "key_semantics": "target_game",
            "rows_file": os.path.relpath(rows_path.resolve(), output_dir.resolve()), "rows_sha256": rows_hash,
            "prediction_column": column, "actual_column": "actual_ppr",
            "scoring_definition": SCORING_DEFINITION,
            "scoring_weights": {target: SCORING[target] for target in TARGETS},
            "folds": [{"test_season": season, "trained_through_season": season - 1,
                       "selection_through_season": seasons[i - 1] if name == "plan_a" and i else None}
                      for i, season in enumerate(seasons)],
            "provenance": {"selector_report": os.path.relpath(report_path.resolve(), output_dir.resolve()),
                           "selector_report_sha256": report_hash, "raw_truth_hash_in_selector": truth_hash,
                           "cutoff_basis": "Upper bounds from reviewed season-forward allocation/team-total code; selector uses prior folds. Historical model weights are not reverified.",
                           "verified_saved_input_hashes": input_hashes},
        }
        (output_dir / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    audit = {"status": "plan_a_inputs_verified_production_export_pending", "rows": len(rows),
             "folds": report["input_audit"]["folds"], "scoring_definition": SCORING_DEFINITION,
             "raw_truth_csv_sha256": file_sha256(output_dir / "raw_truth.csv"),
             "population_keys_sha256": file_sha256(output_dir / "population_keys.csv"),
             "selector_report_sha256": report_hash, "selector_rows_sha256": rows_hash,
             "verified_saved_input_files": len(input_hashes),
             "production_requirements": ["Completed row-level prediction export and manifest",
                                         "Target-game keys, not the origin-week keys of target_1w",
                                         "Predictions for the same eight-component scoring definition",
                                         "Explicit population file if coverage differs; no automatic intersection"]}
    (output_dir / "preparation_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, default=ROOT / "data/experiments/full_ppr_raw_truth_20260924")
    parser.add_argument("--db", type=Path, default=Path(DB_PATH))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        audit = prepare(args.selector_dir, args.db, args.output_dir)
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        parser.exit(2, f"PPR preparation stopped: {exc}\n")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

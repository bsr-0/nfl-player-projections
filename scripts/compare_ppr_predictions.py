#!/usr/bin/env python3
"""Compare two completed prediction exports under the raw eight-component PPR contract."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.paired_ppr_comparison import (
    comparison_report, file_sha256, load_predictions, pair_predictions, read_rows,
)


def run_comparison(baseline_manifest: Path, candidate_manifest: Path, truth_path: Path,
                   output_dir: Path, population_path: Path | None = None,
                   n_bootstrap: int = 2000, seed: int = 42) -> dict:
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}; choose a new run directory")
    baseline, base_meta, base_input = load_predictions(baseline_manifest)
    candidate, cand_meta, cand_input = load_predictions(candidate_manifest)
    truth_hash = file_sha256(truth_path)
    raw = read_rows(truth_path)
    population_hash = file_sha256(population_path) if population_path else None
    population = read_rows(population_path) if population_path else None
    rows, coverage, excluded = pair_predictions(baseline, candidate, raw, population)
    report = comparison_report(rows, n_bootstrap, seed)
    report.update({"generated_at": datetime.now(timezone.utc).isoformat(),
                   "baseline_label": base_meta["label"], "candidate_label": cand_meta["label"],
                   "coverage": coverage, "prediction_manifests": {"baseline": base_meta, "candidate": cand_meta},
                   "inputs": {"baseline": base_input, "candidate": cand_input,
                              "truth": {"path": str(truth_path.resolve()), "sha256": truth_hash},
                              "population": {"path": str(population_path.resolve()), "sha256": population_hash}
                              if population_path else None}})
    # Refuse to publish against an input that a still-running process changed.
    # Keep every observation, including when two manifests share a row file.
    # A dict would discard an earlier digest if that file changed between loads.
    expected_hashes = [(truth_path, truth_hash)]
    if population_path:
        expected_hashes.append((population_path, population_hash))
    for source in (base_input, cand_input):
        expected_hashes.append((Path(source["rows"]), source["rows_sha256"]))
        expected_hashes.append((Path(source["manifest"]), source["manifest_sha256"]))
    if any(file_sha256(path) != digest for path, digest in expected_hashes):
        raise ValueError("an input changed during comparison; wait for completed exports")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "paired_rows.csv"
    rows.to_csv(rows_path, index=False)
    for name, frame in excluded.items():
        frame.to_csv(output_dir / f"excluded_{name}_keys.csv", index=False)
    # Recompute from the serialized rows, independently of the group report.
    saved = read_rows(rows_path)
    for name in ("baseline", "candidate"):
        actual_mae = float((saved[f"predicted_ppr_{name}"] - saved.actual_ppr).abs().mean())
        if abs(actual_mae - report["pooled"][f"{name}_mae"]) > 1e-12:
            raise ValueError("saved row-level MAE does not match report")
    report["output"] = {"paired_rows_sha256": file_sha256(rows_path), "saved_mae_recomputed": True}
    # report.json is the completion marker; partial output dirs have no report.
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path, help="Raw eight component columns and target-game keys")
    parser.add_argument("--population-keys", type=Path, help="Predeclared target games; both exports must cover every row")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing directories are refused")
    parser.add_argument("--n-bootstrap", default=2000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    try:
        report = run_comparison(args.baseline_manifest, args.candidate_manifest, args.truth,
                                args.output_dir, args.population_keys, args.n_bootstrap, args.seed)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"PPR comparison stopped: {exc}\n")
    print(json.dumps({"pooled": report["pooled"], "coverage": report["coverage"],
                      "report": str(args.output_dir / "report.json")}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

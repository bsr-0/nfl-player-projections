#!/usr/bin/env python3
"""Run a frozen-panel, seasonal Plan B joint MAE allocation experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.plan_b_joint_mae_backtester import ARMS, KEYS, run_backtest
from src.models.team_hierarchical.features import PLAN_B_TARGETS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=PLAN_B_TARGETS)
    ap.add_argument("--preflight-dir", required=True, type=Path)
    ap.add_argument("--test-seasons", nargs="+", type=int, required=True)
    ap.add_argument("--prior-run-predictions", type=Path,
                    help="For 2023-2025 confirmation, require exact old-run row/label/baseline equality")
    ap.add_argument("--first-confirmation-run", action="store_true",
                    help="Explicit opt-out of --prior-run-predictions for a population that has never "
                         "been confirmed against 2023+ before (e.g. after uncapping the roster-slot "
                         "population, or for a target with no prior confirmation at all) -- there is no "
                         "honest prior run to compare against. Requires the preflight's own coverage.json "
                         "to show excluded_rows == 0 (population-complete) as a substitute provenance "
                         "check, recorded in the manifest so this is auditable rather than a silently "
                         "skipped guard. Mutually exclusive with --prior-run-predictions.")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--epsilon", type=float, default=0.001)
    ap.add_argument("--penalty", type=float, default=0.00001)
    ap.add_argument("--smooth-width", type=float, default=0.002)
    ap.add_argument("--other-weight", type=float, default=1.0,
                    help="Weight on the omitted-player bucket's mean absolute error relative to "
                         "the mean player-row error (1.0 = original equal weighting, 0.0 = "
                         "unconstrained row-MAE-only, see joint_other_mae.py module docstring)")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.test_seasons != sorted(set(args.test_seasons)):
        ap.error("test seasons must be unique and increasing")
    if args.prior_run_predictions and args.first_confirmation_run:
        ap.error("--prior-run-predictions and --first-confirmation-run are mutually exclusive")
    if not (args.prior_run_predictions or args.first_confirmation_run) and any(s >= 2023 for s in args.test_seasons):
        ap.error("2023+ confirmation requires --prior-run-predictions (matched-row verification) "
                 "or an explicit --first-confirmation-run (population-completeness verification)")
    output = args.output_dir.resolve()
    if output.exists():
        ap.error(f"output exists: {output}")
    output.mkdir(parents=True)
    manifest = {"target": args.target, "test_seasons": args.test_seasons,
                "epsilon": args.epsilon, "penalty": args.penalty,
                "smooth_width": args.smooth_width, "other_weight": args.other_weight,
                "n_bootstrap": args.n_bootstrap,
                "code_sha256": {name: sha256(ROOT / name) for name in (
                    "src/models/team_hierarchical/joint_other.py",
                    "src/models/team_hierarchical/joint_other_mae.py",
                    "src/evaluation/plan_b_joint_mae_backtester.py",
                    "scripts/evaluate_plan_b_joint_mae.py",
                    "src/evaluation/team_hierarchical_backtester.py",
                    "src/models/team_hierarchical/features.py")}}
    try:
        preflight = args.preflight_dir.resolve()
        preflight_manifest = json.loads((preflight / "manifest.json").read_text())
        if preflight_manifest["target"] != args.target or preflight_manifest["mode"] != "preflight":
            raise ValueError("preflight target or mode mismatch")
        panel_path = preflight / "input_panel.csv"
        panel_hash = sha256(panel_path)
        if panel_hash != preflight_manifest["joined_input"]["sha256"]:
            raise ValueError("frozen panel SHA-256 differs from its preflight manifest")
        manifest.update(input_panel=str(panel_path), input_panel_sha256=panel_hash,
                        preflight_manifest_sha256=sha256(preflight / "manifest.json"))
        panel = pd.read_csv(panel_path, dtype={"player_id": str, "team": str, "position": str, "slot": str},
                            float_precision="round_trip")
        if len(panel) != preflight_manifest["joined_input"]["rows"]:
            raise ValueError("frozen panel row count differs from its preflight manifest")
        if args.first_confirmation_run:
            coverage = json.loads((preflight / "coverage.json").read_text())
            if coverage.get("excluded_rows") != 0:
                raise ValueError(
                    "--first-confirmation-run requires a population-complete preflight "
                    f"(coverage.json excluded_rows == 0), got {coverage.get('excluded_rows')!r}")
            manifest["first_confirmation_run"] = {
                "coverage_sha256": sha256(preflight / "coverage.json"),
                "eligible_rows": coverage.get("eligible_rows"),
                "excluded_rows": coverage.get("excluded_rows")}
        rows, report = run_backtest(panel, args.target, args.test_seasons,
                                    epsilon=args.epsilon, penalty=args.penalty,
                                    smooth_width=args.smooth_width,
                                    other_weight=args.other_weight,
                                    n_bootstrap=args.n_bootstrap)
        if args.prior_run_predictions:
            old_path = args.prior_run_predictions.resolve()
            prior = pd.read_csv(old_path, dtype={"player_id": str, "team": str, "position": str},
                                float_precision="round_trip")
            prior = prior.sort_values(KEYS).reset_index(drop=True)
            current = rows.sort_values(KEYS).reset_index(drop=True)
            pd.testing.assert_frame_equal(current[KEYS], prior[KEYS])
            for col in ("actual_share", "rolling3"):
                if not np.allclose(current[col].to_numpy(float), prior[col].to_numpy(float), rtol=0, atol=1e-14):
                    raise ValueError(f"prior run {col} differs on matched rows")
            manifest["prior_run_predictions"] = {"path": str(old_path), "sha256": sha256(old_path),
                                                 "rows": len(prior)}
        rows.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
        saved = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
        if len(saved) != len(rows):
            raise ValueError("saved-row count differs from in-memory predictions")
        for arm in ARMS:
            measured = float(np.abs(saved[arm] - saved.actual_share).mean())
            if not np.isclose(measured, report["pooled"][arm]["mae"], atol=1e-12, rtol=0):
                raise ValueError(f"saved-row MAE mismatch: {arm}")
        manifest["predictions_sha256"] = sha256(output / "predictions.csv")
        report.update(predictions_sha256=manifest["predictions_sha256"], saved_row_mae_verified=True)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(f"Completed {args.target} on {len(rows):,} held-out rows: {output}", flush=True)
        return 0
    except Exception as exc:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (output / "failure.json").write_text(json.dumps({"status": "failed", "type": type(exc).__name__,
                                                          "error": str(exc)}, indent=2) + "\n")
        print(f"Plan B joint MAE run failed: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only Plan B preflight by default; --run explicitly enables fitting."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sqlite3
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels

from config.settings import DB_PATH
from src.evaluation.team_hierarchical_backtester import ARMS, KEYS, run_backtest, validate_panel
from src.models.team_allocation.features import VOLUME_COLS
from src.models.team_hierarchical.features import load_slot_share_rows


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--slots-csv", type=Path)
    parser.add_argument("--target", required=True, choices=VOLUME_COLS)
    parser.add_argument("--seasons", nargs=2, type=int, default=[2013, 2025], metavar=("FIRST", "LAST"))
    parser.add_argument("--test-seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--run", action="store_true", help="Fit models; omitted means preflight only")
    parser.add_argument("--preflight-manifest", type=Path,
                        help="Required with --run; pins exact slot CSV and joined input-panel hashes")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    lo, hi = args.seasons
    if lo > hi or args.test_seasons != sorted(set(args.test_seasons)) or not all(lo < s <= hi for s in args.test_seasons):
        parser.error("require FIRST < each increasing unique test season <= LAST")
    if args.maxiter < 1 or args.n_bootstrap < 100:
        parser.error("maxiter must be positive and n-bootstrap must be >= 100")
    if args.run and args.preflight_manifest is None:
        parser.error("--run requires --preflight-manifest from a completed preflight")
    if args.preflight_manifest and not args.run:
        parser.error("--preflight-manifest applies only to --run")
    out = args.output_dir.resolve()
    if out.exists():
        parser.error(f"output already exists: {out}; choose a new directory")
    out.mkdir(parents=True)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "run" if args.run else "preflight", "target": args.target,
                "database": str(args.db.resolve()), "database_access": "read-only transaction",
                "seasons": [lo, hi], "test_seasons": args.test_seasons,
                "parameters": {"maxiter": args.maxiter, "n_bootstrap": args.n_bootstrap, "seed": args.seed},
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "sklearn": sklearn.__version__, "statsmodels": statsmodels.__version__},
                "code_sha256": {}}
    for relative in ("scripts/evaluate_plan_b_shares.py", "src/evaluation/team_hierarchical_backtester.py",
                     "src/models/team_hierarchical/features.py", "src/models/team_hierarchical/models.py",
                     "src/models/team_allocation/features.py", "src/evaluation/paired_ppr_comparison.py",
                     "src/utils/leakage.py"):
        manifest["code_sha256"][relative] = _sha((ROOT / relative).read_bytes())
    _write_json(out / "manifest.json", manifest)
    try:
        expected = None
        if args.preflight_manifest:
            expected_blob = args.preflight_manifest.read_bytes()
            expected = json.loads(expected_blob)
            if (expected.get("mode") != "preflight" or expected.get("target") != args.target
                    or expected.get("seasons") != [lo, hi]
                    or expected.get("test_seasons") != args.test_seasons
                    or "slots_csv" not in expected or "joined_input" not in expected):
                raise ValueError("preflight manifest does not match the run target, seasons, or required inputs")
            manifest["preflight_manifest"] = {"path": str(args.preflight_manifest.resolve()),
                                               "sha256": _sha(expected_blob)}
        slots = None
        if args.slots_csv:
            blob = args.slots_csv.read_bytes()
            slots = pd.read_csv(io.BytesIO(blob), dtype={"player_id": str, "team": str})
            manifest["slots_csv"] = {"path": str(args.slots_csv.resolve()), "sha256": _sha(blob), "rows": len(slots)}
        if expected and (slots is None or manifest["slots_csv"]["sha256"] != expected["slots_csv"]["sha256"]):
            raise ValueError("slot CSV hash differs from the passed preflight; rebuild and preflight again")
        with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")  # one consistent SQLite snapshot for both tables
            panel, coverage = load_slot_share_rows(args.target, list(range(lo, hi + 1)), conn,
                                                   slots_frame=slots, return_coverage=True)
        panel = panel.sort_values(KEYS).reset_index(drop=True)
        # Hash the actual queried/joined snapshot, not a concurrently changing
        # database file. Save it so the model input is independently reviewable.
        snapshot = panel.to_csv(index=False, float_format="%.17g").encode()
        if expected and (_sha(snapshot) != expected["joined_input"]["sha256"]
                         or len(panel) != expected["joined_input"]["rows"]):
            raise ValueError("joined player-panel hash or row count differs from the passed preflight; do not mix inputs")
        (out / "input_panel.csv").write_bytes(snapshot)
        manifest["joined_input"] = {"sha256": _sha(snapshot), "rows": len(panel), "columns": list(panel)}
        excluded = pd.DataFrame(coverage.pop("excluded_keys"), columns=KEYS)
        excluded.to_csv(out / "excluded_keys.csv", index=False)
        manifest["excluded_keys_sha256"] = _sha((out / "excluded_keys.csv").read_bytes())
        _write_json(out / "coverage.json", coverage)
        _write_json(out / "manifest.json", manifest)
        readiness = validate_panel(panel, args.target, args.test_seasons)
        readiness.update(status="ready_to_attempt_fit", coverage=coverage)
        _write_json(out / "preflight.json", readiness)
        if not args.run:
            print(f"Preflight passed: {len(panel):,} represented rows; no models fitted. {out}")
            return 0
        rows, report = run_backtest(panel, args.target, args.test_seasons, maxiter=args.maxiter,
                                    n_bootstrap=args.n_bootstrap, seed=args.seed)
        rows.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
        saved = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
        # Independently recalculate the reported pooled MAEs from the saved rows.
        if len(saved) != sum(f["n_test"] for f in report["folds"]):
            raise ValueError("saved prediction row count differs from declared folds")
        for arm in ARMS:
            measured = float(np.abs(saved[arm] - saved.actual_share).mean())
            if not np.isclose(measured, report["pooled"][arm]["mae"], rtol=0, atol=1e-12):
                raise ValueError(f"saved-row MAE disagrees with report: {arm}")
        report.update(coverage=coverage, saved_row_mae_verified=True,
                      predictions_sha256=_sha((out / "predictions.csv").read_bytes()))
        _write_json(out / "report.json", report)
        print(f"Completed {args.target}: {len(rows):,} held-out rows. {out}")
        return 0
    except Exception as exc:
        # Never publish a success report after a partial/failed fit.
        _write_json(out / "manifest.json", manifest)
        _write_json(out / "failure.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        print(f"Plan B stopped: {exc}\nFailure details: {out / 'failure.json'}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

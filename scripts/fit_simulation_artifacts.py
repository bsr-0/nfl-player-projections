#!/usr/bin/env python3
"""Fit the calibrated-copula simulation artifacts that generate_simulation_data.py serves.

Fits on one independently verified production OOF panel, keyed to the game each
actual came from, using the same selection rule the backtest applies at its
next week. Optionally attaches a verified backtest run on the same panel as the
evidence for (or against) the served dependence; that evidence is recorded,
never used to change the fit.

    python scripts/fit_simulation_artifacts.py --oof-run-dir data/experiments/oof_panels/<run_id>
    python scripts/fit_simulation_artifacts.py --oof-run-dir ... --backtest-run-dir data/experiments/calibrated_sim_<date>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_oof_panel import verify as verify_oof_panel  # noqa: E402
from src.models.calibrated_simulation import (  # noqa: E402
    ANALOG_K,
    LEGACY_MIN_STRATUM_ROWS,
    candidate_grid,
    fit_simulation_artifacts,
    save_simulation_artifacts,
)
from src.models.oof_capture import target_game_panel  # noqa: E402

DEFAULT_ROOT = PROJECT_ROOT / "data" / "models" / "simulation"


def backtest_evidence(backtest_dir: Path, panel_sha256: str) -> dict:
    """The verified backtest's primaries and gate, if it scored this exact panel."""
    from scripts.verify_calibrated_simulation import verify as verify_backtest
    verification = verify_backtest(backtest_dir)
    manifest = json.loads((backtest_dir / "manifest.json").read_text())
    if manifest.get("source_oof_panel_sha256") != panel_sha256:
        raise ValueError("backtest run was scored on a different OOF panel than the one being fitted")
    report = json.loads((backtest_dir / "report.json").read_text())
    return {
        "run_dir": str(backtest_dir.resolve()), "verification": verification["status"],
        "phase": report["phase"], "scoring_population": report["scoring_population"],
        "primary_comparisons": [
            {key: entry[key] for key in ("candidate", "baseline", "table", "metric", "point_estimate",
                                         "ci_low", "ci_high", "p_value", "significant_holm")}
            for entry in report["comparisons"]["primary"]],
        "promotion_gate": report["promotion_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--backtest-run-dir", type=Path, default=None)
    parser.add_argument("--legacy-min-stratum-rows", type=int, nargs="+", default=list(LEGACY_MIN_STRATUM_ROWS))
    parser.add_argument("--analog-k", type=int, nargs="+", default=list(ANALOG_K))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from src.models.position_models import _git_commit
    candidates = candidate_grid(args.legacy_min_stratum_rows, args.analog_k)
    verification = verify_oof_panel(args.oof_run_dir)
    evidence = (backtest_evidence(args.backtest_run_dir, verification["panel_sha256"])
                if args.backtest_run_dir else "no backtest attached; dependence gain unproven")
    panel = target_game_panel(pd.read_parquet(args.oof_run_dir / "panel.parquet"))
    artifacts = fit_simulation_artifacts(panel, candidates=candidates, seed=args.seed)
    run_dir = save_simulation_artifacts(artifacts, args.output_root, provenance={
        "source_oof_run_dir": str(args.oof_run_dir.resolve()),
        "source_oof_panel_sha256": verification["panel_sha256"], "git_commit": _git_commit(),
        "candidates": [c.id for c in candidates], "seed": int(args.seed),
        "dependence_evidence": evidence,
    })
    selection = artifacts.selection
    print(f"marginal: {selection['candidate']} ({selection['reason']}); dependence: role_factor "
          f"({artifacts.dependence.diagnostics.get('status')}); fitted through "
          f"{artifacts.fitted_through[0]} week {artifacts.fitted_through[1]} on {artifacts.n_rows} rows")
    print(f"wrote {run_dir} and {args.output_root / 'latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

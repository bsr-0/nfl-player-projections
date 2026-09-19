#!/usr/bin/env python3
"""Compare matched Phase 7 PPR folds with Phase 2 OOF usage predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.single_week_ppr.participation_integration import run_participation_integration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2-oof", type=Path, required=True)
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--positions", nargs="+", default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("data/experiments/phase3_participation"))
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--phase2-model", default="hist_gbm",
                    help="model label selected from Phase 2's OOF file")
    args = ap.parse_args()
    _, summary, report = run_participation_integration(
        args.phase2_oof, positions=args.positions, seasons=args.seasons,
        output_dir=args.output_dir, n_bootstrap=args.n_bootstrap, phase2_model=args.phase2_model,
    )
    print(summary.to_string(index=False))
    print(f"all requested folds completed: {report['all_requested_folds_completed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

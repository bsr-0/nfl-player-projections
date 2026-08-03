#!/usr/bin/env python3
"""
Run the walk-forward backtest at multiple Ridge alpha values and emit a
per-position comparison summary.

Implements Step 3 of the April 14 council verdict
(council-transcript-20260414-034617.md): sweep alpha in {0.3, 1, 3, 10}
and assess whether the QB/TE std_ratio - r gap is reproduced across
folds.

Each run produces its own ``ts_backtest_*_predictions.csv`` plus the
companion ``*.json`` metrics file.  After all alphas finish, this script
calls ``scripts/analyze_std_ratio.py`` on each predictions CSV and prints
a side-by-side comparison so the gap's stability can be eyeballed before
authorizing per-position alpha tuning.

Usage:
    python scripts/run_alpha_sweep.py                        # season auto, alphas {0.3, 1, 3, 10}
    python scripts/run_alpha_sweep.py --season 2025
    python scripts/run_alpha_sweep.py --alphas 0.1 0.3 1 3 10 30
    python scripts/run_alpha_sweep.py --positions QB TE      # subset of positions
    python scripts/run_alpha_sweep.py --analyze-only         # skip runs, only re-analyze last sweep

Requires pandas / numpy / scikit-learn (same dependencies as
run_ts_backtest.py).  The analysis step itself is stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ALPHAS = (0.3, 1.0, 3.0, 10.0)
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
ANALYZE_SCRIPT = PROJECT_ROOT / "scripts" / "analyze_std_ratio.py"


def _latest_predictions_csv() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        RESULTS_DIR.glob("ts_backtest_*_predictions.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_backtest(season: int | None, alpha: float, positions: List[str] | None) -> Path:
    """Run a single backtest at ``alpha`` and return the produced predictions CSV."""
    before = {p.name for p in RESULTS_DIR.glob("ts_backtest_*_predictions.csv")}

    from src.evaluation.ts_backtester import run_ts_backtest  # lazy import

    print(f"\n{'=' * 78}")
    print(f"  Backtest: season={season or 'auto'}  alpha={alpha}  positions={positions or 'all'}")
    print(f"{'=' * 78}")

    run_ts_backtest(
        season=season,
        model_type="ridge",
        positions=positions,
        verbose=True,
        ridge_alpha=alpha,
    )

    # Find the new predictions CSV (newest file not present before).
    candidates = sorted(
        (p for p in RESULTS_DIR.glob("ts_backtest_*_predictions.csv") if p.name not in before),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("Backtest finished but no new predictions CSV was found in data/backtest_results/")
    return candidates[0]


def _analyze(csv_path: Path) -> Dict:
    """Invoke analyze_std_ratio.py --json on a predictions CSV and return the parsed report."""
    result = subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT), "--json", str(csv_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _format_sweep_table(reports: Dict[float, Tuple[Path, Dict]]) -> str:
    """Side-by-side per-position gap table across alpha values."""
    alphas = sorted(reports)
    positions = sorted({pos for _, r in reports.values() for pos in r["per_position"]})

    lines = []
    lines.append("Alpha sweep — per-position gap = std_ratio - r")
    lines.append("=" * 78)
    lines.append("")

    header = f"  {'pos':<4} " + " ".join(f"a={a:>5.2f}" for a in alphas)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    # Aggregate gap row
    lines.append("  Season-aggregate gap:")
    for pos in positions:
        cells = []
        for a in alphas:
            _, rep = reports[a]
            s = rep["per_position"].get(pos, {})
            cells.append(f"{s.get('gap', float('nan')):>+7.3f}")
        lines.append(f"  {pos:<4} " + " ".join(cells))

    # Per-week mean & SE row
    lines.append("")
    lines.append("  Per-week gap mean +/- SE:")
    for pos in positions:
        cells = []
        for a in alphas:
            _, rep = reports[a]
            wstats = rep["per_position_per_week"].get(pos, {})
            gaps = [w["gap"] for w in wstats.values()]
            if not gaps:
                cells.append("   n/a  ")
                continue
            from statistics import mean, stdev
            m = mean(gaps)
            se = (stdev(gaps) / (len(gaps) ** 0.5)) if len(gaps) > 1 else 0.0
            cells.append(f"{m:+.2f}±{se:.2f}")
        lines.append(f"  {pos:<4} " + " ".join(f"{c:>9}" for c in cells))

    lines.append("")
    lines.append("Verdict thresholds (council 2026-04-14, Step 3):")
    lines.append("  - If gap_mean is stable across alphas AND |gap_mean| > 2*SE, authorize per-position alpha tuning.")
    lines.append("  - If gap collapses at lower alpha (e.g. a=0.3) or SE > |gap_mean|, the gap is an artifact; no fix.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", "-s", type=int, default=None, help="Season to backtest (default: latest).")
    parser.add_argument(
        "--alphas", "-a",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHAS),
        help=f"Ridge alpha values to sweep (default: {' '.join(str(x) for x in DEFAULT_ALPHAS)}).",
    )
    parser.add_argument(
        "--positions", "-p",
        nargs="+",
        default=None,
        help="Positions to backtest (default: QB RB WR TE).",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip backtests; re-analyze the most recent existing predictions CSV per alpha is not possible "
             "(the file name does not encode alpha).  Use this only after a fresh sweep, with --alpha-csv.",
    )
    parser.add_argument(
        "--alpha-csv",
        nargs="+",
        default=None,
        metavar="ALPHA=PATH",
        help="Override sweep with explicit (alpha, csv) pairs, e.g. 0.3=path/to/preds.csv 1.0=path/to/preds.csv. "
             "Useful with --analyze-only when you already have the four runs on disk.",
    )
    args = parser.parse_args()

    reports: Dict[float, Tuple[Path, Dict]] = {}

    if args.alpha_csv:
        # Manual mapping path: skip backtest execution entirely.
        for spec in args.alpha_csv:
            if "=" not in spec:
                print(f"error: --alpha-csv entries must be ALPHA=PATH, got {spec!r}", file=sys.stderr)
                return 2
            alpha_str, path_str = spec.split("=", 1)
            alpha = float(alpha_str)
            csv_path = Path(path_str)
            if not csv_path.exists():
                print(f"error: {csv_path} does not exist", file=sys.stderr)
                return 2
            reports[alpha] = (csv_path, _analyze(csv_path))
    elif args.analyze_only:
        print("error: --analyze-only requires --alpha-csv to map alphas to predictions CSVs.", file=sys.stderr)
        return 2
    else:
        for alpha in args.alphas:
            csv_path = _run_backtest(args.season, alpha, args.positions)
            print(f"  predictions written: {csv_path.name}")
            reports[alpha] = (csv_path, _analyze(csv_path))

    print()
    print(_format_sweep_table(reports))

    # Write a machine-readable summary alongside the run.
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    summary = {
        "alphas": sorted(reports),
        "season": args.season,
        "positions": args.positions,
        "runs": {
            str(a): {
                "predictions_csv": _rel(p),
                "per_position": rep["per_position"],
            }
            for a, (p, rep) in reports.items()
        },
    }
    summary_path = RESULTS_DIR / "alpha_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSummary written: {summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fail-closed promotion/readiness gate for reconstruction-only Plan A.

The yardage candidate may be research-ready while still being ineligible to
replace production because it does not model receptions, touchdowns, or
passing production. This gate checks both conditions explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=Path("data/experiments/plan_a_reconstruction_served_report.json"))
    ap.add_argument("--artifact-dir", type=Path, default=Path("data/experiments/plan_a_reconstruction_served_artifacts"))
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    checks = {}
    for season in args.seasons:
        row = report.get("report", {}).get(str(season), {})
        ci = row.get("vs_rolling3_bootstrap", {})
        checks[str(season)] = {
            "artifact_rushing_yards": (args.artifact_dir / "rushing_yards" / f"pre{season}.joblib").exists(),
            "artifact_receiving_yards": (args.artifact_dir / "receiving_yards" / f"pre{season}.joblib").exists(),
            "same_row_comparison": row.get("n_rows", 0) > 0,
            "ci_excludes_zero": bool(ci) and ci.get("ci_high", 1.0) < 0,
        }
    research_ready = all(all(v for v in check.values()) for check in checks.values())
    payload = {
        "status": "research_ready_not_promotable" if research_ready else "blocked",
        "research_ready": research_ready,
        "production_promotable": False,
        "reason": "Yardage-only reconstruction does not model receptions, touchdowns, or passing production; do not replace full-PPR production artifacts.",
        "checks": checks,
        "report": str(args.report),
        "artifact_dir": str(args.artifact_dir),
    }
    print(json.dumps(payload, indent=2))
    return 0 if research_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

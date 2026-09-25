#!/usr/bin/env python3
"""Reload a frozen joint Plan A artifact and write a full-PPR shadow export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.team_allocation.features import load_share_rows
from src.models.team_allocation.joint_serving import JointPlanAArtifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--weeks", nargs="+", type=int, help="Optional completed weeks within the requested seasons.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--include-actual", action="store_true")
    args = ap.parse_args()

    artifact = JointPlanAArtifact.load(args.artifact)
    rows = load_share_rows(seasons=args.seasons)
    if args.weeks:
        rows = rows[rows.week.isin(args.weeks)].copy()
        if rows.empty:
            raise ValueError("no rows for requested season/week shadow export")
    export = artifact.predict(rows, include_actual=args.include_actual)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.output, index=False)
    metadata = artifact.metadata() | {
        "output": str(args.output),
        "seasons": args.seasons,
        "weeks": args.weeks,
        "n_rows": int(len(export)),
        "include_actual": bool(args.include_actual),
        "columns": list(export.columns),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    print(json.dumps(metadata, indent=2, default=float))


if __name__ == "__main__":
    main()

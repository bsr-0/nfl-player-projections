#!/usr/bin/env python3
"""Freeze the selected full-PPR Plan A candidate into one immutable artifact."""
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
    ap.add_argument("--selector-report", type=Path, required=True)
    ap.add_argument("--train-through", type=int, required=True)
    ap.add_argument("--train-from", type=int, help="Optional first training season for an exact historical fold replay.")
    ap.add_argument("--version", required=True)
    ap.add_argument(
        "--selector-fold", type=int,
        help="Freeze the choices applied to this held-out fold, for an honest historical shadow export.",
    )
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    args = ap.parse_args()

    selector = json.loads(args.selector_report.read_text())
    if args.selector_fold is None:
        allocation_arms = selector["final_allocation_arms"]
        team_total_arms = selector["final_team_total_arms"]
        source_selector = str(args.selector_report)
    else:
        matches = [row for row in selector["folds"] if int(row["fold"]) == args.selector_fold]
        if len(matches) != 1:
            raise ValueError(f"selector report has no unique fold {args.selector_fold}")
        allocation_arms = matches[0]["allocation_arms"]
        team_total_arms = matches[0]["team_total_arms"]
        source_selector = f"{args.selector_report}#fold={args.selector_fold}"
    artifact = JointPlanAArtifact.fit(
        load_share_rows(),
        allocation_arms=allocation_arms,
        team_total_arms=team_total_arms,
        train_through=args.train_through,
        train_from=args.train_from,
        version=args.version,
        source_selector=source_selector,
    )
    artifact.save(args.artifact)
    metadata = artifact.metadata() | {
        "selector_noninferiority_margin": selector.get("noninferiority_margin"),
        "selector_inner_gate_bootstraps": selector.get("inner_gate_bootstraps"),
        "artifact": str(args.artifact),
        "selector_fold": args.selector_fold,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    print(json.dumps(metadata, indent=2, default=float))


if __name__ == "__main__":
    main()

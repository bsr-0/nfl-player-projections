#!/usr/bin/env python3
"""Main entrypoint for the Phase 1 → 2 → 3 participation system.

This system remains experimental: it creates canonical population data,
produces Phase 2 OOF usage artifacts, and evaluates them in Phase 3.  It never
edits Phase 7's serving feature list or FINAL_CONFIG.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "experiments" / "participation_system"


def stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """Construct deterministic stage commands; separately unit-tested."""
    root = args.output_root
    python = sys.executable
    commands: list[tuple[str, list[str]]] = []
    if "phase1" in args.stages:
        commands.append(("phase1", [
            python, "scripts/build_canonical_player_weeks.py", "--write", "--db", str(args.db), "--seasons",
            str(args.seasons[0]), str(args.seasons[1]), "--audit-csv", str(root / "phase1_audit.csv"),
        ]))
    if "phase2" in args.stages:
        command = [python, "scripts/run_phase2_participation.py", "--db", str(args.db),
                   "--min-train-seasons", str(args.min_train_seasons),
                   "--output-dir", str(root / "phase2")]
        if args.include_pregame_injury:
            command.append("--include-pregame-injury")
        commands.append(("phase2", command))
    if "phase3" in args.stages:
        commands.append(("phase3", [
            python, "scripts/run_phase3_participation_integration.py", "--phase2-oof",
            str(root / "phase2" / "oof_predictions.csv"), "--seasons",
            *[str(s) for s in args.phase3_seasons], "--output-dir", str(root / "phase3"),
            "--n-bootstrap", str(args.n_bootstrap),
            "--phase2-model", args.phase2_model,
        ]))
    return commands


def _canonical_exists(db: Path) -> bool:
    if not db.exists():
        return False
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canonical_player_weeks'"
        ).fetchone() is not None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "nfl_data.db")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--stages", nargs="+", choices=("phase1", "phase2", "phase3"),
                    default=["phase1", "phase2", "phase3"])
    ap.add_argument("--seasons", nargs=2, type=int, default=[2013, 2026])
    ap.add_argument("--min-train-seasons", type=int, default=3)
    ap.add_argument("--include-pregame-injury", action="store_true")
    ap.add_argument("--phase3-seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--phase2-model", default="hist_gbm",
                    help="Phase 2 model label promoted to the Phase 3 experiment")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.seasons = sorted(args.seasons)
    args.output_root = args.output_root.resolve()
    commands = stage_commands(args)

    # A partial run is allowed only when its durable prerequisite exists.
    if "phase2" in args.stages and "phase1" not in args.stages and not _canonical_exists(args.db):
        raise SystemExit("phase2 requires canonical_player_weeks; include phase1 or build it first")
    if "phase3" in args.stages and "phase2" not in args.stages:
        required = args.output_root / "phase2" / "oof_predictions.csv"
        if not required.exists():
            raise SystemExit(f"phase3 requires prior canonical Phase 2 OOF artifact: {required}")

    if args.dry_run:
        for stage, command in commands:
            print(f"{stage}: {' '.join(command)}")
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    for stage, command in commands:
        print(f"\n=== participation system: {stage} ===")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        completed.append(stage)
    manifest = {
        "system": "participation_opportunity", "completed_stages": completed,
        "database": str(args.db), "seasons": args.seasons,
        "include_pregame_injury": bool(args.include_pregame_injury),
        "phase2_model": args.phase2_model,
        "phase2_dir": str(args.output_root / "phase2"),
        "phase3_dir": str(args.output_root / "phase3"),
    }
    (args.output_root / "participation_system_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\ncompleted participation system stages: {', '.join(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

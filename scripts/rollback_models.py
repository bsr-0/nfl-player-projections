#!/usr/bin/env python
"""List or restore archived weekly model artifacts.

`model_metadata.json` claimed `rollback_available: true` for years while no
weights were archived anywhere, so there was no command to run when a
training run destroyed the previous models (GAPS.md 2026-09-24). Snapshots
are now taken before each production training run; this is how you use one.

    python scripts/rollback_models.py --list
    python scripts/rollback_models.py --restore 20260925T031500Z
    python scripts/rollback_models.py --restore latest

Restoring overwrites the served artifacts in place. It does not archive the
current state first -- you are restoring because the current state is
unwanted, and snapshotting it would evict a good snapshot under the default
two-version retention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import MODELS_DIR  # noqa: E402
from src.utils.model_rollback import available_rollbacks, restore_models  # noqa: E402


def _describe(snapshot: Path) -> str:
    files = [f for f in snapshot.iterdir() if f.is_file()]
    size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    return f"{snapshot.name}  ({len(files)} files, {size_mb:.0f} MB)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="show available snapshots")
    group.add_argument("--restore", metavar="VERSION",
                       help="snapshot name, or 'latest' for the most recent")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    snapshots = available_rollbacks(args.models_dir)
    if not snapshots:
        print(f"No rollback snapshots in {args.models_dir / 'rollback'}.")
        print("Snapshots are created by production training runs "
              "(python -m src.models.train); walk-forward runs do not create them.")
        return 1

    if args.list:
        print(f"Rollback snapshots in {args.models_dir / 'rollback'} (oldest first):")
        for snapshot in snapshots:
            print(f"  {_describe(snapshot)}")
        return 0

    wanted = snapshots[-1] if args.restore == "latest" else args.models_dir / "rollback" / args.restore
    if wanted not in snapshots:
        print(f"Unknown snapshot {args.restore!r}. Available: "
              f"{', '.join(s.name for s in snapshots)}")
        return 1

    restored = restore_models(wanted, args.models_dir)
    print(f"Restored {len(restored)} artifact(s) from {wanted.name}:")
    for path in restored:
        print(f"  {path.name}")
    print("\nThe served models are now the snapshot's. model_metadata.json was "
          "restored alongside them, so its training_date describes what is "
          "actually on disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

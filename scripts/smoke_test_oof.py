#!/usr/bin/env python
"""Fast end-to-end check that a walk-forward OOF run can complete.

A real run (`python -m src.models.train --walk-forward`) spends 15-20 hours
in Optuna, and every failure so far came after that: the per-fold metrics
write, coverage accounting, game context, the panel verifier. This runs the
same walk-forward -- same folds, positions, data gates, feature pipeline,
capture, panel assembly -- with hyperparameter tuning off and the --fast
config (fewer CV folds), then runs scripts/verify_oof_panel.py on the result.
Most of a real run is Optuna, so the end-of-run steps are reached far sooner;
how much sooner depends on the data and has not been measured.

    python scripts/smoke_test_oof.py                  # all positions, all folds
    python scripts/smoke_test_oof.py --positions QB   # quicker, less coverage

Exit code 0 means a verified panel was written; anything else is a failure,
printed with its traceback and elapsed time.

Isolation: models train in a temporary MODELS_DIR, and every OOF output goes
to data/experiments/oof_smoke/<UTC stamp>/ -- never the real oof_panels/,
"latest" pointer or walk_forward_fold_metrics.json. Isolation is CHECKED, not
assumed: the run hashes everything a validation run must not change -- the
served model/preprocessing files, data/advanced_model_results.json,
data/utilization_percentile_bounds.json and data/backtest_results/*.json --
before and after, and fails if any differ. (Validation folds used to
overwrite exactly those; see GAPS.md 2026-09-28.)
Not isolated: like a real run it reads (and may refresh) the database and
writes lineage snapshots under data/artifacts/, so do not run it while a
real training job is using the database.

What a pass does NOT show: that tuned training succeeds (Optuna is off), or
anything about accuracy -- the models are untuned and the smoke panel's
metrics are meaningless. It shows the run reaches a verified panel.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR  # noqa: E402

SMOKE_ROOT = DATA_DIR / "experiments" / "oof_smoke"


def _protected_state() -> dict:
    """Content hashes of the files a validation run must leave untouched."""
    import hashlib

    import config.settings as settings
    from src.utils.model_rollback import covered_artifacts

    paths = list(covered_artifacts(settings.PRODUCTION_MODELS_DIR))
    paths += [DATA_DIR / "advanced_model_results.json",
              DATA_DIR / "utilization_percentile_bounds.json"]
    paths += sorted((DATA_DIR / "backtest_results").glob("*.json"))
    def _digest(path: Path) -> str:
        # Chunked: served weights run to hundreds of MB.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return {str(path): _digest(path) for path in paths if path.is_file()}


def _changed_since(before: dict) -> list:
    after = _protected_state()
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


def run_smoke(out_dir: Path, *, positions=None, skip_cache_check: bool = False,
              skip_quality_gate: bool = False) -> dict:
    """Run the untuned walk-forward into `out_dir` and verify its panel."""
    from scripts.verify_oof_panel import verify
    from src.models.train import train_models
    from src.utils.models_dir import redirect_models_dir

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    before = _protected_state()
    try:
        with tempfile.TemporaryDirectory() as tmp, redirect_models_dir(tmp):
            result = train_models(
                positions=positions, walk_forward=True, tune_hyperparameters=False,
                fast=True, oof_label="smoke", oof_output_dir=out_dir,
                skip_cache_check=skip_cache_check, skip_quality_gate=skip_quality_gate)
        if result is None:
            # train_models returns None only when a data gate blocks training.
            raise RuntimeError("training was blocked before walk-forward ran (see the "
                               "TRAINING BLOCKED report above); a real run would stop there too")
        runs = sorted(p for p in (out_dir / "oof_panels").glob("*") if p.is_dir())
        if len(runs) != 1:
            raise RuntimeError(f"expected exactly one OOF panel run in {out_dir / 'oof_panels'}, "
                               f"found {len(runs)}")
        verification = verify(runs[0])
        (runs[0] / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    finally:
        leaked = _changed_since(before)
        if leaked:
            print("\nLEAK: the validation run modified files it must not touch:\n  "
                  + "\n  ".join(leaked), file=sys.stderr)
    if leaked:
        raise RuntimeError(f"validation run modified {len(leaked)} protected file(s): {leaked}")
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", nargs="+", default=None,
                        help="positions to run (default: all, as a real run does)")
    parser.add_argument("--skip-cache-check", action="store_true",
                        help="pass through to train_models; mirror what the real run uses")
    parser.add_argument("--skip-quality-gate", action="store_true",
                        help="pass through to train_models; mirror what the real run uses")
    args = parser.parse_args()

    out_dir = SMOKE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = time.monotonic()
    try:
        verification = run_smoke(out_dir, positions=args.positions,
                                 skip_cache_check=args.skip_cache_check,
                                 skip_quality_gate=args.skip_quality_gate)
    except BaseException:
        traceback.print_exc()
        print(f"\nSMOKE FAILED after {(time.monotonic() - started) / 60:.1f} min "
              f"(outputs so far: {out_dir})")
        return 1
    print(f"\nSMOKE PASSED in {(time.monotonic() - started) / 60:.1f} min: verified panel with "
          f"{verification['n_rows']:,} rows, seasons {verification['seasons']}, "
          f"positions {verification['positions']}")
    print(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plan A acceptance gate -- criterion 1 (see docs/TEAM_LEVEL_ALLOCATION_MODELS.md).

Reads the metadata JSON sidecars scripts/train_team_share_model.py already
writes and exits non-zero unless at least one tunable arm (ridge/xgboost)
RELIABLY beats the rolling-3 baseline on share MAE for a target -- "reliably"
meaning the paired bootstrap CI on the MAE delta (src/evaluation/
team_share_backtester.py's bootstrap_mae_delta) excludes 0, not just that the
point-estimate MAE happens to be lower. A lower point estimate that could
easily be noise is not a pass.

This turns "did it beat rolling3" from something read off stdout into a
repeatable check a script (or CI) can act on -- same spirit as the
SeasonAwareTimeSeriesSplit(strict=True) change earlier in this branch: fail
loud, don't rely on someone reading output carefully.

Usage:
    python scripts/check_team_share_acceptance.py                 # all 4 targets
    python scripts/check_team_share_acceptance.py --target targets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODELS_DIR
from src.models.team_allocation.features import VOLUME_COLS


def check_target(target: str) -> bool:
    path = MODELS_DIR / f"team_share_{target}_model_metadata.json"
    if not path.exists():
        print(f"[{target}] SKIP: no metadata found at {path} -- run scripts/train_team_share_model.py first")
        return False

    meta = json.loads(path.read_text())
    pooled = meta.get("backtest", {}).get("pooled", {})
    baseline = pooled.get("rolling3")
    if not baseline:
        print(f"[{target}] FAIL: metadata has no rolling3 baseline in backtest.pooled")
        return False
    baseline_mae = baseline["mae"]

    passed = False
    for arm in ("ridge", "xgboost"):
        m = pooled.get(arm)
        if not m:
            print(f"[{target}] {arm}: no results in metadata, skipping")
            continue
        boot = m.get("vs_rolling3_bootstrap")
        if not boot:
            print(
                f"[{target}] {arm}: FAIL -- no vs_rolling3_bootstrap in metadata "
                "(metadata predates the bootstrap-CI change; re-run scripts/train_team_share_model.py)"
            )
            continue
        reliably_better = bool(boot.get("significant_improvement"))
        status = "PASS" if reliably_better else "FAIL"
        print(
            f"[{target}] {arm}: mae={m['mae']:.4f} vs rolling3={baseline_mae:.4f}  "
            f"delta CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] -> {status}"
        )
        passed = passed or reliably_better

    if passed:
        print(f"[{target}] ACCEPTANCE CRITERION 1: PASSED")
    else:
        print(
            f"[{target}] ACCEPTANCE CRITERION 1: FAILED -- no arm reliably beats "
            "the rolling-3 baseline (95% bootstrap CI on the MAE delta does not "
            "exclude 0 for any tunable arm)"
        )
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=VOLUME_COLS, default=None, help="Default: check all 4.")
    args = ap.parse_args()
    targets = [args.target] if args.target else list(VOLUME_COLS)

    results = {t: check_target(t) for t in targets}
    print()
    failed = [t for t, ok in results.items() if not ok]
    if failed:
        print(f"FAILED targets: {failed}")
        return 1
    print(f"All checked targets ({targets}) passed acceptance criterion 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

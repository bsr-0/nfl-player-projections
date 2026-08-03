#!/usr/bin/env python3
"""
Diagnose whether Vegas features (implied_team_total, spread) are actually
populated in the training pipeline, or silently falling back to constants.

This script is the first step of the predictive-ceiling Phase 1 work
(docs/PREDICTIVE_CEILING_PLAN.md, docs/PHASE_1_VEGAS_FINDINGS.md).  It
answers the question: "before we build new feature wiring, are the
features we already declare actually live in the model?"

Stdlib + sqlite3 only -- runs anywhere, no pandas/numpy/sklearn needed.

Checks (in order, fail-fast):
1. Does config/settings.py declare implied_team_total / spread in
   CAUSAL_FEATURES?  (Should be YES -- they're already in the list.)
2. Does the local schedule table cache spread_line / total_line?
   (Currently NO -- the schema only stores team / score data.)
3. Is nfl_data_py importable in this environment?
   (Required for the runtime fetch path that populates Vegas features.)
4. Does feature_engineering._create_vegas_game_script_features raise
   when nfl_data_py is unavailable, or silently fall back?  (After
   this PR: it logs a warning.  Before: pure silent fallback.)

Exit code: 0 if all four checks pass (Vegas features are live); 1 if
any check fails (Vegas features are silently degraded).

Usage:
    python scripts/check_vegas_features.py
    python scripts/check_vegas_features.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from importlib import util as importlib_util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.py"
DB_PATH = PROJECT_ROOT / "data" / "nfl_data.db"
FE_PATH = PROJECT_ROOT / "src" / "features" / "feature_engineering.py"


# ---------------------------------------------------------------------------
# Check 1: CAUSAL_FEATURES declares Vegas
# ---------------------------------------------------------------------------

def check_causal_features_declared() -> dict:
    text = SETTINGS_PATH.read_text()
    # Find the CAUSAL_FEATURES dict body (cheap textual scan, not full parse).
    m = re.search(r"CAUSAL_FEATURES\s*=\s*\{(.*?)\n\}\n", text, re.DOTALL)
    if not m:
        return {"name": "causal_features_declared", "pass": False, "detail": "CAUSAL_FEATURES not found in settings"}
    body = m.group(1)
    positions = ["QB", "RB", "WR", "TE"]
    missing = []
    for pos in positions:
        # crude per-position slice
        pos_match = re.search(rf'"{pos}"\s*:\s*\[(.*?)\]', body, re.DOTALL)
        if not pos_match:
            missing.append(f"{pos} block missing")
            continue
        pos_body = pos_match.group(1)
        for feat in ("implied_team_total", "spread"):
            if feat not in pos_body:
                missing.append(f"{pos}.{feat}")
    return {
        "name": "causal_features_declared",
        "pass": not missing,
        "detail": "OK" if not missing else f"missing: {', '.join(missing)}",
    }


# ---------------------------------------------------------------------------
# Check 2: schedule table caches Vegas data
# ---------------------------------------------------------------------------

def check_schedule_has_vegas_columns() -> dict:
    if not DB_PATH.exists():
        return {"name": "schedule_caches_vegas", "pass": False, "detail": f"db not found at {DB_PATH}"}
    con = sqlite3.connect(str(DB_PATH))
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(schedule)").fetchall()]
    finally:
        con.close()
    needed = {"spread_line", "total_line"}
    missing = sorted(needed - set(cols))
    return {
        "name": "schedule_caches_vegas",
        "pass": not missing,
        "detail": "OK" if not missing else f"schedule table missing columns: {missing}; runtime relies on nfl_data_py only",
    }


# ---------------------------------------------------------------------------
# Check 3: nfl_data_py importable
# ---------------------------------------------------------------------------

def check_nfl_data_py_available() -> dict:
    spec = importlib_util.find_spec("nfl_data_py")
    return {
        "name": "nfl_data_py_available",
        "pass": spec is not None,
        "detail": "OK" if spec is not None else "nfl_data_py not installed; Vegas runtime fetch will silently fall back",
    }


# ---------------------------------------------------------------------------
# Check 4: silent fallback patched
# ---------------------------------------------------------------------------

def check_silent_fallback_patched() -> dict:
    text = FE_PATH.read_text()
    # Look for the bare `except Exception: pass` pattern in the Vegas function.
    m = re.search(
        r"def _create_vegas_game_script_features\(self.*?def ",
        text,
        re.DOTALL,
    )
    if not m:
        return {"name": "silent_fallback_patched", "pass": False, "detail": "vegas function not found"}
    body = m.group(0)
    has_silent_pass = bool(re.search(r"except Exception:\s*\n\s*pass\b", body))
    has_warning_log = "Vegas-line load from nfl_data_py failed" in body
    if has_silent_pass and not has_warning_log:
        return {"name": "silent_fallback_patched", "pass": False, "detail": "bare `except Exception: pass` still present"}
    if has_warning_log:
        return {"name": "silent_fallback_patched", "pass": True, "detail": "OK (warning log on fallback)"}
    return {"name": "silent_fallback_patched", "pass": False, "detail": "uncertain — manual review required"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of formatted text.")
    args = parser.parse_args()

    results = [
        check_causal_features_declared(),
        check_schedule_has_vegas_columns(),
        check_nfl_data_py_available(),
        check_silent_fallback_patched(),
    ]
    overall = all(r["pass"] for r in results)

    if args.json:
        print(json.dumps({"overall_pass": overall, "checks": results}, indent=2))
    else:
        print("Vegas-feature readiness check")
        print("=" * 60)
        for r in results:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"  [{mark}] {r['name']}: {r['detail']}")
        print()
        print(f"Overall: {'READY' if overall else 'NOT READY — see docs/PHASE_1_VEGAS_FINDINGS.md'}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())

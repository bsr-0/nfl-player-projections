#!/usr/bin/env python3
"""Manual, one-command refresh of everything the Pages site serves.

Not a scheduler and not CI -- AUDIT_REPORT.md #11 ("no orchestrated
refresh") is real, but the pipeline depends on data/nfl_data.db, a 573MB
gitignored local file, so a cron/CI job would have nowhere to run against.
This just replaces "remember to run 5 scripts in the right order, and
remember docs/data/players_*.json needs a manual copy" with one command
you still run and review by hand.

Chains, in order:
    1. src.data.auto_refresh    -- rosters/weekly stats/schedule/quality gates
    2. scripts/backfill_injuries.py  -- current-season injury reports
    3. scripts/backfill_adp.py       -- current-season market ADP
    4. scripts/generate_draft_data.py -- writes data/players_{POS}.json
    5. copy data/players_{POS}.json -> docs/data/ (generate_draft_data.py
       does not do this itself -- see the ADP-fix session note in git log)
    6. scripts/generate_weekly_data.py -- writes docs/data/weekly_*.json
       directly, auto-detects season_prorated vs weekly_model

Does NOT commit or push anything. Review `git status`/`git diff` on
docs/data and data/players_*.json yourself before committing.

Usage:
    python scripts/refresh_site_data.py
    python scripts/refresh_site_data.py --season 2026
    python scripts/refresh_site_data.py --skip auto_refresh,adp
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"
POSITIONS = ["QB", "RB", "WR", "TE"]

STEPS = ["auto_refresh", "injuries", "adp", "draft_data", "sync_docs", "weekly_data"]


def _run(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\nFAILED at step: {label} (exit {result.returncode})")
        sys.exit(result.returncode)


def sync_docs_data() -> None:
    print(f"\n{'=' * 60}\nsync_docs: data/players_*.json -> docs/data/\n{'=' * 60}")
    for pos in POSITIONS:
        src = DATA_DIR / f"players_{pos}.json"
        dst = DOCS_DATA_DIR / f"players_{pos}.json"
        if not src.exists():
            print(f"  skip {src.name}: not found (did generate_draft_data.py run?)")
            continue
        shutil.copy(src, dst)
        print(f"  copied {src.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None,
                    help="Season for injuries/ADP backfill (default: current NFL season)")
    ap.add_argument("--skip", default="",
                    help=f"Comma-separated steps to skip, from: {','.join(STEPS)}")
    args = ap.parse_args()

    from src.utils.nfl_calendar import get_current_nfl_season
    season = args.season or get_current_nfl_season()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(STEPS)
    if unknown:
        ap.error(f"--skip: unknown step(s) {sorted(unknown)}; valid: {STEPS}")

    print(f"Refreshing site data for season {season}. Skipping: {sorted(skip) or 'nothing'}")

    py = sys.executable

    if "auto_refresh" not in skip:
        _run("1/6 auto_refresh (rosters/weekly stats/schedule/quality gates)",
             [py, "-m", "src.data.auto_refresh"])
    if "injuries" not in skip:
        _run(f"2/6 backfill_injuries ({season})",
             [py, "scripts/backfill_injuries.py", "--seasons", str(season), str(season)])
    if "adp" not in skip:
        _run(f"3/6 backfill_adp ({season})",
             [py, "scripts/backfill_adp.py", "--seasons", str(season), str(season)])
    if "draft_data" not in skip:
        _run("4/6 generate_draft_data", [py, "scripts/generate_draft_data.py"])
    if "sync_docs" not in skip:
        sync_docs_data()
    if "weekly_data" not in skip:
        _run("6/6 generate_weekly_data", [py, "scripts/generate_weekly_data.py"])

    print(f"\n{'=' * 60}\nDone. Review before committing:\n"
          f"  git status docs/data data/players_*.json\n"
          f"  git diff --stat docs/data data/players_*.json\n{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

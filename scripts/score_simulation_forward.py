#!/usr/bin/env python3
"""Score the published simulation on forward weeks (docs/SIMULATION_FORWARD_CHECK.md).

    python scripts/score_simulation_forward.py --weeks 5 8     # display gate (after week 8 is final)
    python scripts/score_simulation_forward.py --weeks 5 13    # correlation verdict + gate re-check

Run only at the checkpoints the pre-registration names. Writes
data/experiments/simulation_forward_2026/report_wk{FIRST}-{LAST}.json and the
per-row scores beside it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DB_PATH  # noqa: E402
from src.evaluation.simulation_forward import correlation_verdict, display_gate, score_weeks  # noqa: E402

OUT = ROOT / "data" / "experiments" / "simulation_forward_2026"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", type=int, nargs=2, required=True, metavar=("FIRST", "LAST"))
    args = ap.parse_args()
    weeks = list(range(args.weeks[0], args.weeks[1] + 1))
    con = sqlite3.connect(str(DB_PATH))
    scores, report = score_weeks(args.season, weeks, con=con, artifacts_root=ROOT / "data" / "models" / "simulation")
    if scores.empty:
        print("nothing to score", report)
        return 1
    result = {"season": args.season, "weeks": weeks, "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              **report, "correlation_verdict": correlation_verdict(scores),
              "display_gate": {arm: display_gate(scores, arm) for arm in ("role_factor", "independent")},
              "summary": scores.groupby(["population", "kind", "arm"])[["crps", "cov50", "cov80"]].mean().round(4)
              .reset_index().to_dict("records")}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"wk{weeks[0]}-{weeks[-1]}"
    (OUT / f"report_{tag}.json").write_text(json.dumps(result, indent=2))
    scores.to_csv(OUT / f"scores_{tag}.csv", index=False)
    print(json.dumps({k: result[k] for k in ("scored_games", "not_published", "other_centre",
                                             "correlation_verdict", "display_gate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

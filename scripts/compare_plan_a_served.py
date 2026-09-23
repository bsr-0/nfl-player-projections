#!/usr/bin/env python3
"""Compare Plan A holdout predictions with a served prediction export.

The join is strict on player/week/team and refuses to compare aggregates from
different populations. The current repository has Plan A metadata sidecars
but no saved ``team_share_*_*.joblib`` artifacts, so this reports that gap
explicitly until a serving export is supplied.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

KEYS = ["player_id", "season", "week", "team"]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("holdout_csv", type=Path)
    ap.add_argument("--served-csv", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    report = {"holdout_csv": str(args.holdout_csv), "served_csv": str(args.served_csv) if args.served_csv else None}
    if not args.served_csv:
        report["status"] = "unavailable"
        report["reason"] = "No Plan A served prediction export was supplied; metadata JSON is not row-level and cannot support an exact-row comparison."
    else:
        left, right = pd.read_csv(args.holdout_csv), pd.read_csv(args.served_csv)
        required = set(KEYS + ["predicted_share"])
        for name, frame in (("holdout", left), ("served", right)):
            missing = required - set(frame.columns)
            if missing: raise SystemExit(f"{name} CSV missing columns: {sorted(missing)}")
        joined = left.merge(right[KEYS + ["predicted_share"]], on=KEYS, suffixes=("_holdout", "_served"), validate="one_to_one")
        report["status"] = "ok"; report["matched_rows"] = len(joined)
        report["holdout_mae"] = float((joined.actual_share - joined.predicted_share_holdout).abs().mean())
        report["served_mae"] = float((joined.actual_share - joined.predicted_share_served).abs().mean())
        report["served_minus_holdout_mae"] = report["served_mae"] - report["holdout_mae"]
    text = json.dumps(report, indent=2) + "\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "ok" else 2

if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Compare stale served metrics with a completed corrected walk-forward CSV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("validation_csv", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    frame = pd.read_csv(args.validation_csv)
    required = {"position", "mae"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"validation CSV missing columns: {sorted(missing)}")
    metadata_path = ROOT / "data/models/model_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = frame.groupby("position", as_index=False).agg(
        validation_mae=("mae", "mean"), validation_rows=("mae", "size"))
    stale = {}
    for pos, values in metadata.get("test_metrics", {}).items():
        fp = values.get("FP (owner objective)") or values.get("FP", {})
        if "mae" in fp:
            stale[pos] = fp["mae"]
    summary["served_stale_mae"] = summary["position"].map(stale)
    summary["delta_corrected_minus_stale"] = summary["validation_mae"] - summary["served_stale_mae"]
    report = {"validation_source": str(args.validation_csv), "served_metadata_source": str(metadata_path),
              "rows": summary.to_dict("records")}
    text = json.dumps(report, indent=2, default=str) + "\n"
    if args.output: args.output.write_text(text)
    print(text, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

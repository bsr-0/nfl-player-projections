#!/usr/bin/env python3
"""Read-only audit of served weekly artifacts and their Vegas lineage."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "data" / "models"
VEGAS_FIX_DATE = datetime(2026, 9, 19, tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, help="Optional JSON report path")
    args = ap.parse_args()
    metadata = {}
    p = MODEL_DIR / "model_metadata.json"
    if p.exists():
        metadata = json.loads(p.read_text())
    rows = []
    for path in sorted(MODEL_DIR.glob("model_*_1w.joblib")):
        stat = path.stat()
        payload = joblib.load(path)
        features = payload.get("feature_names", []) if isinstance(payload, dict) else []
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "position": payload.get("position") if isinstance(payload, dict) else None,
            "feature_count": len(features),
            "feature_version": metadata.get("feature_version"),
            "training_date": metadata.get("training_date"),
            "file_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "trained_before_vegas_fix": stat.st_mtime < VEGAS_FIX_DATE.timestamp(),
            "vegas_features": [f for f in features if f in {"implied_team_total", "is_favorite", "win_probability", "spread"}],
        })
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vegas_fix_date": VEGAS_FIX_DATE.isoformat(),
        "vegas_formula_source": "src/data/external_data.py:763-780",
        "metadata_source": str(p.relative_to(ROOT)) if p.exists() else None,
        "models": rows,
        "read_only": True,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

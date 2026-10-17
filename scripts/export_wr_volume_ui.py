#!/usr/bin/env python3
"""Export the WR volume shadow arm (C1') to the site: docs/data/wr_volume_{season}_wk{N}.json.

Reads only the shadow test's hashed, pre-deadline weekly forecasts (the weeks `wr_volume_shadow.py check`
calls valid) and writes {player_id: c1} for WRs. Reads no outcome, scores nothing, and touches no pinned
file. The UI shows it as an experimental column beside the published projection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts import wr_volume_shadow as shadow

DOCS_DATA = ROOT / "docs/data"


def export(valid: dict[int, str], out_dir: Path = DOCS_DATA) -> list[Path]:
    written = []
    for week, path in sorted(valid.items()):
        df = pd.read_csv(path, dtype={"player_id": str})
        df = df[df.position == "WR"].dropna(subset=["c1"])
        target = out_dir / f"wr_volume_{shadow.SEASON}_wk{week}.json"
        target.write_text(json.dumps({p: round(float(v), 2) for p, v in zip(df.player_id, df.c1)}) + "\n")
        written.append(target)
    return written


def main() -> None:
    status = shadow.check()
    valid = {int(w): str(shadow.OUT / f"week_{int(w):02d}/forecasts.csv") for w in status["valid_weeks"]}
    for p in export(valid):
        print("wrote", p.relative_to(ROOT))
    if not valid:
        print("no valid shadow weeks yet; nothing exported")


if __name__ == "__main__":
    main()

"""Emit weekly-model projections for docs/weekly.html.

Source is the weekly ensemble (NFLPredictor -> EnsemblePredictor), the same
path validated against real 2025 outcomes on 2026-08-31. NOT
generate_weekly_projections.py, which multiplies a season projection by learned
matchup multipliers and reads a board.json that no longer exists.

Writes docs/data/weekly_{POS}.json plus docs/data/weekly_meta.json. The meta
file carries the target week, whether a schedule was available, and the
measured bias per position -- the page shows that caveat rather than presenting
point estimates as unbiased.

Usage:
    python scripts/generate_weekly_data.py
    python scripts/generate_weekly_data.py --as-of 2025 10   # a past week
"""
import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

OUT_DIR = Path("docs/data")

# Serving-path bias measured on 2025 weeks 6/10/14 against a model trained
# 2018-2024 only (GAPS.md 2026-08-31). Published so the page can state what it
# is known to get wrong instead of implying calibrated point estimates.
MEASURED = {
    "QB": {"mae": 7.09, "bias": -1.30},
    "RB": {"mae": 4.43, "bias": -1.62},
    "WR": {"mae": 4.03, "bias": -1.45},
    "TE": {"mae": 3.00, "bias": -1.22},
}

KEEP = [
    "player_id", "name", "position", "team", "opponent", "home_away",
    "predicted_points", "prediction_std",
    "prediction_ci80_lower", "prediction_ci80_upper",
    "prediction_ci95_lower", "prediction_ci95_upper",
    "is_rookie", "injury_score",
]


def _clean(df: pd.DataFrame) -> list:
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float).round(2)
    return json.loads(out.to_json(orient="records"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", nargs=2, type=int, metavar=("SEASON", "WEEK"),
                    default=None)
    ap.add_argument("--top-n", type=int, default=300)
    args = ap.parse_args()

    from src.predict import NFLPredictor, get_prediction_target_week

    p = NFLPredictor()
    if not p.initialize():
        print("no trained models; run `python -m src.models.train` first")
        return 1

    as_of = tuple(args.as_of) if args.as_of else None
    season, week = as_of if as_of else get_prediction_target_week()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for pos in ("QB", "RB", "WR", "TE"):
        df = p.predict(n_weeks=1, position=pos, top_n=args.top_n, as_of=as_of)
        if df.empty:
            print(f"  {pos}: no rows")
            continue
        df = df.sort_values("predicted_points", ascending=False)
        rows = _clean(df)
        (OUT_DIR / f"weekly_{pos}.json").write_text(json.dumps(rows))
        counts[pos] = len(rows)
        pts = pd.to_numeric(df["predicted_points"], errors="coerce")
        print(f"  {pos}: {len(rows)} players, "
              f"median {pts.median():.1f}, max {pts.max():.1f}")

    # Is this a cold-start week? Week 1 means no current-season form exists:
    # veterans carry last season's final games, rookies run on draft capital.
    cold_start = int(week) <= 1
    meta = {
        "season": int(season),
        "week": int(week),
        "cold_start": cold_start,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "measured": MEASURED,
        "model": "weekly ensemble (1w horizon), log1p+smearing calibration",
    }
    (OUT_DIR / "weekly_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {sum(counts.values())} rows for {season} week {week}"
          f"{' (cold start)' if cold_start else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Phase 4C.1 — Fit position-specific rookie-year PPG priors from
historical draft classes.

For each rookie (player's first season in player_weekly_stats) whose
draft class is in the historical fit window, compute their rookie-
year fantasy PPG and bucket by (position, round_bucket).  The median
within each bucket becomes the prior for future rookies in that
bucket.  Undrafted free agents form their own bucket per position.

Output: ``data/rookie_priors.json`` with a 2-level dict:

    {
      "QB": {"rd1": 14.2, "rd2_3": 9.5, "rd4_7": 6.1, "UDFA": 4.0},
      "RB": {"rd1": 10.1, ...},
      "WR": {...},
      "TE": {...},
      "_metadata": {"fit_window": [2006, 2023], "n_rookies": 1234, ...}
    }

The feature-engineering consumer loads this JSON at runtime to fill
``prev_season_ppg`` for rookies (rows where the player has no prior
season and the NaN would otherwise be defaulted to 0).

Usage:
    python scripts/compute_rookie_priors.py
    python scripts/compute_rookie_priors.py --fit-until 2022
    python scripts/compute_rookie_priors.py --min-games 4
    python scripts/compute_rookie_priors.py --output /tmp/priors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


def _round_bucket(round_num: Optional[int]) -> str:
    if round_num is None or pd.isna(round_num):
        return "UDFA"
    r = int(round_num)
    if r == 1:
        return "rd1"
    if r in (2, 3):
        return "rd2_3"
    if r in (4, 5, 6, 7):
        return "rd4_7"
    return "UDFA"  # (no post-merger rounds > 7 but be safe)


def _rookie_year_ppg(
    db_path: Path, fit_until: int, min_games: int
) -> pd.DataFrame:
    """Return a DataFrame with one row per rookie: player_id, position,
    rookie_season, draft_round (or NaN for UDFA), ppg, games."""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        # Determine each player's first season.  Only keep players
        # whose first season is in the fit window.
        first_season = pd.read_sql(
            "SELECT player_id, MIN(season) AS rookie_season "
            "  FROM player_weekly_stats GROUP BY player_id",
            conn,
        )
        first_season = first_season[
            first_season["rookie_season"] <= fit_until
        ].copy()

        # Join to positions
        players = pd.read_sql(
            "SELECT player_id, position FROM players", conn
        )
        rookies = first_season.merge(players, on="player_id", how="left")
        rookies = rookies[rookies["position"].isin(FANTASY_POSITIONS)].copy()

        # Rookie-year stats
        stats = pd.read_sql(
            "SELECT player_id, season, fantasy_points FROM player_weekly_stats",
            conn,
        )
        stats = stats.merge(
            rookies[["player_id", "rookie_season"]],
            left_on=["player_id", "season"],
            right_on=["player_id", "rookie_season"],
        )
        per_player = (
            stats.groupby("player_id")["fantasy_points"]
                 .agg(["sum", "count"])
                 .reset_index()
                 .rename(columns={"sum": "fp_total", "count": "games"})
        )
        rookies = rookies.merge(per_player, on="player_id", how="left")
        rookies["games"] = rookies["games"].fillna(0).astype(int)
        rookies["ppg"] = rookies["fp_total"] / rookies["games"].replace(0, pd.NA)
        rookies = rookies[rookies["games"] >= min_games].copy()

        # Draft round from draft_picks (NULL → UDFA)
        drafts = pd.read_sql(
            "SELECT player_id, draft_season, draft_round "
            "  FROM draft_picks WHERE player_id IS NOT NULL",
            conn,
        )
        # Guard: keep only the draft row that matches this player's
        # first season (should be unique; sanity check anyway).
        rookies = rookies.merge(
            drafts,
            left_on=["player_id", "rookie_season"],
            right_on=["player_id", "draft_season"],
            how="left",
        )
    return rookies


def _fit_priors(rookies: pd.DataFrame) -> Dict:
    priors: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    rookies["round_bucket"] = rookies["draft_round"].map(_round_bucket)
    for pos in FANTASY_POSITIONS:
        sub = rookies[rookies["position"] == pos]
        priors[pos] = {}
        counts[pos] = {}
        for bucket in ["rd1", "rd2_3", "rd4_7", "UDFA"]:
            cell = sub[sub["round_bucket"] == bucket]
            if len(cell) >= 5:
                priors[pos][bucket] = round(float(cell["ppg"].median()), 2)
                counts[pos][bucket] = int(len(cell))
            else:
                # Too few observations — fallback to position-wide median
                priors[pos][bucket] = round(float(sub["ppg"].median()), 2)
                counts[pos][bucket] = int(len(cell))
    return priors, counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fit-until", type=int, default=2023,
                    help="Inclusive upper bound for rookie season used in the fit. "
                         "Default 2023 (so 2024 and 2025 rookies are held out).")
    ap.add_argument("--min-games", type=int, default=4,
                    help="Minimum rookie-year games played to include in the fit.")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "data" / "rookie_priors.json")
    ap.add_argument("--db-path", type=Path,
                    default=PROJECT_ROOT / "data" / "nfl_data.db")
    args = ap.parse_args()

    rookies = _rookie_year_ppg(args.db_path, args.fit_until, args.min_games)
    print(f"Rookies included: {len(rookies):,}")
    print(f"  by position:")
    for pos in FANTASY_POSITIONS:
        n = (rookies["position"] == pos).sum()
        print(f"    {pos}: {n}")
    priors, counts = _fit_priors(rookies)

    print()
    print(f"{'position':<10} {'rd1':>12} {'rd2_3':>12} {'rd4_7':>12} {'UDFA':>12}")
    for pos in FANTASY_POSITIONS:
        vals = priors[pos]
        cs = counts[pos]
        print(f"  {pos:<8} "
              f"{vals['rd1']:>7.2f} (n={cs['rd1']:<3})  "
              f"{vals['rd2_3']:>5.2f} (n={cs['rd2_3']:<3})  "
              f"{vals['rd4_7']:>5.2f} (n={cs['rd4_7']:<3})  "
              f"{vals['UDFA']:>5.2f} (n={cs['UDFA']:<3})")

    out = {
        **priors,
        "_metadata": {
            "fit_window": [int(rookies["rookie_season"].min()), args.fit_until],
            "n_rookies": int(len(rookies)),
            "min_games": args.min_games,
            "sample_sizes": counts,
            "scoring": "PPR",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

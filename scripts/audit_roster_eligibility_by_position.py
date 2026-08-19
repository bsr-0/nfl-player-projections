#!/usr/bin/env python
"""Cross-position check: does the roster-eligibility bug generalise beyond QB?

The QB audit found that only 1 of 644 pre-debut synthetic weeks belonged to
a listed starter, and 45.7% of manufactured points came from players who
were on IR, on a practice squad, declared inactive, or not in the league.
`possible_weeks_for_player` now filters synthetic candidates to weeks the
player was on an active roster (GAPS.md).

This is a VALIDATION CHECKPOINT, not a repeat of the QB investigation. It
answers one question -- how much of each position's synthetic population was
invalid -- and deliberately does no model fitting. A position is only worth
a deeper look if the numbers here are alarming.

It calls the production filter with require_active_roster True and False and
diffs the result, so what is measured is exactly what ships.

Manufactured points are ESTIMATED, not measured: pricing them exactly would
mean constructing every synthetic row and running the model, which is the
refit this checkpoint exists to avoid. The proxy is the player's own
realised PPG that season (falling back to prior season, then position
median). The QB row can be checked against the measured 51% to see how well
the proxy tracks.

Usage:
    python scripts/audit_roster_eligibility_by_position.py
"""
import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.season_projection import (
    possible_weeks_for_player, REGULAR_SEASON_MAX_WEEK,
)

# weekly_rosters status in an excluded week -> why it was excluded.
REASON = {
    "INA": "inactive", "RES": "IR / reserve", "DEV": "practice squad",
    "CUT": "waived", "RET": "retired", "EXE": "exempt", "E01": "exempt",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", nargs="+", default=["QB", "RB", "WR", "TE"])
    ap.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/roster_eligibility_by_position.csv"))
    args = ap.parse_args()

    from config.settings import DB_PATH
    from src.utils.database import DatabaseManager
    db = DatabaseManager()
    conn = sqlite3.connect(str(DB_PATH))
    lo, hi = min(args.seasons), max(args.seasons)

    rosters = pd.read_sql(
        f"""SELECT player_id, season, week, status FROM weekly_rosters
            WHERE game_type='REG' AND season BETWEEN {lo} AND {hi}""", conn)
    rosters["week"] = rosters.week.astype(int)
    status_by = {(r.player_id, r.season, r.week): r.status
                 for r in rosters.drop_duplicates(
                     ["player_id", "season", "week"]).itertuples()}

    # First roster week of the season separates "signed later" from
    # "off the roster this week".
    first_roster = rosters.groupby(["player_id", "season"])["week"].min().to_dict()

    rows = []
    for position in args.positions:
        hist = db.get_all_players_for_training(position=position)
        hist = hist[hist.season.isin(args.seasons)]
        hist = hist[hist.week <= REGULAR_SEASON_MAX_WEEK]

        ppg_season = hist.groupby(["player_id", "season"])["fantasy_points"].mean()
        pos_median = float(hist["fantasy_points"].median())

        for (pid, season), g in hist.groupby(["player_id", "season"]):
            g = g.sort_values("week")
            rtbw = {int(w): sub["team"].iloc[0]
                    for w, sub in g.groupby(g["week"].astype(int))}
            if not rtbw:
                continue
            kept, _ = possible_weeks_for_player(
                db, rtbw, season, player_id=pid, require_active_roster=True)
            allw, _ = possible_weeks_for_player(
                db, rtbw, season, player_id=pid, require_active_roster=False)

            candidates = [w for w in allw if w not in rtbw]
            kept_syn = {w for w in kept if w not in rtbw}
            removed = [w for w in candidates if w not in kept_syn]
            if not candidates:
                continue

            ppg = ppg_season.get((pid, season), np.nan)
            if not np.isfinite(ppg):
                ppg = pos_median

            for w in removed:
                st = status_by.get((pid, season, w))
                if st is None:
                    fr = first_roster.get((pid, season))
                    reason = ("pre-acquisition" if fr is None or w < fr
                              else "not rostered")
                else:
                    reason = REASON.get(st, "other")
                rows.append({"position": position, "season": season,
                             "player_id": pid, "week": w, "reason": reason,
                             "est_points": ppg})
            rows.append({"position": position, "season": season,
                         "player_id": pid, "week": -1, "reason": "_candidates",
                         "est_points": ppg * len(candidates),
                         "n_candidates": len(candidates)})

    d = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.output, index=False)

    print("\n" + "=" * 86)
    print("ROSTER-ELIGIBILITY IMPACT BY POSITION  (weeks exact; points ESTIMATED)")
    print("=" * 86)
    summary = []
    for position in args.positions:
        p = d[d.position == position]
        cand = p[p.reason == "_candidates"]
        rem = p[p.reason != "_candidates"]
        n_cand = int(cand.n_candidates.sum())
        pts_cand = cand.est_points.sum()
        summary.append({
            "position": position,
            "candidate_weeks": n_cand,
            "removed": len(rem),
            "%_weeks_removed": 100 * len(rem) / n_cand if n_cand else np.nan,
            "%_points_removed": 100 * rem.est_points.sum() / pts_cand if pts_cand else np.nan,
            "players_affected": rem.player_id.nunique(),
        })
    s = pd.DataFrame(summary).set_index("position")
    print(s.round(1).to_string())

    print("\n\nREMOVED WEEKS BY REASON (% of that position's removed weeks)\n")
    rem = d[d.reason != "_candidates"]
    pv = pd.crosstab(rem.position, rem.reason, normalize="index") * 100
    pv = pv.reindex([p for p in args.positions if p in pv.index])
    print(pv.round(1).to_string())
    print("\ncounts:")
    print(pd.crosstab(rem.position, rem.reason).reindex(
        [p for p in args.positions if p in pv.index]).to_string())
    print(f"\nWrote {len(d)} rows -> {args.output}")
    conn.close()


if __name__ == "__main__":
    main()

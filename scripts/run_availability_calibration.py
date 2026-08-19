#!/usr/bin/env python
"""Calibration of the availability estimators themselves, independent of
any model: estimated P(plays) vs. realized play/no-play, by how many games
have elapsed.

This is the interpretability check on shrinkage behaviour -- does the
estimator progressively incorporate current-season information without
becoming over-reactive after 1-2 games? It needs no predictions, so it runs
in seconds and can't be confounded by model quality.

At target week W the estimator forecasts P(player appears in week W). The
realized outcome is simply whether a real row exists for week W. Averaged
over player-weeks grouped by n_elapsed (team games already played), mean
estimate vs. mean realized is a proper reliability curve.

Usage:
    python scripts/run_availability_calibration.py --positions QB
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.models.single_week_ppr.availability import (
    AVAILABILITY_ESTIMATORS, PlayerAvailabilityHistory, SHRINKAGE_K,
)
from src.models.single_week_ppr.evaluate import DEFAULT_VALIDATION_SEASONS


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", nargs="+", default=None)
    ap.add_argument("--seasons", nargs="+", type=int, default=list(DEFAULT_VALIDATION_SEASONS))
    ap.add_argument("--output", type=Path,
                    default=Path("data/experiments/availability_calibration.csv"))
    args = ap.parse_args()

    from config.settings import POSITIONS
    from src.models.single_week_ppr.season_projection import possible_weeks_for_team
    from src.utils.database import DatabaseManager

    positions = list(args.positions) if args.positions else list(POSITIONS)
    db = DatabaseManager()
    rows = []

    for position in positions:
        hist_df = db.get_all_players_for_training(position=position)
        hist_df = hist_df[hist_df["week"] <= 18]
        team_weeks = {}
        for t in hist_df["team"].dropna().unique():
            for s in hist_df["season"].dropna().unique():
                wk = possible_weeks_for_team(db, t, int(s))
                if wk:
                    team_weeks[(t, int(s))] = wk
        hist = PlayerAvailabilityHistory(hist_df, team_weeks)

        for season in args.seasons:
            season_rows = hist_df[hist_df["season"] == season]
            print(f"  {position}/{season}: {season_rows['player_id'].nunique()} players")
            for player_id, g in season_rows.groupby("player_id"):
                team = g.sort_values("week")["team"].iloc[0]
                weeks = hist.team_regular_weeks(team, season)
                if len(weeks) == 0:
                    continue
                played = set(hist.weeks_played(player_id, season).tolist())
                for w in weeks:
                    n_elapsed = int((weeks < w).sum())
                    rec = {"position": position, "season": season, "player": player_id,
                           "week": int(w), "n_elapsed": n_elapsed,
                           "actually_played": int(w in played)}
                    for name, est in AVAILABILITY_ESTIMATORS.items():
                        rec[name] = est(hist, player_id, season, team, int(w))
                    rows.append(rec)

    if not rows:
        print("No results.")
        return
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n{len(df)} player-weeks -> {args.output}")

    names = list(AVAILABILITY_ESTIMATORS)
    pd.set_option("display.width", 170)

    print("\n" + "=" * 96)
    print("CALIBRATION: mean estimated P(plays) vs realized, by games elapsed")
    print("(want estimate ~ realized; over-reactive estimators swing early)")
    print("=" * 96)
    df["elapsed_bin"] = pd.cut(df.n_elapsed, [-1, 1, 2, 4, 8, 12, 18],
                               labels=["0-1", "2", "3-4", "5-8", "9-12", "13+"])
    agg = df.groupby("elapsed_bin", observed=True).agg(
        n=("actually_played", "size"), realized=("actually_played", "mean"),
        **{k: (k, "mean") for k in names}).round(3)
    print(agg.to_string())

    print("\n" + "=" * 96)
    print("CALIBRATION ERROR (mean estimate - realized; 0 = perfectly calibrated)")
    print("=" * 96)
    err = pd.DataFrame({k: agg[k] - agg["realized"] for k in names}).round(3)
    err["n"] = agg["n"]
    print(err.to_string())
    print("\nMean |calibration error| across bins:")
    for k in names:
        print(f"  {k:22s} {err[k].abs().mean():.4f}")

    print("\n" + "=" * 96)
    print(f"SHRINKAGE WEIGHT SCHEDULE (k={SHRINKAGE_K:g}): weight on current-season evidence")
    print("=" * 96)
    for n in (1, 2, 4, 8, 12, 16):
        print(f"  n_elapsed={n:2d} -> {n / (n + SHRINKAGE_K):.0%} current / "
              f"{1 - n / (n + SHRINKAGE_K):.0%} prior")

    print("\n" + "=" * 96)
    print("REACTIVITY: spread of the estimate at low evidence (n_elapsed <= 2)")
    print("(a very wide spread this early means the estimator is over-reacting)")
    print("=" * 96)
    early = df[df.n_elapsed <= 2]
    print(early[names].std().round(3).to_string())


if __name__ == "__main__":
    main()

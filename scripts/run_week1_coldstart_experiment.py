#!/usr/bin/env python3
"""How well does the number the board publishes for week 1 predict week 1?

Before kickoff the site publishes `season total / 17` per week -- the step 8
season model divided by a season (`generate_weekly_data.py`, mode
`season_prorated`). Every existing measurement of that model is a SEASON
number: rookie season-total MAE 39.81, cold-start games-played MAE 3.3-4.8.
Nothing measures the thing a week-1 lineup decision actually reads.

This runs the production path backwards over past seasons and scores it
against real week-1 PPR, split by whether the player had any NFL history at
all, because that split is the whole question -- a cold-start player's pace is
almost entirely the shrinkage target that PR #96 changed.

Arms:
    step8_pace     the published number: step 8 season total / 17
    position_mean  the position's mean week-1 points, fitted leave-one-season-
                   out. The floor any model has to clear to be worth running.
    prior_ppg      the player's own PPG last season. Undefined for cold-start
                   rows by construction, which is exactly why they are hard;
                   they fall back to position_mean.

Leakage: step 8 is refit per target season on pairs strictly before it, with
availability fit `before_season=S`. position_mean excludes the target season.
prior_ppg reads season S-1, which is known before week 1 is played.

POPULATION CAVEAT, stated because it flatters every arm: only players who
recorded a week-1 row are scored. A rookie drafted and inactive in week 1 is
not here, so this measures accuracy given that he played, not the harder
question of whether he would.

Usage:
    python scripts/run_week1_coldstart_experiment.py
    python scripts/run_week1_coldstart_experiment.py --seasons 2023 2025
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, DB_PATH

# Step 8's own training window, matching generate_draft_data.py.
TRAIN_FROM = 2019
OUT_DIR = DATA_DIR / "backtest_results"


def step8_pace(season: int) -> pd.DataFrame:
    """Refit step 8 as of `season` and return its published per-week number."""
    from src.utils.database import DatabaseManager
    from src.models.preseason_features import build_multiyear_season_pairs
    from src.models.single_week_ppr.season_availability import load_player_seasons
    from src.models.season_step8 import Step8SeasonModel, possible_games_for_players

    db = DatabaseManager()
    panel = load_player_seasons()
    pairs = build_multiyear_season_pairs(db, list(range(TRAIN_FROM, season)),
                                         inference_season=season)
    train = pairs[pairs["target_season"] < season]
    infer = pairs[pairs["target_season"] == season].copy()
    if train.empty or infer.empty:
        return pd.DataFrame()

    model = Step8SeasonModel().fit(train, panel, before_season=season)
    preds = model.predict(infer, possible_games=possible_games_for_players(
        infer, season))
    cold = infer.get("is_cold_start")
    return pd.DataFrame({
        "player_id": infer["player_id"].values,
        "position": infer["position"].values,
        "cold_start": (cold.fillna(0).astype(int).values if cold is not None
                       else np.zeros(len(infer), dtype=int)),
        # The published week number is the season total over a season.
        "step8_pace": np.asarray(preds) / 17.0,
        "train_seasons": season - TRAIN_FROM,
    })


def week1_actuals(season: int) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql(
            "SELECT player_id, fantasy_points AS actual FROM player_weekly_stats "
            "WHERE season = ? AND week = 1 AND fantasy_points IS NOT NULL",
            con, params=[int(season)])
    finally:
        con.close()


def prior_ppg(season: int) -> pd.DataFrame:
    con = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql(
            "SELECT player_id, AVG(fantasy_points) AS prior_ppg "
            "FROM player_weekly_stats WHERE season = ? "
            "GROUP BY player_id", con, params=[int(season) - 1])
    finally:
        con.close()


def _score(g: pd.DataFrame, arm: str) -> dict:
    err = g[arm] - g["actual"]
    ss_res = float((err ** 2).sum())
    ss_tot = float(((g["actual"] - g["actual"].mean()) ** 2).sum())
    return {"n": int(len(g)), "mae": round(float(err.abs().mean()), 2),
            "bias": round(float(err.mean()), 2),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
            "r2": round(1 - ss_res / ss_tot, 3) if ss_tot else None}


def evaluate(rows: pd.DataFrame, arms) -> dict:
    out = {}
    for label, g in (("all", rows), ("cold_start", rows[rows.cold_start == 1]),
                     ("veteran", rows[rows.cold_start == 0])):
        if g.empty:
            continue
        out[label] = {arm: _score(g, arm) for arm in arms}
        out[label]["mean_actual"] = round(float(g["actual"].mean()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=[2021, 2025])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    first, last = args.seasons
    frames = []
    for season in range(first, last + 1):
        print(f"fitting step 8 as of {season} ...", flush=True)
        pace = step8_pace(season)
        if pace.empty:
            print(f"  no pairs for {season}; skipped")
            continue
        rows = (pace.merge(week1_actuals(season), on="player_id", how="inner")
                    .merge(prior_ppg(season), on="player_id", how="left"))
        rows["season"] = season
        print(f"  {len(rows)} players with a week-1 row "
              f"({int(rows.cold_start.sum())} cold start)")
        frames.append(rows)

    if not frames:
        print("nothing to score")
        return 1
    rows = pd.concat(frames, ignore_index=True)

    # Leave-one-season-out so the floor is not fitted on the season it scores.
    # With a single season there is nothing to leave out; say so rather than
    # silently scoring against NaN and reporting an empty table.
    single_season = rows["season"].nunique() == 1
    parts = []
    for season, g in rows.groupby("season"):
        other = rows if single_season else rows[rows.season != season]
        means = other.groupby("position")["actual"].mean()
        parts.append(g.assign(position_mean=g["position"].map(means)))
    rows = pd.concat(parts, ignore_index=True).dropna(subset=["position_mean"])
    # A cold-start player has no last season; the floor stands in for it.
    rows["prior_ppg"] = rows["prior_ppg"].fillna(rows["position_mean"])

    arms = ["step8_pace", "position_mean", "prior_ppg"]
    result = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "seasons": [first, last],
        "train_from": TRAIN_FROM,
        "population": "players with a week-1 row in player_weekly_stats",
        "position_mean_basis": ("in-sample (single season)" if single_season
                                else "leave-one-season-out"),
        "pooled": evaluate(rows, arms),
        "by_season": {int(s): evaluate(g, arms)
                      for s, g in rows.groupby("season")},
        "by_position_cold_start": {
            pos: evaluate(g, arms).get("all")
            for pos, g in rows[rows.cold_start == 1].groupby("position")},
    }

    for slice_name, scores in result["pooled"].items():
        print(f"\n{slice_name}  (mean actual {scores['mean_actual']})")
        for arm in arms:
            s = scores[arm]
            print(f"  {arm:<14} n={s['n']:<5} MAE {s['mae']:>5}  "
                  f"RMSE {s['rmse']:>5}  bias {s['bias']:>+6}  R2 {s['r2']:>+7}")

    out = args.out or (OUT_DIR / f"week1_coldstart_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

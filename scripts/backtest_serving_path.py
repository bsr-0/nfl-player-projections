"""Score the SERVING path against outcomes that actually happened.

Everything validated during the 2026-08-29/30 audit measured the TRAINING
pipeline: the fold harness trains and scores inside _prepare_training_data and
never calls predict(). That is how two serving defects survived with a green
suite -- a dead call (TypeError on every prediction) and, before it, team
shares computed on a position-filtered frame that inflated every weekly
projection roughly 8x.

This runs the real production path via `predict(as_of=(season, week))`, which
truncates player history to games completed before the target week, and
compares its output to what those players actually scored.

The number that matters is the comparison against the fold harness. If serving
MAE lands near the training-side MAE (QB 6.04 / RB 4.49 / WR 4.23 / TE 2.87),
the serving path is sound. If it is materially worse, there is a third defect.

Usage:
    python scripts/backtest_serving_path.py --season 2025 --weeks 6 10 14
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.utils.database import DatabaseManager


def actuals_for(db: DatabaseManager, season: int, week: int) -> pd.DataFrame:
    frames = []
    for pos in ("QB", "RB", "WR", "TE"):
        d = db.get_all_players_for_training(position=pos, min_games=1)
        d = d[(d["season"] == season) & (d["week"] == week)]
        if len(d):
            frames.append(d[["player_id", "position", "fantasy_points"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(
        columns={"fantasy_points": "actual"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", type=int, nargs="+", default=[6, 10, 14])
    args = ap.parse_args()

    from src.predict import NFLPredictor

    p = NFLPredictor()
    if not p.initialize():
        print("no trained models")
        return 1
    db = DatabaseManager()

    rows = []
    for week in args.weeks:
        act = actuals_for(db, args.season, week)
        if act.empty:
            print(f"{args.season} wk {week}: no actuals, skipped")
            continue
        for pos in ("QB", "RB", "WR", "TE"):
            try:
                pred = p.predict(n_weeks=1, position=pos, top_n=500,
                                 as_of=(args.season, week))
            except Exception as e:
                print(f"  {pos} wk{week}: FAILED {type(e).__name__}: {e}")
                continue
            if pred.empty:
                continue
            m = pred[["player_id", "predicted_points"]].merge(
                act[act["position"] == pos], on="player_id", how="inner")
            m = m[pd.to_numeric(m["actual"], errors="coerce").notna()]
            if len(m) < 10:
                continue
            err = (pd.to_numeric(m["predicted_points"], errors="coerce")
                   - m["actual"]).abs()
            rows.append(dict(
                season=args.season, week=week, position=pos, n=len(m),
                mae=err.mean(),
                bias=(pd.to_numeric(m["predicted_points"], errors="coerce")
                      - m["actual"]).mean(),
                pred_mean=pd.to_numeric(m["predicted_points"],
                                        errors="coerce").mean(),
                actual_mean=m["actual"].mean(),
            ))
            print(f"  {pos} wk{week}: n={len(m)} MAE={err.mean():.2f} "
                  f"pred_mean={rows[-1]['pred_mean']:.2f} "
                  f"actual_mean={rows[-1]['actual_mean']:.2f}")

    if not rows:
        print("\nnothing scored")
        return 1

    df = pd.DataFrame(rows)
    print("\n\nSERVING-PATH MAE by position (averaged over weeks)")
    agg = df.groupby("position").apply(
        lambda g: pd.Series({
            "n": int(g["n"].sum()),
            "serving_mae": np.average(g["mae"], weights=g["n"]),
            "bias": np.average(g["bias"], weights=g["n"]),
            "pred_mean": np.average(g["pred_mean"], weights=g["n"]),
            "actual_mean": np.average(g["actual_mean"], weights=g["n"]),
        })).round(3)
    print(agg.to_string())

    # Training-side OOF MAE from the 2026-08-30 01:32 retrain, for reference.
    train_mae = {"QB": 6.04, "RB": 4.49, "WR": 4.23, "TE": 2.87}
    print("\nserving vs training-side OOF MAE (fold harness):")
    for pos in agg.index:
        s = float(agg.loc[pos, "serving_mae"])
        t = train_mae.get(pos, float("nan"))
        print(f"  {pos}: serving {s:5.2f}   training {t:5.2f}   "
              f"delta {s - t:+.2f}")

    out = Path("data/experiments") / f"serving_backtest_{args.season}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

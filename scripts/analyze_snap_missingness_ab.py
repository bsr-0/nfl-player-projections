#!/usr/bin/env python
"""Pre-registered analysis of the snap-missingness A/B (GAPS.md 2026-08-19).

Reads the per-row predictions written by experiment_snap_missingness.py and
reports the cuts fixed BEFORE the runs: overall, era, position, and the
decisive known-vs-uncertain split. Comparisons are PAIRED -- both variants
predict the same rows -- so the test is on the per-row change in absolute
error, not on two independent means.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import sqlite3
from scipy import stats

D = Path(sys.argv[1] if len(sys.argv) > 1 else "data/experiments/snap_missingness_v2")
SEASONS = [2016, 2017, 2018, 2024]


def load_known_status() -> pd.DataFrame:
    """Recompute snap_share_pct_roll3_known exactly as the feature does."""
    from config.settings import DB_PATH
    conn = sqlite3.connect(str(DB_PATH))
    db = pd.read_sql(
        "SELECT w.player_id, w.season, w.week, w.snap_count, w.team_snaps "
        "FROM player_weekly_stats w JOIN players p ON p.player_id = w.player_id "
        "WHERE p.position IN ('RB','WR','TE') ORDER BY w.player_id, w.season, w.week",
        conn)
    conn.close()
    share = (pd.to_numeric(db.snap_count, errors="coerce")
             / pd.to_numeric(db.team_snaps, errors="coerce") * 100)
    db["ss"] = share.where(pd.to_numeric(db.team_snaps, errors="coerce") > 0)
    count = lambda x: x.shift(1).rolling(3, min_periods=1).count()
    n_known = db.groupby("player_id")["ss"].transform(count)
    n_avail = pd.Series(1.0, index=db.index).groupby(db.player_id).transform(count)
    db["known"] = (n_known / n_avail).fillna(0)
    return db[["player_id", "season", "week", "known"]]


def main() -> int:
    frames = []
    for variant in ("A", "B"):
        for season in SEASONS:
            d = pd.read_csv(D / f"preds_{variant}_{season}.csv")
            d["variant"] = variant
            frames.append(d)
    preds = pd.concat(frames, ignore_index=True).dropna(subset=["actual", "predicted"])
    preds = preds.merge(load_known_status(), on=["player_id", "season", "week"], how="left")
    preds["err"] = (preds.predicted - preds.actual).abs()

    key = ["player_id", "season", "week", "position", "known"]
    paired = preds.pivot_table(index=key, columns="variant", values="err").reset_index().dropna()
    paired["diff"] = paired.B - paired.A
    uncertain = paired[paired.known < 1]

    print(f"paired rows {len(paired):,} | uncertain {len(uncertain):,}\n")
    print("=== paired change in MAE (B - A), uncertain rows ===")
    for name, sub in [("all", uncertain),
                      ("2016-17", uncertain[uncertain.season < 2018]),
                      ("2018+", uncertain[uncertain.season >= 2018])]:
        boot = [np.mean(np.random.choice(sub["diff"], len(sub))) for _ in range(2000)]
        _, pval = stats.wilcoxon(sub["diff"])
        print(f"  {name:8s} n={len(sub):5d} mean={sub['diff'].mean():+.4f} "
              f"CI=[{np.percentile(boot, 2.5):+.4f},{np.percentile(boot, 97.5):+.4f}] p={pval:.2e}")

    print("\n=== known rows (regression check) ===")
    known = paired[paired.known >= 1]
    print(f"  n={len(known):,} mean={known['diff'].mean():+.4f}")

    print("\n=== per position, uncertain rows ===")
    for pos in ("RB", "WR", "TE"):
        sub = uncertain[uncertain.position == pos]
        print(f"  {pos}: n={len(sub):4d} mean={sub['diff'].mean():+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

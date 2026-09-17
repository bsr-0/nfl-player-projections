"""Does shrinking the weekly model toward the season pace beat the weekly model?

The weekly page serves two different numbers: before kickoff, the Step 8
season total / 17 (a pace); once games exist, the weekly ensemble. The
2025 walk-forward shows the ensemble only edging a flat season average
(RMSE +2.9%, MAE -0.6%), which raises the question this script answers
before anyone plumbs a Step 8 feature into the ensemble:

    blend_k = w(g) * weekly + (1 - w(g)) * pace,   w(g) = g / (g + kappa)

where g is the games the player has played so far this season and kappa is
the only free parameter. kappa -> 0 is the weekly model; kappa -> inf is the
pace. The whole point of a games-played weight is that it is the cheapest
possible consolidation: one scalar, no retrain.

Inputs, both out-of-fold with respect to the season scored:
  * the latest trusted serving-path walk-forward rows
    (data/backtest_results/backtest_<season>_<date>.rows.csv), and
  * the step8 arm of data/experiments/walk_forward_player_predictions.csv,
    a season total projected before the season from prior seasons only.

kappa is chosen out-of-sample: players are split in two by id, kappa is
fit on one half and scored on the other, both ways. The in-sample curve is
printed too, but the headline is the out-of-fold number. A paired bootstrap
over players gives a CI on (blend MAE - weekly MAE); the blend "wins" only if
that interval is entirely below zero, the bar the cold-start experiment set.

Usage:
    python scripts/run_pace_blend_experiment.py [--season 2025] [--rows PATH]
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH  # noqa: E402

GAMES_PER_SEASON = 17
KAPPAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, np.inf]
OUT_DIR = PROJECT_ROOT / "data" / "experiments"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest_results"
WF_PATH = OUT_DIR / "walk_forward_player_predictions.csv"


def latest_rows(season: int) -> Path:
    cands = sorted(BACKTEST_DIR.glob(f"backtest_{season}_*.rows.csv"))
    cands = [c for c in cands if "UNTRUSTED" not in c.name and "PARTIAL" not in c.name]
    if not cands:
        raise SystemExit(f"no trusted rows file for {season}; run "
                         f"`python -m src.evaluation.backtester --season {season}` first")
    return cands[-1]


def games_before(season: int) -> pd.DataFrame:
    """Games each player had played in `season` before each week (from the DB,
    so a week the model skipped still counts as played)."""
    with sqlite3.connect(str(DB_PATH)) as conn:
        g = pd.read_sql(
            "SELECT player_id, week FROM player_weekly_stats "
            "WHERE season = ? AND week <= 18 AND fantasy_points IS NOT NULL",
            conn, params=(int(season),))
    g = g.sort_values(["player_id", "week"])
    g["games_before"] = g.groupby("player_id").cumcount()
    return g


def blend(df: pd.DataFrame, kappa: float) -> np.ndarray:
    g = df["games_before"].to_numpy(dtype=float)
    if kappa == 0.0:
        w = np.ones_like(g)
    elif np.isinf(kappa):
        w = np.zeros_like(g)
    else:
        w = g / (g + kappa)
    return w * df["weekly"].to_numpy() + (1.0 - w) * df["pace"].to_numpy()


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def fold_of(player_id: str) -> int:
    return int(hashlib.md5(player_id.encode()).hexdigest(), 16) % 2


def paired_bootstrap(df: pd.DataFrame, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """CI on MAE(a) - MAE(b), resampling PLAYERS (rows within a player are
    not independent)."""
    rng = np.random.default_rng(seed)
    y = df["fantasy_points"].to_numpy()
    err = np.abs(pred_a - y) - np.abs(pred_b - y)
    per_player = pd.Series(err).groupby(df["player_id"].to_numpy()).agg(["sum", "count"])
    sums, counts = per_player["sum"].to_numpy(), per_player["count"].to_numpy()
    n = len(sums)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(sums[idx].sum() / counts[idx].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_diff": float(err.mean()), "ci95": [float(lo), float(hi)],
            "wins": bool(hi < 0), "n_players": int(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--rows", type=Path, default=None)
    args = ap.parse_args()

    rows_path = args.rows or latest_rows(args.season)
    rows = pd.read_csv(rows_path)
    wf = pd.read_csv(WF_PATH)
    s8 = wf[(wf["arm"] == "step8") & (wf["season"] == args.season)][["player_id", "pred"]]
    s8 = s8.rename(columns={"pred": "step8_total"})

    df = rows.merge(s8, on="player_id", how="inner")
    df = df.merge(games_before(args.season), on=["player_id", "week"], how="left")
    df = df[df["predicted_points"].notna() & df["games_before"].notna()].copy()
    df["weekly"] = df["predicted_points"].astype(float)
    df["pace"] = df["step8_total"].astype(float) / GAMES_PER_SEASON
    df["fold"] = df["player_id"].map(fold_of)
    y = df["fantasy_points"].to_numpy()

    print(f"rows file: {rows_path.name}")
    print(f"scored rows with a step8 projection: {len(df)} of {len(rows)} "
          f"({df.player_id.nunique()} players; step8 covers {len(s8)})")
    print(f"games_before: min {df.games_before.min():.0f}, max {df.games_before.max():.0f}, "
          f"rows at 0: {(df.games_before == 0).sum()}")

    # --- in-sample curve over kappa --------------------------------------
    print("\nkappa   w(g=1)  w(g=4)  w(g=8)      MAE     RMSE")
    curve = {}
    for k in KAPPAS:
        p = blend(df, k)
        curve[k] = {"mae": mae(p, y), "rmse": rmse(p, y)}
        wk = (lambda g: 1.0 if k == 0 else (0.0 if np.isinf(k) else g / (g + k)))
        label = "weekly" if k == 0 else ("pace" if np.isinf(k) else f"{k:g}")
        print(f"{label:>6}  {wk(1):6.2f}  {wk(4):6.2f}  {wk(8):6.2f}   {curve[k]['mae']:7.3f}  {curve[k]['rmse']:7.3f}")

    # --- out-of-fold kappa -----------------------------------------------
    oof = np.full(len(df), np.nan)
    chosen = {}
    for f in (0, 1):
        train, test = df[df.fold != f], df[df.fold == f]
        best = min((k for k in KAPPAS), key=lambda k: mae(blend(train, k), train["fantasy_points"]))
        chosen[f] = best
        oof[(df.fold == f).to_numpy()] = blend(test, best)
    print(f"\nkappa chosen per fold (fit on the other half): {chosen}")

    weekly, pace = df["weekly"].to_numpy(), df["pace"].to_numpy()
    summary = {
        "weekly": {"mae": mae(weekly, y), "rmse": rmse(weekly, y)},
        "pace_step8_over_17": {"mae": mae(pace, y), "rmse": rmse(pace, y)},
        "blend_oof": {"mae": mae(oof, y), "rmse": rmse(oof, y)},
    }
    print("\nout-of-fold headline")
    for name, m in summary.items():
        print(f"  {name:22s} MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}")

    boot = {
        "blend_vs_weekly": paired_bootstrap(df, oof, weekly),
        "blend_vs_pace": paired_bootstrap(df, oof, pace),
        "pace_vs_weekly": paired_bootstrap(df, pace, weekly),
    }
    print("\npaired bootstrap over players, MAE difference (negative = first is better)")
    for name, b in boot.items():
        print(f"  {name:16s} {b['mae_diff']:+.3f}  CI95 [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]"
              f"  {'WIN' if b['wins'] else 'no'}")

    # --- where does each one win? ------------------------------------------
    df["blend_oof"] = oof
    df["g_bucket"] = pd.cut(df["games_before"], [-1, 0, 3, 8, 99], labels=["0", "1-3", "4-8", "9+"])
    print("\nMAE by games played so far (weekly / pace / blend)")
    by_g = {}
    for b, grp in df.groupby("g_bucket", observed=True):
        yy = grp["fantasy_points"].to_numpy()
        by_g[str(b)] = {"n": int(len(grp)), "weekly": mae(grp.weekly, yy),
                        "pace": mae(grp.pace, yy), "blend": mae(grp.blend_oof, yy)}
        print(f"  g={b:>4} n={len(grp):5d}   {by_g[str(b)]['weekly']:.3f} / "
              f"{by_g[str(b)]['pace']:.3f} / {by_g[str(b)]['blend']:.3f}")
    print("\nMAE by position (weekly / pace / blend)")
    by_pos = {}
    for pos, grp in df.groupby("position"):
        yy = grp["fantasy_points"].to_numpy()
        by_pos[pos] = {"n": int(len(grp)), "weekly": mae(grp.weekly, yy),
                       "pace": mae(grp.pace, yy), "blend": mae(grp.blend_oof, yy)}
        print(f"  {pos}  n={len(grp):5d}   {by_pos[pos]['weekly']:.3f} / "
              f"{by_pos[pos]['pace']:.3f} / {by_pos[pos]['blend']:.3f}")

    out = {
        "season": args.season,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_file": rows_path.name,
        "n_rows": int(len(df)),
        "n_players": int(df.player_id.nunique()),
        "kappa_grid": [None if np.isinf(k) else k for k in KAPPAS],
        "in_sample_curve": {("inf" if np.isinf(k) else str(k)): v for k, v in curve.items()},
        "kappa_chosen_per_fold": {str(f): (None if np.isinf(k) else k) for f, k in chosen.items()},
        "headline_oof": summary,
        "paired_bootstrap": boot,
        "by_games_before": by_g,
        "by_position": by_pos,
    }
    out_path = OUT_DIR / f"pace_blend_{args.season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

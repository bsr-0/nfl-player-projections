"""Paired comparison of two serving-path walk-forward row files.

Both arms are scored on the SAME (player, season, week) rows, so the
difference is the model change and nothing else. A paired bootstrap over
players gives the CI; an arm "wins" only when the whole interval is on one
side of zero, the bar every experiment in this repo uses.

Also applies the games-played shrinkage toward the Step 8 pace on top of
each arm (scripts/run_pace_blend_experiment.py, kappa fixed at the value
that experiment chose) to answer the follow-up question: if the pace is a
FEATURE, is there anything left for the blend to add?

Usage:
    python scripts/compare_ab_rows.py A.rows.csv B.rows.csv [--label-a x --label-b y]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_pace_blend_experiment import (  # noqa: E402
    GAMES_PER_SEASON, WF_PATH, games_before, mae, paired_bootstrap, rmse,
)

KAPPA = 3.0
KEY = ["player_id", "season", "week"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a = pd.read_csv(args.a)[KEY + ["position", "fantasy_points", "predicted_points"]]
    b = pd.read_csv(args.b)[KEY + ["predicted_points"]]
    df = a.merge(b, on=KEY, suffixes=("_a", "_b"), how="inner")
    df = df[df.predicted_points_a.notna() & df.predicted_points_b.notna()].copy()
    season = int(df["season"].iloc[0])
    y = df["fantasy_points"].to_numpy()
    pa, pb = df["predicted_points_a"].to_numpy(), df["predicted_points_b"].to_numpy()

    print(f"{args.label_a}: {args.a.name}\n{args.label_b}: {args.b.name}")
    print(f"paired rows: {len(df)} (A had {len(a)}, B had {len(b)}), players {df.player_id.nunique()}")

    out = {"season": season, "n_rows": int(len(df)), "a": str(args.a), "b": str(args.b),
           "generated_at": datetime.now().isoformat(timespec="seconds")}
    out["headline"] = {
        args.label_a: {"mae": mae(pa, y), "rmse": rmse(pa, y)},
        args.label_b: {"mae": mae(pb, y), "rmse": rmse(pb, y)},
    }
    boot = paired_bootstrap(df, pb, pa)
    out["bootstrap_b_minus_a"] = boot
    print(f"\n{'':10s} {'MAE':>8s} {'RMSE':>8s}")
    for k, m in out["headline"].items():
        print(f"{k:10s} {m['mae']:8.3f} {m['rmse']:8.3f}")
    print(f"\n{args.label_b} - {args.label_a} MAE: {boot['mae_diff']:+.3f}  "
          f"CI95 [{boot['ci95'][0]:+.3f}, {boot['ci95'][1]:+.3f}]  "
          f"{'B WINS' if boot['wins'] else ('A WINS' if boot['ci95'][0] > 0 else 'no difference')}")

    # --- by games played / position ----------------------------------------
    df = df.merge(games_before(season), on=["player_id", "week"], how="left")
    df["g_bucket"] = pd.cut(df["games_before"], [-1, 0, 3, 8, 99], labels=["0", "1-3", "4-8", "9+"])
    print(f"\nMAE by games played so far ({args.label_a} / {args.label_b})")
    out["by_games_before"] = {}
    for g, grp in df.groupby("g_bucket", observed=True):
        yy = grp.fantasy_points.to_numpy()
        m = {"n": int(len(grp)), args.label_a: mae(grp.predicted_points_a, yy),
             args.label_b: mae(grp.predicted_points_b, yy)}
        out["by_games_before"][str(g)] = m
        print(f"  g={g:>4} n={m['n']:5d}   {m[args.label_a]:.3f} / {m[args.label_b]:.3f}")
    print(f"\nMAE by position ({args.label_a} / {args.label_b})")
    out["by_position"] = {}
    for pos, grp in df.groupby("position"):
        yy = grp.fantasy_points.to_numpy()
        m = {"n": int(len(grp)), args.label_a: mae(grp.predicted_points_a, yy),
             args.label_b: mae(grp.predicted_points_b, yy)}
        out["by_position"][pos] = m
        print(f"  {pos}  n={m['n']:5d}   {m[args.label_a]:.3f} / {m[args.label_b]:.3f}")

    # --- blend on top of each arm ------------------------------------------
    wf = pd.read_csv(WF_PATH)
    s8 = wf[(wf.arm == "step8") & (wf.season == season)][["player_id", "pred"]]
    sub = df.merge(s8, on="player_id", how="inner").copy()
    sub["pace"] = sub["pred"] / GAMES_PER_SEASON
    g = sub["games_before"].to_numpy(dtype=float)
    w = g / (g + KAPPA)
    yy = sub.fantasy_points.to_numpy()
    out["blend_on_top"] = {}
    print(f"\nkappa={KAPPA:g} blend toward Step 8 pace, on the {len(sub)} rows with a pace")
    for label, col in ((args.label_a, "predicted_points_a"), (args.label_b, "predicted_points_b")):
        raw = sub[col].to_numpy()
        bl = w * raw + (1 - w) * sub["pace"].to_numpy()
        bt = paired_bootstrap(sub, bl, raw)
        out["blend_on_top"][label] = {"raw_mae": mae(raw, yy), "blend_mae": mae(bl, yy), "bootstrap": bt}
        print(f"  {label:10s} raw {mae(raw, yy):.3f} -> blend {mae(bl, yy):.3f}   "
              f"diff {bt['mae_diff']:+.3f} CI95 [{bt['ci95'][0]:+.3f}, {bt['ci95'][1]:+.3f}] "
              f"{'blend still helps' if bt['wins'] else 'no further gain'}")

    out_path = args.out or (PROJECT_ROOT / "data" / "experiments" /
                            f"ab_compare_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwritten {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the live pace blend to the held-out served fold; verify it against production.

Live serving is not the weekly model alone: src/predict.py shrinks every
forecast toward the Step 8 season pace,

    served = w * weekly + (1 - w) * pace,   w = g / (g + PACE_BLEND_KAPPA),

with g the player's games already played this season. This script applies
that exact function (NFLPredictor._blend_toward_season_pace, not a copy) to a
served-fold export, so the 2025 incumbent is what production would have served.

    export  per target week T, blend the fold's predictions with history =
            the player's season rows in player_weekly_stats with week < T.
    verify  run production predict(as_of=(season, week)) for a completed week
            and assert this script's history construction reproduces its games
            counts and blended points exactly.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH, PACE_BLEND_KAPPA, STEP8_PACE_TABLE
from src.evaluation.paired_ppr_comparison import KEY, check_keys, file_sha256
from src.predict import NFLPredictor


def season_history(season: int, before_week: int) -> pd.DataFrame:
    """Player-season rows completed before `before_week`, as the blend counts games."""
    with sqlite3.connect(str(DB_PATH)) as con:
        h = pd.read_sql("SELECT player_id, season, week FROM player_weekly_stats "
                        "WHERE season = ? AND week < ?", con, params=(season, before_week))
    if h.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("player_weekly_stats has duplicate player-weeks")
    return h


def blend(results: pd.DataFrame, history: pd.DataFrame, season: int) -> pd.DataFrame:
    shell = NFLPredictor.__new__(NFLPredictor)   # the blend needs no fitted models
    return NFLPredictor._blend_toward_season_pace(shell, results, history, season, 1)


def export(fold_dir: Path, output_dir: Path) -> dict:
    fold_dir, output_dir = fold_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    pred_path = fold_dir / "predictions.csv"
    rows = pd.read_csv(pred_path, dtype={"player_id": str})
    check_keys(rows, "served fold")
    seasons = rows.season.unique()
    if len(seasons) != 1:
        raise ValueError("served fold must cover one season")
    season = int(seasons[0])
    parts = []
    for week, part in rows.groupby("week", sort=True):
        res = part.rename(columns={"predicted_ppr": "predicted_points"})
        out = blend(res, season_history(season, int(week)), season)
        parts.append(out)
    out = pd.concat(parts).rename(columns={"predicted_points": "predicted_ppr",
                                           "predicted_points_model": "predicted_ppr_weekly_model"})
    out = out.sort_values(KEY).reset_index(drop=True)
    if len(out) != len(rows) or not np.isfinite(out.predicted_ppr).all():
        raise ValueError("blend changed the row count or produced nonfinite predictions")
    output_dir.mkdir(parents=True)
    out_path = output_dir / "predictions.csv"
    out.to_csv(out_path, index=False)
    manifest = {
        "schema_version": 1, "label": "served_live_pace_blend", "test_season": season,
        "pace_blend_kappa": PACE_BLEND_KAPPA,
        "blend_function": "src.predict.NFLPredictor._blend_toward_season_pace",
        "games_played": "player_weekly_stats rows in the target season with week < target week",
        "inputs": {"served_fold_predictions": str(pred_path.relative_to(ROOT)),
                   "served_fold_predictions_sha256": file_sha256(pred_path),
                   "served_fold_manifest_sha256": file_sha256(fold_dir / "manifest.json"),
                   "step8_pace_table": str(STEP8_PACE_TABLE.relative_to(ROOT)),
                   "step8_pace_table_sha256": file_sha256(STEP8_PACE_TABLE)},
        "rows": len(out), "rows_with_pace": int(out.pace_prior.notna().sum()),
        "mean_pace_weight": float(out.pace_weight.mean()),
        "predictions_sha256": file_sha256(out_path),
        "script_sha256": file_sha256(Path(__file__)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(season: int, week: int) -> dict:
    live = NFLPredictor()
    live.initialize()
    prod = live.predict(n_weeks=1, top_n=100_000, as_of=(season, week))
    need = {"player_id", "predicted_points", "predicted_points_model", "games_played_season"}
    if not need <= set(prod.columns):
        raise ValueError(f"production output lacks {sorted(need - set(prod.columns))}")
    prod = prod.drop_duplicates("player_id")
    res = prod[["player_id"]].assign(predicted_points=prod.predicted_points_model.to_numpy())
    mine = blend(res, season_history(season, week), season).set_index("player_id")
    prod = prod.set_index("player_id")
    games_diff = (mine.games_played_season - prod.games_played_season).abs()
    pts_diff = (mine.predicted_points - prod.predicted_points).abs()
    report = {"season": season, "week": week, "players": int(len(prod)),
              "games_mismatches": int((games_diff > 0).sum()),
              "max_abs_points_diff": float(pts_diff.max()),
              "players_shrunk": int((prod.pace_weight > 0).sum()) if "pace_weight" in prod else None}
    print(json.dumps(report, indent=2))
    if report["games_mismatches"] or report["max_abs_points_diff"] > 1e-9:
        bad = games_diff[games_diff > 0].index[:10].tolist()
        raise SystemExit(f"blend export does not reproduce production; e.g. {bad}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--fold-dir", type=Path, required=True)
    e.add_argument("--output-dir", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--season", type=int, required=True)
    v.add_argument("--week", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        print(json.dumps(export(a.fold_dir, a.output_dir), indent=2))
    else:
        verify(a.season, a.week)


if __name__ == "__main__":
    main()

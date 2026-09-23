#!/usr/bin/env python3
"""Evaluate Plan A improvements against the rolling-3 share baseline.

This is an experiment-only harness. It emits fold-level predictions/metrics
under data/experiments and never saves production models.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config.settings import TEAM_ALLOCATION_MODEL_CONFIG
from src.evaluation.team_share_backtester import bootstrap_mae_delta
from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.baseline import RollingShareBaseline
from src.models.team_allocation.features import ROLL_WINDOW, VOLUME_COLS, feature_columns, filter_population, load_share_rows
from src.models.team_allocation.models import ShareRidgeModel, ShareXGBModel

def _fit(kind, X, y):
    m = ShareRidgeModel() if kind == "ridge" else ShareXGBModel()
    return m.fit(X, y)

def _renorm(pred, ids):
    out = np.asarray(pred, dtype=float).copy()
    frame = pd.DataFrame({"p": out, "team": ids["team"].to_numpy(), "week": ids["week"].to_numpy(), "season": ids["season"].to_numpy()})
    sums = frame.groupby(["season", "week", "team"])["p"].transform("sum").to_numpy()
    return np.divide(out, sums, out=np.zeros_like(out), where=sums > 0)

def _best_alpha(y_true, model_pred, baseline):
    return min(np.arange(0, 1.01, .05), key=lambda a: mean_absolute_error(
        y_true, a * model_pred + (1-a) * baseline))

def run(target, seasons=None, n_test_seasons=None, by_role=False):
    label = f"share_of_team_{target}"; roll = f"{label}_roll{ROLL_WINDOW}"
    df = filter_population(load_share_rows(seasons=seasons), target).reset_index(drop=True)
    cols = feature_columns(df); X, y = df[cols], df[label].to_numpy(float)
    seasons_arr = df.season.to_numpy(); splitter = SeasonAwareTimeSeriesSplit(
    n_splits=n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"],
        seasons=seasons_arr, gap_seasons=TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"], strict=True)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X)):
        train, test = X.iloc[tr], X.iloc[te]; yt, yv = y[tr], y[te]
        base = RollingShareBaseline(roll).fit(train, yt).predict(X.iloc[te])
        for kind in ("ridge", "xgb"):
            model = _fit(kind, train, yt)
            raw = np.asarray(model.predict(test), float)
            # Residual learning: fit the model to deviations from rolling-3.
            residual_model = _fit(kind, train, yt - X.iloc[tr][roll].fillna(0).to_numpy())
            residual = np.clip(X.iloc[te][roll].fillna(0).to_numpy() + residual_model.predict(test), 0, 1)
            # Fold-local blend weight selected on the latest training season.
            last = max(seasons_arr[tr]); cal = seasons_arr[tr] == last
            alpha = .5
            alpha_map = {}
            if cal.any() and (~cal).sum() >= 20:
                cal_model = _fit(kind, X.iloc[tr][~cal], yt[~cal])
                cal_pred = np.asarray(cal_model.predict(X.iloc[tr][cal]), float)
                cal_base = X.iloc[tr][roll].fillna(0).to_numpy()[cal]
                cal_pos = df.iloc[tr].position.to_numpy()[cal]
                cal_cold = df.iloc[tr].is_cold_start.to_numpy()[cal]
                groups = [("all", "all")] if not by_role else sorted(set(zip(cal_pos, cal_cold)), key=str)
                for group in groups:
                    mask = np.ones(len(cal_pred), dtype=bool) if group == ("all", "all") else ((cal_pos == group[0]) & (cal_cold == group[1]))
                    if mask.sum() >= 20:
                        alpha_map[group] = _best_alpha(y[tr][cal][mask], cal_pred[mask], cal_base[mask])
                alpha = alpha_map.get(("all", "all"), .5)
            if by_role:
                test_pos = df.iloc[te].position.to_numpy(); test_cold = df.iloc[te].is_cold_start.to_numpy()
                alpha_vec = np.array([alpha_map.get((p, c), alpha) for p, c in zip(test_pos, test_cold)])
            else:
                alpha_vec = np.full(len(raw), alpha)
            blend = np.clip(alpha_vec * raw + (1-alpha_vec) * base, 0, 1)
            renorm = _renorm(blend, df.iloc[te][["season", "week", "team"]])
            for name, pred in ((f"{kind}", raw), (f"{kind}_residual", residual), (f"{kind}_blend", blend), (f"{kind}_blend_renorm", renorm)):
                part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
                part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, name, yv, pred
                part["blend_alpha"] = alpha_vec
                rows.append(part)
        part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
        part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, "rolling3", yv, base
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    metrics = []
    for arm, g in out.groupby("arm"):
        metrics.append({"target": target, "arm": arm, "n": len(g), "mae": mean_absolute_error(g.actual_share, g.predicted_share),
                        "mae_established": mean_absolute_error(g[g.is_cold_start == 0].actual_share, g[g.is_cold_start == 0].predicted_share),
                        "mae_cold_start": mean_absolute_error(g[g.is_cold_start == 1].actual_share, g[g.is_cold_start == 1].predicted_share) if (g.is_cold_start == 1).any() else None})
    base = out[out.arm == "rolling3"].sort_values(["fold", "player_id"])
    for m in metrics:
        if m["arm"] != "rolling3":
            cand = out[out.arm == m["arm"]].sort_values(["fold", "player_id"])
            m["vs_rolling3_bootstrap"] = bootstrap_mae_delta(base.actual_share.to_numpy(), cand.predicted_share.to_numpy(), base.predicted_share.to_numpy())
    return out, metrics

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", choices=VOLUME_COLS, default="targets"); ap.add_argument("--seasons", nargs=2, type=int); ap.add_argument("--n-test-seasons", type=int, help="Use 1 for a final-season-only confirmation"); ap.add_argument("--by-role", action="store_true", help="Calibrate alpha separately by position and cold-start flag"); ap.add_argument("--output-dir", type=Path, default=Path("data/experiments/plan_a_improvements")); args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1]+1)) if args.seasons else None
    out, metrics = run(args.target, seasons, args.n_test_seasons, args.by_role); args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / f"{args.target}_predictions.csv", index=False)
    (args.output_dir / f"{args.target}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(json.dumps(metrics, indent=2, default=float))
if __name__ == "__main__": main()

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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

def _fit_bounded(kind, X, y):
    """Fit on logit shares and return a bounded predictor wrapper."""
    model = _fit(kind, X, np.log(np.clip(y, 1e-5, 1 - 1e-5) / np.clip(1 - y, 1e-5, 1)))
    model._bounded_logit = True
    return model

def _bounded_predict(model, X):
    return 1 / (1 + np.exp(-np.clip(np.asarray(model.predict(X), float), -30, 30)))

def _two_stage_position_prediction(train, test, y_train, positions_train,
                                    positions_test, kind, active_threshold=0.0):
    """Predict rushing share as P(active) * E[positive share | active]."""
    out = np.zeros(len(test), dtype=float)
    for position in np.unique(positions_test):
        tr_mask = positions_train == position
        te_mask = positions_test == position
        if tr_mask.sum() < 40 or te_mask.sum() == 0:
            continue
        active = y_train[tr_mask] > active_threshold
        if active.all() or (~active).all() or active.sum() < 20:
            continue
        classifier = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        classifier.fit(train.loc[tr_mask], active.astype(int))
        probability = classifier.predict_proba(test.loc[te_mask])[:, 1]
        positive_model = _fit(kind, train.loc[tr_mask].loc[active], y_train[tr_mask][active])
        positive = np.clip(np.asarray(positive_model.predict(test.loc[te_mask]), float), 0, 1)
        out[te_mask] = probability * positive
    return out

def _renorm(pred, ids):
    out = np.asarray(pred, dtype=float).copy()
    frame = pd.DataFrame({"p": out, "team": ids["team"].to_numpy(), "week": ids["week"].to_numpy(), "season": ids["season"].to_numpy()})
    sums = frame.groupby(["season", "week", "team"])["p"].transform("sum").to_numpy()
    return np.divide(out, sums, out=np.zeros_like(out), where=sums > 0)

def _best_alpha(y_true, model_pred, baseline):
    return min(np.arange(0, 1.01, .05), key=lambda a: mean_absolute_error(
        y_true, a * model_pred + (1-a) * baseline))

def _feature_columns_for_target(df, target, carry_context=False):
    """Add only lagged team-rush context for the rushing-target experiment."""
    cols = feature_columns(df)
    if target == "rushing_attempts" and carry_context:
        for name in ("share_of_team_rushing_attempts_s2d",
                     "team_rushing_attempts_roll3", "team_rushing_attempts_s2d",
                     "team_rushing_yards_roll3", "team_rushing_yards_s2d"):
            if name in df.columns and name not in cols:
                cols.append(name)
    return cols

def run(target, seasons=None, n_test_seasons=None, by_role=False, by_volume=False,
        by_position_volume=False, alpha_shrinkage_k=100, carry_context=False,
        guard_group_loss=False, dedicated_rb=False, two_stage_rushing=False,
        active_threshold=0.0):
    label = f"share_of_team_{target}"; roll = f"{label}_roll{ROLL_WINDOW}"
    df = filter_population(load_share_rows(seasons=seasons), target).reset_index(drop=True)
    cols = _feature_columns_for_target(df, target, carry_context); X, y = df[cols], df[label].to_numpy(float)
    seasons_arr = df.season.to_numpy(); splitter = SeasonAwareTimeSeriesSplit(
    n_splits=n_test_seasons or TEAM_ALLOCATION_MODEL_CONFIG["n_walk_forward_test_seasons"],
        seasons=seasons_arr, gap_seasons=TEAM_ALLOCATION_MODEL_CONFIG["cv_gap_seasons"], strict=True)
    rows = []
    for fold, (tr, te) in enumerate(splitter.split(X)):
        train, test = X.iloc[tr], X.iloc[te]; yt, yv = y[tr], y[te]
        base = RollingShareBaseline(roll).fit(train, yt).predict(X.iloc[te])
        test_positions = df.iloc[te].position.to_numpy()
        for kind in ("ridge", "xgb"):
            model = _fit(kind, train, yt)
            raw = np.asarray(model.predict(test), float)
            bounded_model = _fit_bounded(kind, train, yt)
            bounded = _bounded_predict(bounded_model, test)
            # Residual learning: fit the model to deviations from rolling-3.
            residual_model = _fit(kind, train, yt - X.iloc[tr][roll].fillna(0).to_numpy())
            residual = np.clip(X.iloc[te][roll].fillna(0).to_numpy() + residual_model.predict(test), 0, 1)
            # Fold-local blend weight selected on the latest training season.
            last = max(seasons_arr[tr]); cal = seasons_arr[tr] == last
            train_base_all = X.iloc[tr][roll].fillna(0).to_numpy()
            q1, q2 = np.quantile(train_base_all, [1/3, 2/3])
            train_tier = np.where(train_base_all <= q1, "low", np.where(train_base_all <= q2, "mid", "high"))
            test_base = X.iloc[te][roll].fillna(0).to_numpy()
            test_tier = np.where(test_base <= q1, "low", np.where(test_base <= q2, "mid", "high"))
            alpha = .5
            alpha_map = {}
            if cal.any() and (~cal).sum() >= 20:
                cal_model = _fit(kind, X.iloc[tr][~cal], yt[~cal])
                cal_pred = np.asarray(cal_model.predict(X.iloc[tr][cal]), float)
                cal_base = X.iloc[tr][roll].fillna(0).to_numpy()[cal]
                cal_pos = df.iloc[tr].position.to_numpy()[cal]
                cal_cold = df.iloc[tr].is_cold_start.to_numpy()[cal]
                if by_position_volume:
                    groups = sorted(set(zip(cal_pos, train_tier[cal])), key=str)
                elif by_volume:
                    groups = sorted(set(train_tier[cal]), key=str)
                elif by_role:
                    groups = sorted(set(zip(cal_pos, cal_cold)), key=str)
                else:
                    groups = [("all", "all")]
                # Always estimate a global calibration alpha. Group-specific
                # alphas are shrunk toward it, preventing sparse position /
                # volume cells from overfitting one calibration season.
                global_alpha = _best_alpha(y[tr][cal], cal_pred, cal_base)
                alpha_map[("all", "all")] = global_alpha
                for group in groups:
                    if by_position_volume:
                        mask = (cal_pos == group[0]) & (train_tier[cal] == group[1])
                    elif by_volume:
                        mask = train_tier[cal] == group
                    else:
                        mask = np.ones(len(cal_pred), dtype=bool) if group == ("all", "all") else ((cal_pos == group[0]) & (cal_cold == group[1]))
                    if mask.sum() >= 20:
                        raw_alpha = _best_alpha(y[tr][cal][mask], cal_pred[mask], cal_base[mask])
                        if guard_group_loss:
                            raw_mae = mean_absolute_error(
                                y[tr][cal][mask],
                                raw_alpha * cal_pred[mask] + (1 - raw_alpha) * cal_base[mask],
                            )
                            base_mae = mean_absolute_error(y[tr][cal][mask], cal_base[mask])
                            if raw_mae >= base_mae:
                                raw_alpha = 0.0
                        if by_position_volume:
                            n_group = int(mask.sum())
                            weight = n_group / (n_group + alpha_shrinkage_k)
                            alpha_map[group] = weight * raw_alpha + (1 - weight) * global_alpha
                        else:
                            alpha_map[group] = raw_alpha
                alpha = alpha_map.get(("all", "all"), .5)
            if by_role:
                test_pos = df.iloc[te].position.to_numpy(); test_cold = df.iloc[te].is_cold_start.to_numpy()
                alpha_vec = np.array([alpha_map.get((p, c), alpha) for p, c in zip(test_pos, test_cold)])
            elif by_position_volume:
                test_pos = df.iloc[te].position.to_numpy()
                alpha_vec = np.array([alpha_map.get((p, t), alpha) for p, t in zip(test_pos, test_tier)])
            elif by_volume:
                alpha_vec = np.array([alpha_map.get(t, alpha) for t in test_tier])
            else:
                alpha_vec = np.full(len(raw), alpha)
            blend = np.clip(alpha_vec * raw + (1-alpha_vec) * base, 0, 1)
            renorm = _renorm(blend, df.iloc[te][["season", "week", "team"]])
            for name, pred in ((f"{kind}", raw), (f"{kind}_bounded", bounded), (f"{kind}_residual", residual), (f"{kind}_blend", blend), (f"{kind}_blend_renorm", renorm)):
                part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
                part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, name, yv, pred
                part["blend_alpha"] = alpha_vec
                rows.append(part)
            if target == "rushing_attempts" and dedicated_rb:
                train_positions = df.iloc[tr].position.to_numpy()
                rb_train = train_positions == "RB"
                rb_test = test_positions == "RB"
                dedicated_model = _fit(kind, train.loc[rb_train], yt[rb_train])
                dedicated = base.copy()
                dedicated[rb_test] = (
                    alpha_vec[rb_test] * np.asarray(dedicated_model.predict(test.loc[rb_test]), float)
                    + (1 - alpha_vec[rb_test]) * base[rb_test]
                )
                dedicated = _renorm(dedicated, df.iloc[te][["season", "week", "team"]])
                part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
                part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, f"{kind}_dedicated_rushing", yv, dedicated
                part["blend_alpha"] = alpha_vec
                rows.append(part)
            if target == "rushing_attempts" and two_stage_rushing:
                train_positions = df.iloc[tr].position.to_numpy()
                two_stage = _two_stage_position_prediction(
                    train, test, yt, train_positions, test_positions, kind,
                    active_threshold=active_threshold,
                )
                two_stage_blend = np.clip(alpha_vec * two_stage + (1 - alpha_vec) * base, 0, 1)
                two_stage_renorm = _renorm(two_stage_blend, df.iloc[te][["season", "week", "team"]])
                part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
                part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, f"{kind}_two_stage_rushing", yv, two_stage_renorm
                part["blend_alpha"] = alpha_vec
                rows.append(part)
        part = df.iloc[te][["player_id", "season", "week", "team", "position", "is_cold_start"]].copy()
        part["fold"], part["arm"], part["actual_share"], part["predicted_share"] = fold, "rolling3", yv, base
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    metrics = []
    for arm, g in out.groupby("arm"):
        group_cols = ["season", "week", "team"]
        sums = g.groupby(group_cols).agg(pred_sum=("predicted_share", "sum"), actual_sum=("actual_share", "sum"), n_players=("player_id", "size"))
        sparse = g.merge(sums["n_players"].reset_index(), on=group_cols)
        baseline = g.predicted_share if arm == "rolling3" else None
        row = {"target": target, "arm": arm, "n": len(g), "mae": mean_absolute_error(g.actual_share, g.predicted_share),
                        "mae_established": mean_absolute_error(g[g.is_cold_start == 0].actual_share, g[g.is_cold_start == 0].predicted_share),
                        "mae_cold_start": mean_absolute_error(g[g.is_cold_start == 1].actual_share, g[g.is_cold_start == 1].predicted_share) if (g.is_cold_start == 1).any() else None,
                        "mean_abs_team_sum_error": float(np.abs(sums.pred_sum - 1).mean()),
                        "p95_abs_team_sum_error": float(np.abs(sums.pred_sum - 1).quantile(.95)),
                        "actual_team_sum_error": float(np.abs(sums.actual_sum - 1).mean()),
                        "sparse_group_mean_abs_sum_error": float(np.abs(sums.loc[sums.n_players <= 2, "pred_sum"] - 1).mean()) if (sums.n_players <= 2).any() else None,
                        "cold_start_mae": mean_absolute_error(g[g.is_cold_start == 1].actual_share, g[g.is_cold_start == 1].predicted_share) if (g.is_cold_start == 1).any() else None}
        row["by_position"] = {str(k): float(mean_absolute_error(v.actual_share, v.predicted_share)) for k, v in g.groupby("position")}
        row["by_volume_tier"] = {str(k): float(mean_absolute_error(v.actual_share, v.predicted_share)) for k, v in g.assign(volume_tier=pd.qcut(g.predicted_share.rank(method="first"), 3, labels=["low", "mid", "high"])).groupby("volume_tier")}
        metrics.append(row)
    base = out[out.arm == "rolling3"].sort_values(["fold", "player_id"])
    for m in metrics:
        if m["arm"] != "rolling3":
            cand = out[out.arm == m["arm"]].sort_values(["fold", "player_id"])
            m["vs_rolling3_bootstrap"] = bootstrap_mae_delta(base.actual_share.to_numpy(), cand.predicted_share.to_numpy(), base.predicted_share.to_numpy())
    for kind in ("ridge", "xgb"):
        raw = out[out.arm == f"{kind}_blend"].set_index(["player_id", "season", "week"])
        renorm = out[out.arm == f"{kind}_blend_renorm"].set_index(["player_id", "season", "week"])
        if not raw.empty and not renorm.empty:
            delta = (renorm.predicted_share - raw.predicted_share).abs()
            for m in metrics:
                if m["arm"] == f"{kind}_blend_renorm":
                    m["mean_abs_renorm_distortion"] = float(delta.mean())
                    m["p95_abs_renorm_distortion"] = float(delta.quantile(.95))
    return out, metrics

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", choices=VOLUME_COLS, default="targets"); ap.add_argument("--seasons", nargs=2, type=int); ap.add_argument("--n-test-seasons", type=int, help="Use 1 for a final-season-only confirmation"); ap.add_argument("--by-role", action="store_true", help="Calibrate alpha separately by position and cold-start flag"); ap.add_argument("--by-volume", action="store_true", help="Calibrate alpha separately by low/mid/high baseline share tier"); ap.add_argument("--by-position-volume", action="store_true", help="Calibrate position x volume alpha with shrinkage toward global alpha"); ap.add_argument("--alpha-shrinkage-k", type=float, default=100.0, help="Pseudo-count for position x volume alpha shrinkage"); ap.add_argument("--carry-context", action="store_true", help="For rushing attempts only, include lagged team rushing totals"); ap.add_argument("--guard-group-loss", action="store_true", help="Fallback to rolling-3 when a calibration group does not improve"); ap.add_argument("--dedicated-rb", action="store_true", help="For rushing attempts, fit the learned arm on RB rows only and keep other positions on rolling-3"); ap.add_argument("--two-stage-rushing", action="store_true", help="For rushing attempts, classify active rows then regress positive share per position"); ap.add_argument("--active-threshold", type=float, default=0.0, help="Positive-share threshold for two-stage rushing activity"); ap.add_argument("--output-dir", type=Path, default=Path("data/experiments/plan_a_improvements")); args = ap.parse_args()
    seasons = list(range(args.seasons[0], args.seasons[1]+1)) if args.seasons else None
    out, metrics = run(args.target, seasons, args.n_test_seasons, args.by_role, args.by_volume, args.by_position_volume, args.alpha_shrinkage_k, args.carry_context, args.guard_group_loss, args.dedicated_rb, args.two_stage_rushing, args.active_threshold); args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / f"{args.target}_predictions.csv", index=False)
    (args.output_dir / f"{args.target}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    print(json.dumps(metrics, indent=2, default=float))
if __name__ == "__main__": main()

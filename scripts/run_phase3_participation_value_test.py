#!/usr/bin/env python3
"""Preregistered Phase 3 test of Phase 2 participation outputs in weekly PPR.

This is an offline experiment only. It reuses the canonical Phase 2 OOF file,
the existing Phase 7 fold loader, and identical matched player-week keys for
every arm. Nothing here changes FINAL_CONFIG or serving predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CAUSAL_FEATURES, POSITIONS
from src.models.single_week_ppr.evaluate import (
    _architectures_for_fold,
    compute_metrics,
    run_fold,
)
from src.models.single_week_ppr.final_config import FINAL_CONFIG
from src.models.single_week_ppr.participation_integration import (
    KEY,
    _feature_matrix,
    attach_phase2_oof,
    validate_phase2_manifest,
    validate_phase2_oof,
)
from src.models.single_week_ppr.windows import compute_recency_weights, window_to_season_list
from src.utils.leakage import filter_feature_columns

SEED = 42
N_BOOTSTRAP = 1000
ARMS = {
    "A_baseline": [],
    "B_p_participation": ["p2_participation_probability"],
    "C_conditional_snap_share": ["p2_conditional_snap_share"],
    "D_unconditional_expected_snap_share": ["p2_expected_snap_share"],
    "E_rb_full_replace": ["p2_expected_snap_share"],
}
USAGE_TOKENS = ("snap_share", "rush_share", "target_share", "targets_roll3_mean")


def _features(position: str, train: pd.DataFrame, test: pd.DataFrame, arm: str) -> list[str]:
    cols = filter_feature_columns(CAUSAL_FEATURES.get(position, []))
    cols = [c for c in cols if c in train.columns and c in test.columns]
    if arm == "E_rb_full_replace":
        cols = [c for c in cols if not any(token in c for token in USAGE_TOKENS)]
    required = ARMS[arm]
    if not all(c in train.columns and c in test.columns for c in required):
        raise ValueError(f"missing Phase 2 feature(s) for {arm}")
    cols += required
    if not cols:
        raise ValueError(f"no features available for {position}/{arm}")
    return list(dict.fromkeys(cols))


def _importance(model, feature_names: list[str]) -> dict[str, float]:
    obj = model
    vals = None
    for _ in range(4):
        if hasattr(obj, "feature_importances_"):
            vals = np.asarray(obj.feature_importances_, dtype=float)
            break
        obj = getattr(obj, "model", None)
        if obj is None:
            break
    if vals is None or len(vals) != len(feature_names):
        return {}
    total = float(vals.sum())
    if total > 0:
        vals = vals / total
    return {f: float(v) for f, v in zip(feature_names, vals)}


def _bootstrap_delta(rows: pd.DataFrame, n: int = N_BOOTSTRAP) -> dict:
    required = ["player_id", "actual", "baseline_prediction", "candidate_prediction"]
    clean = rows[required].dropna()
    if clean.empty:
        return {"n": 0, "mae_delta_candidate_minus_baseline": np.nan,
                "improvement_baseline_minus_candidate": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan}
    clean = clean.copy()
    clean["delta"] = (clean.candidate_prediction - clean.actual).abs() - (
        clean.baseline_prediction - clean.actual
    ).abs()
    players = clean.player_id.drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    estimates = np.empty(n)
    by_player = {p: clean[clean.player_id.eq(p)] for p in players}
    for i in range(n):
        draw = rng.choice(players, size=len(players), replace=True)
        estimates[i] = pd.concat([by_player[p] for p in draw], ignore_index=True).delta.mean()
    observed = float(clean.delta.mean())
    return {
        "n": int(len(clean)),
        "n_players": int(len(players)),
        "mae_delta_candidate_minus_baseline": observed,
        "improvement_baseline_minus_candidate": -observed,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase2-oof", type=Path,
                    default=Path("data/experiments/participation_system/phase2/oof_predictions.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("data/experiments/participation_system/phase3"))
    ap.add_argument("--phase2-model", default="hist_gbm")
    ap.add_argument("--seasons", nargs="+", type=int, default=None)
    ap.add_argument("--positions", nargs="+", default=None)
    args = ap.parse_args()

    manifest = validate_phase2_manifest(args.phase2_oof)
    raw_oof = pd.read_csv(args.phase2_oof)
    oof = validate_phase2_oof(raw_oof, model=args.phase2_model)
    conditional = raw_oof.loc[raw_oof.model.eq(args.phase2_model), KEY + ["conditional_snap_share"]].copy()
    conditional = conditional.drop_duplicates(KEY)
    oof = oof.merge(conditional, on=KEY, how="left", validate="one_to_one")
    oof = oof.rename(columns={"conditional_snap_share": "p2_conditional_snap_share"})
    if "p2_conditional_snap_share" not in oof.columns:
        raise ValueError("Phase 2 OOF does not contain conditional_snap_share")
    seasons = args.seasons or sorted(oof["season"].unique().tolist())
    positions = args.positions or list(POSITIONS)
    available_seasons = sorted(oof["season"].unique())
    # The first OOF season has no strictly prior Phase 2 training season.
    # Treat that boundary as intentionally not evaluable, not as a failed fold.
    eligible_seasons = [s for s in seasons if any(
        window_to_season_list(FINAL_CONFIG[p]["window"], int(s), available_seasons)
        for p in positions
    )]
    excluded_seasons = [s for s in seasons if s not in eligible_seasons]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    row_frames, fold_metrics, importance_rows, failures, leakage_checks = [], [], [], [], []

    for position in positions:
        cfg = FINAL_CONFIG[position]
        for season in eligible_seasons:
            try:
                available = sorted(oof["season"].unique())
                train_seasons = window_to_season_list(cfg["window"], int(season), available)
                if not train_seasons:
                    raise ValueError("no training seasons")
                train_df, test_df, _, _ = run_fold(
                    position, int(season), False,
                    train_seasons_override=train_seasons, fit_existing_models=False,
                )
                train = attach_phase2_oof(train_df[train_df.position.eq(position)].copy(), oof)
                test = attach_phase2_oof(test_df[test_df.position.eq(position)].copy(), oof)
                test = test.dropna(subset=["fantasy_points"]).copy()
                if len(train) < 20 or len(test) < 20:
                    raise ValueError(f"insufficient matched rows train={len(train)} test={len(test)}")
                leakage_checks.append({
                    "position": position, "season": int(season), "rows": int(len(test)),
                    "max_phase2_train_season": int(test.phase2_train_max_season.max()),
                    "test_season": int(season),
                    "strictly_prior": bool(test.phase2_train_max_season.lt(season).all()),
                })
                if not test.phase2_train_max_season.lt(season).all():
                    raise ValueError("joined Phase 2 OOF contains current/future training provenance")
                weights = compute_recency_weights(train["season"], cfg["weighting"])
                shared_idx = test.index
                for arm in [a for a in ARMS if a != "E_rb_full_replace" or position == "RB"]:
                    features = _features(position, train, test, arm)
                    X_train, y_train = _feature_matrix(train, features)
                    X_test, y_test = _feature_matrix(test.loc[shared_idx], features)
                    model = _architectures_for_fold()[cfg["architecture"]]
                    fit_kwargs = {}
                    if weights is not None:
                        fit_kwargs["sample_weight"] = pd.Series(weights, index=train.index).reindex(X_train.index)
                    model.fit(X_train, y_train, **fit_kwargs)
                    pred = pd.Series(model.predict(X_test), index=X_test.index, dtype=float)
                    fold_metrics.append({
                        "position": position, "season": int(season), "arm": arm,
                        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                        "n_features": int(len(features)), **compute_metrics(y_test, pred),
                    })
                    frame = test.loc[X_test.index, KEY].copy()
                    frame["position"] = position
                    frame["arm"] = arm
                    frame["actual"] = y_test
                    frame["prediction"] = pred
                    prior_counts = train[train.season.lt(season)].groupby("player_id").size()
                    frame["prior_player_weeks"] = frame.player_id.map(prior_counts).fillna(0).astype(int)
                    frame["history_group"] = np.where(frame.prior_player_weeks < 3, "sparse", "established")
                    row_frames.append(frame)
                    for feature, value in _importance(model, features).items():
                        importance_rows.append({
                            "position": position, "season": int(season), "arm": arm,
                            "feature": feature, "importance": value,
                            "is_participation_feature": feature.startswith("p2_"),
                            "is_incumbent_usage_feature": any(t in feature for t in USAGE_TOKENS),
                        })
            except Exception as exc:
                failures.append({
                    "position": position, "season": int(season),
                    "error_type": type(exc).__name__, "error": str(exc),
                })

    rows = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    metrics = pd.DataFrame(fold_metrics)
    importance = pd.DataFrame(importance_rows)
    bootstrap = []
    if not rows.empty:
        baseline = rows[rows.arm.eq("A_baseline")][
            KEY + ["position", "actual", "prediction", "history_group"]
        ].rename(columns={"prediction": "baseline_prediction"})
        for arm in [a for a in ARMS if a != "A_baseline"]:
            cand = rows[rows.arm.eq(arm)][
                KEY + ["position", "prediction", "history_group"]
            ].rename(columns={"prediction": "candidate_prediction"})
            joined = baseline.merge(
                cand, on=KEY + ["position", "history_group"],
                how="inner", validate="one_to_one",
            )
            for subgroup, group in [
                ("all", joined),
                ("sparse", joined[joined.history_group.eq("sparse")]),
                ("established", joined[joined.history_group.eq("established")]),
            ]:
                bootstrap.append({
                    "arm": arm, "subgroup": subgroup, "position": "ALL", "season": "ALL",
                    **_bootstrap_delta(group),
                })
                for (pos, yr), sub in group.groupby(["position", "season"]):
                    bootstrap.append({
                        "arm": arm, "subgroup": subgroup, "position": pos, "season": int(yr),
                        **_bootstrap_delta(sub),
                    })

    qb_decisions = []
    for arm in [a for a in ARMS if a != "A_baseline"]:
        if rows.empty:
            q = {"improvement_baseline_minus_candidate": np.nan, "ci95_high": np.nan}
        else:
            base = rows[(rows.arm == "A_baseline") & (rows.position == "QB")][
                KEY + ["position", "actual", "prediction"]
            ].rename(columns={"prediction": "baseline_prediction"})
            cand = rows[(rows.arm == arm) & (rows.position == "QB")][
                KEY + ["position", "actual", "prediction"]
            ].rename(columns={"prediction": "candidate_prediction"})
            q = _bootstrap_delta(base.merge(
                cand, on=KEY + ["position", "actual"],
                how="inner", validate="one_to_one",
            ))
        qb_decisions.append({
            "arm": arm, "qb_bootstrap": q,
            "passes_margin": bool(q.get("improvement_baseline_minus_candidate", np.nan) >= 0.15),
            "passes_ci": bool(q.get("ci95_high", np.nan) < 0),
            "passes_primary_gate": bool(
                q.get("improvement_baseline_minus_candidate", np.nan) >= 0.15
                and q.get("ci95_high", np.nan) < 0
            ),
        })

    decision = {
        "phase": 3,
        "decision": "pending_review" if failures else "no_variant_promoted",
        "primary_position": "QB",
        "serving_change": "none",
        "failures": failures,
        "all_requested_folds_completed": not failures,
        "excluded_boundary_seasons": excluded_seasons,
        "primary_gate_results": qb_decisions,
        "rule": "promote only when QB MAE improvement >= 0.15 and paired 95% bootstrap CI for candidate-minus-baseline is below zero; RB is non-gating",
        "selected_phase2_model": args.phase2_model,
        "phase2_manifest": manifest,
    }
    metrics.to_csv(output / "phase3_fold_metrics.csv", index=False)
    rows.to_csv(output / "phase3_row_predictions.csv", index=False)
    importance.to_csv(output / "phase3_feature_importance.csv", index=False)
    (output / "phase3_bootstrap.json").write_text(json.dumps(bootstrap, indent=2, default=str) + "\n")
    (output / "phase3_evaluation.json").write_text(json.dumps({
        "fold_metrics": fold_metrics, "leakage_checks": leakage_checks,
        "failures": failures, "arms": ARMS, "season_list": eligible_seasons,
        "excluded_boundary_seasons": excluded_seasons, "positions": positions,
    }, indent=2, default=str) + "\n")
    (output / "phase3_decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n")
    print(json.dumps({
        "decision": decision["decision"], "folds": len(fold_metrics), "rows": len(rows),
        "failures": len(failures), "primary_gate_results": qb_decisions,
    }, indent=2, default=str))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

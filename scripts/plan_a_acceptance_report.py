#!/usr/bin/env python3
"""Build the Plan A accuracy, segment, constraint, and serving gate report.

This consumes only retained row-level validation artifacts. It never promotes
or writes a served model. A served comparison is marked blocked unless a
row-level export is supplied and joins one-to-one on player/week/team.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "data" / "experiments"
KEYS = ["player_id", "season", "week", "team"]
TARGETS = {
    "targets": {
        2024: ("plan_a_volume_second_holdout", "xgb_blend_renorm"),
        2025: ("plan_a_volume_shrinkage", "xgb_blend_renorm"),
    },
    "receiving_yards": {
        2024: ("plan_a_volume_second_holdout", "xgb_blend_renorm"),
        2025: ("plan_a_volume_2025_remaining", "xgb_blend_renorm"),
    },
    "rushing_yards": {
        2024: ("plan_a_volume_second_holdout", "ridge_blend_renorm"),
        2025: ("plan_a_volume_2025_remaining", "ridge_blend_renorm"),
    },
    "rushing_attempts": {
        2024: ("plan_a_two_stage_rushing_2024", "xgb_two_stage_rushing"),
        2025: ("plan_a_two_stage_rushing_2025", "xgb_two_stage_rushing"),
    },
}
SERVED_ARMS = {
    "targets": "xgb_volume_blend_renorm",
    "receiving_yards": "xgb_volume_blend_renorm",
    "rushing_yards": "ridge_volume_blend_renorm",
    "rushing_attempts": "xgb_two_stage_rushing",
}


def _segment_metrics(base: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    joined = base[KEYS + ["actual_share", "predicted_share", "position", "is_cold_start"]].merge(
        candidate[KEYS + ["predicted_share"]], on=KEYS, suffixes=("_base", "_candidate"), validate="one_to_one"
    )
    rank = joined["predicted_share_base"].rank(method="first")
    joined["volume_tier"] = pd.qcut(rank, 3, labels=["low", "mid", "high"])
    result = {}
    for name, groups in (
        ("position", joined.groupby("position", observed=True)),
        ("cold_start", joined.groupby("is_cold_start", observed=True)),
        ("volume_tier", joined.groupby("volume_tier", observed=True)),
    ):
        result[name] = {}
        for key, group in groups:
            base_mae = float(np.abs(group.actual_share - group.predicted_share_base).mean())
            candidate_mae = float(np.abs(group.actual_share - group.predicted_share_candidate).mean())
            result[name][str(key)] = {
                "n": int(len(group)),
                "baseline_mae": base_mae,
                "candidate_mae": candidate_mae,
                "delta_candidate_minus_baseline": candidate_mae - base_mae,
            }
    return result


def _constraint_metrics(candidate: pd.DataFrame) -> dict:
    grouped = candidate.groupby(["season", "week", "team"], observed=True).agg(
        predicted_sum=("predicted_share", "sum"), n_players=("player_id", "size")
    )
    errors = (grouped["predicted_sum"] - 1).abs()
    sparse = grouped[grouped.n_players <= 2]
    return {
        "group_count": int(len(grouped)),
        "sparse_group_count": int(len(sparse)),
        "mean_abs_team_sum_error": float(errors.mean()),
        "p95_abs_team_sum_error": float(errors.quantile(0.95)),
        "max_abs_team_sum_error": float(errors.max()),
        "sparse_mean_abs_team_sum_error": float((sparse.predicted_sum - 1).abs().mean()) if len(sparse) else None,
        "sparse_max_abs_team_sum_error": float((sparse.predicted_sum - 1).abs().max()) if len(sparse) else None,
    }


def _evaluate_file(path: Path, arm: str, season: int) -> dict:
    frame = pd.read_csv(path)
    required = set(KEYS + ["actual_share", "predicted_share", "arm", "position", "is_cold_start"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    base = frame[frame.arm.eq("rolling3")].copy()
    candidate = frame[frame.arm.eq(arm)].copy()
    if base[KEYS].duplicated().any() or candidate[KEYS].duplicated().any():
        raise ValueError(f"{path} has duplicate row keys for baseline or {arm}")
    if set(map(tuple, base[KEYS].to_numpy())) != set(map(tuple, candidate[KEYS].to_numpy())):
        raise ValueError(f"{path} baseline and {arm} populations differ")
    joined = base[KEYS + ["actual_share", "predicted_share"]].merge(
        candidate[KEYS + ["predicted_share"]], on=KEYS, suffixes=("_base", "_candidate"), validate="one_to_one"
    )
    base_mae = float(np.abs(joined.actual_share - joined.predicted_share_base).mean())
    candidate_mae = float(np.abs(joined.actual_share - joined.predicted_share_candidate).mean())
    metrics_path = path.with_name(path.name.replace("_predictions.csv", "_metrics.json"))
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    metric = next((row for row in metrics if row.get("arm") == arm), {})
    return {
        "season": season,
        "source": str(path),
        "arm": arm,
        "rows": int(len(joined)),
        "baseline_mae": base_mae,
        "candidate_mae": candidate_mae,
        "delta_candidate_minus_baseline": candidate_mae - base_mae,
        "bootstrap": metric.get("vs_rolling3_bootstrap"),
        "segments": _segment_metrics(base, candidate),
        "constraints": _constraint_metrics(candidate),
    }


def _served_status(served_csv: Path | None) -> dict:
    if served_csv is None:
        return {"status": "blocked", "reason": "No row-level Plan A served export supplied."}
    if not served_csv.exists():
        return {"status": "blocked", "reason": f"Served export does not exist: {served_csv}"}
    frame = pd.read_csv(served_csv)
    required = set(KEYS + ["target", "predicted_share", "arm"])
    missing = required - set(frame.columns)
    if missing:
        return {"status": "blocked", "reason": f"Served export missing columns: {sorted(missing)}"}
    if frame[KEYS + ["target"]].duplicated().any():
        return {"status": "blocked", "reason": "Served export has duplicate target/player/week/team rows."}
    return {"status": "ready", "rows": int(len(frame)), "source": str(served_csv)}


def _compare_served(served_csv: Path, accuracy_gate: dict) -> dict:
    served = pd.read_csv(served_csv)
    comparisons = {}
    for target, seasons in TARGETS.items():
        for season, (folder, arm) in seasons.items():
            label = f"{target}/{season}"
            holdout_path = EXPERIMENTS / folder / f"{target}_predictions.csv"
            holdout = pd.read_csv(holdout_path)
            holdout = holdout[holdout.arm.eq(arm)]
            right = served[(served.target == target) & (served.season == season)]
            expected_arm = SERVED_ARMS[target]
            served_arms = sorted(right.arm.dropna().astype(str).unique())
            left_keys = set(map(tuple, holdout[KEYS].to_numpy()))
            right_keys = set(map(tuple, right[KEYS].to_numpy()))
            if left_keys != right_keys:
                comparisons[label] = {
                    "status": "blocked",
                    "reason": "Served and validation row-key populations differ.",
                    "holdout_rows": len(left_keys),
                    "served_rows": len(right_keys),
                }
                continue
            joined = holdout[KEYS + ["actual_share", "predicted_share"]].merge(
                right[KEYS + ["predicted_share"]], on=KEYS,
                suffixes=("_holdout", "_served"), validate="one_to_one"
            )
            holdout_mae = float(np.abs(joined.actual_share - joined.predicted_share_holdout).mean())
            served_mae = float(np.abs(joined.actual_share - joined.predicted_share_served).mean())
            arm_matches = len(served_arms) == 1 and served_arms[0] == expected_arm
            comparisons[label] = {
                "status": "ok" if arm_matches else "arm_mismatch",
                "expected_arm": expected_arm,
                "served_arms": served_arms,
                "matched_rows": int(len(joined)),
                "holdout_mae": holdout_mae,
                "served_mae": served_mae,
                "served_minus_holdout_mae": served_mae - holdout_mae,
            }
    blocked = [v for v in comparisons.values() if v["status"] != "ok"]
    return {"status": "blocked" if blocked else "ok", "comparisons": comparisons}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=EXPERIMENTS / "plan_a_acceptance_report.json")
    ap.add_argument("--served-csv", type=Path)
    args = ap.parse_args()
    report = {"accuracy_gate": {}, "served_comparison": _served_status(args.served_csv)}
    all_pass = True
    for target, seasons in TARGETS.items():
        target_report = {}
        for season, (folder, arm) in seasons.items():
            path = EXPERIMENTS / folder / f"{target}_predictions.csv"
            if not path.exists():
                target_report[str(season)] = {"status": "missing", "source": str(path)}
                all_pass = False
                continue
            result = _evaluate_file(path, arm, season)
            bootstrap = result.get("bootstrap") or {}
            result["accuracy_pass"] = bool(
                bootstrap.get("significant_improvement") is True
                and bootstrap.get("ci_high", 1) < 0
            )
            all_pass = all_pass and result["accuracy_pass"]
            target_report[str(season)] = result
        report["accuracy_gate"][target] = target_report
    if args.served_csv and report["served_comparison"]["status"] == "ready":
        report["served_comparison"] = _compare_served(args.served_csv, report["accuracy_gate"])
    report["status"] = "accuracy_and_constraints_ready" if all_pass else "accuracy_gate_failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2, default=float))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

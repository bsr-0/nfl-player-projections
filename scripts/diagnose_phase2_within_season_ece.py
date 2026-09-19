#!/usr/bin/env python3
"""Diagnose Phase 2 QB/RB calibration by early vs late test-season weeks.

This script does not train a new calibrator. It checks whether raw hist_gbm
ECE is concentrated in weeks 1-9, which would point toward cold-start/season
transfer rather than a stable calibration correction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "player_id",
    "season",
    "week",
    "team",
    "position",
    "phase2_test_season",
    "phase2_train_max_season",
]


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    if len(y) == 0:
        return float("nan")
    clipped = np.clip(probability, 0.0, 1.0)
    bucket = np.digitize(clipped, np.linspace(0.0, 1.0, bins + 1)[1:-1], right=True)
    total = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            total += mask.sum() / len(y) * abs(clipped[mask].mean() - y[mask].mean())
    return float(total)


def summarize_rows(rows: pd.DataFrame, probability_column: str) -> dict[str, float | int]:
    y = rows["participation_actual"].to_numpy(dtype=float)
    probability = rows[probability_column].to_numpy(dtype=float)
    return {
        "n": int(len(rows)),
        "base_rate": float(np.mean(y)) if len(rows) else float("nan"),
        "mean_probability": float(np.mean(probability)) if len(rows) else float("nan"),
        "ece": ece(y, probability),
    }


def weighted_mean(records: Iterable[dict], metric: str) -> float:
    numerator = 0.0
    denominator = 0
    for record in records:
        value = record[metric]
        n = record["n"]
        if not np.isnan(value):
            numerator += value * n
            denominator += n
    return float(numerator / denominator) if denominator else float("nan")


def build_diagnostic(frame: pd.DataFrame) -> dict:
    key = KEY_COLUMNS
    gbm = frame[frame.model.eq("hist_gbm")][key + ["participation_actual", "participation_probability"]]
    gbm = gbm.rename(columns={"participation_probability": "gbm_probability"})
    prior_baseline = frame[frame.model.eq("position_rate")][key + ["participation_probability"]]
    prior_baseline = prior_baseline.rename(columns={"participation_probability": "prior_position_rate_probability"})
    paired = gbm.merge(prior_baseline, on=key, how="inner", validate="one_to_one")
    paired = paired[paired.position.isin(["QB", "RB"])].copy()
    paired["week_half"] = np.where(paired.week <= 9, "weeks_1_9", "weeks_10_18")

    fold_records: list[dict] = []
    for (position, test_season), fold in paired.groupby(["position", "phase2_test_season"], sort=True):
        week9 = fold[fold.week <= 9]
        late = fold[fold.week >= 10]
        week9_rate = float(week9.participation_actual.mean()) if len(week9) else float("nan")
        fold = fold.copy()
        fold["week9_position_rate_probability"] = week9_rate

        early_rows = fold[fold.week <= 9]
        late_rows = fold[fold.week >= 10]
        for half_name, rows, baseline_column in [
            ("weeks_1_9", early_rows, "prior_position_rate_probability"),
            ("weeks_10_18", late_rows, "week9_position_rate_probability"),
        ]:
            if rows.empty:
                continue
            gbm_summary = summarize_rows(rows, "gbm_probability")
            baseline_summary = summarize_rows(rows, baseline_column)
            fold_records.append({
                "position": position,
                "test_season": int(test_season),
                "half": half_name,
                "n": gbm_summary["n"],
                "actual_base_rate": gbm_summary["base_rate"],
                "hist_gbm_mean_probability": gbm_summary["mean_probability"],
                "hist_gbm_ece": gbm_summary["ece"],
                "baseline": (
                    "prior_training_position_rate"
                    if half_name == "weeks_1_9"
                    else "test_season_weeks_1_9_position_rate"
                ),
                "baseline_mean_probability": baseline_summary["mean_probability"],
                "baseline_ece": baseline_summary["ece"],
                "ece_difference_hist_gbm_minus_baseline": gbm_summary["ece"] - baseline_summary["ece"],
            })

    aggregates: list[dict] = []
    for (position, half), group in pd.DataFrame(fold_records).groupby(["position", "half"], sort=True):
        records = group.to_dict("records")
        aggregates.append({
            "position": position,
            "half": half,
            "n": int(group.n.sum()),
            "hist_gbm_ece": weighted_mean(records, "hist_gbm_ece"),
            "baseline_ece": weighted_mean(records, "baseline_ece"),
            "ece_difference_hist_gbm_minus_baseline": weighted_mean(
                records, "ece_difference_hist_gbm_minus_baseline"
            ),
            "actual_base_rate": float(np.average(group.actual_base_rate, weights=group.n)),
            "hist_gbm_mean_probability": float(np.average(group.hist_gbm_mean_probability, weights=group.n)),
            "baseline_mean_probability": float(np.average(group.baseline_mean_probability, weights=group.n)),
        })

    interpretation: list[dict] = []
    aggregate_frame = pd.DataFrame(aggregates)
    for position in ["QB", "RB"]:
        position_rows = aggregate_frame[aggregate_frame.position.eq(position)].set_index("half")
        if {"weeks_1_9", "weeks_10_18"}.issubset(position_rows.index):
            early = float(position_rows.loc["weeks_1_9", "hist_gbm_ece"])
            late = float(position_rows.loc["weeks_10_18", "hist_gbm_ece"])
            interpretation.append({
                "position": position,
                "early_hist_gbm_ece": early,
                "late_hist_gbm_ece": late,
                "late_minus_early_hist_gbm_ece": late - early,
                "pattern": (
                    "early_worse_late_better"
                    if early > late
                    else "late_worse_or_similar"
                ),
            })

    bin_records: list[dict] = []
    edges = np.linspace(0.0, 1.0, 11)
    paired["probability_bin"] = pd.cut(
        paired["gbm_probability"].clip(0.0, 1.0),
        bins=edges,
        include_lowest=True,
        right=True,
    )
    for (position, half, probability_bin), group in paired.groupby(
        ["position", "week_half", "probability_bin"],
        observed=False,
        sort=True,
    ):
        if group.empty:
            continue
        total = len(paired[paired.position.eq(position) & paired.week_half.eq(half)])
        predicted = float(group.gbm_probability.mean())
        observed = float(group.participation_actual.mean())
        bin_records.append({
            "position": position,
            "half": half,
            "probability_bin": str(probability_bin),
            "n": int(len(group)),
            "mass": float(len(group) / total) if total else float("nan"),
            "mean_prediction": predicted,
            "observed_rate": observed,
            "prediction_minus_observed": predicted - observed,
        })

    return {
        "description": "Raw hist_gbm ECE split by early and late test-season weeks for QB/RB.",
        "bins": 10,
        "model": "hist_gbm",
        "positions": ["QB", "RB"],
        "early_half": {
            "weeks": [1, 9],
            "baseline": "prior-training position_rate OOF prediction",
            "reason": "Avoids scoring weeks 1-9 with a baseline fit on those same weeks."
        },
        "late_half": {
            "weeks": [10, 18],
            "baseline": "position rate fit on weeks 1-9 of the same test season",
            "reason": "Valid in-season baseline available before weeks 10-18."
        },
        "aggregates": aggregates,
        "bin_diagnostics": bin_records,
        "folds": fold_records,
        "interpretation": interpretation,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="data/experiments/participation_system/phase2/oof_predictions.csv")
    parser.add_argument("--evaluation", default="data/experiments/participation_system/phase2/evaluation.json")
    parser.add_argument(
        "--diagnostic-output",
        default="data/experiments/participation_system/phase2/within_season_ece_diagnostic.json",
    )
    parser.add_argument("--decision-output", default="data/experiments/participation_system/phase2/phase2_decision.json")
    args = parser.parse_args()

    frame = pd.read_csv(args.oof)
    diagnostic = build_diagnostic(frame)
    write_json(Path(args.diagnostic_output), diagnostic)

    evaluation_path = Path(args.evaluation)
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["within_season_ece_diagnostic"] = diagnostic
    write_json(evaluation_path, evaluation)

    failed_positions = [
        item["position"]
        for item in diagnostic["interpretation"]
        if item["pattern"] != "early_worse_late_better"
    ]
    decision = {
        "phase": "phase2",
        "incumbent_model": "hist_gbm",
        "serving_change": "none",
        "calibration_promotion": "none",
        "decision": "keep_raw_hist_gbm_incumbent_and_do_not_modify_phase7",
        "diagnostic_artifact": str(Path(args.diagnostic_output)),
        "preregistration_artifact": "data/experiments/participation_system/phase2/preregistration_v2.json",
        "rationale": (
            "QB/RB calibration remains under diagnosis; no new calibrator is promoted. "
            "If QB/RB still fail after this diagnostic, raw hist_gbm should be documented as "
            "ranking-oriented for those positions and in-season recalibration should be a separate ticket."
        ),
        "diagnostic_patterns": diagnostic["interpretation"],
        "positions_without_early_worse_late_better_pattern": failed_positions,
    }
    write_json(Path(args.decision_output), decision)
    print(json.dumps({
        "diagnostic_output": args.diagnostic_output,
        "decision_output": args.decision_output,
        "aggregates": diagnostic["aggregates"],
        "interpretation": diagnostic["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()

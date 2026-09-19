#!/usr/bin/env python3
"""Bootstrap paired Phase 2 calibration differences by position.

This is a diagnostic for future pre-registered non-inferiority gates. It does
not change the existing Phase 2 acceptance decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(np.clip(probability, 0.0, 1.0), edges[1:-1], right=True)
    total = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            total += mask.sum() / len(y) * abs(probability[mask].mean() - y[mask].mean())
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="data/experiments/participation_system/phase2/oof_predictions.csv")
    parser.add_argument("--output", default="data/experiments/participation_system/phase2/calibration_bootstrap.json")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.005,
                        help="Illustrative future ECE non-inferiority margin; not applied to current gate.")
    args = parser.parse_args()

    frame = pd.read_csv(args.oof)
    key = ["player_id", "season", "week", "team", "position",
           "phase2_test_season", "phase2_train_max_season"]
    gbm = frame[frame.model.eq("hist_gbm")][key + ["participation_actual", "participation_probability"]]
    gbm = gbm.rename(columns={"participation_probability": "gbm_probability"})
    baseline = frame[frame.model.eq("position_rate")][key + ["participation_probability"]]
    baseline = baseline.rename(columns={"participation_probability": "baseline_probability"})
    paired = gbm.merge(baseline, on=key, how="inner", validate="one_to_one")
    rng = np.random.default_rng(args.seed)
    results = []
    for position, group in paired.groupby("position", sort=True):
        y = group.participation_actual.to_numpy(dtype=float)
        gbm_probability = group.gbm_probability.to_numpy(dtype=float)
        baseline_probability = group.baseline_probability.to_numpy(dtype=float)
        n = len(group)
        deltas = np.empty(args.replicates)
        for replicate in range(args.replicates):
            indices = rng.integers(0, n, n)
            deltas[replicate] = ece(y[indices], gbm_probability[indices]) - ece(
                y[indices], baseline_probability[indices]
            )
        results.append({
            "position": position,
            "n": n,
            "observed_ece_difference": float(ece(y, gbm_probability) - ece(y, baseline_probability)),
            "bootstrap_ci_95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "probability_difference_le_zero": float(np.mean(deltas <= 0)),
            "illustrative_non_inferiority_margin": args.margin,
        })
    output = {
        "metric": "ECE(hist_gbm) - ECE(position_rate)",
        "bins": 10,
        "replicates": args.replicates,
        "seed": args.seed,
        "paired_rows": int(len(paired)),
        "note": "Diagnostic only; does not alter the preregistered Phase 2 gate.",
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

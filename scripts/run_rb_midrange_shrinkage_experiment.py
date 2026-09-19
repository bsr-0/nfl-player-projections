#!/usr/bin/env python3
"""Run the preregistered RB-only mid-range shrinkage experiment.

The candidate is intentionally narrow:
- RB rows only.
- Raw hist_gbm probabilities outside 0.4-0.8 pass through unchanged.
- A single prior-season-only shrinkage weight is fit per test season.
- The correction shrinks in-band probabilities toward the prior RB base rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


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


def ece_contribution(y: np.ndarray, probability: np.ndarray, low: float, high: float) -> float:
    mask = (probability > low) & (probability <= high)
    if not mask.any():
        return 0.0
    return float(mask.sum() / len(y) * abs(probability[mask].mean() - y[mask].mean()))


def taper_weight(probability: np.ndarray) -> np.ndarray:
    weight = np.zeros(len(probability), dtype=float)
    lower = (probability >= 0.4) & (probability < 0.45)
    full = (probability >= 0.45) & (probability <= 0.75)
    upper = (probability > 0.75) & (probability <= 0.8)
    weight[lower] = (probability[lower] - 0.4) / 0.05
    weight[full] = 1.0
    weight[upper] = (0.8 - probability[upper]) / 0.05
    return np.clip(weight, 0.0, 1.0)


def fit_shrinkage_weight(y: np.ndarray, probability: np.ndarray, target_rate: float) -> float:
    weights = np.linspace(0.0, 1.0, 1001)
    best_weight = 0.0
    best_loss = float("inf")
    for weight in weights:
        candidate = (1.0 - weight) * probability + weight * target_rate
        loss = log_loss(y, np.clip(candidate, 1e-6, 1 - 1e-6), labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return best_weight


def apply_shrinkage(probability: np.ndarray, shrinkage_weight: float, target_rate: float) -> np.ndarray:
    local_weight = taper_weight(probability) * shrinkage_weight
    candidate = (1.0 - local_weight) * probability + local_weight * target_rate
    return np.clip(candidate, 0.0, 1.0)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": ece(y, probability),
    }


def bin_table(frame: pd.DataFrame, probability_column: str) -> list[dict]:
    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, 11)
    bins = pd.cut(
        frame[probability_column].clip(0.0, 1.0),
        bins=edges,
        include_lowest=True,
        right=True,
    )
    for probability_bin, group in frame.groupby(bins, observed=False, sort=True):
        if group.empty:
            continue
        predicted = float(group[probability_column].mean())
        observed = float(group.participation_actual.mean())
        rows.append({
            "probability_bin": str(probability_bin),
            "n": int(len(group)),
            "mass": float(len(group) / len(frame)),
            "mean_prediction": predicted,
            "observed_rate": observed,
            "prediction_minus_observed": predicted - observed,
            "ece_contribution": float(len(group) / len(frame) * abs(predicted - observed)),
        })
    return rows


def bootstrap_delta(y: np.ndarray, raw: np.ndarray, candidate: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(replicates)
    for replicate in range(replicates):
        indices = rng.integers(0, n, n)
        deltas[replicate] = ece(y[indices], candidate[indices]) - ece(y[indices], raw[indices])
    return {
        "observed_ece_difference_candidate_minus_hist_gbm": float(ece(y, candidate) - ece(y, raw)),
        "bootstrap_ci_95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_difference_le_zero": float(np.mean(deltas <= 0)),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default="data/experiments/participation_system/phase2/oof_predictions.csv")
    parser.add_argument(
        "--evaluation-output",
        default="data/experiments/participation_system/phase2/rb_midrange_shrinkage_evaluation.json",
    )
    parser.add_argument(
        "--bootstrap-output",
        default="data/experiments/participation_system/phase2/rb_midrange_shrinkage_bootstrap.json",
    )
    parser.add_argument(
        "--bin-output",
        default="data/experiments/participation_system/phase2/rb_midrange_shrinkage_bin_diagnostics.json",
    )
    parser.add_argument(
        "--decision-output",
        default="data/experiments/participation_system/phase2/rb_midrange_shrinkage_decision.json",
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--min-fit-rows", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_csv(args.oof)
    hist = frame[frame.model.eq("hist_gbm")].copy()
    rb = hist[hist.position.eq("RB")].copy()
    rb = rb.rename(columns={"participation_probability": "hist_gbm_probability"})
    rb["candidate_probability"] = rb["hist_gbm_probability"]
    rb["fit_mode"] = "raw_fallback_unset"
    fold_details: list[dict] = []

    for test_season in sorted(rb.phase2_test_season.unique()):
        test_mask = rb.phase2_test_season.eq(test_season)
        prior = rb[rb.phase2_test_season.lt(test_season)]
        prior_in_band = prior[
            prior.hist_gbm_probability.ge(0.4) & prior.hist_gbm_probability.le(0.8)
        ]
        test_probability = rb.loc[test_mask, "hist_gbm_probability"].to_numpy(float)
        if len(prior_in_band) < args.min_fit_rows or prior_in_band.participation_actual.nunique() < 2:
            rb.loc[test_mask, "fit_mode"] = "raw_fallback_insufficient_prior_in_band_support"
            fold_details.append({
                "test_season": int(test_season),
                "prior_in_band_rows": int(len(prior_in_band)),
                "prior_target_rate": None,
                "shrinkage_weight": 0.0,
                "fit_mode": "raw_fallback_insufficient_prior_in_band_support",
                "test_in_band_rows": int(((test_probability >= 0.4) & (test_probability <= 0.8)).sum()),
            })
            continue

        y_fit = prior_in_band.participation_actual.to_numpy(dtype=int)
        p_fit = prior_in_band.hist_gbm_probability.to_numpy(float)
        target_rate = float(y_fit.mean())
        shrinkage_weight = fit_shrinkage_weight(y_fit, p_fit, target_rate)
        candidate = apply_shrinkage(test_probability, shrinkage_weight, target_rate)
        rb.loc[test_mask, "candidate_probability"] = candidate
        rb.loc[test_mask, "fit_mode"] = "prior_oof_partial_pooling_shrinkage"
        fold_details.append({
            "test_season": int(test_season),
            "prior_in_band_rows": int(len(prior_in_band)),
            "prior_target_rate": target_rate,
            "shrinkage_weight": shrinkage_weight,
            "fit_mode": "prior_oof_partial_pooling_shrinkage",
            "test_in_band_rows": int(((test_probability >= 0.4) & (test_probability <= 0.8)).sum()),
        })

    y = rb.participation_actual.to_numpy(dtype=int)
    raw = rb.hist_gbm_probability.to_numpy(float)
    candidate = rb.candidate_probability.to_numpy(float)
    in_band = (raw >= 0.4) & (raw <= 0.8)
    outside_band = ~in_band
    top_bin = (raw > 0.9) & (raw <= 1.0)

    raw_metrics = metrics(y, raw)
    candidate_metrics = metrics(y, candidate)
    raw_in_band_metrics = metrics(y[in_band], raw[in_band])
    candidate_in_band_metrics = metrics(y[in_band], candidate[in_band])
    raw_top_contribution = ece_contribution(y, raw, 0.9, 1.0)
    candidate_top_contribution = ece_contribution(y, candidate, 0.9, 1.0)
    outside_band_max_abs_change = float(np.max(np.abs(candidate[outside_band] - raw[outside_band])))
    non_rb_unchanged = True

    evaluation = {
        "model": "hist_gbm_rb_midrange_shrinkage",
        "incumbent": "hist_gbm",
        "position": "RB",
        "probability_band": [0.4, 0.8],
        "full_strength_band": [0.45, 0.75],
        "fit_method": "prior_oof_partial_pooling_shrinkage_to_prior_rb_in_band_rate",
        "folds": fold_details,
        "fallback_folds": [row for row in fold_details if row["fit_mode"].startswith("raw_fallback")],
        "overall": {
            "hist_gbm": raw_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_hist_gbm": {
                "brier": candidate_metrics["brier"] - raw_metrics["brier"],
                "log_loss": candidate_metrics["log_loss"] - raw_metrics["log_loss"],
                "ece": candidate_metrics["ece"] - raw_metrics["ece"],
            },
        },
        "in_band": {
            "hist_gbm": raw_in_band_metrics,
            "candidate": candidate_in_band_metrics,
            "delta_candidate_minus_hist_gbm": {
                "brier": candidate_in_band_metrics["brier"] - raw_in_band_metrics["brier"],
                "log_loss": candidate_in_band_metrics["log_loss"] - raw_in_band_metrics["log_loss"],
                "ece": candidate_in_band_metrics["ece"] - raw_in_band_metrics["ece"],
            },
        },
        "guardrails": {
            "outside_band_max_abs_probability_change": outside_band_max_abs_change,
            "non_rb_rows_unchanged_by_construction": non_rb_unchanged,
            "top_bin_ece_contribution_hist_gbm": raw_top_contribution,
            "top_bin_ece_contribution_candidate": candidate_top_contribution,
            "top_bin_ece_contribution_delta": candidate_top_contribution - raw_top_contribution,
        },
    }

    bootstrap = {
        "metric": "ECE(hist_gbm_rb_midrange_shrinkage) - ECE(hist_gbm)",
        "position": "RB",
        "bins": 10,
        "replicates": args.replicates,
        "seed": args.seed,
        "non_inferiority_margin": args.margin,
        "paired_rows": int(len(rb)),
        **bootstrap_delta(y, raw, candidate, args.replicates, args.seed),
    }

    raw_bins = bin_table(rb.assign(probability=raw), "probability")
    candidate_bins = bin_table(rb.assign(probability=candidate), "probability")
    bin_diagnostics = {
        "position": "RB",
        "hist_gbm": raw_bins,
        "candidate": candidate_bins,
        "top_bin": {
            "hist_gbm_ece_contribution": raw_top_contribution,
            "candidate_ece_contribution": candidate_top_contribution,
            "delta": candidate_top_contribution - raw_top_contribution,
        },
    }

    passes = {
        "top_bin_not_worse": candidate_top_contribution - raw_top_contribution <= 0.0,
        "outside_band_unchanged": outside_band_max_abs_change <= 1e-12,
        "brier_not_worse": candidate_metrics["brier"] <= raw_metrics["brier"],
        "log_loss_not_worse": candidate_metrics["log_loss"] <= raw_metrics["log_loss"],
        "in_band_ece_improved": candidate_in_band_metrics["ece"] < raw_in_band_metrics["ece"],
        "overall_ece_non_inferior": bootstrap["bootstrap_ci_95"][1] <= args.margin,
        "no_unreported_fallbacks": True,
    }
    decision = {
        "candidate": "hist_gbm_rb_midrange_shrinkage",
        "incumbent": "hist_gbm",
        "position": "RB",
        "decision": "promote_candidate" if all(passes.values()) else "reject_candidate_keep_raw_hist_gbm",
        "phase7_predictions_modified": False,
        "passes": passes,
        "rationale": (
            "Candidate can be promoted only if all preregistered guardrails pass. "
            "Otherwise raw hist_gbm remains incumbent and RB probabilities stay flagged "
            "as uncalibrated for literal probability reads."
        ),
        "evaluation_artifact": args.evaluation_output,
        "bootstrap_artifact": args.bootstrap_output,
        "bin_diagnostics_artifact": args.bin_output,
    }

    write_json(Path(args.evaluation_output), evaluation)
    write_json(Path(args.bootstrap_output), bootstrap)
    write_json(Path(args.bin_output), bin_diagnostics)
    write_json(Path(args.decision_output), decision)
    print(json.dumps({
        "decision": decision["decision"],
        "passes": passes,
        "overall_delta": evaluation["overall"]["delta_candidate_minus_hist_gbm"],
        "in_band_delta": evaluation["in_band"]["delta_candidate_minus_hist_gbm"],
        "top_bin_delta": evaluation["guardrails"]["top_bin_ece_contribution_delta"],
        "fallback_folds": evaluation["fallback_folds"],
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Phase 2 walk-forward participation/opportunity experiments.

This script reads ``canonical_player_weeks`` and writes only experiment
artifacts.  It does not update Phase 7, production features or PPR targets.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH
from src.models.participation_opportunity import (
    FEATURES, PRIMARY_THRESHOLD, TARGET_THRESHOLDS, build_causal_features,
    expanding_season_folds, feature_columns_for_ablation, make_classifier,
    make_opportunity_regressor, predict_asof_week, target_name, TEAM_COMPETITION_MODEL_FEATURES,
    team_grouping_provenance, validate_causal_contract,
)


def calibration_rows(y: pd.Series, probability: np.ndarray, bins: int = 10) -> list[dict]:
    data = pd.DataFrame({"actual": y.astype(float).to_numpy(), "probability": probability})
    data["bin"] = pd.cut(data["probability"], np.linspace(0, 1, bins + 1),
                         include_lowest=True, duplicates="drop")
    return [
        {"bin": str(label), "n": int(len(g)), "mean_probability": float(g.probability.mean()),
         "observed_rate": float(g.actual.mean())}
        for label, g in data.groupby("bin", observed=True)
    ]


def classification_metrics(y: pd.Series, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    bins = calibration_rows(y, probability)
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, clipped)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "ece": float(sum(r["n"] * abs(r["mean_probability"] - r["observed_rate"])
                         for r in bins) / len(y)),
    }


def classification_slices(test: pd.DataFrame, target: str, probability: np.ndarray) -> list[dict]:
    """Return aggregate, position and cold-start diagnostics for one prediction."""
    scored = test[[target, "position", "cold_start"]].copy()
    scored["probability"] = probability
    masks = [("all", pd.Series(True, index=scored.index))]
    masks.extend((f"position:{p}", scored.position.eq(p)) for p in sorted(scored.position.unique()))
    masks.extend([
        ("cold_start", scored.cold_start.eq(1)),
        ("has_history", scored.cold_start.eq(0)),
    ])
    rows = []
    for segment, mask in masks:
        part = scored[mask]
        if part.empty:
            continue
        row = classification_metrics(part[target].astype(int), part.probability.to_numpy())
        row["segment"] = segment
        rows.append(row)
    return rows


def baseline_probability(train: pd.DataFrame, test: pd.DataFrame, target: str, mode: str) -> np.ndarray:
    global_rate = float(train[target].mean())
    position_rate = train.groupby("position")[target].mean().to_dict()
    fallback = test["position"].map(position_rate).fillna(global_rate).to_numpy(float)
    if mode == "position_rate":
        return fallback
    source = test[f"{target}_lag1"] if mode == "previous_game" else test[f"{target}_roll3"]
    return source.fillna(pd.Series(fallback, index=test.index)).to_numpy(float)


def position_platt_gbm_probability(
    train: pd.DataFrame, test: pd.DataFrame, target: str,
) -> tuple[np.ndarray, dict[str, str]]:
    """Score a strictly past-only, position-specific Platt-calibrated GBM.

    The latest *training* season is reserved for calibration.  A model is fit
    only on seasons before it, then its probabilities become the one-feature
    input to a logistic (Platt) calibrator.  Consequently neither the base
    estimator nor the calibrator can see labels from the held-out season.

    A rare position/fold with only one class in either sub-window falls back
    to the raw GBM fit on the complete past training data.  The fallback is
    recorded in the report rather than silently fabricating calibration.
    """
    probability = pd.Series(index=test.index, dtype=float)
    modes: dict[str, str] = {}
    for position, test_part in test.groupby("position", sort=True):
        past = train[train.position.eq(position)]
        calibration_season = int(past.season.max())
        base_train = past[past.season.lt(calibration_season)]
        calibration = past[past.season.eq(calibration_season)]
        viable = (
            not base_train.empty and not calibration.empty
            and base_train[target].nunique() == 2 and calibration[target].nunique() == 2
        )
        if viable:
            base = make_classifier("hist_gbm").fit(base_train[FEATURES], base_train[target].astype(int))
            calibration_probability = base.predict_proba(calibration[FEATURES])[:, 1]
            platt = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(
                calibration_probability.reshape(-1, 1), calibration[target].astype(int),
            )
            raw_test_probability = base.predict_proba(test_part[FEATURES])[:, 1]
            probability.loc[test_part.index] = platt.predict_proba(raw_test_probability.reshape(-1, 1))[:, 1]
            modes[str(position)] = f"platt_holdout_{calibration_season}"
        else:
            # This is still temporally valid, but is explicitly not counted as
            # calibrated evidence for this position/fold.
            raw = make_classifier("hist_gbm").fit(past[FEATURES], past[target].astype(int))
            probability.loc[test_part.index] = raw.predict_proba(test_part[FEATURES])[:, 1]
            modes[str(position)] = "raw_fallback_insufficient_calibration_classes"
    return probability.loc[test.index].to_numpy(float), modes


def position_isotonic_gbm_probability(
    train: pd.DataFrame, test: pd.DataFrame, target: str, min_calibration_rows: int = 200,
    calibration_seasons: int = 1,
) -> tuple[np.ndarray, dict[str, str], dict[str, dict]]:
    """Score a strictly past-only, position-specific isotonic-calibrated GBM.

    The latest training season is reserved for calibration.  That season is
    split chronologically: the earlier calibration-season rows fit the
    isotonic map and the later rows are held out from that fit.  The fitted
    map is then applied to the outer test season.  Isotonic is used
    because it can correct mid-range probabilities without forcing a global
    sigmoid shift onto an already-calibrated high-probability bin.  Small or
    single-class calibration windows fall back to the raw GBM and are recorded.
    """
    probability = pd.Series(index=test.index, dtype=float)
    modes: dict[str, str] = {}
    curves: dict[str, dict] = {}
    for position, test_part in test.groupby("position", sort=True):
        past = train[train.position.eq(position)]
        calibration_season = int(past.season.max())
        selected_seasons = sorted(past.season.unique())[-calibration_seasons:]
        base_train = past[~past.season.isin(selected_seasons)]
        calibration = past[past.season.isin(selected_seasons)]
        calibration = calibration.sort_values(["week", "player_id"])
        split_at = int(len(calibration) * 0.7)
        fit_calibration = calibration.iloc[:split_at]
        heldout_calibration = calibration.iloc[split_at:]
        viable = (
            len(calibration) >= min_calibration_rows
            and len(fit_calibration) >= min_calibration_rows // 2
            and not heldout_calibration.empty
            and not base_train.empty
            and fit_calibration[target].nunique() == 2
            and heldout_calibration[target].nunique() == 2
            and base_train[target].nunique() == 2
        )
        if viable:
            base = make_classifier("hist_gbm").fit(base_train[FEATURES], base_train[target].astype(int))
            calibration_probability = base.predict_proba(fit_calibration[FEATURES])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                calibration_probability, fit_calibration[target].astype(int)
            )
            raw_test_probability = base.predict_proba(test_part[FEATURES])[:, 1]
            probability.loc[test_part.index] = iso.predict(raw_test_probability)
            modes[str(position)] = (
                f"isotonic_crossfit_{selected_seasons[0]}_{selected_seasons[-1]}"
                f"_fit{len(fit_calibration)}_heldout{len(heldout_calibration)}"
            )
            curves[str(position)] = {
                "status": "fit",
                "fit_rows": int(len(fit_calibration)),
                "heldout_rows": int(len(heldout_calibration)),
                "x_thresholds": [float(v) for v in iso.X_thresholds_],
                "y_thresholds": [float(v) for v in iso.y_thresholds_],
            }
        else:
            raw = make_classifier("hist_gbm").fit(past[FEATURES], past[target].astype(int))
            probability.loc[test_part.index] = raw.predict_proba(test_part[FEATURES])[:, 1]
            modes[str(position)] = "raw_fallback_insufficient_calibration_support"
            curves[str(position)] = {
                "status": "raw_fallback_insufficient_calibration_support",
                "fit_rows": int(len(fit_calibration)),
                "heldout_rows": int(len(heldout_calibration)),
            }
    return probability.loc[test.index].to_numpy(float), modes, curves


def evaluate(frame: pd.DataFrame, min_train_seasons: int = 3) -> tuple[pd.DataFrame, dict]:
    validate_causal_contract(frame)
    observed = frame[frame["label_observed"].eq(1)].copy()
    predictions: list[pd.DataFrame] = []
    report: dict = {"contract": {
        "unknown_rows_are_negatives": False,
        "features": FEATURES,
        "primary_threshold": PRIMARY_THRESHOLD,
        "production_ppr_modified": False,
        "team_competition_feature_columns": TEAM_COMPETITION_MODEL_FEATURES,
    }, "team_grouping_provenance": team_grouping_provenance(frame), "coverage": {},
        "classification": [], "opportunity": [], "ablations": []}

    coverage = frame.groupby(["season", "position"], dropna=False).agg(
        rows=("player_id", "size"), observed_labels=("label_observed", "sum")
    ).reset_index()
    coverage["coverage"] = coverage.observed_labels / coverage.rows
    report["coverage"] = coverage.to_dict("records")

    for fold in expanding_season_folds(observed, min_train_seasons):
        train, test = observed.loc[fold.train_index], observed.loc[fold.test_index]
        fold_predictions: list[pd.DataFrame] = []
        for threshold in TARGET_THRESHOLDS:
            target = target_name(threshold)
            y_train, y_test = train[target].astype(int), test[target].astype(int)
            if y_train.nunique() < 2 or y_test.empty:
                continue
            for baseline in ("position_rate", "previous_game", "rolling3"):
                p = baseline_probability(train, test, target, baseline)
                for row in classification_slices(test, target, p):
                    row.update({"season": fold.test_season, "threshold": threshold, "model": baseline})
                    if row["segment"] == "all":
                        row["calibration"] = calibration_rows(y_test, p)
                    report["classification"].append(row)
                if baseline == "position_rate" and threshold == PRIMARY_THRESHOLD:
                    pred = test[["player_id", "season", "week", "team", "position"]].copy()
                    pred["participation_probability"] = p
                    pred["participation_actual"] = y_test.to_numpy()
                    pred["model"] = baseline
                    pred["phase2_test_season"] = fold.test_season
                    pred["phase2_train_max_season"] = int(train["season"].max())
                    predictions.append(pred)
                    fold_predictions.append(pred)

            candidate_probabilities = {}
            for kind in ("logistic", "hist_gbm"):
                model = make_classifier(kind)
                model.fit(train[FEATURES], y_train)
                candidate_probabilities[kind] = model.predict_proba(test[FEATURES])[:, 1]
            team_model = make_classifier("hist_gbm", feature_columns=TEAM_COMPETITION_MODEL_FEATURES)
            team_model.fit(train[TEAM_COMPETITION_MODEL_FEATURES], y_train)
            candidate_probabilities["hist_gbm_team_competition"] = team_model.predict_proba(
                test[TEAM_COMPETITION_MODEL_FEATURES]
            )[:, 1]
            calibrated_p, calibration_modes = position_platt_gbm_probability(train, test, target)
            candidate_probabilities["hist_gbm_platt_position"] = calibrated_p
            isotonic_p, isotonic_modes, isotonic_curves = position_isotonic_gbm_probability(train, test, target)
            candidate_probabilities["hist_gbm_isotonic_position"] = isotonic_p
            pooled_isotonic_p, pooled_isotonic_modes, pooled_isotonic_curves = position_isotonic_gbm_probability(
                train, test, target, calibration_seasons=3
            )
            candidate_probabilities["hist_gbm_isotonic_pooled3"] = pooled_isotonic_p
            for kind, p in candidate_probabilities.items():
                for row in classification_slices(test, target, p):
                    row.update({"season": fold.test_season, "threshold": threshold, "model": kind})
                    if kind == "hist_gbm_platt_position":
                        row["calibration_modes"] = calibration_modes
                    if kind == "hist_gbm_isotonic_position":
                        row["calibration_modes"] = isotonic_modes
                        if row["segment"] == "all":
                            row["isotonic_curves"] = isotonic_curves
                    if kind == "hist_gbm_isotonic_pooled3":
                        row["calibration_modes"] = pooled_isotonic_modes
                        if row["segment"] == "all":
                            row["isotonic_curves"] = pooled_isotonic_curves
                    if row["segment"] == "all":
                        row["calibration"] = calibration_rows(y_test, p)
                    report["classification"].append(row)
                if threshold == PRIMARY_THRESHOLD:
                    pred = test[["player_id", "season", "week", "team", "position"]].copy()
                    pred["participation_probability"] = p
                    pred["participation_actual"] = y_test.to_numpy()
                    pred["model"] = kind
                    pred["phase2_test_season"] = fold.test_season
                    pred["phase2_train_max_season"] = int(train["season"].max())
                    predictions.append(pred)
                    fold_predictions.append(pred)

        # Conditional opportunity uses only primary-positive rows.  Evaluation
        # is therefore conditional too; the unconditional downstream product
        # can later be formed from P(play) * E(share | play).
        primary = target_name(PRIMARY_THRESHOLD)
        opp_train = train[train[primary].eq(1)]
        opp_test = test[test[primary].eq(1)]
        if len(opp_train) >= 100 and not opp_test.empty:
            baseline = opp_test["snap_share_roll3"].fillna(opp_train.snap_share.median()).clip(0, 1)
            reg = make_opportunity_regressor().fit(opp_train[FEATURES], opp_train.snap_share)
            all_opp_p = np.clip(reg.predict(test[FEATURES]), 0, 1)
            team_reg = make_opportunity_regressor(feature_columns=TEAM_COMPETITION_MODEL_FEATURES).fit(
                opp_train[TEAM_COMPETITION_MODEL_FEATURES], opp_train.snap_share
            )
            all_team_opp_p = np.clip(team_reg.predict(test[TEAM_COMPETITION_MODEL_FEATURES]), 0, 1)
            opp_p = pd.Series(all_opp_p, index=test.index).loc[opp_test.index]
            team_opp_p = pd.Series(all_team_opp_p, index=test.index).loc[opp_test.index]
            segments = [("all", opp_test)] + [
                (f"position:{p}", opp_test[opp_test.position.eq(p)])
                for p in sorted(opp_test.position.unique())
            ] + [("cold_start", opp_test[opp_test.cold_start.eq(1)]),
                 ("has_history", opp_test[opp_test.cold_start.eq(0)])]
            for kind, prediction in (("rolling3", baseline), ("hist_gbm_mae", opp_p),
                                     ("hist_gbm_team_competition_mae", team_opp_p)):
                for segment, subset in segments:
                    if subset.empty:
                        continue
                    segment_p = prediction.loc[subset.index]
                    report["opportunity"].append({
                        "season": fold.test_season, "model": kind, "segment": segment,
                        "n": len(subset), "mae": float(mean_absolute_error(subset.snap_share, segment_p)),
                        "rmse": float(mean_squared_error(subset.snap_share, segment_p) ** 0.5),
                    })
            for pred in fold_predictions:
                conditional = (all_team_opp_p if pred["model"].iloc[0] == "hist_gbm_team_competition"
                               else all_opp_p)
                pred["conditional_snap_share"] = conditional
                pred["expected_snap_share"] = (
                    pred["participation_probability"] * pred["conditional_snap_share"]
                )

        # Primary-target ablations use logistic regression to isolate feature
        # value without conflating each addition with model-family changes.
        y_train, y_test = train[primary].astype(int), test[primary].astype(int)
        if y_train.nunique() >= 2:
            for ablation in ("history", "usage", "role"):
                cols = feature_columns_for_ablation(ablation)
                base = make_classifier("logistic")
                # Rebuild preprocessing through column selection by blanking
                # excluded values; the declared model schema remains stable.
                train_x, test_x = train[FEATURES].copy(), test[FEATURES].copy()
                for col in set(FEATURES) - set(cols):
                    train_x[col] = np.nan if col in train_x.select_dtypes(exclude="object") else "EXCLUDED"
                    test_x[col] = np.nan if col in test_x.select_dtypes(exclude="object") else "EXCLUDED"
                base.fit(train_x, y_train)
                p = base.predict_proba(test_x)[:, 1]
                row = classification_metrics(y_test, p)
                row.update({"season": fold.test_season, "ablation": ablation})
                report["ablations"].append(row)

    return (pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()), report


def summarize(report: dict) -> dict:
    out = {}
    for section, keys in (("classification", ["threshold", "model", "segment"]),
                          ("opportunity", ["model", "segment"]), ("ablations", ["ablation"])):
        data = pd.DataFrame(report[section])
        if data.empty:
            out[section] = []
            continue
        metrics = [c for c in ("brier", "log_loss", "ece", "mae", "rmse") if c in data]
        out[section] = data.groupby(keys, dropna=False)[metrics].mean().reset_index().to_dict("records")
    return out


def add_kickoff_filtered_injury_score(panel: pd.DataFrame) -> pd.DataFrame:
    """Reuse the one repository-authoritative pregame injury-cache guard."""
    from src.features.feature_engineering import FeatureEngineer
    return FeatureEngineer()._merge_injury_data_from_cache(panel.copy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--min-train-seasons", type=int, default=3)
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "experiments" / "phase2")
    ap.add_argument("--predict-season", type=int, default=None,
                    help="also write Phase 2 predictions for a target season/week")
    ap.add_argument("--predict-week", type=int, default=1,
                    help="target week for --predict-season; Week 1 is the pre-season path")
    ap.add_argument("--include-pregame-injury", action="store_true",
                    help="include only kickoff-filtered player_injuries cache scores as an optional ablation")
    args = ap.parse_args()
    with sqlite3.connect(args.db) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='canonical_player_weeks'").fetchone()
        if not exists:
            raise SystemExit("canonical_player_weeks is missing; run Phase 1 builder with --write first")
        panel = pd.read_sql("SELECT * FROM canonical_player_weeks", conn)

    if args.include_pregame_injury:
        panel = add_kickoff_filtered_injury_score(panel)
    frame = build_causal_features(panel, include_pregame_injury=args.include_pregame_injury)
    predictions, report = evaluate(frame, args.min_train_seasons)
    report["summary"] = summarize(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    manifest = {
        "system": "participation_opportunity", "phase": 2,
        "panel_source": "canonical_player_weeks", "oof_file": "oof_predictions.csv",
        "oof_provenance_columns": ["phase2_test_season", "phase2_train_max_season"],
        "feature_columns": FEATURES,
        "include_pregame_injury": bool(args.include_pregame_injury),
        "target_thresholds": list(TARGET_THRESHOLDS),
        "primary_threshold": PRIMARY_THRESHOLD,
        "min_train_seasons": args.min_train_seasons,
        "classifier_candidates": [
            "logistic", "hist_gbm", "hist_gbm_platt_position", "hist_gbm_isotonic_position",
            "hist_gbm_isotonic_pooled3",
            "hist_gbm_team_competition",
        ],
        "team_competition_feature_columns": TEAM_COMPETITION_MODEL_FEATURES,
    }
    (args.output_dir / "phase2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.predict_season is not None:
        future = predict_asof_week(frame, args.predict_season, args.predict_week)
        future.to_csv(args.output_dir / f"predictions_{args.predict_season}_week_{args.predict_week}.csv", index=False)
        print(f"wrote {len(future):,} strictly as-of prediction rows for {args.predict_season} week {args.predict_week}")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {len(predictions):,} OOF prediction rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

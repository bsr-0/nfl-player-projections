#!/usr/bin/env python3
"""
Audit: 2025 Backtest Pipeline Data Leakage & Distribution Shift

Diagnoses why the 2025 backtest collapsed (R²=-0.685, +3.6pt over-prediction)
versus the 2024 backtest (R²=0.233, well-calibrated).

Runs 5 checks:
  1. Percentile bounds validation (zero-width bounds detection)
  2. Utilization score distribution shift (train vs test)
  3. Feature-level distribution comparison
  4. Player-level trace (verify no temporal leakage)
  5. Naive baseline comparison (does the model beat "last year's average"?)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import POSITIONS, MODELS_DIR, DATA_DIR, MODEL_CONFIG
from src.utils.database import DatabaseManager
from src.features.utilization_score import (
    UtilizationScoreCalculator,
    load_percentile_bounds,
    calculate_utilization_scores,
)
from src.features.feature_engineering import FeatureEngineer

REPORT_PATH = DATA_DIR / "audit_2025_backtest_report.json"
BOUNDS_PATH = MODELS_DIR / "utilization_percentile_bounds.json"
WEIGHTS_PATH = MODELS_DIR / "utilization_weights.json"
TOP_FEATURES_PATH = MODELS_DIR / "top10_features_per_position.json"
BACKTEST_2025_PATH = DATA_DIR / "backtest_results" / "backtest_2025_20260215.json"


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ── Check 1: Percentile Bounds Validation ──────────────────────────────────

def check_percentile_bounds() -> dict:
    _header("CHECK 1: Percentile Bounds Validation")

    with open(BOUNDS_PATH) as f:
        raw_bounds = json.load(f)
    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)

    # Map component names to weight keys
    # bounds keys: "WR|snap_share_pct" -> weight key: "snap_share"
    def _component_to_weight_key(component: str) -> str:
        return (
            component.replace("_pct", "")
            .replace("_share", "")
            .replace("_rate", "")
            .replace("redzone_opp", "redzone_opportunity")
            .replace("redzone_targets", "redzone_targets")
        )

    zero_width = []
    full_range = []
    valid = []

    for key, (lo, hi) in raw_bounds.items():
        if key.startswith("__"):
            continue
        pos, component = key.split("|", 1)
        width = hi - lo

        if abs(width) < 1e-9:
            zero_width.append({"key": key, "position": pos, "component": component, "lo": lo, "hi": hi})
        elif width >= 99.0:
            full_range.append({"key": key, "position": pos, "component": component, "lo": lo, "hi": hi})
        else:
            valid.append({"key": key, "position": pos, "component": component, "lo": lo, "hi": hi, "width": width})

    # Compute weighted impact of corrupted bounds per position
    impact = {}
    for pos in POSITIONS:
        pos_weights = weights.get(pos, {})
        total_weight = sum(pos_weights.values())
        corrupted_weight = 0.0
        corrupted_components = []
        for entry in zero_width:
            if entry["position"] != pos:
                continue
            # Try to find matching weight key
            comp = entry["component"]
            for wk, wv in pos_weights.items():
                if wk in comp or comp.startswith(wk):
                    corrupted_weight += wv
                    corrupted_components.append(f"{wk}={wv:.3f}")
                    break
        impact[pos] = {
            "total_weight": round(total_weight, 3),
            "corrupted_weight": round(corrupted_weight, 3),
            "corrupted_pct": round(corrupted_weight / total_weight * 100, 1) if total_weight > 0 else 0,
            "corrupted_components": corrupted_components,
        }

    # Print results
    print(f"\nZero-width bounds ({len(zero_width)} found):")
    for z in zero_width:
        print(f"  FAIL  {z['key']}: [{z['lo']}, {z['hi']}]")

    print(f"\nFull-range bounds ({len(full_range)} found):")
    for f_entry in full_range:
        print(f"  WARN  {f_entry['key']}: [{f_entry['lo']}, {f_entry['hi']}] (passthrough, no normalization)")

    print(f"\nValid bounds ({len(valid)} found):")
    for v in valid:
        print(f"  OK    {v['key']}: [{v['lo']:.1f}, {v['hi']:.1f}] (width={v['width']:.1f})")

    print(f"\nWeighted impact of zero-width bounds per position:")
    for pos, info in impact.items():
        status = "FAIL" if info["corrupted_pct"] > 5 else "OK"
        print(f"  {status}  {pos}: {info['corrupted_pct']}% of utilization weight corrupted {info['corrupted_components']}")

    return {
        "zero_width_count": len(zero_width),
        "full_range_count": len(full_range),
        "valid_count": len(valid),
        "zero_width": zero_width,
        "full_range": full_range,
        "impact_by_position": impact,
    }


# ── Check 2: Utilization Score Distribution Shift ──────────────────────────

def check_utilization_distribution(db: DatabaseManager) -> dict:
    _header("CHECK 2: Utilization Score Distribution (Train vs Test)")

    # Load raw data
    all_data = []
    for pos in POSITIONS:
        d = db.get_all_players_for_training(position=pos, min_games=1)
        if len(d) > 0:
            all_data.append(d)
    if not all_data:
        print("  ERROR: No data loaded")
        return {"error": "no data"}

    combined = pd.concat(all_data, ignore_index=True)
    train = combined[combined["season"] <= 2024].copy()
    test = combined[combined["season"] == 2025].copy()

    print(f"  Train: {len(train)} rows (seasons <= 2024)")
    print(f"  Test:  {len(test)} rows (season 2025)")

    if test.empty:
        print("  WARNING: No 2025 data found in database")
        return {"error": "no 2025 data"}

    # Calculate utilization with persisted bounds
    bounds = load_percentile_bounds(BOUNDS_PATH)
    train_util = calculate_utilization_scores(train.copy(), weights=None, percentile_bounds=bounds)
    test_util = calculate_utilization_scores(test.copy(), weights=None, percentile_bounds=bounds)

    # Calculate utilization WITHOUT bounds (rank-based fallback)
    train_util_rank = calculate_utilization_scores(train.copy(), weights=None, percentile_bounds=None)
    test_util_rank = calculate_utilization_scores(test.copy(), weights=None, percentile_bounds=None)

    results = {}
    print(f"\n  {'Position':<6} {'Split':<7} {'Method':<12} {'Mean':>6} {'Std':>6} {'P25':>6} {'P50':>6} {'P75':>6}")
    print("  " + "-" * 60)

    for pos in POSITIONS:
        pos_results = {}
        for label, df_b, df_r in [
            ("train", train_util, train_util_rank),
            ("test", test_util, test_util_rank),
        ]:
            mask = df_b["position"] == pos
            s_bounds = df_b.loc[mask, "utilization_score"].dropna()
            s_rank = df_r.loc[mask, "utilization_score"].dropna()

            for method, s in [("bounds", s_bounds), ("rank", s_rank)]:
                stats = {
                    "mean": round(float(s.mean()), 2) if len(s) > 0 else None,
                    "std": round(float(s.std()), 2) if len(s) > 0 else None,
                    "p25": round(float(s.quantile(0.25)), 2) if len(s) > 0 else None,
                    "p50": round(float(s.quantile(0.50)), 2) if len(s) > 0 else None,
                    "p75": round(float(s.quantile(0.75)), 2) if len(s) > 0 else None,
                    "n": len(s),
                }
                pos_results[f"{label}_{method}"] = stats
                if stats["mean"] is not None:
                    print(f"  {pos:<6} {label:<7} {method:<12} {stats['mean']:>6.1f} {stats['std']:>6.1f} {stats['p25']:>6.1f} {stats['p50']:>6.1f} {stats['p75']:>6.1f}")

        # Distribution shift metrics
        train_mean = pos_results.get("train_bounds", {}).get("mean")
        test_mean = pos_results.get("test_bounds", {}).get("mean")
        train_std = pos_results.get("train_bounds", {}).get("std")
        if train_mean and test_mean and train_std and train_std > 0:
            shift = (test_mean - train_mean) / train_std
            pos_results["mean_shift_z"] = round(shift, 2)
            status = "FAIL" if abs(shift) > 0.5 else ("WARN" if abs(shift) > 0.25 else "OK")
            print(f"  {pos:<6} → Mean shift (z-score): {shift:+.2f} [{status}]")

        results[pos] = pos_results
        print()

    return results


# ── Check 3: Feature-Level Distribution Comparison ─────────────────────────

def check_feature_distributions(db: DatabaseManager) -> dict:
    _header("CHECK 3: Feature-Level Distribution Comparison")

    # Load top features per position
    try:
        with open(TOP_FEATURES_PATH) as f:
            top_features_raw = json.load(f)
    except FileNotFoundError:
        print("  WARNING: top10_features_per_position.json not found, using defaults")
        top_features_raw = {}

    top_features = {}
    for pos, entries in top_features_raw.items():
        top_features[pos] = [e["feature"] for e in entries[:15]]

    # Load data and run feature engineering
    all_data = []
    for pos in POSITIONS:
        d = db.get_all_players_for_training(position=pos, min_games=1)
        if len(d) > 0:
            all_data.append(d)
    combined = pd.concat(all_data, ignore_index=True)
    train = combined[combined["season"] <= 2024].copy()
    test = combined[combined["season"] == 2025].copy()

    if test.empty:
        return {"error": "no 2025 data"}

    # Run feature engineering
    fe = FeatureEngineer()
    train_feat = fe.create_features(train, include_target=True)
    # For test, combine with train for temporal context then extract test rows
    combined_sorted = pd.concat([train, test], ignore_index=True).sort_values(
        ["player_id", "season", "week"]
    ).reset_index(drop=True)
    combined_feat = fe.create_features(combined_sorted, include_target=True)
    test_feat = combined_feat[combined_feat["season"] == 2025].copy()

    results = {}
    print(f"\n  {'Position':<5} {'Feature':<35} {'Train Mean':>10} {'Test Mean':>10} {'Shift(z)':>9} {'Status':>7}")
    print("  " + "-" * 80)

    for pos in POSITIONS:
        pos_train = train_feat[train_feat["position"] == pos]
        pos_test = test_feat[test_feat["position"] == pos]
        features = top_features.get(pos, [])
        pos_results = []

        for feat in features:
            if feat not in pos_train.columns or feat not in pos_test.columns:
                continue
            tr = pd.to_numeric(pos_train[feat], errors="coerce").dropna()
            te = pd.to_numeric(pos_test[feat], errors="coerce").dropna()
            if len(tr) < 10 or len(te) < 10:
                continue

            tr_mean, tr_std = float(tr.mean()), float(tr.std())
            te_mean = float(te.mean())
            shift_z = (te_mean - tr_mean) / tr_std if tr_std > 0 else 0
            var_ratio = float(te.std()) / tr_std if tr_std > 0 else 1

            status = "FAIL" if abs(shift_z) > 1.0 else ("WARN" if abs(shift_z) > 0.5 else "OK")
            print(f"  {pos:<5} {feat:<35} {tr_mean:>10.2f} {te_mean:>10.2f} {shift_z:>+9.2f} {status:>7}")

            pos_results.append({
                "feature": feat,
                "train_mean": round(tr_mean, 3),
                "test_mean": round(te_mean, 3),
                "shift_z": round(shift_z, 3),
                "var_ratio": round(var_ratio, 3),
                "status": status,
            })

        results[pos] = pos_results

    return results


# ── Check 4: Player-Level Trace ────────────────────────────────────────────

def check_player_trace(db: DatabaseManager) -> dict:
    _header("CHECK 4: Player-Level Temporal Leakage Trace")

    all_data = []
    for pos in POSITIONS:
        d = db.get_all_players_for_training(position=pos, min_games=1)
        if len(d) > 0:
            all_data.append(d)
    combined = pd.concat(all_data, ignore_index=True)

    # Find players with data in both 2024 and 2025, pick top 3 by 2024 games per position
    results = {}
    for pos in POSITIONS:
        pos_df = combined[combined["position"] == pos]
        has_2024 = set(pos_df[pos_df["season"] == 2024]["player_id"].unique())
        has_2025 = set(pos_df[pos_df["season"] == 2025]["player_id"].unique())
        both = has_2024 & has_2025

        if not both:
            print(f"  {pos}: No players with data in both 2024 and 2025")
            results[pos] = {"error": "no overlapping players"}
            continue

        # Pick top 3 by 2024 fantasy points total
        candidates = pos_df[(pos_df["player_id"].isin(both)) & (pos_df["season"] == 2024)]
        top_players = (
            candidates.groupby("player_id")["fantasy_points"]
            .sum()
            .nlargest(3)
            .index.tolist()
        )

        pos_results = []
        for pid in top_players:
            player_data = pos_df[pos_df["player_id"] == pid].sort_values(["season", "week"])
            name = player_data["name"].iloc[0] if "name" in player_data.columns else pid

            # Get Week 1 of 2025
            w1_2025 = player_data[(player_data["season"] == 2025) & (player_data["week"] == 1)]
            if w1_2025.empty:
                # Try earliest 2025 week
                w1_2025 = player_data[player_data["season"] == 2025].head(1)
            if w1_2025.empty:
                continue

            week_num = int(w1_2025["week"].iloc[0])
            actual_fp = float(w1_2025["fantasy_points"].iloc[0]) if "fantasy_points" in w1_2025.columns else None

            # Run feature engineering on this player's full history up to this point
            # to verify rolling features only use past data
            hist = player_data[
                (player_data["season"] < 2025) |
                ((player_data["season"] == 2025) & (player_data["week"] <= week_num))
            ].copy()

            fe = FeatureEngineer()
            feat = fe.create_features(hist, include_target=False)
            row = feat[(feat["season"] == 2025) & (feat["week"] == week_num)]

            if row.empty:
                continue

            # Check rolling features: they should be based on 2024 data only
            trace = {"player_id": pid, "name": name, "week": week_num}

            # Get last 2024 game for comparison
            last_2024 = player_data[player_data["season"] == 2024].tail(1)

            rolling_cols = [c for c in row.columns if "roll" in c and "mean" in c][:5]
            lag_cols = [c for c in row.columns if "lag" in c][:5]

            trace["actual_fp"] = actual_fp
            trace["rolling_features"] = {}
            for rc in rolling_cols:
                raw = row[rc].iloc[0]
                try:
                    val = float(raw) if pd.notna(raw) else None
                except (ValueError, TypeError):
                    val = None
                trace["rolling_features"][rc] = val

            trace["lag_features"] = {}
            for lc in lag_cols:
                raw = row[lc].iloc[0]
                try:
                    val = float(raw) if pd.notna(raw) else None
                except (ValueError, TypeError):
                    val = None
                trace["lag_features"][lc] = val

            # Verify: rolling features should NOT equal the current week's raw stats
            # (that would indicate leakage)
            leakage_flags = []
            if actual_fp is not None:
                for rc in rolling_cols:
                    val = trace["rolling_features"].get(rc)
                    if val is not None and abs(val - actual_fp) < 0.01:
                        leakage_flags.append(f"{rc} == actual_fp ({val})")

            trace["leakage_flags"] = leakage_flags
            status = "FAIL" if leakage_flags else "OK"

            print(f"  {status}  {pos} {name} (Week {week_num}, 2025)")
            print(f"        Actual FP: {actual_fp}")
            for rc in rolling_cols[:3]:
                val = trace["rolling_features"].get(rc)
                print(f"        {rc}: {val}")
            for lc in lag_cols[:3]:
                val = trace["lag_features"].get(lc)
                print(f"        {lc}: {val}")
            if leakage_flags:
                for flag in leakage_flags:
                    print(f"        LEAKAGE: {flag}")
            print()

            pos_results.append(trace)

        results[pos] = pos_results

    return results


# ── Check 5: Naive Baseline Comparison ─────────────────────────────────────

def check_naive_baseline(db: DatabaseManager) -> dict:
    _header("CHECK 5: Naive Baseline vs Model (2025)")

    # Load backtest results
    try:
        with open(BACKTEST_2025_PATH) as f:
            backtest = json.load(f)
    except FileNotFoundError:
        print("  WARNING: 2025 backtest results not found")
        return {"error": "no backtest file"}

    # Load raw data for baseline computation
    all_data = []
    for pos in POSITIONS:
        d = db.get_all_players_for_training(position=pos, min_games=1)
        if len(d) > 0:
            all_data.append(d)
    combined = pd.concat(all_data, ignore_index=True)

    # Baseline: each player's 2024 PPG applied to every 2025 week
    season_2024 = combined[combined["season"] == 2024]
    season_2025 = combined[combined["season"] == 2025]

    if season_2025.empty:
        print("  No 2025 data available")
        return {"error": "no 2025 data"}

    ppg_2024 = season_2024.groupby("player_id")["fantasy_points"].mean()

    # Merge baseline predictions onto 2025 data
    baseline = season_2025[["player_id", "position", "fantasy_points"]].copy()
    baseline["baseline_pred"] = baseline["player_id"].map(ppg_2024)
    # Players without 2024 data: use position average
    pos_avg_2024 = season_2024.groupby("position")["fantasy_points"].mean()
    for pos in POSITIONS:
        mask = (baseline["position"] == pos) & baseline["baseline_pred"].isna()
        baseline.loc[mask, "baseline_pred"] = pos_avg_2024.get(pos, 8.0)

    baseline = baseline.dropna(subset=["fantasy_points", "baseline_pred"])

    results = {"overall": {}, "by_position": {}}

    # Overall baseline MAE
    baseline_mae = float(np.abs(baseline["fantasy_points"] - baseline["baseline_pred"]).mean())
    model_mae = backtest["metrics"]["mae"]
    model_avg_pred = backtest["metrics"]["avg_predicted"]
    model_avg_actual = backtest["metrics"]["avg_actual"]
    baseline_avg_pred = float(baseline["baseline_pred"].mean())
    baseline_bias = baseline_avg_pred - float(baseline["fantasy_points"].mean())

    results["overall"] = {
        "baseline_mae": round(baseline_mae, 2),
        "model_mae": model_mae,
        "model_beats_baseline": model_mae < baseline_mae,
        "baseline_avg_pred": round(baseline_avg_pred, 2),
        "baseline_bias": round(baseline_bias, 2),
        "model_bias": round(model_avg_pred - model_avg_actual, 2),
    }

    print(f"\n  {'Metric':<25} {'Model':>10} {'Baseline':>10} {'Winner':>10}")
    print("  " + "-" * 58)
    winner = "Model" if model_mae < baseline_mae else "Baseline"
    print(f"  {'MAE':<25} {model_mae:>10.2f} {baseline_mae:>10.2f} {winner:>10}")
    print(f"  {'Avg Predicted':<25} {model_avg_pred:>10.2f} {baseline_avg_pred:>10.2f}")
    print(f"  {'Avg Actual':<25} {model_avg_actual:>10.2f} {float(baseline['fantasy_points'].mean()):>10.2f}")
    print(f"  {'Bias (pred-actual)':<25} {model_avg_pred - model_avg_actual:>+10.2f} {baseline_bias:>+10.2f}")

    # Per-position
    print(f"\n  {'Position':<6} {'Model MAE':>10} {'Base MAE':>10} {'Model Bias':>11} {'Base Bias':>11} {'Winner':>10}")
    print("  " + "-" * 62)

    for pos in POSITIONS:
        pos_baseline = baseline[baseline["position"] == pos]
        if pos_baseline.empty:
            continue
        b_mae = float(np.abs(pos_baseline["fantasy_points"] - pos_baseline["baseline_pred"]).mean())
        b_bias = float(pos_baseline["baseline_pred"].mean() - pos_baseline["fantasy_points"].mean())

        pos_backtest = backtest.get("by_position", {}).get(pos, {})
        m_mae = pos_backtest.get("mae", None)
        m_bias = (pos_backtest.get("avg_predicted", 0) - pos_backtest.get("avg_actual", 0))

        if m_mae is not None:
            winner = "Model" if m_mae < b_mae else "Baseline"
            print(f"  {pos:<6} {m_mae:>10.2f} {b_mae:>10.2f} {m_bias:>+11.2f} {b_bias:>+11.2f} {winner:>10}")
            results["by_position"][pos] = {
                "model_mae": m_mae,
                "baseline_mae": round(b_mae, 2),
                "model_bias": round(m_bias, 2),
                "baseline_bias": round(b_bias, 2),
                "model_beats_baseline": m_mae < b_mae,
            }

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  2025 BACKTEST PIPELINE AUDIT")
    print("  Diagnosing R²=-0.685, +3.6pt systematic over-prediction")
    print("=" * 60)

    db = DatabaseManager()
    report = {}

    report["check_1_percentile_bounds"] = check_percentile_bounds()
    report["check_2_utilization_distribution"] = check_utilization_distribution(db)
    report["check_3_feature_distributions"] = check_feature_distributions(db)
    report["check_4_player_trace"] = check_player_trace(db)
    report["check_5_naive_baseline"] = check_naive_baseline(db)

    # Summary
    _header("AUDIT SUMMARY")

    c1 = report["check_1_percentile_bounds"]
    print(f"  Check 1 (Bounds):       {c1['zero_width_count']} zero-width, {c1['full_range_count']} full-range")

    c2 = report["check_2_utilization_distribution"]
    shifts = []
    for pos in POSITIONS:
        z = c2.get(pos, {}).get("mean_shift_z")
        if z is not None:
            shifts.append(f"{pos}={z:+.2f}")
    print(f"  Check 2 (Util Shift):   {', '.join(shifts) if shifts else 'N/A'}")

    c3 = report["check_3_feature_distributions"]
    fail_count = sum(
        1 for pos_feats in c3.values()
        if isinstance(pos_feats, list)
        for f in pos_feats
        if f.get("status") == "FAIL"
    )
    warn_count = sum(
        1 for pos_feats in c3.values()
        if isinstance(pos_feats, list)
        for f in pos_feats
        if f.get("status") == "WARN"
    )
    print(f"  Check 3 (Features):     {fail_count} FAIL, {warn_count} WARN")

    c4 = report["check_4_player_trace"]
    leakage_found = any(
        flag
        for pos_players in c4.values()
        if isinstance(pos_players, list)
        for p in pos_players
        for flag in p.get("leakage_flags", [])
    )
    print(f"  Check 4 (Leakage):      {'LEAKAGE DETECTED' if leakage_found else 'No temporal leakage found'}")

    c5 = report["check_5_naive_baseline"]
    if "overall" in c5:
        beats = c5["overall"].get("model_beats_baseline", None)
        print(f"  Check 5 (Baseline):     Model {'beats' if beats else 'LOSES to'} naive baseline")

    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

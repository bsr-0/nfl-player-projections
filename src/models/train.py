"""Training script for NFL prediction models."""
import argparse
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Suppress SciPy/NumPy version mismatch warning (env may have numpy>=1.23 with older scipy)
warnings.filterwarnings(
    "ignore",
    message=".*NumPy version.*required for this version of SciPy.*",
    category=UserWarning,
    module="scipy",
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    POSITIONS,
    MODELS_DIR,
    DATA_DIR,
    MODEL_CONFIG,
    FAST_MODEL_CONFIG,
    QB_TARGET_CHOICE_FILENAME,
    FEATURE_VERSION,
    FEATURE_VERSION_FILENAME,
    MIN_TRAINING_SEASONS_1W,
    MIN_TRAINING_SEASONS_18W,
    MIN_TRAINING_SEASONS_4W,
    MIN_PLAYERS_PER_POSITION,
    RETRAINING_CONFIG,
)
from src.utils.database import DatabaseManager
from src.utils.data_manager import DataManager, auto_refresh_data
from src.features.feature_engineering import FeatureEngineer, PositionFeatureEngineer
from src.features.utilization_score import (
    calculate_utilization_scores,
    recalculate_utilization_with_weights,
    UtilizationScoreCalculator,
    save_percentile_bounds,
    load_percentile_bounds,
    validate_percentile_bounds_meta,
)
from src.features.utilization_weight_optimizer import fit_utilization_weights, UTIL_COMPONENTS
from src.features.dimensionality_reduction import PositionDimensionalityReducer
from src.models.ensemble import ModelTrainer
from src.models.robust_validation import RobustTimeSeriesCV
from src.evaluation.backtester import ModelBacktester
from src.data.lineage import (
    find_artifact_ids,
    get_artifact_id,
    persist_dataframe_artifact,
    set_artifact_id,
    utc_now_iso,
)
from src.models.utilization_to_fp import train_utilization_to_fp_per_position
from src.data.quality_gates import run_quality_gates, validate_training_cache_integrity
from src.evaluation.explainability import (
    get_top10_feature_importance_per_position,
    explain_with_shap,
    partial_dependence_plots,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from src.models.data_loading import load_training_data, create_sample_data  # noqa: F401
from src.models.data_quality import (  # noqa: F401
    _report_missingness,
    _validate_critical_missingness,
    _report_outliers_3sigma,
    _write_json_artifact,
    _check_distribution_shift,
    _report_train_serve_feature_parity,
)
from src.models.feature_preparation import (  # noqa: F401
    _create_horizon_targets,
    _apply_with_temporal_context,
    add_utilization_scores,
    add_engineered_features,
    add_advanced_features,
    prepare_features,
    _infer_bounded_columns,
    _apply_bounded_scaling,
    _prepare_training_data,
)


def _load_qb_target_choice() -> str:
    """Load QB target choice from disk; default 'util' if missing."""
    qb_choice_path = MODELS_DIR / QB_TARGET_CHOICE_FILENAME
    if not qb_choice_path.exists():
        return "util"
    try:
        with open(qb_choice_path) as f:
            return json.load(f).get("qb_target", "util")
    except Exception as e:
        logger.warning("QB target choice load failed, defaulting to 'util': %s", e)
        return "util"


def _safe_mape(y_true, y_pred):
    """Calculate MAPE with denominator floor for stability near zero actuals."""
    denom_floor = float(MODEL_CONFIG.get("mape_denominator_floor", 3.0))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return None
    denom = np.maximum(np.abs(y_true[mask]), denom_floor)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom) * 100)


def _predictions_are_fantasy_points(position: str, converters: dict, qb_target: str) -> bool:
    """Return whether 1-week predictions for this position are in fantasy-point units."""
    pos_target_cfg = MODEL_CONFIG.get("position_target_type", {})
    target_type = pos_target_cfg.get(position, "util")
    if target_type in {"fp", "component"}:
        return True
    if position == "QB" and qb_target == "fp":
        return True
    return (
        position in converters
        and target_type != "fp"
        and (position != "QB" or qb_target == "util")
    )


def _select_backtest_actual_series(
    df: pd.DataFrame,
    position: str,
    converters: dict,
    qb_target: str,
) -> Optional[pd.Series]:
    """Choose the correct evaluation target without mixing FP and utilization units."""
    if _predictions_are_fantasy_points(position, converters, qb_target):
        if "target_1w" in df.columns:
            return df["target_1w"]
        return None

    if "target_util_1w" in df.columns:
        return df["target_util_1w"]
    return None


def _populate_actual_for_backtest(
    df: pd.DataFrame,
    converters: dict,
    qb_target: str,
) -> pd.DataFrame:
    """Populate `actual_for_backtest` using unit-consistent targets per position."""
    df = df.copy()
    df["actual_for_backtest"] = np.nan
    for position in df["position"].dropna().unique():
        pos_mask = df["position"] == position
        pos_actual = _select_backtest_actual_series(df.loc[pos_mask], position, converters, qb_target)
        if pos_actual is not None:
            df.loc[pos_mask, "actual_for_backtest"] = pos_actual

    if df["actual_for_backtest"].isna().all():
        if "target_1w" in df.columns:
            df["actual_for_backtest"] = df["target_1w"]
        elif "target_util_1w" in df.columns:
            df["actual_for_backtest"] = df["target_util_1w"]
        else:
            df["actual_for_backtest"] = df.get("fantasy_points", np.nan)
    return df




def _report_test_metrics(trainer, test_data: pd.DataFrame, train_data: pd.DataFrame) -> dict:
    """Evaluate and return model performance on held-out test set.

    Returns dict of {position: {rmse, mae, r2, ...}} for persistence.
    Also prints metrics to stdout.
    Returns {} when test_data is empty (draft prep / projection mode — no actuals yet).
    """
    if test_data.empty:
        from src.utils.nfl_calendar import is_draft_prep_window
        if is_draft_prep_window():
            print("  [DRAFT PREP MODE] No test actuals — evaluation metrics skipped.")
            print("  Run walk-forward (--walk-forward) to see 2025-validated performance.")
        else:
            print("  Test set is empty — evaluation metrics skipped.")
        return {}

    test_metrics = {}
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from src.evaluation.metrics import spearman_rank_correlation
    from src.models.utilization_to_fp import UtilizationToFPConverter
    
    qb_target = _load_qb_target_choice()
    converters = {}
    for pos in ["RB", "WR", "TE", "QB"]:
        try:
            conv = UtilizationToFPConverter.load(pos)
            if getattr(conv, "is_fitted", False):
                converters[pos] = conv
        except Exception as e:
            logger.warning("Converter load for %s in test metrics: %s", pos, e)

    for position in trainer.trained_models:
        multi_model = trainer.trained_models[position]
        pos_test = test_data[test_data["position"] == position]
        if len(pos_test) < 10:
            continue

        # Component mode: multi_model is None placeholder
        if multi_model is None:
            # Use component predictor feature names if available
            comp = trainer.component_predictors.get(position)
            if comp is None:
                continue
            feature_names = getattr(comp, "feature_names", [])
            available = [c for c in feature_names if c in pos_test.columns]
            if len(available) < max(len(feature_names) * 0.5, 1):
                continue
        else:
            model = multi_model.models.get(1) or list(multi_model.models.values())[0]
            available = [c for c in model.feature_names if c in pos_test.columns]
            if len(available) < len(model.feature_names) * 0.5:
                continue
        
        def _compute_metrics(pos, label, y_true, y_pred):
            rmse = float((mean_squared_error(y_true, y_pred)) ** 0.5)
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            mape = _safe_mape(y_true.values, y_pred)
            within_7 = float((np.abs(y_true.values - y_pred) <= 7).mean() * 100)
            within_10 = float((np.abs(y_true.values - y_pred) <= 10).mean() * 100)
            rho = spearman_rank_correlation(y_true.values, np.asarray(y_pred), top_n=None) if len(y_true) >= 5 else float('nan')
            mape_str = f", MAPE={mape:.1f}%" if mape is not None else ""
            rho_str = f", ρ={rho:.3f}" if np.isfinite(rho) else ""
            print(f"  {pos} (test {label}): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}{mape_str}, ≤7pt={within_7:.1f}%, ≤10pt={within_10:.1f}%{rho_str}")
            m = {"rmse": rmse, "mae": mae, "r2": r2,
                 "within_7pt_pct": within_7, "within_10pt_pct": within_10,
                 "n_samples": int(len(y_true))}
            if mape is not None:
                m["mape"] = float(mape)
            if np.isfinite(rho):
                m["spearman_rho"] = float(rho)
            test_metrics.setdefault(pos, {})[label] = m
        
        # Component mode: use component predictor directly for FP predictions
        if multi_model is None:
            comp = trainer.component_predictors.get(position)
            if comp is None:
                continue
            if "target_1w" in pos_test.columns:
                y_test = pos_test["target_1w"]
                valid = ~y_test.isna()
                if valid.sum() >= 5:
                    pos_subset = pos_test.loc[valid]
                    preds_fp = comp.predict(pos_subset)
                    _compute_metrics(position, "FP (component)", y_test[valid], preds_fp)
            continue

        # QB: report owner-facing FP metric (convert util->FP when needed).
        if position == "QB":
            if "target_1w" not in pos_test.columns and "target_util_1w" not in pos_test.columns:
                continue
            target_col = "target_1w" if "target_1w" in pos_test.columns else "target_util_1w"
            y_act = pos_test[target_col]
            valid = ~y_act.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                preds = multi_model.predict(pos_subset, n_weeks=1)
                label = "FP (owner objective)" if target_col == "target_1w" else "util (fallback)"
                preds_out = preds
                if target_col == "target_1w" and qb_target == "util" and "QB" in converters:
                    try:
                        eff_df = pos_subset.copy()
                        eff_df["utilization_score"] = preds
                        preds_out = converters["QB"].predict(preds, efficiency_df=eff_df)
                    except Exception as e:
                        logger.warning("QB FP conversion in metrics: %s", e)
                        preds_out = preds
                _compute_metrics(position, label, y_act[valid], preds_out)
            continue

        # RB/WR/TE: primary utilization, optional FP
        util_col = "target_util_1w"
        if util_col in pos_test.columns:
            y_util = pos_test[util_col]
            valid = ~y_util.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                pred_util = multi_model.predict(pos_subset, n_weeks=1)
                _compute_metrics(position, "util", y_util[valid], pred_util)
        if "target_1w" in pos_test.columns:
            y_test = pos_test["target_1w"]
            valid = ~y_test.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                preds = multi_model.predict(pos_subset, n_weeks=1)
                preds_fp = preds
                if position in converters:
                    try:
                        eff_df = pos_subset.copy()
                        eff_df["utilization_score"] = preds
                        preds_fp = converters[position].predict(preds, efficiency_df=eff_df)
                    except Exception as e:
                        logger.warning("FP conversion for %s in metrics: %s", position, e)
                        preds_fp = preds
                _compute_metrics(position, "FP", y_test[valid], preds_fp)

    # Compare against FantasyPros industry RMSE benchmarks
    if test_metrics:
        from src.evaluation.metrics import generate_expert_baselines
        expert_benchmarks = generate_expert_baselines(np.array([]), np.array([]))
        print("\n  --- vs FantasyPros industry RMSE ---")
        for pos, pos_results in test_metrics.items():
            exp_rmse = expert_benchmarks.get(pos)
            if exp_rmse is None:
                continue
            # Use the first available metric set (FP preferred, then util)
            for label in ["FP", "FP (owner objective)", "util"]:
                if label in pos_results:
                    model_rmse = pos_results[label]["rmse"]
                    delta_pct = (exp_rmse - model_rmse) / exp_rmse * 100
                    status = "BEAT" if delta_pct > 0 else "BEHIND"
                    print(f"  {pos}: {model_rmse:.2f} vs {exp_rmse:.1f} ({delta_pct:+.1f}%) [{status}]")
                    pos_results[label]["expert_benchmark_rmse"] = exp_rmse
                    pos_results[label]["vs_expert_pct"] = round(delta_pct, 1)
                    break

    return test_metrics


def _run_backtest_after_training(trainer, test_data: pd.DataFrame,
                                 train_seasons: list, actual_test_season: int,
                                 train_data: pd.DataFrame = None):
    """
    Run full backtest using the trained ensemble on held-out test data.
    Saves backtest results and app-compatible advanced_model_results.json.
    """
    if test_data.empty or len(test_data) < 10:
        return
    test_data = test_data.copy()
    test_data["predicted_points"] = np.nan
    test_data["predicted_utilization"] = np.nan

    # Load utilization->FP converters when present.
    converters = {}
    try:
        from src.models.utilization_to_fp import UtilizationToFPConverter
        for pos in ["RB", "WR", "TE", "QB"]:
            try:
                c = UtilizationToFPConverter.load(pos)
                if getattr(c, "is_fitted", False):
                    converters[pos] = c
            except Exception as e:
                logger.warning("Converter load for %s skipped: %s", pos, e)
    except Exception as e:
        logger.warning("UtilizationToFPConverter import failed: %s", e)
        converters = {}
    qb_target = _load_qb_target_choice()
    import time as _time
    _pred_start = _time.perf_counter()
    _n_predicted = 0
    for position in trainer.trained_models:
        multi_model = trainer.trained_models[position]
        pos_mask = test_data["position"] == position
        pos_test = test_data.loc[pos_mask]
        if len(pos_test) < 5:
            continue

        # Component mode: multi_model is None placeholder
        if multi_model is None:
            comp = trainer.component_predictors.get(position)
            if comp is None:
                continue
            pos_test = test_data.loc[pos_mask].copy()
            preds = comp.predict(pos_test)
        else:
            model = multi_model.models.get(1) or list(multi_model.models.values())[0]
            medians = getattr(model, "feature_medians", {})
            for fn in getattr(model, "feature_names", []):
                if fn not in pos_test.columns:
                    test_data.loc[pos_mask, fn] = medians.get(fn, 0)
            pos_test = test_data.loc[pos_mask].copy()
            preds = multi_model.predict(pos_test, n_weeks=1)
        _n_predicted += len(pos_test)
        test_data.loc[pos_mask, "predicted_utilization"] = preds
        # Default: set points equal to raw model output.
        test_data.loc[pos_mask, "predicted_points"] = preds
        # Convert utilization -> fantasy points only for positions trained on util targets.
        # Positions configured with target_type="fp" predict FP directly and skip conversion.
        pos_target_cfg = MODEL_CONFIG.get("position_target_type", {})
        pos_target_type = pos_target_cfg.get(position, "util")
        should_convert = (position in converters
                          and pos_target_type == "util"
                          and (position != "QB" or qb_target == "util"))
        if should_convert:
            eff_df = pos_test.copy()
            eff_df["utilization_score"] = preds
            try:
                fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
                test_data.loc[pos_mask, "predicted_points"] = fp_pred
            except Exception as e:
                logger.warning("Backtest FP conversion for %s skipped: %s", position, e)
    _pred_elapsed = _time.perf_counter() - _pred_start
    if _n_predicted > 0:
        _per_player = _pred_elapsed / _n_predicted
        from config.settings import MAX_PREDICTION_TIME_PER_PLAYER_SECONDS
        print(f"  Prediction speed: {_per_player:.4f}s/player ({_n_predicted} players in {_pred_elapsed:.2f}s)"
              f"  {'OK' if _per_player <= MAX_PREDICTION_TIME_PER_PLAYER_SECONDS else 'SLOW (>' + str(MAX_PREDICTION_TIME_PER_PLAYER_SECONDS) + 's)'}")
    valid_preds = test_data["predicted_points"].notna()
    if valid_preds.sum() < 10:
        return
    test_data = _populate_actual_for_backtest(test_data, converters, qb_target)
    backtester = ModelBacktester()
    pred_col = "predicted_points"
    actual_col = "actual_for_backtest"

    # C5 fix: compute calibration residuals from training data (not test data)
    # to avoid circular CI calculation that always appears calibrated.
    calibration_errors = None
    if train_data is not None and not train_data.empty:
        calibration_errors = {}
        for position in trainer.trained_models:
            try:
                multi_model = trainer.trained_models[position]
                pos_train = train_data[train_data["position"] == position].copy()
                if len(pos_train) < 20:
                    continue
                if multi_model is None:
                    comp = trainer.component_predictors.get(position)
                    if comp is None:
                        continue
                    train_preds = comp.predict(pos_train)
                    train_actual = _select_backtest_actual_series(pos_train, position, converters, qb_target)
                    preds_for_cal = train_preds
                else:
                    model = multi_model.models.get(1) or list(multi_model.models.values())[0]
                    fnames = getattr(model, "feature_names", [])
                    medians = getattr(model, "feature_medians", {})
                    for fn in fnames:
                        if fn not in pos_train.columns:
                            pos_train[fn] = medians.get(fn, 0)
                    train_preds = multi_model.predict(pos_train, n_weeks=1)
                    train_actual = _select_backtest_actual_series(pos_train, position, converters, qb_target)
                    preds_for_cal = train_preds
                    _ptc = MODEL_CONFIG.get("position_target_type", {})
                    should_convert = (position in converters
                                      and _ptc.get(position, "util") == "util"
                                      and (position != "QB" or qb_target == "util"))
                    if should_convert:
                        eff_df = pos_train.copy()
                        eff_df["utilization_score"] = train_preds
                        try:
                            preds_for_cal = converters[position].predict(train_preds, efficiency_df=eff_df)
                        except Exception:
                            pass

                if train_actual is not None:
                    residuals = (train_actual.values - np.asarray(preds_for_cal, dtype=float))
                    valid_resid = residuals[np.isfinite(residuals)]
                    if len(valid_resid) >= 20:
                        calibration_errors[position] = valid_resid
            except Exception as e:
                logger.warning("Calibration residual computation for %s failed: %s", position, e)

    test_data = backtester.calculate_confidence_intervals(
        test_data,
        pred_col=pred_col,
        actual_col=actual_col,
        confidence=0.80,
        lower_col="prediction_ci80_lower",
        upper_col="prediction_ci80_upper",
        calibration_errors=calibration_errors,
    )
    test_data = backtester.calculate_confidence_intervals(
        test_data,
        pred_col=pred_col,
        actual_col=actual_col,
        confidence=0.95,
        lower_col="prediction_ci95_lower",
        upper_col="prediction_ci95_upper",
        calibration_errors=calibration_errors,
    )
    results = backtester.backtest_season(
        predictions=test_data,
        actuals=test_data,
        season=actual_test_season,
        prediction_col=pred_col,
        actual_col=actual_col,
    )
    ci_mask = test_data[[actual_col, "prediction_ci80_lower", "prediction_ci80_upper"]].dropna()
    if len(ci_mask) > 0:
        results["confidence_band_coverage_10pt"] = float(
            ((ci_mask[actual_col] >= ci_mask["prediction_ci80_lower"])
             & (ci_mask[actual_col] <= ci_mask["prediction_ci80_upper"])).mean() * 100
        )
    results["train_seasons"] = train_seasons
    results["test_season"] = actual_test_season
    results["model_source"] = "production_ensemble"
    results["feature_counts"] = {
        pos: (
            len(getattr(trainer.component_predictors.get(pos), "feature_names", []))
            if m is None
            else len(getattr(m.models.get(1) or list(m.models.values())[0], "feature_names", []))
        )
        for pos, m in trainer.trained_models.items()
    }
    baseline_comp = backtester.compare_to_baseline(
        test_data, actual_col=actual_col, pred_col=pred_col
    )
    if "error" not in baseline_comp:
        results["baseline_comparison"] = baseline_comp
    multi_baseline = backtester.compare_to_multiple_baselines(
        test_data, actual_col=actual_col, pred_col=pred_col
    )
    if "error" not in multi_baseline:
        results["multiple_baseline_comparison"] = multi_baseline
        # Print baseline comparison prominently
        print("\n  === BASELINE COMPARISON (model must beat all) ===")
        for bl_name, bl_data in multi_baseline.get("baselines", {}).items():
            bl_rmse = bl_data.get("rmse", "?")
            bl_beat = bl_data.get("model_beats_baseline", False)
            bl_imp = bl_data.get("improvement_pct", 0)
            status = "PASS" if bl_beat else "FAIL"
            print(f"    {bl_name}: baseline RMSE={bl_rmse}, improvement={bl_imp:+.1f}% [{status}]")
        model_rmse = multi_baseline.get("model", {}).get("rmse", "?")
        print(f"    Model RMSE: {model_rmse}")

    # Additional baselines via baselines.py module (C4: real expert-projection baselines)
    try:
        from src.evaluation.baselines import compare_model_to_baselines, format_baseline_report
        if pred_col in test_data.columns and actual_col in test_data.columns:
            valid_bl = test_data.dropna(subset=[pred_col, actual_col])
            if len(valid_bl) >= 30:
                bl_comparison = compare_model_to_baselines(
                    valid_bl, valid_bl[pred_col], target_col=actual_col
                )
                results["strong_baseline_comparison"] = bl_comparison
                report_text = format_baseline_report(bl_comparison)
                print(f"\n{report_text}")
    except Exception as e:
        logger.warning("Strong baseline comparison skipped: %s", e)

    expert_csv = DATA_DIR / "expert_consensus.csv"
    if expert_csv.exists():
        expert_comp = backtester.compare_to_expert_consensus(
            test_data,
            expert_csv_path=str(expert_csv),
            actual_col=actual_col,
            pred_col=pred_col,
            player_key="name",
        )
        if "error" not in expert_comp:
            results["expert_comparison"] = expert_comp
            print(
                "  Expert benchmark: "
                f"model RMSE={expert_comp['model_rmse']} vs expert RMSE={expert_comp['expert_rmse']} "
                f"({expert_comp['model_vs_expert_pct']}% better)"
            )
    # --- Expert industry benchmark comparison (FantasyPros published RMSE) ---
    # This runs unconditionally using published accuracy data, no CSV needed.
    from src.evaluation.metrics import generate_expert_baselines
    expert_rmse_benchmarks = generate_expert_baselines(
        np.array([]), np.array([])  # Only need the lookup dict
    )
    by_position = results.get("by_position", {})
    if by_position:
        expert_comp = {"by_position": {}}
        model_rmse_all, expert_rmse_all, n_total = [], [], 0
        print("\n  === EXPERT BENCHMARK (vs FantasyPros industry RMSE) ===")
        for pos in ["QB", "RB", "WR", "TE"]:
            pos_data = by_position.get(pos, {})
            pos_rmse = pos_data.get("rmse")
            exp_rmse = expert_rmse_benchmarks.get(pos)
            if pos_rmse is not None and exp_rmse is not None:
                beat_pct = round((exp_rmse - pos_rmse) / exp_rmse * 100, 1)
                n_pos = pos_data.get("n_samples", 0)
                status = "BEAT" if beat_pct > 0 else "BEHIND"
                print(f"    {pos}: model RMSE={pos_rmse:.2f} vs expert RMSE={exp_rmse:.1f} "
                      f"({beat_pct:+.1f}%) [{status}]")
                expert_comp["by_position"][pos] = {
                    "model_rmse": round(pos_rmse, 2),
                    "expert_rmse": exp_rmse,
                    "beat_pct": beat_pct,
                    "n_matched": n_pos,
                }
                model_rmse_all.append(pos_rmse)
                expert_rmse_all.append(exp_rmse)
                n_total += n_pos
        if model_rmse_all:
            avg_model = np.mean(model_rmse_all)
            avg_expert = np.mean(expert_rmse_all)
            overall_pct = round((avg_expert - avg_model) / avg_expert * 100, 1)
            expert_comp["model_rmse"] = round(avg_model, 2)
            expert_comp["expert_rmse"] = round(avg_expert, 2)
            expert_comp["model_vs_expert_pct"] = overall_pct
            expert_comp["source"] = "FantasyPros industry average 2019-2024"
            expert_comp["n_total"] = n_total
            print(f"    Overall: model avg RMSE={avg_model:.2f} vs expert avg RMSE={avg_expert:.1f} "
                  f"({overall_pct:+.1f}%)")
        results["expert_comparison"] = expert_comp

    # --- Success criteria evaluation (per requirements Section VII) ---
    from src.evaluation.backtester import check_success_criteria, print_success_criteria_report
    success_criteria = check_success_criteria(results)
    results["success_criteria"] = success_criteria
    print_success_criteria_report(success_criteria)

    # --- Model drift detection: compare against previous backtest if available ---
    try:
        prev_files = sorted(backtester.results_dir.glob("backtest_*.json"))
        if len(prev_files) >= 2:
            prev_path = prev_files[-2]  # second-to-last = previous run
            with open(prev_path) as f:
                prev_results = json.load(f)
            prev_rmse = prev_results.get("metrics", {}).get("rmse")
            curr_rmse = results.get("metrics", {}).get("rmse")
            if prev_rmse and curr_rmse and prev_rmse > 0:
                drift_pct = (curr_rmse - prev_rmse) / prev_rmse * 100
                drift_threshold_pct = float(RETRAINING_CONFIG.get("degradation_threshold_pct", 20.0))
                results["model_drift"] = {
                    "previous_rmse": prev_rmse,
                    "current_rmse": curr_rmse,
                    "drift_pct": round(drift_pct, 1),
                    "drift_threshold_pct": drift_threshold_pct,
                    "degradation_exceeds_threshold": drift_pct > drift_threshold_pct,
                }
                if drift_pct > drift_threshold_pct:
                    print(f"\n  *** WARNING: Model drift detected! RMSE degraded {drift_pct:.1f}% vs previous run. "
                          f"Consider rollback (prev RMSE={prev_rmse}, current={curr_rmse}).")
                else:
                    print(f"\n  Model drift: {drift_pct:+.1f}% vs previous (stable)")
    except Exception as e:
        logger.warning("Model drift detection skipped: %s", e)

    # H3: Compare ensemble vs simple Ridge to justify model complexity
    if train_data is not None and not train_data.empty:
        simple_comparison = {}
        print("\n  === MODEL COMPLEXITY COMPARISON (ensemble vs Ridge) ===")
        for position in trainer.trained_models:
            try:
                multi_model = trainer.trained_models[position]
                if multi_model is None:
                    comp = trainer.component_predictors.get(position)
                    fnames = getattr(comp, "feature_names", []) if comp else []
                else:
                    base = multi_model.models.get(1) or list(multi_model.models.values())[0]
                    fnames = getattr(base, "feature_names", [])
                if len(fnames) < 5:
                    continue
                pos_train = train_data[train_data["position"] == position].copy()
                pos_test = test_data[test_data["position"] == position].copy()
                if len(pos_train) < 50 or len(pos_test) < 10:
                    continue
                train_target = _select_backtest_actual_series(pos_train, position, converters, qb_target)
                test_target = _select_backtest_actual_series(pos_test, position, converters, qb_target)
                if train_target is None or test_target is None:
                    continue
                pos_train["_comparison_target"] = train_target
                pos_test["_comparison_target"] = test_target
                valid_train = pos_train.dropna(subset=["_comparison_target"])
                valid_test = pos_test.dropna(subset=["_comparison_target"])
                if len(valid_train) < 50 or len(valid_test) < 10:
                    continue
                avail_feats = [f for f in fnames if f in valid_train.columns and f in valid_test.columns]
                if len(avail_feats) < 5:
                    continue
                X_tr = valid_train[avail_feats].fillna(0)
                y_tr = valid_train["_comparison_target"]
                X_te = valid_test[avail_feats].fillna(0)
                y_te = valid_test["_comparison_target"]
                # Simple Ridge
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_tr, y_tr)
                ridge_preds = ridge.predict(X_te)
                from sklearn.metrics import mean_squared_error as _mse
                ridge_rmse = float(np.sqrt(_mse(y_te, ridge_preds)))
                # Ensemble
                if multi_model is None:
                    comp = trainer.component_predictors.get(position)
                    ensemble_preds = comp.predict(valid_test) if comp is not None else np.full(len(valid_test), np.nan)
                else:
                    ensemble_preds = multi_model.predict(valid_test, n_weeks=1)
                ensemble_rmse = float(np.sqrt(_mse(y_te, ensemble_preds)))
                improvement = ((ridge_rmse - ensemble_rmse) / ridge_rmse * 100) if ridge_rmse > 0 else 0
                status = "JUSTIFIED" if improvement > 0 else "NOT JUSTIFIED"
                print(f"    {position}: Ridge RMSE={ridge_rmse:.2f}, Ensemble RMSE={ensemble_rmse:.2f} "
                      f"({improvement:+.1f}% improvement) [{status}]")
                simple_comparison[position] = {
                    "ridge_rmse": round(ridge_rmse, 3),
                    "ensemble_rmse": round(ensemble_rmse, 3),
                    "improvement_pct": round(improvement, 1),
                    "complexity_justified": improvement > 0,
                }
            except Exception as e:
                logger.warning("Simple model comparison for %s failed: %s", position, e)
        results["simple_model_comparison"] = simple_comparison

    backtester.save_results(results)

    # Write app-compatible results (all rubric metrics per position)
    backtest_results_app = {}
    for pos, pm in results.get("by_position", {}).items():
        backtest_results_app[pos] = {
            "rmse": pm["rmse"],
            "mae": pm["mae"],
            "r2": pm["r2"],
            "mape": pm.get("mape"),
            "correlation": pm.get("correlation"),
            "directional_accuracy_pct": pm.get("directional_accuracy_pct"),
            "within_3_pts_pct": pm.get("within_3_pts_pct"),
            "within_5_pts_pct": pm.get("within_5_pts_pct"),
            "within_7_pts_pct": pm.get("within_7_pts_pct"),
            "within_10_pts_pct": pm.get("within_10_pts_pct"),
            "spearman_rho": pm.get("spearman_rho"),
            "tier_classification_accuracy": pm.get("tier_classification_accuracy"),
            "boom_bust": pm.get("boom_bust"),
            "vor_rank_correlation": pm.get("vor_rank_correlation"),
            "mae_rmse_ratio": pm.get("mae_rmse_ratio"),
        }
    # Per-position Spearman from backtest
    spearman_by_pos = results.get("spearman_by_position", {})
    for pos, rho in spearman_by_pos.items():
        if pos in backtest_results_app:
            backtest_results_app[pos]["spearman_top50"] = rho

    app_results_path = DATA_DIR / "advanced_model_results.json"
    app_payload = {
        "_metadata": {
            "source": "train.py primary pipeline",
            "authoritative": True,
            "description": (
                "Single source of truth for model performance. "
                "Other result files (ml_evaluation_results.json, "
                "model_comparison_results.json) are from secondary/exploratory "
                "pipelines and may not reflect production model accuracy."
            ),
        },
        "timestamp": datetime.now().isoformat(),
        "train_seasons": train_seasons,
        "test_season": actual_test_season,
        "backtest_results": backtest_results_app,
        "success_criteria": success_criteria,
        "multiple_baseline_comparison": results.get("multiple_baseline_comparison"),
        "model_drift": results.get("model_drift"),
        "confidence_band_coverage_10pt": results.get("confidence_band_coverage_10pt"),
    }
    with open(app_results_path, "w") as f:
        json.dump(app_payload, f, indent=2, default=str)
    print(f"\nBacktest complete. App results written to {app_results_path.name}")

    # Remove stale secondary result files that may show misleading metrics
    for stale_file in ["ml_evaluation_results.json", "model_comparison_results.json",
                       "approach_comparison_results.json", "feature_engineering_results.json"]:
        stale_path = DATA_DIR / stale_file
        if stale_path.exists():
            stale_path.unlink()
            print(f"  Removed stale result file: {stale_file}")

    return results


def _run_robust_cv_report(train_data: pd.DataFrame):
    """Run RobustTimeSeriesCV and report per-fold metrics."""
    gap = MODEL_CONFIG.get("cv_gap_seasons", 0)
    validator = RobustTimeSeriesCV(
        n_splits=3, min_train_seasons=1, scale_features=True, gap_seasons=gap
    )
    exclude_cols = [
        "player_id", "name", "position", "team", "season", "week",
        "fantasy_points", "target", "opponent", "home_away",
        "created_at", "updated_at", "id", "birth_date", "college",
        "game_id", "game_time"
    ]
    
    for position in ["QB", "RB", "WR", "TE"]:
        pos_df = train_data[train_data["position"] == position].copy()
        if len(pos_df) < 200 or "season" not in pos_df.columns:
            continue
        target_col = "target_util_1w" if "target_util_1w" in pos_df.columns else ("target_1w" if "target_1w" in pos_df.columns else "fantasy_points")
        pos_df = pos_df.dropna(subset=[target_col])
        if len(pos_df) < 100:
            continue
        feature_cols = [c for c in pos_df.columns 
                       if c not in exclude_cols and not c.startswith("target_")
                       and pos_df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns
        feature_cols = filter_feature_columns(feature_cols)
        assert_no_leakage_columns(feature_cols, context=f"cv features ({position})")
        if len(feature_cols) < 5:
            continue
        try:
            result = validator.validate(
                pos_df, Ridge, {"alpha": 1.0},
                feature_cols, target_col=target_col, position=position
            )
            print(f"  {position} CV: RMSE={result.rmse:.2f} ± {np.std([f['rmse'] for f in result.fold_results]):.2f}, R²={result.r2:.3f}")
        except Exception as e:
            print(f"  {position} CV: skipped ({e})")


def _run_one_fold(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    train_seasons: list,
    actual_test_season: int,
    positions: list,
    tune_hyperparameters: bool,
    n_trials: int,
):
    """
    Run one fold: prepare features, train models, run backtest.
    Uses MODELS_DIR from config (patch for walk-forward). Returns (trainer, backtest_results).
    """
    train_data, test_data, trainer = _prepare_training_data(
        train_data, test_data, positions, tune_hyperparameters, n_trials,
        fast=False,
    )

    # Backtest prediction loop
    test_data = test_data.copy()
    test_data["predicted_points"] = np.nan
    test_data["predicted_utilization"] = np.nan

    converters = {}
    try:
        from src.models.utilization_to_fp import UtilizationToFPConverter
        for pos in ["RB", "WR", "TE", "QB"]:
            try:
                c = UtilizationToFPConverter.load(pos)
                if getattr(c, "is_fitted", False):
                    converters[pos] = c
            except Exception as e:
                logger.warning("Converter load for %s skipped: %s", pos, e)
    except Exception as e:
        logger.warning("UtilizationToFPConverter import failed: %s", e)
        converters = {}

    qb_target = _load_qb_target_choice()
    for position in positions:
        if position not in trainer.trained_models:
            continue
        multi_model = trainer.trained_models[position]
        pos_mask = test_data["position"] == position
        pos_test = test_data.loc[pos_mask]
        if len(pos_test) < 5:
            continue
        if multi_model is None:
            comp = trainer.component_predictors.get(position)
            if comp is None:
                continue
            pos_test = test_data.loc[pos_mask].copy()
            preds = comp.predict(pos_test)
        else:
            base = multi_model.models.get(1) or list(multi_model.models.values())[0]
            medians = getattr(base, "feature_medians", {})
            for fn in getattr(base, "feature_names", []):
                if fn not in pos_test.columns:
                    test_data.loc[pos_mask, fn] = medians.get(fn, 0)
            pos_test = test_data.loc[pos_mask].copy()
            preds = multi_model.predict(pos_test, n_weeks=1)
        test_data.loc[pos_mask, "predicted_utilization"] = preds
        test_data.loc[pos_mask, "predicted_points"] = preds
        _ptc2 = MODEL_CONFIG.get("position_target_type", {})
        should_convert = (position in converters
                          and _ptc2.get(position, "util") == "util"
                          and (position != "QB" or qb_target == "util"))
        if should_convert:
            eff_df = pos_test.copy()
            eff_df["utilization_score"] = preds
            try:
                fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
                test_data.loc[pos_mask, "predicted_points"] = fp_pred
            except Exception as e:
                logger.warning("FP conversion for %s skipped: %s", position, e)
    test_data = _populate_actual_for_backtest(test_data, converters, qb_target)
    results = _run_backtest_after_training(trainer, test_data, train_seasons, actual_test_season,
                                               train_data=train_data)
    return trainer, results


def train_models(positions: list = None,
                 tune_hyperparameters: bool = True,
                 n_trials: int = None,
                 test_season: int = None,
                 optimize_training_years: bool = False,
                 walk_forward: bool = False,
                 strict_requirements: bool = None,
                 fast: bool = False,
                 skip_cache_check: bool = False,
                 loyo_backtest: bool = False,
                 loyo_seasons: list = None):
    """
    Main training function with automatic train/test split.

    Args:
        positions: Positions to train models for
        tune_hyperparameters: Whether to tune hyperparameters
        n_trials: Number of Optuna trials
        test_season: Override test season (None = use latest available)
        walk_forward: If True, run walk-forward validation (train on 1..N-1, test on N) for last 4 seasons and report mean +/- std RMSE/MAE.
        fast: If True, apply FAST_MODEL_CONFIG overrides for ~8-10x faster training
              with minimal accuracy loss.
        loyo_backtest: If True, run LOYO walk-forward backtest across multiple seasons.
        loyo_seasons: Override test seasons for LOYO (default: 2018-2024).
    """
    # Apply fast-mode overrides before reading any config values
    if fast:
        print("[FAST MODE] Applying reduced training config for faster iteration.")
        for key, val in FAST_MODEL_CONFIG.items():
            MODEL_CONFIG[key] = val

    positions = positions or POSITIONS
    n_trials = n_trials or MODEL_CONFIG["n_optuna_trials"]
    if strict_requirements is None:
        strict_requirements = bool(MODEL_CONFIG.get("strict_requirements_default", False))

    # Pre-training data integrity gate — block on critical cache corruption
    if skip_cache_check:
        print("⚠ Skipping data integrity gate (--skip-cache-check)")
        cache_result = None
    else:
        cache_result = validate_training_cache_integrity()
    if cache_result is not None and not cache_result.passed:
        print("\n" + "=" * 60)
        print("TRAINING BLOCKED: Data integrity gate FAILED")
        print("=" * 60)
        for failure in cache_result.report.get("failures", []):
            print(f"  ✗ {failure}")
        print("\nFix the data issues above and re-run. Use --skip-cache-check to bypass (not recommended).")

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        gate_path = MODELS_DIR / "data_integrity_gate_report.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gate_path, "w") as f:
            json.dump(cache_result.report, f, indent=2, cls=_NumpyEncoder)
        print(f"Full report: {gate_path}")
        return
    else:
        print("✓ Data integrity gate passed")

    # H5: Experiment tracking
    from src.evaluation.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    experiment_run_id = tracker.start_run(
        config={
            "positions": positions,
            "tune_hyperparameters": tune_hyperparameters,
            "n_trials": n_trials,
            "test_season": test_season,
            "walk_forward": walk_forward,
            "fast": fast,
            "strict_requirements": strict_requirements,
        },
        description=f"train_models {'fast' if fast else 'full'} run",
    )

    print("=" * 60)
    print("NFL Player Performance Model Training")
    print(f"  Experiment run: {experiment_run_id}")
    print("=" * 60)
    
    # Load data with automatic train/test split (test = latest season)
    print("\n[1/5] Loading training data...")
    train_data, test_data, train_seasons, actual_test_season, context_data = load_training_data(
        positions,
        test_season=test_season,
        n_train_seasons=None,  # Use all available by default
        optimize_training_years=optimize_training_years,
        strict_requirements=strict_requirements,
        return_context=True,
    )
    print(f"Training records: {len(train_data)}")
    if context_data is not None and not context_data.empty:
        ctx_seasons = sorted({int(s) for s in context_data["season"].dropna().unique()})
        print(f"Pre-window context: {len(context_data)} records from "
              f"{min(ctx_seasons)}-{max(ctx_seasons)} "
              f"(warms up lookback features; not trained on)")
    if test_data.empty:
        from src.utils.nfl_calendar import is_draft_prep_window
        if is_draft_prep_window():
            print(f"Test records: 0  [DRAFT PREP MODE — projecting {actual_test_season}, no actuals yet]")
        else:
            print(f"Test records: 0  (no data for season {actual_test_season})")
    else:
        print(f"Test records: {len(test_data)}")

    # Scoring environment shift detection (council: non-stationarity check)
    try:
        from src.evaluation.monitoring import detect_scoring_environment_shift
        shift_report = detect_scoring_environment_shift(train_data)
        if shift_report["shifts_detected"]:
            print(f"\n  === SCORING ENVIRONMENT SHIFT DETECTED ===")
            for s in shift_report["shifts_detected"]:
                print(f"    {s['position']} {s['season']}: "
                      f"{s['prev_season_mean']:.1f} -> {s['season_mean']:.1f} "
                      f"({s['pct_change']:+.1f}%)")
            if shift_report["recommendation"]:
                print(f"    Recommendation: {shift_report['recommendation']}")
            print()
        else:
            print("  Scoring environment: stable across training window")
        _write_json_artifact(
            MODELS_DIR / "scoring_environment_report.json",
            shift_report,
            "scoring environment shift report",
        )
    except Exception as e:
        logger.warning("Scoring environment shift detection skipped: %s", e)

    # Lineage: only include seasons that actually have data in the DB
    lineage_seasons = sorted(set(train_seasons) | (set() if test_data.empty else {actual_test_season}))
    bronze_parent_ids = find_artifact_ids(
        layer="bronze",
        source="nflverse_weekly_stats",
        seasons=lineage_seasons,
    )
    tracker.log_params(experiment_run_id, {"bronze_parent_artifact_ids": bronze_parent_ids})

    # Optional walk-forward validation: train on 1..N-1, test on N for last 4 seasons
    if walk_forward:
        # When test_data is empty (draft prep / future season), only walk-forward over
        # seasons that have actual game data — exclude the projection target year.
        if test_data.empty:
            all_seasons = sorted(train_seasons)
        else:
            all_seasons = sorted(set(train_seasons) | {actual_test_season})
        test_seasons_wf = all_seasons[-4:] if len(all_seasons) >= 4 else all_seasons[-2:]
        import tempfile
        from pathlib import Path
        import config.settings as settings
        old_models_dir = settings.MODELS_DIR
        wf_metrics = []
        for ts in test_seasons_wf:
            td, td_test, tr_ss, _ = load_training_data(
                positions,
                test_season=ts,
                optimize_training_years=False,
                strict_requirements=strict_requirements,
            )
            if len(td_test) < 20:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                settings.MODELS_DIR = Path(tmp)
                try:
                    _, res = _run_one_fold(td, td_test, tr_ss, ts, positions, tune_hyperparameters, n_trials)
                    if res:
                        wf_metrics.append(res.get("by_position", {}))
                except Exception as e:
                    print(f"  Walk-forward fold {ts} failed: {e}")
                settings.MODELS_DIR = old_models_dir
        if wf_metrics:
            print("\n" + "=" * 60)
            print("Walk-Forward Validation Summary (mean +/- std)")
            print("=" * 60)
            for pos in POSITIONS:
                rmses = [m[pos]["rmse"] for m in wf_metrics if pos in m and isinstance(m[pos].get("rmse"), (int, float))]
                maes = [m[pos]["mae"] for m in wf_metrics if pos in m and isinstance(m[pos].get("mae"), (int, float))]
                if rmses:
                    print(f"  {pos}: RMSE {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}  MAE {np.mean(maes):.2f} +/- {np.std(maes):.2f}")
            return None, train_data, test_data, actual_test_season
        settings.MODELS_DIR = old_models_dir

    # LOYO walk-forward backtest: train fresh per season, test on each in range
    if loyo_backtest:
        from src.evaluation.backtester import run_loyo_backtest
        run_loyo_backtest(
            test_seasons=loyo_seasons,
            positions=positions,
            tune_hyperparameters=tune_hyperparameters,
            n_trials=n_trials,
            fast=fast,
            strict_requirements=strict_requirements,
        )
        return None, train_data, test_data, actual_test_season

    # Shared preprocessing: DVP, external, season-long, utilization, targets,
    # feature engineering, bounded scaling, winsorization, model training, util-to-fp.
    print("\n[2/5] Preparing features, engineering, and training...")
    train_data, test_data, trainer = _prepare_training_data(
        train_data, test_data, positions, tune_hyperparameters, n_trials,
        fast=fast, context_data=context_data,
    )

    # Data quality checks (train_models-only, not needed in walk-forward folds)
    print("\n[3/5] Data quality checks...", flush=True)
    train_missing = _report_missingness(train_data, "train", threshold=0.05)
    _validate_critical_missingness(train_data, "train", threshold=0.05)
    train_outliers = _report_outliers_3sigma(
        train_data,
        "train",
        cols=["fantasy_points", "target_1w", "target_4w", "target_util_1w", "utilization_score"],
    )
    if test_data.empty:
        # Draft prep mode: test set is the future season with no actuals yet.
        # Skip test-data quality checks — they'd produce meaningless output.
        test_missing = {}
        dist_shift = {"skipped": "draft_prep_mode_no_test_actuals"}
        print("  Test data quality checks skipped (draft prep mode — no actuals for projection season).")
    else:
        test_missing = _report_missingness(test_data, "test", threshold=0.05)
        _validate_critical_missingness(test_data, "test", threshold=0.05)
        _report_train_serve_feature_parity(train_data, test_data)
        # H2: Adversarial validation to detect train/test distribution shift
        dist_shift = _check_distribution_shift(train_data, test_data)
    quality_payload = {
        "generated_at": datetime.now().isoformat(),
        "strict_requirements": bool(strict_requirements),
        "distribution_shift": dist_shift,
        "missingness_above_5pct": {
            "train": train_missing,
            "test": test_missing,
        },
        "train_outliers_3sigma": train_outliers,
    }
    _write_json_artifact(MODELS_DIR / "data_quality_report.json", quality_payload, "data quality report")

    train_players_per_pos = {
        pos: int(train_data[train_data["position"] == pos]["player_id"].nunique())
        for pos in POSITIONS
    }
    requirement_gates = {
        "generated_at": datetime.now().isoformat(),
        "strict_requirements": bool(strict_requirements),
        "training_seasons": len(train_seasons),
        "min_training_seasons": {
            "1w": MIN_TRAINING_SEASONS_1W,
            "4w": MIN_TRAINING_SEASONS_4W,
            "18w": MIN_TRAINING_SEASONS_18W,
        },
        "seasons_gate": {
            "1w_pass": len(train_seasons) >= MIN_TRAINING_SEASONS_1W,
            "4w_pass": (not MODEL_CONFIG.get("use_4w_hybrid", True)) or len(train_seasons) >= MIN_TRAINING_SEASONS_4W,
            "18w_pass": (not MODEL_CONFIG.get("use_18w_deep", True)) or len(train_seasons) >= MIN_TRAINING_SEASONS_18W,
        },
        "players_per_position": train_players_per_pos,
        "min_players_per_position": MIN_PLAYERS_PER_POSITION,
        "players_gate": {
            pos: train_players_per_pos.get(pos, 0) >= MIN_PLAYERS_PER_POSITION.get(pos, 0)
            for pos in POSITIONS
        },
    }
    _write_json_artifact(MODELS_DIR / "training_requirements_gate.json", requirement_gates, "requirements gate report")

    # Horizon-specific models: 4-week LSTM+ARIMA hybrid, 18-week deep (when enabled)
    # Skip in fast mode: PyTorch/statsmodels horizon models trigger segfaults from
    # native memory corruption when run after heavy XGBoost/RF component training.
    horizon_status: Dict[str, Dict[str, str]] = {pos: {} for pos in positions}
    target_semantics: Dict[str, Dict[str, str]] = {pos: {} for pos in positions}
    if fast:
        print("\n[4c/5] Horizon-specific models skipped (fast mode)")
        for pos in positions:
            horizon_status[pos] = {"hybrid_4w": "skipped_fast_mode", "deep_18w": "skipped_fast_mode"}
            target_semantics[pos] = {"1w": "component", "4w": "component", "18w": "component"}
    else:
        print("\n[4c/5] Training horizon-specific models (4w hybrid, 18w deep)...")
        try:
            from src.models.horizon_models import (
                Hybrid4WeekModel,
                DeepSeasonLongModel,
                HAS_TF,
                HAS_ARIMA,
            )
            n_seasons = len(train_seasons)
            if not HAS_TF:
                print("  Horizon note: PyTorch unavailable; LSTM/deep components disabled.")
            if not HAS_ARIMA:
                print("  Horizon note: statsmodels unavailable; ARIMA component disabled.")
            for position in positions:
                if position not in trainer.trained_models:
                    horizon_status[position]["hybrid_4w"] = "base_model_missing"
                    horizon_status[position]["deep_18w"] = "base_model_missing"
                    target_semantics[position]["1w"] = "base_model_missing"
                    target_semantics[position]["4w"] = "base_model_missing"
                    target_semantics[position]["18w"] = "base_model_missing"
                    continue
                multi = trainer.trained_models[position]
                if multi is None:
                    # Component mode: use component predictor features for horizon models
                    comp = trainer.component_predictors.get(position)
                    feature_cols = getattr(comp, "feature_names", []) if comp else []
                    if not feature_cols:
                        horizon_status[position]["hybrid_4w"] = "component_mode_no_features"
                        horizon_status[position]["deep_18w"] = "component_mode_no_features"
                        target_semantics[position]["1w"] = "component"
                        target_semantics[position]["4w"] = "component"
                        target_semantics[position]["18w"] = "component"
                        continue
                else:
                    base = multi.models.get(1) or list(multi.models.values())[0]
                    feature_cols = getattr(base, "feature_names", [])
                # Track semantically intended targets: QB may be fp/util; skill positions are utilization-first.
                target_semantics[position]["1w"] = "target_1w_or_target_util_1w_trainer_selected"
                target_semantics[position]["4w"] = "target_util_4w_preferred_over_target_4w"
                target_semantics[position]["18w"] = "target_util_18w_preferred_over_target_18w"
                if len(feature_cols) < 5:
                    horizon_status[position]["hybrid_4w"] = "insufficient_features"
                    horizon_status[position]["deep_18w"] = "insufficient_features"
                    continue
                pos_data = train_data[train_data["position"] == position].copy()
                pos_data = pos_data.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
                X_pos = pos_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                player_ids = pos_data["player_id"].values
                seasons_arr = pos_data["season"].values if "season" in pos_data.columns else None
    
                if MODEL_CONFIG.get("use_4w_hybrid", True) and n_seasons >= MIN_TRAINING_SEASONS_4W:
                    y_4w = pos_data.get("target_util_4w", pos_data.get("target_4w"))
                    if not HAS_TF or not HAS_ARIMA:
                        reason = []
                        if not HAS_TF:
                            reason.append("tensorflow_missing")
                        if not HAS_ARIMA:
                            reason.append("statsmodels_missing")
                        horizon_status[position]["hybrid_4w"] = "unavailable:" + ",".join(reason)
                    elif y_4w is not None and y_4w.notna().sum() >= 100:
                        try:
                            hybrid = Hybrid4WeekModel(position)
                            hybrid.fit(pos_data, y_4w, player_ids, feature_cols,
                                      epochs=MODEL_CONFIG.get("lstm_epochs", 80),
                                      seasons=seasons_arr)
                            if hybrid.is_fitted:
                                hybrid.save()
                                horizon_status[position]["hybrid_4w"] = "trained_and_saved"
                                print(f"  {position}: 4-week hybrid model saved")
                            else:
                                horizon_status[position]["hybrid_4w"] = "fit_not_converged"
                        except Exception as e:
                            horizon_status[position]["hybrid_4w"] = f"fit_failed:{e}"
                            print(f"  4-week hybrid skip for {position}: {e}")
                    else:
                        horizon_status[position]["hybrid_4w"] = "insufficient_targets"
                elif not MODEL_CONFIG.get("use_4w_hybrid", True):
                    horizon_status[position]["hybrid_4w"] = "disabled_by_config"
                else:
                    horizon_status[position]["hybrid_4w"] = "insufficient_training_seasons"
    
                if MODEL_CONFIG.get("use_18w_deep", True) and n_seasons >= MIN_TRAINING_SEASONS_18W:
                    y_18w = pos_data.get("target_util_18w", pos_data.get("target_18w"))
                    if not HAS_TF:
                        horizon_status[position]["deep_18w"] = "unavailable:tensorflow_missing"
                    elif y_18w is not None and y_18w.notna().sum() >= 80:
                        try:
                            deep = DeepSeasonLongModel(position, n_features=min(150, len(feature_cols)))
                            X_arr = X_pos.values.astype(np.float64)
                            y_arr = y_18w.values.astype(np.float64)
                            valid = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
                            if valid.sum() >= 80:
                                seasons_18w = seasons_arr[valid] if seasons_arr is not None else None
                                deep.fit(
                                    X_arr[valid], y_arr[valid],
                                    feature_names=feature_cols,
                                    epochs=MODEL_CONFIG.get("deep_epochs", 100),
                                    batch_size=MODEL_CONFIG.get("deep_batch_size", 64),
                                    seasons=seasons_18w,
                                )
                                if deep.is_fitted:
                                    deep.save()
                                    horizon_status[position]["deep_18w"] = "trained_and_saved"
                                    print(f"  {position}: 18-week deep model saved")
                                else:
                                    horizon_status[position]["deep_18w"] = "fit_not_converged"
                            else:
                                horizon_status[position]["deep_18w"] = "insufficient_valid_rows"
                        except Exception as e:
                            horizon_status[position]["deep_18w"] = f"fit_failed:{e}"
                            print(f"  18-week deep skip for {position}: {e}")
                    else:
                        horizon_status[position]["deep_18w"] = "insufficient_targets"
                elif not MODEL_CONFIG.get("use_18w_deep", True):
                    horizon_status[position]["deep_18w"] = "disabled_by_config"
                else:
                    horizon_status[position]["deep_18w"] = "insufficient_training_seasons"
        except ImportError:
            print("  Horizon models skipped (TensorFlow not available).")
            for position in positions:
                horizon_status[position]["hybrid_4w"] = "unavailable:horizon_module_import_error"
                horizon_status[position]["deep_18w"] = "unavailable:horizon_module_import_error"
        except Exception as e:
            print(f"  Horizon-specific training skipped: {e}")
            for position in positions:
                horizon_status[position]["hybrid_4w"] = f"unexpected_error:{e}"
                horizon_status[position]["deep_18w"] = f"unexpected_error:{e}"
    try:
        with open(MODELS_DIR / "horizon_model_status.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "train_seasons": train_seasons,
                    "status_by_position": horizon_status,
                    "target_semantics": target_semantics,
                },
                f,
                indent=2,
            )
    except Exception as e:
        print(f"  Horizon status write skipped: {e}")

    # Top-10 feature importance per position (explainability)
    try:
        top10 = get_top10_feature_importance_per_position(
            trainer.trained_models,
            output_path=MODELS_DIR / "top10_features_per_position.json",
        )
        for pos, feats in top10.items():
            print(f"  Top-3 {pos}: {[f['feature'] for f in feats[:3]]}")
        if MODEL_CONFIG.get("enable_shap_pdp", True):
            explain_dir = MODELS_DIR / "explainability"
            explain_dir.mkdir(parents=True, exist_ok=True)
            for pos, multi in trainer.trained_models.items():
                try:
                    base = multi.models.get(1) or list(multi.models.values())[0]
                    feature_cols = getattr(base, "feature_names", [])
                    if len(feature_cols) < 5:
                        continue
                    pos_data = train_data[train_data["position"] == pos].copy()
                    if pos_data.empty:
                        continue
                    X_pos = pos_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

                    shap_result = explain_with_shap(
                        base,
                        X_pos,
                        feature_cols,
                        n_samples=int(MODEL_CONFIG.get("shap_samples", 200)),
                    )
                    if shap_result is not None:
                        shap_vals = np.asarray(shap_result["values"])
                        if shap_vals.ndim == 2 and shap_vals.shape[1] == len(feature_cols):
                            mean_abs = np.mean(np.abs(shap_vals), axis=0)
                            order = np.argsort(mean_abs)[::-1][:10]
                            payload = {
                                "position": pos,
                                "top_shap_features": [
                                    {"feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i])}
                                    for i in order
                                ],
                            }
                            with open(explain_dir / f"shap_{pos.lower()}.json", "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2)

                    partial_dependence_plots(
                        base,
                        X_pos,
                        top_n=int(MODEL_CONFIG.get("pdp_top_n", 5)),
                        grid_resolution=20,
                        output_path=explain_dir / f"pdp_{pos.lower()}.json",
                    )
                except Exception as pos_err:
                    print(f"  Explainability skip for {pos}: {pos_err}")
    except Exception as e:
        print(f"  Top-10 feature export skipped: {e}")

    # Feature importance stability tracking across training runs (per requirements)
    # Compare current top-10 features to previous run; flag drift if overlap < 60%
    try:
        top10_path = MODELS_DIR / "top10_features_per_position.json"
        history_path = MODELS_DIR / "feature_importance_history.json"
        if top10_path.exists():
            with open(top10_path, encoding="utf-8") as f:
                current_top10 = json.load(f)
            # Load history (list of previous snapshots)
            history = []
            if history_path.exists():
                try:
                    with open(history_path, encoding="utf-8") as f:
                        history = json.load(f)
                except Exception as e:
                    logger.warning("Feature importance history load failed: %s", e)
                    history = []
            # Compare to most recent previous snapshot
            stability_report = {}
            if history:
                prev = history[-1].get("top10", {})
                for pos in POSITIONS:
                    curr_feats = set(f["feature"] for f in current_top10.get(pos, []))
                    prev_feats = set(f["feature"] for f in prev.get(pos, []))
                    if curr_feats and prev_feats:
                        overlap = len(curr_feats & prev_feats)
                        overlap_pct = overlap / max(len(curr_feats), 1) * 100
                        stability_report[pos] = {
                            "overlap_pct": round(overlap_pct, 1),
                            "stable": overlap_pct >= 60.0,
                            "new_features": sorted(curr_feats - prev_feats),
                            "dropped_features": sorted(prev_feats - curr_feats),
                        }
                        status = "STABLE" if overlap_pct >= 60 else "DRIFT"
                        print(f"  {pos} feature stability: {overlap_pct:.0f}% overlap ({status})")
                    else:
                        stability_report[pos] = {"overlap_pct": None, "stable": True, "new_features": [], "dropped_features": []}
            # Append current snapshot to history (keep last 10 runs)
            history.append({
                "date": datetime.now().isoformat(),
                "feature_version": FEATURE_VERSION.strip(),
                "train_seasons": train_seasons,
                "top10": current_top10,
                "stability": stability_report,
            })
            history = history[-10:]  # Keep last 10 snapshots
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, default=str)
            print(f"  Feature importance history updated ({len(history)} snapshots)")
    except Exception as e:
        print(f"  Feature stability tracking skipped: {e}")

    # Report QB target choice if present
    qb_choice_path = MODELS_DIR / QB_TARGET_CHOICE_FILENAME
    if qb_choice_path.exists():
        with open(qb_choice_path) as f:
            qb_choice = json.load(f)
        print(f"\nQB target: {qb_choice.get('qb_target', 'util')}")
    
    # Evaluate on test data
    print("\n[5/5] Evaluating on test data...")
    held_out_test_metrics = {}
    if len(test_data) > 0:
        print(f"  Test season: {actual_test_season}")
        print(f"  Test records: {len(test_data)}")
        held_out_test_metrics = _report_test_metrics(trainer, test_data, train_data)
        _run_backtest_after_training(trainer, test_data, train_seasons, actual_test_season,
                                     train_data=train_data)

    # Rolling-origin time-series CV validation report (skip in fast mode)
    # This satisfies the rolling-origin CV requirement: train on seasons 1..N-1,
    # validate on season N, across multiple folds with season-based splits.
    if not fast:
        print("\n[6/6] Rolling-origin time-series CV validation (Ridge baseline)...")
        _run_robust_cv_report(train_data)
    else:
        print("\n[6/6] Rolling-origin CV validation skipped (fast mode)")
    
    # Save label baseline for drift detection
    try:
        from src.evaluation.monitoring import ModelMonitor
        target_cols = [c for c in train_data.columns if c.startswith("target_") and "util" not in c]
        if "fantasy_points" in train_data.columns:
            target_cols = ["fantasy_points"] + target_cols
        for col in target_cols[:3]:  # Save baselines for top targets
            vals = train_data[col].dropna().values
            if len(vals) > 50:
                baseline_path = MODELS_DIR / f"label_baseline_{col}.json"
                ModelMonitor.save_label_baseline(vals, baseline_path)
        # Also save a generic label baseline
        if "fantasy_points" in train_data.columns:
            fp_vals = train_data["fantasy_points"].dropna().values
            if len(fp_vals) > 50:
                ModelMonitor.save_label_baseline(fp_vals, MODELS_DIR / "label_baseline.json")
                print(f"  Label baseline saved for drift detection (Section 18.3)")
    except Exception as e:
        logger.warning("Label baseline save failed: %s", e)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Train seasons: {train_seasons}")
    print(f"Test season: {actual_test_season}")
    summary = trainer.get_training_summary()
    print(summary.to_string(index=False))

    # Train preseason projector (season-total model for draft tool)
    print("\nTraining preseason projector (season-total model)...")
    try:
        from src.models.preseason_projector import PreseasonProjector
        proj, pairs_df = PreseasonProjector.train(seasons=train_seasons + [actual_test_season])
        projector_path = MODELS_DIR / "preseason_projector.json"
        proj.save(projector_path)
        for pos, model in proj.models.items():
            n = int((pairs_df["position"] == pos).sum())
            coef_top = sorted(zip(proj.feature_names[pos], model.coef_.tolist()), key=lambda x: abs(x[1]), reverse=True)[:3]
            print(f"  {pos}: {n} samples, top features: {coef_top}")
        print(f"  Preseason projector saved to {projector_path}")
    except Exception as e:
        print(f"  WARNING: Preseason projector training failed: {e}")

    print("\nModels saved to:", MODELS_DIR)
    # Persist feature version so prediction path can detect stale (old-feature) models
    version_path = MODELS_DIR / FEATURE_VERSION_FILENAME
    version_path.write_text(FEATURE_VERSION.strip(), encoding="utf-8")
    print(f"Feature version written: {FEATURE_VERSION_FILENAME} = {FEATURE_VERSION}")
    # Model metadata for versioning and monitoring (training date, feature version)
    metadata_path = MODELS_DIR / "model_metadata.json"
    try:
        # Load previous metadata for rollback tracking
        prev_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    prev_metadata = json.load(f)
            except Exception as e:
                logger.warning("Previous metadata load failed: %s", e)

        # Collect per-position OOF metrics (from cross-validation during training)
        oof_metrics = {}
        for pos, m in trainer.training_metrics.items():
            oof_metrics[pos] = m

        # Archive previous metadata for rollback (keep last 5 versions)
        version_history_path = MODELS_DIR / "model_version_history.json"
        version_history = []
        if version_history_path.exists():
            try:
                with open(version_history_path, encoding="utf-8") as f:
                    version_history = json.load(f)
            except Exception as e:
                logger.warning("Version history load failed: %s", e)
                version_history = []
        if prev_metadata.get("training_date"):
            version_history.append(prev_metadata)
            version_history = version_history[-5:]  # Keep last 5 versions for rollback
            with open(version_history_path, "w", encoding="utf-8") as f:
                json.dump(version_history, f, indent=2, default=str)
            print(f"  Archived previous model version ({len(version_history)} versions available for rollback)")

        metadata = {
            "training_date": datetime.now().isoformat(),
            "feature_version": FEATURE_VERSION.strip(),
            "train_seasons": train_seasons,
            "test_season": actual_test_season,
            "positions_trained": list(trainer.trained_models.keys()),
            "oof_metrics": oof_metrics,
            "test_metrics": held_out_test_metrics,
            "n_features_per_position": {
                pos: (
                    len(getattr(trainer.component_predictors.get(pos), "feature_names", []))
                    if m is None
                    else len(getattr(
                        (m.models.get(1) or list(m.models.values())[0]) if hasattr(m, "models") else m,
                        "feature_names", []
                    ))
                )
                for pos, m in trainer.trained_models.items()
            },
            "previous_training_date": prev_metadata.get("training_date"),
            "previous_feature_version": prev_metadata.get("feature_version"),
            "rollback_available": bool(prev_metadata.get("training_date")),
            "n_rollback_versions": len(version_history),
            "horizon_status_file": str(MODELS_DIR / "horizon_model_status.json"),
            "bounded_scaler_file": str(MODELS_DIR / "feature_scaler_bounded.joblib"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Model metadata written: {metadata_path.name}")

        # Unified monitoring artifact: model version + performance + explainability pointers.
        top10_path = MODELS_DIR / "top10_features_per_position.json"
        top10_payload = {}
        if top10_path.exists():
            try:
                with open(top10_path, encoding="utf-8") as f:
                    top10_payload = json.load(f)
            except Exception as e:
                logger.warning("Top-10 features load failed: %s", e)
                top10_payload = {}
        # Load feature stability data if available
        feature_stability = {}
        try:
            hist_path = MODELS_DIR / "feature_importance_history.json"
            if hist_path.exists():
                with open(hist_path, encoding="utf-8") as f:
                    hist = json.load(f)
                if hist:
                    feature_stability = hist[-1].get("stability", {})
        except Exception as e:
            logger.warning("Feature stability data load failed: %s", e)

        monitoring_summary = {
            "generated_at": datetime.now().isoformat(),
            "feature_version": FEATURE_VERSION.strip(),
            "metadata_file": str(metadata_path),
            "training_metadata": metadata,
            "top10_features_per_position": top10_payload,
            "feature_importance_stability": feature_stability,
            "drift_threshold_pct": float(MODEL_CONFIG.get("drift_threshold_pct", 20.0)),
            "retraining_config": {
                "schedule": RETRAINING_CONFIG.get("retrain_day", "Tuesday"),
                "auto_retrain": RETRAINING_CONFIG.get("auto_retrain", True),
                "degradation_threshold_pct": RETRAINING_CONFIG.get("degradation_threshold_pct", 20.0),
            },
        }
        monitoring_path = MODELS_DIR / "model_monitoring_report.json"
        with open(monitoring_path, "w", encoding="utf-8") as f:
            json.dump(monitoring_summary, f, indent=2, default=str)
        print(f"Monitoring report written: {monitoring_path.name}")
    except Exception as e:
        print(f"Model metadata write skipped: {e}")
    # Log explicit dataset hashes and gold artifact lineage
    try:
        train_lineage_parents = [
            i for i in [get_artifact_id(train_data)] + find_artifact_ids(
                layer="bronze",
                source="nflverse_weekly_stats",
                seasons=train_seasons,
            ) if i
        ]
        test_lineage_parents = [
            i for i in [get_artifact_id(test_data)] + find_artifact_ids(
                layer="bronze",
                source="nflverse_weekly_stats",
                seasons=[actual_test_season],
            ) if i
        ]
        tracker.log_dataset_hash(
            experiment_run_id,
            train_data,
            label="training_matrix_gold",
            parent_artifact_ids=train_lineage_parents,
        )
        tracker.log_dataset_hash(
            experiment_run_id,
            test_data,
            label="holdout_matrix_gold",
            parent_artifact_ids=test_lineage_parents,
        )

        target_definition = {
            "1w": "target_util_1w for non-QB; QB trainer-selected util/fp",
            "4w": "target_util_4w preferred over target_4w",
            "18w": "target_util_18w preferred over target_18w",
        }
        gold_train_meta = persist_dataframe_artifact(
            train_data.copy(),
            layer="gold",
            table="model_training_matrix",
            run_id=experiment_run_id,
            metadata={
                "source": "src.models.train.train_models",
                "seasons": train_seasons,
                "feature_version": FEATURE_VERSION.strip(),
                "target_definition": target_definition,
                "pulled_at": utc_now_iso(),
            },
            parent_artifact_ids=train_lineage_parents,
        )
        tracker.log_params(experiment_run_id, {"gold_training_artifact_id": gold_train_meta["artifact_id"]})
    except Exception as e:
        logger.warning("Dataset hash/artifact lineage logging failed: %s", e)

    # Log experiment metrics and end tracking run
    try:
        tracker.log_params(experiment_run_id, {
            "train_seasons": train_seasons,
            "test_season": actual_test_season,
            "n_train_records": len(train_data),
            "n_test_records": len(test_data),
            "positions_trained": list(trainer.trained_models.keys()),
        })
        tracker.end_run(experiment_run_id)
        print(f"Experiment {experiment_run_id} logged to {tracker.log_file}")
    except Exception as e:
        logger.warning("Experiment tracking finalization failed: %s", e)

    print("Training complete!")

    return trainer, train_data, test_data, actual_test_season


def main():
    parser = argparse.ArgumentParser(description="Train NFL prediction models")
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="Positions to train (e.g., QB RB WR TE)"
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning (use defaults)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials for hyperparameter tuning"
    )
    parser.add_argument(
        "--test-season",
        type=int,
        default=None,
        help="Override test season (default: latest available)"
    )
    parser.add_argument(
        "--optimize-years",
        action="store_true",
        help="Dynamically select optimal training years per position"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation (train on 1..N-1, test on N for last 4 seasons); report mean +/- std RMSE/MAE"
    )
    parser.add_argument(
        "--strict-requirements",
        action="store_true",
        help="Fail training when minimum data requirements are not met (seasons/player counts)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast training mode: ~8-10x faster with minimal accuracy loss. "
             "Reduces Optuna trials, CV folds, stability bootstrap, LSTM/deep epochs, "
             "and skips SHAP/PDP and robust CV report."
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="Skip the data integrity gate (not recommended). Use when the gate "
             "fails on stale cache and you want to retrain with fresh data loading."
    )
    parser.add_argument(
        "--loyo-backtest",
        action="store_true",
        help="Run LOYO walk-forward backtest: train fresh per season, "
             "test on each of 2018-2024. Reports aggregate metrics + stability."
    )
    parser.add_argument(
        "--loyo-seasons",
        nargs="+",
        type=int,
        default=None,
        help="Override test seasons for LOYO backtest (e.g., 2020 2021 2022)"
    )
    args = parser.parse_args()

    train_models(
        positions=args.positions,
        tune_hyperparameters=not args.no_tune,
        n_trials=args.trials,
        test_season=args.test_season,
        optimize_training_years=args.optimize_years,
        walk_forward=args.walk_forward,
        strict_requirements=args.strict_requirements,
        fast=args.fast,
        skip_cache_check=args.skip_cache_check,
        loyo_backtest=args.loyo_backtest,
        loyo_seasons=args.loyo_seasons,
    )


if __name__ == "__main__":
    main()

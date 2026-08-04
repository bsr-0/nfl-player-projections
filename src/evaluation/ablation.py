"""Ablation study: remove key features to test if model has real edge.

The two features under investigation are:
  - **season_position_rank**: Prior-season finishing rank (44% feature importance).
  - **utilization_score** (and all derived rolling/lag/ewm variants): Custom
    utilization metric (20% feature importance).

Together these features dominate the model.  If removing them causes only a
small performance drop, the model has learned genuine signal from other features.
If performance collapses, the model is essentially a repackaged ADP/utilization
lookup with no real edge.

Usage:
    python -m src.evaluation.ablation [--fast] [--positions QB RB WR TE]

The script:
  1. Loads the same training/test data as the main pipeline.
  2. Runs the full feature engineering pipeline.
  3. Trains three model variants:
     a. **full**: All features (baseline for comparison).
     b. **no_rank**: season_position_rank removed.
     c. **no_util**: utilization_score and all derived features removed.
     d. **no_rank_no_util**: Both removed.
  4. Reports RMSE/MAE/R² per position for each variant and the delta vs full.
  5. Saves results to data/models/ablation_results.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import POSITIONS, MODELS_DIR, DATA_DIR, MODEL_CONFIG, FAST_MODEL_CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature identification helpers
# ---------------------------------------------------------------------------

# Patterns that match utilization_score and all derived rolling/lag/ewm features
_UTIL_PATTERNS = [
    r"^utilization_score$",
    r"^utilization_score_",   # utilization_score_roll3_mean, utilization_score_lag_1, etc.
    r"_utilization_score$",   # position_mean_utilization_score, etc.
    r"_utilization_score_",   # player_mean_utilization_score_roll3, etc.
    r"^util_",                # util_ewm_3, util_volatility_roll5, etc.
    r"_util_ewm",
    r"_util_roll",
    r"_util_lag",
    r"utilization_vol",
    r"util_regression_to_mean",
    r"pos_util_expanding_mean",
    r"player_util_ewm",
]

_RANK_PATTERNS = [
    r"^season_position_rank$",
    r"season_position_rank_",  # any derived from it
    r"_season_position_rank",
    r"^position_rank$",
    r"^estimated_adp$",
    r"^projected_adp$",
]


def identify_utilization_columns(columns: List[str]) -> List[str]:
    """Return column names related to utilization_score."""
    matched = []
    for col in columns:
        col_lower = col.lower()
        for pat in _UTIL_PATTERNS:
            if re.search(pat, col_lower):
                matched.append(col)
                break
    return sorted(set(matched))


def identify_rank_columns(columns: List[str]) -> List[str]:
    """Return column names related to season_position_rank / ADP."""
    matched = []
    for col in columns:
        col_lower = col.lower()
        for pat in _RANK_PATTERNS:
            if re.search(pat, col_lower):
                matched.append(col)
                break
    return sorted(set(matched))


# ---------------------------------------------------------------------------
# Ablation variant training
# ---------------------------------------------------------------------------

def _train_ablation_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    positions: List[str],
    drop_columns: List[str],
    variant_name: str,
    tune: bool = False,
    n_trials: int = 15,
) -> Dict[str, Dict[str, float]]:
    """Train a model variant with specified columns removed and evaluate.

    Returns per-position metrics dict: {pos: {rmse, mae, r2, n_test}}.
    """
    from src.models.ensemble import ModelTrainer

    train = train_df.copy()
    test = test_df.copy()

    # Drop the ablated columns
    existing_drops = [c for c in drop_columns if c in train.columns]
    if existing_drops:
        train = train.drop(columns=existing_drops)
        test = test.drop(columns=[c for c in existing_drops if c in test.columns])

    print(f"\n  [{variant_name}] Dropped {len(existing_drops)} columns, "
          f"{len(train.columns)} features remaining")

    trainer = ModelTrainer()
    trainer.train_all_positions(
        train, positions=positions, tune_hyperparameters=tune,
        n_weeks_list=[1], test_data=None,
    )

    results: Dict[str, Dict[str, float]] = {}
    for position in positions:
        if position not in trainer.trained_models:
            results[position] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_test": 0}
            continue

        multi_model = trainer.trained_models[position]
        if multi_model is None:
            results[position] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_test": 0}
            continue
        pos_test = test[test["position"] == position].copy()
        if len(pos_test) < 10:
            results[position] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_test": len(pos_test)}
            continue

        model = multi_model.models.get(1) or list(multi_model.models.values())[0]
        medians = getattr(model, "feature_medians", {})
        for fn in getattr(model, "feature_names", []):
            if fn not in pos_test.columns:
                pos_test[fn] = medians.get(fn, 0)

        # Get target column (prefer target_1w for FP, fallback to target_util_1w)
        target_col = None
        for candidate in ["target_1w", "target_util_1w"]:
            if candidate in pos_test.columns and pos_test[candidate].notna().sum() >= 10:
                target_col = candidate
                break
        if target_col is None:
            results[position] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_test": len(pos_test)}
            continue

        y_true = pos_test[target_col]
        valid = y_true.notna()
        if valid.sum() < 10:
            results[position] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_test": int(valid.sum())}
            continue

        preds = multi_model.predict(pos_test.loc[valid], n_weeks=1)
        y = y_true[valid].values
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))

        results[position] = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "n_test": int(valid.sum()),
        }
        print(f"    {position}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}  (n={valid.sum()})")

    return results


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation_study(
    positions: Optional[List[str]] = None,
    fast: bool = True,
    test_season: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full ablation study and return results.

    Returns a dict with keys: "full", "no_rank", "no_util", "no_rank_no_util",
    each containing per-position metrics + a "summary" with deltas.
    """
    from src.models.data_loading import load_training_data
    from src.models.feature_preparation import _prepare_training_data

    if fast:
        for key, val in FAST_MODEL_CONFIG.items():
            MODEL_CONFIG[key] = val

    positions = positions or POSITIONS
    n_trials = MODEL_CONFIG.get("n_optuna_trials", 15)

    print("=" * 70)
    print("ABLATION STUDY: Testing model edge without key features")
    print("=" * 70)

    # Load and prepare data (shared across all variants)
    print("\n[1/5] Loading data...")
    train_data, test_data, train_seasons, actual_test_season = load_training_data(
        positions, test_season=test_season,
    )

    print("\n[2/5] Preparing features (shared pipeline)...")
    train_data, test_data, _ = _prepare_training_data(
        train_data, test_data, positions,
        tune_hyperparameters=False, n_trials=n_trials, fast=fast,
    )

    # Identify columns to ablate
    all_cols = list(train_data.columns)
    util_cols = identify_utilization_columns(all_cols)
    rank_cols = identify_rank_columns(all_cols)
    print(f"\n  Utilization-related columns to ablate ({len(util_cols)}): {util_cols[:10]}{'...' if len(util_cols) > 10 else ''}")
    print(f"  Rank-related columns to ablate ({len(rank_cols)}): {rank_cols[:10]}{'...' if len(rank_cols) > 10 else ''}")

    results: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "test_season": actual_test_season,
        "train_seasons": train_seasons,
        "ablated_util_columns": util_cols,
        "ablated_rank_columns": rank_cols,
    }

    # Variant A: Full model (all features)
    print("\n[3/5] Training FULL model (all features)...")
    results["full"] = _train_ablation_variant(
        train_data, test_data, positions, drop_columns=[],
        variant_name="full", tune=False, n_trials=n_trials,
    )

    # Variant B: No season_position_rank
    print("\n[4a/5] Training NO_RANK model (season_position_rank removed)...")
    results["no_rank"] = _train_ablation_variant(
        train_data, test_data, positions, drop_columns=rank_cols,
        variant_name="no_rank", tune=False, n_trials=n_trials,
    )

    # Variant C: No utilization_score
    print("\n[4b/5] Training NO_UTIL model (utilization_score removed)...")
    results["no_util"] = _train_ablation_variant(
        train_data, test_data, positions, drop_columns=util_cols,
        variant_name="no_util", tune=False, n_trials=n_trials,
    )

    # Variant D: Neither
    print("\n[4c/5] Training NO_RANK_NO_UTIL model (both removed)...")
    results["no_rank_no_util"] = _train_ablation_variant(
        train_data, test_data, positions, drop_columns=rank_cols + util_cols,
        variant_name="no_rank_no_util", tune=False, n_trials=n_trials,
    )

    # Summary: compute deltas
    print("\n[5/5] Computing ablation deltas...")
    summary = _compute_ablation_summary(results, positions)
    results["summary"] = summary

    # Save results
    out_path = MODELS_DIR / "ablation_results.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved ablation results: {out_path}")
    except Exception as e:
        print(f"  Ablation results save failed: {e}")

    # Print summary report
    print("\n" + format_ablation_report(results, positions))

    return results


def _compute_ablation_summary(
    results: Dict[str, Any],
    positions: List[str],
) -> Dict[str, Any]:
    """Compute per-position RMSE deltas for each ablation variant vs full."""
    summary: Dict[str, Any] = {}
    full = results.get("full", {})

    for variant in ["no_rank", "no_util", "no_rank_no_util"]:
        variant_data = results.get(variant, {})
        pos_deltas = {}
        for pos in positions:
            full_rmse = full.get(pos, {}).get("rmse")
            var_rmse = variant_data.get(pos, {}).get("rmse")
            if full_rmse and var_rmse and np.isfinite(full_rmse) and np.isfinite(var_rmse):
                delta = var_rmse - full_rmse
                pct = (delta / full_rmse) * 100 if full_rmse > 0 else 0.0
                pos_deltas[pos] = {
                    "rmse_delta": round(delta, 4),
                    "rmse_pct_change": round(pct, 2),
                    "feature_contributes": delta > 0.05,  # Feature removal hurt performance
                }
            else:
                pos_deltas[pos] = {"rmse_delta": None, "rmse_pct_change": None, "feature_contributes": None}
        summary[variant] = pos_deltas

    # Overall verdict
    both_removed = summary.get("no_rank_no_util", {})
    large_degradation = any(
        (v.get("rmse_pct_change") or 0) > 15.0
        for v in both_removed.values()
        if v.get("rmse_pct_change") is not None
    )
    minimal_degradation = all(
        abs(v.get("rmse_pct_change") or 0) < 5.0
        for v in both_removed.values()
        if v.get("rmse_pct_change") is not None
    )

    if minimal_degradation:
        summary["verdict"] = (
            "MODEL HAS REAL EDGE: Removing season_position_rank and utilization_score "
            "causes <5% RMSE degradation. The model learns genuine signal from other features."
        )
    elif large_degradation:
        summary["verdict"] = (
            "MODEL DEPENDS HEAVILY ON RANK/UTIL: Removing these features causes >15% RMSE "
            "degradation for at least one position. The model may be a sophisticated "
            "ADP/utilization lookup rather than learning novel patterns."
        )
    else:
        summary["verdict"] = (
            "MIXED RESULTS: Some positions degrade moderately (5-15%) without rank/util features. "
            "The model has partial edge beyond these features but also relies on them."
        )

    return summary


def format_ablation_report(results: Dict[str, Any], positions: List[str]) -> str:
    """Format ablation results as a readable text report."""
    lines = [
        "=" * 70,
        "ABLATION STUDY RESULTS",
        "=" * 70,
        "",
    ]

    # Table header
    lines.append(f"  {'Position':<10} {'Full RMSE':<12} {'No Rank':<12} {'No Util':<12} {'No Both':<12}")
    lines.append(f"  {'-'*10:<10} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")

    full = results.get("full", {})
    no_rank = results.get("no_rank", {})
    no_util = results.get("no_util", {})
    no_both = results.get("no_rank_no_util", {})

    for pos in positions:
        f_rmse = full.get(pos, {}).get("rmse")
        nr_rmse = no_rank.get(pos, {}).get("rmse")
        nu_rmse = no_util.get(pos, {}).get("rmse")
        nb_rmse = no_both.get(pos, {}).get("rmse")

        def _fmt(val, ref):
            if val is None or not np.isfinite(val):
                return "N/A"
            if ref and np.isfinite(ref) and ref > 0:
                pct = (val - ref) / ref * 100
                return f"{val:.3f} ({pct:+.1f}%)"
            return f"{val:.3f}"

        lines.append(
            f"  {pos:<10} "
            f"{_fmt(f_rmse, None):<12} "
            f"{_fmt(nr_rmse, f_rmse):<12} "
            f"{_fmt(nu_rmse, f_rmse):<12} "
            f"{_fmt(nb_rmse, f_rmse):<12}"
        )

    lines.append("")

    # R² comparison
    lines.append(f"  {'Position':<10} {'Full R²':<12} {'No Rank':<12} {'No Util':<12} {'No Both':<12}")
    lines.append(f"  {'-'*10:<10} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12} {'-'*12:<12}")
    for pos in positions:
        f_r2 = full.get(pos, {}).get("r2")
        nr_r2 = no_rank.get(pos, {}).get("r2")
        nu_r2 = no_util.get(pos, {}).get("r2")
        nb_r2 = no_both.get(pos, {}).get("r2")

        def _fmt_r2(val):
            if val is None or not np.isfinite(val):
                return "N/A"
            return f"{val:.3f}"

        lines.append(
            f"  {pos:<10} "
            f"{_fmt_r2(f_r2):<12} "
            f"{_fmt_r2(nr_r2):<12} "
            f"{_fmt_r2(nu_r2):<12} "
            f"{_fmt_r2(nb_r2):<12}"
        )

    lines.append("")

    # Verdict
    summary = results.get("summary", {})
    verdict = summary.get("verdict", "No verdict available.")
    lines.append("-" * 70)
    lines.append(f"VERDICT: {verdict}")
    lines.append("-" * 70)

    # Feature-level contribution
    lines.append("")
    lines.append("Feature contribution by position:")
    for variant_name, label in [("no_rank", "season_position_rank"), ("no_util", "utilization_score")]:
        variant_summary = summary.get(variant_name, {})
        for pos in positions:
            delta = variant_summary.get(pos, {})
            pct = delta.get("rmse_pct_change")
            contributes = delta.get("feature_contributes")
            if pct is not None:
                status = "contributes" if contributes else "redundant"
                lines.append(f"  {label:<30} {pos}: {pct:+.1f}% RMSE change ({status})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: test if model has real edge without key features"
    )
    parser.add_argument(
        "--positions", nargs="+", default=None,
        help="Positions to ablate (default: all)",
    )
    parser.add_argument(
        "--fast", action="store_true", default=True,
        help="Use fast training config (default: True)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full training config (overrides --fast)",
    )
    parser.add_argument(
        "--test-season", type=int, default=None,
        help="Override test season",
    )
    args = parser.parse_args()

    run_ablation_study(
        positions=args.positions,
        fast=not args.full,
        test_season=args.test_season,
    )


if __name__ == "__main__":
    main()

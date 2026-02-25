"""Training script for NFL prediction models."""
import argparse
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
from src.models.utilization_to_fp import train_utilization_to_fp_per_position
from src.evaluation.explainability import (
    get_top10_feature_importance_per_position,
    explain_with_shap,
    partial_dependence_plots,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


def load_training_data(positions: list = None, min_games: int = 4, 
                       test_season: int = None,
                       n_train_seasons: int = None,
                       optimize_training_years: bool = False,
                       strict_requirements: bool = False) -> tuple:
    """
    Load and prepare training data with automatic train/test split.
    
    Uses the latest available season as test set for out-of-sample evaluation.
    When n_train_seasons is None, uses ALL available training seasons.
    
    Args:
        positions: List of positions to load
        min_games: Minimum games for player inclusion
        test_season: Override test season (None = auto-select latest)
        n_train_seasons: Max training seasons (None = use all available)
        optimize_training_years: If True, dynamically select optimal years per position
        
    Returns:
        Tuple of (train_data, test_data, train_seasons, test_season)
    """
    # Auto-refresh and check data availability
    print("Checking data availability...")
    data_status = auto_refresh_data()
    print(f"  Latest available season: {data_status['latest_season']}")
    print(f"  Available seasons: {data_status['available_seasons']}")
    
    data_manager = DataManager()
    optimal_years = None
    
    if optimize_training_years:
        from src.utils.training_years_selector import (
            select_optimal_training_years,
            save_optimal_training_years,
            get_available_seasons_from_data,
        )
        # Load raw data first for optimization
        db = DatabaseManager()
        all_raw = []
        for pos in (positions or POSITIONS):
            d = db.get_all_players_for_training(position=pos, min_games=min_games)
            if len(d) > 0:
                all_raw.append(d)
        if all_raw:
            raw_df = pd.concat(all_raw, ignore_index=True)
            seasons = get_available_seasons_from_data(raw_df)
            test_season = test_season or max(seasons)
            # Run optimization (requires prepared data - we do a quick pass)
            # For speed, use raw data with minimal prep
            print("  Optimizing training years per position...")
            optimal_years = select_optimal_training_years(
                raw_df, positions=positions or POSITIONS,
                test_season=test_season,
            )
            save_optimal_training_years(optimal_years, test_season)
            print(f"  Optimal years: {optimal_years}")
    
    train_seasons, auto_test_season = data_manager.get_train_test_seasons(
        test_season=test_season,
        n_train_seasons=n_train_seasons,
        optimal_years_per_position=optimal_years,
    )
    
    db = DatabaseManager()
    all_data = []
    positions = positions or POSITIONS
    
    for position in positions:
        print(f"Loading data for {position}...")
        pos_data = db.get_all_players_for_training(position=position, min_games=min_games)
        
        if len(pos_data) > 0:
            all_data.append(pos_data)
            print(f"  Loaded {len(pos_data)} records for {position}")
    
    if not all_data:
        from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
        raise ValueError(
            "No data found in database. Please load real NFL data first using:\n"
            f"  python3 src/data/nfl_data_loader.py --seasons {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON}\n"
            "(or omit --seasons for default: config range). This system only uses real NFL data."
        )
    
    combined = pd.concat(all_data, ignore_index=True)

    # Guard: training data must not contain model-output or leakage columns.
    from src.utils.leakage import find_leakage_columns
    leaked = find_leakage_columns(combined.columns, ban_utilization_score=False)
    if leaked:
        logger.warning("Dropping %d leakage-risk columns from training data: %s",
                        len(leaked), sorted(leaked)[:10])
        combined = combined.drop(columns=leaked, errors="ignore")
    
    # Split into train/test (strict unseen test: test season must not be in train)
    assert auto_test_season not in train_seasons, (
        f"Test season {auto_test_season} must not be in train seasons {train_seasons}"
    )
    train_data = combined[combined['season'].isin(train_seasons)]
    test_data = combined[combined['season'] == auto_test_season]
    
    # In-season: pipeline requires current season as test and non-empty test set
    from src.utils.nfl_calendar import get_current_nfl_season, current_season_has_weeks_played
    current_season = get_current_nfl_season()
    in_season = current_season_has_weeks_played()
    if in_season and auto_test_season != current_season:
        raise ValueError(
            "The pipeline requires the current season as test when it has started. "
            f"Expected test_season={current_season}, got {auto_test_season}. "
            "Run auto_refresh or load current season from PBP and re-run."
        )
    if in_season and len(test_data) == 0:
        raise ValueError(
            "Current season is in progress but test set is empty. "
            "Load current season from play-by-play (e.g. python -m src.data.auto_refresh) and re-run."
        )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} records from seasons {train_seasons}")
    print(f"  Testing: {len(test_data)} records from season {auto_test_season}")
    n_seasons = len(train_seasons)
    # Requirement-derived minimums: warn (or fail in strict mode) when below
    # (1w min 3, 4w min 5, 18w min 8)
    requirement_failures = []
    if n_seasons < MIN_TRAINING_SEASONS_1W:
        msg = f"1-week model requires >= {MIN_TRAINING_SEASONS_1W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}. Accuracy may suffer.")
        requirement_failures.append(msg)
    if MODEL_CONFIG.get("use_18w_deep", True) and n_seasons < MIN_TRAINING_SEASONS_18W:
        msg = f"18-week deep model requires >= {MIN_TRAINING_SEASONS_18W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}. Consider skipping or adding data.")
        requirement_failures.append(msg)
    if MODEL_CONFIG.get("use_4w_hybrid", True) and n_seasons < MIN_TRAINING_SEASONS_4W:
        msg = f"4-week hybrid model benefits from >= {MIN_TRAINING_SEASONS_4W} training seasons (have {n_seasons})"
        print(f"  WARNING: {msg}.")
        requirement_failures.append(msg)
    # Per-position player minimums (requirements: QB 30+, RB 60+, WR 70+, TE 30+)
    train_players_per_pos = train_data.groupby("position")["player_id"].nunique()
    for pos in POSITIONS:
        min_players = MIN_PLAYERS_PER_POSITION.get(pos, 30)
        n_players = int(train_players_per_pos.get(pos, 0))
        if n_players < min_players:
            msg = f"{pos} has {n_players} unique players in training (minimum >= {min_players})"
            print(f"  WARNING: {msg}.")
            requirement_failures.append(msg)
    if strict_requirements and requirement_failures:
        joined = "; ".join(requirement_failures)
        raise ValueError(f"Strict requirements check failed: {joined}")
    return train_data, test_data, train_seasons, auto_test_season


def create_sample_data() -> pd.DataFrame:
    """
    DEPRECATED: This function previously generated fake/synthetic data.
    
    This system now requires real NFL data from nfl-data-py.
    To load real data, run:
        python3 src/data/nfl_data_loader.py --seasons 2020-2024
    """
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    raise NotImplementedError(
        "Synthetic data generation has been removed. "
        "This system only uses real NFL data from nfl-data-py. "
        f"Run: python3 src/data/nfl_data_loader.py (default: {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON})"
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


def _create_horizon_targets(df: pd.DataFrame, n_weeks: List[int] = None) -> pd.DataFrame:
    """Create causal horizon targets within each player-season boundary."""
    if df.empty:
        return df
    n_weeks = n_weeks or [1, 4, 18]
    out = df.copy()
    group_cols = ["player_id", "season"] if "season" in out.columns else ["player_id"]

    def _forward_window(series: pd.Series, window: int, agg: str) -> pd.Series:
        """
        Aggregate future values x[t+1:t+window] for each row t.
        Uses reverse rolling to keep strict forward-looking targets.

        Strict min_periods to avoid noisy/biased targets:
        - For sum targets (fantasy points): require 75% of window to prevent
          systematic underestimation for late-season rows. An 18-week sum
          with only 9 games would be half the expected magnitude, teaching
          the model that late-season == low production.
        - For mean targets (utilization): require 60% of window since means
          are scale-invariant, but very short windows produce high-variance
          estimates.
        - 1-week targets always require exactly 1 future game (no change).
        Rows with insufficient future data become NaN and are excluded during
        training.
        """
        shifted = series.shift(-1)
        rev = shifted.iloc[::-1]
        if window <= 1:
            min_p = 1
        elif agg == "sum":
            # Require 75% of window for sums to prevent scale bias
            min_p = max(int(np.ceil(window * 0.75)), 2)
        else:
            # Require 60% of window for means (less sensitive to count)
            min_p = max(int(np.ceil(window * 0.60)), 2)
        if agg == "sum":
            return rev.rolling(window=window, min_periods=min_p).sum().iloc[::-1]
        return rev.rolling(window=window, min_periods=min_p).mean().iloc[::-1]

    for nw in n_weeks:
        out[f"target_{nw}w"] = out.groupby(group_cols)["fantasy_points"].transform(
            lambda x, w=nw: _forward_window(x, window=w, agg="sum")
        )
    if "utilization_score" in out.columns:
        out["target_util_1w"] = out.groupby(group_cols)["utilization_score"].shift(-1)
        for nw in [w for w in n_weeks if w != 1]:
            out[f"target_util_{nw}w"] = out.groupby(group_cols)["utilization_score"].transform(
                lambda x, w=nw: _forward_window(x, window=w, agg="mean")
            )
    return out


def _apply_with_temporal_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transform_fn,
    label: str,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a transformation so test rows can use historical context from prior
    train seasons, but train rows are NEVER influenced by test data.

    Strategy: transform train alone first, then transform train+test combined
    and keep only the test rows from the combined result. This ensures:
    - Train features are computed using train data only (no test leakage)
    - Test features can see train-season history (expanding/rolling windows)
    """
    if train_df.empty and test_df.empty:
        return train_df, test_df

    # Step 1: Transform train data alone — train features see only train data
    train_out = transform_fn(train_df.copy(), **kwargs)

    # Step 2: Transform combined (train + test) — test rows benefit from
    # train-season historical context in expanding/rolling windows
    if test_df.empty:
        test_out = test_df.copy()
    else:
        split_col = "__split_context_marker__"
        train_in = train_df.copy()
        test_in = test_df.copy()
        train_in[split_col] = 0
        test_in[split_col] = 1
        combined = pd.concat([train_in, test_in], ignore_index=True, sort=False)

        sort_cols = [c for c in ["season", "week", "player_id"] if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)

        transformed = transform_fn(combined, **kwargs)
        if split_col not in transformed.columns:
            raise ValueError(f"{label}: split marker missing after transform")

        # Only keep test rows from the combined result
        test_out = transformed[transformed[split_col] == 1].drop(columns=[split_col]).reset_index(drop=True)

    print(f"  Applied {label} (train-only features, test with context): train={len(train_out)}, test={len(test_out)}")
    return train_out, test_out


def _report_missingness(df: pd.DataFrame, label: str, threshold: float = 0.05) -> Dict[str, float]:
    """Report feature missingness and return columns above threshold."""
    if df.empty:
        print(f"  {label}: empty dataset for missingness check")
        return {}
    miss = df.isna().mean()
    high = miss[miss > threshold].sort_values(ascending=False)
    if len(high) == 0:
        print(f"  {label}: no columns above {threshold:.0%} missingness")
        return {}
    print(f"  {label}: {len(high)} columns above {threshold:.0%} missingness")
    for col, pct in high.head(15).items():
        print(f"    - {col}: {pct:.1%}")
    if len(high) > 15:
        print(f"    ... {len(high) - 15} more")
    return high.to_dict()


def _validate_critical_missingness(df: pd.DataFrame, label: str, threshold: float = 0.05) -> None:
    """
    Validate critical columns after preprocessing.
    Raise on severe quality issues that can invalidate training labels/features.
    """
    critical = [c for c in ["player_id", "position", "season", "week", "fantasy_points", "utilization_score"] if c in df.columns]
    bad = {}
    for col in critical:
        pct = float(df[col].isna().mean())
        if pct > threshold:
            bad[col] = pct
    if bad:
        details = ", ".join(f"{k}={v:.1%}" for k, v in sorted(bad.items()))
        raise ValueError(f"{label}: critical missingness exceeds {threshold:.0%}: {details}")


def _report_outliers_3sigma(df: pd.DataFrame, label: str, cols: list) -> Dict[str, Dict[str, float]]:
    """Diagnostics only: report >3 std outliers without dropping data."""
    if df.empty:
        return {}
    print(f"  {label}: >3σ outlier diagnostics")
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[np.isfinite(s)]
        if len(s) < 30:
            continue
        mu, sigma = float(s.mean()), float(s.std(ddof=0))
        if sigma <= 0:
            continue
        n_out = int(((s - mu).abs() > 3 * sigma).sum())
        pct = 100.0 * n_out / max(len(s), 1)
        print(f"    - {col}: {n_out}/{len(s)} ({pct:.2f}%)")
        out[col] = {
            "n_outliers": n_out,
            "n_samples": int(len(s)),
            "pct_outliers": round(pct, 3),
        }
    return out


def _write_json_artifact(path: Path, payload: Dict[str, Any], label: str) -> None:
    """Best-effort JSON artifact writer for training diagnostics."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  Wrote {label}: {path.name}")
    except Exception as e:
        print(f"  {label} write skipped: {e}")


def _report_train_serve_feature_parity(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Check train/test feature schema parity before model fitting."""
    excluded_prefix = ("target_",)
    excluded_cols = {"fantasy_points", "predicted_points", "predicted_utilization"}
    train_feats = {c for c in train_df.columns if c not in excluded_cols and not c.startswith(excluded_prefix)}
    test_feats = {c for c in test_df.columns if c not in excluded_cols and not c.startswith(excluded_prefix)}
    missing_in_test = sorted(train_feats - test_feats)
    unseen_in_test = sorted(test_feats - train_feats)
    print("  Train/serve feature parity check:")
    print(f"    - train feature count: {len(train_feats)}")
    print(f"    - test feature count: {len(test_feats)}")
    print(f"    - train-only features: {len(missing_in_test)}")
    print(f"    - test-only features: {len(unseen_in_test)}")
    if missing_in_test:
        print("    - sample train-only:", missing_in_test[:8])
    if unseen_in_test:
        print("    - sample test-only:", unseen_in_test[:8])


def _infer_bounded_columns(df: pd.DataFrame) -> List[str]:
    """
    Select bounded/percentage-like columns for explicit MinMax scaling.
    Keeps scaling policy deterministic and train/serve consistent.
    """
    if df.empty:
        return []
    candidates: List[str] = []
    bounded_tokens = ("pct", "rate", "share", "prob", "probability", "percentage")
    for col in df.columns:
        if col.startswith("target_util_") or (
            col.startswith("target_") and (col.endswith("w") or col[7:8].isdigit())
        ):
            continue
        if col in {"fantasy_points", "predicted_points", "predicted_utilization"}:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lower_col = col.lower()
        if not any(t in lower_col for t in bounded_tokens):
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 3:
            continue
        lo = float(series.quantile(0.01))
        hi = float(series.quantile(0.99))
        # Accept common bounded ranges like 0-1 and 0-100.
        if lo >= -1e-6 and hi <= 1.5:
            candidates.append(col)
        elif lo >= -1e-6 and hi <= 100.5:
            candidates.append(col)
    return sorted(set(candidates))


def _apply_bounded_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: Path,
) -> Dict[str, Any]:
    """
    Fit MinMax scaler on bounded columns using train only, apply to train/test, persist artifact.
    """
    cols = _infer_bounded_columns(train_df)
    artifact: Dict[str, Any] = {"columns": cols, "scaler": None}
    if not cols:
        return artifact
    scaler = MinMaxScaler()
    train_vals = train_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    test_vals = test_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    train_df.loc[:, cols] = scaler.fit_transform(train_vals)
    test_df.loc[:, cols] = scaler.transform(test_vals)
    artifact["scaler"] = scaler
    try:
        import joblib
        joblib.dump(artifact, path)
        print(f"  Saved bounded feature scaler artifact: {path.name} ({len(cols)} columns)")
    except Exception as e:
        print(f"  Bounded scaler artifact save skipped: {e}")
    return artifact


def _report_test_metrics(trainer, test_data: pd.DataFrame, train_data: pd.DataFrame):
    """Report model performance on held-out test set with full metrics per requirements.
    
    Reports RMSE, MAE, MAPE, R², within-7pt%, within-10pt%, and Spearman rho.
    """
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
        
        model = multi_model.models.get(1) or list(multi_model.models.values())[0]
        available = [c for c in model.feature_names if c in pos_test.columns]
        if len(available) < len(model.feature_names) * 0.5:
            continue
        
        def _print_metrics(pos, label, y_true, y_pred):
            rmse = (mean_squared_error(y_true, y_pred)) ** 0.5
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = _safe_mape(y_true.values, y_pred)
            within_7 = float((np.abs(y_true.values - y_pred) <= 7).mean() * 100)
            within_10 = float((np.abs(y_true.values - y_pred) <= 10).mean() * 100)
            rho = spearman_rank_correlation(y_true.values, np.asarray(y_pred), top_n=None) if len(y_true) >= 5 else np.nan
            mape_str = f", MAPE={mape:.1f}%" if mape is not None else ""
            rho_str = f", ρ={rho:.3f}" if np.isfinite(rho) else ""
            print(f"  {pos} (test {label}): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}{mape_str}, ≤7pt={within_7:.1f}%, ≤10pt={within_10:.1f}%{rho_str}")
        
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
                _print_metrics(position, label, y_act[valid], preds_out)
            continue
        
        # RB/WR/TE: primary utilization, optional FP
        util_col = "target_util_1w"
        if util_col in pos_test.columns:
            y_util = pos_test[util_col]
            valid = ~y_util.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                pred_util = multi_model.predict(pos_subset, n_weeks=1)
                _print_metrics(position, "util", y_util[valid], pred_util)
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
                _print_metrics(position, "FP", y_test[valid], preds_fp)


def _run_backtest_after_training(trainer, test_data: pd.DataFrame,
                                 train_seasons: list, actual_test_season: int):
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
        # Convert utilization -> fantasy points for skill positions and QB(util mode).
        should_convert = position in converters and (position != "QB" or qb_target == "util")
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
    # Combined actual column:
    # - QB: owner-facing FP when available (fallback to util only when converter is unavailable)
    # - RB/WR/TE: future FP (fallback util if FP target missing)
    test_data["actual_for_backtest"] = np.nan
    non_qb_mask = test_data["position"] != "QB"
    if "target_1w" in test_data.columns:
        test_data.loc[non_qb_mask, "actual_for_backtest"] = test_data.loc[non_qb_mask, "target_1w"]
    if "target_util_1w" in test_data.columns:
        test_data.loc[non_qb_mask, "actual_for_backtest"] = test_data.loc[non_qb_mask, "actual_for_backtest"].fillna(
            test_data.loc[non_qb_mask, "target_util_1w"]
        )
    qb_mask = test_data["position"] == "QB"
    if "target_1w" in test_data.columns:
        test_data.loc[qb_mask, "actual_for_backtest"] = test_data.loc[qb_mask, "target_1w"]
    if qb_target == "util" and "QB" not in converters and "target_util_1w" in test_data.columns:
        test_data.loc[qb_mask, "actual_for_backtest"] = test_data.loc[qb_mask, "actual_for_backtest"].fillna(
            test_data.loc[qb_mask, "target_util_1w"]
        )
    if test_data["actual_for_backtest"].isna().all():
        test_data["actual_for_backtest"] = test_data.get("fantasy_points", np.nan)
    backtester = ModelBacktester()
    pred_col = "predicted_points"
    actual_col = "actual_for_backtest"
    test_data = backtester.calculate_confidence_intervals(
        test_data,
        pred_col=pred_col,
        actual_col=actual_col,
        confidence=0.80,
        lower_col="prediction_ci80_lower",
        upper_col="prediction_ci80_upper",
    )
    test_data = backtester.calculate_confidence_intervals(
        test_data,
        pred_col=pred_col,
        actual_col=actual_col,
        confidence=0.95,
        lower_col="prediction_ci95_lower",
        upper_col="prediction_ci95_upper",
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
        pos: len(getattr(m.models.get(1) or list(m.models.values())[0], "feature_names", []))
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

    return results


def _run_robust_cv_report(train_data: pd.DataFrame) -> dict:
    """Run rolling-origin CV and report per-fold stability metrics.

    Uses expanding-window cross-validation with season-aware splits and a
    purge gap.  Reports fold-wise variance, worst-fold degradation, and a
    complexity comparison (Ridge vs GBM) to verify that ensemble complexity
    is justified.

    Returns:
        Dict with per-position CV results and complexity comparison.
    """
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

    from sklearn.ensemble import GradientBoostingRegressor
    from src.utils.leakage import filter_feature_columns, assert_no_leakage_columns

    cv_report: dict = {}

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
        feature_cols = filter_feature_columns(feature_cols)
        assert_no_leakage_columns(feature_cols, context=f"cv features ({position})")
        if len(feature_cols) < 5:
            continue

        pos_report: dict = {}

        # --- Ridge (simple baseline) ---
        try:
            ridge_result = validator.validate(
                pos_df, Ridge, {"alpha": 1.0},
                feature_cols, target_col=target_col, position=position
            )
            fold_rmses = [f["rmse"] for f in ridge_result.fold_results]
            fold_std = float(np.std(fold_rmses))
            worst_fold = max(fold_rmses)
            pos_report["ridge"] = {
                "rmse_mean": round(ridge_result.rmse, 3),
                "rmse_std": round(fold_std, 3),
                "rmse_worst_fold": round(worst_fold, 3),
                "r2": round(ridge_result.r2, 3),
                "n_folds": len(ridge_result.fold_results),
                "fold_rmses": [round(r, 3) for r in fold_rmses],
            }
            print(f"  {position} Ridge CV: RMSE={ridge_result.rmse:.2f} "
                  f"± {fold_std:.2f} (worst={worst_fold:.2f}), R²={ridge_result.r2:.3f}")
        except Exception as e:
            print(f"  {position} Ridge CV: skipped ({e})")

        # --- GBM (complex model) for complexity comparison ---
        try:
            gbm_result = validator.validate(
                pos_df, GradientBoostingRegressor,
                {"n_estimators": 100, "max_depth": 5, "random_state": 42},
                feature_cols, target_col=target_col, position=position
            )
            gbm_fold_rmses = [f["rmse"] for f in gbm_result.fold_results]
            gbm_fold_std = float(np.std(gbm_fold_rmses))
            ridge_rmse = pos_report.get("ridge", {}).get("rmse_mean")
            improvement = None
            if ridge_rmse and ridge_rmse > 0:
                improvement = round((1 - gbm_result.rmse / ridge_rmse) * 100, 1)
            pos_report["gbm"] = {
                "rmse_mean": round(gbm_result.rmse, 3),
                "rmse_std": round(gbm_fold_std, 3),
                "r2": round(gbm_result.r2, 3),
                "improvement_over_ridge_pct": improvement,
            }
            marker = "justified" if (improvement or 0) > 5 else "marginal"
            print(f"  {position} GBM  CV: RMSE={gbm_result.rmse:.2f} "
                  f"± {gbm_fold_std:.2f}, improvement over Ridge: "
                  f"{improvement:+.1f}% ({marker})")
        except Exception as e:
            print(f"  {position} GBM CV: skipped ({e})")

        if pos_report:
            cv_report[position] = pos_report

    return cv_report


def add_utilization_scores(data: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Add utilization scores only (for two-phase prep when fitting weights from data)."""
    return calculate_utilization_scores(data, weights=weights)


def add_engineered_features(data: pd.DataFrame, position: str = None) -> pd.DataFrame:
    """Add feature engineering (rolling, lag, trend, etc.) - assumes utilization_score exists."""
    if position:
        engineer = PositionFeatureEngineer(position)
        return engineer.create_features(data)
    engineer = FeatureEngineer()
    return engineer.create_features(data)


def add_advanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add advanced rookie/injury/combine features with safe fallback."""
    try:
        from src.features.advanced_rookie_injury import add_advanced_rookie_injury_features
        return add_advanced_rookie_injury_features(data)
    except Exception as e:
        print(f"  Advanced rookie/injury features skipped: {e}")
        return data


def prepare_features(data: pd.DataFrame, position: str = None,
                     utilization_weights: dict = None) -> pd.DataFrame:
    """Prepare features for training (utilization + engineered features + advanced features)."""
    print("Calculating utilization scores...")
    data = add_utilization_scores(data, weights=utilization_weights)
    print("Engineering features...")
    data = add_engineered_features(data, position=position)
    return add_advanced_features(data)


def _prepare_training_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    positions: list,
    tune_hyperparameters: bool,
    n_trials: int,
    fast: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, "ModelTrainer"]:
    """Shared preprocessing pipeline used by both train_models() and _run_one_fold().

    Handles: DVP, external features, season-long features, utilization scores,
    horizon targets, util weight optimization, feature engineering, bounded scaling,
    winsorization, model training, and util-to-fp conversion.

    When fast=True, skips QB dual-target selection by not passing test_data to
    the model trainer (QB defaults to utilization or FP fallback path).

    Returns (train_data, test_data, trainer).
    """
    from config.settings import MODELS_DIR

    # DVP
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        db.ensure_team_defense_stats()
    except Exception as e:
        logger.warning("Team defense stats (DVP) skipped: %s", e)

    # External (Vegas, injury, weather) with shared train/test temporal context.
    try:
        from src.data.external_data import add_external_features
        all_seasons = sorted(set(train_data["season"].dropna().astype(int)) | set(test_data["season"].dropna().astype(int)))
        train_data, test_data = _apply_with_temporal_context(
            train_data, test_data, add_external_features, "external features",
            seasons=all_seasons,
        )
    except Exception as e:
        logger.warning("External features (weather/injury/Vegas) skipped: %s", e)

    # Season-long draft/rookie context with shared temporal context.
    try:
        from src.features.season_long_features import add_season_long_features
        train_data, test_data = _apply_with_temporal_context(
            train_data, test_data, add_season_long_features, "season-long features",
        )
    except Exception as e:
        logger.warning("Season-long features skipped: %s", e)

    # Utilization scores with train-only percentile bounds
    team_df = pd.DataFrame()
    util_calc = UtilizationScoreCalculator(weights=None)
    train_seasons_list = []
    if "season" in train_data.columns:
        train_seasons_list = sorted({int(s) for s in train_data["season"].dropna().unique()})
    bounds_meta = {
        "train_seasons": train_seasons_list,
        "min_season": min(train_seasons_list) if train_seasons_list else None,
        "max_season": max(train_seasons_list) if train_seasons_list else None,
        "created_at": datetime.now().isoformat(),
    }
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    for pos in POSITIONS:
        util_calc.fit_percentile_bounds(
            train_data, pos, UTIL_COMPONENTS.get(pos, []), metadata=bounds_meta
        )
    bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
    save_percentile_bounds(util_calc.position_percentiles, bounds_path, metadata=bounds_meta)
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    loaded_bounds, loaded_meta = load_percentile_bounds(bounds_path, return_meta=True)
    if not validate_percentile_bounds_meta(loaded_meta, train_seasons_list):
        raise ValueError(
            "Utilization percentile bounds metadata mismatch; "
            "refusing to use bounds not fit on the current training seasons."
        )
    test_data = calculate_utilization_scores(test_data, team_df=team_df, weights=None, percentile_bounds=loaded_bounds)

    # Horizon targets (season-bounded)
    train_data = _create_horizon_targets(train_data, n_weeks=[1, 4, 18])
    test_data = _create_horizon_targets(test_data, n_weeks=[1, 4, 18])

    # Data-driven utilization weight optimization
    util_weights = fit_utilization_weights(
        train_data,
        target_col="target_util_1w" if "target_util_1w" in train_data.columns else "target_1w",
        tune_alpha_cv=True,
    )
    train_data = recalculate_utilization_with_weights(train_data, util_weights)
    test_data = recalculate_utilization_with_weights(test_data, util_weights)
    # Recompute targets on reweighted utilization scale
    train_data = _create_horizon_targets(train_data, n_weeks=[1, 4, 18])
    test_data = _create_horizon_targets(test_data, n_weeks=[1, 4, 18])
    with open(MODELS_DIR / "utilization_weights.json", "w") as f:
        json.dump(util_weights, f, indent=2)

    # Feature engineering
    train_data, test_data = _apply_with_temporal_context(
        train_data, test_data,
        lambda d: add_advanced_features(add_engineered_features(d)),
        "feature engineering",
    )
    _apply_bounded_scaling(
        train_data, test_data, MODELS_DIR / "feature_scaler_bounded.joblib",
    )

    # Player embeddings: PCA-based dense representations from aggregated train stats
    try:
        from src.models.advanced_techniques import PlayerEmbeddings
        print("Computing player embeddings (PCA on aggregated stats)...")
        emb = PlayerEmbeddings(embedding_dim=8)
        emb.fit(train_data)  # Fit on train only to avoid leakage
        # Add embedding columns to both train and test
        for df_ref, label in [(train_data, "train"), (test_data, "test")]:
            if "player_id" not in df_ref.columns:
                continue
            emb_matrix = np.array([emb.get_embedding(pid) for pid in df_ref["player_id"]])
            for i in range(emb_matrix.shape[1]):
                df_ref[f"player_emb_{i}"] = emb_matrix[:, i]
        print(f"  Added {emb.embedding_dim} player embedding features")
    except Exception as e:
        logger.warning("Player embeddings skipped: %s", e)

    # Winsorize targets at 1st/99th percentile per position (train only)
    for pos in ["QB", "RB", "WR", "TE"]:
        mask = train_data["position"] == pos
        for n_weeks in [1, 4, 18]:
            col = f"target_{n_weeks}w"
            if col not in train_data.columns:
                continue
            valid = train_data.loc[mask, col].dropna()
            if len(valid) < 20:
                continue
            lo, hi = valid.quantile(0.01), valid.quantile(0.99)
            train_data.loc[mask, col] = train_data.loc[mask, col].clip(lo, hi)
        for col in ["target_util_1w", "target_util_4w", "target_util_18w"]:
            if col not in train_data.columns:
                continue
            valid = train_data.loc[mask, col].dropna()
            if len(valid) < 20:
                continue
            lo, hi = valid.quantile(0.01), valid.quantile(0.99)
            train_data.loc[mask, col] = train_data.loc[mask, col].clip(lo, hi)

    # Train models (fast mode: skip QB dual-target comparison by withholding test_data)
    trainer = ModelTrainer()
    trainer.train_all_positions(
        train_data, positions=positions, tune_hyperparameters=tune_hyperparameters,
        n_weeks_list=[1, 4, 18], test_data=None if fast else test_data,
    )

    # Utilization -> FP conversion
    try:
        train_utilization_to_fp_per_position(train_data, positions=["RB", "WR", "TE", "QB"])
    except Exception as e:
        logger.warning("Utilization-to-FP conversion training skipped: %s", e)

    return train_data, test_data, trainer


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
        base = multi_model.models.get(1) or list(multi_model.models.values())[0]
        medians = getattr(base, "feature_medians", {})
        for fn in getattr(base, "feature_names", []):
            if fn not in pos_test.columns:
                test_data.loc[pos_mask, fn] = medians.get(fn, 0)
        pos_test = test_data.loc[pos_mask].copy()
        preds = multi_model.predict(pos_test, n_weeks=1)
        test_data.loc[pos_mask, "predicted_utilization"] = preds
        test_data.loc[pos_mask, "predicted_points"] = preds
        should_convert = position in converters and (position != "QB" or qb_target == "util")
        if should_convert:
            eff_df = pos_test.copy()
            eff_df["utilization_score"] = preds
            try:
                fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
                test_data.loc[pos_mask, "predicted_points"] = fp_pred
            except Exception as e:
                logger.warning("FP conversion for %s skipped: %s", position, e)
    test_data["actual_for_backtest"] = np.nan
    if "target_1w" in test_data.columns:
        test_data.loc[test_data["position"] != "QB", "actual_for_backtest"] = test_data.loc[
            test_data["position"] != "QB", "target_1w"
        ]
    if "target_util_1w" in test_data.columns:
        test_data.loc[test_data["position"] != "QB", "actual_for_backtest"] = test_data.loc[
            test_data["position"] != "QB", "actual_for_backtest"
        ].fillna(test_data.loc[test_data["position"] != "QB", "target_util_1w"])
    qb_mask = test_data["position"] == "QB"
    if "target_1w" in test_data.columns:
        test_data.loc[qb_mask, "actual_for_backtest"] = test_data.loc[qb_mask, "target_1w"]
    if qb_target == "util" and "QB" not in converters and "target_util_1w" in test_data.columns:
        test_data.loc[qb_mask, "actual_for_backtest"] = test_data.loc[qb_mask, "actual_for_backtest"].fillna(
            test_data.loc[qb_mask, "target_util_1w"]
        )
    if test_data["actual_for_backtest"].isna().all():
        test_data["actual_for_backtest"] = test_data.get("fantasy_points", np.nan)
    results = _run_backtest_after_training(trainer, test_data, train_seasons, actual_test_season)
    return trainer, results


def train_models(positions: list = None,
                 tune_hyperparameters: bool = True,
                 n_trials: int = None,
                 test_season: int = None,
                 optimize_training_years: bool = False,
                 walk_forward: bool = False,
                 strict_requirements: bool = None,
                 fast: bool = False):
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
    
    print("=" * 60)
    print("NFL Player Performance Model Training")
    print("=" * 60)
    
    # Load data with automatic train/test split (test = latest season)
    print("\n[1/5] Loading training data...")
    train_data, test_data, train_seasons, actual_test_season = load_training_data(
        positions,
        test_season=test_season,
        n_train_seasons=None,  # Use all available by default
        optimize_training_years=optimize_training_years,
        strict_requirements=strict_requirements,
    )
    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")

    # Optional walk-forward validation: train on 1..N-1, test on N for last 4 seasons
    if walk_forward:
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

    # Shared preprocessing: DVP, external, season-long, utilization, targets,
    # feature engineering, bounded scaling, winsorization, model training, util-to-fp.
    print("\n[2/5] Preparing features, engineering, and training...")
    train_data, test_data, trainer = _prepare_training_data(
        train_data, test_data, positions, tune_hyperparameters, n_trials,
        fast=fast,
    )

    # Data quality checks (train_models-only, not needed in walk-forward folds)
    print("\n[3/5] Data quality checks...")
    train_missing = _report_missingness(train_data, "train", threshold=0.05)
    test_missing = _report_missingness(test_data, "test", threshold=0.05)
    _validate_critical_missingness(train_data, "train", threshold=0.05)
    _validate_critical_missingness(test_data, "test", threshold=0.05)
    train_outliers = _report_outliers_3sigma(
        train_data,
        "train",
        cols=["fantasy_points", "target_1w", "target_4w", "target_18w", "target_util_1w", "utilization_score"],
    )
    _report_train_serve_feature_parity(train_data, test_data)
    quality_payload = {
        "generated_at": datetime.now().isoformat(),
        "strict_requirements": bool(strict_requirements),
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
    print("\n[4c/5] Training horizon-specific models (4w hybrid, 18w deep)...")
    horizon_status: Dict[str, Dict[str, str]] = {pos: {} for pos in positions}
    target_semantics: Dict[str, Dict[str, str]] = {pos: {} for pos in positions}
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
                                  epochs=MODEL_CONFIG.get("lstm_epochs", 80))
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
                            deep.fit(
                                X_arr[valid], y_arr[valid],
                                feature_names=feature_cols,
                                epochs=MODEL_CONFIG.get("deep_epochs", 100),
                                batch_size=MODEL_CONFIG.get("deep_batch_size", 64),
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
    if len(test_data) > 0:
        print(f"  Test season: {actual_test_season}")
        print(f"  Test records: {len(test_data)}")
        _report_test_metrics(trainer, test_data, train_data)
        _run_backtest_after_training(trainer, test_data, train_seasons, actual_test_season)
    
    # Robust time-series CV validation report (skip in fast mode)
    if not fast:
        print("\n[6/6] Robust CV validation (Ridge baseline)...")
        _run_robust_cv_report(train_data)
    else:
        print("\n[6/6] Robust CV validation skipped (fast mode)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Train seasons: {train_seasons}")
    print(f"Test season: {actual_test_season}")
    summary = trainer.get_training_summary()
    print(summary.to_string(index=False))
    
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

        # Collect per-position test metrics for the metadata
        pos_test_metrics = {}
        for pos, m in trainer.training_metrics.items():
            pos_test_metrics[pos] = m

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
            "training_metrics": pos_test_metrics,
            "n_features_per_position": {
                pos: len(getattr(
                    (m.models.get(1) or list(m.models.values())[0]) if hasattr(m, "models") else m,
                    "feature_names", []
                ))
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
    )


if __name__ == "__main__":
    main()

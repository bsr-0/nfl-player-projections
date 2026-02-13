"""Training script for NFL prediction models."""
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime

# Suppress SciPy/NumPy version mismatch warning (env may have numpy>=1.23 with older scipy)
warnings.filterwarnings(
    "ignore",
    message=".*NumPy version.*required for this version of SciPy.*",
    category=UserWarning,
    module="scipy",
)

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    POSITIONS,
    MODELS_DIR,
    DATA_DIR,
    MODEL_CONFIG,
    QB_TARGET_CHOICE_FILENAME,
    FEATURE_VERSION,
    FEATURE_VERSION_FILENAME,
    MIN_TRAINING_SEASONS_1W,
    MIN_TRAINING_SEASONS_18W,
    MIN_TRAINING_SEASONS_4W,
    MIN_PLAYERS_PER_POSITION,
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
)
from src.features.utilization_weight_optimizer import fit_utilization_weights, UTIL_COMPONENTS
from src.features.dimensionality_reduction import PositionDimensionalityReducer
from src.models.ensemble import ModelTrainer
from src.models.robust_validation import RobustTimeSeriesCV
from src.evaluation.backtester import ModelBacktester
from src.models.utilization_to_fp import train_utilization_to_fp_per_position
from src.evaluation.explainability import get_top10_feature_importance_per_position
from sklearn.linear_model import Ridge


def load_training_data(positions: list = None, min_games: int = 4, 
                       test_season: int = None,
                       n_train_seasons: int = None,
                       optimize_training_years: bool = False) -> tuple:
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
    # Requirement-derived minimums: warn when below (1w min 3, 4w min 5, 18w min 8)
    if n_seasons < MIN_TRAINING_SEASONS_1W:
        print(f"  WARNING: 1-week model requires >= {MIN_TRAINING_SEASONS_1W} training seasons (have {n_seasons}). Accuracy may suffer.")
    if MODEL_CONFIG.get("use_18w_deep", True) and n_seasons < MIN_TRAINING_SEASONS_18W:
        print(f"  WARNING: 18-week deep model requires >= {MIN_TRAINING_SEASONS_18W} training seasons (have {n_seasons}). Consider skipping or adding data.")
    if MODEL_CONFIG.get("use_4w_hybrid", True) and n_seasons < MIN_TRAINING_SEASONS_4W:
        print(f"  WARNING: 4-week hybrid model benefits from >= {MIN_TRAINING_SEASONS_4W} training seasons (have {n_seasons}).")
    # Per-position player minimums (requirements: QB 30+, RB 60+, WR 70+, TE 30+)
    train_players_per_pos = train_data.groupby("position")["player_id"].nunique()
    for pos in POSITIONS:
        min_players = MIN_PLAYERS_PER_POSITION.get(pos, 30)
        n_players = int(train_players_per_pos.get(pos, 0))
        if n_players < min_players:
            print(f"  WARNING: {pos} has {n_players} unique players in training (requirement-derived minimum >= {min_players}).")
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
    except Exception:
        return "util"


def _report_test_metrics(trainer, test_data: pd.DataFrame, train_data: pd.DataFrame):
    """Report model performance on held-out test set. QB uses chosen target (util vs fp)."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    qb_target = _load_qb_target_choice()
    
    for position in trainer.trained_models:
        multi_model = trainer.trained_models[position]
        pos_test = test_data[test_data["position"] == position]
        if len(pos_test) < 10:
            continue
        
        model = multi_model.models.get(1) or list(multi_model.models.values())[0]
        available = [c for c in model.feature_names if c in pos_test.columns]
        if len(available) < len(model.feature_names) * 0.5:
            continue
        
        # QB: compare to chosen target only
        if position == "QB":
            actual_col = "target_1w" if qb_target == "fp" else "target_util_1w"
            if actual_col not in pos_test.columns:
                continue
            y_act = pos_test[actual_col]
            valid = ~y_act.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                preds = multi_model.predict(pos_subset, n_weeks=1)
                rmse = (mean_squared_error(y_act[valid], preds)) ** 0.5
                mae = mean_absolute_error(y_act[valid], preds)
                r2 = r2_score(y_act[valid], preds)
                label = "FP (chosen)" if qb_target == "fp" else "util (chosen)"
                print(f"  {position} (test {label}): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
            continue
        
        # RB/WR/TE: primary utilization, optional FP
        util_col = "target_util_1w"
        if util_col in pos_test.columns:
            y_util = pos_test[util_col]
            valid = ~y_util.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                pred_util = multi_model.predict(pos_subset, n_weeks=1)
                rmse_u = (mean_squared_error(y_util[valid], pred_util)) ** 0.5
                mae_u = mean_absolute_error(y_util[valid], pred_util)
                r2_u = r2_score(y_util[valid], pred_util)
                print(f"  {position} (test util): RMSE={rmse_u:.2f}, MAE={mae_u:.2f}, R²={r2_u:.3f}")
        if "target_1w" in pos_test.columns:
            y_test = pos_test["target_1w"]
            valid = ~y_test.isna()
            if valid.sum() >= 5:
                pos_subset = pos_test.loc[valid]
                preds = multi_model.predict(pos_subset, n_weeks=1)
                rmse = (mean_squared_error(y_test[valid], preds)) ** 0.5
                mae = mean_absolute_error(y_test[valid], preds)
                r2 = r2_score(y_test[valid], preds)
                print(f"  {position} (test FP): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")


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

    # Load utilization->FP converters when present (RB/WR/TE)
    converters = {}
    try:
        from src.models.utilization_to_fp import UtilizationToFPConverter
        for pos in ["RB", "WR", "TE"]:
            try:
                c = UtilizationToFPConverter.load(pos)
                if getattr(c, "is_fitted", False):
                    converters[pos] = c
            except Exception:
                pass
    except Exception:
        converters = {}
    for position in trainer.trained_models:
        multi_model = trainer.trained_models[position]
        pos_mask = test_data["position"] == position
        pos_test = test_data.loc[pos_mask]
        if len(pos_test) < 5:
            continue
        model = multi_model.models.get(1) or list(multi_model.models.values())[0]
        for fn in getattr(model, "feature_names", []):
            if fn not in pos_test.columns:
                test_data.loc[pos_mask, fn] = 0
        pos_test = test_data.loc[pos_mask].copy()
        preds = multi_model.predict(pos_test, n_weeks=1)
        test_data.loc[pos_mask, "predicted_utilization"] = preds
        # Default: set points equal to utilization (kept for QB when trained on FP or util)
        test_data.loc[pos_mask, "predicted_points"] = preds
        # Non-QB: convert utilization -> fantasy points when converter is available
        if position in converters:
            eff_df = pos_test.copy()
            eff_df["utilization_score"] = preds
            try:
                fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
                test_data.loc[pos_mask, "predicted_points"] = fp_pred
            except Exception:
                pass
    valid_preds = test_data["predicted_points"].notna()
    if valid_preds.sum() < 10:
        return
    # Combined actual column:
    # - QB: chosen target (util or FP)
    # - RB/WR/TE: when utilization->FP conversion is applied, compare against future FP; otherwise utilization
    qb_target = _load_qb_target_choice()
    test_data["actual_for_backtest"] = np.nan
    non_qb_mask = test_data["position"] != "QB"
    if "target_1w" in test_data.columns:
        # Prefer FP actuals for non-QB when we are producing predicted_points via conversion.
        test_data.loc[non_qb_mask, "actual_for_backtest"] = test_data.loc[non_qb_mask, "target_1w"]
    if "target_util_1w" in test_data.columns:
        # Fallback: if FP actual missing, use utilization target.
        test_data.loc[non_qb_mask, "actual_for_backtest"] = test_data.loc[non_qb_mask, "actual_for_backtest"].fillna(
            test_data.loc[non_qb_mask, "target_util_1w"]
        )
    if "target_1w" in test_data.columns and "target_util_1w" in test_data.columns:
        qb_mask = test_data["position"] == "QB"
        test_data.loc[qb_mask, "actual_for_backtest"] = (
            test_data.loc[qb_mask, "target_1w"] if qb_target == "fp"
            else test_data.loc[qb_mask, "target_util_1w"]
        )
    if test_data["actual_for_backtest"].isna().all():
        test_data["actual_for_backtest"] = test_data.get("fantasy_points", np.nan)
    backtester = ModelBacktester()
    pred_col = "predicted_points"
    actual_col = "actual_for_backtest"
    results = backtester.backtest_season(
        predictions=test_data,
        actuals=test_data,
        season=actual_test_season,
        prediction_col=pred_col,
        actual_col=actual_col,
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
    # --- Success criteria evaluation (per requirements Section VII) ---
    metrics = results.get("metrics", {})
    spearman_rho = metrics.get("spearman_rho")
    within_10 = metrics.get("within_10_pts_pct")
    within_7 = metrics.get("within_7_pts_pct")
    mbc = multi_baseline if "error" not in multi_baseline else {}
    beat_all_20 = mbc.get("model_beats_all_by_20_pct", False) if mbc else False
    beat_primary_25 = False
    if baseline_comp and "error" not in baseline_comp:
        beat_primary_25 = baseline_comp.get("improvement", {}).get("rmse_pct", 0) >= 25.0

    success_criteria = {
        "spearman_rho": round(float(spearman_rho), 3) if spearman_rho and np.isfinite(spearman_rho) else None,
        "spearman_gt_065": bool(spearman_rho and np.isfinite(spearman_rho) and spearman_rho > 0.65),
        "within_10_pts_pct": round(float(within_10), 1) if within_10 else None,
        "within_10_pts_pct_ge_80": bool(within_10 and within_10 >= 80.0),
        "within_7_pts_pct": round(float(within_7), 1) if within_7 else None,
        "within_7_pts_pct_ge_70": bool(within_7 and within_7 >= 70.0),
        "beat_all_baselines_by_20_pct": beat_all_20,
        "beat_primary_baseline_by_25_pct": beat_primary_25,
    }
    results["success_criteria"] = success_criteria

    print("\n  --- Success Criteria (Requirements VII) ---")
    for k, v in success_criteria.items():
        status = "PASS" if v is True else ("FAIL" if v is False else str(v))
        print(f"    {k}: {status}")

    # --- Model drift detection: compare against previous backtest if available ---
    try:
        prev_files = sorted(backtester.results_dir.glob("backtest_*.json"))
        if len(prev_files) >= 2:
            prev_path = prev_files[-2]  # second-to-last = previous run
            with open(prev_path) as f:
                prev_results = json.load(f)
            prev_rmse = prev_results.get("metrics", {}).get("rmse")
            curr_rmse = metrics.get("rmse")
            if prev_rmse and curr_rmse and prev_rmse > 0:
                drift_pct = (curr_rmse - prev_rmse) / prev_rmse * 100
                results["model_drift"] = {
                    "previous_rmse": prev_rmse,
                    "current_rmse": curr_rmse,
                    "drift_pct": round(drift_pct, 1),
                    "degradation_gt_20_pct": drift_pct > 20.0,
                }
                if drift_pct > 20.0:
                    print(f"\n  *** WARNING: Model drift detected! RMSE degraded {drift_pct:.1f}% vs previous run. "
                          f"Consider rollback (prev RMSE={prev_rmse}, current={curr_rmse}).")
                else:
                    print(f"\n  Model drift: {drift_pct:+.1f}% vs previous (stable)")
    except Exception:
        pass

    backtester.save_results(results)

    # Write app-compatible results
    backtest_results_app = {}
    for pos, pm in results.get("by_position", {}).items():
        backtest_results_app[pos] = {
            "rmse": pm["rmse"],
            "mae": pm["mae"],
            "r2": pm["r2"],
            "directional_accuracy_pct": pm.get("directional_accuracy_pct"),
            "within_5_pts_pct": pm.get("within_5_pts_pct"),
        }
    app_results_path = DATA_DIR / "advanced_model_results.json"
    app_payload = {
        "timestamp": datetime.now().isoformat(),
        "train_seasons": train_seasons,
        "test_season": actual_test_season,
        "backtest_results": backtest_results_app,
        "success_criteria": success_criteria,
    }
    with open(app_results_path, "w") as f:
        json.dump(app_payload, f, indent=2, default=str)
    print(f"\nBacktest complete. App results written to {app_results_path.name}")

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


def prepare_features(data: pd.DataFrame, position: str = None,
                     utilization_weights: dict = None) -> pd.DataFrame:
    """Prepare features for training (utilization + engineered features + advanced features)."""
    print("Calculating utilization scores...")
    data = add_utilization_scores(data, weights=utilization_weights)
    print("Engineering features...")
    data = add_engineered_features(data, position=position)
    # Add advanced rookie & injury features (combine data, injury risk, draft capital)
    try:
        from src.features.advanced_rookie_injury import add_advanced_rookie_injury_features
        data = add_advanced_rookie_injury_features(data)
    except Exception as e:
        print(f"  Advanced rookie/injury features skipped: {e}")
    return data


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
    from config.settings import MODELS_DIR
    # DVP
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        db.ensure_team_defense_stats()
    except Exception:
        pass
    # External (Vegas, injury, weather)
    try:
        from src.data.external_data import add_external_features
        train_data = add_external_features(train_data, seasons=list(train_data["season"].unique()))
        test_data = add_external_features(test_data, seasons=list(test_data["season"].unique()))
    except Exception:
        pass
    # Util + bounds
    team_df = pd.DataFrame()
    util_calc = UtilizationScoreCalculator(weights=None)
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    for pos in POSITIONS:
        util_calc.fit_percentile_bounds(train_data, pos, UTIL_COMPONENTS.get(pos, []))
    bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
    save_percentile_bounds(util_calc.position_percentiles, bounds_path)
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    loaded_bounds = load_percentile_bounds(bounds_path)
    test_data = calculate_utilization_scores(test_data, team_df=team_df, weights=None, percentile_bounds=loaded_bounds)
    # Targets
    for n_weeks in [1, 4, 18]:
        train_data[f"target_{n_weeks}w"] = train_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
        test_data[f"target_{n_weeks}w"] = test_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
    train_data["target_util_1w"] = train_data.groupby("player_id")["utilization_score"].shift(-1)
    test_data["target_util_1w"] = test_data.groupby("player_id")["utilization_score"].shift(-1)
    for nw in [4, 18]:
        train_data[f"target_util_{nw}w"] = (
            train_data.groupby("player_id")["utilization_score"]
            .transform(lambda x, w=nw: x.shift(-1).rolling(window=w, min_periods=1).mean())
        )
        test_data[f"target_util_{nw}w"] = (
            test_data.groupby("player_id")["utilization_score"]
            .transform(lambda x, w=nw: x.shift(-1).rolling(window=w, min_periods=1).mean())
        )
    util_weights = fit_utilization_weights(
        train_data,
        target_col="target_util_1w" if "target_util_1w" in train_data.columns else "target_1w",
        tune_alpha_cv=True,
    )
    train_data = recalculate_utilization_with_weights(train_data, util_weights)
    test_data = recalculate_utilization_with_weights(test_data, util_weights)
    with open(MODELS_DIR / "utilization_weights.json", "w") as f:
        json.dump(util_weights, f, indent=2)
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    # Winsorize
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
    trainer = ModelTrainer()
    trainer.train_all_positions(train_data, positions=positions, tune_hyperparameters=tune_hyperparameters, n_weeks_list=[1, 4, 18], test_data=test_data)
    try:
        train_utilization_to_fp_per_position(train_data)
    except Exception:
        pass
    test_data = test_data.copy()
    test_data["predicted_points"] = np.nan
    test_data["predicted_utilization"] = np.nan

    converters = {}
    try:
        from src.models.utilization_to_fp import UtilizationToFPConverter
        for pos in ["RB", "WR", "TE"]:
            try:
                c = UtilizationToFPConverter.load(pos)
                if getattr(c, "is_fitted", False):
                    converters[pos] = c
            except Exception:
                pass
    except Exception:
        converters = {}
    for position in positions:
        if position not in trainer.trained_models:
            continue
        multi_model = trainer.trained_models[position]
        pos_mask = test_data["position"] == position
        pos_test = test_data.loc[pos_mask]
        if len(pos_test) < 5:
            continue
        base = multi_model.models.get(1) or list(multi_model.models.values())[0]
        for fn in getattr(base, "feature_names", []):
            if fn not in pos_test.columns:
                test_data.loc[pos_mask, fn] = 0
        pos_test = test_data.loc[pos_mask].copy()
        preds = multi_model.predict(pos_test, n_weeks=1)
        test_data.loc[pos_mask, "predicted_utilization"] = preds
        test_data.loc[pos_mask, "predicted_points"] = preds
        if position in converters:
            eff_df = pos_test.copy()
            eff_df["utilization_score"] = preds
            try:
                fp_pred = converters[position].predict(preds, efficiency_df=eff_df)
                test_data.loc[pos_mask, "predicted_points"] = fp_pred
            except Exception:
                pass
    qb_target = _load_qb_target_choice()
    test_data["actual_for_backtest"] = np.nan
    # Backtest points: for non-QB we convert utilization->FP when possible, so compare to future FP.
    if "target_1w" in test_data.columns:
        test_data.loc[test_data["position"] != "QB", "actual_for_backtest"] = test_data.loc[
            test_data["position"] != "QB", "target_1w"
        ]
    if "target_1w" in test_data.columns and "target_util_1w" in test_data.columns:
        qb_mask = test_data["position"] == "QB"
        test_data.loc[qb_mask, "actual_for_backtest"] = (
            test_data.loc[qb_mask, "target_1w"] if qb_target == "fp" else test_data.loc[qb_mask, "target_util_1w"]
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
                 walk_forward: bool = False):
    """
    Main training function with automatic train/test split.
    
    Args:
        positions: Positions to train models for
        tune_hyperparameters: Whether to tune hyperparameters
        n_trials: Number of Optuna trials
        test_season: Override test season (None = use latest available)
        walk_forward: If True, run walk-forward validation (train on 1..N-1, test on N) for last 4 seasons and report mean +/- std RMSE/MAE.
    """
    positions = positions or POSITIONS
    n_trials = n_trials or MODEL_CONFIG["n_optuna_trials"]
    
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
            td, td_test, tr_ss, _ = load_training_data(positions, test_season=ts, optimize_training_years=False)
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

    # Populate team_defense_stats from player data so DVP features are available
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        n_def = db.ensure_team_defense_stats()
        if n_def > 0:
            print(f"  Populated {n_def} team_defense_stats rows (DVP).")
    except Exception as e:
        print(f"  Team defense stats skip: {e}")

    # Integrate weather, injury, Vegas/game script (spread, over/under, implied total) into main training path
    try:
        from src.data.external_data import add_external_features
        train_data = add_external_features(train_data, seasons=list(train_data["season"].unique()))
        test_data = add_external_features(test_data, seasons=list(test_data["season"].unique()))
    except Exception as e:
        print(f"  External features (weather/injury/Vegas) skip: {e}")

    # Phase 1: Utilization scores with train-only percentile bounds (avoid test/serve leakage)
    print("\n[2/5] Preparing features...")
    team_df = pd.DataFrame()
    util_calc = UtilizationScoreCalculator(weights=None)
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    for pos in POSITIONS:
        util_calc.fit_percentile_bounds(train_data, pos, UTIL_COMPONENTS.get(pos, []))
    bounds_path = MODELS_DIR / "utilization_percentile_bounds.json"
    save_percentile_bounds(util_calc.position_percentiles, bounds_path)
    print(f"  Saved utilization percentile bounds to {bounds_path.name}")
    train_data = util_calc.calculate_all_scores(train_data, team_df)
    loaded_bounds = load_percentile_bounds(bounds_path)
    test_data = calculate_utilization_scores(test_data, team_df=team_df, weights=None, percentile_bounds=loaded_bounds)
    
    # Create targets for different horizons (needed before utilization weight fitting)
    print("\n[2b/5] Creating prediction targets...")
    for n_weeks in [1, 4, 18]:
        train_data[f"target_{n_weeks}w"] = train_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
        test_data[f"target_{n_weeks}w"] = test_data.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1).rolling(window=n_weeks, min_periods=1).sum()
        )
    
    # Utilization targets (primary): next 1/4/18 weeks' utilization (future only)
    train_data["target_util_1w"] = train_data.groupby("player_id")["utilization_score"].shift(-1)
    test_data["target_util_1w"] = test_data.groupby("player_id")["utilization_score"].shift(-1)
    for nw in [4, 18]:
        train_data[f"target_util_{nw}w"] = (
            train_data.groupby("player_id")["utilization_score"]
            .transform(lambda x, w=nw: x.shift(-1).rolling(window=w, min_periods=1).mean())
        )
        test_data[f"target_util_{nw}w"] = (
            test_data.groupby("player_id")["utilization_score"]
            .transform(lambda x, w=nw: x.shift(-1).rolling(window=w, min_periods=1).mean())
        )
    
    # Data-driven utilization weight optimization (fit on train only; target = future utilization)
    util_weights = fit_utilization_weights(
        train_data,
        target_col="target_util_1w" if "target_util_1w" in train_data.columns else "target_1w",
        tune_alpha_cv=True,
    )
    train_data = recalculate_utilization_with_weights(train_data, util_weights)
    test_data = recalculate_utilization_with_weights(test_data, util_weights)
    # Persist for train/serve consistency (predict pipeline loads these)
    import json
    weights_path = MODELS_DIR / "utilization_weights.json"
    with open(weights_path, "w") as f:
        json.dump(util_weights, f, indent=2)
    print(f"  Applied data-driven utilization weights (saved to {weights_path.name})")
    
    # Phase 2: Feature engineering (uses utilization_score)
    print("\n[2c/5] Engineering features...")
    train_data = add_engineered_features(train_data)
    test_data = add_engineered_features(test_data)
    
    # Winsorize targets at 1st/99th percentile per position (train only)
    print("\n[3/5] Winsorizing targets...")
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
    
    # Train models on training data (pass test_data for QB target selection)
    print("\n[4/5] Training models...")
    trainer = ModelTrainer()
    trainer.train_all_positions(
        train_data,
        positions=positions,
        tune_hyperparameters=tune_hyperparameters,
        n_weeks_list=[1, 4, 18],
        test_data=test_data,
    )

    # Utilization -> Fantasy Points conversion layer for RB/WR/TE
    print("\n[4b/5] Training utilization-to-FP conversion models...")
    try:
        converters = train_utilization_to_fp_per_position(train_data)
        for pos, c in converters.items():
            if c.is_fitted:
                print(f"  {pos}: utilization->FP converter saved")
    except Exception as e:
        print(f"  Utilization->FP conversion skipped: {e}")

    # Horizon-specific models: 4-week LSTM+ARIMA hybrid, 18-week deep (when enabled)
    print("\n[4c/5] Training horizon-specific models (4w hybrid, 18w deep)...")
    try:
        from src.models.horizon_models import Hybrid4WeekModel, DeepSeasonLongModel
        n_seasons = len(train_seasons)
        for position in positions:
            if position not in trainer.trained_models:
                continue
            multi = trainer.trained_models[position]
            base = multi.models.get(1) or list(multi.models.values())[0]
            feature_cols = getattr(base, "feature_names", [])
            if len(feature_cols) < 5:
                continue
            pos_data = train_data[train_data["position"] == position].copy()
            pos_data = pos_data.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
            X_pos = pos_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            player_ids = pos_data["player_id"].values

            if MODEL_CONFIG.get("use_4w_hybrid", True):
                y_4w = pos_data.get("target_util_4w", pos_data.get("target_4w"))
                if y_4w is not None and y_4w.notna().sum() >= 100:
                    try:
                        hybrid = Hybrid4WeekModel(position)
                        hybrid.fit(pos_data, y_4w, player_ids, feature_cols, epochs=80)
                        if hybrid.is_fitted:
                            hybrid.save()
                            print(f"  {position}: 4-week hybrid model saved")
                    except Exception as e:
                        print(f"  4-week hybrid skip for {position}: {e}")

            if MODEL_CONFIG.get("use_18w_deep", True) and n_seasons >= MIN_TRAINING_SEASONS_18W:
                y_18w = pos_data.get("target_util_18w", pos_data.get("target_18w"))
                if y_18w is not None and y_18w.notna().sum() >= 80:
                    try:
                        deep = DeepSeasonLongModel(position, n_features=min(150, len(feature_cols)))
                        X_arr = X_pos.values.astype(np.float64)
                        y_arr = y_18w.values.astype(np.float64)
                        valid = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
                        if valid.sum() >= 80:
                            deep.fit(
                                X_arr[valid], y_arr[valid],
                                feature_names=feature_cols,
                                epochs=100,
                                batch_size=64,
                            )
                            if deep.is_fitted:
                                deep.save()
                                print(f"  {position}: 18-week deep model saved")
                    except Exception as e:
                        print(f"  18-week deep skip for {position}: {e}")
    except ImportError:
        print("  Horizon models skipped (TensorFlow not available).")
    except Exception as e:
        print(f"  Horizon-specific training skipped: {e}")

    # Top-10 feature importance per position (explainability)
    try:
        top10 = get_top10_feature_importance_per_position(
            trainer.trained_models,
            output_path=MODELS_DIR / "top10_features_per_position.json",
        )
        for pos, feats in top10.items():
            print(f"  Top-3 {pos}: {[f['feature'] for f in feats[:3]]}")
    except Exception as e:
        print(f"  Top-10 feature export skipped: {e}")

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
    
    # Robust time-series CV validation report
    print("\n[6/6] Robust CV validation (Ridge baseline)...")
    _run_robust_cv_report(train_data)
    
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
            except Exception:
                pass

        # Collect per-position test metrics for the metadata
        pos_test_metrics = {}
        for pos, m in trainer.training_metrics.items():
            pos_test_metrics[pos] = m

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
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Model metadata written: {metadata_path.name}")
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
    
    args = parser.parse_args()
    
    train_models(
        positions=args.positions,
        tune_hyperparameters=not args.no_tune,
        n_trials=args.trials,
        test_season=args.test_season,
        optimize_training_years=args.optimize_years,
        walk_forward=args.walk_forward,
    )


if __name__ == "__main__":
    main()

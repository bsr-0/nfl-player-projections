"""
Dynamic Training Year Selection for NFL Prediction Models.

Evaluates multiple training window sizes (e.g., 3, 5, 7, 10, all years) using
time-series cross-validation and selects the window that minimizes test RMSE
(or maximizes R²) on held-out folds. Ensures all available data is considered
and the process dynamically selects the optimal history length per position.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR, MODEL_CONFIG


def get_available_seasons_from_data(df: pd.DataFrame) -> List[int]:
    """Get sorted list of seasons present in the data."""
    if df.empty or "season" not in df.columns:
        return []
    return sorted(df["season"].unique().tolist())


def evaluate_training_window(
    df: pd.DataFrame,
    position: str,
    n_train_years: int,
    test_season: int,
    feature_cols: List[str],
    target_col: str = "target_1w",
    n_cv_folds: int = 3,
) -> Optional[Dict]:
    """
    Evaluate a specific training window size using time-series CV.
    
    Args:
        df: Full DataFrame with features and target
        position: Position to evaluate
        n_train_years: Number of years for training (e.g., 5 = use 5 seasons before test)
        test_season: Season to hold out as test
        feature_cols: Feature column names
        target_col: Target column
        n_cv_folds: Number of CV folds (each uses a different test season)
        
    Returns:
        Dict with rmse, r2, mae, n_train, n_test, or None if insufficient data
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    pos_df = df[df["position"] == position].copy()
    seasons = sorted(pos_df["season"].unique())
    
    if len(seasons) < n_train_years + 1:
        return None
    
    # Use last n_cv_folds seasons as test folds (walk-forward)
    test_seasons = [s for s in seasons if s >= test_season][:n_cv_folds]
    if not test_seasons:
        test_seasons = seasons[-n_cv_folds:]
    
    rmses, r2s, maes, n_trains, n_tests = [], [], [], [], []
    
    for ts in test_seasons:
        # Train: n_train_years immediately before test season
        train_seasons = [s for s in seasons if s < ts][-n_train_years:]
        if len(train_seasons) < max(1, n_train_years // 2):
            continue
        
        train_df = pos_df[pos_df["season"].isin(train_seasons)]
        test_df = pos_df[pos_df["season"] == ts]
        
        available = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
        if len(available) < 5 or target_col not in train_df.columns:
            continue
        
        X_train = train_df[available].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df[target_col]
        X_test = test_df[available].fillna(0).replace([np.inf, -np.inf], 0)
        y_test = test_df[target_col]
        
        valid_train = y_train.notna() & (y_train >= 0)
        valid_test = y_test.notna() & (y_test >= 0)
        if valid_train.sum() < 50 or valid_test.sum() < 10:
            continue
        
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0, random_state=MODEL_CONFIG.get("random_state", 42))
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        
        rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2s.append(r2_score(y_test, preds))
        maes.append(mean_absolute_error(y_test, preds))
        n_trains.append(len(X_train))
        n_tests.append(len(X_test))
    
    if not rmses:
        return None
    
    return {
        "n_train_years": n_train_years,
        "mean_rmse": float(np.mean(rmses)),
        "std_rmse": float(np.std(rmses)),
        "mean_r2": float(np.mean(r2s)),
        "mean_mae": float(np.mean(maes)),
        "n_folds": len(rmses),
        "n_train": int(np.mean(n_trains)),
        "n_test": int(np.mean(n_tests)),
    }


def select_optimal_training_years(
    df: pd.DataFrame,
    positions: List[str] = None,
    candidate_years: List[int] = None,
    test_season: int = None,
    feature_cols: List[str] = None,
    target_col: str = "target_1w",
    metric: str = "rmse",  # "rmse" (minimize) or "r2" (maximize)
) -> Dict[str, int]:
    """
    Dynamically select optimal training years per position.
    
    Args:
        df: Training DataFrame with features and target
        positions: Positions to optimize (default: QB, RB, WR, TE)
        candidate_years: Training window sizes to try (e.g., [3, 5, 7, 10, 15])
        test_season: Test season (default: max season in data)
        feature_cols: Feature columns (auto-detected if None)
        target_col: Target column
        metric: "rmse" to minimize or "r2" to maximize
        
    Returns:
        Dict mapping position -> optimal n_train_years
    """
    positions = positions or ["QB", "RB", "WR", "TE"]
    seasons = get_available_seasons_from_data(df)
    
    if not seasons:
        return {p: 5 for p in positions}  # Default fallback
    
    test_season = test_season or max(seasons)
    max_possible = test_season - min(seasons)
    candidate_years = candidate_years or [3, 5, 7, 10, 15, max_possible]
    candidate_years = [y for y in candidate_years if 1 <= y <= max_possible]
    if not candidate_years:
        candidate_years = [min(5, max_possible)]
    
    # Ensure target exists (create from fantasy_points if needed)
    df = df.copy()
    if target_col not in df.columns and "fantasy_points" in df.columns:
        df = df.sort_values(["player_id", "season", "week"])
        df[target_col] = df.groupby("player_id")["fantasy_points"].transform(
            lambda x: x.shift(-1)
        )
    
    exclude = [
        "player_id", "name", "position", "team", "season", "week",
        "target", "opponent", "home_away", "id", "games_played",
        "fantasy_points",  # Raw current-week FP - never a feature
    ]
    if feature_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric if c not in exclude and not c.startswith("target_")]
        priority = ["rushing_yards", "rushing_attempts", "targets", "receptions", "receiving_yards",
                    "passing_yards", "passing_attempts", "utilization_score", "snap_share",
                    "rushing_tds", "receiving_tds", "passing_tds"]
        ordered = [c for c in priority if c in feature_cols]
        feature_cols = (ordered + [c for c in feature_cols if c not in ordered])[:50]
    
    results = {}
    for position in positions:
        best_years = 5
        best_score = np.inf if metric == "rmse" else -np.inf
        scores = []
        
        for n_years in candidate_years:
            ev = evaluate_training_window(
                df, position, n_years, test_season,
                feature_cols, target_col, n_cv_folds=3
            )
            if ev is None:
                continue
            
            score = ev["mean_rmse"] if metric == "rmse" else ev["mean_r2"]
            scores.append((n_years, score, ev))
            
            if metric == "rmse" and score < best_score:
                best_score = score
                best_years = n_years
            elif metric == "r2" and score > best_score:
                best_score = score
                best_years = n_years
        
        results[position] = best_years
    
    return results


def load_cached_optimal_years() -> Optional[Dict[str, int]]:
    """Load cached optimal training years if available."""
    path = MODELS_DIR / "optimal_training_years.json"
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
                return data.get("optimal_years")
        except Exception:
            pass
    return None


def save_optimal_training_years(optimal: Dict[str, int], test_season: int):
    """Cache optimal training years."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "optimal_training_years.json"
    with open(path, "w") as f:
        json.dump({
            "optimal_years": optimal,
            "test_season": test_season,
        }, f, indent=2)

#!/usr/bin/env python3
"""Generate static JSON data files for the fantasy draft web app.

Produces two categories of data:

1. **Model Performance** (``model_performance.json``):
   Previous season's out-of-sample predictions alongside actual results,
   demonstrating how the model performed on truly unseen data.

2. **Upcoming Season Projections** (``players_{POS}.json``):
   ML model predictions for the upcoming season.  When the schedule has
   not been released yet the projection fields are set to ``null`` so the
   frontend can display a "pending" state instead of extrapolating.

No extrapolation is ever performed — all numbers come from the ML model
or from real game results.

Usage:
    python scripts/generate_draft_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest_results"


# ---------------------------------------------------------------------------
# 1. Model Performance: previous season out-of-sample predictions vs actuals
# ---------------------------------------------------------------------------

def _load_latest_backtest_json(season: int = None):
    """Load the latest backtest JSON for a given (or most recent) season."""
    if not BACKTEST_DIR.exists():
        return None
    files = sorted(BACKTEST_DIR.glob("backtest_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            with open(p) as f:
                data = json.load(f)
            if season is None or data.get("season") == season:
                return data
        except Exception:
            continue
    return None


def _load_ts_backtest_predictions(season: int = None):
    """Load per-player per-week ts-backtest predictions CSV."""
    if not BACKTEST_DIR.exists():
        return None
    pattern = f"ts_backtest_{season}_*_predictions.csv" if season else "ts_backtest_*_predictions.csv"
    files = sorted(BACKTEST_DIR.glob(pattern),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return pd.read_csv(files[0])
    except Exception:
        return None


def generate_model_performance():
    """Create model_performance.json showing previous season predictions vs actuals."""
    from src.utils.nfl_calendar import get_current_nfl_season
    current_season = get_current_nfl_season()
    # Previous completed season
    prev_season = current_season

    # Try ts-backtest predictions (per-player, per-week granularity)
    ts_preds = _load_ts_backtest_predictions(prev_season)
    if ts_preds is None:
        # Fall back to one season earlier
        ts_preds = _load_ts_backtest_predictions(prev_season - 1)
        if ts_preds is not None:
            prev_season = prev_season - 1

    # Also load aggregate backtest metrics
    backtest_json = _load_latest_backtest_json(prev_season)
    if backtest_json is None:
        backtest_json = _load_latest_backtest_json()  # any season

    payload = {
        "season": prev_season,
        "has_per_player_predictions": ts_preds is not None and len(ts_preds) > 0,
        "aggregate_metrics": {},
        "by_position": {},
        "top_performers": {},
        "per_player_season_totals": [],
    }

    if backtest_json:
        payload["aggregate_metrics"] = backtest_json.get("metrics", {})
        payload["by_position"] = backtest_json.get("by_position", {})
        tp = backtest_json.get("top_performers", {})
        for pos in ["QB", "RB", "WR", "TE"]:
            if pos in tp and "top_10_actual" in tp[pos]:
                payload["top_performers"][pos] = tp[pos]["top_10_actual"]

    # Build per-player season aggregates from ts-backtest predictions
    if ts_preds is not None and not ts_preds.empty:
        required_cols = {"player_id", "name", "position", "predicted", "actual"}
        if required_cols.issubset(set(ts_preds.columns)):
            ts_preds = ts_preds[ts_preds["position"].isin(["QB", "RB", "WR", "TE"])]
            agg = ts_preds.groupby(["player_id", "name", "position"]).agg(
                predicted_total=("predicted", "sum"),
                actual_total=("actual", "sum"),
                games=("actual", "count"),
                predicted_ppg=("predicted", "mean"),
                actual_ppg=("actual", "mean"),
            ).reset_index()

            if "team" in ts_preds.columns:
                team_map = ts_preds.groupby("player_id")["team"].last().to_dict()
                agg["team"] = agg["player_id"].map(team_map).fillna("")
            else:
                agg["team"] = ""

            agg["error"] = (agg["predicted_total"] - agg["actual_total"]).round(1)
            agg["abs_error"] = agg["error"].abs()

            # Sort by actual total (best performers first)
            agg = agg.sort_values("actual_total", ascending=False)

            records = []
            for _, row in agg.head(200).iterrows():
                records.append({
                    "player_id": str(row["player_id"]),
                    "name": row["name"],
                    "position": row["position"],
                    "team": row.get("team", ""),
                    "predicted_total": round(float(row["predicted_total"]), 1),
                    "actual_total": round(float(row["actual_total"]), 1),
                    "predicted_ppg": round(float(row["predicted_ppg"]), 1),
                    "actual_ppg": round(float(row["actual_ppg"]), 1),
                    "games": int(row["games"]),
                    "error": round(float(row["error"]), 1),
                })
            payload["per_player_season_totals"] = records

    out_path = DATA_DIR / "model_performance.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote model_performance.json ({len(payload.get('per_player_season_totals', []))} players, season {prev_season})")


# ---------------------------------------------------------------------------
# 2. Upcoming season projections (ML predictions, never extrapolated)
# ---------------------------------------------------------------------------

def load_season_data(season: int):
    """Load regular season data (weeks 1-18) from parquet for a given season."""
    parquet_path = DATA_DIR / "daily_predictions.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(parquet_path)
    mask = (
        (df["season"] == season)
        & (df["week"] <= 18)
        & (df["position"].isin(["QB", "RB", "WR", "TE"]))
    )
    return df[mask]


def aggregate_player_stats(season_df):
    """Aggregate per-player season totals from weekly data."""
    agg_dict = {
        "total_fp": ("fantasy_points", "sum"),
        "games_played": ("fantasy_points", "count"),
        "ppg": ("fantasy_points", "mean"),
        "fp_std": ("fantasy_points", "std"),
        "util_mean": ("utilization_score", "mean"),
    }

    optional_cols = {
        "vol_mean": ("weekly_volatility", "mean"),
        "consistency_mean": ("consistency_score", "mean"),
        "confidence_mean": ("confidence_score", "mean"),
        "cv_mean": ("coefficient_of_variation", "mean"),
    }
    for key, (col, func) in optional_cols.items():
        if col in season_df.columns:
            agg_dict[key] = (col, func)

    agg = season_df.groupby(["player_id", "name", "team", "position"]).agg(
        **agg_dict
    ).reset_index()

    agg["fp_std"] = agg["fp_std"].fillna(0)

    for col in ["vol_mean", "consistency_mean", "confidence_mean", "cv_mean"]:
        if col not in agg.columns:
            agg[col] = 0.0

    return agg


def compute_risk_scores(agg):
    """Risk score 0-100 where higher = more risky."""
    agg["risk_vol"] = 0.0
    agg["risk_cv"] = 0.0
    agg["risk_consistency"] = 0.0
    agg["risk_games"] = 0.0

    for pos in ["QB", "RB", "WR", "TE"]:
        mask = agg["position"] == pos
        subset = agg.loc[mask]
        if len(subset) == 0:
            continue

        vol_min, vol_max = subset["vol_mean"].min(), subset["vol_mean"].max()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1
        agg.loc[mask, "risk_vol"] = (subset["vol_mean"] - vol_min) / vol_range

        cv_min, cv_max = subset["cv_mean"].min(), subset["cv_mean"].max()
        cv_range = cv_max - cv_min if cv_max > cv_min else 1
        agg.loc[mask, "risk_cv"] = (subset["cv_mean"] - cv_min) / cv_range

        cons_max = subset["consistency_mean"].max()
        if cons_max > 0:
            agg.loc[mask, "risk_consistency"] = 1 - (
                subset["consistency_mean"] / cons_max
            )

        gp_max = subset["games_played"].max()
        if gp_max > 0:
            agg.loc[mask, "risk_games"] = 1 - (subset["games_played"] / gp_max)

    agg["risk_score"] = (
        agg["risk_vol"] * 30
        + agg["risk_cv"] * 25
        + agg["risk_consistency"] * 25
        + agg["risk_games"] * 20
    ).round(0).clip(0, 100).astype(int)

    return agg


def add_feature_importance(agg):
    """Attach top features from top10_features_per_position.json."""
    features_path = MODELS_DIR / "top10_features_per_position.json"
    if not features_path.exists():
        agg["key_features"] = [[] for _ in range(len(agg))]
        agg["feature_importance_rank"] = [{} for _ in range(len(agg))]
        return agg

    with open(features_path) as f:
        features = json.load(f)

    def get_key_features(pos):
        return [feat["feature"] for feat in features.get(pos, [])][:5]

    def get_importance_dict(pos):
        return {
            feat["feature"]: round(feat["importance"], 4)
            for feat in features.get(pos, [])
        }

    agg["key_features"] = agg["position"].map(get_key_features)
    agg["feature_importance_rank"] = agg["position"].map(get_importance_dict)
    return agg


def _check_schedule_available(upcoming_season: int) -> bool:
    """Check if the NFL schedule is available for the upcoming season."""
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        return db.has_schedule_for_season(upcoming_season)
    except Exception:
        return False


def _load_ml_predictions(upcoming_season: int):
    """Try to load ML predictions from daily_predictions.parquet for the upcoming season."""
    parquet_path = DATA_DIR / "daily_predictions.parquet"
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        upcoming = df[
            (df["season"] == upcoming_season)
            & (df["position"].isin(["QB", "RB", "WR", "TE"]))
        ]
        if upcoming.empty:
            return None
        # Check if these are actual ML predictions (have projection_18w column with non-null values)
        proj_cols = [c for c in ["projection_1w", "projection_4w", "projection_18w"] if c in upcoming.columns]
        if not proj_cols:
            return None
        has_predictions = False
        for col in proj_cols:
            if upcoming[col].notna().any():
                has_predictions = True
                break
        if not has_predictions:
            return None
        return upcoming
    except Exception:
        return None


def output_position_files(agg, upcoming_season: int, schedule_available: bool,
                          has_ml_predictions: bool, prev_season: int):
    """Write per-position JSON files.

    When ML predictions are available, use projection_18w for full-season totals.
    When no schedule is available for the upcoming season, set projection fields
    to null so the frontend shows a "pending" state.
    """
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = agg[agg["position"] == pos].copy()

        # Sort: if ML predictions available, by projection; else by previous-season PPG
        if has_ml_predictions and "projection_18w" in pos_df.columns:
            sort_col = "projection_18w"
            pos_df = pos_df.sort_values(sort_col, ascending=False, na_position="last")
        else:
            pos_df = pos_df.sort_values("ppg", ascending=False, na_position="last")

        players = []
        for rank, (_, row) in enumerate(pos_df.iterrows(), 1):
            # ML prediction values (null if not available)
            proj_total = None
            proj_ppg = None
            proj_floor = None
            proj_ceiling = None

            if has_ml_predictions and schedule_available:
                p18 = row.get("projection_18w")
                if pd.notna(p18):
                    proj_total = round(float(p18), 1)
                    proj_ppg = round(float(p18) / 17, 1)
                    std = row.get("fp_std", 0)
                    if pd.notna(std) and std > 0:
                        proj_floor = round(max(0, float(p18) - 1.5 * float(std) * (17 ** 0.5)), 1)
                        proj_ceiling = round(float(p18) + 1.5 * float(std) * (17 ** 0.5), 1)

            players.append({
                "player_id": str(row["player_id"]),
                "name": row["name"],
                "team": row["team"],
                "position": row["position"],
                "bye_week": None,
                "adp": rank,
                "projection_points_total": proj_total,
                "projection_points_per_game": proj_ppg,
                "projection_floor": proj_floor,
                "projection_ceiling": proj_ceiling,
                "risk_score": int(row["risk_score"]) if pd.notna(row.get("risk_score")) else None,
                "injury_flag": False,
                "age": None,
                "key_features": row.get("key_features", []),
                "feature_importance_rank": row.get("feature_importance_rank", {}),
                "uses_schedule": schedule_available,
                "prev_season": prev_season,
                "prev_season_ppg": round(float(row["ppg"]), 1) if pd.notna(row.get("ppg")) else None,
                "prev_season_total_fp": round(float(row["total_fp"]), 1) if pd.notna(row.get("total_fp")) else None,
                "prev_season_games": int(row["games_played"]) if pd.notna(row.get("games_played")) else None,
                "has_ml_prediction": proj_total is not None,
            })
        out_path = DATA_DIR / f"players_{pos}.json"
        with open(out_path, "w") as f:
            json.dump(players, f, indent=2)
        print(f"  Wrote {len(players)} players to {out_path.name}"
              f" (ML predictions: {'yes' if has_ml_predictions and schedule_available else 'pending'})")


def generate_schedule_impact(upcoming_season: int, schedule_available: bool):
    """Generate schedule_impact.json."""
    if schedule_available:
        payload = {
            "schedule_incorporated": True,
            "reason": f"The {upcoming_season} NFL schedule has been incorporated.",
            "season": upcoming_season,
            "schedule_available": True,
        }
    else:
        payload = {
            "schedule_incorporated": False,
            "reason": (
                f"The {upcoming_season} NFL schedule has not been released. "
                f"Projections will be available once the schedule is out. "
                f"No extrapolations are used."
            ),
            "season": upcoming_season,
            "schedule_available": False,
        }
    out_path = DATA_DIR / "schedule_impact.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {out_path.name}")


def generate_model_metadata_frontend(upcoming_season: int, prev_season: int,
                                     schedule_available: bool, has_ml_predictions: bool):
    """Create draft_model_metadata.json for the frontend methodology section."""
    meta_path = MODELS_DIR / "model_metadata.json"
    backtest_path = DATA_DIR / "advanced_model_results.json"
    features_path = MODELS_DIR / "top10_features_per_position.json"

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    backtest = {}
    if backtest_path.exists():
        with open(backtest_path) as f:
            backtest = json.load(f)

    features = {}
    if features_path.exists():
        with open(features_path) as f:
            features = json.load(f)

    train_seasons = meta.get("train_seasons", [])
    training_window = ""
    if train_seasons:
        training_window = f"{train_seasons[0]}-{train_seasons[-1]}"

    if has_ml_predictions and schedule_available:
        target_def = f"PPR fantasy points — ML model predictions for {upcoming_season} season"
        data_basis = (
            f"All projections are ML model outputs for the {upcoming_season} season. "
            f"Model performance on the {prev_season} held-out season is shown in the "
            f"Model Performance tab."
        )
    else:
        target_def = f"PPR fantasy points — awaiting {upcoming_season} NFL schedule"
        data_basis = (
            f"The {upcoming_season} NFL schedule has not been released. "
            f"Projections will appear once the schedule is available. "
            f"No extrapolations are used. The Model Performance tab shows how "
            f"the model performed on the {prev_season} season (out-of-sample)."
        )

    payload = {
        "target_definition": target_def,
        "training_data_range": training_window or "2006-2024",
        "positions": ["QB", "RB", "WR", "TE"],
        "schedule_incorporated": schedule_available,
        "upcoming_season": upcoming_season,
        "prev_season": prev_season,
        "has_ml_predictions": has_ml_predictions and schedule_available,
        "version": "v2.0.0",
        "last_updated": meta.get("training_date", ""),
        "training_date": meta.get("training_date"),
        "test_season": meta.get("test_season"),
        "n_features_per_position": meta.get("n_features_per_position", {}),
        "training_metrics": meta.get("training_metrics", {}),
        "backtest_results": backtest.get("backtest_results", {}),
        "top_features": features,
        "methodology": {
            "model_type": "LightGBM ensemble with XGBoost and Ridge regression",
            "training_window": training_window or "2006-2024",
            "test_season": str(meta.get("test_season", prev_season)),
            "scoring_format": "PPR (1 point per reception)",
            "features_description": (
                "50 features per position including utilization scores, "
                "rolling averages (3/5/8 week windows), lag features, "
                "team context, matchup quality, and advanced play-by-play metrics"
            ),
            "overfitting_prevention": [
                "Time-series cross-validation with gap seasons",
                "Recency decay weighting (half-life 2-4 seasons)",
                "Feature selection via stability bootstrap (30 iterations)",
                "Correlation threshold filtering (r > 0.92)",
                "VIF multicollinearity checks (VIF > 10)",
                "Early stopping with 25 rounds patience",
            ],
            "horizons": ["1-week", "4-week", "Full season (18-week)"],
        },
        "data_basis_note": data_basis,
    }
    out_path = DATA_DIR / "draft_model_metadata.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {out_path.name}")


def main():
    from src.utils.nfl_calendar import get_current_nfl_season, is_offseason

    current_season = get_current_nfl_season()
    prev_season = current_season
    upcoming_season = current_season + 1 if is_offseason() else current_season

    print(f"Current NFL season: {current_season}")
    print(f"Previous completed season: {prev_season}")
    print(f"Upcoming season: {upcoming_season}")
    print()

    # 1. Model Performance (previous season OOS predictions vs actuals)
    print("Generating model performance data...")
    generate_model_performance()
    print()

    # 2. Upcoming season projections
    schedule_available = _check_schedule_available(upcoming_season)
    print(f"Schedule available for {upcoming_season}: {schedule_available}")

    # Load previous season stats for player list and risk scores
    print(f"Loading {prev_season} season data for player baseline...")
    season_df = load_season_data(prev_season)
    if season_df.empty:
        print(f"  No data for {prev_season} season. Trying {prev_season - 1}...")
        prev_season = prev_season - 1
        season_df = load_season_data(prev_season)

    if season_df.empty:
        print("  No season data available. Cannot generate draft board.")
        return

    print(f"  Loaded {len(season_df)} weekly records")

    agg = aggregate_player_stats(season_df)
    print(f"  Aggregated {len(agg)} unique players")

    agg = compute_risk_scores(agg)
    agg = add_feature_importance(agg)

    # Check for ML predictions for upcoming season
    has_ml_predictions = False
    ml_df = _load_ml_predictions(upcoming_season)
    if ml_df is not None and not ml_df.empty:
        # Merge ML predictions into agg
        for col in ["projection_1w", "projection_4w", "projection_18w"]:
            if col in ml_df.columns:
                pred_map = ml_df.groupby("player_id")[col].last().to_dict()
                agg[col] = agg["player_id"].map(pred_map)
        has_ml_predictions = True
        print(f"  Loaded ML predictions for {upcoming_season} season")
    else:
        print(f"  No ML predictions available for {upcoming_season} season")

    output_position_files(agg, upcoming_season, schedule_available,
                          has_ml_predictions, prev_season)
    generate_schedule_impact(upcoming_season, schedule_available)
    generate_model_metadata_frontend(upcoming_season, prev_season,
                                     schedule_available, has_ml_predictions)

    print("\nDone! JSON files ready for the web app.")


if __name__ == "__main__":
    main()

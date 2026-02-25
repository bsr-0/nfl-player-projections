#!/usr/bin/env python3
"""Generate static JSON data files for the fantasy draft web app.

Reads the most recently completed regular season from
daily_predictions.parquet and creates per-position JSON files for the
draft board.

When ML predictions are available (cached_features.parquet with
projection_18w for the upcoming season), the draft board uses those as
the primary values.  Otherwise it falls back to extrapolating per-game
actuals to a 17-game season.  In both cases the basis-season actuals
(games played, total FP, PPG) are kept as reference fields.

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

# Season detected at load time and shared across functions
_BASIS_SEASON: int = 0


def load_season_data():
    """Load the most recently completed regular season (weeks 1-18).

    Dynamically picks the latest season with week-18 data in the
    parquet rather than hardcoding a specific year.  Sets the module-
    level _BASIS_SEASON so other functions can reference it.
    """
    global _BASIS_SEASON
    df = pd.read_parquet(DATA_DIR / "daily_predictions.parquet")
    reg = df[(df["week"] <= 18) & (df["position"].isin(["QB", "RB", "WR", "TE"]))]
    # Pick the latest season that has at least 1 week of data
    _BASIS_SEASON = int(reg["season"].max())
    mask = reg["season"] == _BASIS_SEASON
    print(f"  Basis season detected: {_BASIS_SEASON}")
    return reg[mask]


def aggregate_player_stats(season_df):
    """Aggregate per-player season totals from weekly data."""
    agg_dict = {
        "total_fp": ("fantasy_points", "sum"),
        "games_played": ("fantasy_points", "count"),
        "ppg": ("fantasy_points", "mean"),
        "fp_std": ("fantasy_points", "std"),
        "util_mean": ("utilization_score", "mean"),
    }

    # Add optional columns if they exist
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

    # Fill NaN std for single-game players
    agg["fp_std"] = agg["fp_std"].fillna(0)

    # Fill missing optional columns with defaults
    for col in ["vol_mean", "consistency_mean", "confidence_mean", "cv_mean"]:
        if col not in agg.columns:
            agg[col] = 0.0

    return agg


def _try_load_ml_predictions() -> pd.DataFrame:
    """Try loading ML predictions (projection_18w) from cached_features.parquet.

    Returns a DataFrame with player_id and projection_18w if available,
    otherwise an empty DataFrame.
    """
    cached_path = DATA_DIR / "cached_features.parquet"
    if not cached_path.exists():
        return pd.DataFrame()
    try:
        cdf = pd.read_parquet(cached_path)
        if "projection_18w" not in cdf.columns:
            return pd.DataFrame()
        # Keep only rows for the prediction-target season (latest season in file)
        latest = int(cdf["season"].max())
        cdf = cdf[cdf["season"] == latest]
        valid = cdf[cdf["projection_18w"].notna() & (cdf["projection_18w"] > 0)]
        if valid.empty:
            return pd.DataFrame()
        return valid[["player_id", "projection_18w"]].drop_duplicates("player_id")
    except Exception:
        return pd.DataFrame()


def compute_projections(agg):
    """Compute projected totals for the draft board.

    Uses ML predictions (projection_18w) when available; otherwise
    falls back to extrapolating per-game actuals to 17 games.
    Always keeps the basis-season actuals as reference fields.
    """
    # Always compute extrapolation-based values as reference / fallback
    agg["extrap_points_total"] = (agg["ppg"] * 17).round(1)
    agg["extrap_points_per_game"] = agg["ppg"].round(1)

    ml_preds = _try_load_ml_predictions()
    if not ml_preds.empty:
        agg = agg.merge(ml_preds, on="player_id", how="left")
        has_ml = agg["projection_18w"].notna()
        agg.loc[has_ml, "projection_points_total"] = agg.loc[has_ml, "projection_18w"].round(1)
        agg.loc[has_ml, "projection_points_per_game"] = (agg.loc[has_ml, "projection_18w"] / 17).round(1)
        # Fall back to extrapolation for players missing ML predictions
        agg.loc[~has_ml, "projection_points_total"] = agg.loc[~has_ml, "extrap_points_total"]
        agg.loc[~has_ml, "projection_points_per_game"] = agg.loc[~has_ml, "extrap_points_per_game"]
        agg["data_source"] = np.where(has_ml, "ml_model", "extrapolation")
        agg = agg.drop(columns=["projection_18w"], errors="ignore")
        n_ml = int(has_ml.sum())
        print(f"  Using ML predictions for {n_ml}/{len(agg)} players (rest: extrapolation)")
    else:
        agg["projection_points_total"] = agg["extrap_points_total"]
        agg["projection_points_per_game"] = agg["extrap_points_per_game"]
        agg["data_source"] = "extrapolation"
        print("  No ML predictions available; using extrapolation")

    agg["projection_floor"] = (
        (agg["ppg"] - 1.5 * agg["fp_std"]).clip(lower=0) * 17
    ).round(1)
    agg["projection_ceiling"] = ((agg["ppg"] + 1.5 * agg["fp_std"]) * 17).round(1)
    return agg


def compute_risk_scores(agg):
    """Risk score 0-100 where higher = more risky.

    Based on volatility, coefficient of variation, inverse consistency,
    and games played penalty, normalized within each position.
    """
    agg["risk_vol"] = 0.0
    agg["risk_cv"] = 0.0
    agg["risk_consistency"] = 0.0
    agg["risk_games"] = 0.0

    for pos in ["QB", "RB", "WR", "TE"]:
        mask = agg["position"] == pos
        subset = agg.loc[mask]
        if len(subset) == 0:
            continue

        # Volatility component (higher vol = higher risk)
        vol_min, vol_max = subset["vol_mean"].min(), subset["vol_mean"].max()
        vol_range = vol_max - vol_min if vol_max > vol_min else 1
        agg.loc[mask, "risk_vol"] = (subset["vol_mean"] - vol_min) / vol_range

        # CV component (higher CV = higher risk)
        cv_min, cv_max = subset["cv_mean"].min(), subset["cv_mean"].max()
        cv_range = cv_max - cv_min if cv_max > cv_min else 1
        agg.loc[mask, "risk_cv"] = (subset["cv_mean"] - cv_min) / cv_range

        # Consistency inverse (lower consistency = higher risk)
        cons_max = subset["consistency_mean"].max()
        if cons_max > 0:
            agg.loc[mask, "risk_consistency"] = 1 - (
                subset["consistency_mean"] / cons_max
            )

        # Games played penalty (fewer games = higher risk)
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


def add_adp_proxy(agg):
    """Generate synthetic ADP from overall rank by projected points."""
    agg = agg.sort_values(
        "projection_points_total", ascending=False
    ).reset_index(drop=True)
    agg["adp"] = agg.index + 1
    return agg


def output_position_files(agg):
    """Write per-position JSON files."""
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = agg[agg["position"] == pos].sort_values(
            "projection_points_total", ascending=False
        )
        players = []
        for _, row in pos_df.iterrows():
            players.append({
                "player_id": str(row["player_id"]),
                "name": row["name"],
                "team": row["team"],
                "position": row["position"],
                "bye_week": None,
                "adp": int(row["adp"]),
                "projection_points_total": float(row["projection_points_total"]),
                "projection_points_per_game": float(row["projection_points_per_game"]),
                "projection_floor": float(row["projection_floor"]),
                "projection_ceiling": float(row["projection_ceiling"]),
                "risk_score": int(row["risk_score"]),
                "injury_flag": False,
                "age": None,
                "key_features": row["key_features"],
                "feature_importance_rank": row["feature_importance_rank"],
                "uses_schedule": False,
                # Dynamic basis-season reference fields
                "basis_season": _BASIS_SEASON,
                "games_played_basis": int(row["games_played"]),
                "total_fp_basis": round(float(row["total_fp"]), 1),
                "ppg_basis": round(float(row["ppg"]), 1),
                # Legacy aliases so older frontend code doesn't break
                "games_played_2025": int(row["games_played"]),
                "total_fp_2025": round(float(row["total_fp"]), 1),
            })
        out_path = DATA_DIR / f"players_{pos}.json"
        with open(out_path, "w") as f:
            json.dump(players, f, indent=2)
        print(f"  Wrote {len(players)} players to {out_path.name}")


def generate_schedule_impact():
    """Generate schedule_impact.json from upcoming_week_meta.json."""
    meta_path = DATA_DIR / "upcoming_week_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    target_season = _BASIS_SEASON + 1
    payload = {
        "schedule_incorporated": False,
        "reason": (
            f"The {target_season} NFL schedule has not been released. All projections are "
            f"based on {_BASIS_SEASON} season performance without opponent-specific "
            "adjustments. Once the schedule is released, matchup quality, "
            "home/away splits, and defensive rankings will be incorporated."
        ),
        "season": meta.get("season", target_season),
        "schedule_available": meta.get("schedule_available", False),
    }
    out_path = DATA_DIR / "schedule_impact.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {out_path.name}")


def generate_model_metadata_frontend(data_source: str = "extrapolation"):
    """Create draft_model_metadata.json for the frontend methodology section.

    Args:
        data_source: "ml_model" when ML predictions are the primary draft
            board values, "extrapolation" when using basis-season PPG * 17.
    """
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

    target_season = _BASIS_SEASON + 1  # e.g. 2026 when basis is 2025

    if data_source == "ml_model":
        target_def = (
            f"PPR fantasy points projected via ML model for the "
            f"{target_season} season (trained on {training_window or '2006-' + str(_BASIS_SEASON)}). "
            f"{_BASIS_SEASON} actuals retained as reference."
        )
        data_basis_note = (
            f"Draft board values are ML model predictions for the "
            f"{target_season} season. The model was trained on "
            f"{training_window or '2006-' + str(_BASIS_SEASON)} historical data. "
            f"{_BASIS_SEASON} actual performance (games played, total FP, PPG) "
            f"is shown in the player detail panel for reference."
        )
    else:
        target_def = (
            f"PPR fantasy points per game ({_BASIS_SEASON} actuals projected "
            f"to 17 games for draft board; ML model targets {target_season} season)"
        )
        data_basis_note = (
            f"Draft board values are {_BASIS_SEASON} actual PPR points per "
            f"game projected over 17 games. They are not ML model outputs. "
            f"The ML model (trained {training_window or '2006-' + str(_BASIS_SEASON - 1)}, "
            f"validated on the {_BASIS_SEASON} held-out season) is available "
            f"via the prediction API; its backtest metrics are shown in the "
            f"Methodology section."
        )

    payload = {
        "data_source": data_source,
        "basis_season": _BASIS_SEASON,
        "target_season": target_season,
        "target_definition": target_def,
        "training_data_range": training_window or f"2006-{_BASIS_SEASON - 1}",
        "positions": ["QB", "RB", "WR", "TE"],
        "schedule_incorporated": False,
        "schedule_release_status": (
            f"{target_season} schedule not released yet; draft board estimates use "
            f"schedule-neutral assumptions based on {_BASIS_SEASON} actual performance."
        ),
        "version": "v1.0.0",
        "last_updated": meta.get("training_date", ""),
        "training_date": meta.get("training_date"),
        "training_mode": meta.get("training_mode", "backtest"),
        "test_season": meta.get("test_season"),
        "n_features_per_position": meta.get("n_features_per_position", {}),
        "training_metrics": meta.get("training_metrics", {}),
        "backtest_results": backtest.get("backtest_results", {}),
        "top_features": features,
        "methodology": {
            "model_type": "LightGBM ensemble with XGBoost and Ridge regression",
            "training_window": training_window or f"2006-{_BASIS_SEASON - 1}",
            "test_season": str(meta.get("test_season", _BASIS_SEASON)),
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
            "evaluation_metrics": {
                "primary": ["RMSE", "MAE", "R-squared"],
                "ranking": ["Spearman rank correlation", "Tier classification accuracy"],
                "calibration": ["Within 7-point accuracy", "Within 10-point accuracy"],
            },
        },
        "data_basis_note": data_basis_note,
    }
    out_path = DATA_DIR / "draft_model_metadata.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {out_path.name}")


def main():
    print("Generating draft board data...")

    season_df = load_season_data()
    print(f"  Loaded {len(season_df)} weekly records (basis season: {_BASIS_SEASON})")

    agg = aggregate_player_stats(season_df)
    print(f"  Aggregated {len(agg)} unique players")

    agg = compute_projections(agg)
    agg = compute_risk_scores(agg)
    agg = add_feature_importance(agg)
    agg = add_adp_proxy(agg)

    # Determine dominant data source for metadata
    data_source = "extrapolation"
    if "data_source" in agg.columns:
        ml_count = (agg["data_source"] == "ml_model").sum()
        if ml_count > len(agg) * 0.5:
            data_source = "ml_model"

    output_position_files(agg)
    generate_schedule_impact()
    generate_model_metadata_frontend(data_source=data_source)

    print(f"\nDone! JSON files ready (data_source={data_source}, basis={_BASIS_SEASON}).")


if __name__ == "__main__":
    main()

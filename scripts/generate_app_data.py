#!/usr/bin/env python3
"""
Generate app data with ML predictions.

Runs the trained models (EnsemblePredictor) to produce predictions and merges
them into the feature data so the web app displays ML-powered projections.

Usage:
    python scripts/generate_app_data.py              # Update cached_features with predictions
    python scripts/generate_app_data.py --parquet    # Also save to daily_predictions.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def generate_app_data(save_daily: bool = False) -> bool:
    """
    Generate feature data with ML predictions for the web app.
    
    1. Load data from DB (or cached_features if exists)
    2. Run NFLPredictor to get predictions
    3. Merge predicted_points, projection_1w, projection_4w, projection_18w
    4. Save to data/cached_features.parquet (and optionally daily_predictions.parquet)
    
    Returns:
        True if successful
    """
    data_dir = Path(__file__).parent.parent / "data"
    cached_path = data_dir / "cached_features.parquet"
    daily_path = data_dir / "daily_predictions.parquet"
    
    print("Generating app data with ML predictions...")
    
    # Auto-refresh so current season completed weeks are in DB
    try:
        from src.utils.data_manager import auto_refresh_data
        auto_refresh_data()
    except Exception as e:
        print(f"  Auto-refresh skipped: {e}")

    # Prediction target (upcoming week from calendar, e.g. season week 22 = Super Bowl)
    from src.predict import get_prediction_target_week
    pred_season, pred_week = get_prediction_target_week()

    # Schedule available for prediction week: if not, we will not store matchup (opponent/home_away) so UI and data stay aligned
    from src.utils.database import DatabaseManager
    _db = DatabaseManager()
    schedule_available_for_pred = _db.has_schedule_for_season(pred_season)

    # Decide whether to use cache or rebuild from DB (use DB when cache is behind prediction target)
    full_df = None
    if cached_path.exists():
        cache_df = pd.read_parquet(cached_path)
        cache_latest_season = int(cache_df["season"].max())
        cache_latest_week = int(cache_df[cache_df["season"] == cache_latest_season]["week"].max())
        cache_behind = (
            cache_latest_season < pred_season
            or (cache_latest_season == pred_season and cache_latest_week < pred_week)
        )
        if not cache_behind:
            full_df = cache_df
            print(f"  Loaded {len(full_df)} rows from cached_features.parquet")
        else:
            print(f"  Cache is behind prediction target ({pred_season} week {pred_week}); rebuilding from DB")

    # Load predictor and get predictions
    try:
        from src.predict import NFLPredictor
        predictor = NFLPredictor()
        if not predictor.initialize():
            print("Warning: No trained models. Run: python -m src.models.train")
            print("App will use fantasy_points/fp_rolling as projection fallback.")
            return False
    except Exception as e:
        print(f"Could not load predictor: {e}")
        return False
    
    # Get predictions for multiple horizons (1w, 4w, 18w, plus dynamic default)
    from src.utils.nfl_calendar import get_current_nfl_week, is_offseason
    week_info = get_current_nfl_week()
    cur_week_num = int(week_info.get("week_num", pred_week or 1) or 1)
    if cur_week_num < 1:
        cur_week_num = 1
    # Default horizon: full season if offseason, else remaining weeks in season
    default_horizon = 18 if is_offseason() else max(1, 18 - min(cur_week_num, 18) + 1)
    horizons = [1, 4, 18]
    if default_horizon not in horizons:
        horizons.append(default_horizon)
    pred_dfs = {}
    
    for n_weeks in horizons:
        try:
            df = predictor.predict(n_weeks=n_weeks, top_n=2000)
            if not df.empty:
                cols = ["player_id", "name", "position", "team", "predicted_points"]
                if "opponent" in df.columns:
                    cols.append("opponent")
                if "home_away" in df.columns:
                    cols.append("home_away")
                if "predicted_utilization" in df.columns:
                    cols.append("predicted_utilization")
                pred_dfs[n_weeks] = df[[c for c in cols if c in df.columns]].copy()
                pred_dfs[n_weeks] = pred_dfs[n_weeks].rename(
                    columns={"predicted_points": f"projection_{n_weeks}w"}
                )
        except Exception as e:
            print(f"  Prediction for {n_weeks}w failed: {e}")

    # K/DST predictions (statistical model, not ML)
    try:
        from src.models.kicker_dst_predictor import KickerDSTPredictor, load_kicker_dst_history
        from src.predict import get_schedule_map_for_week
        kd_history = load_kicker_dst_history()
        if not kd_history.empty:
            kd_predictor = KickerDSTPredictor(_db)
            kd_schedule = get_schedule_map_for_week(_db, pred_season, pred_week)
            for n_weeks in horizons:
                kd_pred = kd_predictor.predict_all(kd_history, n_weeks=n_weeks, schedule_map=kd_schedule)
                if not kd_pred.empty:
                    cols = ["player_id", "name", "position", "team", f"projection_{n_weeks}w"]
                    if "opponent" in kd_pred.columns:
                        cols.append("opponent")
                    if "home_away" in kd_pred.columns:
                        cols.append("home_away")
                    kd_out = kd_pred[[c for c in cols if c in kd_pred.columns]].copy()
                    if n_weeks in pred_dfs:
                        pred_dfs[n_weeks] = pd.concat([pred_dfs[n_weeks], kd_out], ignore_index=True)
                    else:
                        pred_dfs[n_weeks] = kd_out
                    print(f"  Added {len(kd_out)} K/DST predictions for {n_weeks}w horizon")
    except Exception as e:
        print(f"  K/DST predictions skipped: {e}")
    
    if not pred_dfs:
        print("No predictions generated.")
        return False

    # Build full_df from DB if not using cache
    if full_df is None:
        from src.utils.database import DatabaseManager
        from src.features.utilization import engineer_all_features
        from src.features.qb_features import add_qb_features

        db = DatabaseManager()
        full_df = db.get_all_players_for_training(min_games=1)
        if full_df.empty:
            from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
            print(f"No data in database. Run: python -m src.data.nfl_data_loader (default: {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON})")
            return False
        # Filter to eligible (active) players only for the prediction rows
        from src.data.nfl_data_loader import filter_to_eligible_players
        full_df = filter_to_eligible_players(full_df)
        full_df = engineer_all_features(full_df)
        full_df = add_qb_features(full_df)
        print(f"  Computed features for {len(full_df)} rows from DB")
    
    # Add projection columns (initially NaN)
    horizons = list(pred_dfs.keys())
    for n_weeks in horizons:
        col = f"projection_{n_weeks}w"
        if col not in full_df.columns:
            full_df[col] = np.nan
    
    # Merge predictions only into LATEST season/week rows (where app shows rankings)
    latest_season = full_df["season"].max()
    latest_week = full_df[full_df["season"] == latest_season]["week"].max()
    latest_mask = (full_df["season"] == latest_season) & (full_df["week"] == latest_week)
    latest_indices = full_df.index[latest_mask]
    
    for n_weeks, pdf in pred_dfs.items():
        col = f"projection_{n_weeks}w"
        pred_map = pdf.set_index("player_id")[col].to_dict()
        # Update only latest week rows
        latest_players = full_df.loc[latest_mask, "player_id"]
        full_df.loc[latest_mask, col] = latest_players.map(pred_map).values
    
    # Attach upcoming matchup (opponent, home_away) for app display.
    # Predictions are for the prediction target week (nfl_calendar); these columns
    # are the upcoming game's opponent and home/away for that week (or ""/unknown if no schedule).
    first_pdf = pred_dfs.get(1)
    if first_pdf is None or first_pdf.empty:
        first_pdf = list(pred_dfs.values())[0] if pred_dfs else None
    if first_pdf is not None and not first_pdf.empty and "opponent" in first_pdf.columns and "home_away" in first_pdf.columns and schedule_available_for_pred:
        opp_map = first_pdf.set_index("player_id")["opponent"].fillna("").astype(str).to_dict()
        ha_map = first_pdf.set_index("player_id")["home_away"].fillna("unknown").astype(str).to_dict()
        full_df["upcoming_opponent"] = ""
        full_df["upcoming_home_away"] = "unknown"
        full_df.loc[latest_mask, "upcoming_opponent"] = full_df.loc[latest_mask, "player_id"].map(opp_map).fillna("").astype(str)
        full_df.loc[latest_mask, "upcoming_home_away"] = full_df.loc[latest_mask, "player_id"].map(ha_map).fillna("unknown").astype(str)
    else:
        full_df["upcoming_opponent"] = ""
        full_df["upcoming_home_away"] = "unknown"
    
    # Add predicted_points = projection_1w for compatibility
    if "projection_1w" in full_df.columns:
        full_df["predicted_points"] = full_df["projection_1w"]
    
    # Optional: upcoming week label for app (e.g. "Super Bowl")
    from src.utils.nfl_calendar import get_week_label
    upcoming_label = get_week_label(pred_week, pred_season)
    if is_offseason():
        default_label = "Full Season Projections"
    else:
        start_wk = int(pred_week or cur_week_num or 1)
        start_wk = max(1, min(start_wk, 18))
        default_label = f"Rest of Season (Weeks {start_wk}\u201318)"
    default_horizon_label = f"{pred_season} Season \u00b7 {default_label}"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "upcoming_week_meta.json"
    import json
    try:
        with open(meta_path, "w") as f:
            json.dump({
                "season": pred_season,
                "week": pred_week,
                "label": upcoming_label,
                "schedule_available": schedule_available_for_pred,
                "default_horizon": int(default_horizon),
                "default_horizon_label": default_horizon_label,
            }, f, indent=2)
    except Exception:
        pass

    # Validation: when prediction target is Super Bowl (week 22), log if schedule has matchup
    if pred_week == 22:
        from src.utils.database import DatabaseManager
        from src.predict import get_schedule_map_for_week
        _db = DatabaseManager()
        schedule_map = get_schedule_map_for_week(_db, pred_season, pred_week)
        sb_teams = {"SEA", "NE", "NEP"}
        in_map = [t for t in sb_teams if t in schedule_map]
        if schedule_map and in_map:
            print(f"  Validation: Super Bowl (season {pred_season}) schedule has matchup (teams in map: {in_map})")
        elif not schedule_map:
            print(f"  Validation: No schedule for season {pred_season} week 22 (Super Bowl); run auto_refresh to load schedules")

    # Build prediction-target rows (upcoming week) so parquet max(season/week) is the upcoming week
    first_pdf = pred_dfs.get(1)
    if first_pdf is not None and not first_pdf.empty:
        upcoming_rows = first_pdf.copy()
        upcoming_rows["season"] = pred_season
        upcoming_rows["week"] = pred_week
        upcoming_rows = upcoming_rows.rename(columns={
            "opponent": "upcoming_opponent",
            "home_away": "upcoming_home_away",
        })
        if not schedule_available_for_pred:
            upcoming_rows["upcoming_opponent"] = ""
            upcoming_rows["upcoming_home_away"] = "unknown"
        if "predicted_utilization" in upcoming_rows.columns:
            upcoming_rows["utilization_score"] = upcoming_rows["predicted_utilization"]
            upcoming_rows = upcoming_rows.drop(columns=["predicted_utilization"], errors="ignore")
        if "projection_1w" in upcoming_rows.columns:
            upcoming_rows["predicted_points"] = upcoming_rows["projection_1w"]
        # Current-season roster from nfl-data-py so team reflects trades/signings
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            loader = NFLDataLoader()
            current_roster = loader.load_rosters([pred_season])
            if not current_roster.empty and "player_id" in current_roster.columns and "team" in current_roster.columns:
                roster_team = current_roster.groupby("player_id")["team"].last().to_dict()
                existing_team = upcoming_rows["team"].copy()
                upcoming_rows["team"] = upcoming_rows["player_id"].map(roster_team)
                upcoming_rows["team"] = upcoming_rows["team"].fillna(existing_team)
                print(f"  Aligned team with current-season roster ({pred_season}, {len(roster_team)} players)")
        except Exception as e:
            print(f"  Current-season roster refresh skipped: {e}")
        # Merge projection_4w and projection_18w from other horizons
        for n_weeks in (4, 18):
            if n_weeks in pred_dfs and not pred_dfs[n_weeks].empty and f"projection_{n_weeks}w" in pred_dfs[n_weeks].columns:
                merge_df = pred_dfs[n_weeks][["player_id", f"projection_{n_weeks}w"]].drop_duplicates(subset=["player_id"])
                upcoming_rows = upcoming_rows.merge(merge_df, on="player_id", how="left")
        # Next-season roster: when horizon can span next season (e.g. SB week), attach team_next_season
        next_season = pred_season + 1
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            from config.settings import POSITIONS
            loader = NFLDataLoader()
            roster_df = loader.load_rosters([next_season])
            if not roster_df.empty and "player_id" in roster_df.columns and "team" in roster_df.columns:
                # one row per player (last team if multiple)
                roster_team = roster_df.groupby("player_id")["team"].last().to_dict()
                upcoming_rows["team_next_season"] = upcoming_rows["player_id"].map(roster_team)
                print(f"  Attached team_next_season for {next_season} ({len(roster_team)} players)")
            else:
                upcoming_rows["team_next_season"] = np.nan
        except Exception as e:
            upcoming_rows["team_next_season"] = np.nan
            print(f"  Next-season roster skipped: {e}")
        # Ensure projection_* and team_next_season exist in full_df so concat preserves them
        for col in ["projection_1w", "projection_4w", "projection_18w", "team_next_season"]:
            if col not in full_df.columns:
                full_df[col] = np.nan
        # Ensure all columns from full_df exist in upcoming_rows (NaN for display-only rows)
        for col in full_df.columns:
            if col not in upcoming_rows.columns:
                upcoming_rows[col] = np.nan
        # Align column order: full_df columns, explicitly keeping projection_* in upcoming_rows
        cols_for_upcoming = [c for c in full_df.columns if c in upcoming_rows.columns]
        upcoming_rows = upcoming_rows[cols_for_upcoming]
        # Avoid duplicate rows for prediction week: drop existing (pred_season, pred_week) from full_df before concat
        if "season" in full_df.columns and "week" in full_df.columns:
            full_df = full_df[~((full_df["season"] == pred_season) & (full_df["week"] == pred_week))]
        full_df = pd.concat([full_df, upcoming_rows], ignore_index=True)
        print(f"  Added {len(upcoming_rows)} prediction-target rows for {pred_season} week {pred_week}")
        # Validation: log that 4w/18w differ from 1w
        if "projection_1w" in upcoming_rows.columns and "projection_18w" in upcoming_rows.columns:
            u1 = upcoming_rows["projection_1w"].dropna()
            u18 = upcoming_rows["projection_18w"].dropna()
            if len(u1) > 0 and len(u18) > 0:
                print(f"  Validation: projection_1w range [{u1.min():.1f}, {u1.max():.1f}], projection_18w range [{u18.min():.1f}, {u18.max():.1f}]")

    # Save
    full_df.to_parquet(cached_path, index=False)
    print(f"  Saved to {cached_path}")

    if save_daily:
        full_df.to_parquet(daily_path, index=False)
        print(f"  Saved to {daily_path}")

    print("Done. Web app will use ML projections.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", action="store_true", help="Also save to daily_predictions.parquet")
    args = parser.parse_args()
    success = generate_app_data(save_daily=args.parquet)
    sys.exit(0 if success else 1)

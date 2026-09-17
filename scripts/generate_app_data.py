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


def _load_authoritative_position_map():
    try:
        from src.utils.database import DatabaseManager
        return DatabaseManager().get_authoritative_player_positions()
    except Exception:
        return {}


def _apply_authoritative_positions(df: pd.DataFrame, pos_map: dict) -> pd.DataFrame:
    if df.empty or not pos_map or "player_id" not in df.columns:
        return df
    out = df.copy()
    authoritative = out["player_id"].astype(str).map(pos_map)
    if "position" in out.columns:
        out["position"] = authoritative.where(authoritative.notna(), out["position"])
    else:
        out["position"] = authoritative
    return out


def _db_stats_fingerprint(db) -> dict:
    """Cheap content fingerprint of player_weekly_stats.

    created_at alone is not enough: the COALESCE upsert in insert_player_weekly_stats
    updates rows in place, so a re-ingest that corrects values leaves COUNT and
    MAX(created_at) unchanged. The stat sums move whenever values do.
    """
    try:
        with db._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*), ROUND(COALESCE(SUM(fantasy_points), 0), 1), "
                "COALESCE(SUM(rushing_yards), 0), COALESCE(SUM(receiving_yards), 0), "
                "COALESCE(SUM(fumbles_lost), 0), MAX(created_at) FROM player_weekly_stats"
            ).fetchone()
    except Exception as e:  # noqa: BLE001 -- a fingerprint failure must not block generation
        return {"error": str(e)}
    keys = ("n_rows", "sum_fp", "sum_rush_yds", "sum_rec_yds", "sum_fumbles_lost", "max_created_at")
    return dict(zip(keys, [None if v is None else (float(v) if isinstance(v, float) else v) for v in row]))


def _fingerprint_sidecar(cached_path: Path) -> Path:
    return cached_path.with_suffix(".fingerprint.json")


def _read_cache_fingerprint(cached_path: Path):
    import json
    try:
        return json.loads(_fingerprint_sidecar(cached_path).read_text())
    except (OSError, ValueError):
        return None   # no sidecar (cache predates this check) -> treated as stale


def _write_cache_fingerprint(cached_path: Path, fingerprint: dict) -> None:
    import json
    _fingerprint_sidecar(cached_path).write_text(json.dumps(fingerprint, indent=2, default=str))


def generate_app_data(save_daily: bool = False) -> bool:
    """
    Generate feature data with ML predictions for the web app.
    
    1. Load data from DB (or cached_features if exists)
    2. Run NFLPredictor to get predictions
    3. Merge predicted_points and projection_1w (the only trained horizon --
       4w and 18w were retired; see TRAINING_HORIZONS in config/settings.py)
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
    corrected_positions = _db.reconcile_player_positions_from_rosters()
    if corrected_positions:
        print(f"  Reconciled {corrected_positions} player positions from roster snapshots")
    authoritative_pos_map = _db.get_authoritative_player_positions()
    schedule_available_for_pred = _db.has_schedule_for_season(pred_season)

    # Decide whether to use cache or rebuild from DB. Two staleness tests:
    # behind the prediction target week, OR built from a different DB state.
    # The week test alone let a re-ingest of past seasons (2025 was reloaded
    # 2026-08-11..19, after the cache was built) go unnoticed for as long as
    # the target week stood still: the board showed J.Allen's 2025 total as
    # 226.6 (QB rushing zeroed) and every fumble-loser 2-6 points high,
    # because those corrections only ever existed in the DB.
    db_fingerprint = _db_stats_fingerprint(_db)
    full_df = None
    if cached_path.exists():
        cache_df = pd.read_parquet(cached_path)
        cache_latest_season = int(cache_df["season"].max())
        cache_latest_week = int(cache_df[cache_df["season"] == cache_latest_season]["week"].max())
        cache_behind = (
            cache_latest_season < pred_season
            or (cache_latest_season == pred_season and cache_latest_week < pred_week)
        )
        cache_fingerprint = _read_cache_fingerprint(cached_path)
        if cache_behind:
            print(f"  Cache is behind prediction target ({pred_season} week {pred_week}); rebuilding from DB")
        elif cache_fingerprint != db_fingerprint:
            print(f"  Cache was built from a different player_weekly_stats state "
                  f"(cache {cache_fingerprint}, DB {db_fingerprint}); rebuilding from DB")
        else:
            full_df = cache_df
            print(f"  Loaded {len(full_df)} rows from cached_features.parquet")

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
    
    # Get predictions for the 1-week horizon only. TRAINING_HORIZONS
    # (config/settings.py) trains 1-week only as of 2026-09-16 -- 4 was
    # retired the same way 18 was on 2026-08-29 (unconsumed: projection_4w
    # never reached players_{pos}.json/the site; never validated against
    # summing real per-week predictions, which is what
    # generate_weekly_data.py's build_weekly_model() already does for the
    # actual "next N weeks" UI). MultiWeekModel.predict raises for any
    # n_weeks with no trained representative (position_models.py) instead
    # of silently substituting the nearest one's unscaled output under the
    # wrong label -- which is what produced e.g. QB predicted_ppg values
    # off by roughly 4/18 before that guard existed. Requesting only 1 here
    # avoids relying on that guard rather than just not asking for horizons
    # nothing trains.
    from src.utils.nfl_calendar import get_current_nfl_week, is_offseason
    from config.settings import MODEL_CONFIG
    week_info = get_current_nfl_week()
    cur_week_num = int(week_info.get("week_num", pred_week or 1) or 1)
    if cur_week_num < 1:
        cur_week_num = 1
    # default_horizon is the width of the "rest of season pace" window used
    # for the label written to upcoming_week_meta.json below (e.g. "Weeks
    # 3-8"). It is NOT a horizon anything predicts: only 1-week is trained
    # (TRAINING_HORIZONS), and multi-week views are built by summing real
    # per-week predictions in generate_weekly_data.py's build_weekly_model()
    # -- not duplicated here. The 8-week cap is the old
    # MODEL_CONFIG["horizon_long_threshold"] - 1, kept as a label width only.
    MAX_LABEL_HORIZON = 8
    default_horizon = (MAX_LABEL_HORIZON if is_offseason()
                        else min(MAX_LABEL_HORIZON, max(1, 18 - min(cur_week_num, 18) + 1)))
    horizons = [1]
    pred_dfs = {}
    
    for n_weeks in horizons:
        try:
            df = predictor.predict(n_weeks=n_weeks, top_n=2000)
            if not df.empty:
                df = _apply_authoritative_positions(df, authoritative_pos_map)
                cols = ["player_id", "name", "position", "team", "predicted_points"]
                if "opponent" in df.columns:
                    cols.append("opponent")
                if "home_away" in df.columns:
                    cols.append("home_away")
                if "predicted_utilization" in df.columns:
                    cols.append("predicted_utilization")
                # Availability is reported beside the prediction, not folded
                # into it (see NFLPredictor._attach_injury_availability).
                for extra in ("expected_points", "injury_adjustment"):
                    if extra in df.columns:
                        cols.append(extra)
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
                    kd_pred = _apply_authoritative_positions(kd_pred, authoritative_pos_map)
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
        full_df = _apply_authoritative_positions(full_df, authoritative_pos_map)
        full_df = engineer_all_features(full_df)
        full_df = add_qb_features(full_df)
        print(f"  Computed features for {len(full_df)} rows from DB")
    else:
        full_df = _apply_authoritative_positions(full_df, authoritative_pos_map)
    
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
    start_wk = int(pred_week or cur_week_num or 1)
    start_wk = max(1, min(start_wk, 18))
    end_wk = start_wk + default_horizon - 1
    # Honest label: only claims the horizon actually predicted
    # (MAX_TRAINED_HORIZON weeks), never "full season" through week 18.
    default_label = f"Weeks {start_wk}\u2013{end_wk}" if end_wk > start_wk else f"Week {start_wk}"
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
        # (A projection_4w merge used to live here. Horizon 4 was retired
        # 2026-09-16 -- see TRAINING_HORIZONS -- so pred_dfs only ever has
        # key 1 and the loop was dead.)
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
        for col in ["projection_1w", "team_next_season",
                    "expected_points", "injury_adjustment"]:
            if col not in full_df.columns:
                full_df[col] = np.nan
        # Ensure all columns from full_df exist in upcoming_rows (NaN for display-only rows)
        for col in full_df.columns:
            if col not in upcoming_rows.columns:
                upcoming_rows[col] = np.nan
        # Align column order: full_df columns, explicitly keeping projection_* in upcoming_rows
        cols_for_upcoming = [c for c in full_df.columns if c in upcoming_rows.columns]
        upcoming_rows = upcoming_rows[cols_for_upcoming]
        # Avoid duplicate rows for prediction week: only drop existing rows if they
        # are prediction stubs (all stats null), NOT real game data.
        if "season" in full_df.columns and "week" in full_df.columns:
            pred_week_mask = (full_df["season"] == pred_season) & (full_df["week"] == pred_week)
            if pred_week_mask.any():
                stat_cols = [c for c in ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards"]
                             if c in full_df.columns]
                if stat_cols:
                    existing_rows = full_df[pred_week_mask]
                    is_stub = existing_rows[stat_cols].isnull().all(axis=1)
                    if is_stub.all():
                        # All existing rows are stubs — safe to replace
                        full_df = full_df[~pred_week_mask]
                    else:
                        # Real game data exists — keep it, only add players not already present
                        existing_pids = set(full_df.loc[pred_week_mask, "player_id"])
                        upcoming_rows = upcoming_rows[~upcoming_rows["player_id"].isin(existing_pids)]
                        print(f"  Preserved {(~is_stub).sum()} real game data rows for {pred_season} week {pred_week}")
                else:
                    full_df = full_df[~pred_week_mask]
        full_df = pd.concat([full_df, upcoming_rows], ignore_index=True)
        print(f"  Added {len(upcoming_rows)} prediction-target rows for {pred_season} week {pred_week}")

    # projection_18w is retired (see MAX_TRAINED_HORIZON above) but a cache
    # or daily_predictions file built before this fix can still carry stale
    # values for it (e.g. old rows where it was fabricated as
    # projection_1w * 18) that this run's `if col not in full_df.columns`
    # guards never touch, since the column already exists. Drop it outright
    # rather than let it persist as a landmine for anything that still reads
    # this file expecting a real 18-week number.
    full_df = full_df.drop(columns=["projection_18w"], errors="ignore")

    # Save (atomic write: temp file then rename to prevent corruption on crash)
    import tempfile, os
    def _atomic_save(df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=path.parent)
        try:
            os.close(fd)
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    _atomic_save(full_df, cached_path)
    _write_cache_fingerprint(cached_path, db_fingerprint)
    print(f"  Saved to {cached_path}")

    if save_daily:
        _atomic_save(full_df, daily_path)
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

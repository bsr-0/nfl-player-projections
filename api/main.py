"""
FastAPI backend for NFL Fantasy Predictor SPA.

Serves all data required by the frontend: hero stats, pipeline, training years,
EDA sample, utilization weights, advanced results, backtest, predictions.
Run from project root: python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import math
import json
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

# Schedule team codes (e.g. NE, SEA) may differ from roster/parquet (e.g. NEP, PAT). Map schedule code -> roster aliases.
TEAM_ALIASES: Dict[str, Set[str]] = {
    "NE": {"NEP", "PAT"},   # Patriots: schedule often "NE", roster may be "NEP" or "PAT"
}
# No alias needed for SEA; other teams use schedule abbreviation as-is.

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Ensure project root is on path when running uvicorn api.main:app
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import MODELS_DIR

from src.app_data import (
    load_advanced_model_results,
    load_backtest_results,
    load_eda_sample,
    load_predictions_parquet,
    load_training_years_analysis,
    load_qb_target_choice,
    get_utilization_weights_merged,
    load_ts_backtest_results,
    load_ts_backtest_predictions,
)

app = FastAPI(
    title="NFL Predictor API",
    description="Data API for NFL Fantasy Predictor web app",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _teams_playing_for_week(schedule_codes_upper: Set[str]) -> Set[str]:
    """Expand schedule team codes with roster aliases so parquet/roster team column matches."""
    out = set(schedule_codes_upper)
    for code in schedule_codes_upper:
        if code in TEAM_ALIASES:
            out |= TEAM_ALIASES[code]
    return out


def _json_safe_value(v: Any) -> Any:
    """Convert a value to something JSON-serializable (replace nan/inf with None)."""
    if v is None:
        return None
    if hasattr(v, "shape") and getattr(v, "size", 0):
        try:
            return _json_safe_value(v.flat[0]) if v.size else None
        except (IndexError, TypeError, ValueError):
            return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if hasattr(v, "__int__") and not isinstance(v, bool):
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
    if hasattr(v, "__float__"):
        try:
            x = float(v)
            return None if math.isnan(x) or math.isinf(x) else x
        except (ValueError, TypeError):
            pass
    try:
        return str(v)
    except Exception:
        return None


def _load_optional_json(path: Path) -> Dict[str, Any]:
    """Load optional JSON payload safely."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Hero
# -----------------------------------------------------------------------------
@app.get("/api/hero")
def get_hero() -> Dict[str, Any]:
    """Record count and optional correlation for hero section."""
    df_eda, stats = load_eda_sample(8000)
    record_count = stats.get("row_count", 0) or (len(df_eda) if df_eda is not None and not df_eda.empty else 0)
    res = load_advanced_model_results()
    backlist = load_backtest_results()
    correlation = None
    if res and res.get("backtest_results"):
        by_pos = res["backtest_results"]
        r2_list = [by_pos[p].get("r2") for p in by_pos if isinstance(by_pos[p], dict)]
        if r2_list:
            import numpy as np
            correlation = round(float(np.sqrt(max(0, np.mean(r2_list)))) * 100, 1)
    if correlation is None and backlist:
        latest = backlist[0]
        metrics = latest.get("metrics", {})
        corr = metrics.get("correlation")
        if corr is not None:
            correlation = round(float(corr) * 100, 1)
    return {"record_count": record_count, "correlation": correlation}


# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------
@app.get("/api/data-pipeline")
def get_data_pipeline() -> Dict[str, Any]:
    """Stats for pipeline visualization: row_count, seasons, n_features, health."""
    df_eda, stats = load_eda_sample(5000)
    row_count = stats.get("row_count", 0) or (len(df_eda) if df_eda is not None and not df_eda.empty else 0)
    seasons = stats.get("seasons", [])
    n_features = stats.get("n_features", 0)
    season_range = f"{min(seasons)}–{max(seasons)}" if seasons else ""
    health = "OK" if row_count > 1000 else "Low volume"
    return {
        "row_count": row_count,
        "seasons": seasons,
        "season_range": season_range,
        "n_features": n_features,
        "health": health,
    }


# -----------------------------------------------------------------------------
# Training years
# -----------------------------------------------------------------------------
@app.get("/api/training-years")
def get_training_years() -> List[Dict[str, Any]]:
    """Training years analysis for radial cards (position, test_correlation, n_train, train_range)."""
    df = load_training_years_analysis()
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


# -----------------------------------------------------------------------------
# EDA
# -----------------------------------------------------------------------------
@app.get("/api/eda")
def get_eda(max_rows: int = Query(5000, ge=100, le=20000)) -> Dict[str, Any]:
    """Sample + stats for heatmap and fantasy distribution (filter by position on client)."""
    df, stats = load_eda_sample(max_rows=max_rows)
    if df is None or df.empty:
        return {"sample": [], "stats": {"row_count": 0, "seasons": [], "n_features": 0}}
    # Convert to JSON-serializable; replace nan/inf with None
    sample = df.head(max_rows)
    records = []
    for _, row in sample.iterrows():
        d = {}
        for c in sample.columns:
            d[c] = _json_safe_value(row[c])
        records.append(d)
    return {
        "sample": records,
        "stats": {
            "row_count": stats.get("row_count", len(records)),
            "seasons": stats.get("seasons", []),
            "n_features": stats.get("n_features", 0),
        },
    }


# -----------------------------------------------------------------------------
# Utilization weights
# -----------------------------------------------------------------------------
@app.get("/api/utilization-weights")
def get_utilization_weights() -> Dict[str, Dict[str, float]]:
    """Per-position utilization component weights."""
    return get_utilization_weights_merged()


# -----------------------------------------------------------------------------
# Advanced results
# -----------------------------------------------------------------------------
@app.get("/api/advanced-results")
def get_advanced_results() -> Dict[str, Any]:
    """Full advanced_model_results.json (by_position, comparison, train_seasons, test_season, baseline_comparison, timestamp)."""
    res = load_advanced_model_results()
    if res is None:
        return {}
    return res


@app.get("/api/model-config")
def get_model_config() -> Dict[str, Any]:
    """Serve model configuration metadata needed by UI (e.g. QB dependent variable)."""
    model_metadata = _load_optional_json(MODELS_DIR / "model_metadata.json")
    horizon_status = _load_optional_json(MODELS_DIR / "horizon_model_status.json")
    monitoring = _load_optional_json(MODELS_DIR / "model_monitoring_report.json")
    return {
        "qb_target": load_qb_target_choice(),  # "util" or "fp"
        "model_metadata": model_metadata,
        "horizon_model_status": horizon_status,
        "monitoring_report": monitoring,
    }


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
@app.get("/api/backtest")
def get_backtest() -> Dict[str, Any]:
    """List of backtest files + latest payload (by_position, by_week, metrics, baseline_comparison)."""
    backlist = load_backtest_results()
    res = load_advanced_model_results()
    latest = backlist[0] if backlist else (res or {})
    by_position = latest.get("by_position") or (res.get("backtest_results") if res else None) or {}
    by_week = latest.get("by_week") or {}
    metrics = latest.get("metrics") or {}
    baseline_comparison = (res or {}).get("baseline_comparison")
    files = []
    if backlist:
        for b in backlist:
            files.append({
                "season": b.get("season"),
                "backtest_date": b.get("backtest_date"),
            })
    return {
        "files": files,
        "latest": {
            "by_position": by_position,
            "by_week": by_week,
            "metrics": metrics,
            "baseline_comparison": baseline_comparison,
            "season": latest.get("season"),
            "backtest_date": latest.get("backtest_date"),
        },
        "train_seasons": (res or {}).get("train_seasons"),
        "test_season": (res or {}).get("test_season"),
    }


@app.get("/api/backtest/{season:int}")
def get_backtest_season(season: int) -> Dict[str, Any]:
    """Backtest payload for a specific season (by_week, etc.)."""
    backlist = load_backtest_results()
    for b in backlist:
        if b.get("season") == season:
            return b
    return JSONResponse(status_code=404, content={"detail": "Backtest not found for season"})


# -----------------------------------------------------------------------------
# Time-Series Backtest (expanding window, weekly refit)
# -----------------------------------------------------------------------------
@app.get("/api/ts-backtest")
def get_ts_backtest() -> Dict[str, Any]:
    """List available time-series backtest results + latest metrics summary."""
    ts_results = load_ts_backtest_results()
    if not ts_results:
        return {"results": [], "latest": None, "available_seasons": []}
    available_seasons = sorted({r.get("season") for r in ts_results if r.get("season")})
    return {
        "results": ts_results,
        "latest": ts_results[0],
        "available_seasons": available_seasons,
    }


@app.get("/api/ts-backtest/{season:int}")
def get_ts_backtest_season(season: int) -> Dict[str, Any]:
    """Metrics for a specific season's time-series backtest."""
    ts_results = load_ts_backtest_results()
    for r in ts_results:
        if r.get("season") == season:
            return r
    return JSONResponse(status_code=404, content={"detail": f"No TS backtest for season {season}"})


@app.get("/api/ts-backtest/{season:int}/predictions")
def get_ts_backtest_predictions(
    season: int,
    position: Optional[str] = Query(None),
    week: Optional[int] = Query(None),
) -> Dict[str, Any]:
    """Per-player, per-week predictions vs actuals from the time-series backtest."""
    df = load_ts_backtest_predictions(season)
    if df is None or df.empty:
        return {"rows": [], "season": season}
    if position:
        df = df[df["position"].str.upper() == position.upper()]
    if week is not None:
        df = df[df["week"] == week]
    records = []
    for _, row in df.iterrows():
        d = {}
        for c in df.columns:
            d[c] = _json_safe_value(row[c])
        records.append(d)
    return {"rows": records, "season": season}


# -----------------------------------------------------------------------------
# Predictions
# -----------------------------------------------------------------------------
def _parse_horizon(horizon: Optional[str]) -> Optional[int]:
    """Parse horizon query: None or 'all' -> None (all horizons); '1','4','18' -> int."""
    if horizon is None or str(horizon).strip().lower() == "all":
        return None
    try:
        n = int(horizon)
        if n in (1, 4, 18):
            return n
    except (ValueError, TypeError):
        pass
    return None


@app.get("/api/predictions")
def get_predictions(
    position: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    horizon: Optional[str] = Query(None, description="Time horizon: 1, 4, 18, or 'all' for all"),
):
    """Predictions table from parquet as JSON. Returns only prediction-target (upcoming) week rows. Optional filter by position, name (substring), and horizon."""
    try:
        return _get_predictions_impl(position=position, name=name, horizon=horizon)
    except Exception as e:
        logger.exception("Predictions endpoint failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "type": type(e).__name__},
        )


def _get_predictions_impl(
    position: Optional[str] = None,
    name: Optional[str] = None,
    horizon: Optional[str] = None,
) -> Dict[str, Any]:
    horizon_int = _parse_horizon(horizon)
    df = load_predictions_parquet()
    if df is None or df.empty:
        return {
            "rows": [],
            "qb_target": load_qb_target_choice(),
            "week_label": "",
            "upcoming_week_label": None,
            "schedule_available": True,
            "schedule_note": "",
            "schedule_by_horizon": {"1": True, "4": True, "18": True},
            "horizon_weeks": [],
            "horizon_weeks_label": "",
        }

    meta_path = _PROJECT_ROOT / "data" / "upcoming_week_meta.json"
    pred_season = None
    pred_week = None
    upcoming_label = None
    if meta_path.exists():
        try:
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            pred_season = meta.get("season")
            pred_week = meta.get("week")
            upcoming_label = meta.get("label")
        except Exception:
            pass

    # Filter to prediction-target week when meta is present so we serve upcoming week only
    if pred_season is not None and pred_week is not None and "season" in df.columns and "week" in df.columns:
        df = df[(df["season"] == pred_season) & (df["week"] == pred_week)]
    elif "season" in df.columns and "week" in df.columns:
        latest_season = int(df["season"].max())
        latest_week = int(df[df["season"] == latest_season]["week"].max())
        df = df[(df["season"] == latest_season) & (df["week"] == latest_week)]

    if position:
        if "position" in df.columns:
            df = df[df["position"].astype(str).str.upper() == position.upper()]
    if name and name.strip() and "name" in df.columns:
        df = df[df["name"].astype(str).str.lower().str.contains(name.strip().lower(), na=False)]

    # When horizon is exactly 1 week, show only players from teams playing in the upcoming week (e.g. Super Bowl: Patriots and Seahawks only). When "all" horizons, do not filter so chart can show 1w/4w/18w for everyone.
    if horizon_int == 1 and pred_season is not None and pred_week is not None and "team" in df.columns and not df.empty:
        try:
            from src.utils.database import DatabaseManager
            _db = DatabaseManager()
            schedule = _db.get_schedule(season=pred_season, week=pred_week)
            if schedule is not None and not schedule.empty and "home_team" in schedule.columns and "away_team" in schedule.columns:
                schedule_codes = set(
                    schedule["home_team"].dropna().astype(str).str.strip().str.upper()
                ) | set(
                    schedule["away_team"].dropna().astype(str).str.strip().str.upper()
                )
                teams_playing = _teams_playing_for_week(schedule_codes)
                df["_team_upper"] = df["team"].astype(str).str.strip().str.upper()
                df = df[df["_team_upper"].isin(teams_playing)].drop(columns=["_team_upper"], errors="ignore")
        except Exception as e:
            logger.debug("1-week team filter skipped (schedule or db unavailable): %s", e)

    # Deduplicate: one row per player, keep row with largest projection for selected horizon (or projection_1w when "all")
    if not df.empty:
        if "player_id" in df.columns:
            df["_player_key"] = df["player_id"].astype(str).fillna("")
        else:
            name_s = df["name"].astype(str).fillna("") if "name" in df.columns else pd.Series("", index=df.index)
            pos_s = df["position"].astype(str).fillna("") if "position" in df.columns else pd.Series("", index=df.index)
            team_s = df["team"].astype(str).fillna("") if "team" in df.columns else pd.Series("", index=df.index)
            df["_player_key"] = name_s + "|" + pos_s + "|" + team_s
        sort_col = None
        if horizon_int in (1, 4, 18):
            sort_col = f"projection_{horizon_int}w"
            if sort_col not in df.columns and horizon_int == 1:
                sort_col = "predicted_points" if "predicted_points" in df.columns else None
            elif sort_col not in df.columns:
                sort_col = None
        elif horizon_int is None:
            sort_col = "projection_1w" if "projection_1w" in df.columns else ("predicted_points" if "predicted_points" in df.columns else None)
        if sort_col:
            # Prefer rows with non-empty team when projection ties, so response includes team for display
            if "team" in df.columns:
                df["_has_team"] = df["team"].astype(str).str.strip().fillna("").ne("")
                df = df.sort_values(
                    [sort_col, "_has_team"],
                    ascending=[False, False],
                    na_position="last",
                )
                df = df.drop(columns=["_has_team"], errors="ignore")
            else:
                df = df.sort_values(sort_col, ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["_player_key"], keep="first")
        df = df.drop(columns=["_player_key"], errors="ignore")

    # Week label: prefer upcoming label from meta
    if upcoming_label:
        week_label = upcoming_label
    elif pred_season is not None and pred_week is not None:
        week_label = f"Season {pred_season}, Week {pred_week}"
    elif "season" in df.columns and "week" in df.columns and not df.empty:
        latest_season = int(df["season"].max())
        latest_week = int(df[df["season"] == latest_season]["week"].max())
        week_label = f"Season {latest_season}, Week {latest_week}"
    else:
        week_label = ""

    # Column for projected_points by horizon (do not fall back to 1w when user chose 4w/18w). When "all", no single proj_col; frontend uses projection_1w/4w/18w.
    proj_col = None
    if horizon_int in (1, 4, 18):
        proj_col = f"projection_{horizon_int}w"
        if proj_col not in df.columns:
            if horizon_int == 1:
                proj_col = "predicted_points" if "predicted_points" in df.columns else None
            else:
                proj_col = None  # keep horizon-specific; do not show 1w as 18w
    records = []
    for _, row in df.iterrows():
        d = {}
        for c in df.columns:
            d[c] = _json_safe_value(row[c])
        if proj_col and proj_col in df.columns:
            d["projected_points"] = _json_safe_value(row[proj_col])
        records.append(d)

    # Horizon calendar and schedule-availability: compute for 1w, 4w, 18w so frontend can show "Schedule used" per horizon
    horizon_weeks_list: List[List[int]] = []
    horizon_weeks_label = ""
    schedule_available = True
    schedule_note = "Includes schedule and matchups for all weeks."
    schedule_by_horizon: Dict[str, bool] = {"1": True, "4": True, "18": True}
    h = horizon_int if horizon_int in (1, 4, 18) else 1
    try:
        from src.utils.nfl_calendar import get_next_n_nfl_weeks, get_current_nfl_season
        from src.utils.database import DatabaseManager
        weeks_tuples = get_next_n_nfl_weeks(None, h)
        horizon_weeks_list = [[s, w] for s, w in weeks_tuples]
        if weeks_tuples:
            first_s, first_w = weeks_tuples[0]
            last_s, last_w = weeks_tuples[-1]
            if first_s == last_s:
                horizon_weeks_label = f"Next {h} NFL week{'s' if h > 1 else ''}: {first_s}–{first_w}" + (f" through {last_w}" if last_w != first_w else "")
            else:
                horizon_weeks_label = f"Next {h} NFL weeks: {first_s}–{first_w} through {last_s}–{last_w}"
        cur_season = get_current_nfl_season()
        db = DatabaseManager()
        for horizon_weeks in (1, 4, 18):
            w_tuples = get_next_n_nfl_weeks(None, horizon_weeks)
            seasons_in_h = list({s for s, _ in w_tuples})
            avail = True
            for s in seasons_in_h:
                if not db.has_schedule_for_season(s):
                    avail = False
                    break
            schedule_by_horizon[str(horizon_weeks)] = avail
            if not avail:
                schedule_available = False
        # Build horizon-aware schedule_note after the loop
        if schedule_available:
            schedule_note = "Includes schedule and matchups for all weeks."
        else:
            parts = []
            for h in ("1", "4", "18"):
                if schedule_by_horizon.get(h):
                    parts.append(f"{h}w: schedule available")
                else:
                    parts.append(f"{h}w: includes next season (not yet released) or missing")
            schedule_note = "; ".join(parts) + "."
    except Exception:
        pass

    out = {
        "rows": records,
        "qb_target": load_qb_target_choice(),
        "week_label": week_label,
        "schedule_available": schedule_available,
        "schedule_note": schedule_note,
        "schedule_by_horizon": schedule_by_horizon,
        "horizon_weeks": horizon_weeks_list,
        "horizon_weeks_label": horizon_weeks_label or "",
    }
    if upcoming_label:
        out["upcoming_week_label"] = upcoming_label
    return out


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# Serve frontend: static build if present, else a fallback page
_dist = _PROJECT_ROOT / "frontend" / "dist"

if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
else:
    _FALLBACK_HTML = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>NFL Predictor API</title></head>
    <body style="font-family: system-ui; max-width: 600px; margin: 2rem auto; padding: 1rem; background: #0a0e1f; color: #cbd5e1;">
    <h1 style="color: #00f5ff;">NFL Predictor API</h1>
    <p>API is running. To use the web app:</p>
    <ol>
    <li>Run the frontend dev server: <code>cd frontend && npm install && npm run dev</code></li>
    <li>Open the URL Vite prints (e.g. <a href="http://localhost:5173" style="color: #a78bfa;">http://localhost:5173</a>)</li>
    <li>Or build and serve from here: <code>cd frontend && npm run build</code>, then restart this server</li>
    </ol>
    <p><a href="/api/health" style="color: #a78bfa;">/api/health</a> &middot; <a href="/api/hero" style="color: #a78bfa;">/api/hero</a></p>
    </body>
    </html>
    """

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return _FALLBACK_HTML

    @app.get("/favicon.ico")
    def favicon():
        from starlette.responses import Response
        return Response(status_code=204)

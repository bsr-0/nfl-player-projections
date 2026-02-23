"""
Data loading and Plotly theme helpers for the narrative NFL predictor web app.

Loads all app JSON/CSV/parquet and optionally samples DB for EDA.
Provides a shared Plotly layout (Catppuccin Mocha-inspired dark theme).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

# Project root: assume we are in src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def _models_dir() -> Path:
    return _PROJECT_ROOT / "data" / "models"


# -----------------------------------------------------------------------------
# Plotly theme (Catppuccin Mocha-inspired)
# -----------------------------------------------------------------------------
THEME = {
    "paper_bgcolor": "#0a0e27",
    "plot_bgcolor": "#1a1f3a",
    "font": {"color": "#e2e8f0", "family": "Inter, sans-serif"},
    "title": {"font": {"color": "#f1f5f9"}, "x": 0.5, "xanchor": "center"},
    "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
    "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
    "colorway": ["#00f5ff", "#b794f6", "#4ade80", "#fbbf24", "#f472b6", "#94a3b8"],
    "margin": {"t": 40, "r": 20, "b": 40, "l": 50},
    "hoverlabel": {"bgcolor": "#1a1f3a", "font_color": "#e2e8f0", "bordercolor": "#475569"},
    "legend": {"bgcolor": "rgba(10,14,39,0.8)", "bordercolor": "#334155"},
}


def nfl_plotly_layout(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """Apply dark theme to a Plotly figure. Optionally set title."""
    layout = dict(
        paper_bgcolor=THEME["paper_bgcolor"],
        plot_bgcolor=THEME["plot_bgcolor"],
        font=THEME["font"],
        margin=THEME.get("margin", {}),
    )
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig


# -----------------------------------------------------------------------------
# Data loaders (return None or empty when missing)
# -----------------------------------------------------------------------------
def load_advanced_model_results() -> Optional[Dict[str, Any]]:
    """Load data/advanced_model_results.json."""
    path = _data_dir() / "advanced_model_results.json"
    if not path.exists():
        return None
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_backtest_results() -> List[Dict[str, Any]]:
    """Load all JSON files from data/backtest_results/ (newest first)."""
    dir_path = _data_dir() / "backtest_results"
    if not dir_path.exists():
        return []
    files = sorted(dir_path.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            import json
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def load_ts_backtest_results() -> List[Dict[str, Any]]:
    """Load all ts_backtest_*.json files from data/backtest_results/ (newest first)."""
    dir_path = _data_dir() / "backtest_results"
    if not dir_path.exists():
        return []
    files = sorted(dir_path.glob("ts_backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            import json
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def load_ts_backtest_predictions(season: int = None) -> Optional["pd.DataFrame"]:
    """Load the latest ts_backtest predictions CSV for a given season."""
    dir_path = _data_dir() / "backtest_results"
    if not dir_path.exists():
        return None
    pattern = f"ts_backtest_{season}_*_predictions.csv" if season else "ts_backtest_*_predictions.csv"
    files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return pd.read_csv(files[0])
    except Exception:
        return None


def load_utilization_weights_from_file() -> Optional[Dict[str, Any]]:
    """Load data/models/utilization_weights.json if present."""
    path = _models_dir() / "utilization_weights.json"
    if not path.exists():
        return None
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_training_years_analysis() -> Optional[pd.DataFrame]:
    """Load data/training_years_analysis.csv."""
    path = _data_dir() / "training_years_analysis.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_training_years_optimization() -> Optional[pd.DataFrame]:
    """Load data/training_years_optimization.csv."""
    path = _data_dir() / "training_years_optimization.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_optimal_training_config() -> Optional[Dict[str, Any]]:
    """Load data/optimal_training_config.json."""
    path = _data_dir() / "optimal_training_config.json"
    if not path.exists():
        return None
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# Standard projection columns and display labels (1w, 4w, 5w, 18w supported)
PROJECTION_LABELS = {
    "projection_1w": "1 week",
    "projection_4w": "4 weeks",
    "projection_5w": "5 weeks",
    "projection_18w": "18 weeks",
    "predicted_points": "1 week",
}


def get_projection_columns_and_labels(df: Optional[pd.DataFrame]) -> List[tuple]:
    """Return list of (column_name, display_label) for projection cols present in df."""
    if df is None or df.empty:
        return []
    return [(c, PROJECTION_LABELS[c]) for c in PROJECTION_LABELS if c in df.columns]


def load_predictions_parquet() -> Optional[pd.DataFrame]:
    """Load data/daily_predictions.parquet or data/cached_features.parquet."""
    for name in ("daily_predictions.parquet", "cached_features.parquet"):
        path = _data_dir() / name
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                continue
    return None


def load_upcoming_week_meta() -> Optional[Dict[str, Any]]:
    """Load data/upcoming_week_meta.json (season, week, label). Returns None if missing."""
    path = _data_dir() / "upcoming_week_meta.json"
    if not path.exists():
        return None
    try:
        import json
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_qb_target_choice() -> str:
    """Load QB target (util/fp) from data/models/qb_target_choice.json. Default 'util'."""
    path = _models_dir() / "qb_target_choice.json"
    if not path.exists():
        return "util"
    try:
        import json
        with open(path) as f:
            return json.load(f).get("qb_target", "util")
    except Exception:
        return "util"


def load_eda_sample(max_rows: int = 5000) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load a sample of player data for EDA (correlations, distributions).
    Returns (df, stats_dict) where stats_dict has keys like row_count, seasons, etc.
    Uses DB if available; else falls back to parquet. Cached by caller (e.g. st.cache_data).
    """
    stats = {"row_count": 0, "seasons": [], "n_features": 0}
    try:
        from config.settings import DATA_DIR
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        df = db.get_all_players_for_training(min_games=1)
        if df is not None and not df.empty:
            # Filter to eligible (active) players so retired players don't appear
            try:
                from src.data.nfl_data_loader import filter_to_eligible_players
                df = filter_to_eligible_players(df)
            except Exception:
                pass
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            stats["row_count"] = len(df)
            if "season" in df.columns:
                stats["seasons"] = sorted(df["season"].unique().tolist())
            numeric = df.select_dtypes(include=["number"]).columns.tolist()
            stats["n_features"] = len(numeric)
            return df, stats
    except Exception:
        pass
    df = load_predictions_parquet()
    if df is not None and not df.empty:
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        stats["row_count"] = len(df)
        if "season" in df.columns:
            stats["seasons"] = sorted(df["season"].unique().tolist())
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        stats["n_features"] = len(numeric)
        return df, stats
    return None, stats


def get_utilization_weights_merged() -> Dict[str, Dict[str, float]]:
    """
    Merge config UTILIZATION_WEIGHTS with data-driven weights from file.
    File weights (per position) override config when present.
    """
    from config.settings import UTILIZATION_WEIGHTS
    out = {pos: dict(weights) for pos, weights in UTILIZATION_WEIGHTS.items()}
    file_weights = load_utilization_weights_from_file()
    if file_weights and isinstance(file_weights, dict):
        # File may have "weights": { "RB": {...}, ... } or direct position keys
        nested = file_weights.get("weights", file_weights)
        for pos, wdict in nested.items():
            if isinstance(wdict, dict) and pos in out:
                out[pos] = {k: float(v) for k, v in wdict.items() if isinstance(v, (int, float))}
    return out

"""
Predictor for Kicker and DST positions using ML models with statistical fallback.

Uses gradient boosting on engineered features (rolling averages, matchup context,
team trends) for K and DST projections. Falls back to weighted rolling averages
when insufficient training data is available.

Produces output compatible with the offensive EnsemblePredictor so
predictions merge seamlessly into the app data pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import SCORING_KICKER, SCORING_DST, MODELS_DIR

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _build_kicker_features(grp: pd.DataFrame) -> Dict[str, float]:
    """Engineer features for a single kicker from their game history."""
    fp = grp["fantasy_points"].values
    n = len(fp)
    feats = {
        "fp_mean_4": fp[-4:].mean() if n >= 4 else fp.mean(),
        "fp_mean_8": fp[-8:].mean() if n >= 8 else fp.mean(),
        "fp_season_avg": fp.mean(),
        "fp_std_8": np.std(fp[-8:]) if n >= 8 else np.std(fp),
        "fp_trend": (fp[-4:].mean() - fp[-8:].mean()) if n >= 8 else 0.0,
        "games_played": float(n),
    }
    if "fg_att" in grp.columns and "fg_made" in grp.columns:
        feats["fg_rate"] = grp["fg_made"].sum() / max(grp["fg_att"].sum(), 1)
        feats["fg_att_per_game"] = grp["fg_att"].mean()
    else:
        feats["fg_rate"] = 0.85
        feats["fg_att_per_game"] = 1.8
    if "xp_made" in grp.columns:
        feats["xp_per_game"] = grp["xp_made"].mean()
    else:
        feats["xp_per_game"] = 3.0
    return feats


def _build_dst_features(grp: pd.DataFrame) -> Dict[str, float]:
    """Engineer features for a single DST from their game history."""
    fp = grp["fantasy_points"].values
    n = len(fp)
    feats = {
        "fp_mean_4": fp[-4:].mean() if n >= 4 else fp.mean(),
        "fp_mean_8": fp[-8:].mean() if n >= 8 else fp.mean(),
        "fp_season_avg": fp.mean(),
        "fp_std_8": np.std(fp[-8:]) if n >= 8 else np.std(fp),
        "fp_trend": (fp[-4:].mean() - fp[-8:].mean()) if n >= 8 else 0.0,
        "games_played": float(n),
    }
    feats["sack_rate"] = grp["sacks"].mean() if "sacks" in grp.columns else 2.0
    feats["int_rate"] = grp["interceptions"].mean() if "interceptions" in grp.columns else 0.8
    feats["pa_avg"] = grp["points_allowed"].mean() if "points_allowed" in grp.columns else 21.0
    if "fumble_recoveries" in grp.columns:
        feats["fr_rate"] = grp["fumble_recoveries"].mean()
    else:
        feats["fr_rate"] = 0.5
    return feats


class KickerDSTPredictor:
    """ML-enhanced predictor for K and DST positions."""

    KICKER_FEATURE_COLS = [
        "fp_mean_4", "fp_mean_8", "fp_season_avg", "fp_std_8", "fp_trend",
        "games_played", "fg_rate", "fg_att_per_game", "xp_per_game",
        "is_home",
    ]
    DST_FEATURE_COLS = [
        "fp_mean_4", "fp_mean_8", "fp_season_avg", "fp_std_8", "fp_trend",
        "games_played", "sack_rate", "int_rate", "pa_avg", "fr_rate",
        "is_home",
    ]

    def __init__(self, db=None):
        if db is None:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
        self.db = db
        self._kicker_model = None
        self._dst_model = None
        self._load_ml_models()

    def _load_ml_models(self) -> None:
        """Load persisted ML models if available."""
        k_path = MODELS_DIR / "kicker_ml.joblib"
        d_path = MODELS_DIR / "dst_ml.joblib"
        try:
            if k_path.exists():
                self._kicker_model = joblib.load(k_path)
        except Exception:
            self._kicker_model = None
        try:
            if d_path.exists():
                self._dst_model = joblib.load(d_path)
        except Exception:
            self._dst_model = None

    @staticmethod
    def train_kicker_model(
        data: pd.DataFrame, save: bool = True
    ) -> Optional["GradientBoostingRegressor"]:
        """Train a gradient boosting model for kicker projections.

        Args:
            data: Historical K weekly stats with fantasy_points column.
            save: Whether to persist the model to disk.

        Returns:
            Fitted model or None if insufficient data.
        """
        if not HAS_SKLEARN or data.empty:
            return None
        k = data[data["position"] == "K"].copy()
        if k.empty:
            return None
        k = k.sort_values(["player_id", "season", "week"])

        rows = []
        for pid, grp in k.groupby("player_id"):
            grp = grp.sort_values(["season", "week"])
            if len(grp) < 5:
                continue
            # Use each game as a training example with rolling features
            for i in range(4, len(grp)):
                hist = grp.iloc[:i]
                target = grp.iloc[i]["fantasy_points"]
                feats = _build_kicker_features(hist)
                feats["is_home"] = 1.0 if grp.iloc[i].get("home_away") == "home" else 0.0
                feats["target"] = target
                rows.append(feats)

        if len(rows) < 30:
            return None

        df = pd.DataFrame(rows)
        feature_cols = [c for c in KickerDSTPredictor.KICKER_FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y = df["target"].values

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        model._feature_cols = feature_cols

        if save:
            joblib.dump(model, MODELS_DIR / "kicker_ml.joblib")
        return model

    @staticmethod
    def train_dst_model(
        data: pd.DataFrame, save: bool = True
    ) -> Optional["GradientBoostingRegressor"]:
        """Train a gradient boosting model for DST projections."""
        if not HAS_SKLEARN or data.empty:
            return None
        d = data[data["position"] == "DST"].copy()
        if d.empty:
            return None
        d = d.sort_values(["player_id", "season", "week"])

        rows = []
        for pid, grp in d.groupby("player_id"):
            grp = grp.sort_values(["season", "week"])
            if len(grp) < 5:
                continue
            for i in range(4, len(grp)):
                hist = grp.iloc[:i]
                target = grp.iloc[i]["fantasy_points"]
                feats = _build_dst_features(hist)
                feats["is_home"] = 1.0 if grp.iloc[i].get("home_away") == "home" else 0.0
                feats["target"] = target
                rows.append(feats)

        if len(rows) < 30:
            return None

        df = pd.DataFrame(rows)
        feature_cols = [c for c in KickerDSTPredictor.DST_FEATURE_COLS if c in df.columns]
        X = df[feature_cols].fillna(0).values
        y = df["target"].values

        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        model.fit(X, y)
        model._feature_cols = feature_cols

        if save:
            joblib.dump(model, MODELS_DIR / "dst_ml.joblib")
        return model

    def predict_kickers(
        self,
        kicker_data: pd.DataFrame,
        n_weeks: int = 1,
        schedule_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Predict kicker fantasy points using ML model with statistical fallback.

        Args:
            kicker_data: DataFrame with K weekly stats (from kicker_dst_aggregator)
            n_weeks: Prediction horizon (1, 4, or 18 weeks)
            schedule_map: team -> (opponent, home_away) for upcoming week

        Returns:
            DataFrame with columns: player_id, name, position, team, opponent,
            home_away, predicted_points, prediction_std, projection_{n_weeks}w
        """
        if kicker_data.empty:
            return pd.DataFrame()

        k = kicker_data[kicker_data["position"] == "K"].copy()
        if k.empty:
            return pd.DataFrame()

        k = k.sort_values(["player_id", "season", "week"])

        results = []
        for pid, grp in k.groupby("player_id"):
            grp = grp.sort_values(["season", "week"])
            if len(grp) < 2:
                continue

            latest = grp.iloc[-1]
            fp = grp["fantasy_points"].values

            team = str(latest.get("team", ""))
            opponent = ""
            home_away = "unknown"
            if schedule_map and team in schedule_map:
                opponent, home_away = schedule_map[team]

            feats = _build_kicker_features(grp)
            feats["is_home"] = 1.0 if home_away == "home" else 0.0

            # Try ML prediction
            proj_per_week = None
            if self._kicker_model is not None:
                try:
                    fcols = getattr(self._kicker_model, "_feature_cols", self.KICKER_FEATURE_COLS)
                    x = np.array([[feats.get(c, 0.0) for c in fcols]])
                    proj_per_week = float(self._kicker_model.predict(x)[0])
                except Exception:
                    proj_per_week = None

            # Statistical fallback
            if proj_per_week is None:
                recent_4 = fp[-4:].mean() if len(fp) >= 4 else fp.mean()
                recent_8 = fp[-8:].mean() if len(fp) >= 8 else fp.mean()
                season_avg = fp.mean()
                proj_per_week = 0.50 * recent_4 + 0.30 * recent_8 + 0.20 * season_avg
                if home_away == "home":
                    proj_per_week *= 1.03

            proj_total = round(proj_per_week * n_weeks, 1)
            std = round(np.std(fp[-8:]) if len(fp) >= 8 else np.std(fp), 2) * np.sqrt(n_weeks)

            results.append({
                "player_id": pid,
                "name": str(latest.get("name", "")),
                "position": "K",
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "predicted_points": proj_total,
                "prediction_std": round(std, 2),
                f"projection_{n_weeks}w": proj_total,
                "fg_rate": round(feats.get("fg_rate", 0.85), 3),
                "games_played": len(grp),
                "model_type": "ml" if self._kicker_model is not None else "statistical",
            })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def predict_dst(
        self,
        dst_data: pd.DataFrame,
        n_weeks: int = 1,
        schedule_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Predict DST fantasy points using ML model with statistical fallback.

        Args:
            dst_data: DataFrame with DST weekly stats (from kicker_dst_aggregator)
            n_weeks: Prediction horizon
            schedule_map: team -> (opponent, home_away) for upcoming week

        Returns:
            DataFrame with prediction columns compatible with app pipeline
        """
        if dst_data.empty:
            return pd.DataFrame()

        d = dst_data[dst_data["position"] == "DST"].copy()
        if d.empty:
            return pd.DataFrame()

        d = d.sort_values(["player_id", "season", "week"])

        results = []
        for pid, grp in d.groupby("player_id"):
            grp = grp.sort_values(["season", "week"])
            if len(grp) < 2:
                continue

            latest = grp.iloc[-1]
            fp = grp["fantasy_points"].values

            team = str(latest.get("team", ""))
            opponent = ""
            home_away = "unknown"
            if schedule_map and team in schedule_map:
                opponent, home_away = schedule_map[team]

            feats = _build_dst_features(grp)
            feats["is_home"] = 1.0 if home_away == "home" else 0.0

            # Try ML prediction
            proj_per_week = None
            if self._dst_model is not None:
                try:
                    fcols = getattr(self._dst_model, "_feature_cols", self.DST_FEATURE_COLS)
                    x = np.array([[feats.get(c, 0.0) for c in fcols]])
                    proj_per_week = float(self._dst_model.predict(x)[0])
                except Exception:
                    proj_per_week = None

            # Statistical fallback
            if proj_per_week is None:
                recent_4 = fp[-4:].mean() if len(fp) >= 4 else fp.mean()
                recent_8 = fp[-8:].mean() if len(fp) >= 8 else fp.mean()
                season_avg = fp.mean()
                proj_per_week = 0.45 * recent_4 + 0.35 * recent_8 + 0.20 * season_avg
                if home_away == "home":
                    proj_per_week *= 1.05

            proj_total = round(proj_per_week * n_weeks, 1)
            std = round(np.std(fp[-8:]) if len(fp) >= 8 else np.std(fp), 2) * np.sqrt(n_weeks)

            results.append({
                "player_id": pid,
                "name": str(latest.get("name", "")),
                "position": "DST",
                "team": team,
                "opponent": opponent,
                "home_away": home_away,
                "predicted_points": proj_total,
                "prediction_std": round(std, 2),
                f"projection_{n_weeks}w": proj_total,
                "sack_rate": round(feats.get("sack_rate", 2.0), 2),
                "int_rate": round(feats.get("int_rate", 0.8), 2),
                "points_allowed_avg": round(feats.get("pa_avg", 21.0), 1),
                "games_played": len(grp),
                "model_type": "ml" if self._dst_model is not None else "statistical",
            })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def predict_all(
        self,
        data: pd.DataFrame,
        n_weeks: int = 1,
        schedule_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Predict both K and DST, returning combined DataFrame.

        Args:
            data: DataFrame containing K and/or DST weekly stats
            n_weeks: Prediction horizon
            schedule_map: team -> (opponent, home_away)

        Returns:
            Combined predictions for K and DST
        """
        dfs = []

        k_pred = self.predict_kickers(data, n_weeks=n_weeks, schedule_map=schedule_map)
        if not k_pred.empty:
            dfs.append(k_pred)

        dst_pred = self.predict_dst(data, n_weeks=n_weeks, schedule_map=schedule_map)
        if not dst_pred.empty:
            dfs.append(dst_pred)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)


def load_kicker_dst_history(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load historical K/DST data from PBP for prediction.

    Falls back to cached data if available.
    """
    from src.data.kicker_dst_aggregator import load_kicker_dst_data
    from config.settings import SEASONS_TO_SCRAPE

    if seasons is None:
        # Use last 3 seasons for prediction context
        from src.utils.nfl_calendar import get_current_nfl_season
        current = get_current_nfl_season()
        seasons = list(range(current - 2, current + 1))

    return load_kicker_dst_data(seasons)

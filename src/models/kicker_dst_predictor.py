"""
Simple statistical predictor for Kicker and DST positions.

Uses rolling averages, matchup adjustments, and team scoring context
to project K and DST fantasy points. These positions don't use the
utilization-based ML pipeline designed for offensive skill positions.

Produces output compatible with the offensive EnsemblePredictor so
predictions merge seamlessly into the app data pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import SCORING_KICKER, SCORING_DST


class KickerDSTPredictor:
    """Statistical predictor for K and DST positions."""

    def __init__(self, db=None):
        if db is None:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
        self.db = db

    def predict_kickers(
        self,
        kicker_data: pd.DataFrame,
        n_weeks: int = 1,
        schedule_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Predict kicker fantasy points using rolling averages and team context.

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

        # Compute rolling stats per kicker
        results = []
        for pid, grp in k.groupby("player_id"):
            grp = grp.sort_values(["season", "week"])
            if len(grp) < 2:
                continue

            latest = grp.iloc[-1]
            fp = grp["fantasy_points"].values

            # Rolling averages (recent games weighted more)
            recent_4 = fp[-4:].mean() if len(fp) >= 4 else fp.mean()
            recent_8 = fp[-8:].mean() if len(fp) >= 8 else fp.mean()
            season_avg = fp.mean()

            # Weighted projection: 50% recent-4, 30% recent-8, 20% season
            proj_per_week = 0.50 * recent_4 + 0.30 * recent_8 + 0.20 * season_avg

            # FG rate trend (improvement indicator)
            if "fg_att" in grp.columns and "fg_made" in grp.columns:
                fg_att = grp["fg_att"].sum()
                fg_made = grp["fg_made"].sum()
                fg_rate = fg_made / max(fg_att, 1)
            else:
                fg_rate = 0.85  # league average

            # Team and opponent context
            team = str(latest.get("team", ""))
            opponent = ""
            home_away = "unknown"
            if schedule_map and team in schedule_map:
                opponent, home_away = schedule_map[team]

            # Home kickers get slight boost (indoor/familiar conditions)
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
                "fg_rate": round(fg_rate, 3),
                "games_played": len(grp),
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
        Predict DST fantasy points using rolling averages and opponent context.

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

            # Rolling averages
            recent_4 = fp[-4:].mean() if len(fp) >= 4 else fp.mean()
            recent_8 = fp[-8:].mean() if len(fp) >= 8 else fp.mean()
            season_avg = fp.mean()

            # Weighted projection
            proj_per_week = 0.45 * recent_4 + 0.35 * recent_8 + 0.20 * season_avg

            # Defensive stats trends
            sack_rate = grp["sacks"].mean() if "sacks" in grp.columns else 2.0
            int_rate = grp["interceptions"].mean() if "interceptions" in grp.columns else 0.8
            pa_avg = grp["points_allowed"].mean() if "points_allowed" in grp.columns else 21.0

            team = str(latest.get("team", ""))
            opponent = ""
            home_away = "unknown"
            if schedule_map and team in schedule_map:
                opponent, home_away = schedule_map[team]

            # Home field advantage for defense
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
                "sack_rate": round(sack_rate, 2),
                "int_rate": round(int_rate, 2),
                "points_allowed_avg": round(pa_avg, 1),
                "games_played": len(grp),
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

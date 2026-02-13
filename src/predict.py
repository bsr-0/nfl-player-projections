"""Main prediction interface for NFL player performance."""
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    POSITIONS, MAX_PREDICTION_WEEKS, MODELS_DIR, MIN_GAMES_FOR_PREDICTION,
    SCORING_FORMATS, MAX_PREDICTION_TIME_PER_PLAYER_SECONDS,
)
from src.utils.database import DatabaseManager
from src.features.feature_engineering import FeatureEngineer
from src.features.utilization_score import UtilizationScoreCalculator
from src.models.ensemble import EnsemblePredictor
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week
# Note: This system only uses real NFL data from nfl-data-py


def get_prediction_target_week() -> Tuple[int, int]:
    """
    Return (season, week_num) for the upcoming game week we are predicting.
    Uses current NFL season and current week (or week 1 if preseason).
    """
    info = get_current_nfl_week()
    season = info.get("season", get_current_nfl_season())
    week_num = info.get("week_num", 0)
    if week_num < 1:
        week_num = 1  # Predict for week 1 when in preseason
    return season, week_num


def get_schedule_map_for_week(db: DatabaseManager, season: int, week: int) -> Dict[str, Tuple[str, str]]:
    """
    Build team -> (opponent, home_away) for the given season/week from schedule.
    Returns empty dict if no schedule; callers use neutral opponent/home_away.
    """
    schedule = db.get_schedule(season=season, week=week)
    if schedule is None or schedule.empty or "home_team" not in schedule.columns or "away_team" not in schedule.columns:
        return {}
    out = {}
    for _, row in schedule.iterrows():
        home = str(row["home_team"]).strip() if pd.notna(row.get("home_team")) else ""
        away = str(row["away_team"]).strip() if pd.notna(row.get("away_team")) else ""
        if home:
            out[home] = (away, "home")
        if away:
            out[away] = (home, "away")
    return out


def _load_utilization_weights():
    """Load persisted utilization weights for train/serve consistency."""
    import json
    path = MODELS_DIR / "utilization_weights.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


class NFLPredictor:
    """
    Main prediction interface for NFL player fantasy performance.
    
    Supports:
    - Single week predictions (next week)
    - Multi-week predictions (up to 18 weeks / full season)
    - Player-specific predictions
    - Position-filtered predictions
    - Rankings and comparisons
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        util_weights = _load_utilization_weights()
        self.utilization_calculator = UtilizationScoreCalculator(weights=util_weights)
        self.feature_engineer = FeatureEngineer()
        self.predictor = EnsemblePredictor()
        self.is_initialized = False
    
    def initialize(self):
        """Load models and prepare for predictions."""
        print("Initializing NFL Predictor...")
        
        # Load trained models
        self.predictor.load_models()
        
        if not self.predictor.is_loaded:
            print("Warning: No trained models found. Please run training first.")
            print("  python -m src.models.train")
            return False
        
        self.is_initialized = True
        print("Predictor initialized successfully.")
        return True
    
    def predict_next_week(self, position: str = None, 
                          top_n: int = 50,
                          scoring_format: str = "ppr") -> pd.DataFrame:
        """
        Predict next week's fantasy performance.
        
        Args:
            position: Optional position filter (QB, RB, WR, TE)
            top_n: Number of top players to return
            scoring_format: 'ppr', 'half_ppr', or 'standard'
            
        Returns:
            DataFrame with predictions ranked by projected points
        """
        return self.predict(n_weeks=1, position=position, top_n=top_n, scoring_format=scoring_format)
    
    def predict_season(self, position: str = None,
                       top_n: int = 50,
                       scoring_format: str = "ppr") -> pd.DataFrame:
        """
        Predict full season (18 weeks) fantasy performance.
        
        Args:
            position: Optional position filter
            top_n: Number of top players to return
            scoring_format: 'ppr', 'half_ppr', or 'standard'
            
        Returns:
            DataFrame with season projections
        """
        return self.predict(n_weeks=18, position=position, top_n=top_n, scoring_format=scoring_format)
    
    @staticmethod
    def _adjust_scoring_format(results: pd.DataFrame, scoring_format: str, n_weeks: int) -> pd.DataFrame:
        """Adjust predicted_points for scoring format (Half-PPR or Standard).

        Models are trained on PPR data. For other formats, subtract the reception
        point difference per predicted reception volume.
        """
        if scoring_format == "ppr" or scoring_format not in SCORING_FORMATS:
            results["scoring_format"] = "ppr"
            return results
        ppr_rec_pts = SCORING_FORMATS["ppr"]["receptions"]      # 1.0
        fmt_rec_pts = SCORING_FORMATS[scoring_format]["receptions"]  # 0.5 or 0
        diff = ppr_rec_pts - fmt_rec_pts  # positive means we need to subtract
        if diff == 0:
            results["scoring_format"] = scoring_format
            return results
        # Estimate receptions from data if available
        if "receptions" in results.columns:
            rec_est = results["receptions"].fillna(0) * n_weeks
        elif "receptions_roll3_mean" in results.columns:
            rec_est = results["receptions_roll3_mean"].fillna(0) * n_weeks
        else:
            rec_est = 0
        for col in ["predicted_points", "prediction_ci80_lower", "prediction_ci80_upper",
                    "prediction_ci95_lower", "prediction_ci95_upper"]:
            if col in results.columns:
                results[col] = results[col] - diff * rec_est
        results["scoring_format"] = scoring_format
        return results

    def predict(self, n_weeks: int = 1, 
                position: str = None,
                player_name: str = None,
                top_n: int = 50,
                scoring_format: str = "ppr") -> pd.DataFrame:
        """
        Make predictions for specified parameters.
        
        Args:
            n_weeks: Number of weeks to predict (1-18)
            position: Optional position filter
            player_name: Optional specific player
            top_n: Number of top players to return
            scoring_format: 'ppr', 'half_ppr', or 'standard'
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_initialized:
            if not self.initialize():
                return pd.DataFrame()
        
        # Validate n_weeks
        n_weeks = max(1, min(n_weeks, MAX_PREDICTION_WEEKS))
        
        # Prediction target week (upcoming game) and ensure schedule is loaded when available
        pred_season, pred_week = get_prediction_target_week()
        try:
            from src.utils.data_manager import DataManager
            DataManager().ensure_schedule_loaded(pred_season)
        except Exception:
            pass  # Non-fatal; predictions proceed with neutral matchup if no schedule
        
        # Load player data (min_games=1 to include rookies for cold-start handling)
        player_data = self._load_player_data(position, min_games=1)
        
        if player_data.empty:
            from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
            raise ValueError(
                "No player data available. Please load real NFL data first using:\n"
                f"  python3 src/data/nfl_data_loader.py --seasons {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON}\n"
                "(or omit --seasons for default). This system only uses real NFL data."
            )
        
        # Filter by player name if specified
        if player_name:
            player_data = player_data[
                player_data["name"].str.lower().str.contains(player_name.lower())
            ]
            if player_data.empty:
                print(f"No player found matching '{player_name}'")
                return pd.DataFrame()
        
        # Prepare features
        player_data = self._prepare_features(player_data)
        
        # Get most recent data per player and game count for cold-start detection
        games_per_player = player_data.groupby("player_id").size().reset_index(name="games_count")
        latest_data = player_data.groupby("player_id").last().reset_index()
        latest_data = latest_data.merge(games_per_player, on="player_id")
        
        # Overwrite season/week/opponent/home_away for the upcoming game so matchup features are correct
        schedule_map = get_schedule_map_for_week(self.db, pred_season, pred_week)
        latest_data["season"] = pred_season
        latest_data["week"] = pred_week
        if schedule_map:
            def _opp(row):
                t = row.get("team")
                if pd.isna(t):
                    return ""
                t = str(t).strip()
                return schedule_map.get(t, ("", "unknown"))[0]
            def _ha(row):
                t = row.get("team")
                if pd.isna(t):
                    return "unknown"
                t = str(t).strip()
                return schedule_map.get(t, ("", "unknown"))[1]
            latest_data["opponent"] = latest_data.apply(_opp, axis=1)
            latest_data["home_away"] = latest_data.apply(_ha, axis=1)
        else:
            latest_data["opponent"] = ""
            latest_data["home_away"] = "unknown"
        
        # Refresh schedule- and opponent-dependent features for the prediction row
        latest_data = self.feature_engineer.refresh_matchup_features(latest_data)
        
        # Make predictions (with speed monitoring per requirements: <5s per player)
        t_start = time.perf_counter()
        results = self.predictor.predict(latest_data, n_weeks=n_weeks)
        t_elapsed = time.perf_counter() - t_start
        n_players = len(latest_data)
        if n_players > 0:
            time_per_player = t_elapsed / n_players
            if time_per_player > MAX_PREDICTION_TIME_PER_PLAYER_SECONDS:
                import warnings
                warnings.warn(
                    f"Prediction speed {time_per_player:.2f}s/player exceeds "
                    f"{MAX_PREDICTION_TIME_PER_PLAYER_SECONDS}s target "
                    f"({n_players} players in {t_elapsed:.1f}s)",
                    UserWarning,
                    stacklevel=2,
                )
        
        # Cold-start: use position-average projection for rookies (high uncertainty)
        results = self._apply_cold_start_fallback(
            results, latest_data, n_weeks, MIN_GAMES_FOR_PREDICTION
        )
        
        # Add useful columns (primary = utilization)
        results["predicted_ppg"] = results["predicted_points"] / n_weeks
        results["n_weeks"] = n_weeks
        if "predicted_utilization" not in results.columns:
            results["predicted_utilization"] = results["predicted_points"]
        
        # Get utilization tier from predicted utilization
        if "predicted_utilization" in results.columns:
            results["util_tier"] = results.apply(
                lambda row: self.utilization_calculator.get_utilization_tier(
                    row.get("predicted_utilization", row.get("predicted_points", 50)),
                    row.get("position", "RB")
                ), axis=1
            )
        elif "utilization_score" in results.columns:
            results["util_tier"] = results["utilization_score"].apply(
                lambda x: self.utilization_calculator.get_utilization_tier(x, "RB")
            )
        
        # Sort and rank by predicted utilization (primary) then predicted_points
        sort_col = "predicted_utilization" if "predicted_utilization" in results.columns else "predicted_points"
        results = results.sort_values(sort_col, ascending=False)
        results["overall_rank"] = range(1, len(results) + 1)
        
        # Add position rank
        for pos in POSITIONS:
            mask = results["position"] == pos
            results.loc[mask, "position_rank"] = range(1, mask.sum() + 1)
        
        # Widen CIs for rookies / volatile players (requirement: wider for volatile, rookies, injury-prone)
        if "games_count" in latest_data.columns and "prediction_std" in results.columns:
            gc_merged = latest_data[["player_id", "games_count"]].drop_duplicates("player_id")
            results = results.merge(gc_merged, on="player_id", how="left", suffixes=("", "_ci"))
            gc_col = "games_count_ci" if "games_count_ci" in results.columns else "games_count"
            rookie_mask = results[gc_col].fillna(0) < MIN_GAMES_FOR_PREDICTION
            if rookie_mask.any() and "prediction_std" in results.columns:
                results.loc[rookie_mask, "prediction_std"] = results.loc[rookie_mask, "prediction_std"].fillna(5.0) * 1.5
                z80, z95 = 1.28, 1.96
                pts = results.loc[rookie_mask, "predicted_points"]
                std = results.loc[rookie_mask, "prediction_std"]
                results.loc[rookie_mask, "prediction_ci80_lower"] = (pts - z80 * std).clip(lower=0)
                results.loc[rookie_mask, "prediction_ci80_upper"] = pts + z80 * std
                results.loc[rookie_mask, "prediction_ci95_lower"] = (pts - z95 * std).clip(lower=0)
                results.loc[rookie_mask, "prediction_ci95_upper"] = pts + z95 * std
            results = results.drop(columns=[c for c in [gc_col] if c in results.columns and c != "games_count"], errors="ignore")

        # Ensure CI lower bounds are non-negative
        for ci_col in ["prediction_ci80_lower", "prediction_ci95_lower"]:
            if ci_col in results.columns:
                results[ci_col] = results[ci_col].clip(lower=0)

        # Select output columns (primary = predicted utilization; include matchup for app)
        output_cols = [
            "overall_rank", "position_rank", "name", "position", "team",
            "predicted_utilization", "predicted_points", "predicted_ppg", "n_weeks",
            "opponent", "home_away",
            "prediction_ci80_lower", "prediction_ci80_upper",
            "prediction_ci95_lower", "prediction_ci95_upper",
        ]
        output_cols = [c for c in output_cols if c in results.columns]
        if "utilization_score" in results.columns and "utilization_score" not in output_cols:
            output_cols.append("utilization_score")
        if "util_tier" in results.columns:
            output_cols.append("util_tier")
        if "player_id" not in output_cols and "player_id" in results.columns:
            output_cols.insert(0, "player_id")
        
        # Adjust for scoring format (Half-PPR / Standard) if not default PPR
        results = self._adjust_scoring_format(results, scoring_format, n_weeks)
        if "scoring_format" in results.columns and "scoring_format" not in output_cols:
            output_cols.append("scoring_format")

        available_cols = [c for c in output_cols if c in results.columns]
        results = results[available_cols]
        
        # Return top N
        return results.head(top_n)
    
    def predict_player(self, player_name: str, n_weeks: int = 1) -> dict:
        """
        Get detailed prediction for a specific player.
        
        Args:
            player_name: Player name to search for
            n_weeks: Weeks to predict
            
        Returns:
            Dict with prediction details
        """
        results = self.predict(n_weeks=n_weeks, player_name=player_name, top_n=10)
        
        if results.empty:
            return {"error": f"Player '{player_name}' not found"}
        
        player = results.iloc[0]
        
        # Get expected PPG range based on utilization
        util_score = player.get("utilization_score", 50)
        position = player.get("position", "RB")
        ppg_range = self.utilization_calculator.get_expected_ppg_range(util_score, position)
        
        pred_util = player.get("predicted_utilization", player.get("predicted_points", util_score))
        return {
            "name": player["name"],
            "position": player["position"],
            "team": player.get("team", "N/A"),
            "n_weeks": n_weeks,
            "predicted_utilization": round(pred_util, 1),
            "predicted_total_points": round(player["predicted_points"], 1),
            "predicted_ppg": round(player["predicted_ppg"], 1),
            "utilization_score": round(util_score, 1),
            "utilization_tier": player.get("util_tier", "N/A"),
            "expected_ppg_range": ppg_range,
            "overall_rank": int(player["overall_rank"]),
            "position_rank": int(player.get("position_rank", 0)),
        }
    
    def compare_players(self, player_names: List[str], 
                        n_weeks: int = 1) -> pd.DataFrame:
        """
        Compare multiple players side by side.
        
        Args:
            player_names: List of player names
            n_weeks: Weeks to predict
            
        Returns:
            DataFrame comparing players
        """
        comparisons = []
        
        for name in player_names:
            result = self.predict_player(name, n_weeks)
            if "error" not in result:
                comparisons.append(result)
        
        if not comparisons:
            return pd.DataFrame()
        
        return pd.DataFrame(comparisons)
    
    def get_rankings(self, n_weeks: int = 1, 
                     position: str = None) -> pd.DataFrame:
        """
        Get player rankings for specified timeframe.
        
        Args:
            n_weeks: Prediction horizon
            position: Optional position filter
            
        Returns:
            DataFrame with rankings
        """
        return self.predict(n_weeks=n_weeks, position=position, top_n=100)
    
    def _load_player_data(self, position: str = None, min_games: int = 1) -> pd.DataFrame:
        """Load player data from database."""
        try:
            return self.db.get_all_players_for_training(position=position, min_games=min_games)
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _apply_cold_start_fallback(self, results: pd.DataFrame, latest_data: pd.DataFrame,
                                    n_weeks: int, min_games: int) -> pd.DataFrame:
        """Replace ML predictions with position-average for rookies (high uncertainty)."""
        if "games_count" not in latest_data.columns:
            return results
        # Merge may produce games_count_y if results already had games_count
        results = results.merge(
            latest_data[["player_id", "games_count"]], on="player_id", how="left", suffixes=("", "_from_latest")
        )
        gc_col = "games_count_from_latest" if "games_count_from_latest" in results.columns else "games_count"
        cold_start_mask = results[gc_col] < min_games
        if not cold_start_mask.any():
            results = results.drop(columns=[c for c in ["games_count", "games_count_from_latest"] if c in results.columns], errors="ignore")
            return results
        
        # Position-average from players with sufficient history
        sufficient = results[~cold_start_mask]
        if sufficient.empty:
            results = results.drop(columns=[c for c in ["games_count", "games_count_from_latest"] if c in results.columns], errors="ignore")
            return results
        
        pred_col = "predicted_utilization" if "predicted_utilization" in sufficient.columns else "predicted_points"
        pos_avg = sufficient.groupby("position")[pred_col].mean().to_dict()
        pos_std = sufficient.groupby("position")[pred_col].std().to_dict()
        
        for idx in results[cold_start_mask].index:
            pos = results.loc[idx, "position"]
            avg = pos_avg.get(pos, 50.0)
            std = pos_std.get(pos, 10.0)
            results.loc[idx, "predicted_points"] = avg
            if "predicted_utilization" in results.columns:
                results.loc[idx, "predicted_utilization"] = avg
            if "prediction_std" in results.columns:
                results.loc[idx, "prediction_std"] = max(std * 2, 3.0 * n_weeks) if pd.notna(std) else 5.0 * n_weeks
        
        results = results.drop(columns=[c for c in ["games_count", "games_count_from_latest"] if c in results.columns], errors="ignore")
        return results
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        # Calculate utilization scores
        data = self.utilization_calculator.calculate_all_scores(data, pd.DataFrame())
        
        # Engineer features
        data = self.feature_engineer.create_features(data, include_target=False)
        
        return data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Player Fantasy Performance Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict next week's top players
  python -m src.predict --weeks 1

  # Predict full season for RBs
  python -m src.predict --weeks 18 --position RB

  # Predict specific player
  python -m src.predict --player "Patrick Mahomes" --weeks 4

  # Compare players
  python -m src.predict --compare "Josh Allen" "Jalen Hurts" "Lamar Jackson"
        """
    )
    
    parser.add_argument(
        "--weeks", "-w",
        type=int,
        default=1,
        help="Number of weeks to predict (1-18, default: 1)"
    )
    
    parser.add_argument(
        "--position", "-p",
        type=str,
        choices=POSITIONS,
        default=None,
        help="Filter by position (QB, RB, WR, TE)"
    )
    
    parser.add_argument(
        "--player",
        type=str,
        default=None,
        help="Predict for specific player by name"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare multiple players"
    )
    
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=25,
        help="Number of top players to show (default: 25)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (CSV)"
    )
    
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["ppr", "half_ppr", "standard"],
        default="ppr",
        help="Scoring format: ppr (default), half_ppr, or standard"
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    print("=" * 60)
    print("NFL Fantasy Performance Predictor")
    print("=" * 60)
    
    # Handle different prediction modes
    if args.compare:
        print(f"\nComparing players for {args.weeks} week(s)...")
        results = predictor.compare_players(args.compare, args.weeks)
        
    elif args.player:
        print(f"\nPredicting {args.player} for {args.weeks} week(s)...")
        result = predictor.predict_player(args.player, args.weeks)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\n{result['name']} ({result['position']}) - {result['team']}")
        print("-" * 40)
        print(f"Prediction Period: {result['n_weeks']} week(s)")
        print(f"Projected Total Points: {result['predicted_total_points']}")
        print(f"Projected PPG: {result['predicted_ppg']}")
        print(f"Utilization Score: {result['utilization_score']}")
        print(f"Utilization Tier: {result['utilization_tier']}")
        print(f"Expected PPG Range: {result['expected_ppg_range']['min']:.1f} - {result['expected_ppg_range']['max']:.1f}")
        print(f"Overall Rank: #{result['overall_rank']}")
        print(f"Position Rank: #{result['position_rank']}")
        return
        
    else:
        position_str = f" {args.position}" if args.position else ""
        print(f"\nPredicting top{position_str} players for {args.weeks} week(s)...")
        results = predictor.predict(
            n_weeks=args.weeks,
            position=args.position,
            top_n=args.top,
            scoring_format=args.scoring
        )
    
    # Display results
    if results.empty:
        print("No predictions available. Please ensure models are trained.")
        return
    
    print(f"\nTop {len(results)} Players - {args.weeks} Week Projection")
    print("-" * 80)
    
    # Format for display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results.to_string(index=False))
    
    # Save to file if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

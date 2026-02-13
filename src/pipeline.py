"""Main pipeline orchestration for NFL prediction workflow."""
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import POSITIONS, SEASONS_TO_SCRAPE
from src.scrapers.run_scrapers import run_all_scrapers
from src.models.train import train_models
from src.predict import NFLPredictor


class NFLPredictionPipeline:
    """
    End-to-end pipeline for NFL player performance prediction.
    
    Stages:
    1. Data Collection: Scrape latest player and team stats
    2. Data Processing: Calculate utilization scores, engineer features
    3. Model Training: Train position-specific models with tuning
    4. Prediction: Generate predictions for specified timeframe
    5. Evaluation: Assess model performance
    """
    
    def __init__(self):
        self.predictor = None
    
    def run_full_pipeline(self, 
                          seasons: List[int] = None,
                          positions: List[str] = None,
                          tune_hyperparameters: bool = True,
                          prediction_weeks: int = 1):
        """
        Run the complete pipeline from data collection to prediction.
        
        Args:
            seasons: Seasons to scrape (default: 2020-2024)
            positions: Positions to train (default: all)
            tune_hyperparameters: Whether to tune model hyperparameters
            prediction_weeks: Weeks to predict after training
        """
        seasons = seasons or SEASONS_TO_SCRAPE
        positions = positions or POSITIONS
        
        print("=" * 70)
        print("NFL PLAYER PERFORMANCE PREDICTION PIPELINE")
        print("=" * 70)
        
        # Stage 1: Data Collection
        print("\n" + "=" * 70)
        print("STAGE 1: DATA COLLECTION")
        print("=" * 70)
        self._run_data_collection(seasons)
        
        # Stage 2: Model Training
        print("\n" + "=" * 70)
        print("STAGE 2: MODEL TRAINING")
        print("=" * 70)
        self._run_training(positions, tune_hyperparameters)
        
        # Stage 3: Generate Predictions
        print("\n" + "=" * 70)
        print("STAGE 3: PREDICTIONS")
        print("=" * 70)
        self._run_predictions(prediction_weeks, positions)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
    
    def run_refresh_and_predict(self, prediction_weeks: int = 1):
        """
        Quick pipeline: refresh data and make predictions with existing models.
        
        Args:
            prediction_weeks: Weeks to predict
        """
        print("=" * 70)
        print("REFRESH AND PREDICT")
        print("=" * 70)
        
        # Refresh latest data
        print("\nRefreshing latest data...")
        run_all_scrapers(refresh_only=True)
        
        # Make predictions
        print("\nGenerating predictions...")
        self._run_predictions(prediction_weeks)
    
    def _run_data_collection(self, seasons: List[int]):
        """Run data collection stage."""
        print(f"Scraping data for seasons: {seasons}")
        run_all_scrapers(seasons=seasons, refresh_only=False)
    
    def _run_training(self, positions: List[str], tune_hyperparameters: bool):
        """Run model training stage."""
        print(f"Training models for positions: {positions}")
        print(f"Hyperparameter tuning: {'Enabled' if tune_hyperparameters else 'Disabled'}")
        train_models(
            positions=positions,
            tune_hyperparameters=tune_hyperparameters
        )
    
    def _run_predictions(self, n_weeks: int, positions: List[str] = None):
        """Run prediction stage."""
        positions = positions or POSITIONS
        
        self.predictor = NFLPredictor()
        self.predictor.initialize()
        
        print(f"\nGenerating {n_weeks}-week predictions...")
        
        # Overall rankings
        print("\n--- OVERALL TOP 25 ---")
        overall = self.predictor.predict(n_weeks=n_weeks, top_n=25)
        if not overall.empty:
            print(overall.to_string(index=False))
        
        # Position-specific rankings
        for position in positions:
            print(f"\n--- TOP 10 {position}s ---")
            pos_rankings = self.predictor.predict(
                n_weeks=n_weeks, 
                position=position, 
                top_n=10
            )
            if not pos_rankings.empty:
                print(pos_rankings.to_string(index=False))


def main():
    """Main CLI for pipeline."""
    parser = argparse.ArgumentParser(
        description="NFL Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  full      Run complete pipeline (scrape, train, predict)
  refresh   Refresh data and predict with existing models
  scrape    Only run data scrapers
  train     Only train models
  predict   Only make predictions

Examples:
  # Run full pipeline
  python -m src.pipeline full --seasons 2022-2024

  # Quick refresh and predict
  python -m src.pipeline refresh --weeks 1

  # Train models only
  python -m src.pipeline train --positions QB RB --no-tune
        """
    )
    
    parser.add_argument(
        "command",
        choices=["full", "refresh", "scrape", "train", "predict"],
        help="Pipeline command to run"
    )
    
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Seasons to process (e.g., '2022-2024' or '2023,2024')"
    )
    
    parser.add_argument(
        "--positions",
        nargs="+",
        default=None,
        help="Positions to process (QB RB WR TE)"
    )
    
    parser.add_argument(
        "--weeks", "-w",
        type=int,
        default=1,
        help="Weeks to predict (default: 1)"
    )
    
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning"
    )
    
    args = parser.parse_args()
    
    # Parse seasons
    seasons = None
    if args.seasons:
        if "-" in args.seasons:
            start, end = args.seasons.split("-")
            seasons = list(range(int(start), int(end) + 1))
        else:
            seasons = [int(s.strip()) for s in args.seasons.split(",")]
    
    pipeline = NFLPredictionPipeline()
    
    if args.command == "full":
        pipeline.run_full_pipeline(
            seasons=seasons,
            positions=args.positions,
            tune_hyperparameters=not args.no_tune,
            prediction_weeks=args.weeks
        )
    
    elif args.command == "refresh":
        pipeline.run_refresh_and_predict(prediction_weeks=args.weeks)
    
    elif args.command == "scrape":
        run_all_scrapers(seasons=seasons, refresh_only=False)
    
    elif args.command == "train":
        train_models(
            positions=args.positions,
            tune_hyperparameters=not args.no_tune
        )
    
    elif args.command == "predict":
        predictor = NFLPredictor()
        predictor.initialize()
        
        if args.positions:
            for pos in args.positions:
                print(f"\n--- {pos} Rankings ({args.weeks} weeks) ---")
                results = predictor.predict(n_weeks=args.weeks, position=pos, top_n=25)
                print(results.to_string(index=False))
        else:
            print(f"\n--- Overall Rankings ({args.weeks} weeks) ---")
            results = predictor.predict(n_weeks=args.weeks, top_n=50)
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()

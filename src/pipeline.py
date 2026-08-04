"""Main pipeline orchestration for NFL prediction workflow.

Includes a lightweight DAG-based pipeline orchestrator per Directive V7
Section 19 (Data Engineering and Pipeline Resilience).
"""
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import POSITIONS, SEASONS_TO_SCRAPE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight DAG Pipeline Orchestrator (Directive V7 Section 19)
# ---------------------------------------------------------------------------

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage:
    """A single stage in the pipeline DAG."""

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        depends_on: Optional[List[str]] = None,
        cache_ttl_seconds: Optional[float] = None,
    ):
        self.name = name
        self.func = func
        self.depends_on: List[str] = depends_on or []
        self.cache_ttl_seconds = cache_ttl_seconds
        self.status = StageStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.duration_seconds: float = 0.0

    def __repr__(self) -> str:
        return f"PipelineStage({self.name!r}, status={self.status})"


class PipelineDAG:
    """Lightweight DAG-based pipeline with dependency resolution and checkpointing.

    Features per Directive V7 Section 19:
    - Explicit dependency declarations between stages
    - Idempotent execution via checkpoint caching
    - Deterministic execution order (topological sort)
    - Stage-level retry on failure
    - Status tracking and logging
    """

    def __init__(
        self,
        name: str = "pipeline",
        cache_dir: Optional[Path] = None,
        max_retries: int = 1,
    ):
        self.name = name
        self.stages: Dict[str, PipelineStage] = {}
        self.cache_dir = cache_dir or Path("data/pipeline_cache")
        self.max_retries = max_retries
        self._execution_log: List[Dict[str, Any]] = []

    def add_stage(
        self,
        name: str,
        func: Callable[..., Any],
        depends_on: Optional[List[str]] = None,
        cache_ttl_seconds: Optional[float] = None,
    ) -> "PipelineDAG":
        """Register a pipeline stage."""
        stage = PipelineStage(name, func, depends_on, cache_ttl_seconds)
        self.stages[name] = stage
        return self

    def _topological_sort(self) -> List[str]:
        """Return stages in dependency-respecting order."""
        visited: Set[str] = set()
        order: List[str] = []

        def _visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            stage = self.stages[name]
            for dep in stage.depends_on:
                if dep not in self.stages:
                    raise ValueError(
                        f"Stage {name!r} depends on unknown stage {dep!r}"
                    )
                _visit(dep)
            order.append(name)

        for name in self.stages:
            _visit(name)
        return order

    def _check_cache(self, stage: PipelineStage, config_hash: str) -> bool:
        """Check if a stage has a fresh cached result."""
        if stage.cache_ttl_seconds is None:
            return False
        cache_file = self.cache_dir / f"{stage.name}_{config_hash}.json"
        if not cache_file.exists():
            return False
        try:
            meta = json.loads(cache_file.read_text())
            cached_at = meta.get("completed_at", 0)
            age = time.time() - cached_at
            return age < stage.cache_ttl_seconds
        except (json.JSONDecodeError, KeyError):
            return False

    def _write_cache(self, stage: PipelineStage, config_hash: str) -> None:
        """Write cache marker for a completed stage."""
        if stage.cache_ttl_seconds is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{stage.name}_{config_hash}.json"
        cache_file.write_text(json.dumps({
            "stage": stage.name,
            "completed_at": time.time(),
            "duration_seconds": stage.duration_seconds,
        }))

    def run(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Execute the pipeline in topological order.

        Args:
            config: Optional config dict used for cache key computation.
            **kwargs: Passed to each stage function.

        Returns:
            Dict with stage statuses and execution summary.
        """
        config = config or {}
        config_hash = hashlib.md5(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        order = self._topological_sort()
        logger.info("Pipeline %s: executing %d stages: %s", self.name, len(order), order)

        for stage_name in order:
            stage = self.stages[stage_name]

            # Check dependencies
            failed_deps = [
                d for d in stage.depends_on
                if self.stages[d].status == StageStatus.FAILED
            ]
            if failed_deps:
                stage.status = StageStatus.SKIPPED
                stage.error = f"Skipped: dependencies failed: {failed_deps}"
                logger.warning("Stage %s skipped (deps failed: %s)", stage_name, failed_deps)
                continue

            # Check cache
            if self._check_cache(stage, config_hash):
                stage.status = StageStatus.SKIPPED
                logger.info("Stage %s: cached result still fresh, skipping", stage_name)
                continue

            # Execute with retry
            stage.status = StageStatus.RUNNING
            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    start = time.monotonic()
                    stage.result = stage.func(**kwargs)
                    stage.duration_seconds = time.monotonic() - start
                    stage.status = StageStatus.COMPLETED
                    self._write_cache(stage, config_hash)
                    logger.info(
                        "Stage %s completed in %.1fs",
                        stage_name, stage.duration_seconds,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        logger.warning(
                            "Stage %s attempt %d failed: %s, retrying",
                            stage_name, attempt + 1, exc,
                        )
                    else:
                        stage.status = StageStatus.FAILED
                        stage.error = str(exc)
                        logger.error("Stage %s failed after %d attempts: %s",
                                     stage_name, self.max_retries + 1, exc)

            self._execution_log.append({
                "stage": stage_name,
                "status": stage.status.value,
                "duration_seconds": round(stage.duration_seconds, 2),
                "error": stage.error,
                "timestamp": datetime.now().isoformat(),
            })

        summary = {
            "pipeline": self.name,
            "stages": {s.name: s.status.value for s in self.stages.values()},
            "completed": sum(1 for s in self.stages.values() if s.status == StageStatus.COMPLETED),
            "failed": sum(1 for s in self.stages.values() if s.status == StageStatus.FAILED),
            "skipped": sum(1 for s in self.stages.values() if s.status == StageStatus.SKIPPED),
            "total_duration_seconds": round(
                sum(s.duration_seconds for s in self.stages.values()), 2
            ),
            "execution_log": self._execution_log,
        }
        logger.info("Pipeline %s complete: %s", self.name, summary["stages"])
        return summary
from src.scrapers.run_scrapers import run_all_scrapers
from src.models.train import train_models
from src.predict import NFLPredictor
from src.evaluation.backtester import ModelBacktester


class NFLPredictionPipeline:
    """
    End-to-end pipeline for NFL player performance prediction.
    
    Stages:
    1. Data Collection: Load latest player and team stats
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
            seasons: Seasons to load (default: 2020-current season)
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
        print(f"Loading data for seasons: {seasons}")
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

    def run_evaluation(self, positions: List[str] = None, n_seasons: int = 3):
        """
        Run backtesting and evaluation on the trained models.
        
        Generates a comprehensive metrics report including:
        - RMSE, MAE, R2, MAPE per position
        - Fantasy-specific metrics (Spearman, tier accuracy, boom/bust, VOR)
        - Position benchmark checks
        - Naive baseline comparison
        - Success criteria verification
        
        Args:
            positions: Positions to evaluate (default: all).
            n_seasons: Number of most recent seasons to backtest.
        """
        positions = positions or POSITIONS
        
        print("=" * 70)
        print("MODEL EVALUATION & BACKTESTING")
        print("=" * 70)
        
        try:
            backtester = ModelBacktester()
            results = backtester.run_multi_season_backtest(n_seasons=n_seasons)
            
            if results:
                print("\n--- Backtest Results ---")
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"\n  {key}:")
                        for k2, v2 in value.items():
                            print(f"    {k2}: {v2}")
                    else:
                        print(f"  {key}: {value}")
                
                # Check success criteria
                if hasattr(backtester, 'check_success_criteria'):
                    print("\n--- Success Criteria ---")
                    criteria = backtester.check_success_criteria(results)
                    for criterion, passed in criteria.items():
                        status = "PASS" if passed else "FAIL"
                        print(f"  [{status}] {criterion}")
            else:
                print("  No backtest results generated (models may not be trained yet)")
                
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main CLI for pipeline."""
    parser = argparse.ArgumentParser(
        description="NFL Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  full      Run complete pipeline (load, train, predict)
  refresh   Refresh data and predict with existing models
  load      Only run data loader
  train     Only train models
  predict   Only make predictions
  evaluate  Run backtesting and evaluation on trained models

Examples:
  # Run full pipeline
  python -m src.pipeline full --seasons 2022-2024

  # Quick refresh and predict
  python -m src.pipeline refresh --weeks 1

  # Train models only
  python -m src.pipeline train --positions QB RB --no-tune

  # Evaluate models with 3-season backtest
  python -m src.pipeline evaluate --backtest-seasons 3
        """
    )
    
    parser.add_argument(
        "command",
        choices=["full", "refresh", "load", "train", "predict", "evaluate"],
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
    
    parser.add_argument(
        "--backtest-seasons",
        type=int,
        default=3,
        help="Number of seasons to backtest (default: 3)"
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
    
    elif args.command == "load":
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
    
    elif args.command == "evaluate":
        pipeline.run_evaluation(
            positions=args.positions,
            n_seasons=args.backtest_seasons
        )


if __name__ == "__main__":
    main()

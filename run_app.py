#!/usr/bin/env python3
"""
NFL Fantasy Predictor - Startup Script

Data and models are saved so future runs can be fast:
- Player/team data: stored in data/nfl_data.db (only refreshed with --refresh).
- Schedule data is refreshed from nfl-data-py on --refresh; new seasons (e.g. next year's
  schedule in spring) are loaded automatically when available.
- Trained models: stored in data/models/*.joblib (train once with python -m src.models.train).
- Prediction cache: data/cached_features.parquet and data/daily_predictions.parquet;
  with --with-predictions we skip regeneration if cache is newer than --max-prediction-cache-hours (default 24).

Usage:
    python run_app.py                        # Start app; use existing DB and cache (fast)
    python run_app.py --skip-data            # Fastest: skip data check, use existing parquet only
    python run_app.py --with-predictions     # Ensure predictions exist; use cache if < 24h old
    python run_app.py --force-predictions    # Always regenerate predictions
    python run_app.py --refresh --with-predictions  # Full refresh + regenerate predictions
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def _default_season_range():
    """Default season range from config (single source of truth)."""
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    return list(range(MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON + 1))


def parse_seasons(args):
    """Parse season arguments - supports list and range. Defaults from config."""
    if args.from_year is not None and args.to_year is not None:
        return list(range(args.from_year, args.to_year + 1))
    if args.seasons:
        return args.seasons
    return _default_season_range()


def main():
    parser = argparse.ArgumentParser(
        description='NFL Fantasy Predictor Startup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                              # Default: config range (e.g. 2020–current season)
  python run_app.py --seasons 2022 2023 2024    # Specific seasons
  python run_app.py --from-year 2020 --to-year <current>  # Range (use current NFL season)
  python run_app.py --refresh                    # Force data refresh (picks up new schedules from nfl-data-py)
        """
    )
    parser.add_argument('--seasons', nargs='+', type=int,
                        help='Seasons to load (e.g., --seasons 2022 2023 2024)')
    from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
    parser.add_argument('--from-year', type=int, dest='from_year', default=None,
                        help=f'Start year for range (default: {MIN_HISTORICAL_YEAR})')
    parser.add_argument('--to-year', type=int, dest='to_year', default=None,
                        help=f'End year for range (default: current NFL season, {CURRENT_NFL_SEASON})')
    parser.add_argument('--refresh', action='store_true',
                        help='Force refresh data from source')
    parser.add_argument('--port', type=int, default=8501,
                        help='Port for web app (default: 8501)')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data loading, just start app')
    parser.add_argument('--with-predictions', action='store_true',
                        help='Run ML predictions and merge into cached data before launch (skips if cache is fresh)')
    parser.add_argument('--force-predictions', action='store_true',
                        help='Always regenerate predictions (ignore cache age); implies --with-predictions')
    parser.add_argument('--max-prediction-cache-hours', type=float, default=24,
                        help='Use cached predictions if parquet is newer than this many hours (default: 24)')
    
    args = parser.parse_args()
    if args.force_predictions:
        args.with_predictions = True
    
    # Parse seasons from either format
    seasons = parse_seasons(args)
    
    print("=" * 60)
    print("🏈 NFL Fantasy Predictor")
    print("=" * 60)
    print(f"\n📅 Seasons: {seasons}")
    print(f"🔄 Refresh: {args.refresh}")
    print(f"🌐 Port: {args.port}")
    
    # Step 1: Load/refresh data if needed (prefer nfl-data-py; do not scrape PFR when data already exists)
    if not args.skip_data:
        print("\n📊 Checking data...")
        try:
            from src.utils.database import DatabaseManager
            db = DatabaseManager()
            existing_seasons = db.get_seasons_with_data()
            needs_seasons = [s for s in seasons if s not in existing_seasons]

            if not needs_seasons:
                # All requested seasons already have data in DB
                if args.refresh:
                    print("⏳ Refreshing from nfl-data-py (skipping PFR scrapers)...")
                    from src.utils.data_manager import auto_refresh_data
                    auto_refresh_data(force_check=True)
                    print("✅ Data refreshed from nfl-data-py.")
                else:
                    df = db.get_player_stats()
                    if not df.empty:
                        print(f"✅ Data already loaded: {len(df)} records")
                        print(f"   Seasons available: {list(sorted(existing_seasons))}")
                        print(f"   Positions: {df['position'].unique().tolist()}")
                    else:
                        print("✅ Data already in database.")
            else:
                # Missing one or more seasons: try nfl-data-py first, then PFR only if needed
                print("⏳ Loading data (this may take a moment)...")
                try:
                    from src.data.nfl_data_loader import NFLDataLoader
                    loader = NFLDataLoader()
                    loader.load_weekly_data(needs_seasons, store_in_db=True, use_pbp_fallback=True)
                except Exception as e:
                    print(f"   nfl-data-py load warning: {e}")
                existing_after = db.get_seasons_with_data()
                still_needs = [s for s in seasons if s not in existing_after]
                if still_needs:
                    print(f"   Loading missing seasons {still_needs} from PFR (fallback)...")
                    from src.scrapers.run_scrapers import run_all_scrapers
                    run_all_scrapers(seasons=still_needs, force_rescrape=False)
                else:
                    print("✅ Data loaded from nfl-data-py.")
                print("✅ Data loaded successfully!")
        except Exception as e:
            print(f"⚠️ Data check warning: {e}")
            print("   Continuing with existing data...")
    
    # Step 2: Generate ML predictions for app (optional); use cache if fresh
    if args.with_predictions:
        project_root = Path(__file__).parent
        data_dir = project_root / "data"
        cache_paths = [data_dir / "daily_predictions.parquet", data_dir / "cached_features.parquet"]
        use_cache = False
        if not args.force_predictions and args.max_prediction_cache_hours > 0:
            import time
            now = time.time()
            max_age_secs = args.max_prediction_cache_hours * 3600
            for p in cache_paths:
                if p.exists():
                    age = now - p.stat().st_mtime
                    if age <= max_age_secs:
                        use_cache = True
                        print(f"\n✅ Using cached predictions ({p.name} is {int(age / 3600)}h old; run with --force-predictions to regenerate)")
                        break
        if not use_cache:
            print("\n📈 Generating ML predictions for web app...")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "generate_app_data",
                    project_root / "scripts" / "generate_app_data.py"
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                mod.generate_app_data(save_daily=True)
            except Exception as e:
                print(f"⚠️ Prediction generation failed: {e}")
                print("   App will use fallback projections (fantasy_points, fp_rolling)")
    
    # Step 3: Build frontend if needed, then launch FastAPI app
    project_root = Path(__file__).parent
    frontend_dist = project_root / "frontend" / "dist"
    if not frontend_dist.exists():
        print("\n📦 Building frontend (first run or after changes)...")
        try:
            subprocess.run(
                ["npm", "run", "build"],
                cwd=project_root / "frontend",
                check=True,
                capture_output=False,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️ Frontend build failed: {e}")
            print("   API will still run; open http://localhost:<port> for API docs or build manually: cd frontend && npm run build")

    print(f"\n🚀 Starting web app on port {args.port}...")
    print(f"   URL: http://localhost:{args.port}")
    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "api.main:app",
            "--host", "0.0.0.0",
            "--port", str(args.port),
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped.")


if __name__ == "__main__":
    main()

"""Data manager for automatic season selection and data availability."""
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, SEASONS_TO_SCRAPE, TRAINING_START_YEAR_DEFAULT
from src.utils.database import DatabaseManager


class DataManager:
    """
    Manages data availability and automatic season selection.
    
    Automatically uses the latest available data for training/testing,
    and periodically checks for new season data availability.
    """
    
    CACHE_FILE = DATA_DIR / "data_availability_cache.json"
    CHECK_INTERVAL_HOURS = 6  # How often to re-check for new data
    
    def __init__(self):
        self.db = DatabaseManager()
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cached availability data."""
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_check": None, "available_seasons": [], "latest_season": None}
    
    def _save_cache(self):
        """Save availability cache atomically (write to temp, then rename)."""
        import tempfile, os
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp", dir=self.CACHE_FILE.parent
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._cache, f)
            os.replace(tmp_path, self.CACHE_FILE)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    
    def _should_recheck(self) -> bool:
        """Check if we should re-check data availability."""
        if not self._cache.get("last_check"):
            return True
        
        last_check = datetime.fromisoformat(self._cache["last_check"])
        return datetime.now() - last_check > timedelta(hours=self.CHECK_INTERVAL_HOURS)
    
    def check_data_availability(self, force: bool = False) -> Dict:
        """
        Check what seasons have data available.
        
        Args:
            force: Force re-check even if cache is fresh
            
        Returns:
            Dict with availability info
        """
        if not force and not self._should_recheck():
            return {
                "available_seasons": self._cache.get("available_seasons", []),
                "latest_season": self._cache.get("latest_season"),
                "from_cache": True
            }
        
        print("Checking data availability...")
        
        from src.scrapers.schedule_scraper import (
            check_schedule_availability, 
            get_available_seasons,
            get_latest_available_season
        )
        
        available = get_available_seasons()
        latest = get_latest_available_season()
        
        # Update cache
        self._cache["last_check"] = datetime.now().isoformat()
        self._cache["available_seasons"] = available
        self._cache["latest_season"] = latest
        self._save_cache()
        
        print(f"  Available seasons: {available}")
        print(f"  Latest season: {latest}")
        
        return {
            "available_seasons": available,
            "latest_season": latest,
            "from_cache": False
        }
    
    def get_available_seasons_from_db(self) -> List[int]:
        """Get available seasons from player_weekly_stats (source of truth for train/test)."""
        try:
            return self.db.get_seasons_with_data()
        except Exception:
            return []
    
    @staticmethod
    def _season_has_completed_games(season: int) -> bool:
        """Thin delegate; see nfl_calendar.season_has_completed_games."""
        from src.utils.nfl_calendar import season_has_completed_games
        return season_has_completed_games(season)

    def get_train_test_seasons(self,
                                test_season: int = None,
                                n_train_seasons: int = None,
                                optimal_years_per_position: Dict[str, int] = None) -> Tuple[List[int], int]:
        """
        Get training and test seasons automatically.

        Normal season (Sept–May): test_season = latest available (e.g. 2025 in-season).
        Draft prep window (June–Aug): test_season = upcoming calendar year (e.g. 2026),
          which has no game data yet.  Training then includes ALL available historical
          seasons (e.g. 2018–2025), with 2025 serving as the internal validation fold.
        In-season (week ≥ 1 played): test_season = current season (forced).

        Args:
            test_season: Override test season (None = auto-select; forced when in-season)
            n_train_seasons: Max seasons for training (None = use all available)
            optimal_years_per_position: Optional per-position optimal years (uses min for union)

        Returns:
            Tuple of (train_seasons, test_season)
        """
        from src.utils.nfl_calendar import (
            get_current_nfl_season,
            current_season_has_weeks_played,
            is_draft_prep_window,
            get_projection_season,
        )

        # Prefer DB seasons so train/test split matches actual data (no test_season with 0 rows)
        available = self.get_available_seasons_from_db()
        if not available:
            availability = self.check_data_availability()
            available = availability["available_seasons"]

        if not available:
            from config.settings import MIN_HISTORICAL_YEAR, CURRENT_NFL_SEASON
            raise ValueError(f"No season data available. Load data with: python -m src.data.nfl_data_loader (default: {MIN_HISTORICAL_YEAR}-{CURRENT_NFL_SEASON})")

        available = sorted(available)
        current_season = get_current_nfl_season()

        # Draft prep window (June–Aug) must be checked BEFORE in-season logic.
        # current_season_has_weeks_played() returns True in summer because the most
        # recently completed season (e.g. 2025) had weeks played — but those ended in
        # February.  During draft prep the target is the UPCOMING season (e.g. 2026),
        # and training includes all completed history (e.g. 2018–2025).
        # Extended beyond the calendar draft-prep window to cover the gap
        # between that window closing and the season's first kickoff. On
        # 2026-09-03 the window had closed (September) but 2026 had 0 completed
        # games, so this fell through to in-season handling and picked the last
        # COMPLETED season as the test set -- training on 2018-2024 and
        # silently discarding 2025, a full season of data, from production
        # models. The season a model should project is the one that has not
        # been played yet, whatever the calendar calls today.
        _projection_season = get_projection_season()
        _pre_kickoff = not self._season_has_completed_games(_projection_season)
        if (is_draft_prep_window() or _pre_kickoff) and test_season is None:
            test_season = _projection_season
            # test_season (e.g. 2026) is intentionally not in available — empty test set.
        else:
            # `current_season_has_weeks_played()` is a CALENDAR check --
            # week_num >= 1 -- and says nothing about whether a game has
            # actually been played. Between the nominal start of week 1 and
            # kickoff it returns True while no result exists anywhere, so the
            # guard below demanded database rows that cannot exist and blocked
            # all training. Hit for real on 2026-09-03: calendar week 1, but
            # 2026 had 272 scheduled games and 0 scores, and nfl_data_py
            # returned HTTP 404 for 2026 weekly data.
            #
            # Require an actual completed game before treating the season as in
            # progress.
            in_season = (current_season_has_weeks_played()
                         and self._season_has_completed_games(current_season))

            # In-season: current season must be in DB and must be test season
            if in_season and current_season not in available:
                raise ValueError(
                    f"Current season {current_season} has completed games but is not in the database. "
                    "Run data refresh so current season is loaded from play-by-play (e.g. python -m src.data.auto_refresh)."
                )
            if in_season and current_season in available and test_season is None:
                # Default to current season when in-season and caller didn't specify.
                # Don't override explicit test_season (LOYO backtest sets its own).
                test_season = current_season
            else:
                if test_season is None:
                    test_season = max(available)
                # Only fall back to latest available when the requested season isn't in the DB
                # AND we're not in the draft prep window (where the future season legitimately
                # has no data yet and an empty test set is expected and correct).
                if test_season not in available and not is_draft_prep_window():
                    test_season = max(available)

        # Get training seasons: all seasons before test, enforcing hard 2018+ cutoff.
        # Pre-2018 data lacks NGS, snap counts, and modern play-calling norms — it's
        # noise for current projections (TRAINING_START_YEAR_DEFAULT = 2018 is now hard).
        # During draft prep this includes the most recently completed season (e.g. 2025).
        train_seasons = [s for s in available if s < test_season and s >= TRAINING_START_YEAR_DEFAULT]

        # Optionally limit training window
        if n_train_seasons is not None and n_train_seasons > 0:
            train_seasons = sorted(train_seasons)[-n_train_seasons:]
        
        if optimal_years_per_position:
            n = max(optimal_years_per_position.values()) if optimal_years_per_position else len(train_seasons)
            train_seasons = sorted(train_seasons)[-n:]
        
        train_seasons = sorted(train_seasons)
        
        print(f"Train seasons: {train_seasons} ({len(train_seasons)} years)")
        print(f"Test season: {test_season}")
        
        return train_seasons, test_season
    
    def get_prediction_season(self) -> int:
        """
        Get the season to use for predictions.
        
        Returns current NFL season if available, otherwise latest available season.
        Uses shared nfl_calendar for current season (no duplicated calendar logic).
        """
        from src.utils.nfl_calendar import get_current_nfl_season
        target_season = get_current_nfl_season()
        availability = self.check_data_availability()
        
        # Check if target season is available
        if target_season in availability["available_seasons"]:
            return target_season
        
        # Check next year (for schedule release in spring)
        if target_season + 1 in availability["available_seasons"]:
            return target_season + 1
        
        # Fall back to latest available
        return availability["latest_season"]
    
    def load_training_data(self, 
                           position: str = None,
                           min_games: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data with automatic season selection.
        
        Args:
            position: Filter by position (None = all)
            min_games: Minimum games for player inclusion
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_seasons, test_season = self.get_train_test_seasons()
        
        # Load all player data
        all_data = self.db.get_all_players_for_training(
            position=position,
            min_games=min_games
        )
        
        if all_data.empty:
            raise ValueError("No training data available")
        
        # Split by season
        train_df = all_data[all_data['season'].isin(train_seasons)]
        test_df = all_data[all_data['season'] == test_season]
        
        print(f"Training data: {len(train_df)} rows from seasons {train_seasons}")
        print(f"Test data: {len(test_df)} rows from season {test_season}")
        
        return train_df, test_df
    
    def ensure_schedule_loaded(self, season: int = None) -> bool:
        """
        Ensure schedule data is loaded for a season.
        
        Schedules: nfl-data-py first; scraper fallback when season not yet available.
        Attempts to import if not already in database.
        
        Args:
            season: Season to check (None = latest available)
            
        Returns:
            True if schedule is available
        """
        if season is None:
            season = self.get_prediction_season()
        
        # Check if already in database
        if self.db.has_schedule_for_season(season):
            return True
        
        # Try nfl-data-py first (same source as auto_refresh / training)
        try:
            from src.data.nfl_data_loader import NFLDataLoader
            loader = NFLDataLoader()
            df = loader.load_schedules([season], store_in_db=True)
            if df is not None and len(df) > 0:
                return True
        except Exception:
            pass
        
        # Fallback: scraper when nfl-data-py has no schedule (e.g. future season)
        try:
            from src.scrapers.schedule_scraper import import_schedule_to_db
            count = import_schedule_to_db(season)
            return count > 0
        except Exception:
            return False
    
    def get_status_report(self) -> str:
        """Get a status report of data availability."""
        availability = self.check_data_availability(force=True)
        
        report = []
        report.append("=" * 50)
        report.append("NFL Predictor Data Status")
        report.append("=" * 50)
        report.append(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Available seasons: {availability['available_seasons']}")
        report.append(f"Latest season: {availability['latest_season']}")
        
        # Check specific seasons
        current_year = datetime.now().year
        for year in [current_year + 1, current_year, current_year - 1]:
            from src.scrapers.schedule_scraper import check_schedule_availability
            status = "✓ Available" if check_schedule_availability(year) else "✗ Not available"
            report.append(f"  {year}: {status}")
        
        # Database stats
        report.append("\nDatabase Statistics:")
        try:
            player_count = len(self.db.get_player_stats())
            report.append(f"  Player weekly records: {player_count}")
        except Exception:
            report.append("  Player data: Unable to query")
        
        try:
            schedule = self.db.get_schedule()
            report.append(f"  Schedule records: {len(schedule)}")
            if not schedule.empty:
                seasons_in_db = schedule['season'].unique()
                report.append(f"  Seasons in schedule DB: {sorted(seasons_in_db)}")
        except Exception:
            report.append("  Schedule data: Unable to query")
        
        report.append("=" * 50)
        
        return "\n".join(report)


def auto_refresh_data(force_check: bool = False) -> Dict:
    """
    Convenience function to auto-refresh data and check availability.
    
    Loads any missing weekly data (e.g. current season) from nfl-data-py so
    the ML pipeline sees the latest season. Then uses DB seasons for train/test
    split so test set is never empty.
    
    Args:
        force_check: Force re-check even if recently checked
        
    Returns:
        Dict with current data status
    """
    # Load missing seasons/weeks from nfl-data-py (current season) so DB has latest
    try:
        from src.data.auto_refresh import NFLDataRefresher
        refresher = NFLDataRefresher()
        refresh_result = refresher.refresh(force=False)
        if isinstance(refresh_result, dict) and refresh_result.get("blocked"):
            import warnings
            warnings.warn(
                "Auto-refresh data quality gates FAILED — proceeding with existing DB data. "
                "Check the quality gate report for details.",
                UserWarning,
            )
    except Exception as e:
        # Non-fatal: proceed with whatever is in DB
        import warnings
        warnings.warn(f"Auto-refresh load skipped: {e}", UserWarning)
    
    manager = DataManager()
    
    # Check availability (schedule/cache for display; train/test uses DB)
    availability = manager.check_data_availability(force=force_check)
    
    # Ensure latest schedule is loaded
    latest = availability["latest_season"]
    if latest:
        manager.ensure_schedule_loaded(latest)
    
    # Train/test split uses DB seasons (so test set has data)
    train_seasons, test_season = manager.get_train_test_seasons()
    
    return {
        "available_seasons": availability["available_seasons"],
        "latest_season": latest,
        "prediction_season": manager.get_prediction_season(),
        "train_test_split": (train_seasons, test_season),
    }


if __name__ == "__main__":
    # Print status report
    manager = DataManager()
    print(manager.get_status_report())
    
    # Show auto-selected train/test split
    print("\nAuto-selected train/test split:")
    status = auto_refresh_data(force_check=True)
    print(f"  Train seasons: {status['train_test_split'][0]}")
    print(f"  Test season: {status['train_test_split'][1]}")
    print(f"  Prediction season: {status['prediction_season']}")

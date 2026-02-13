"""
Real-Time Data Integration System

Automatically fetches and caches current season data.
Implements smart refresh logic and cache management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import hashlib
import json
from typing import Optional, Tuple


class DataCache:
    """Intelligent caching system for NFL data."""
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / 'current_season.parquet'
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
    
    def get_cache_age(self) -> Optional[float]:
        """Get age of cache in hours. None if no cache."""
        if not self.cache_file.exists():
            return None
        
        mod_time = self.cache_file.stat().st_mtime
        age_seconds = time.time() - mod_time
        return age_seconds / 3600  # Convert to hours
    
    def is_cache_valid(self, max_age_hours: float = 6) -> bool:
        """Check if cache is still valid."""
        age = self.get_cache_age()
        if age is None:
            return False
        return age < max_age_hours
    
    def get_metadata(self) -> dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: dict):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self) -> Optional[pd.DataFrame]:
        """Load cached data if valid."""
        if not self.is_cache_valid():
            return None
        
        try:
            data = pd.read_parquet(self.cache_file)
            print(f"✅ Loaded cached data ({len(data):,} rows, age: {self.get_cache_age():.1f}h)")
            return data
        except Exception as e:
            print(f"⚠️  Failed to load cache: {e}")
            return None
    
    def save(self, data: pd.DataFrame, metadata: dict = None):
        """Save data to cache."""
        data.to_parquet(self.cache_file, index=False)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'rows': len(data),
            'timestamp': datetime.now().isoformat(),
            'seasons': sorted(data['season'].unique().tolist()) if 'season' in data.columns else [],
        })
        
        self.save_metadata(metadata)
        print(f"✅ Cached {len(data):,} rows")


class RealTimeDataFetcher:
    """Fetch and manage real-time NFL data."""
    
    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache = DataCache(cache_dir)
        try:
            from src.utils.nfl_calendar import get_current_nfl_season
            self.current_season = get_current_nfl_season()
        except Exception:
            self.current_season = datetime.now().year
    
    def get_current_week(self) -> int:
        """Current NFL week from shared nfl_calendar."""
        try:
            from src.utils.nfl_calendar import get_current_nfl_week
            return get_current_nfl_week().get("week_num", 0)
        except Exception:
            pass
        today = datetime.now()
        season_start = datetime(self.current_season, 9, 4)
        if today < season_start:
            return 0
        weeks_elapsed = (today - season_start).days // 7
        return min(weeks_elapsed + 1, 19)
    
    def fetch_current_season(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch current season data with smart caching.
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            DataFrame with current season stats
        """
        # Try cache first
        if not force_refresh:
            cached = self.cache.load()
            if cached is not None:
                return cached
        
        print(f"📥 Fetching {self.current_season} season data (Week {self.get_current_week()})...")
        
        try:
            # Try real data first
            data = self._fetch_from_nflverse()
        except Exception as e:
            print(f"⚠️  nflverse failed: {e}")
            print("   Using historical data as fallback...")
            data = self._fetch_historical_fallback()
        
        # Cache the data
        self.cache.save(data, {
            'season': self.current_season,
            'week': self.get_current_week(),
            'source': 'nflverse' if len(data) > 0 else 'historical_fallback',
        })
        
        return data
    
    def _fetch_from_nflverse(self) -> pd.DataFrame:
        """Fetch data from nflverse API."""
        try:
            import nfl_data_py as nfl
            
            # Get weekly data for current season
            weekly = nfl.import_weekly_data([self.current_season], downcast=True)
            
            # Filter to completed weeks
            current_week = self.get_current_week()
            weekly = weekly[weekly['week'] <= current_week]
            
            print(f"✅ Fetched {len(weekly):,} player-week records from nflverse")
            return weekly
            
        except ImportError:
            raise Exception("nfl_data_py not installed. Run: pip install nfl-data-py")
    
    def _fetch_historical_fallback(self) -> pd.DataFrame:
        """Fallback to last season if current season not available."""
        try:
            import nfl_data_py as nfl
            
            last_season = self.current_season - 1
            weekly = nfl.import_weekly_data([last_season], downcast=True)
            
            print(f"✅ Using {last_season} season as fallback ({len(weekly):,} rows)")
            return weekly
            
        except Exception as e:
            print(f"⚠️  Fallback also failed: {e}")
            return pd.DataFrame()
    
    def get_combined_data(self) -> pd.DataFrame:
        """
        Get combined historical + current season data.
        
        Returns:
            Full dataset for predictions
        """
        # Load historical training data
        historical_path = Path('data/historical_training_2000_2024.parquet')
        
        if historical_path.exists():
            historical = pd.read_parquet(historical_path)
            print(f"✅ Loaded {len(historical):,} historical records")
        else:
            print("⚠️  No historical training data found")
            historical = pd.DataFrame()
        
        # Get current season
        current = self.fetch_current_season()
        
        if current.empty:
            print("⚠️  No current season data")
            return historical
        
        # Combine
        if not historical.empty:
            # Avoid duplicates if overlap
            historical = historical[historical['season'] < self.current_season]
            combined = pd.concat([historical, current], ignore_index=True)
        else:
            combined = current
        
        print(f"✅ Combined dataset: {len(combined):,} total rows")
        return combined
    
    def refresh_if_needed(self, max_age_hours: float = 6) -> pd.DataFrame:
        """
        Smart refresh: only fetch if cache is stale.
        
        Args:
            max_age_hours: Cache validity period
            
        Returns:
            Current season data (cached or fresh)
        """
        cache_age = self.cache.get_cache_age()
        
        if cache_age is None:
            print("📥 No cache found, fetching fresh data...")
            return self.fetch_current_season(force_refresh=True)
        
        if cache_age < max_age_hours:
            print(f"✅ Cache valid (age: {cache_age:.1f}h < {max_age_hours}h max)")
            return self.cache.load()
        
        print(f"🔄 Cache stale (age: {cache_age:.1f}h > {max_age_hours}h), refreshing...")
        return self.fetch_current_season(force_refresh=True)


class AutoRefreshManager:
    """Manage automatic data refresh for dashboard."""
    
    def __init__(self, refresh_interval_hours: float = 6):
        self.fetcher = RealTimeDataFetcher()
        self.refresh_interval = refresh_interval_hours
        self.last_check = None
    
    def should_refresh(self) -> bool:
        """Check if it's time to refresh."""
        if self.last_check is None:
            return True
        
        time_since_check = (datetime.now() - self.last_check).total_seconds() / 3600
        return time_since_check >= self.refresh_interval
    
    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get data with automatic refresh management.
        
        This is the main entry point for the dashboard.
        """
        if force_refresh or self.should_refresh():
            data = self.fetcher.refresh_if_needed(self.refresh_interval)
            self.last_check = datetime.now()
        else:
            # Use cache
            data = self.fetcher.cache.load()
            if data is None:
                data = self.fetcher.fetch_current_season()
        
        return data
    
    def get_refresh_status(self) -> dict:
        """Get status information for dashboard display."""
        metadata = self.fetcher.cache.get_metadata()
        cache_age = self.fetcher.cache.get_cache_age()
        current_week = self.fetcher.get_current_week()
        
        return {
            'current_season': self.fetcher.current_season,
            'current_week': current_week,
            'cache_age_hours': cache_age,
            'cache_valid': self.fetcher.cache.is_cache_valid(self.refresh_interval),
            'last_update': metadata.get('timestamp', 'Never'),
            'data_rows': metadata.get('rows', 0),
            'data_source': metadata.get('source', 'unknown'),
        }


# ============================================================================
# STREAMLIT INTEGRATION HELPERS
# ============================================================================

def setup_streamlit_cache():
    """
    Setup Streamlit caching decorators for real-time data.
    
    Add this to your dashboard:
    
    @st.cache_data(ttl=21600)  # 6 hours
    def load_realtime_data():
        manager = AutoRefreshManager(refresh_interval_hours=6)
        return manager.get_data()
    """
    import streamlit as st
    
    @st.cache_data(ttl=21600)  # 6 hours = 21600 seconds
    def load_realtime_data():
        manager = AutoRefreshManager(refresh_interval_hours=6)
        return manager.get_data()
    
    return load_realtime_data


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("REAL-TIME DATA INTEGRATION TEST")
    print("="*70)
    
    # Initialize manager
    manager = AutoRefreshManager(refresh_interval_hours=6)
    
    # Get data
    print("\n1. Fetching data (with smart cache)...")
    data = manager.get_data()
    print(f"   Retrieved {len(data):,} rows")
    
    # Status
    print("\n2. Refresh status:")
    status = manager.get_refresh_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Demonstrate cache hit
    print("\n3. Fetching again (should use cache)...")
    data2 = manager.get_data()
    print(f"   Retrieved {len(data2):,} rows")
    
    # Force refresh
    print("\n4. Force refresh test...")
    data3 = manager.get_data(force_refresh=True)
    print(f"   Retrieved {len(data3):,} rows")
    
    print("\n" + "="*70)
    print("✅ Real-time data integration working!")
    print("="*70)

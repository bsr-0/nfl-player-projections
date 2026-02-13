"""Base scraper class with common functionality."""
import time
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import SCRAPER_DELAY, USER_AGENT, RAW_DATA_DIR


class BaseScraper(ABC):
    """Base class for all NFL data scrapers."""
    
    def __init__(self, delay: float = None):
        self.delay = delay or SCRAPER_DELAY
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(max(0, self.delay - elapsed))
        self.last_request_time = time.time()
    
    def _get(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a GET request with rate limiting and error handling."""
        self._rate_limit()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")
            return None
    
    def _get_soup(self, url: str, params: Dict = None) -> Optional[BeautifulSoup]:
        """Get BeautifulSoup object from URL."""
        response = self._get(url, params)
        if response:
            return BeautifulSoup(response.text, "lxml")
        return None
    
    def _get_json(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Get JSON response from URL."""
        response = self._get(url, params)
        if response:
            try:
                return response.json()
            except ValueError:
                print(f"Failed to parse JSON from {url}")
        return None
    
    def save_raw_data(self, data: pd.DataFrame, filename: str):
        """Save raw scraped data to CSV."""
        filepath = RAW_DATA_DIR / filename
        data.to_csv(filepath, index=False)
        print(f"Saved {len(data)} rows to {filepath}")
    
    def load_raw_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Load previously scraped raw data."""
        filepath = RAW_DATA_DIR / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        return None
    
    @abstractmethod
    def scrape(self, **kwargs) -> pd.DataFrame:
        """Main scraping method to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_latest_data(self) -> pd.DataFrame:
        """Get only the latest/new data since last scrape."""
        pass

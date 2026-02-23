"""Base scraper class with common functionality."""
import time
import hashlib
import json
import random
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    SCRAPER_DELAY,
    SCRAPER_DELAY_JITTER,
    USER_AGENT,
    USER_AGENT_POOL,
    RAW_DATA_DIR,
    SCRAPER_CACHE_TTL_HOURS,
    SCRAPER_CACHE_SUBDIR,
)


class CachedResponse:
    """Minimal response shim for cached HTTP payloads."""

    def __init__(self, text: str, url: str):
        self.text = text
        self.url = url
        self.status_code = 200

    def json(self):
        return json.loads(self.text)


class BaseScraper(ABC):
    """Base class for all NFL data scrapers."""
    
    def __init__(self, delay: float = None):
        self.delay = delay or SCRAPER_DELAY
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
        self.cache_dir = RAW_DATA_DIR / SCRAPER_CACHE_SUBDIR
        self.cache_ttl_seconds = int(SCRAPER_CACHE_TTL_HOURS * 3600)
        self.last_request_time = 0

    def _pick_user_agent(self) -> str:
        pool = USER_AGENT_POOL if isinstance(USER_AGENT_POOL, list) and USER_AGENT_POOL else []
        if pool:
            return random.choice(pool)
        return USER_AGENT

    def _cache_key(self, url: str, params: Optional[Dict]) -> str:
        payload = {"url": url, "params": params or {}}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest

    def _cache_paths(self, key: str) -> Dict[str, Path]:
        return {
            "body": self.cache_dir / f"{key}.txt",
            "meta": self.cache_dir / f"{key}.json",
        }

    def _load_cache(self, url: str, params: Optional[Dict], allow_stale: bool = False) -> Optional[CachedResponse]:
        key = self._cache_key(url, params)
        paths = self._cache_paths(key)
        if not paths["body"].exists() or not paths["meta"].exists():
            return None
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            fetched_at = float(meta.get("fetched_at", 0))
            age = time.time() - fetched_at
            if allow_stale or age <= self.cache_ttl_seconds:
                text = paths["body"].read_text(encoding="utf-8")
                return CachedResponse(text=text, url=url)
        except Exception:
            return None
        return None

    def _save_cache(self, url: str, params: Optional[Dict], text: str) -> None:
        key = self._cache_key(url, params)
        paths = self._cache_paths(key)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            paths["body"].write_text(text, encoding="utf-8")
            meta = {"url": url, "params": params or {}, "fetched_at": time.time()}
            paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        jitter = random.uniform(0, SCRAPER_DELAY_JITTER) if SCRAPER_DELAY_JITTER else 0
        target_delay = self.delay + jitter
        if elapsed < target_delay:
            time.sleep(max(0, target_delay - elapsed))
        self.last_request_time = time.time()
    
    def _get(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a GET request with rate limiting and error handling."""
        cached = self._load_cache(url, params)
        if cached is not None:
            return cached
        self._rate_limit()
        try:
            headers = {}
            headers["User-Agent"] = self._pick_user_agent()
            response = self.session.get(url, params=params, timeout=30, headers=headers or None)
            response.raise_for_status()
            self._save_cache(url, params, response.text)
            return response
        except requests.RequestException as e:
            stale = self._load_cache(url, params, allow_stale=True)
            if stale is not None:
                print(f"Request failed for {url}: {e} (using cached response)")
                return stale
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

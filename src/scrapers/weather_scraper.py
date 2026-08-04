"""Game-day weather scraper using Open-Meteo historical archive API.

Free, no API key required.  Fetches hourly temperature, wind speed, and
precipitation at each NFL stadium for every game in the schedule table,
then stores one row per game in game_weather.

Dome/retractable-roof stadiums are marked is_dome=1 with NULL weather fields
because indoor conditions are controlled and weather is irrelevant.

Usage:
    from src.scrapers.weather_scraper import WeatherScraper
    scraper = WeatherScraper()
    scraper.scrape_seasons(range(2006, 2026))
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Stadium reference data
# ---------------------------------------------------------------------------

# (latitude, longitude) for each team's home stadium
STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "ARI": (33.5277, -112.2626),  # State Farm Stadium, Glendale AZ
    "ATL": (33.7554, -84.4008),   # Mercedes-Benz Stadium, Atlanta GA
    "BAL": (39.2780, -76.6227),   # M&T Bank Stadium, Baltimore MD
    "BUF": (42.7738, -78.7870),   # Highmark Stadium, Orchard Park NY
    "CAR": (35.2258, -80.8528),   # Bank of America Stadium, Charlotte NC
    "CHI": (41.8623, -87.6167),   # Soldier Field, Chicago IL
    "CIN": (39.0955, -84.5160),   # Paycor Stadium, Cincinnati OH
    "CLE": (41.5061, -81.6995),   # Cleveland Browns Stadium, Cleveland OH
    "DAL": (32.7473, -97.0945),   # AT&T Stadium, Arlington TX (dome)
    "DEN": (39.7439, -105.0201),  # Empower Field, Denver CO
    "DET": (42.3400, -83.0456),   # Ford Field, Detroit MI (dome)
    "GB":  (44.5013, -88.0622),   # Lambeau Field, Green Bay WI
    "HOU": (29.6847, -95.4107),   # NRG Stadium, Houston TX (retractable)
    "IND": (39.7601, -86.1638),   # Lucas Oil Stadium, Indianapolis IN (dome)
    "JAX": (30.3239, -81.6373),   # EverBank Stadium, Jacksonville FL
    "KC":  (39.0489, -94.4839),   # Arrowhead Stadium, Kansas City MO
    "LA":  (33.9535, -118.3392),  # SoFi Stadium, Inglewood CA (dome)
    "LAC": (33.9535, -118.3392),  # SoFi Stadium, Inglewood CA (dome)
    "LV":  (36.0909, -115.1833),  # Allegiant Stadium, Las Vegas NV (dome)
    "MIA": (25.9580, -80.2389),   # Hard Rock Stadium, Miami Gardens FL
    "MIN": (44.9740, -93.2575),   # U.S. Bank Stadium, Minneapolis MN (dome)
    "NE":  (42.0909, -71.2643),   # Gillette Stadium, Foxborough MA
    "NO":  (29.9511, -90.0812),   # Caesars Superdome, New Orleans LA (dome)
    "NYG": (40.8128, -74.0742),   # MetLife Stadium, East Rutherford NJ
    "NYJ": (40.8128, -74.0742),   # MetLife Stadium, East Rutherford NJ
    "PHI": (39.9008, -75.1675),   # Lincoln Financial Field, Philadelphia PA
    "PIT": (40.4468, -80.0158),   # Acrisure Stadium, Pittsburgh PA
    "SF":  (37.4032, -121.9699),  # Levi's Stadium, Santa Clara CA
    "SEA": (47.5952, -122.3316),  # Lumen Field, Seattle WA
    "TB":  (27.9759, -82.5033),   # Raymond James Stadium, Tampa FL
    "TEN": (36.1665, -86.7713),   # Nissan Stadium, Nashville TN
    "WAS": (38.9076, -76.8645),   # Northwest Stadium, Landover MD
    # Relocated franchises (pre-move home)
    "OAK": (37.7516, -122.2005),  # Oakland Coliseum
    "SD":  (32.7831, -117.1197),  # Qualcomm Stadium
    "STL": (38.6328, -90.1884),   # Edward Jones Dome
}

STADIUM_NAMES: Dict[str, str] = {
    "ARI": "State Farm Stadium",
    "ATL": "Mercedes-Benz Stadium",
    "BAL": "M&T Bank Stadium",
    "BUF": "Highmark Stadium",
    "CAR": "Bank of America Stadium",
    "CHI": "Soldier Field",
    "CIN": "Paycor Stadium",
    "CLE": "Cleveland Browns Stadium",
    "DAL": "AT&T Stadium",
    "DEN": "Empower Field at Mile High",
    "DET": "Ford Field",
    "GB":  "Lambeau Field",
    "HOU": "NRG Stadium",
    "IND": "Lucas Oil Stadium",
    "JAX": "EverBank Stadium",
    "KC":  "GEHA Field at Arrowhead Stadium",
    "LA":  "SoFi Stadium",
    "LAC": "SoFi Stadium",
    "LV":  "Allegiant Stadium",
    "MIA": "Hard Rock Stadium",
    "MIN": "U.S. Bank Stadium",
    "NE":  "Gillette Stadium",
    "NO":  "Caesars Superdome",
    "NYG": "MetLife Stadium",
    "NYJ": "MetLife Stadium",
    "PHI": "Lincoln Financial Field",
    "PIT": "Acrisure Stadium",
    "SF":  "Levi's Stadium",
    "SEA": "Lumen Field",
    "TB":  "Raymond James Stadium",
    "TEN": "Nissan Stadium",
    "WAS": "Northwest Stadium",
}

# Teams that play in climate-controlled environments (dome or retractable closed)
DOME_TEAMS = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LA", "LAC", "LV", "MIN", "NO"}

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# WeatherScraper
# ---------------------------------------------------------------------------


class WeatherScraper:
    """Fetches historical game-day weather from Open-Meteo (free, no key)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nfl-projections-weather/1.0"})
        self._last_request = 0.0
        self._delay = 0.5  # Open-Meteo is generous; 0.5s is fine

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request = time.time()

    def fetch_game_weather(
        self, lat: float, lon: float, date_str: str, kickoff_hour_utc: int = 18
    ) -> Optional[Dict]:
        """Fetch hourly weather for a single game location/date.

        Args:
            lat, lon: Stadium coordinates
            date_str: 'YYYY-MM-DD'
            kickoff_hour_utc: UTC hour for kickoff (default 18 = 1pm ET)

        Returns:
            Dict with temp_f, wind_mph, precip_mm — or None on error.
        """
        self._rate_limit()
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,windspeed_10m,precipitation",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "mm",
            "timezone": "UTC",
        }
        try:
            resp = self.session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    [weather] request failed for {date_str}: {e}")
            return None

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("windspeed_10m", [])
        precips = hourly.get("precipitation", [])

        # Find the kickoff hour index
        target_suffix = f"T{kickoff_hour_utc:02d}:00"
        idx = next(
            (i for i, t in enumerate(times) if t.endswith(target_suffix)), None
        )
        if idx is None:
            # Fallback: use the closest available hour
            idx = min(kickoff_hour_utc, len(temps) - 1) if temps else None

        if idx is None or idx >= len(temps):
            return None

        return {
            "temp_f": temps[idx] if idx < len(temps) else None,
            "wind_mph": winds[idx] if idx < len(winds) else None,
            "precip_mm": precips[idx] if idx < len(precips) else None,
        }

    def scrape_seasons(
        self,
        seasons: List[int],
        skip_existing: bool = True,
    ) -> int:
        """Scrape weather for all games in the given seasons.

        Reads games from the schedule table, fetches Open-Meteo data for each
        outdoor game, and stores results in game_weather.

        Returns total rows upserted.
        """
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        total = 0

        with db._get_connection() as conn:
            for season in seasons:
                games = conn.execute(
                    "SELECT season, week, home_team, away_team, game_time "
                    "FROM schedule WHERE season = ? ORDER BY week, home_team",
                    (season,),
                ).fetchall()

                if not games:
                    print(f"  Season {season}: no schedule data, skipping")
                    continue

                existing = set()
                if skip_existing:
                    rows = conn.execute(
                        "SELECT home_team || '_' || week FROM game_weather WHERE season = ?",
                        (season,),
                    ).fetchall()
                    existing = {r[0] for r in rows}

                print(f"  Season {season}: {len(games)} games", end="")
                season_count = 0

                for game in games:
                    s, week, home, away, game_time = game
                    key = f"{home}_{week}"
                    if key in existing:
                        continue

                    is_dome = 1 if home in DOME_TEAMS else 0
                    coords = STADIUM_COORDS.get(home)
                    stadium = STADIUM_NAMES.get(home)

                    # Parse game date
                    if game_time:
                        game_date = str(game_time)[:10]
                    else:
                        continue  # can't fetch without a date

                    # Kickoff UTC hour — default 18 (1pm ET); rough but good enough
                    kickoff_hour_utc = 18

                    weather = None
                    if not is_dome and coords:
                        weather = self.fetch_game_weather(
                            coords[0], coords[1], game_date, kickoff_hour_utc
                        )

                    try:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO game_weather
                                (season, week, game_date, home_team, away_team,
                                 stadium, is_dome, kickoff_time,
                                 temp_f, wind_mph, precip_mm)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                season, week, game_date, home, away,
                                stadium, is_dome, f"{game_date}T{kickoff_hour_utc:02d}:00Z",
                                weather["temp_f"] if weather else None,
                                weather["wind_mph"] if weather else None,
                                weather["precip_mm"] if weather else None,
                            ),
                        )
                        season_count += 1
                    except Exception as e:
                        print(f"\n    [db] insert error {home} wk{week}: {e}")

                conn.commit()
                total += season_count
                print(f" → {season_count} rows")

        return total

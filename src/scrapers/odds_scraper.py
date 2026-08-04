"""NFL odds scraper — The Odds API (https://api.the-odds-api.com).

Fetches game lines (spreads, totals, moneylines) and player props for all NFL
games and persists them to the local SQLite database.  Supports both current
upcoming odds and historical snapshots.

API key is read from config.settings.ODDS_API_KEY (loaded from .env).
The key is deliberately stripped from all cache files so it is never stored
in plaintext on disk.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import ODDS_API_KEY, CURRENT_NFL_SEASON, MIN_HISTORICAL_YEAR
from src.scrapers.base_scraper import BaseScraper, CachedResponse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full team name → abbreviation (matches nfl-data-py / schedule table format)
TEAM_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",  # pre-relocation
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",  # pre-relocation
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",  # pre-relocation
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


def _normalize_team(name: str) -> str:
    """Convert full team name to abbreviation; return as-is if already short."""
    if not name:
        return name
    if name in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[name]
    if len(name) <= 4 and name.isupper():
        return name  # already an abbreviation
    # Partial match fallback
    name_lower = name.lower()
    for full, abbr in TEAM_NAME_TO_ABBR.items():
        if name_lower in full.lower() or full.lower() in name_lower:
            return abbr
    return name


BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Markets we pull for game-level odds
GAME_MARKETS = "h2h,spreads,totals"

# Player prop markets (fantasy-relevant)
PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_rush_yds",
    "player_rush_tds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",
    "player_reception_tds",
    "player_kicking_points",
    "player_pass_interceptions",
    "player_sacks",
    "player_tackles_assists",
]

GAME_EXTRA_MARKETS = "team_totals,alternate_spreads,alternate_totals,h2h_h1,h2h_h2,spreads_h1,spreads_h2,totals_h1,totals_h2,h2h_q1,h2h_q2,h2h_q3,h2h_q4,spreads_q1,totals_q1"

# US bookmakers we care about (used to filter responses and control credit spend)
US_REGIONS = "us"

# Snapshot time suffix appended to each NFL game date for historical calls.
# 17:00 UTC = noon ET — before the earliest Sunday 1pm kickoffs.
SNAPSHOT_TIME_SUFFIX = "T17:00:00Z"

# For Thursday/Saturday/Monday games the date itself is correct; Sunday sweeps all.
# We use a single snapshot per gameday which covers all games on that date.


# ---------------------------------------------------------------------------
# NFLOddsScraper
# ---------------------------------------------------------------------------


class NFLOddsScraper(BaseScraper):
    """Scrapes NFL odds from The Odds API.

    Inherits caching and rate-limiting from BaseScraper but overrides the
    HTTP layer to strip the API key from all cache files.
    """

    def __init__(self):
        super().__init__()
        self.api_key = ODDS_API_KEY
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY is not set.  Add it to .env and reload."
            )
        self._requests_used: Optional[int] = None
        self._requests_remaining: Optional[int] = None

    # ------------------------------------------------------------------
    # Cache helpers — key/metadata exclude the API key
    # ------------------------------------------------------------------

    def _safe_cache_key(self, url: str, params: Optional[Dict]) -> str:
        """Cache key derived from url + params WITHOUT the api key."""
        safe = {k: v for k, v in (params or {}).items() if k != "apiKey"}
        payload = {"url": url, "params": safe}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _load_odds_cache(self, url: str, params: Dict) -> Optional[CachedResponse]:
        key = self._safe_cache_key(url, params)
        paths = self._cache_paths(key)
        if not paths["body"].exists() or not paths["meta"].exists():
            return None
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            age = time.time() - float(meta.get("fetched_at", 0))
            if age <= self.cache_ttl_seconds:
                text = paths["body"].read_text(encoding="utf-8")
                return CachedResponse(text=text, url=url)
        except Exception:
            return None
        return None

    def _save_odds_cache(self, url: str, params: Dict, text: str) -> None:
        key = self._safe_cache_key(url, params)
        paths = self._cache_paths(key)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            paths["body"].write_text(text, encoding="utf-8")
            safe = {k: v for k, v in params.items() if k != "apiKey"}
            meta = {"url": url, "params": safe, "fetched_at": time.time()}
            paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _odds_get(self, url: str, params: Dict) -> Optional[Dict]:
        """GET with caching (key excludes apiKey) and credit tracking."""
        cached = self._load_odds_cache(url, params)
        if cached is not None:
            try:
                return cached.json()
            except Exception:
                return None

        self._rate_limit()
        full_params = {**params, "apiKey": self.api_key}
        try:
            resp = self.session.get(url, params=full_params, timeout=30)
            resp.raise_for_status()
            # Track API credit usage from response headers
            self._requests_used = int(resp.headers.get("x-requests-used", 0) or 0)
            self._requests_remaining = int(
                resp.headers.get("x-requests-remaining", -1) or -1
            )
            self._save_odds_cache(url, params, resp.text)
            return resp.json()
        except Exception as e:
            print(f"  [odds_scraper] request failed {url}: {e}")
            return None

    def _log_credits(self) -> None:
        if self._requests_remaining is not None and self._requests_remaining >= 0:
            print(
                f"  [odds_api] used={self._requests_used} "
                f"remaining={self._requests_remaining}"
            )

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def fetch_events(self) -> List[Dict]:
        """Return upcoming NFL events (id, commence_time, home/away team)."""
        url = f"{BASE_URL}/sports/{SPORT}/events"
        data = self._odds_get(url, {})
        return data if isinstance(data, list) else []

    def fetch_current_odds(
        self,
        markets: str = GAME_MARKETS,
        regions: str = US_REGIONS,
        odds_format: str = "american",
    ) -> List[Dict]:
        """Fetch odds for all upcoming NFL games."""
        url = f"{BASE_URL}/sports/{SPORT}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        data = self._odds_get(url, params)
        self._log_credits()
        return data if isinstance(data, list) else []

    def fetch_historical_odds(
        self,
        date_iso: str,
        markets: str = GAME_MARKETS,
        regions: str = US_REGIONS,
        odds_format: str = "american",
    ) -> Tuple[List[Dict], Optional[str]]:
        """Fetch a historical odds snapshot at a specific UTC datetime.

        Args:
            date_iso: ISO-8601 UTC datetime, e.g. '2023-09-10T17:00:00Z'

        Returns:
            (events list, timestamp string from API) — events is empty on error
            or when the historical tier is not available.
        """
        url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
        params = {
            "date": date_iso,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        data = self._odds_get(url, params)
        self._log_credits()
        if not isinstance(data, dict):
            return [], None
        events = data.get("data", [])
        timestamp = data.get("timestamp")
        return (events if isinstance(events, list) else []), timestamp

    def fetch_event_player_props(
        self,
        event_id: str,
        markets: Optional[List[str]] = None,
        regions: str = US_REGIONS,
        odds_format: str = "american",
    ) -> Optional[Dict]:
        """Fetch player props for a single event.

        Args:
            event_id: The Odds API event ID (from fetch_events / fetch_current_odds)
            markets: list of market strings; defaults to PROP_MARKETS
        """
        url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
        params = {
            "regions": regions,
            "markets": ",".join(markets or PROP_MARKETS),
            "oddsFormat": odds_format,
        }
        data = self._odds_get(url, params)
        self._log_credits()
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Normalisation — API response → DataFrames
    # ------------------------------------------------------------------

    def normalize_game_odds(
        self, events: List[Dict], fetched_at: str
    ) -> pd.DataFrame:
        """Parse a list of event dicts (from current or historical endpoint) into rows."""
        rows = []
        for event in events:
            event_id = event.get("id", "")
            commence_time = event.get("commence_time", "")
            home_team = _normalize_team(event.get("home_team", ""))
            away_team = _normalize_team(event.get("away_team", ""))

            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    market_key = market.get("key", "")
                    outcomes = market.get("outcomes", [])
                    row = _parse_market_outcomes(
                        event_id,
                        commence_time,
                        home_team,
                        away_team,
                        book_key,
                        market_key,
                        outcomes,
                        fetched_at,
                    )
                    if row:
                        rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def normalize_player_props(
        self, event_data: Dict, fetched_at: str
    ) -> pd.DataFrame:
        """Parse a single event's player-prop response into rows."""
        rows = []
        event_id = event_data.get("id", "")
        commence_time = event_data.get("commence_time", "")
        home_team = _normalize_team(event_data.get("home_team", ""))
        away_team = _normalize_team(event_data.get("away_team", ""))

        for bookmaker in event_data.get("bookmakers", []):
            book_key = bookmaker.get("key", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event_id,
                            "game_id": None,
                            "season": None,
                            "week": None,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": book_key,
                            "market": market_key,
                            "player_name": outcome.get("description") or outcome.get("name", ""),
                            "description": outcome.get("name", ""),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                            "fetched_at": fetched_at,
                        }
                    )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def save_game_odds_to_db(self, df: pd.DataFrame) -> int:
        """Upsert game odds rows. Returns count of new rows inserted."""
        if df.empty:
            return 0
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        count = 0
        with db._get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO game_odds
                            (event_id, game_id, season, week, commence_time,
                             home_team, away_team, bookmaker, market,
                             home_price, away_price, home_point, away_point,
                             fetched_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row.get("event_id"),
                            row.get("game_id"),
                            row.get("season"),
                            row.get("week"),
                            row.get("commence_time"),
                            row.get("home_team"),
                            row.get("away_team"),
                            row.get("bookmaker"),
                            row.get("market"),
                            row.get("home_price"),
                            row.get("away_price"),
                            row.get("home_point"),
                            row.get("away_point"),
                            row.get("fetched_at"),
                        ),
                    )
                    count += cursor.rowcount
                except Exception as e:
                    print(f"  [db] insert error: {e}")
            conn.commit()
        return count

    def save_player_props_to_db(self, df: pd.DataFrame) -> int:
        """Upsert player prop rows. Returns count of new rows inserted."""
        if df.empty:
            return 0
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        count = 0
        with db._get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO player_props_odds
                            (event_id, game_id, season, week, commence_time,
                             home_team, away_team, bookmaker, market,
                             player_name, description, price, point, fetched_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            row.get("event_id"),
                            row.get("game_id"),
                            row.get("season"),
                            row.get("week"),
                            row.get("commence_time"),
                            row.get("home_team"),
                            row.get("away_team"),
                            row.get("bookmaker"),
                            row.get("market"),
                            row.get("player_name"),
                            row.get("description"),
                            row.get("price"),
                            row.get("point"),
                            row.get("fetched_at"),
                        ),
                    )
                    count += cursor.rowcount
                except Exception as e:
                    print(f"  [db] insert error: {e}")
            conn.commit()
        return count

    # ------------------------------------------------------------------
    # High-level scrape workflows
    # ------------------------------------------------------------------

    def scrape_current(self, include_props: bool = True) -> Dict[str, int]:
        """Fetch and store all odds for upcoming NFL games.

        Returns dict with 'game_odds' and 'player_props' insert counts.
        """
        fetched_at = _utcnow()
        totals = {"game_odds": 0, "player_props": 0}

        print("Fetching current game odds...")
        events = self.fetch_current_odds()
        if events:
            df = self.normalize_game_odds(events, fetched_at)
            n = self.save_game_odds_to_db(df)
            totals["game_odds"] += n
            print(f"  Stored {n} game-odds rows ({len(events)} events)")
        else:
            print("  No upcoming game odds returned.")

        if include_props:
            # Re-use event IDs from the current call to fetch props
            event_list = self.fetch_events()
            print(f"Fetching player props for {len(event_list)} events...")
            for event in event_list:
                event_id = event.get("id", "")
                if not event_id:
                    continue
                prop_data = self.fetch_event_player_props(event_id)
                if prop_data:
                    df_props = self.normalize_player_props(prop_data, fetched_at)
                    n = self.save_player_props_to_db(df_props)
                    totals["player_props"] += n
                    print(
                        f"  {event.get('home_team')} vs {event.get('away_team')}: "
                        f"{n} prop rows"
                    )

        return totals

    def scrape_historical_season(
        self,
        year: int,
        markets: str = GAME_MARKETS,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """Fetch historical odds for every game date in a given NFL season.

        Uses nfl_data_py schedule to determine game dates, then fetches one
        snapshot per unique date.  Requires a paid historical tier on The Odds API.

        Args:
            year: NFL season year (e.g. 2023)
            markets: comma-separated market keys
            dry_run: if True, print dates but do not call the API

        Returns:
            dict with 'game_odds' insert count and 'dates_fetched'.
        """
        game_dates = _get_game_dates_for_season(year)
        if not game_dates:
            print(f"  No schedule data found for {year}.")
            return {"game_odds": 0, "dates_fetched": 0}

        print(f"Season {year}: {len(game_dates)} unique game dates to fetch.")
        total_rows = 0
        dates_fetched = 0

        for date_str in sorted(game_dates):
            snapshot_dt = f"{date_str}{SNAPSHOT_TIME_SUFFIX}"
            if dry_run:
                print(f"  [dry-run] would fetch: {snapshot_dt}")
                continue

            print(f"  Fetching {snapshot_dt} ...", end=" ", flush=True)
            events, timestamp = self.fetch_historical_odds(snapshot_dt, markets=markets)

            if not events:
                print("no data (historical tier may be required)")
                continue

            fetched_at = timestamp or snapshot_dt
            df = self.normalize_game_odds(events, fetched_at)
            n = self.save_game_odds_to_db(df)
            total_rows += n
            dates_fetched += 1
            print(f"{len(events)} events, {n} rows stored")

        return {"game_odds": total_rows, "dates_fetched": dates_fetched}

    def scrape_historical_range(
        self,
        start_year: int,
        end_year: int,
        markets: str = GAME_MARKETS,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """Scrape historical odds for a range of seasons."""
        totals: Dict[str, int] = {"game_odds": 0, "dates_fetched": 0}
        for year in range(start_year, end_year + 1):
            print(f"\n=== Season {year} ===")
            result = self.scrape_historical_season(year, markets=markets, dry_run=dry_run)
            totals["game_odds"] += result.get("game_odds", 0)
            totals["dates_fetched"] += result.get("dates_fetched", 0)
        return totals

    def scrape_historical_props(
        self,
        seasons: Optional[List[int]] = None,
        markets: Optional[List[str]] = None,
        credit_floor: int = 5000,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """Fetch historical player props for events already stored in game_odds.

        Uses the event-specific historical endpoint, which requires one API
        call per event (costs ~10 credits × number of markets).

        Args:
            seasons: which seasons to cover (default: [2024, 2025])
            markets: prop market keys (default: PROP_MARKETS)
            credit_floor: stop fetching if remaining credits drop below this
            skip_existing: skip event_ids already in player_props_odds

        Returns:
            dict with 'props_rows' and 'events_fetched'.
        """
        from src.utils.database import DatabaseManager

        seasons = seasons or [2024, 2025]
        markets = markets or PROP_MARKETS
        markets_str = ",".join(markets)

        db = DatabaseManager()
        totals = {"props_rows": 0, "events_fetched": 0}

        # Load existing event_ids to skip if requested
        existing_events: set = set()
        if skip_existing:
            with db._get_connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT event_id FROM player_props_odds"
                ).fetchall()
                existing_events = {r[0] for r in rows}

        # Get distinct events for target seasons from game_odds
        with db._get_connection() as conn:
            placeholders = ",".join("?" * len(seasons))
            events = conn.execute(
                f"""
                SELECT DISTINCT event_id, commence_time, home_team, away_team,
                       season, week
                FROM game_odds
                WHERE season IN ({placeholders})
                ORDER BY commence_time
                """,
                seasons,
            ).fetchall()

        print(f"Historical props: {len(events)} events across seasons {seasons}")

        for event_id, commence_time, home, away, season, week in events:
            # Check credits
            if (
                self._requests_remaining is not None
                and self._requests_remaining < credit_floor
            ):
                print(f"  Credit floor reached ({self._requests_remaining} remaining). Stopping.")
                break

            if event_id in existing_events:
                continue

            # Use commence_time date at 17:00 UTC as snapshot (pre-game)
            date_str = commence_time[:10] if commence_time else None
            if not date_str:
                continue
            snapshot = f"{date_str}T17:00:00Z"

            url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
            params = {
                "date": snapshot,
                "regions": US_REGIONS,
                "markets": markets_str,
                "oddsFormat": "american",
            }
            data = self._odds_get(url, params)
            self._log_credits()

            if not data or not isinstance(data, dict):
                continue

            inner = data.get("data", data)
            if not isinstance(inner, dict) or not inner.get("bookmakers"):
                continue

            fetched_at = data.get("timestamp") or snapshot
            df = self.normalize_player_props(inner, fetched_at)

            # Tag season/week
            if not df.empty:
                df["season"] = season
                df["week"] = week

            n = self.save_player_props_to_db(df)
            totals["props_rows"] += n
            totals["events_fetched"] += 1
            existing_events.add(event_id)

            if totals["events_fetched"] % 50 == 0:
                print(
                    f"  [{season} wk{week}] {away}@{home}: {n} rows "
                    f"(total {totals['props_rows']:,}, "
                    f"credits left ~{self._requests_remaining})"
                )

        print(
            f"Done: {totals['events_fetched']} events, "
            f"{totals['props_rows']:,} prop rows stored"
        )
        return totals

    def fetch_win_totals(self, season: int) -> List[Dict]:
        """Fetch NFL team season win total O/U lines for a given season.

        Uses the futures market.  Results are stored directly in game_odds
        with market='win_totals' and home_team=team abbreviation.

        Returns raw bookmaker rows inserted.
        """
        # Win totals are listed as futures — snapshot at start of season
        snapshot = f"{season}-07-01T17:00:00Z"
        url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
        params = {
            "date": snapshot,
            "regions": US_REGIONS,
            "markets": "team_totals",
            "oddsFormat": "american",
        }
        data = self._odds_get(url, params)
        self._log_credits()

        if not data or not isinstance(data, dict):
            # Try alternate futures sport key
            url2 = f"{BASE_URL}/historical/sports/americanfootball_nfl_super_bowl_winner/odds"
            data = self._odds_get(url2, {"date": snapshot, "regions": US_REGIONS,
                                          "markets": "h2h", "oddsFormat": "american"})
            self._log_credits()

        if not data:
            return []

        events = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        fetched_at = (data.get("timestamp") if isinstance(data, dict) else None) or snapshot

        # Normalise and store as game_odds rows with market='team_totals'
        rows_inserted = 0
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        with db._get_connection() as conn:
            cursor = conn.cursor()
            for event in events:
                event_id = event.get("id", "")
                ct = event.get("commence_time", "")
                home = _normalize_team(event.get("home_team", ""))
                away = _normalize_team(event.get("away_team", ""))
                for bookmaker in event.get("bookmakers", []):
                    book_key = bookmaker.get("key", "")
                    for market in bookmaker.get("markets", []):
                        for outcome in market.get("outcomes", []):
                            try:
                                cursor.execute(
                                    """INSERT OR IGNORE INTO game_odds
                                       (event_id, season, commence_time,
                                        home_team, away_team, bookmaker, market,
                                        home_price, home_point, fetched_at)
                                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                    (event_id, season, ct, home, away, book_key,
                                     market.get("key", "win_totals"),
                                     outcome.get("price"), outcome.get("point"),
                                     fetched_at),
                                )
                                rows_inserted += cursor.rowcount
                            except Exception:
                                pass
            conn.commit()
        return rows_inserted

    def scrape_win_totals(self, seasons: Optional[List[int]] = None) -> Dict[str, int]:
        """Scrape season win total O/U lines for multiple seasons."""
        seasons = seasons or list(range(2020, CURRENT_NFL_SEASON + 1))
        total_rows = 0
        for season in seasons:
            print(f"  Win totals {season} ...", end=" ", flush=True)
            n = self.fetch_win_totals(season)
            total_rows += n if isinstance(n, int) else 0
            print(f"{n} rows")
        return {"win_total_rows": total_rows}

    def scrape_extra_game_markets(
        self,
        seasons: Optional[List[int]] = None,
        market_batches: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Fetch team_totals, alternate lines for all historical game dates.

        Fetches one market at a time (separate calls) to avoid 422 errors from
        the date-based historical endpoint which rejects large market lists.

        Returns dict with 'game_odds' rows stored.
        """
        seasons = seasons or list(range(2020, CURRENT_NFL_SEASON + 1))
        # Each string is one call (single market or small batch that's known to work)
        market_batches = market_batches or [
            "team_totals",
            "alternate_spreads",
            "alternate_totals",
        ]
        total_rows = 0

        for market in market_batches:
            print(f"\n  Market: {market}")
            for year in seasons:
                game_dates = _get_game_dates_for_season(year)
                if not game_dates:
                    continue
                season_rows = 0
                for date_str in sorted(game_dates):
                    snapshot = f"{date_str}{SNAPSHOT_TIME_SUFFIX}"
                    events, timestamp = self.fetch_historical_odds(snapshot, markets=market)
                    if not events:
                        continue
                    fetched_at = timestamp or snapshot
                    df = self.normalize_game_odds(events, fetched_at)
                    if not df.empty:
                        df["season"] = year
                    n = self.save_game_odds_to_db(df)
                    season_rows += n
                total_rows += season_rows
                print(f"    {year}: {season_rows:,} rows")
        return {"game_odds": total_rows}

    def scrape_super_bowl_futures(
        self, seasons: Optional[List[int]] = None
    ) -> Dict[str, int]:
        """Fetch Super Bowl winner odds at the start of each season.

        Uses americanfootball_nfl_super_bowl_winner sport key.
        Stored in game_odds with market='outrights'.
        """
        seasons = seasons or list(range(2020, CURRENT_NFL_SEASON + 1))
        total_rows = 0
        from src.utils.database import DatabaseManager
        db = DatabaseManager()

        for season in seasons:
            snapshot = f"{season}-09-05T17:00:00Z"
            url = f"{BASE_URL}/historical/sports/americanfootball_nfl_super_bowl_winner/odds"
            params = {
                "date": snapshot,
                "regions": US_REGIONS,
                "markets": "outrights",
                "oddsFormat": "american",
            }
            data = self._odds_get(url, params)
            self._log_credits()
            if not data or not isinstance(data, dict):
                print(f"  SB futures {season}: no data")
                continue
            events = data.get("data", [])
            fetched_at = data.get("timestamp") or snapshot
            if not isinstance(events, list) or not events:
                print(f"  SB futures {season}: empty")
                continue

            count = 0
            with db._get_connection() as conn:
                cursor = conn.cursor()
                for event in events:
                    event_id = event.get("id", "")
                    ct = event.get("commence_time", "")
                    for bookmaker in event.get("bookmakers", []):
                        book_key = bookmaker.get("key", "")
                        for market in bookmaker.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                try:
                                    cursor.execute(
                                        """INSERT OR IGNORE INTO game_odds
                                           (event_id, season, commence_time,
                                            home_team, bookmaker, market,
                                            home_price, fetched_at)
                                           VALUES (?,?,?,?,?,?,?,?)""",
                                        (event_id, season, ct,
                                         _normalize_team(outcome.get("name", "")),
                                         book_key, "sb_futures",
                                         outcome.get("price"), fetched_at),
                                    )
                                    count += cursor.rowcount
                                except Exception:
                                    pass
                conn.commit()
            total_rows += count
            print(f"  SB futures {season}: {count} rows")

        return {"sb_futures_rows": total_rows}

    def scrape_preseason(
        self, seasons: Optional[List[int]] = None, markets: str = GAME_MARKETS
    ) -> Dict[str, int]:
        """Fetch NFL preseason game odds for all available seasons."""
        seasons = seasons or list(range(2020, CURRENT_NFL_SEASON + 1))
        total_rows = 0
        sport = "americanfootball_nfl_preseason"

        for season in seasons:
            # Preseason runs Aug 1 – Aug 31 roughly
            # Sample dates across August for each season
            import datetime
            dates = []
            start = datetime.date(season, 8, 1)
            end = datetime.date(season, 9, 5)
            d = start
            while d <= end:
                dates.append(d.strftime("%Y-%m-%d"))
                d += datetime.timedelta(days=1)

            season_rows = 0
            seen_events: set = set()
            for date_str in dates:
                snapshot = f"{date_str}T17:00:00Z"
                url = f"{BASE_URL}/historical/sports/{sport}/odds"
                params = {
                    "date": snapshot,
                    "regions": US_REGIONS,
                    "markets": markets,
                    "oddsFormat": "american",
                }
                data = self._odds_get(url, params)
                if not data or not isinstance(data, dict):
                    continue
                events = data.get("data", [])
                if not isinstance(events, list) or not events:
                    continue
                fetched_at = data.get("timestamp") or snapshot
                # Only process new events
                new_events = [e for e in events if e.get("id") not in seen_events]
                if not new_events:
                    continue
                for e in new_events:
                    seen_events.add(e.get("id", ""))
                df = self.normalize_game_odds(new_events, fetched_at)
                if not df.empty:
                    df["season"] = season
                n = self.save_game_odds_to_db(df)
                season_rows += n

            self._log_credits()
            print(f"  Preseason {season}: {season_rows:,} rows")
            total_rows += season_rows

        return {"preseason_rows": total_rows}

    def scrape_all_remaining(self, credit_floor: int = 2000) -> Dict[str, int]:
        """Pull every remaining data category not yet ingested.

        Order (highest value first, stops at credit_floor):
          1. team_totals + alternate lines (2020-2025)
          2. Super Bowl futures (2020-2025)
          3. NFL preseason game odds (2020-2025)
        """
        totals: Dict[str, int] = {}

        print("\n=== 1/3  team_totals + alternate lines ===")
        r = self.scrape_extra_game_markets()
        totals["extra_game_odds"] = r.get("game_odds", 0)

        if self._requests_remaining is not None and self._requests_remaining < credit_floor:
            print("Credit floor hit — stopping early.")
            return totals

        print("\n=== 2/3  Super Bowl futures 2020-2025 ===")
        r2 = self.scrape_super_bowl_futures()
        totals["sb_futures"] = r2.get("sb_futures_rows", 0)

        if self._requests_remaining is not None and self._requests_remaining < credit_floor:
            print("Credit floor hit — stopping early.")
            return totals

        print("\n=== 3/3  NFL preseason game odds 2020-2025 ===")
        r3 = self.scrape_preseason()
        totals["preseason_rows"] = r3.get("preseason_rows", 0)

        return totals

    # ------------------------------------------------------------------
    # Required abstract method implementations
    # ------------------------------------------------------------------

    def scrape(self, **kwargs) -> pd.DataFrame:
        """Fetch current game odds and return as DataFrame."""
        fetched_at = _utcnow()
        events = self.fetch_current_odds(**kwargs)
        if not events:
            return pd.DataFrame()
        return self.normalize_game_odds(events, fetched_at)

    def get_latest_data(self) -> pd.DataFrame:
        """Alias for scrape() — returns current upcoming game odds."""
        return self.scrape()

    # ------------------------------------------------------------------
    # Quota check
    # ------------------------------------------------------------------

    def check_quota(self) -> Dict:
        """Return current API quota status without consuming sports-data credits."""
        url = f"{BASE_URL}/sports"
        data = self._odds_get(url, {})
        self._log_credits()
        return {
            "requests_used": self._requests_used,
            "requests_remaining": self._requests_remaining,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_market_outcomes(
    event_id: str,
    commence_time: str,
    home_team: str,
    away_team: str,
    bookmaker: str,
    market: str,
    outcomes: List[Dict],
    fetched_at: str,
) -> Optional[Dict]:
    """Convert a market's outcome list to a single flat dict.

    Layout per market type:
    - h2h: home_price = home ML, away_price = away ML
    - spreads: home_point = home spread, home_price = juice; same for away
    - totals: home_point = over line, home_price = over price;
              away_point = under line, away_price = under price
    """
    if not outcomes:
        return None

    # Index by name
    by_name: Dict[str, Dict] = {}
    for o in outcomes:
        by_name[o.get("name", "")] = o

    home_price = away_price = home_point = away_point = None

    if market == "h2h":
        home_price = by_name.get(home_team, {}).get("price")
        away_price = by_name.get(away_team, {}).get("price")
    elif market == "spreads":
        h = by_name.get(home_team, {})
        a = by_name.get(away_team, {})
        home_price = h.get("price")
        home_point = h.get("point")
        away_price = a.get("price")
        away_point = a.get("point")
    elif market == "totals":
        over = by_name.get("Over", {})
        under = by_name.get("Under", {})
        home_price = over.get("price")
        home_point = over.get("point")
        away_price = under.get("price")
        away_point = under.get("point")
    elif market == "team_totals":
        # outcomes keyed by team name in "description"; "name" = Over/Under
        for o in outcomes:
            if o.get("name") == "Over":
                team = _normalize_team(o.get("description", ""))
                if team == home_team:
                    home_point = o.get("point")
                    home_price = o.get("price")
                elif team == away_team:
                    away_point = o.get("point")
                    away_price = o.get("price")
    elif market in ("alternate_spreads", "alternate_totals",
                    "h2h_h1", "h2h_h2", "spreads_h1", "spreads_h2",
                    "totals_h1", "totals_h2", "h2h_q1", "h2h_q2",
                    "h2h_q3", "h2h_q4", "spreads_q1", "totals_q1"):
        # Handle same as their base counterparts
        if "h2h" in market:
            home_price = by_name.get(home_team, {}).get("price")
            away_price = by_name.get(away_team, {}).get("price")
        elif "spreads" in market:
            h = by_name.get(home_team, {})
            a = by_name.get(away_team, {})
            home_price = h.get("price"); home_point = h.get("point")
            away_price = a.get("price"); away_point = a.get("point")
        elif "totals" in market:
            over = by_name.get("Over", {})
            under = by_name.get("Under", {})
            home_price = over.get("price"); home_point = over.get("point")
            away_price = under.get("price"); away_point = under.get("point")
        else:
            return None
    else:
        return None

    return {
        "event_id": event_id,
        "game_id": None,
        "season": None,
        "week": None,
        "commence_time": commence_time,
        "home_team": home_team,
        "away_team": away_team,
        "bookmaker": bookmaker,
        "market": market,
        "home_price": home_price,
        "away_price": away_price,
        "home_point": home_point,
        "away_point": away_point,
        "fetched_at": fetched_at,
    }


def _get_game_dates_for_season(year: int) -> List[str]:
    """Return sorted list of unique 'YYYY-MM-DD' game dates for an NFL season.

    Tries nfl_data_py first; falls back to the local schedule DB table.
    Returns empty list if neither source has data.
    """
    try:
        import nfl_data_py as nfl

        df = nfl.import_schedules([year])
        if df is not None and not df.empty and "gameday" in df.columns:
            dates = df["gameday"].dropna().astype(str).unique().tolist()
            return [d for d in dates if d and d != "nan"]
    except Exception:
        pass

    # Fallback: local schedule table
    try:
        from src.utils.database import DatabaseManager

        db = DatabaseManager()
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT game_time FROM schedule WHERE season = ? "
                "AND game_time IS NOT NULL",
                (year,),
            ).fetchall()
        dates = set()
        for (gt,) in rows:
            if gt:
                dates.add(str(gt)[:10])
        return sorted(dates)
    except Exception:
        pass

    return []

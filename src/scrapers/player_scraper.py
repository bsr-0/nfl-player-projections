"""Scraper for NFL player statistics from Pro Football Reference."""
import re
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

from .base_scraper import BaseScraper

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import POSITIONS, SEASONS_TO_SCRAPE
from src.utils.helpers import (
    normalize_player_name, generate_player_id, standardize_team_name,
    calculate_fantasy_points
)


class PlayerStatsScraper(BaseScraper):
    """Scraper for player weekly statistics from Pro Football Reference."""
    
    BASE_URL = "https://www.pro-football-reference.com"
    
    def __init__(self, delay: float = None):
        super().__init__(delay or 3.0)  # PFR needs slower rate
    
    def scrape(self, seasons: List[int] = None, positions: List[str] = None) -> pd.DataFrame:
        """Scrape player stats for specified seasons and positions."""
        seasons = seasons or SEASONS_TO_SCRAPE
        positions = positions or POSITIONS
        
        all_data = []
        
        for season in seasons:
            print(f"Scraping season {season}...")
            
            # Scrape weekly fantasy data
            fantasy_df = self._scrape_fantasy_weekly(season)
            if fantasy_df is not None and len(fantasy_df) > 0:
                all_data.append(fantasy_df)
            
            # Scrape detailed passing stats
            passing_df = self._scrape_passing_weekly(season)
            if passing_df is not None:
                all_data.append(passing_df)
            
            # Scrape rushing/receiving stats
            skill_df = self._scrape_skill_weekly(season)
            if skill_df is not None:
                all_data.append(skill_df)
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine and deduplicate
        combined = pd.concat(all_data, ignore_index=True)
        combined = self._merge_and_clean(combined)
        
        # Filter to requested positions
        combined = combined[combined["position"].isin(positions)]
        
        # Calculate fantasy points
        combined["fantasy_points"] = combined.apply(
            lambda row: calculate_fantasy_points(row.to_dict()), axis=1
        )
        
        self.save_raw_data(combined, "player_weekly_stats.csv")
        return combined
    
    def _scrape_fantasy_weekly(self, season: int) -> Optional[pd.DataFrame]:
        """Scrape weekly fantasy stats page."""
        url = f"{self.BASE_URL}/years/{season}/fantasy.htm"
        soup = self._get_soup(url)
        
        if not soup:
            return None
        
        table = soup.find("table", {"id": "fantasy"})
        if not table:
            return None
        
        rows = []
        tbody = table.find("tbody")
        if not tbody:
            return None
        
        for tr in tbody.find_all("tr"):
            if tr.get("class") and "thead" in tr.get("class"):
                continue
            
            cells = tr.find_all(["th", "td"])
            if len(cells) < 10:
                continue
            
            player_cell = cells[0]
            player_link = player_cell.find("a")
            if not player_link:
                continue
            
            player_name = player_link.text.strip()
            player_href = player_link.get("href", "")
            
            # Extract player ID from href
            player_id_match = re.search(r"/players/\w/(\w+)\.htm", player_href)
            pfr_id = player_id_match.group(1) if player_id_match else generate_player_id(player_name)
            
            row_data = {
                "player_id": pfr_id,
                "name": player_name,
                "season": season,
                "team": self._get_cell_value(cells, 1),
                "position": self._get_cell_value(cells, 2),
                "games_played": self._get_cell_int(cells, 4),
                "passing_completions": self._get_cell_int(cells, 5),
                "passing_attempts": self._get_cell_int(cells, 6),
                "passing_yards": self._get_cell_int(cells, 7),
                "passing_tds": self._get_cell_int(cells, 8),
                "interceptions": self._get_cell_int(cells, 9),
                "rushing_attempts": self._get_cell_int(cells, 10),
                "rushing_yards": self._get_cell_int(cells, 11),
                "rushing_tds": self._get_cell_int(cells, 13),
                "targets": self._get_cell_int(cells, 14),
                "receptions": self._get_cell_int(cells, 15),
                "receiving_yards": self._get_cell_int(cells, 16),
                "receiving_tds": self._get_cell_int(cells, 18),
                "fumbles_lost": self._get_cell_int(cells, 20),
            }
            
            row_data["team"] = standardize_team_name(row_data["team"])
            rows.append(row_data)
        
        return pd.DataFrame(rows) if rows else None
    
    def _scrape_passing_weekly(self, season: int) -> Optional[pd.DataFrame]:
        """Scrape weekly passing stats."""
        all_weeks = []
        
        for week in range(1, 19):
            url = f"{self.BASE_URL}/years/{season}/week_{week}.htm"
            soup = self._get_soup(url)
            
            if not soup:
                continue
            
            # Find all game boxes
            games = soup.find_all("div", class_="game_summary")
            
            for game in games:
                # Extract passing leaders from game summary
                passing_data = self._extract_game_passing(game, season, week)
                all_weeks.extend(passing_data)
        
        return pd.DataFrame(all_weeks) if all_weeks else None
    
    def _scrape_skill_weekly(self, season: int) -> Optional[pd.DataFrame]:
        """Scrape weekly rushing/receiving stats from game logs."""
        # This would scrape individual game logs
        # For efficiency, we'll rely on the fantasy page which has aggregated data
        return None
    
    def _extract_game_passing(self, game_div, season: int, week: int) -> List[Dict]:
        """Extract passing stats from a game summary div."""
        results = []
        
        # Find team names
        teams = game_div.find_all("td", class_="right")
        
        # Find passing stats tables within game
        tables = game_div.find_all("table")
        
        for table in tables:
            header = table.find("thead")
            if header and "Passing" in header.text:
                tbody = table.find("tbody")
                if tbody:
                    for tr in tbody.find_all("tr"):
                        cells = tr.find_all("td")
                        if len(cells) >= 4:
                            player_link = tr.find("a")
                            if player_link:
                                results.append({
                                    "name": player_link.text.strip(),
                                    "season": season,
                                    "week": week,
                                })
        
        return results
    
    def _get_cell_value(self, cells: list, index: int) -> str:
        """Safely get cell text value."""
        if index < len(cells):
            return cells[index].text.strip()
        return ""
    
    def _get_cell_int(self, cells: list, index: int) -> int:
        """Safely get cell integer value."""
        value = self._get_cell_value(cells, index)
        try:
            return int(value) if value else 0
        except ValueError:
            return 0
    
    def _get_cell_float(self, cells: list, index: int) -> float:
        """Safely get cell float value."""
        value = self._get_cell_value(cells, index)
        try:
            return float(value) if value else 0.0
        except ValueError:
            return 0.0
    
    def _merge_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge duplicate rows and clean data."""
        if df.empty:
            return df
        
        # Group by player, season, week and aggregate
        agg_cols = {
            "name": "first",
            "team": "first",
            "position": "first",
            "games_played": "max",
            "passing_attempts": "max",
            "passing_completions": "max",
            "passing_yards": "max",
            "passing_tds": "max",
            "interceptions": "max",
            "rushing_attempts": "max",
            "rushing_yards": "max",
            "rushing_tds": "max",
            "targets": "max",
            "receptions": "max",
            "receiving_yards": "max",
            "receiving_tds": "max",
            "fumbles_lost": "max",
        }
        
        # Only aggregate columns that exist
        agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
        
        group_cols = ["player_id", "season"]
        if "week" in df.columns:
            group_cols.append("week")
        
        existing_group_cols = [c for c in group_cols if c in df.columns]
        
        if existing_group_cols and agg_cols:
            df = df.groupby(existing_group_cols, as_index=False).agg(agg_cols)
        
        return df
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get only the most recent week's data."""
        import datetime
        
        # Determine current season and week
        today = datetime.date.today()
        if today.month >= 9:
            current_season = today.year
        else:
            current_season = today.year - 1
        
        # Scrape only current season
        return self.scrape(seasons=[current_season])
    
    def scrape_player_game_logs(self, player_id: str, seasons: List[int] = None) -> pd.DataFrame:
        """Scrape detailed game logs for a specific player."""
        seasons = seasons or SEASONS_TO_SCRAPE
        all_games = []
        
        for season in seasons:
            url = f"{self.BASE_URL}/players/{player_id[0].upper()}/{player_id}/gamelog/{season}/"
            soup = self._get_soup(url)
            
            if not soup:
                continue
            
            table = soup.find("table", {"id": "stats"})
            if not table:
                continue
            
            tbody = table.find("tbody")
            if not tbody:
                continue
            
            for tr in tbody.find_all("tr"):
                if tr.get("class") and "thead" in tr.get("class"):
                    continue
                
                cells = tr.find_all(["th", "td"])
                if len(cells) < 10:
                    continue
                
                game_data = self._parse_game_log_row(cells, player_id, season)
                if game_data:
                    all_games.append(game_data)
        
        return pd.DataFrame(all_games)
    
    def _parse_game_log_row(self, cells: list, player_id: str, season: int) -> Optional[Dict]:
        """Parse a single game log row."""
        week_str = self._get_cell_value(cells, 1)
        if not week_str or not week_str.isdigit():
            return None
        
        week = int(week_str)
        
        # Determine home/away
        game_loc = self._get_cell_value(cells, 4)
        home_away = "away" if "@" in game_loc else "home"
        
        opponent = self._get_cell_value(cells, 5)
        opponent = standardize_team_name(opponent)
        
        team = self._get_cell_value(cells, 3)
        team = standardize_team_name(team)
        
        return {
            "player_id": player_id,
            "season": season,
            "week": week,
            "team": team,
            "opponent": opponent,
            "home_away": home_away,
            "games_played": 1,
        }


class NFLDataScraper(BaseScraper):
    """Alternative scraper using NFL.com data via their API."""
    
    API_BASE = "https://api.nfl.com/v3/shield"
    
    def __init__(self, delay: float = None):
        super().__init__(delay or 1.0)
    
    def scrape(self, seasons: List[int] = None, positions: List[str] = None) -> pd.DataFrame:
        """Scrape from NFL API (requires authentication for full access)."""
        # NFL API requires OAuth - this is a placeholder for the structure
        # In practice, you'd need to handle authentication
        print("NFL API scraping requires authentication setup")
        return pd.DataFrame()
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get latest data from NFL API."""
        return self.scrape()


class FantasyProssScraper(BaseScraper):
    """Scraper for FantasyPros projections and stats."""
    
    BASE_URL = "https://www.fantasypros.com/nfl/stats"
    
    def scrape(self, seasons: List[int] = None, positions: List[str] = None) -> pd.DataFrame:
        """Scrape player stats from FantasyPros."""
        seasons = seasons or SEASONS_TO_SCRAPE
        positions = positions or POSITIONS
        
        all_data = []
        
        for season in seasons:
            for position in positions:
                print(f"Scraping FantasyPros {position} stats for {season}...")
                pos_data = self._scrape_position_stats(season, position)
                if pos_data is not None:
                    all_data.append(pos_data)
        
        if not all_data:
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        self.save_raw_data(combined, "fantasypros_stats.csv")
        return combined
    
    def _scrape_position_stats(self, season: int, position: str) -> Optional[pd.DataFrame]:
        """Scrape stats for a specific position and season."""
        url = f"{self.BASE_URL}/{position.lower()}.php"
        params = {"year": season, "scoring": "PPR"}
        
        soup = self._get_soup(url, params)
        if not soup:
            return None
        
        table = soup.find("table", {"id": "data"})
        if not table:
            return None
        
        rows = []
        tbody = table.find("tbody")
        if not tbody:
            return None
        
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 5:
                continue
            
            player_cell = cells[0]
            player_link = player_cell.find("a")
            player_name = player_link.text.strip() if player_link else player_cell.text.strip()
            
            row_data = {
                "name": player_name,
                "position": position,
                "season": season,
                "player_id": generate_player_id(player_name, position),
            }
            
            # Parse position-specific columns
            if position == "QB":
                row_data.update(self._parse_qb_row(cells))
            elif position == "RB":
                row_data.update(self._parse_rb_row(cells))
            elif position in ["WR", "TE"]:
                row_data.update(self._parse_receiver_row(cells))
            
            rows.append(row_data)
        
        return pd.DataFrame(rows) if rows else None
    
    def _parse_qb_row(self, cells: list) -> Dict:
        """Parse QB-specific stats."""
        return {
            "passing_completions": self._get_cell_int(cells, 1),
            "passing_attempts": self._get_cell_int(cells, 2),
            "passing_yards": self._get_cell_int(cells, 4),
            "passing_tds": self._get_cell_int(cells, 6),
            "interceptions": self._get_cell_int(cells, 7),
            "rushing_attempts": self._get_cell_int(cells, 8),
            "rushing_yards": self._get_cell_int(cells, 9),
            "rushing_tds": self._get_cell_int(cells, 10),
            "fumbles_lost": self._get_cell_int(cells, 11),
        }
    
    def _parse_rb_row(self, cells: list) -> Dict:
        """Parse RB-specific stats."""
        return {
            "rushing_attempts": self._get_cell_int(cells, 1),
            "rushing_yards": self._get_cell_int(cells, 2),
            "rushing_tds": self._get_cell_int(cells, 4),
            "targets": self._get_cell_int(cells, 5),
            "receptions": self._get_cell_int(cells, 6),
            "receiving_yards": self._get_cell_int(cells, 7),
            "receiving_tds": self._get_cell_int(cells, 9),
            "fumbles_lost": self._get_cell_int(cells, 10),
        }
    
    def _parse_receiver_row(self, cells: list) -> Dict:
        """Parse WR/TE-specific stats."""
        return {
            "targets": self._get_cell_int(cells, 1),
            "receptions": self._get_cell_int(cells, 2),
            "receiving_yards": self._get_cell_int(cells, 3),
            "receiving_tds": self._get_cell_int(cells, 5),
            "rushing_attempts": self._get_cell_int(cells, 6),
            "rushing_yards": self._get_cell_int(cells, 7),
            "rushing_tds": self._get_cell_int(cells, 8),
            "fumbles_lost": self._get_cell_int(cells, 9),
        }
    
    def _get_cell_int(self, cells: list, index: int) -> int:
        """Safely get cell integer value."""
        if index < len(cells):
            value = cells[index].text.strip().replace(",", "")
            try:
                return int(float(value)) if value else 0
            except ValueError:
                return 0
        return 0
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get current season data."""
        import datetime
        today = datetime.date.today()
        current_season = today.year if today.month >= 9 else today.year - 1
        return self.scrape(seasons=[current_season])

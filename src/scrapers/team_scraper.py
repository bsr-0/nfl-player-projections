"""Scraper for NFL team statistics."""
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

from .base_scraper import BaseScraper

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import SEASONS_TO_SCRAPE
from src.utils.helpers import standardize_team_name


class TeamStatsScraper(BaseScraper):
    """Scraper for team-level statistics from Pro Football Reference."""
    
    BASE_URL = "https://www.pro-football-reference.com"
    
    def __init__(self, delay: float = None):
        super().__init__(delay or 3.0)
    
    def scrape(self, seasons: List[int] = None) -> pd.DataFrame:
        """Scrape team stats for specified seasons."""
        seasons = seasons or SEASONS_TO_SCRAPE
        
        all_offense = []
        all_defense = []
        
        for season in seasons:
            print(f"Scraping team stats for {season}...")
            
            # Offensive stats
            offense_df = self._scrape_team_offense(season)
            if offense_df is not None:
                all_offense.append(offense_df)
            
            # Defensive stats
            defense_df = self._scrape_team_defense(season)
            if defense_df is not None:
                all_defense.append(defense_df)
        
        # Combine offense data
        if all_offense:
            offense_combined = pd.concat(all_offense, ignore_index=True)
            self.save_raw_data(offense_combined, "team_offense_stats.csv")
        else:
            offense_combined = pd.DataFrame()
        
        # Combine defense data
        if all_defense:
            defense_combined = pd.concat(all_defense, ignore_index=True)
            self.save_raw_data(defense_combined, "team_defense_stats.csv")
        else:
            defense_combined = pd.DataFrame()
        
        return offense_combined
    
    def _scrape_team_offense(self, season: int) -> Optional[pd.DataFrame]:
        """Scrape team offensive statistics."""
        url = f"{self.BASE_URL}/years/{season}/"
        soup = self._get_soup(url)
        
        if not soup:
            return None
        
        # Find AFC and NFC standings tables which have team stats
        rows = []
        
        for conf in ["AFC", "NFC"]:
            table = soup.find("table", {"id": f"{conf}"})
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
                
                team_cell = cells[0]
                team_link = team_cell.find("a")
                if not team_link:
                    continue
                
                team_name = team_link.text.strip()
                team_abbr = standardize_team_name(team_name)
                
                row_data = {
                    "team": team_abbr,
                    "season": season,
                    "wins": self._get_cell_int(cells, 1),
                    "losses": self._get_cell_int(cells, 2),
                    "points_scored": self._get_cell_int(cells, 4),
                    "points_allowed": self._get_cell_int(cells, 5),
                    "point_diff": self._get_cell_int(cells, 6),
                }
                rows.append(row_data)
        
        # Also scrape detailed offensive stats
        offense_url = f"{self.BASE_URL}/years/{season}/opp.htm"
        offense_soup = self._get_soup(offense_url)
        
        if offense_soup:
            offense_table = offense_soup.find("table", {"id": "team_stats"})
            if offense_table:
                self._parse_detailed_team_stats(offense_table, rows, season, "offense")
        
        return pd.DataFrame(rows) if rows else None
    
    def _scrape_team_defense(self, season: int) -> Optional[pd.DataFrame]:
        """Scrape team defensive statistics (fantasy points allowed by position)."""
        url = f"{self.BASE_URL}/years/{season}/opp.htm"
        soup = self._get_soup(url)
        
        if not soup:
            return None
        
        rows = []
        
        # Find defensive stats table
        table = soup.find("table", {"id": "team_stats"})
        if not table:
            return None
        
        tbody = table.find("tbody")
        if not tbody:
            return None
        
        for tr in tbody.find_all("tr"):
            if tr.get("class") and "thead" in tr.get("class"):
                continue
            
            cells = tr.find_all(["th", "td"])
            if len(cells) < 15:
                continue
            
            team_cell = cells[0]
            team_link = team_cell.find("a")
            if not team_link:
                continue
            
            team_name = team_link.text.strip()
            team_abbr = standardize_team_name(team_name)
            
            row_data = {
                "team": team_abbr,
                "season": season,
                "games": self._get_cell_int(cells, 1),
                "points_allowed": self._get_cell_int(cells, 2),
                "yards_allowed": self._get_cell_int(cells, 3),
                "plays_against": self._get_cell_int(cells, 4),
                "yards_per_play_allowed": self._get_cell_float(cells, 5),
                "turnovers_forced": self._get_cell_int(cells, 6),
                "fumbles_recovered": self._get_cell_int(cells, 7),
                "first_downs_allowed": self._get_cell_int(cells, 8),
                "passing_yards_allowed": self._get_cell_int(cells, 10),
                "passing_tds_allowed": self._get_cell_int(cells, 11),
                "interceptions": self._get_cell_int(cells, 12),
                "rushing_yards_allowed": self._get_cell_int(cells, 15),
                "rushing_tds_allowed": self._get_cell_int(cells, 16),
            }
            rows.append(row_data)
        
        return pd.DataFrame(rows) if rows else None
    
    def _parse_detailed_team_stats(self, table, existing_rows: List[Dict], 
                                    season: int, stat_type: str):
        """Parse detailed stats and merge with existing rows."""
        tbody = table.find("tbody")
        if not tbody:
            return
        
        team_stats = {}
        for tr in tbody.find_all("tr"):
            if tr.get("class") and "thead" in tr.get("class"):
                continue
            
            cells = tr.find_all(["th", "td"])
            if len(cells) < 10:
                continue
            
            team_cell = cells[0]
            team_link = team_cell.find("a")
            if not team_link:
                continue
            
            team_name = team_link.text.strip()
            team_abbr = standardize_team_name(team_name)
            
            team_stats[team_abbr] = {
                "total_yards": self._get_cell_int(cells, 3),
                "total_plays": self._get_cell_int(cells, 4),
                "passing_yards": self._get_cell_int(cells, 10),
                "passing_tds": self._get_cell_int(cells, 11),
                "rushing_yards": self._get_cell_int(cells, 15),
                "rushing_tds": self._get_cell_int(cells, 16),
            }
        
        # Merge with existing rows
        for row in existing_rows:
            team = row.get("team")
            if team in team_stats:
                row.update(team_stats[team])
    
    def scrape_weekly_team_stats(self, season: int) -> pd.DataFrame:
        """Scrape week-by-week team statistics."""
        all_weeks = []
        
        for week in range(1, 19):
            print(f"Scraping week {week} team stats for {season}...")
            week_data = self._scrape_week_games(season, week)
            all_weeks.extend(week_data)
        
        df = pd.DataFrame(all_weeks)
        if not df.empty:
            self.save_raw_data(df, f"team_weekly_stats_{season}.csv")
        return df
    
    def _scrape_week_games(self, season: int, week: int) -> List[Dict]:
        """Scrape game results for a specific week."""
        url = f"{self.BASE_URL}/years/{season}/week_{week}.htm"
        soup = self._get_soup(url)
        
        if not soup:
            return []
        
        results = []
        
        # Find game summaries
        games = soup.find_all("div", class_="game_summary")
        
        for game in games:
            game_data = self._parse_game_summary(game, season, week)
            results.extend(game_data)
        
        return results
    
    def _parse_game_summary(self, game_div, season: int, week: int) -> List[Dict]:
        """Parse a game summary div into team stats."""
        results = []
        
        # Find team rows
        teams_table = game_div.find("table", class_="teams")
        if not teams_table:
            return results
        
        rows = teams_table.find_all("tr")
        if len(rows) < 2:
            return results
        
        team_data = []
        for tr in rows:
            team_cell = tr.find("td")
            score_cell = tr.find("td", class_="right")
            
            if team_cell and score_cell:
                team_link = team_cell.find("a")
                if team_link:
                    team_name = team_link.text.strip()
                    team_abbr = standardize_team_name(team_name)
                    try:
                        score = int(score_cell.text.strip())
                    except ValueError:
                        score = 0
                    
                    team_data.append({
                        "team": team_abbr,
                        "score": score,
                    })
        
        if len(team_data) == 2:
            # Away team is first, home team is second
            away_team, home_team = team_data[0], team_data[1]
            
            results.append({
                "team": away_team["team"],
                "season": season,
                "week": week,
                "opponent": home_team["team"],
                "home_away": "away",
                "points_scored": away_team["score"],
                "points_allowed": home_team["score"],
            })
            
            results.append({
                "team": home_team["team"],
                "season": season,
                "week": week,
                "opponent": away_team["team"],
                "home_away": "home",
                "points_scored": home_team["score"],
                "points_allowed": away_team["score"],
            })
        
        return results
    
    def scrape_fantasy_points_allowed(self, season: int) -> pd.DataFrame:
        """Scrape fantasy points allowed by position for each team."""
        # This would typically come from a fantasy-specific source
        # For now, we'll calculate it from player stats
        return pd.DataFrame()
    
    def _get_cell_value(self, cells: list, index: int) -> str:
        """Safely get cell text value."""
        if index < len(cells):
            return cells[index].text.strip()
        return ""
    
    def _get_cell_int(self, cells: list, index: int) -> int:
        """Safely get cell integer value."""
        value = self._get_cell_value(cells, index)
        try:
            return int(value.replace(",", "")) if value else 0
        except ValueError:
            return 0
    
    def _get_cell_float(self, cells: list, index: int) -> float:
        """Safely get cell float value."""
        value = self._get_cell_value(cells, index)
        try:
            return float(value) if value else 0.0
        except ValueError:
            return 0.0
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get current season team data."""
        import datetime
        today = datetime.date.today()
        current_season = today.year if today.month >= 9 else today.year - 1
        return self.scrape(seasons=[current_season])


class SnapCountScraper(BaseScraper):
    """
    Scraper for snap count data.
    
    Note: Snap count data is not consistently available from free sources.
    This scraper attempts to get data from nfl_data_py but will gracefully
    skip if unavailable. The model can still function without snap data
    by using other volume metrics (targets, carries, etc.).
    """
    
    def __init__(self, delay: float = None):
        super().__init__(delay or 1.0)
    
    def scrape(self, seasons: List[int] = None) -> pd.DataFrame:
        """
        Attempt to get snap count data.
        
        Returns empty DataFrame if snap data is not available.
        This is not critical - the model uses other volume metrics.
        """
        seasons = seasons or SEASONS_TO_SCRAPE
        
        try:
            import nfl_data_py as nfl
            
            print(f"Checking for snap data in nfl_data_py...")
            weekly = nfl.import_weekly_data(seasons)
            
            # Check for snap-related columns
            snap_cols = [c for c in weekly.columns if 'snap' in c.lower()]
            
            if not snap_cols:
                print("  Snap data not available (this is OK - using volume metrics instead)")
                return pd.DataFrame()
            
            # Extract available snap data
            base_cols = ['player_id', 'player_display_name', 'position', 'recent_team', 
                        'season', 'week']
            available_cols = [c for c in base_cols if c in weekly.columns] + snap_cols
            
            snap_df = weekly[available_cols].copy()
            
            # Standardize column names
            rename_map = {
                'player_display_name': 'name',
                'recent_team': 'team',
            }
            # Add any snap columns found
            for col in snap_cols:
                if 'pct' in col.lower():
                    rename_map[col] = 'snap_percentage'
                elif 'snap' in col.lower():
                    rename_map[col] = 'total_snaps'
            
            snap_df = snap_df.rename(columns=rename_map)
            
            # Remove rows without snap data
            if 'total_snaps' in snap_df.columns:
                snap_df = snap_df.dropna(subset=['total_snaps'])
            
            print(f"  Got {len(snap_df)} snap count records")
            
            if not snap_df.empty:
                self.save_raw_data(snap_df, "snap_counts.csv")
            
            return snap_df
            
        except ImportError:
            print("  nfl_data_py not available, skipping snap counts")
            return pd.DataFrame()
        except Exception as e:
            print(f"  Snap data unavailable: {e}")
            return pd.DataFrame()
    
    def get_latest_data(self) -> pd.DataFrame:
        """Get current season snap data."""
        import datetime
        today = datetime.date.today()
        current_season = today.year if today.month >= 9 else today.year - 1
        return self.scrape(seasons=[current_season])

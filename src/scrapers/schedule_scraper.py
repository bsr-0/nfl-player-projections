"""NFL Schedule scraper for team matchups and strength of schedule."""
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import SCRAPER_DELAY, USER_AGENT
from src.scrapers.base_scraper import BaseScraper


class NFLScheduleScraper(BaseScraper):
    """Scrapes NFL schedule data from NFL.com."""
    
    NFL_SCHEDULE_URL = "https://www.nfl.com/schedules/{year}/REG{week}/"
    
    def scrape(self, seasons: List[int] = None, **kwargs) -> pd.DataFrame:
        """Required abstract method implementation."""
        if not seasons:
            seasons = [datetime.now().year]
        all_data = []
        for year in seasons:
            df = self.scrape_season_schedule(year)
            all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def get_latest_data(self) -> pd.DataFrame:
        """Required abstract method implementation."""
        return self.scrape_season_schedule(datetime.now().year)
    
    # Team abbreviation mappings (NFL.com uses full names sometimes)
    TEAM_ABBREV_MAP = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
        "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    }
    
    def scrape_season_schedule(self, year: int, weeks: List[int] = None) -> pd.DataFrame:
        """
        Scrape full season schedule for a given year.
        
        Args:
            year: NFL season year (e.g. current or next season)
            weeks: Specific weeks to scrape (default: 1-18)
            
        Returns:
            DataFrame with schedule data
        """
        weeks = weeks or list(range(1, 19))
        all_games = []
        
        for week in weeks:
            print(f"  Scraping week {week}...")
            games = self._scrape_week(year, week)
            all_games.extend(games)
            self._rate_limit()
        
        if not all_games:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_games)
        return df
    
    def _scrape_week(self, year: int, week: int) -> List[Dict]:
        """Scrape a single week's schedule."""
        url = self.NFL_SCHEDULE_URL.format(year=year, week=week)
        
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"    Error fetching week {week}: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        games = []
        
        # Look for embedded JSON data in script tags
        for script in soup.find_all('script'):
            if script.string and 'homeTeam' in script.string and 'awayTeam' in script.string:
                games = self._parse_json_schedule(script.string, year, week)
                if games:
                    break
        
        # Fallback: try parsing HTML structure
        if not games:
            games = self._parse_html_schedule(soup, year, week)
        
        return games
    
    def _parse_json_schedule(self, script_content: str, year: int, week: int) -> List[Dict]:
        """Parse schedule from embedded JSON."""
        games = []
        
        try:
            # Find JSON objects with game data
            # Look for patterns like {"id":"...", "homeTeam":..., "awayTeam":...}
            pattern = r'\{[^{}]*"homeTeam"[^{}]*"awayTeam"[^{}]*\}'
            
            # Try to find the main data structure
            # NFL.com often embeds data in __NEXT_DATA__ or similar
            next_data_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', 
                                        script_content, re.DOTALL)
            if next_data_match:
                data = json.loads(next_data_match.group(1))
                games = self._extract_games_from_next_data(data, year, week)
            else:
                # Try to find game objects directly
                # Look for array of games
                games_match = re.search(r'"games"\s*:\s*\[(.*?)\]', script_content, re.DOTALL)
                if games_match:
                    games_json = '[' + games_match.group(1) + ']'
                    try:
                        games_data = json.loads(games_json)
                        for game in games_data:
                            parsed = self._parse_game_object(game, year, week)
                            if parsed:
                                games.append(parsed)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"    JSON parse error: {e}")
        
        return games
    
    def _extract_games_from_next_data(self, data: dict, year: int, week: int) -> List[Dict]:
        """Extract games from Next.js data structure."""
        games = []
        
        def find_games(obj, depth=0):
            if depth > 10:
                return
            if isinstance(obj, dict):
                if 'homeTeam' in obj and 'awayTeam' in obj:
                    parsed = self._parse_game_object(obj, year, week)
                    if parsed:
                        games.append(parsed)
                else:
                    for v in obj.values():
                        find_games(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    find_games(item, depth + 1)
        
        find_games(data)
        return games
    
    def _parse_game_object(self, game: dict, year: int, week: int) -> Optional[Dict]:
        """Parse a single game object."""
        try:
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            
            # Get team abbreviations
            home_abbrev = (home_team.get('abbreviation') or 
                         home_team.get('nickName') or 
                         self._get_abbrev(home_team.get('fullName', '')))
            away_abbrev = (away_team.get('abbreviation') or 
                         away_team.get('nickName') or 
                         self._get_abbrev(away_team.get('fullName', '')))
            
            if not home_abbrev or not away_abbrev:
                return None
            
            # Parse game time
            game_time = game.get('gameTime') or game.get('time') or game.get('dateTime')
            if game_time:
                try:
                    game_datetime = pd.to_datetime(game_time)
                except:
                    game_datetime = None
            else:
                game_datetime = None
            
            return {
                'season': year,
                'week': week,
                'home_team': home_abbrev.upper(),
                'away_team': away_abbrev.upper(),
                'game_id': game.get('id', f"{year}_{week}_{away_abbrev}_{home_abbrev}"),
                'game_time': game_datetime,
                'venue': game.get('venue', {}).get('name') if isinstance(game.get('venue'), dict) else game.get('venue'),
            }
        except Exception as e:
            return None
    
    def _parse_html_schedule(self, soup: BeautifulSoup, year: int, week: int) -> List[Dict]:
        """Fallback: parse schedule from HTML structure."""
        games = []
        
        # Try various CSS selectors that NFL.com might use
        game_containers = (
            soup.select('[data-testid="game-card"]') or
            soup.select('.nfl-c-matchup-strip') or
            soup.select('.d3-o-media-object--game')
        )
        
        for container in game_containers:
            try:
                teams = container.select('.nfl-c-matchup-strip__team-name, .d3-o-club-fullname')
                if len(teams) >= 2:
                    away_team = self._get_abbrev(teams[0].get_text(strip=True))
                    home_team = self._get_abbrev(teams[1].get_text(strip=True))
                    
                    if home_team and away_team:
                        games.append({
                            'season': year,
                            'week': week,
                            'home_team': home_team,
                            'away_team': away_team,
                            'game_id': f"{year}_{week}_{away_team}_{home_team}",
                            'game_time': None,
                            'venue': None,
                        })
            except Exception:
                continue
        
        return games
    
    def _get_abbrev(self, team_name: str) -> str:
        """Convert team name to abbreviation."""
        if not team_name:
            return ""
        
        # Check direct mapping
        if team_name in self.TEAM_ABBREV_MAP:
            return self.TEAM_ABBREV_MAP[team_name]
        
        # Check if already an abbreviation
        if len(team_name) <= 3 and team_name.isupper():
            return team_name
        
        # Try partial match
        team_name_lower = team_name.lower()
        for full_name, abbrev in self.TEAM_ABBREV_MAP.items():
            if team_name_lower in full_name.lower() or full_name.lower() in team_name_lower:
                return abbrev
        
        return team_name[:3].upper()


class StrengthOfScheduleCalculator:
    """Calculates strength of schedule based on opponent historical performance."""
    
    def __init__(self, team_stats_df: pd.DataFrame = None):
        """
        Initialize with historical team stats.
        
        Args:
            team_stats_df: DataFrame with team performance data
        """
        self.team_stats = team_stats_df
        self.team_rankings = {}
    
    def calculate_team_rankings(self, season: int = None) -> Dict[str, float]:
        """
        Calculate team power rankings based on historical performance.
        
        Returns:
            Dict mapping team abbreviation to power rating (0-100)
        """
        if self.team_stats is None or self.team_stats.empty:
            # Return neutral ratings if no data
            return {team: 50.0 for team in [
                "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
                "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
                "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS"
            ]}
        
        # Filter to most recent season if specified
        if season:
            df = self.team_stats[self.team_stats['season'] == season - 1]  # Use prior year
        else:
            # Use most recent season available
            max_season = self.team_stats['season'].max()
            df = self.team_stats[self.team_stats['season'] == max_season]
        
        if df.empty:
            df = self.team_stats
        
        # Calculate power rating based on points scored/allowed, yards, etc.
        team_ratings = {}
        
        for team in df['team'].unique():
            team_data = df[df['team'] == team]
            
            # Aggregate metrics
            pts_scored = team_data['points_scored'].mean() if 'points_scored' in team_data else 20
            pts_allowed = team_data['points_allowed'].mean() if 'points_allowed' in team_data else 20
            total_yards = team_data['total_yards'].mean() if 'total_yards' in team_data else 300
            
            # Simple power rating formula
            point_diff = pts_scored - pts_allowed
            rating = 50 + (point_diff * 2) + (total_yards - 300) / 10
            rating = max(0, min(100, rating))  # Clamp to 0-100
            
            team_ratings[team] = rating
        
        self.team_rankings = team_ratings
        return team_ratings
    
    def calculate_sos(self, schedule_df: pd.DataFrame, team: str) -> Dict[str, float]:
        """
        Calculate strength of schedule for a team.
        
        Args:
            schedule_df: DataFrame with schedule (home_team, away_team, week)
            team: Team abbreviation
            
        Returns:
            Dict with SOS metrics
        """
        if not self.team_rankings:
            self.calculate_team_rankings()
        
        # Get all opponents for this team
        home_games = schedule_df[schedule_df['home_team'] == team]
        away_games = schedule_df[schedule_df['away_team'] == team]
        
        opponents = list(home_games['away_team']) + list(away_games['home_team'])
        
        if not opponents:
            return {'sos_rating': 50.0, 'sos_rank': 16, 'avg_opponent_rating': 50.0}
        
        # Calculate average opponent rating
        opponent_ratings = [self.team_rankings.get(opp, 50.0) for opp in opponents]
        avg_rating = sum(opponent_ratings) / len(opponent_ratings)
        
        return {
            'sos_rating': avg_rating,
            'avg_opponent_rating': avg_rating,
            'num_games': len(opponents),
            'toughest_opponent': max(opponents, key=lambda x: self.team_rankings.get(x, 50)),
            'easiest_opponent': min(opponents, key=lambda x: self.team_rankings.get(x, 50)),
        }
    
    def calculate_weekly_matchup_difficulty(self, schedule_df: pd.DataFrame, 
                                            team: str) -> pd.DataFrame:
        """
        Calculate matchup difficulty for each week.
        
        Args:
            schedule_df: Schedule DataFrame
            team: Team abbreviation
            
        Returns:
            DataFrame with weekly matchup ratings
        """
        if not self.team_rankings:
            self.calculate_team_rankings()
        
        results = []
        
        for _, row in schedule_df.iterrows():
            if row['home_team'] == team:
                opponent = row['away_team']
                home_away = 'home'
            elif row['away_team'] == team:
                opponent = row['home_team']
                home_away = 'away'
            else:
                continue
            
            opp_rating = self.team_rankings.get(opponent, 50.0)
            
            # Adjust for home/away (home team gets ~3 point advantage)
            matchup_difficulty = opp_rating
            if home_away == 'away':
                matchup_difficulty += 5  # Harder on the road
            
            results.append({
                'week': row['week'],
                'opponent': opponent,
                'home_away': home_away,
                'opponent_rating': opp_rating,
                'matchup_difficulty': matchup_difficulty,
            })
        
        return pd.DataFrame(results)
    
    def get_all_teams_sos(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SOS for all teams."""
        all_teams = set(schedule_df['home_team'].unique()) | set(schedule_df['away_team'].unique())
        
        results = []
        for team in all_teams:
            sos = self.calculate_sos(schedule_df, team)
            sos['team'] = team
            results.append(sos)
        
        df = pd.DataFrame(results)
        df = df.sort_values('sos_rating', ascending=False)
        df['sos_rank'] = range(1, len(df) + 1)
        
        return df


def check_schedule_availability(year: int) -> bool:
    """
    Check if schedule data is available for a given year.
    
    Returns True if data is available from any source.
    """
    # Check nfl_data_py
    try:
        import nfl_data_py as nfl
        schedule = nfl.import_schedules([year])
        if not schedule.empty and len(schedule) > 100:  # Full schedule has 272+ games
            return True
    except:
        pass
    
    # Check local CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / f"schedule_{year}.csv"
    if csv_path.exists():
        return True
    
    # Check database
    try:
        from src.utils.database import DatabaseManager
        db = DatabaseManager()
        if db.has_schedule_for_season(year):
            return True
    except:
        pass
    
    return False


def get_latest_available_season() -> int:
    """
    Get the latest season with available schedule data.
    
    Checks from current year backwards until data is found.
    """
    current_year = datetime.now().year
    # Check current year and next year (for upcoming season)
    for year in [current_year + 1, current_year, current_year - 1]:
        if check_schedule_availability(year):
            return year
    return current_year - 1  # Fallback


def get_available_seasons() -> List[int]:
    """
    Get list of all seasons with available data (through current NFL season + 1).
    
    Returns list sorted descending (newest first).
    """
    available = []
    current_year = datetime.now().year
    # NFL season runs into next calendar year; check through next year
    max_year = current_year + 1
    for year in range(max_year, 2019, -1):
        if check_schedule_availability(year):
            available.append(year)
    return sorted(available, reverse=True)


def scrape_schedule(year: int, weeks: List[int] = None) -> pd.DataFrame:
    """
    Convenience function to get schedule data.
    
    Tries multiple sources in order:
    1. nfl_data_py (most reliable when available)
    2. NFL.com scraping
    3. Local CSV file
    """
    # Try nfl_data_py first
    try:
        import nfl_data_py as nfl
        schedule = nfl.import_schedules([year])
        if not schedule.empty:
            # Normalize column names
            df = schedule[['game_id', 'season', 'week', 'home_team', 'away_team']].copy()
            df = df.rename(columns={'gameday': 'game_time'})
            print(f"  Loaded {len(df)} games from nfl_data_py")
            return df
    except Exception as e:
        print(f"  nfl_data_py not available for {year}: {e}")
    
    # Try NFL.com scraping
    scraper = NFLScheduleScraper()
    schedule = scraper.scrape_season_schedule(year, weeks)
    if not schedule.empty:
        return schedule
    
    # Check for local CSV
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / f"schedule_{year}.csv"
    if csv_path.exists():
        print(f"  Loading from local CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    print(f"  No schedule data found for {year}")
    return pd.DataFrame()


def import_schedule_to_db(year: int, csv_path: str = None) -> int:
    """
    Import schedule to database from CSV or API.
    
    Args:
        year: Season year
        csv_path: Optional path to CSV file
        
    Returns:
        Number of games imported
    """
    from src.utils.database import DatabaseManager
    db = DatabaseManager()
    
    if csv_path:
        return db.import_schedule_from_csv(csv_path)
    
    # Try to scrape
    schedule = scrape_schedule(year)
    if schedule.empty:
        return 0
    
    count = 0
    for _, row in schedule.iterrows():
        db.insert_schedule(row.to_dict())
        count += 1
    
    print(f"Imported {count} games for {year} season")
    return count


if __name__ == "__main__":
    import argparse
    
    def _default_schedule_year():
        from src.utils.nfl_calendar import get_current_nfl_season
        return get_current_nfl_season()
    
    parser = argparse.ArgumentParser(description="Scrape NFL schedule")
    parser.add_argument("--year", type=int, default=None,
                        help="Season year (default: current NFL season from nfl_calendar)")
    parser.add_argument("--weeks", type=str, default=None, help="Weeks to scrape (e.g., '1-5' or '1,2,3')")
    
    args = parser.parse_args()
    if args.year is None:
        args.year = _default_schedule_year()
    
    weeks = None
    if args.weeks:
        if '-' in args.weeks:
            start, end = args.weeks.split('-')
            weeks = list(range(int(start), int(end) + 1))
        else:
            weeks = [int(w.strip()) for w in args.weeks.split(',')]
    
    print(f"Scraping {args.year} NFL schedule...")
    schedule = scrape_schedule(args.year, weeks)
    
    if not schedule.empty:
        print(f"\nFound {len(schedule)} games")
        print(schedule.head(20))
        
        # Save to CSV
        output_path = Path(__file__).parent.parent.parent / "data" / "raw" / f"schedule_{args.year}.csv"
        schedule.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")
    else:
        print("No schedule data found")

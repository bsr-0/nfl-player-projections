"""Data scrapers for NFL statistics."""
from .player_scraper import PlayerStatsScraper
from .team_scraper import TeamStatsScraper

__all__ = ["PlayerStatsScraper", "TeamStatsScraper"]

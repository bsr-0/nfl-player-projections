"""
Quick sanity check that we can pull a private ESPN league via the API.

Usage:
    ESPN_LEAGUE_ID=123456 ESPN_YEAR=2026 ESPN_S2=... ESPN_SWID=... python scripts/test_espn_connection.py

Reads credentials from env vars so nothing sensitive ends up in shell history
or gets committed. Doesn't write anything to disk.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from integrations.espn_fantasy import ESPNFantasyConnector


def main():
    league_id = os.environ.get("ESPN_LEAGUE_ID")
    year = os.environ.get("ESPN_YEAR", "2026")
    espn_s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("ESPN_SWID")

    if not league_id:
        print("Set ESPN_LEAGUE_ID (and ESPN_S2 / ESPN_SWID for a private league).")
        sys.exit(1)

    connector = ESPNFantasyConnector(
        league_id=int(league_id),
        year=int(year),
        espn_s2=espn_s2,
        swid=swid,
    )

    if not connector.connect():
        print("Connection failed — check league ID, year, and cookies.")
        sys.exit(1)

    info = connector.get_league_info()
    print("Connected OK")
    print(f"  League: {info['name']}")
    print(f"  Teams:  {info['num_teams']}")
    print(f"  Week:   {info['current_week']}")

    print("\nTeams:")
    for team in connector.get_all_teams():
        print(f"  {team['team_id']:>2}  {team['team_name']:<25} {team['wins']}-{team['losses']}  {team['points_for']:.1f} pts")


if __name__ == "__main__":
    main()

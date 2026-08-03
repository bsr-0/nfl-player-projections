"""
ESPN sync CLI — pulls a team's roster from ESPN and writes docs/data/my_roster.json
for consumption by the lineup optimizer (docs/lineup.html).

Usage:
    python scripts/espn_sync_start_sit.py
"""

import getpass
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "my_roster.json"

# Slot values from ESPNSyncService → output JSON slot
OUTPUT_SLOT_MAP = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "FLEX": "FLEX",
    "SUPERFLEX": "FLEX",
    "BENCH": "BENCH",
    "IR": "BENCH",
}
SKIP_SLOTS = {"K", "DST"}


def prompt_credentials() -> dict:
    """Interactively collect all connection parameters from the user."""
    print("ESPN Fantasy Sync\n" + "-" * 40)

    league_id_raw = input("League ID: ").strip()
    if not league_id_raw.isdigit():
        print("Error: League ID must be an integer.")
        sys.exit(1)
    league_id = int(league_id_raw)

    year_raw = input("Season year [2026]: ").strip() or "2026"
    if not year_raw.isdigit():
        print("Error: Season year must be an integer.")
        sys.exit(1)
    year = int(year_raw)

    is_private = input("Private league? (y/n) [n]: ").strip().lower() in ("y", "yes")

    espn_s2 = None
    swid = None
    if is_private:
        espn_s2 = getpass.getpass("espn_s2 cookie: ").strip()
        swid = getpass.getpass("SWID cookie: ").strip()
        if not espn_s2 or not swid:
            print("Error: Both espn_s2 and SWID are required for private leagues.")
            sys.exit(1)

    team_input = input("Team name or ID: ").strip()
    if not team_input:
        print("Error: Team name or ID is required.")
        sys.exit(1)

    team_id = None
    team_name = None
    if team_input.isdigit():
        team_id = int(team_input)
    else:
        team_name = team_input

    return {
        "league_id": league_id,
        "year": year,
        "espn_s2": espn_s2,
        "swid": swid,
        "team_id": team_id,
        "team_name": team_name,
    }


def build_roster_json(state) -> list[dict]:
    """Map LeagueState.roster to the simple JSON format for lineup.html."""
    rows = []
    for player in state.roster:
        output_slot = OUTPUT_SLOT_MAP.get(player.slot)
        if output_slot is None:
            continue  # Skip K, DST, and anything unexpected
        rows.append(
            {
                "name": player.name,
                "position": player.position,
                "slot": output_slot,
            }
        )
    return rows


def main() -> None:
    # Guard: espn_api must be installed
    try:
        import espn_api  # noqa: F401
    except ImportError:
        print("Error: espn_api package not installed.")
        print("Install it with:  pip install espn_api")
        sys.exit(1)

    # Collect parameters
    params = prompt_credentials()
    print()

    # Connect
    from src.integrations.espn_fantasy import ESPNFantasyConnector
    from src.integrations.espn_sync import ESPNSyncService

    connector = ESPNFantasyConnector(
        league_id=params["league_id"],
        year=params["year"],
        espn_s2=params["espn_s2"],
        swid=params["swid"],
    )

    print("Connecting to ESPN…")
    if not connector.connect():
        print("Error: Could not connect to ESPN. Check your League ID and credentials.")
        sys.exit(1)

    league_info = connector.get_league_info()
    league_name = league_info.get("name", "Unknown League")
    current_week = league_info.get("current_week", "?")

    # Sync
    sync = ESPNSyncService(connector, predictor=None)
    try:
        state = sync.sync(
            team_id=params["team_id"],
            team_name=params["team_name"],
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Tip: Check the team name or ID — use a partial name or the team number.")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during sync: {exc}")
        sys.exit(1)

    # Build output
    roster_json = build_roster_json(state)

    # Write file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(roster_json, indent=2))

    # Summary
    resolved_count = sum(1 for p in state.roster if p.is_resolved())
    total_count = len(state.roster)
    unresolved_names = [
        f"{u['name']} ({u['position']})" for u in state.unresolved_players
    ]

    team_data = connector.get_my_team(
        team_name=params["team_name"], team_id=params["team_id"]
    )
    fallback = params.get("team_name") or f"Team {params.get('team_id')}"
    display_team_name = team_data.get("team_name", fallback)

    print(f"ESPN sync complete — {league_name}, Week {current_week}")
    print(f"Team: {display_team_name}")
    print(f"Resolved: {resolved_count}/{total_count} players")
    if unresolved_names:
        print(f"Unresolved: {unresolved_names}")
    print(f"Wrote {OUTPUT_PATH}")
    print()
    print("Next: open docs/lineup.html in your browser to see the optimizer.")


if __name__ == "__main__":
    main()

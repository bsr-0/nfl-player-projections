"""
fetch_news.py — Pull injury/status data from Sleeper API and write docs/data/news.json.

Matches Sleeper players against our draft board by name, outputs only players
with active injury flags.
"""

import json
import re
import sys
import tempfile
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("WARNING: 'requests' not installed. Run: pip install requests")
    sys.exit(0)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOARD_PATH = PROJECT_ROOT / "docs" / "data" / "board.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "news.json"

SLEEPER_URL = "https://api.sleeper.app/v1/players/nfl"
TIMEOUT = 15
POSITIONS = {"QB", "RB", "WR", "TE"}
FLAGGED_STATUSES = {"Questionable", "Doubtful", "Out", "IR", "PUP", "Suspended"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize(name: str) -> str:
    """Lowercase, strip punctuation and extra spaces."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def strip_suffix(norm: str) -> str:
    """Remove trailing generational suffix (jr/sr/ii/iii/iv) from a normalized name."""
    parts = norm.split()
    if parts and parts[-1] in SUFFIXES:
        return " ".join(parts[:-1])
    return norm


def first_initial_last(name: str) -> str:
    """'Ja'Marr Chase' → 'j chase'; strips generational suffixes first."""
    parts = strip_suffix(normalize(name)).split()
    if len(parts) >= 2:
        return parts[0][0] + " " + parts[-1]
    return strip_suffix(normalize(name))


def last_name(name: str) -> str:
    parts = strip_suffix(normalize(name)).split()
    return parts[-1] if parts else normalize(name)


def build_sleeper_index(sleeper_players: dict) -> tuple[dict, dict, dict]:
    """
    Build three lookup dicts for the three match tiers:
      1. normalized full name → player
      2. first-initial + last → list of players (may collide)
      3. (last_name, pos, team) → player
    """
    by_full: dict[str, dict] = {}
    by_initial_last: dict[str, list] = {}
    by_last_pos_team: dict[tuple, dict] = {}

    for pid, p in sleeper_players.items():
        if p.get("position") not in POSITIONS:
            continue

        full = p.get("full_name") or ""
        pos = p.get("position") or ""
        team = (p.get("team") or "").upper()

        norm = normalize(full)
        if norm:
            by_full[norm] = p
            # Also index without generational suffix so board names like
            # "Patrick Mahomes II" match Sleeper's "Patrick Mahomes"
            stripped = strip_suffix(norm)
            if stripped != norm and stripped not in by_full:
                by_full[stripped] = p

        il = first_initial_last(full)
        by_initial_last.setdefault(il, []).append(p)

        ln = last_name(full)
        if ln and pos and team:
            by_last_pos_team[(ln, pos, team)] = p

    return by_full, by_initial_last, by_last_pos_team


def find_sleeper_player(
    board_name: str,
    board_pos: str,
    board_team: str,
    by_full: dict,
    by_initial_last: dict,
    by_last_pos_team: dict,
) -> dict | None:
    """Try three match tiers; return the Sleeper player dict or None."""
    # Tier 1: exact normalized full name
    norm = normalize(board_name)
    if norm in by_full:
        return by_full[norm]

    # Tier 2: first-initial + last name (only if unique match)
    il = first_initial_last(board_name)
    candidates = by_initial_last.get(il, [])
    if len(candidates) == 1:
        return candidates[0]
    # If multiple candidates, narrow by position
    if len(candidates) > 1:
        pos_filtered = [c for c in candidates if c.get("position") == board_pos]
        if len(pos_filtered) == 1:
            return pos_filtered[0]

    # Tier 3: last name + position + team
    ln = last_name(board_name)
    team = board_team.upper()
    key = (ln, board_pos, team)
    if key in by_last_pos_team:
        return by_last_pos_team[key]

    return None


def best_note(sleeper_player: dict) -> str:
    """
    Build a human-readable note from Sleeper fields.

    The /v1/players/nfl endpoint doesn't carry a news array — it exposes
    injury_body_part and injury_notes instead, which are more reliable.
    """
    parts: list[str] = []
    body_part = (sleeper_player.get("injury_body_part") or "").strip()
    notes = (sleeper_player.get("injury_notes") or "").strip()
    if body_part:
        parts.append(body_part)
    if notes and notes.lower() != body_part.lower():
        parts.append(notes)
    if parts:
        return " — ".join(parts)
    return sleeper_player.get("injury_status") or ""


def main():
    # Load board
    if not BOARD_PATH.exists():
        print(f"ERROR: board.json not found at {BOARD_PATH}")
        sys.exit(1)

    with open(BOARD_PATH) as f:
        board = json.load(f)

    board_players = [p for p in board if p.get("p") in POSITIONS]

    # Fetch Sleeper
    print(f"Fetching Sleeper player data from {SLEEPER_URL} ...")
    try:
        resp = requests.get(SLEEPER_URL, timeout=TIMEOUT)
        resp.raise_for_status()
        sleeper_data = resp.json()
    except requests.exceptions.Timeout:
        print("WARNING: Sleeper API timed out. Skipping news update.")
        sys.exit(0)
    except requests.exceptions.RequestException as e:
        print(f"WARNING: Sleeper API unreachable ({e}). Skipping news update.")
        sys.exit(0)

    n_sleeper = len(sleeper_data)
    by_full, by_initial_last, by_last_pos_team = build_sleeper_index(sleeper_data)

    # Match board players and collect flags
    injured: dict[str, dict] = {}
    matched = 0

    unmatched_names: list[str] = []

    for bp in board_players:
        board_name = bp.get("n", "")
        board_pos = bp.get("p", "")
        board_team = bp.get("t", "")

        sp = find_sleeper_player(
            board_name, board_pos, board_team,
            by_full, by_initial_last, by_last_pos_team,
        )
        if sp is None:
            unmatched_names.append(f"{board_name} ({board_pos})")
            continue

        matched += 1
        status = sp.get("injury_status")
        if status in FLAGGED_STATUSES:
            note = best_note(sp)
            injured[board_name] = {
                "status": status,
                "note": note,
                "pos": board_pos,
                "team": board_team,
            }

    # Print summary
    print(f"Fetched {n_sleeper} Sleeper players, matched {matched}/{len(board_players)} "
          f"board players, {len(injured)} with injury flags")
    if unmatched_names:
        print(f"  Unmatched ({len(unmatched_names)}): {', '.join(unmatched_names)}")

    if injured:
        for name, info in injured.items():
            print(f"  {name} ({info['pos']}, {info['team']}): "
                  f"{info['status']} — {info['note'][:80]}")
    else:
        print("  No injury flags found (may be offseason).")

    # Write output
    output = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "players": injured,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write via temp file in same directory
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

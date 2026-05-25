"""Match ADP player names to player_weekly_stats player_ids.

ADP uses full names ("Josh Allen"), our DB uses abbreviated ("J.Allen").
This module provides a lookup table: (season) -> {player_id: pre_season_ecr}.
"""
import re
import sqlite3
from pathlib import Path
from typing import Dict, Optional

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "nfl_data.db"

# Team abbreviation aliases (ADP -> DB)
_TEAM_ALIASES = {"LAR": "LA", "JAC": "JAX"}


def _normalize_team(team: str) -> str:
    return _TEAM_ALIASES.get(team, team)


def _full_name_to_abbrev(full_name: str) -> str:
    """Convert 'Josh Allen' -> 'J.Allen', 'Amon-Ra St. Brown' -> 'A.St. Brown'."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return full_name
    first = parts[0]
    last = " ".join(parts[1:])
    # Strip suffixes like Jr., II, III, IV, Sr.
    last = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', last)
    return f"{first[0]}.{last}"


def get_preseason_ecr(season: int, db_path: Path = DB_PATH) -> Dict[str, float]:
    """Return {player_id: ecr} for the given season's pre-draft consensus.

    Uses the latest scrape before September 1 of the season year (pre-season
    draft board). Falls back to earliest available scrape if none pre-Sept.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get the latest pre-season scrape date (before Sept 1)
    c.execute(
        """SELECT MAX(scrape_date) FROM adp_history
           WHERE season = ? AND page_type = 'redraft-overall'
           AND scrape_date < ?""",
        (season, f"{season}-09-01"),
    )
    pre_season_date = c.fetchone()[0]

    if not pre_season_date:
        # Fall back to earliest scrape for this season
        c.execute(
            """SELECT MIN(scrape_date) FROM adp_history
               WHERE season = ? AND page_type = 'redraft-overall'""",
            (season,),
        )
        pre_season_date = c.fetchone()[0]

    if not pre_season_date:
        conn.close()
        return {}

    # Get ADP rankings for that date
    c.execute(
        """SELECT player_name, position, team, ecr FROM adp_history
           WHERE season = ? AND page_type = 'redraft-overall' AND scrape_date = ?""",
        (season, pre_season_date),
    )
    adp_rows = c.fetchall()

    # Build lookup from our DB: (abbrev_name, position, team) -> player_id
    # Primary: use the requested season's game data.
    # Fallback: when no game data exists yet (preseason), use the most recent
    # season that does have data — then overlay current team assignments from
    # nfl-data-py rosters so players who switched teams still match.
    c.execute(
        """SELECT DISTINCT p.player_id, p.name, p.position, pws.team
           FROM players p
           JOIN player_weekly_stats pws ON p.player_id = pws.player_id
           WHERE pws.season = ?""",
        (season,),
    )
    rows = c.fetchall()

    if not rows:
        # No game data for this season yet — use most recent available season.
        c.execute(
            """SELECT DISTINCT p.player_id, p.name, p.position, pws.team
               FROM players p
               JOIN player_weekly_stats pws ON p.player_id = pws.player_id
               WHERE pws.season = (SELECT MAX(season) FROM player_weekly_stats)""",
        )
        rows = c.fetchall()

        # Overlay current team assignments from nfl-data-py rosters so players
        # who moved in the offseason still match their ADP team.
        try:
            import nfl_data_py as nfl
            roster_df = nfl.import_seasonal_rosters([season])
            roster_df = roster_df[roster_df["status"] == "ACT"][["player_id", "team"]].dropna()
            team_override = dict(zip(roster_df["player_id"], roster_df["team"]))
        except Exception:
            team_override = {}

        rows = [
            (pid, name, pos, team_override.get(pid, team))
            for pid, name, pos, team in rows
        ]

    db_lookup: Dict[tuple, str] = {}
    for pid, name, pos, team in rows:
        db_lookup[(name, pos, team)] = pid
        # Also index without team for cross-team matches (trades)
        if (name, pos) not in db_lookup:
            db_lookup[(name, pos)] = pid

    conn.close()

    # Match ADP to player_ids
    result: Dict[str, float] = {}
    matched = 0
    for full_name, pos, adp_team, ecr in adp_rows:
        abbrev = _full_name_to_abbrev(full_name)
        norm_team = _normalize_team(adp_team)

        # Try exact match (name + position + team)
        pid = db_lookup.get((abbrev, pos, norm_team))
        # Fall back to name + position only (handles trades)
        if not pid:
            pid = db_lookup.get((abbrev, pos))

        if pid and pid not in result:
            result[pid] = float(ecr)
            matched += 1

    return result


def get_preseason_ecr_series(seasons: list, db_path: Path = DB_PATH) -> Dict[int, Dict[str, float]]:
    """Return {season: {player_id: ecr}} for multiple seasons."""
    return {s: get_preseason_ecr(s, db_path) for s in seasons}


if __name__ == "__main__":
    # Quick test
    for season in [2024, 2025]:
        ecr = get_preseason_ecr(season)
        print(f"\n{season}: matched {len(ecr)} players")
        # Show top 10
        top = sorted(ecr.items(), key=lambda x: x[1])[:10]
        import sqlite3 as _sql
        conn = _sql.connect(DB_PATH)
        c = conn.cursor()
        for pid, rank in top:
            c.execute("SELECT name, position FROM players WHERE player_id = ?", (pid,))
            row = c.fetchone()
            name = row[0] if row else "?"
            pos = row[1] if row else "?"
            print(f"  ECR {rank:5.1f}: {name:20s} {pos}")
        conn.close()

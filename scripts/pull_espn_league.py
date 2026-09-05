"""Pull ESPN fantasy-league data into the private directory.

WHERE THIS WRITES MATTERS. Output goes to config.ESPN_PRIVATE_DIR
(data/espn_private/), never under docs/. docs/ is the GitHub Pages root and
the site fetches relative paths beneath it, so a browser cannot reach anything
outside it -- keeping league data under data/ makes publishing it impossible
rather than merely discouraged. The predecessor script wrote
docs/data/my_roster.json, inside the published tree; the assertion in
_snapshot_dir() exists so that cannot happen again silently.

Each run writes a dated snapshot directory plus a `latest` symlink:

    data/espn_private/2026-09-04T13-05-22/
        league.json       settings: scoring rules, roster slots, playoff format
        teams.json        standings and season totals
        rosters.json      every team's roster
        free_agents.json  the unrostered pool, pulled one position at a time
        matchups.json     head-to-head schedule, one row per team per week
        manifest.json     what was pulled, when, and any failures

Snapshots are kept rather than overwritten: rosters change through the season
and ESPN does not expose history, so a pull is the only record of what a
roster looked like on a given day. The free-agent pool turns over faster than
the rosters do.

Credentials come from the environment only. A public league needs just
ESPN_LEAGUE_ID and ESPN_YEAR.

Usage:
    ESPN_LEAGUE_ID=... ESPN_YEAR=2026 python scripts/pull_espn_league.py
    python scripts/pull_espn_league.py --dry-run
    python scripts/pull_espn_league.py --fa-limit 100
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.integrations.espn_fantasy import ESPNFantasyConnector


def _snapshot_dir(stamp: str) -> Path:
    out = ESPN_PRIVATE_DIR / stamp
    # The whole safeguard is that this path is outside the published tree.
    # Assert it rather than trust it: a future edit to ESPN_PRIVATE_DIR that
    # pointed into docs/ would otherwise publish the user's roster.
    docs = PROJECT_ROOT / "docs"
    resolved = out.resolve()
    if docs.resolve() == resolved or docs.resolve() in resolved.parents:
        raise SystemExit(
            f"refusing to write league data inside the published site: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(path: Path, payload) -> int:
    # allow_nan=False: bare NaN is invalid JSON and silently breaks any reader.
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return path.stat().st_size


# ESPN's position codes, spelled as they appear in the league's roster slots.
FREE_AGENT_POSITIONS = ("QB", "RB", "WR", "TE", "K", "D/ST")


def _free_agent_positions(settings) -> list:
    """The positions this league actually starts.

    ESPN ranks the free-agent pool globally, so one unfiltered call returns
    its top N players overall and buries whole positions; pulling per position
    gives an even pool. Which positions those are comes from the league's own
    roster slots -- a league with no kicker slot should not have its pool
    padded with kickers -- and a composite slot counts for each of its
    components, so "RB/WR/TE" makes all three startable.
    """
    counts = settings.get("position_slot_counts") or {}
    slots = {slot for slot, n in counts.items() if n}
    keep = [pos for pos in FREE_AGENT_POSITIONS
            if any(pos == slot or pos in slot.split("/") for slot in slots)]
    # No recognisable slots (settings unavailable) -- pull the standard set
    # rather than silently pulling nothing.
    return keep or list(FREE_AGENT_POSITIONS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="connect and report counts without writing")
    ap.add_argument("--fa-limit", type=int, default=50, metavar="N",
                    help="free agents to pull per position (default 50)")
    args = ap.parse_args()

    league_id = os.environ.get("ESPN_LEAGUE_ID")
    year = os.environ.get("ESPN_YEAR")
    espn_s2 = os.environ.get("ESPN_S2")
    swid = os.environ.get("ESPN_SWID")

    missing = [n for n, v in (("ESPN_LEAGUE_ID", league_id),
                              ("ESPN_YEAR", year)) if not v]
    if missing:
        print(f"Set {' and '.join(missing)}"
              " (plus ESPN_S2 / ESPN_SWID for a private league).")
        return 1

    connector = ESPNFantasyConnector(league_id=int(league_id), year=int(year),
                                     espn_s2=espn_s2, swid=swid)
    auth = "authenticated" if (espn_s2 and swid) else "anonymous (public league)"
    print(f"Connecting to league {league_id} for {year} — {auth}")
    if not connector.connect():
        print("Connection failed — check league ID, year, and cookies.")
        return 1

    info = connector.get_league_info()
    print(f"Connected: {info.get('name')} — {info.get('num_teams')} teams, "
          f"week {info.get('current_week')}")

    teams = connector.get_all_teams()
    settings = connector.get_league_settings()
    matchups = connector.get_matchups()

    # Rosters are pulled per team. A team that errors should not lose the
    # whole snapshot, so failures are recorded and the rest still writes.
    rosters, failures = [], []
    for t in teams:
        try:
            team = connector.get_my_team(team_id=t["team_id"])
            if "error" in team:
                failures.append({"team_id": t["team_id"], "error": team["error"]})
            else:
                rosters.append(team)
        except Exception as e:                       # noqa: BLE001
            failures.append({"team_id": t["team_id"], "error": str(e)})

    # Same contract as the roster loop: one position failing costs that
    # position, not the snapshot. get_free_agents raises rather than returning
    # [], so a failed request is recorded instead of being written as an
    # empty pool.
    free_agents, fa_failures = [], []
    for pos in _free_agent_positions(settings):
        try:
            free_agents.extend(
                connector.get_free_agents(position=pos, limit=args.fa_limit))
        except Exception as e:                       # noqa: BLE001
            fa_failures.append({"position": pos, "error": str(e)})

    fa_by_position = dict(Counter(p.get("position") for p in free_agents))
    drafted = sum(len(r.get("roster", [])) for r in rosters)
    print(f"  settings    : {len(settings.get('scoring_format', []))} scoring "
          f"rules, slots {settings.get('position_slot_counts')}")
    print(f"  teams       : {len(teams)}")
    print(f"  rosters     : {len(rosters)} teams, {drafted} players")
    print(f"  free agents : {len(free_agents)} -> {fa_by_position}")
    print(f"  matchups    : {len(matchups)} team-weeks")
    if failures:
        print(f"  roster failures      : {len(failures)} -> {failures}")
    if fa_failures:
        print(f"  free-agent failures  : {len(fa_failures)} -> {fa_failures}")

    if drafted == 0:
        print("\n  NOTE: every roster is empty. Either the league has not "
              "drafted yet, or the connection lacks access to rosters.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = _snapshot_dir(stamp)
    written = {
        "league.json": _write(out / "league.json",
                              {"info": info, "settings": settings}),
        "teams.json": _write(out / "teams.json", teams),
        "rosters.json": _write(out / "rosters.json", rosters),
        "free_agents.json": _write(out / "free_agents.json", free_agents),
        "matchups.json": _write(out / "matchups.json", matchups),
    }
    _write(out / "manifest.json", {
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
        "league_id": int(league_id),
        "season": int(year),
        "authenticated": bool(espn_s2 and swid),
        "counts": {"teams": len(teams), "rosters": len(rosters),
                   "rostered_players": drafted, "matchup_rows": len(matchups),
                   "free_agents": len(free_agents),
                   "scoring_rules": len(settings.get("scoring_format", []))},
        "free_agents_by_position": fa_by_position,
        "free_agent_limit_per_position": args.fa_limit,
        "failures": failures,
        "free_agent_failures": fa_failures,
    })

    latest = ESPN_PRIVATE_DIR / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)

    print(f"\nWrote {out.relative_to(PROJECT_ROOT)}/")
    for name, size in written.items():
        print(f"  {name:<15} {size:>8,} bytes")
    print(f"  latest -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

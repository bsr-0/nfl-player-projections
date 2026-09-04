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
        league.json     settings: scoring rules, roster slots, playoff format
        teams.json      standings and season totals
        rosters.json    every team's roster
        matchups.json   head-to-head schedule, one row per team per week
        manifest.json   what was pulled, when, and any failures

Snapshots are kept rather than overwritten: rosters change through the season
and ESPN does not expose history, so a pull is the only record of what a
roster looked like on a given day.

Credentials come from the environment only. A public league needs just
ESPN_LEAGUE_ID and ESPN_YEAR.

Usage:
    ESPN_LEAGUE_ID=... ESPN_YEAR=2026 python scripts/pull_espn_league.py
    python scripts/pull_espn_league.py --dry-run
"""
import argparse
import json
import os
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="connect and report counts without writing")
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

    drafted = sum(len(r.get("roster", [])) for r in rosters)
    print(f"  settings : {len(settings.get('scoring_format', []))} scoring rules, "
          f"slots {settings.get('position_slot_counts')}")
    print(f"  teams    : {len(teams)}")
    print(f"  rosters  : {len(rosters)} teams, {drafted} players")
    print(f"  matchups : {len(matchups)} team-weeks")
    if failures:
        print(f"  failures : {len(failures)} -> {failures}")

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
        "matchups.json": _write(out / "matchups.json", matchups),
    }
    _write(out / "manifest.json", {
        "pulled_at": datetime.now().isoformat(timespec="seconds"),
        "league_id": int(league_id),
        "season": int(year),
        "authenticated": bool(espn_s2 and swid),
        "counts": {"teams": len(teams), "rosters": len(rosters),
                   "rostered_players": drafted, "matchup_rows": len(matchups),
                   "scoring_rules": len(settings.get("scoring_format", []))},
        "failures": failures,
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

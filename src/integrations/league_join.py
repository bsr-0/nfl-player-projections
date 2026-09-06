"""Join an ESPN league snapshot to this project's projections.

Two id systems meet here. ESPN keys players by its own numeric playerId; the
board keys them by GSIS id. nflverse's id map crosses the two and covers the
whole snapshot -- measured 2026-09-05, 346 skill players, none missing from
the map -- so the id join is the mechanism and name matching is only the
fallback. That order matters: "A.Smith" is two different receivers on this
board, so a name that is not unique has to resolve to nothing rather than to a
coin flip.

WHICH NUMBER IS "THE PROJECTION" is not decided here. This module reads
docs/data/weekly_meta.json and serves whatever the pipeline is publishing:

    season_prorated   before kickoff. The weekly file holds the season total
                      over 17, with each week's real opponent and byes removed
                      -- a pace, deliberately carrying no interval.
    weekly_model      once games are played. The same file, same shape, but
                      real per-week predictions with 80% intervals attached.

`generate_weekly_data.py` flips that mode on its own the first time a game has
been played, so nothing here has to change when it does. The mode travels with
the rows as `projection_mode`, and `week_ci_low`/`week_ci_high` populate
themselves when intervals start being published -- a report reading these rows
picks up the real weekly model automatically and can say which it is showing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DATA_DIR, ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.utils.player_names import board_name

# The positions this project models. A league also starts K and D/ST, which
# have no projection here and must be reported as such rather than as zero.
POSITIONS = ("QB", "RB", "WR", "TE")

WEEKLY_DIR = PROJECT_ROOT / "docs" / "data"

# ESPN's team codes where they differ from the board's.
ESPN_TEAM_CODES = {"LAR": "LA", "WSH": "WAS"}

BOARD_COLUMNS = {
    "player_id": "player_id",
    "name": "name",
    "team": "team",
    "position": "position",
    "projection_points_total": "season_total",
    "projection_points_per_game": "season_ppg",
    "projection_floor": "floor",
    "projection_ceiling": "ceiling",
    "projection_source": "source",
    "support_class": "support_class",
    "risk_score": "risk_score",
    "prev_season_games": "prev_season_games",
}

# What a matched player carries over from the board.
PROJECTION_COLUMNS = (
    "player_id", "name", "season_total", "season_ppg", "week_points",
    "week_ci_low", "week_ci_high", "floor", "ceiling", "risk_score",
    "support_class", "source", "opponent", "home_away", "on_bye",
    "projection_mode", "prev_season_games", "measured_mae",
)


@dataclass(frozen=True)
class Snapshot:
    """One `pull_espn_league.py` run, read back."""

    path: Path
    info: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    teams: list = field(default_factory=list)
    rosters: list = field(default_factory=list)
    free_agents: list = field(default_factory=list)
    matchups: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    @property
    def season(self) -> int:
        return int(self.manifest.get("season") or 0)

    @property
    def week(self) -> int:
        """The league's own scoring period, which is ESPN's answer.

        Not the calendar's: deciding the week from dates is what produced the
        2026-09-03 failure where the code believed week 1 had been played.
        """
        return int(self.info.get("current_week") or 1)

    def rostered(self) -> list:
        """Every rostered player, tagged with the team that owns him."""
        return [dict(p, fantasy_team=t.get("team_name"),
                     fantasy_team_id=t.get("team_id"))
                for t in self.rosters for p in t.get("roster", [])]

    def opponent_for(self, team_id: int, week: Optional[int] = None) -> dict:
        """The matchup row for one team in one week, or {}."""
        week = self.week if week is None else week
        for row in self.matchups:
            if row.get("team_id") == team_id and row.get("week") == week:
                return row
        return {}


def load_snapshot(path: Optional[Path] = None) -> Snapshot:
    path = Path(path) if path else ESPN_PRIVATE_DIR / "latest"
    if not path.exists():
        raise FileNotFoundError(
            f"no league snapshot at {path} -- run scripts/pull_espn_league.py")

    def read(name, default):
        f = path / name
        return json.loads(f.read_text()) if f.exists() else default

    league = read("league.json", {})
    return Snapshot(
        path=path.resolve(),
        info=league.get("info", {}),
        settings=league.get("settings", {}),
        teams=read("teams.json", []),
        rosters=read("rosters.json", []),
        free_agents=read("free_agents.json", []),
        matchups=read("matchups.json", []),
        manifest=read("manifest.json", {}),
    )


def load_board() -> pd.DataFrame:
    """The season board, from the canonical data/players_*.json."""
    rows = []
    for pos in POSITIONS:
        f = DATA_DIR / f"players_{pos}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text()))
    if not rows:
        return pd.DataFrame(columns=list(BOARD_COLUMNS.values()))
    df = pd.DataFrame(rows)
    have = {k: v for k, v in BOARD_COLUMNS.items() if k in df.columns}
    return df[list(have)].rename(columns=have)


def load_week(season: int, week: int):
    """One week's published payload and the meta that says what it is."""
    meta_path = WEEKLY_DIR / "weekly_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    f = WEEKLY_DIR / f"weekly_{season}_wk{week}.json"
    if not f.exists():
        return pd.DataFrame(), meta
    return pd.DataFrame(json.loads(f.read_text())), meta


def load_projections(season: int, week: int, board: Optional[pd.DataFrame] = None,
                     weekly: Optional[pd.DataFrame] = None,
                     meta: Optional[dict] = None) -> pd.DataFrame:
    """Board rows carrying week `week`'s number, opponent and interval.

    `board`, `weekly` and `meta` are injectable so the join can be tested
    without the published payloads on disk.
    """
    board = load_board() if board is None else board.copy()
    if weekly is None and meta is None:
        weekly, meta = load_week(season, week)
    meta = meta or {}
    board["projection_mode"] = meta.get("mode")
    # The pipeline publishes its own measured error per position. Carrying it
    # per row is what lets the report state a spread instead of implying the
    # point estimate is exact.
    board["measured_mae"] = board["position"].map(
        {pos: m.get("mae") for pos, m in (meta.get("measured") or {}).items()})

    for col in ("week_points", "week_ci_low", "week_ci_high", "opponent",
                "home_away"):
        board[col] = pd.NA
    if weekly is None or weekly.empty:
        board["on_bye"] = pd.NA
        return board

    rename = {"predicted_points": "week_points",
              "prediction_ci80_lower": "week_ci_low",
              "prediction_ci80_upper": "week_ci_high"}
    keep = ["name", "position", "team", "opponent", "home_away"]
    keep += [c for c in rename if c in weekly.columns]
    merged = board.drop(columns=["week_points", "week_ci_low", "week_ci_high",
                                 "opponent", "home_away"]).merge(
        weekly[[c for c in keep if c in weekly.columns]].rename(columns=rename),
        on=["name", "position", "team"], how="left")
    for col in ("week_points", "week_ci_low", "week_ci_high"):
        if col not in merged.columns:
            merged[col] = pd.NA

    # generate_weekly_data.py drops players on bye from the week's file, so a
    # projected player missing from it is on bye -- distinct from a player who
    # has no projection at all, which is why season_total gates this.
    merged["on_bye"] = merged["season_total"].notna() & merged["week_points"].isna()
    return merged


@lru_cache(maxsize=1)
def espn_id_to_gsis() -> dict:
    """ESPN's numeric playerId -> GSIS id, from nflverse's id map."""
    try:
        import nfl_data_py as nfl
        ids = nfl.import_ids()
    except Exception as e:                           # noqa: BLE001
        print(f"  nflverse id map unavailable ({e}); matching on names only")
        return {}
    ids = ids.dropna(subset=["espn_id", "gsis_id"])
    return {str(int(e)): str(g) for e, g in zip(ids["espn_id"], ids["gsis_id"])}


def join_players(players, projections: pd.DataFrame,
                 crosswalk: Optional[dict] = None) -> pd.DataFrame:
    """Attach a projection to each ESPN player, or say why there isn't one.

    Every player keeps a row. An unmatched one carries null projection columns
    and an `unmatched_reason`, because a report has to be able to say "no
    projection" -- writing 0 for unknown is how fabricated numbers get in.
    """
    crosswalk = espn_id_to_gsis() if crosswalk is None else crosswalk

    ranked = projections.sort_values("season_total", ascending=False,
                                     na_position="last")
    # A player traded mid-season has two board rows (the aggregate groups by
    # team). Same player, same projection -- keep the more complete row.
    # drop=False: the matched row has to keep carrying its own id.
    by_id = ranked.drop_duplicates("player_id").set_index("player_id",
                                                          drop=False)
    ranked = ranked.assign(
        _key=list(zip(ranked["name"], ranked["position"], ranked["team"])))
    counts = ranked["_key"].value_counts()
    # A dict, not an index: pandas reads .loc[("A.Smith", "WR", "DAL")] on an
    # index of tuples as three axes and raises.
    solo = ranked[ranked["_key"].map(counts).eq(1)]
    by_name = {key: row for key, (_, row) in zip(solo["_key"], solo.iterrows())}
    ambiguous = set(counts[counts > 1].index)

    out = []
    for p in players:
        team = ESPN_TEAM_CODES.get(p.get("team"), p.get("team"))
        rec = {
            "espn_player_id": p.get("player_id"),
            "espn_name": p.get("name"),
            "position": p.get("position"),
            "nfl_team": team,
            "lineup_slot": p.get("lineup_slot"),
            "injury_status": p.get("injury_status"),
            "percent_owned": p.get("percent_owned"),
            "espn_projected_avg": p.get("projected_avg_points"),
            "fantasy_team": p.get("fantasy_team"),
            "match_method": None,
            "unmatched_reason": None,
        }
        row = None
        if p.get("position") not in POSITIONS:
            rec["unmatched_reason"] = "position not modelled"
        else:
            gsis = crosswalk.get(str(p.get("player_id")))
            key = (board_name(p.get("name")), p.get("position"), team)
            if gsis is not None and gsis in by_id.index:
                row, rec["match_method"] = by_id.loc[gsis], "espn_id"
            elif key in by_name:
                row, rec["match_method"] = by_name[key], "name_team_pos"
            elif key in ambiguous:
                rec["unmatched_reason"] = "name is not unique on the board"
            else:
                rec["unmatched_reason"] = "not on the board"

        if row is not None:
            rec.update({c: row[c] for c in PROJECTION_COLUMNS if c in row.index})
        out.append(rec)
    return pd.DataFrame(out)


def match_report(joined: pd.DataFrame) -> dict:
    """Match rate and, more usefully, exactly who did not match and why."""
    if joined.empty:
        return {"players": 0, "matched": 0, "by_method": {}, "unmatched": []}
    matched = joined["match_method"].notna()
    unmatched = joined[~matched]
    return {
        "players": int(len(joined)),
        "matched": int(matched.sum()),
        "by_method": joined.loc[matched, "match_method"].value_counts().to_dict(),
        "by_reason": unmatched["unmatched_reason"].value_counts().to_dict(),
        "unmatched": unmatched[["espn_name", "position", "nfl_team",
                                "unmatched_reason"]].to_dict("records"),
    }

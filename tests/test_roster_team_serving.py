"""A player who changed clubs since his last game is served on his roster club.

`latest_data` is each player's last COMPLETED game, so its team was the old
club for anyone traded or signed since; opponent, home/away and team-matchup
features of the upcoming game were then read for the wrong club (57 of 696
listed players in 2025 week 6, GAPS.md 2026-10-08).
"""
import pandas as pd
import pytest

from src.predict import NFLPredictor
from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=tmp_path / "t.db")


def _seed(db, table, rows, with_status=True):
    cols = "season INTEGER, week INTEGER, team TEXT, player_id TEXT" + (", status TEXT" if with_status else "")
    with db._get_connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"CREATE TABLE {table} ({cols})")
        marks = "?,?,?,?" + (",?" if with_status else "")
        conn.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        conn.commit()


def test_most_recent_snapshot_wins_across_tables_not_table_priority(db):
    # weekly_rosters_v2 stops at week 1; weekly_rosters has the later move.
    _seed(db, "weekly_rosters_v2", [(2026, 1, "SEA", "p", "ACT")])
    _seed(db, "weekly_rosters", [(2026, 1, "SEA", "p", "ACT"), (2026, 5, "PHI", "p", "ACT")])
    assert db.get_roster_team_map()["p"] == "PHI"


def test_as_of_ignores_the_target_week_and_later(db):
    _seed(db, "weekly_rosters", [(2025, 5, "SEA", "p", "ACT"), (2025, 6, "PHI", "p", "ACT"),
                                 (2025, 9, "DAL", "p", "ACT")])
    assert db.get_roster_team_map(as_of=(2025, 6))["p"] == "SEA"
    assert db.get_roster_team_map(as_of=(2025, 7))["p"] == "PHI"
    assert db.get_roster_team_map(as_of=(2026, 1))["p"] == "DAL"
    assert "p" not in db.get_roster_team_map(as_of=(2025, 5))


@pytest.mark.parametrize("status", ["CUT", "RET", "TRD", "UFA"])
def test_a_snapshot_without_a_club_is_skipped(db, status):
    _seed(db, "weekly_rosters", [(2025, 5, "SEA", "p", "ACT"), (2025, 6, "PHI", "p", status)])
    assert db.get_roster_team_map()["p"] == "SEA"


def test_practice_squad_and_reserve_still_name_a_club(db):
    _seed(db, "weekly_rosters", [(2025, 5, "SEA", "a", "DEV"), (2025, 5, "SEA", "b", "RES"),
                                 (2025, 5, "SEA", "c", "INA")])
    assert db.get_roster_team_map() == {"a": "SEA", "b": "SEA", "c": "SEA"}


def test_tables_without_a_status_column_are_still_read(db):
    _seed(db, "weekly_rosters", [(2025, 5, "SEA", "p")], with_status=False)
    assert db.get_roster_team_map()["p"] == "SEA"


def test_blank_team_and_missing_tables_are_ignored(db):
    _seed(db, "weekly_rosters", [(2025, 5, "", "p", "ACT")])
    assert db.get_roster_team_map() == {}
    assert DatabaseManager(db_path=db.db_path.with_name("empty.db")).get_roster_team_map() == {}


class _Db:
    def __init__(self, teams):
        self.teams, self.seen = teams, []

    def get_roster_team_map(self, as_of=None):
        self.seen.append(as_of)
        return self.teams


def _predictor(teams):
    shell = NFLPredictor.__new__(NFLPredictor)
    shell.db = _Db(teams)
    return shell


def _latest():
    return pd.DataFrame({"player_id": ["mover", "stayer", "unknown", "legacy", "washington"],
                         "team": ["BAL", "KC", "DEN", "NE", "WAS"]})


def test_movers_are_repointed_and_everyone_else_is_untouched(capsys):
    shell = _predictor({"mover": "WAS", "stayer": "KC", "legacy": "ARZ"})
    out = shell._apply_roster_teams(_latest(), as_of=(2025, 6))
    assert out.set_index("player_id").team.to_dict() == {
        "mover": "WAS", "stayer": "KC", "unknown": "DEN",
        "legacy": "NE", "washington": "WAS"}, "a club not in the frame (legacy abbreviation) must not be introduced"
    assert shell.db.seen == [(2025, 6)], "the replay cutoff must reach the roster lookup"


def test_the_input_frame_is_not_mutated():
    frame = _latest()
    _predictor({"mover": "WAS"})._apply_roster_teams(frame, as_of=None)
    assert frame.set_index("player_id").team["mover"] == "BAL"


def test_no_moves_is_a_noop():
    out = _predictor({"stayer": "KC"})._apply_roster_teams(_latest(), as_of=None)
    assert out.equals(_latest())

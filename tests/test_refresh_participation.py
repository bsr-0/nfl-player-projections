import sqlite3

import pandas as pd

from scripts.refresh_participation import COLUMNS, compare, rows_to_write, upsert


def _con(last_week=3):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE player_weekly_stats (season INT, week INT)")
    con.executemany("INSERT INTO player_weekly_stats VALUES (2026, ?)", [(w,) for w in range(1, last_week + 1)])
    con.execute("CREATE TABLE team_personnel_stats (id INTEGER PRIMARY KEY AUTOINCREMENT, team TEXT, season INT, "
                "week INT, pct_11 REAL, pct_12 REAL, pct_21 REAL, pct_13 REAL, pct_other REAL, n_plays INT, "
                "UNIQUE(team, season, week))")
    return con


def _personnel(rows):
    return pd.DataFrame(rows, columns=COLUMNS["team_personnel_stats"])


def test_only_played_weeks_and_whole_games_are_written():
    d = _personnel([("KC", 2026, 0, .6, .2, .1, .05, .05, 60), ("KC", 2026, 2, .6, .2, .1, .05, .05, 64),
                    ("KC", 2026, 3, .6, .2, .1, .05, .05, 12), ("KC", 2026, 4, .6, .2, .1, .05, .05, 61),
                    ("KC", 2025, 2, .6, .2, .1, .05, .05, 61)])
    rows, info = rows_to_write(_con(last_week=3), "team_personnel_stats", d, 2026)
    assert rows.week.tolist() == [2]
    assert info == {"derived": 4, "after_week_filter": 2, "left_out_short": 1, "last_week": 3}


def test_upserting_twice_replaces_rows_instead_of_duplicating_them():
    con = _con()
    first = _personnel([("KC", 2026, 1, .6, .2, .1, .05, .05, 60), ("BUF", 2026, 1, .7, .1, .1, .05, .05, 62)])
    upsert(con, "team_personnel_stats", first)
    revised = first.assign(pct_11=[.65, .7], n_plays=[61, 62])
    upsert(con, "team_personnel_stats", revised)
    got = con.execute("SELECT team, pct_11, n_plays FROM team_personnel_stats ORDER BY team").fetchall()
    assert got == [("BUF", .7, 62), ("KC", .65, 61)]


def test_compare_separates_missing_keys_from_revised_values():
    stored = _personnel([("KC", 2025, 1, .6, .2, .1, .05, .05, 60), ("BUF", 2025, 1, .7, .1, .1, .05, .05, 62)])
    derived = _personnel([("KC", 2025, 1, .6, .2, .1, .05, .05, 61), ("NYJ", 2025, 1, .7, .1, .1, .05, .05, 62)])
    r = compare(stored, derived, "team_personnel_stats")
    assert (r["shared"], r["only_stored"], r["only_derived"]) == (1, 1, 1)
    assert r["differing_cells"] == {"n_plays": 1}

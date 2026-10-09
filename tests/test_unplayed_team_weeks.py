"""Roster rows of games not yet played must not enter the share panel.

canonical_player_weeks gets a team-week's rows as soon as weekly_rosters lists
them (2026 week 5 had 757 on 2026-10-08, before its games). Built as played
weeks they are zero-volume games and drag the next week's lags toward zero.
"""
import sqlite3

import pandas as pd
import pytest

from scripts.build_prekickoff_share_rows import history
from scripts.build_team_week_player_shares import build_shares, drop_unplayed_team_weeks


@pytest.fixture
def con(tmp_path):
    c = sqlite3.connect(str(tmp_path / "t.db"))
    c.executescript("""
        CREATE TABLE schedule (season INT, week INT, home_team TEXT, away_team TEXT,
                               home_score INT, away_score INT);
        CREATE TABLE canonical_player_weeks (player_id TEXT, season INT, week INT, team TEXT, position TEXT);
        CREATE TABLE player_weekly_stats (player_id TEXT, season INT, week INT, targets INT,
                                          receptions INT, receiving_yards INT, rushing_attempts INT,
                                          rushing_yards INT, passing_attempts INT);
    """)
    return c


def _add(con, season, week, scores, stats_for):
    """Teams AAA vs BBB play; each has one WR. `scores` None = not played."""
    h, a = scores if scores else (None, None)
    con.execute("INSERT INTO schedule VALUES (?,?,?,?,?,?)", (season, week, "AAA", "BBB", h, a))
    for team, pid in (("AAA", "a1"), ("BBB", "b1")):
        con.execute("INSERT INTO canonical_player_weeks VALUES (?,?,?,?,?)", (pid, season, week, team, "WR"))
        if pid in stats_for:
            con.execute("INSERT INTO player_weekly_stats VALUES (?,?,?,?,?,?,?,?,?)",
                        (pid, season, week, 8, 5, 60, 0, 0, 0))


def _frames(con):
    pop = pd.read_sql("SELECT * FROM canonical_player_weeks", con)
    vol = pd.read_sql("SELECT player_id, season, week FROM player_weekly_stats", con)
    return pop, vol


def test_a_played_week_is_kept(con):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    pop, vol = _frames(con)
    kept, dropped = drop_unplayed_team_weeks(con, pop, vol)
    assert len(kept) == 2 and dropped == []


def test_a_scheduled_unscored_week_without_stats_is_dropped(con):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    _add(con, 2025, 2, None, set())
    pop, vol = _frames(con)
    kept, dropped = drop_unplayed_team_weeks(con, pop, vol)
    assert set(kept.week) == {1}
    assert dropped == [(2025, 2, "AAA"), (2025, 2, "BBB")]


def test_a_final_score_with_no_stats_is_a_data_gap_and_raises(con):
    _add(con, 2025, 1, (24, 20), {"a1"})   # BBB's stats never loaded
    pop, vol = _frames(con)
    with pytest.raises(ValueError, match="final score but no player stats"):
        drop_unplayed_team_weeks(con, pop, vol)


def test_a_bye_is_left_alone(con):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    con.execute("INSERT INTO canonical_player_weeks VALUES ('c1', 2025, 1, 'CCC', 'WR')")  # no game
    pop, vol = _frames(con)
    kept, dropped = drop_unplayed_team_weeks(con, pop, vol)
    assert "c1" in set(kept.player_id) and dropped == []


def test_a_started_week_with_stats_is_not_dropped_even_without_a_score(con):
    _add(con, 2025, 1, None, {"a1", "b1"})   # stats loaded before the schedule's scores
    pop, vol = _frames(con)
    kept, dropped = drop_unplayed_team_weeks(con, pop, vol)
    assert len(kept) == 2 and dropped == []


def test_the_share_table_build_leaves_unplayed_weeks_out_and_lags_are_unharmed(con, capsys):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    _add(con, 2025, 2, (24, 20), {"a1", "b1"})
    _add(con, 2025, 3, None, set())
    played_only = build_shares(con, 2025, 2025)
    assert set(played_only.week) == {1, 2}
    assert "unplayed team-weeks" in capsys.readouterr().out
    # Week 2's lag is week 1's share, whether or not week 3's roster rows exist.
    assert played_only[(played_only.week == 2) & (played_only.player_id == "a1")] \
        .share_of_team_targets_s2d.iloc[0] == 1.0


def test_forecasting_a_week_whose_predecessor_has_no_stats_refuses(con):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    _add(con, 2025, 2, None, set())          # rosters loaded, stats not
    with pytest.raises(ValueError, match="week 2 has no stats yet"):
        history(con, 2025, 3)


def test_forecasting_the_first_unplayed_week_is_fine(con):
    _add(con, 2025, 1, (24, 20), {"a1", "b1"})
    _add(con, 2025, 2, None, set())
    pop, vol = history(con, 2025, 2)         # week 1 is played; week 2 is the target
    assert set(pop.week) == {1}

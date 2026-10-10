import sqlite3

from scripts.check_live_inputs import LIMITS, check

TEAMS = [f"T{i:02d}" for i in range(32)]


def _db(zero_weeks=(), zero_share=1.0, with_tables=True, draft_ids=True):
    con = sqlite3.connect(":memory:")
    cols = ", ".join(f"{c} REAL" for c in LIMITS)
    con.execute(f"CREATE TABLE player_weekly_stats (season INT, week INT, team TEXT, player_id TEXT, {cols})")
    for week in (1, 2, 3):
        for i, team in enumerate(TEAMS):
            broken = week in zero_weeks and i < zero_share * len(TEAMS)
            v = 0.0 if broken else 5.0
            con.executemany(
                f"INSERT INTO player_weekly_stats VALUES (2026, ?, ?, ?, {', '.join('?' * len(LIMITS))})",
                [(week, team, f"{team}-{k}", *([v] * len(LIMITS))) for k in range(2)])
    if with_tables:
        for t in ("weekly_pfr", "snap_counts", "ngs_passing", "ngs_receiving", "ngs_rushing"):
            con.execute(f"CREATE TABLE {t} (season INT, week INT)")
            con.executemany(f"INSERT INTO {t} VALUES (2026, ?)", [(1,), (2,), (3,)])
    con.execute("CREATE TABLE draft_picks_v2 (player_id TEXT, draft_season INT)")
    con.executemany("INSERT INTO draft_picks_v2 VALUES (?, 2026)",
                    [("00-0000001" if draft_ids else "ABC123456",)] * 10)
    return con


def test_healthy_inputs_pass_with_no_warnings():
    r = check(_db(), 2026)
    assert r["failures"] == [] and r["warnings"] == [] and r["weeks"] == [1, 2, 3]


def test_a_week_where_every_team_reads_zero_fails_and_names_the_week():
    r = check(_db(zero_weeks=(2,)), 2026)
    assert len(r["failures"]) == 1 and "week 2" in r["failures"][0]
    assert "neutral_targets 100%" in r["failures"][0] and "redzone_targets 100%" in r["failures"][0]


def test_a_partly_broken_week_fails_at_the_column_limit():
    assert check(_db(zero_weeks=(3,), zero_share=0.20), 2026)["failures"] == []
    assert len(check(_db(zero_weeks=(3,), zero_share=0.30), 2026)["failures"]) == 1


def test_redzone_targets_tolerates_more_zero_teams_than_the_stable_columns():
    r = check(_db(zero_weeks=(1,), zero_share=0.40), 2026)
    assert "redzone_targets" not in r["failures"][0] and "neutral_targets" in r["failures"][0]


def test_missing_weekly_tables_and_placeholder_draft_ids_only_warn():
    r = check(_db(with_tables=False, draft_ids=False), 2026)
    assert r["failures"] == []
    assert any("weekly_pfr" in w for w in r["warnings"]) and any("official id" in w for w in r["warnings"])


def test_a_newest_week_missing_from_a_table_warns_but_does_not_fail():
    con = _db()
    con.execute("DELETE FROM snap_counts WHERE week = 3")
    r = check(con, 2026)
    assert r["failures"] == [] and r["warnings"] == ["snap_counts has no 2026 week 3 rows"]


def test_a_season_with_no_rows_fails():
    assert check(_db(), 2031)["failures"] == ["no 2031 rows in player_weekly_stats"]

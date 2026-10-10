import sqlite3

import pandas as pd

from scripts.refresh_draft_identity import apply_updates, is_gsis, plan_updates

KEY = ["draft_season", "draft_round", "draft_pick"]


def _local(rows):
    df = pd.DataFrame(rows, columns=KEY + ["player_id", "position", "pfr_player_id"])
    df.insert(0, "rowid", range(1, len(df) + 1))
    return df


def _up(rows):
    return pd.DataFrame(rows, columns=KEY + ["gsis_id", "position", "pfr_player_id"])


def test_gsis_form_is_strict():
    assert is_gsis("00-0041562")
    assert not any(is_gsis(v) for v in ("MEN516487", "", None, "00-123", float("nan")))


def test_a_placeholder_and_a_missing_id_are_replaced_by_the_official_id():
    local = _local([(2026, 1, 1, "MEN516487", "QB", "MendFe00"), (2026, 2, 33, None, "WR", "StriDe01")])
    up = _up([(2026, 1, 1, "00-0041562", "QB", "MendFe00"), (2026, 2, 33, "00-0041026", "WR", "StriDe01")])
    updates, skipped = plan_updates(local, up)
    assert dict(zip(updates.rowid, updates.new_id)) == {1: "00-0041562", 2: "00-0041026"}
    assert updates.old_id.tolist()[0] == "MEN516487" and pd.isna(updates.old_id.tolist()[1])
    assert sum(skipped.values()) == 0


def test_an_existing_gsis_id_is_kept_and_a_different_one_is_reported_not_applied():
    local = _local([(2025, 1, 1, "00-0000001", "QB", "A"), (2025, 1, 2, "00-0000002", "WR", "B")])
    up = _up([(2025, 1, 1, "00-0000001", "QB", "A"), (2025, 1, 2, "00-0000009", "WR", "B")])
    updates, skipped = plan_updates(local, up)
    assert updates.empty and skipped["already_gsis"] == 1 and skipped["conflict"] == 1


def test_a_pick_whose_position_or_pfr_id_disagrees_is_left_alone():
    local = _local([(2026, 1, 1, "X1", "QB", "MendFe00"), (2026, 1, 2, "X2", "WR", "TateCa00")])
    up = _up([(2026, 1, 1, "00-0000001", "RB", "MendFe00"), (2026, 1, 2, "00-0000002", "WR", "OtheOt00")])
    updates, skipped = plan_updates(local, up)
    assert updates.empty and skipped["identity_mismatch"] == 2


def test_a_pick_without_an_official_id_upstream_keeps_what_it_has():
    local = _local([(2026, 7, 245, "MIL303300", "RB", "MillJa03"), (2026, 7, 246, None, "WR", "ZZ")])
    up = _up([(2026, 7, 245, None, "RB", "MillJa03"), (2026, 7, 246, "MEN516487", "WR", "ZZ")])
    updates, skipped = plan_updates(local, up)
    assert updates.empty and skipped["no_gsis_upstream"] == 2


def test_picks_missing_upstream_are_counted_and_an_id_is_never_assigned_twice():
    local = _local([(2026, 1, 1, "P", "QB", None), (2026, 1, 2, "Q", "QB", None), (2026, 1, 3, "R", "QB", None)])
    up = _up([(2026, 1, 1, "00-0000001", "QB", None), (2026, 1, 2, "00-0000001", "QB", None)])
    updates, skipped = plan_updates(local, up)
    assert updates.rowid.tolist() == [1]
    assert skipped["id_in_use"] == 1 and skipped["no_upstream_row"] == 1


def test_an_id_already_held_by_another_pick_is_not_reassigned():
    local = _local([(2025, 1, 1, "00-0000001", "QB", None), (2026, 1, 1, "P", "QB", None)])
    up = _up([(2025, 1, 1, "00-0000001", "QB", None), (2026, 1, 1, "00-0000001", "QB", None)])
    updates, skipped = plan_updates(local, up)
    assert updates.empty and skipped["id_in_use"] == 1


def test_apply_changes_only_the_planned_rows_and_a_second_plan_is_empty():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE draft_picks_v2 (player_id TEXT, position TEXT, draft_season INT, "
                "draft_round INT, draft_pick INT, pfr_player_id TEXT)")
    con.executemany("INSERT INTO draft_picks_v2 VALUES (?,?,?,?,?,?)", [
        ("MEN516487", "QB", 2026, 1, 1, "MendFe00"), (None, "WR", 2026, 2, 33, "StriDe01"),
        ("00-0000007", "RB", 2026, 1, 3, "LoveJe00")])
    up = _up([(2026, 1, 1, "00-0041562", "QB", "MendFe00"), (2026, 2, 33, "00-0041026", "WR", "StriDe01"),
              (2026, 1, 3, "00-0099999", "RB", "LoveJe00")])
    local = pd.read_sql("SELECT rowid, * FROM draft_picks_v2", con)
    updates, skipped = plan_updates(local, up)
    assert apply_updates(con, updates) == 2 and skipped["conflict"] == 1
    got = dict(con.execute("SELECT draft_pick, player_id FROM draft_picks_v2").fetchall())
    assert got == {1: "00-0041562", 33: "00-0041026", 3: "00-0000007"}
    again, _ = plan_updates(pd.read_sql("SELECT rowid, * FROM draft_picks_v2", con), up)
    assert again.empty


def test_the_update_statement_never_overwrites_a_gsis_id_even_if_planned_wrongly():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE draft_picks_v2 (player_id TEXT)")
    con.execute("INSERT INTO draft_picks_v2 VALUES ('00-0000007')")
    wrong = pd.DataFrame({"rowid": [1], "new_id": ["00-0000008"]})
    assert apply_updates(con, wrong) == 0
    assert con.execute("SELECT player_id FROM draft_picks_v2").fetchone()[0] == "00-0000007"

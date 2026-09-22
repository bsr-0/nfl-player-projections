"""Correctness + leakage invariants for scripts/build_team_week_roster_slots.py.

Slot uniqueness is the one property Plan B's whole data shape depends on;
the tie-break order (depth_chart_rank, then lagged snap_share, then
player_id) must be deterministic and must never let a same-week value leak
into the ordering. Also regression-covers the DB_PATH/cache gotcha found
while wiring this up (see load_depth_chart_rank_asof's docstring).
"""
import sqlite3

import pandas as pd
import pytest

from scripts.build_team_week_roster_slots import (
    MAX_SLOTS_PER_POSITION,
    audit_slot_coverage,
    build_roster_slots,
    load_depth_chart_rank_asof,
    validate_roster_slots,
)


@pytest.fixture(autouse=True)
def _restore_shared_state():
    """load_depth_chart_rank_asof intentionally mutates config.settings.DB_PATH
    as a real, permanent side effect for CLI use (see its docstring) -- fine
    for a one-shot process, but every test in this file calls it, and pytest
    runs the whole suite in one process. Without this, a test here could
    leave config.settings.DB_PATH pointed at an already-deleted tmp_path
    file, breaking any later-running test elsewhere in the suite that reads
    it (tests/test_depth_chart_asof.py and others touch the same lookup)."""
    import config.settings as settings
    from src.features import feature_engineering

    original = settings.DB_PATH
    yield
    settings.DB_PATH = original
    feature_engineering._depth_chart_asof_cache.clear()
from src.utils.database import DatabaseManager

SEASON = 2021
WEEKS = [1, 2, 3, 4]


def _seed(db_path, players, depth_chart=None):
    """`players`: list of (player_id, position) on team AAA every week.
    `depth_chart`: optional {player_id: rank} applied every week; players
    not listed get no depth_charts row at all (falls back to the default)."""
    db = DatabaseManager(db_path=db_path)
    con = sqlite3.connect(str(db_path))
    rows = [
        {"player_id": pid, "season": SEASON, "week": wk, "team": "AAA", "position": pos}
        for wk in WEEKS for pid, pos in players
    ]
    pd.DataFrame(rows).to_sql("canonical_player_weeks", con, index=False, if_exists="replace")

    dc_rows = []
    if depth_chart:
        for wk in WEEKS:
            for pid, rank in depth_chart.items():
                dc_rows.append({"season": SEASON, "week": wk, "gsis_id": pid, "depth_team": rank})
    pd.DataFrame(dc_rows, columns=["season", "week", "gsis_id", "depth_team"]).to_sql(
        "depth_charts", con, index=False, if_exists="replace"
    )
    con.close()
    return db


def _add_snap_shares(db, snap_shares_by_week):
    """`snap_shares_by_week`: {player_id: {week: snap_share}}."""
    for pid, by_week in snap_shares_by_week.items():
        for wk, share in by_week.items():
            db.insert_player_weekly_stats({
                "player_id": pid, "season": SEASON, "week": wk, "team": "AAA", "snap_share": share,
            })


def _build(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        return build_roster_slots(con, SEASON, SEASON, db_path)
    finally:
        con.close()


def test_slot_order_follows_depth_chart_rank(tmp_path):
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB"), ("c", "RB")], depth_chart={"a": 1, "b": 2, "c": 3})
    _add_snap_shares(db, {"a": {}, "b": {}, "c": {}})
    panel = _build(db_path)
    validate_roster_slots(panel)

    week1 = panel[panel.week == 1].set_index("player_id")
    assert week1.loc["a", "slot"] == "RB1"
    assert week1.loc["b", "slot"] == "RB2"
    assert week1.loc["c", "slot"] == "RB3"


def test_tiebreak_uses_lagged_snap_share_when_depth_chart_ties(tmp_path):
    """Both players have NO depth_charts row (both default to rank 3) --
    order must fall back to lagged snap_share, higher share = better slot."""
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB")])  # no depth_chart arg -> both default
    _add_snap_shares(db, {"a": {1: 0.2, 2: 0.2, 3: 0.2}, "b": {1: 0.8, 2: 0.8, 3: 0.8}})
    panel = _build(db_path)
    validate_roster_slots(panel)

    week4 = panel[panel.week == 4].set_index("player_id")  # week 4 has 3 prior weeks of history
    assert week4.loc["b", "slot"] == "RB1"  # higher lagged snap share wins
    assert week4.loc["a", "slot"] == "RB2"


def test_tiebreak_falls_back_to_player_id_when_fully_tied(tmp_path):
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("z_player", "RB"), ("a_player", "RB")])  # both default rank, no snap history
    _add_snap_shares(db, {"z_player": {}, "a_player": {}})
    panel = _build(db_path)
    validate_roster_slots(panel)

    week1 = panel[panel.week == 1].set_index("player_id")
    assert week1.loc["a_player", "slot"] == "RB1"  # alphabetically first wins the final tiebreak
    assert week1.loc["z_player", "slot"] == "RB2"


def test_sentinel_snap_share_spike_does_not_leak_into_its_own_week(tmp_path):
    """A snap-share spike recorded in week 3 must not affect week 3's own
    tiebreak ordering (shift(1) excludes the current row) -- only week 4's."""
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB")])
    _add_snap_shares(db, {
        "a": {1: 0.3, 2: 0.3, 3: 0.99, 4: 0.3},  # spike at week 3
        "b": {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
    })
    panel = _build(db_path)
    validate_roster_slots(panel)

    week3 = panel[panel.week == 3].set_index("player_id")
    # a's spike is THIS week's value -- its lagged feature (from weeks 1-2)
    # is still 0.3, well below b's steady 0.5, so b must still be RB1.
    assert week3.loc["b", "slot"] == "RB1"
    assert week3.loc["a", "snap_share_s2d"] == pytest.approx(0.3)

    week4 = panel[panel.week == 4].set_index("player_id")
    # week 4's lag now includes week 3's spike -- a's average over weeks 1-3
    # ((0.3+0.3+0.99)/3 ~= 0.53) edges out b's steady 0.5.
    assert week4.loc["a", "slot"] == "RB1"


def test_cap_enforcement_drops_excess_players(tmp_path):
    db_path = tmp_path / "test.db"
    cap = MAX_SLOTS_PER_POSITION["RB"]
    players = [(f"rb{i}", "RB") for i in range(cap + 2)]
    depth_chart = {pid: i + 1 for i, (pid, _) in enumerate(players)}
    db = _seed(db_path, players, depth_chart=depth_chart)
    _add_snap_shares(db, {pid: {} for pid, _ in players})
    panel = _build(db_path)
    validate_roster_slots(panel)

    week1 = panel[panel.week == 1]
    assert len(week1[week1.position == "RB"]) == cap
    assert set(week1[week1.position == "RB"]["slot_rank"]) == set(range(1, cap + 1))


def test_validate_rejects_duplicate_slot(tmp_path):
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB")], depth_chart={"a": 1, "b": 2})
    _add_snap_shares(db, {"a": {}, "b": {}})
    panel = _build(db_path)
    dup = pd.concat([panel, panel.iloc[[0]].assign(player_id="c")], ignore_index=True)
    with pytest.raises(ValueError, match="slot uniqueness violated"):
        validate_roster_slots(dup)


def test_validate_rejects_player_in_two_slots_same_week(tmp_path):
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB")], depth_chart={"a": 1, "b": 2})
    _add_snap_shares(db, {"a": {}, "b": {}})
    panel = _build(db_path)
    # Assign row 0 (player "a", slot RB1, week 1) to an otherwise-UNUSED
    # slot (RB4 -- only 2 RBs exist in this fixture) so this violates ONLY
    # the "one player, one slot" invariant, not also the "one slot, one
    # player" invariant the previous test already covers (reassigning to
    # RB2, already held by "b", would trip that check first and never
    # reach the one this test means to isolate).
    dup = pd.concat([panel, panel.iloc[[0]].assign(slot="RB4", slot_rank=4)], ignore_index=True)
    with pytest.raises(ValueError, match="more than one slot"):
        validate_roster_slots(dup)


def test_audit_coverage_reports_default_rank_share(tmp_path):
    db_path = tmp_path / "test.db"
    db = _seed(db_path, [("a", "RB"), ("b", "RB")], depth_chart={"a": 1})  # b has no depth_charts row
    _add_snap_shares(db, {"a": {}, "b": {}})
    panel = _build(db_path)
    coverage = audit_slot_coverage(panel)

    rb2_row = coverage[coverage.slot == "RB2"].iloc[0]
    assert rb2_row["n_filled"] == len(WEEKS)
    assert rb2_row["pct_default_depth_chart_rank"] == pytest.approx(1.0)  # b always defaults


def test_db_path_override_is_actually_respected_not_a_stale_cache(tmp_path):
    """Regression test for the gotcha found wiring this up:
    _load_depth_chart_asof_table ignores its caller's connection and reads
    config.settings.DB_PATH directly, with a process-level cache. Calling
    load_depth_chart_rank_asof against two DIFFERENT db files back to back
    (same process) must return DIFFERENT results, not a stale first-call
    cache."""
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    _seed(db_a, [("x", "RB")], depth_chart={"x": 1})
    _seed(db_b, [("x", "RB")], depth_chart={"x": 2})

    pop = pd.DataFrame([{"player_id": "x", "season": SEASON, "week": 1, "team": "AAA", "position": "RB"}])
    rank_a = load_depth_chart_rank_asof(db_a, pop).iloc[0]
    rank_b = load_depth_chart_rank_asof(db_b, pop).iloc[0]
    assert rank_a == 1
    assert rank_b == 2


def _seed_weekly(db_path, weekly_players, weekly_depth_chart=None):
    """`weekly_players`: {week: [(player_id, position), ...]} -- lets a
    week have a DIFFERENT (possibly smaller) roster than another week, the
    same shape a real bye week or an injury-driven absence from the
    canonical population produces upstream (see
    build_canonical_player_weeks.py's schedule inner-join -- a team with no
    scheduled game that week has no row there at all, which is inherited
    here, not re-implemented). `weekly_depth_chart`: {week: {player_id:
    rank}} -- lets depth-chart rank change week to week, e.g. a real
    in-season promotion."""
    db = DatabaseManager(db_path=db_path)
    con = sqlite3.connect(str(db_path))
    rows = [
        {"player_id": pid, "season": SEASON, "week": wk, "team": "AAA", "position": pos}
        for wk, players in weekly_players.items() for pid, pos in players
    ]
    pd.DataFrame(rows).to_sql("canonical_player_weeks", con, index=False, if_exists="replace")

    dc_rows = []
    if weekly_depth_chart:
        for wk, ranks in weekly_depth_chart.items():
            for pid, rank in ranks.items():
                dc_rows.append({"season": SEASON, "week": wk, "gsis_id": pid, "depth_team": rank})
    pd.DataFrame(dc_rows, columns=["season", "week", "gsis_id", "depth_team"]).to_sql(
        "depth_charts", con, index=False, if_exists="replace"
    )
    con.close()
    return db


def test_bye_week_produces_no_slot_rows_at_all(tmp_path):
    """A bye week isn't a row with empty/zero slots -- it's the ABSENCE of
    any row, inherited structurally from canonical_player_weeks (which
    itself only has rows for games actually scheduled). Confirms this
    builder doesn't accidentally fabricate a phantom team-week for a week
    the team had no game, e.g. via some later merge silently reintroducing
    every week in a range regardless of population."""
    db_path = tmp_path / "test.db"
    # Week 3 is the "bye" -- simply absent from every player's population.
    weekly_players = {
        1: [("a", "RB"), ("b", "RB")],
        2: [("a", "RB"), ("b", "RB")],
        4: [("a", "RB"), ("b", "RB")],
    }
    db = _seed_weekly(db_path, weekly_players, {wk: {"a": 1, "b": 2} for wk in (1, 2, 4)})
    _add_snap_shares(db, {"a": {}, "b": {}})
    con = sqlite3.connect(str(db_path))
    try:
        panel = build_roster_slots(con, SEASON, SEASON, db_path)
    finally:
        con.close()
    validate_roster_slots(panel)

    assert set(panel["week"].unique()) == {1, 2, 4}
    assert 3 not in panel["week"].values


def test_depth_chart_promotion_mid_season_flips_slots_at_the_right_week(tmp_path):
    """Player "b" is promoted from RB2 to RB1 starting week 3. The slot
    flip must land EXACTLY at week 3 -- not bleed backward into weeks 1-2
    (the as-of lookup must not look ahead) and not lag behind into week 4
    only (the as-of lookup must not be stale once the new snapshot exists)."""
    db_path = tmp_path / "test.db"
    weekly_players = {wk: [("a", "RB"), ("b", "RB")] for wk in WEEKS}
    weekly_depth_chart = {
        1: {"a": 1, "b": 2},
        2: {"a": 1, "b": 2},
        3: {"a": 2, "b": 1},  # promotion takes effect
        4: {"a": 2, "b": 1},
    }
    db = _seed_weekly(db_path, weekly_players, weekly_depth_chart)
    _add_snap_shares(db, {"a": {}, "b": {}})
    con = sqlite3.connect(str(db_path))
    try:
        panel = build_roster_slots(con, SEASON, SEASON, db_path)
    finally:
        con.close()
    validate_roster_slots(panel)

    by_week = panel.set_index(["week", "player_id"])["slot"]
    assert by_week[(1, "a")] == "RB1" and by_week[(1, "b")] == "RB2"
    assert by_week[(2, "a")] == "RB1" and by_week[(2, "b")] == "RB2"
    assert by_week[(3, "a")] == "RB2" and by_week[(3, "b")] == "RB1"
    assert by_week[(4, "a")] == "RB2" and by_week[(4, "b")] == "RB1"

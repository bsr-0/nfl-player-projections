"""Smoke tests for scripts/paper_trade_lock.py — the re-council Step 5 harness."""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the module as a script (it has main() but the individual
# functions are importable).
import importlib.util
spec = importlib.util.spec_from_file_location(
    "paper_trade_lock", PROJECT_ROOT / "scripts" / "paper_trade_lock.py"
)
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)


def _make_db(path: Path) -> sqlite3.Connection:
    """Create a minimal schema that the harness needs: paper_trade_entries,
    player_weekly_stats, players."""
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE paper_trade_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL, week INTEGER NOT NULL,
            lock_timestamp TEXT NOT NULL, lock_git_sha TEXT NOT NULL,
            model_config_json TEXT NOT NULL, model_lineup_json TEXT NOT NULL,
            opponent_lineup_json TEXT NOT NULL, opponent_method TEXT NOT NULL,
            notional_entry_usd REAL NOT NULL, notional_payout_multiplier REAL NOT NULL,
            model_actual REAL, opponent_actual REAL, won INTEGER,
            score_timestamp TEXT, notes TEXT,
            UNIQUE(season, week)
        );
        CREATE TABLE players (player_id TEXT PRIMARY KEY, name TEXT, position TEXT);
        CREATE TABLE player_weekly_stats (
            player_id TEXT, season INTEGER, week INTEGER,
            team TEXT, fantasy_points REAL
        );
        CREATE TABLE weekly_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL, season INTEGER NOT NULL,
            week INTEGER NOT NULL, team TEXT, position TEXT,
            status TEXT, status_description_abbr TEXT,
            full_name TEXT, game_type TEXT,
            UNIQUE(player_id, season, week)
        );
        """
    )
    conn.commit()
    return conn


def _populate_fixture(conn: sqlite3.Connection, season: int = 2026) -> None:
    """Seed a tiny roster of players and 3 weeks of history."""
    players = [
        ("QB1", "QB_Alpha", "QB"), ("QB2", "QB_Beta", "QB"),
        ("RB1", "RB_Alpha", "RB"), ("RB2", "RB_Beta", "RB"), ("RB3", "RB_Gamma", "RB"),
        ("WR1", "WR_Alpha", "WR"), ("WR2", "WR_Beta", "WR"), ("WR3", "WR_Gamma", "WR"),
        ("TE1", "TE_Alpha", "TE"), ("TE2", "TE_Beta", "TE"),
    ]
    conn.executemany("INSERT INTO players VALUES (?,?,?)", players)
    # 3 weeks of history so week 4 has a cumulative pool + a prior week
    pws_rows = []
    for week in (1, 2, 3):
        for pid, _, _ in players:
            # Deterministic points: week + hash of player id
            pts = (week * 3.0) + (hash(pid) % 15)
            pws_rows.append((pid, season, week, "TEAM", pts))
    conn.executemany("INSERT INTO player_weekly_stats VALUES (?,?,?,?,?)", pws_rows)
    # Seed weekly_rosters for week 4 — all players ACT by default.  Individual
    # tests can override to test the filter.
    roster_rows = [
        (pid, season, 4, "TEAM", pos, "ACT", "A01", name, "REG")
        for pid, name, pos in players
    ]
    conn.executemany(
        "INSERT INTO weekly_rosters (player_id, season, week, team, position, "
        "status, status_description_abbr, full_name, game_type) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        roster_rows,
    )
    conn.commit()


def _write_predictions_csv(path: Path, season: int = 2026, week: int = 4) -> None:
    rows = [
        {"season": season, "week": week, "player_id": "QB1", "name": "QB_Alpha",
         "position": "QB", "team": "A", "predicted": 20.5},
        {"season": season, "week": week, "player_id": "QB2", "name": "QB_Beta",
         "position": "QB", "team": "B", "predicted": 15.0},
        {"season": season, "week": week, "player_id": "RB1", "name": "RB_Alpha",
         "position": "RB", "team": "A", "predicted": 18.0},
        {"season": season, "week": week, "player_id": "RB2", "name": "RB_Beta",
         "position": "RB", "team": "B", "predicted": 16.0},
        {"season": season, "week": week, "player_id": "RB3", "name": "RB_Gamma",
         "position": "RB", "team": "C", "predicted": 14.0},
        {"season": season, "week": week, "player_id": "WR1", "name": "WR_Alpha",
         "position": "WR", "team": "A", "predicted": 17.0},
        {"season": season, "week": week, "player_id": "WR2", "name": "WR_Beta",
         "position": "WR", "team": "B", "predicted": 15.0},
        {"season": season, "week": week, "player_id": "WR3", "name": "WR_Gamma",
         "position": "WR", "team": "C", "predicted": 13.0},
        {"season": season, "week": week, "player_id": "TE1", "name": "TE_Alpha",
         "position": "TE", "team": "A", "predicted": 12.0},
        {"season": season, "week": week, "player_id": "TE2", "name": "TE_Beta",
         "position": "TE", "team": "B", "predicted": 9.0},
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_model_lineup_picks_top_n_per_position():
    preds = [
        {"player_id": "QB1", "name": "a", "position": "QB", "team": "A", "predicted": 20.0},
        {"player_id": "QB2", "name": "b", "position": "QB", "team": "A", "predicted": 10.0},
        {"player_id": "RB1", "name": "c", "position": "RB", "team": "A", "predicted": 18.0},
        {"player_id": "RB2", "name": "d", "position": "RB", "team": "A", "predicted": 16.0},
        {"player_id": "RB3", "name": "e", "position": "RB", "team": "A", "predicted": 5.0},
        {"player_id": "WR1", "name": "f", "position": "WR", "team": "A", "predicted": 15.0},
        {"player_id": "WR2", "name": "g", "position": "WR", "team": "A", "predicted": 14.0},
        {"player_id": "TE1", "name": "h", "position": "TE", "team": "A", "predicted": 11.0},
    ]
    picks = pt.build_model_lineup(preds)
    assert len(picks) == 6  # QB:1 + RB:2 + WR:2 + TE:1
    picked_ids = {p["player_id"] for p in picks}
    # Top-1 QB: QB1 (20 > 10)
    assert "QB1" in picked_ids and "QB2" not in picked_ids
    # Top-2 RB: RB1, RB2 (18, 16 beat 5)
    assert "RB1" in picked_ids and "RB2" in picked_ids and "RB3" not in picked_ids
    # Top-2 WR: WR1, WR2
    assert "WR1" in picked_ids and "WR2" in picked_ids
    # Top-1 TE: TE1
    assert "TE1" in picked_ids


def test_lock_then_score_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    _populate_fixture(conn)
    preds_csv = tmp_path / "preds.csv"
    _write_predictions_csv(preds_csv)

    row_id = pt.lock_week(conn, season=2026, week=4, predictions_csv=preds_csv)
    assert row_id >= 1

    # Locked row should have the 6-player model + opponent lineups and NULL actuals.
    row = conn.execute(
        "SELECT model_lineup_json, opponent_lineup_json, model_actual, opponent_actual, won "
        "FROM paper_trade_entries WHERE id=?", (row_id,),
    ).fetchone()
    model_lineup = json.loads(row[0])
    opp_lineup = json.loads(row[1])
    assert len(model_lineup) == 6
    assert len(opp_lineup) == 6
    assert row[2] is None  # model_actual unset until score
    assert row[3] is None
    assert row[4] is None

    # Seed week-4 actuals for every player, then score.
    conn.executemany(
        "INSERT INTO player_weekly_stats VALUES (?,?,?,?,?)",
        [(p["player_id"], 2026, 4, "TEAM", 10.0 + (hash(p["player_id"]) % 5))
         for p in model_lineup + opp_lineup],
    )
    conn.commit()

    result = pt.score_week(conn, season=2026, week=4)
    assert result["season"] == 2026 and result["week"] == 4
    assert result["model_actual"] > 0
    assert result["opponent_actual"] > 0

    # Row should now be fully populated.
    row = conn.execute(
        "SELECT model_actual, opponent_actual, won, score_timestamp "
        "FROM paper_trade_entries WHERE id=?", (row_id,),
    ).fetchone()
    assert row[0] == result["model_actual"]
    assert row[1] == result["opponent_actual"]
    assert row[2] in (0, 1)
    assert row[3] is not None


def test_lock_refuses_missing_predictions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    _populate_fixture(conn)
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("season,week,player_id,name,position,team,predicted\n")

    with pytest.raises(SystemExit):
        pt.lock_week(conn, season=2026, week=4, predictions_csv=empty_csv)


def test_score_refuses_unknown_week(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    _populate_fixture(conn)

    with pytest.raises(SystemExit):
        pt.score_week(conn, season=2026, week=99)


def test_active_filter_drops_inactive_players(tmp_path):
    """build_model_lineup filters to status='ACT' when active_ids is passed."""
    preds = [
        {"player_id": "QB1", "name": "a", "position": "QB", "team": "A", "predicted": 30.0},
        {"player_id": "QB2", "name": "b", "position": "QB", "team": "A", "predicted": 10.0},
        {"player_id": "RB1", "name": "c", "position": "RB", "team": "A", "predicted": 25.0},
        {"player_id": "RB2", "name": "d", "position": "RB", "team": "A", "predicted": 20.0},
        {"player_id": "RB3", "name": "e", "position": "RB", "team": "A", "predicted": 18.0},
        {"player_id": "WR1", "name": "f", "position": "WR", "team": "A", "predicted": 15.0},
        {"player_id": "WR2", "name": "g", "position": "WR", "team": "A", "predicted": 14.0},
        {"player_id": "TE1", "name": "h", "position": "TE", "team": "A", "predicted": 11.0},
    ]
    # QB1 and RB1 are top picks but marked inactive — the filter must drop
    # them even though their predictions are highest.
    active_ids = {"QB2", "RB2", "RB3", "WR1", "WR2", "TE1"}

    picks = pt.build_model_lineup(preds, active_ids=active_ids)
    picked_ids = {p["player_id"] for p in picks}

    assert "QB1" not in picked_ids, "QB1 was inactive; filter should drop"
    assert "RB1" not in picked_ids, "RB1 was inactive; filter should drop"
    assert "QB2" in picked_ids, "QB2 is the only active QB — must be picked"
    assert {"RB2", "RB3"}.issubset(picked_ids), "RB2/RB3 are the only active RBs"


def test_load_active_roster_ids_returns_only_act_rows(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    conn.executemany(
        "INSERT INTO weekly_rosters (player_id, season, week, status) "
        "VALUES (?,?,?,?)",
        [
            ("A1", 2026, 1, "ACT"),
            ("A2", 2026, 1, "ACT"),
            ("I1", 2026, 1, "INA"),
            ("R1", 2026, 1, "RES"),
            ("C1", 2026, 1, "CUT"),
            ("A3", 2026, 2, "ACT"),
        ],
    )
    conn.commit()

    week1 = pt.load_active_roster_ids(conn, 2026, 1)
    assert week1 == {"A1", "A2"}

    week2 = pt.load_active_roster_ids(conn, 2026, 2)
    assert week2 == {"A3"}

    week3 = pt.load_active_roster_ids(conn, 2026, 3)
    assert week3 == set()  # empty set, not a crash


def test_lock_refuses_without_roster_cache_when_filter_on(tmp_path):
    """If --no-active-filter is NOT passed but weekly_rosters has no data,
    lock_week must refuse rather than silently disabling the gate."""
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    _populate_fixture(conn)
    # Wipe the ACT rows to simulate a missing-cache scenario.
    conn.execute("DELETE FROM weekly_rosters WHERE season=2026 AND week=4")
    conn.commit()
    preds_csv = tmp_path / "preds.csv"
    _write_predictions_csv(preds_csv)

    with pytest.raises(SystemExit):
        pt.lock_week(conn, season=2026, week=4, predictions_csv=preds_csv)

    # With --no-active-filter the lock should succeed even without cache.
    row_id = pt.lock_week(
        conn, season=2026, week=4, predictions_csv=preds_csv,
        use_active_filter=False,
    )
    assert row_id >= 1


def test_lock_with_filter_drops_inactive_from_lineup(tmp_path):
    db_path = tmp_path / "test.db"
    conn = _make_db(db_path)
    _populate_fixture(conn)
    # Mark QB1 (the top predicted QB per the fixture CSV) as inactive —
    # the filter should promote QB2 to the model's lineup.
    conn.execute(
        "UPDATE weekly_rosters SET status='INA' WHERE player_id='QB1' AND season=2026 AND week=4"
    )
    conn.commit()
    preds_csv = tmp_path / "preds.csv"
    _write_predictions_csv(preds_csv)

    row_id = pt.lock_week(conn, season=2026, week=4, predictions_csv=preds_csv)
    row = conn.execute(
        "SELECT model_lineup_json FROM paper_trade_entries WHERE id=?", (row_id,),
    ).fetchone()
    model_lineup = json.loads(row[0])
    qb_pick = next(p for p in model_lineup if p["position"] == "QB")

    assert qb_pick["player_id"] == "QB2", (
        "active-roster filter should have promoted QB2 after QB1 was marked INA"
    )

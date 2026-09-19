import sqlite3

import pandas as pd
import pytest

from scripts.build_canonical_player_weeks import audit_panel, validate_panel


def _base():
    return pd.DataFrame([
        {
            "player_id":"p1","season":2025,"week":1,"team":"KC","opponent":"LV","position":"QB",
            "participation_state":"confirmed_played","offense_snaps":40,
            "has_stats_row":1,"fantasy_points":12.0,
        },
        {
            "player_id":"p2","season":2025,"week":1,"team":"KC","opponent":"LV","position":"RB",
            "participation_state":"confirmed_zero_snaps","offense_snaps":0,
            "has_stats_row":0,"fantasy_points":pd.NA,
        },
        {
            "player_id":"p3","season":2025,"week":1,"team":"KC","opponent":"LV","position":"WR",
            "participation_state":"unknown","offense_snaps":pd.NA,
            "has_stats_row":0,"fantasy_points":pd.NA,
        },
    ])


def test_three_participation_states_validate():
    validate_panel(_base())


def test_missing_stats_row_cannot_receive_fabricated_ppr():
    df = _base()
    df.loc[df.player_id.eq("p2"), "fantasy_points"] = 0.0
    with pytest.raises(ValueError, match="fabricated fantasy_points"):
        validate_panel(df)


def test_confirmed_played_requires_positive_snaps():
    df = _base()
    df.loc[df.player_id.eq("p1"), "offense_snaps"] = 0
    with pytest.raises(ValueError, match="confirmed_played"):
        validate_panel(df)


def test_confirmed_zero_requires_exact_zero():
    df = _base()
    df.loc[df.player_id.eq("p2"), "offense_snaps"] = 1
    with pytest.raises(ValueError, match="confirmed_zero_snaps"):
        validate_panel(df)


def test_player_week_key_is_unique():
    df = pd.concat([_base(), _base().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(df)


def _full_base():
    df = _base().copy()
    df["has_snap_row"] = [1, 1, 0]
    return df


def test_missing_position_fails_acceptance():
    df = _full_base()
    df.loc[df.player_id.eq("p2"), "position"] = pd.NA
    with pytest.raises(ValueError, match="missing fantasy position"):
        validate_panel(df)


def test_missing_schedule_identity_fails_acceptance():
    df = _full_base()
    df.loc[df.player_id.eq("p1"), "opponent"] = pd.NA
    with pytest.raises(ValueError, match="team/opponent"):
        validate_panel(df)


def test_audit_rejects_altered_existing_ppr_target():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE player_weekly_stats (
               player_id TEXT, season INTEGER, week INTEGER, team TEXT,
               position TEXT, fantasy_points REAL, snap_count INTEGER,
               snap_share REAL
           )"""
    )
    conn.execute(
        "INSERT INTO player_weekly_stats VALUES (?,?,?,?,?,?,?,?)",
        ("p1", 2025, 1, "KC", "QB", 12.0, 40, 0.6),
    )
    df = _full_base().iloc[[0]].copy()
    df["fantasy_points"] = 13.0
    df["missing_stats_row"] = 0
    df["played_without_stats_row"] = 0
    df["zero_snaps_without_stats_row"] = 0
    with pytest.raises(ValueError, match="altered"):
        audit_panel(conn, df, 2025, 2025)
    conn.close()


def test_mapped_snap_row_cannot_be_unknown():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE player_weekly_stats (
               player_id TEXT, season INTEGER, week INTEGER, team TEXT,
               position TEXT, fantasy_points REAL, snap_count INTEGER,
               snap_share REAL
           )"""
    )
    conn.execute(
        "INSERT INTO player_weekly_stats VALUES (?,?,?,?,?,?,?,?)",
        ("p1", 2025, 1, "KC", "QB", 12.0, 40, 0.6),
    )
    df = _full_base().iloc[[0]].copy()
    df["participation_state"] = "unknown"
    df["offense_snaps"] = pd.NA
    df["missing_stats_row"] = 0
    df["played_without_stats_row"] = 0
    df["zero_snaps_without_stats_row"] = 0
    with pytest.raises(ValueError, match="classified as unknown"):
        audit_panel(conn, df, 2025, 2025)
    conn.close()


def test_stats_only_nonfantasy_positions_are_excluded_but_missing_positions_fail():
    from scripts.build_canonical_player_weeks import build_panel
    with sqlite3.connect(":memory:") as conn:
        pd.DataFrame({"season": [2025], "week": [1], "home_team": ["KC"],
                      "away_team": ["LV"]}).to_sql("schedule", conn, index=False)
        pd.DataFrame({"player_id": ["qb", "kicker", "defense", "missing"],
                      "position": ["QB", "K", "DST", None]}).to_sql("players", conn, index=False)
        pd.DataFrame({"player_id": ["qb", "kicker", "defense", "missing"],
                      "season": [2025]*4, "week": [1]*4, "team": ["KC"]*4,
                      "fantasy_points": [10., 3., 5., 1.]}).to_sql("player_weekly_stats", conn, index=False)
        panel = build_panel(conn, 2025, 2025)
        assert set(panel.player_id) == {"qb", "missing"}
        with pytest.raises(ValueError, match="missing fantasy position"):
            validate_panel(panel)


def test_missing_id_mapping_fails_before_snap_labels_are_lost(monkeypatch):
    from scripts import build_canonical_player_weeks as builder
    with sqlite3.connect(":memory:") as conn:
        pd.DataFrame({"season": [2025], "week": [1], "team": ["KC"],
                      "position": ["QB"], "game_type": ["REG"], "pfr_player_id": ["pfr"],
                      "offense_snaps": [50], "offense_pct": [.8]}).to_sql("snap_counts", conn, index=False)
        monkeypatch.setattr(builder, "get_pfr_to_gsis_map", lambda: {})
        with pytest.raises(ValueError, match="mapping unavailable"):
            builder.load_snaps(conn, 2025, 2025)


def test_missing_player_identity_is_rejected():
    panel = _base()
    panel.loc[0, 'player_id'] = None
    with pytest.raises(ValueError, match='identity'):
        validate_panel(panel)


def test_unidentified_roster_rows_are_excluded_with_diagnostic(capsys):
    from scripts.build_canonical_player_weeks import load_rosters
    with sqlite3.connect(':memory:') as conn:
        pd.DataFrame({'player_id': ['known', None], 'season': [2025, 2025],
                      'week': [1, 1], 'team': ['KC', 'KC'], 'position': ['WR', 'WR']
                      }).to_sql('weekly_rosters', conn, index=False)
        assert load_rosters(conn, 2025, 2025).player_id.tolist() == ['known']
        assert 'excluding 1 roster rows without player IDs' in capsys.readouterr().out

"""A drafted rookie with zero games this season must still get a row.

Before 2026-09-17 (the pace blend) and the days after it, such a player had
NO row anywhere in the serving frame at all -- it is built entirely from
player_weekly_stats, which has no room for a game that has not been
played -- so he never reached predict(), full stop: not predicted from the
pace, invisible.

_drafted_rookie_stub_rows builds a synthetic row from draft_picks_v2 (real
signal: draft capital, combine, college, current team) with every
own-history column left NaN, the same value a real debut row gets once
first_nfl_season marks it as one. Its own model prediction is real but,
per data/experiments/rookie_debut_stub_2022_2025_README.md, measurably
worse than the pace for a real debut week -- _blend_toward_season_pace
still serves pure pace at g=0. These tests pin the row CONSTRUCTION only.
"""
import numpy as np
import pandas as pd
import pytest

from src.predict import NFLPredictor
from src.utils.database import DatabaseManager


@pytest.fixture
def db(tmp_path):
    d = DatabaseManager(tmp_path / "rookie_stub.db")
    with d._get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS weekly_rosters_v2 "
                     "(season INTEGER, week INTEGER, team TEXT, position TEXT, player_id TEXT)")
        conn.executemany(
            "INSERT INTO players (player_id, name, position, birth_date) VALUES (?,?,?,?)",
            [("rook1", "New Rookie", "WR", "2003-01-01"),
             ("rook2", "Undebuted Vet Class", "RB", "2002-06-01")],
            # "ghost" is deliberately absent from `players` -- a draft-day ID
            # in the feed that never reached a real player record.
        )
        conn.executemany(
            "INSERT INTO draft_picks_v2 (player_id, position, college, draft_season, "
            "draft_round, draft_pick, draft_team) VALUES (?,?,?,?,?,?,?)",
            [("rook1", "WR", "State U", 2026, 2, 45, "DAL"),
             ("rook2", "RB", "Tech", 2026, 5, 150, "SEA"),
             # "ghost" is in the draft feed but never reached `players` --
             # must be excluded, not served with a blank name.
             ("ghost", "QB", "Nowhere", 2026, 7, 240, "NYJ")],
        )
        conn.execute("INSERT INTO draft_values (pick, otc) VALUES (45, 12.5)")
        conn.executemany(
            "INSERT INTO weekly_rosters_v2 (player_id, team, season, week, position) "
            "VALUES (?,?,?,?,?)",
            [("rook1", "DAL", 2026, 1, "WR")],  # on Dallas's active roster
        )
        conn.commit()
    return d


@pytest.fixture
def predictor(db):
    p = NFLPredictor.__new__(NFLPredictor)
    p.db = db
    return p


def _empty_player_data(extra_ids=()):
    """A player_data frame with the real 95-column schema and one row for
    an unrelated veteran (real usage always has thousands of these; the
    stub builder copies its own row's schema/dtypes from whatever is
    already there), plus any of the rookies under test who supposedly
    already have a row this season."""
    cols = ["player_id", "name", "position", "season", "week", "team",
            "fantasy_points", "birth_date", "data_source", "draft_round",
            "draft_pick", "draft_pick_value", "draft_season", "draft_college",
            "is_undrafted", "first_nfl_season", "injury_score"]
    rows = [{"player_id": "some_veteran", "season": 2026, "week": 1, "fantasy_points": 12.0}]
    rows += [{"player_id": pid, "season": 2026, "week": 1, "fantasy_points": 10.0}
            for pid in extra_ids]
    return pd.DataFrame(rows, columns=cols)


def test_a_drafted_rookie_with_no_players_row_is_excluded(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    assert set(stub["player_id"]) == {"rook1", "rook2"}, \
        "'ghost' has no players-table identity and must not get a synthetic row"
    assert "some_veteran" not in set(stub["player_id"])


def test_static_attributes_come_from_the_draft_record(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    row = stub.set_index("player_id").loc["rook1"]
    assert row["name"] == "New Rookie"
    assert row["position"] == "WR"
    assert row["draft_round"] == 2 and row["draft_pick"] == 45
    assert row["draft_pick_value"] == 12.5
    assert row["draft_college"] == "State U"
    assert row["is_undrafted"] == 0
    assert row["first_nfl_season"] == 2026
    assert row["season"] == 2026
    assert row["injury_score"] == 1.0, "presumed healthy, same convention as season_projection's cold start"


def test_team_prefers_current_roster_over_draft_team(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    row = stub.set_index("player_id").loc["rook1"]
    # Drafted by nobody in particular here, but rostered on DAL -- a trade
    # or elevation after the draft must win.
    assert row["team"] == "DAL"


def test_team_falls_back_to_draft_team_with_no_roster_row(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    row = stub.set_index("player_id").loc["rook2"]
    assert row["team"] == "SEA"


def test_own_history_columns_are_nan_not_zero(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    row = stub.set_index("player_id").loc["rook1"]
    assert np.isnan(row["fantasy_points"]), \
        "NaN means no game happened; 0 would assert he played and scored nothing"


def test_a_player_who_already_has_a_row_this_season_gets_no_stub(predictor):
    already_played = _empty_player_data(extra_ids=["rook1"])
    stub = predictor._drafted_rookie_stub_rows(already_played, 2026)
    assert "rook1" not in set(stub["player_id"])
    assert "rook2" in set(stub["player_id"])


def test_no_draft_class_for_the_season_returns_empty(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2099)
    assert stub.empty


def test_games_played_excludes_stub_rows_from_the_blend_weight(predictor, monkeypatch):
    """_blend_toward_season_pace must not count a synthetic row as a game
    played -- fantasy_points on it reads 0.0 post-pipeline (calculate_all_
    scores's blanket fill), not NaN, so the exclusion has to key off
    data_source, not fantasy_points.notna()."""
    import config.settings as settings
    monkeypatch.setattr(settings, "PACE_BLEND_KAPPA", 3.0)
    monkeypatch.setattr(settings, "PACE_BLEND_CI_SCALE", 1.5)
    monkeypatch.setattr(NFLPredictor, "_PACE_TABLE_CACHE", pd.DataFrame({
        "player_id": ["rook1"], "season": [2026.0], "step8_pace": [5.0],
    }))
    history = pd.DataFrame({
        "player_id": ["rook1"], "season": [2026], "week": [1],
        "fantasy_points": [0.0],  # post-pipeline value of a stub row
        "data_source": ["rookie_debut_stub"],
    })
    results = pd.DataFrame({"player_id": ["rook1"], "predicted_points": [1.5]})
    out = predictor._blend_toward_season_pace(results, history, 2026, n_weeks=1)
    assert out.loc[0, "games_played_season"] == 0
    assert out.loc[0, "pace_weight"] == 1.0
    assert out.loc[0, "predicted_points"] == 5.0

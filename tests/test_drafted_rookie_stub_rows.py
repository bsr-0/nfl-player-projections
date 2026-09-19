"""A rookie -- drafted or undrafted -- with zero games this season must
still get a row.

Before 2026-09-17 (the pace blend) such a player had NO row anywhere in
the serving frame at all -- it is built entirely from player_weekly_stats,
which has no room for a game that has not been played -- so he never
reached predict(), full stop: not predicted from the pace, invisible. The
drafted half was fixed 2026-09-17; the undrafted half (~39% of everyone who
reaches the league) the next day, 2026-09-18.

_drafted_rookie_stub_rows builds a synthetic row from real signal --
draft_picks_v2 for a drafted rookie, a roster snapshot (years_exp <= 1,
never drafted, no prior history) for an undrafted one -- with every
own-history column left NaN, the same value a real debut row gets once
first_nfl_season marks it as one. Its own model prediction is real but,
per data/experiments/rookie_debut_stub_2022_2025_README.md (drafted) and
undrafted_rookie_debut_stub_2022_2025_README.md (undrafted), not shown to
beat the pace either way -- _blend_toward_season_pace still serves pure
pace at g=0 for both. These tests pin the row CONSTRUCTION only.
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
        conn.execute("CREATE TABLE IF NOT EXISTS rosters "
                     "(player_id TEXT, position TEXT, team TEXT, season INTEGER, "
                     "years_exp REAL, college TEXT)")
        conn.executemany(
            "INSERT INTO players (player_id, name, position, birth_date) VALUES (?,?,?,?)",
            [("rook1", "New Rookie", "WR", "2003-01-01"),
             ("rook2", "Undebuted Vet Class", "RB", "2002-06-01"),
             ("udfa_rookie", "UDFA Debut", "WR", "2003-05-01"),
             ("udfa_year1", "Development Squad", "TE", "2002-11-01"),
             ("camp_body", "Long Tenured", "RB", "2001-02-01")],
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
        conn.executemany(
            "INSERT INTO rosters (player_id, position, team, season, years_exp, college) "
            "VALUES (?,?,?,?,?,?)",
            [("udfa_rookie", "WR", "KC", 2026, 0, "Small State"),
             # Development-squad-then-real-year pattern: years_exp reads 1
             # despite no real regular-season history either -- see the
             # _cold_start_rows_incoming docstring for the real cases this
             # `<= 1` cutoff (not `== 0`) exists to catch.
             ("udfa_year1", "TE", "GB", 2026, 1, "Mid State"),
             # 3 years on a roster without ever debuting -- not a rookie.
             ("camp_body", "RB", "MIA", 2026, 6, "Old State")],
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
    ids = set(stub["player_id"])
    assert {"rook1", "rook2"} <= ids
    assert "ghost" not in ids, \
        "'ghost' has no players-table identity and must not get a synthetic row"
    assert "some_veteran" not in ids


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


def test_undrafted_rookie_gets_a_stub_with_the_sentinel_values(predictor):
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    row = stub.set_index("player_id").loc["udfa_rookie"]
    assert row["name"] == "UDFA Debut"
    assert row["team"] == "KC"
    assert row["draft_college"] == "Small State"
    assert row["is_undrafted"] == 1
    assert row["draft_round"] == 8 and row["draft_pick"] == 400, \
        "UNDRAFTED_ROUND/UNDRAFTED_PICK, matching advanced_rookie_injury's convention"
    assert row["draft_pick_value"] == 0.0
    assert row["first_nfl_season"] == 2026


def test_development_squad_year_one_still_counts_as_a_rookie(predictor):
    """years_exp <= 1, not == 0: a real development-squad-then-debut
    pattern reads years_exp==1 on the roster feed despite no regular-season
    history either. See _cold_start_rows_incoming's docstring."""
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    assert "udfa_year1" in set(stub["player_id"])


def test_long_tenured_never_drafted_player_is_not_treated_as_a_rookie(predictor):
    """3 seasons on a roster without ever debuting is a real, if marginal,
    veteran -- not this year's incoming class, however undrafted he is."""
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    assert "camp_body" not in set(stub["player_id"])


def test_undrafted_rookie_with_prior_history_gets_no_stub(predictor):
    already_played = _empty_player_data(extra_ids=[])
    # Give him a row from an EARLIER season -- real history, not "already
    # has a row this season" (a different, already-tested exclusion).
    prior = pd.DataFrame([{"player_id": "udfa_rookie", "season": 2024, "week": 10,
                          "fantasy_points": 3.0}])
    frame = pd.concat([already_played, prior], ignore_index=True)
    stub = predictor._drafted_rookie_stub_rows(frame, 2026)
    assert "udfa_rookie" not in set(stub["player_id"])


def test_drafted_and_undrafted_do_not_duplicate_a_player(predictor):
    """A player could in principle satisfy both queries (e.g. a data
    inconsistency); drop_duplicates in the union must keep exactly one row."""
    stub = predictor._drafted_rookie_stub_rows(_empty_player_data(), 2026)
    assert stub["player_id"].is_unique

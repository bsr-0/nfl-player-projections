"""Joining an ESPN league to this project's projections.

Everything the league report will say rests on this join, so what it must not
do is guess: an unmatched player keeps his row with null projections and a
stated reason, never a zero and never a same-name player from another team.

The second thing pinned here is that the join does not decide which number is
"the projection". It serves whatever docs/data/weekly_meta.json says the
pipeline is publishing, so the switch from season pace to the real weekly
model after week 1 needs no code change.
"""
import pandas as pd
import pytest

from src.integrations.league_join import (
    Snapshot, VETERAN_PER_GAME_WEIGHT, join_players, load_projections,
    match_report,
)

BOARD = pd.DataFrame([
    {"player_id": "00-0036223", "name": "J.Gibbs", "team": "DET",
     "position": "RB", "season_total": 270.3, "season_ppg": 15.9,
     "floor": 200.0, "ceiling": 320.0, "risk_score": 40,
     "support_class": "starter", "source": "preseason_model",
     "prev_season_games": 16},
    # Two different receivers who abbreviate to the same name.
    {"player_id": "00-0000001", "name": "A.Smith", "team": "DAL",
     "position": "WR", "season_total": 34.3, "season_ppg": 2.0,
     "floor": 10.0, "ceiling": 60.0, "risk_score": 70,
     "support_class": "backup", "source": "preseason_model",
     "prev_season_games": None},
    {"player_id": "00-0000002", "name": "A.Smith", "team": "NYJ",
     "position": "WR", "season_total": 34.0, "season_ppg": 2.0,
     "floor": 10.0, "ceiling": 60.0, "risk_score": 70,
     "support_class": "backup", "source": "preseason_model",
     "prev_season_games": 12},
    {"player_id": "00-0000003", "name": "K.Walker", "team": "SEA",
     "position": "RB", "season_total": 180.0, "season_ppg": 10.6,
     "floor": 120.0, "ceiling": 240.0, "risk_score": 50,
     "support_class": "starter", "source": "preseason_model",
     "prev_season_games": 15},
])

PRORATED = pd.DataFrame([
    {"name": "J.Gibbs", "position": "RB", "team": "DET", "opponent": "NO",
     "home_away": "home", "predicted_points": 15.9},
    {"name": "K.Walker", "position": "RB", "team": "SEA", "opponent": "SF",
     "home_away": "away", "predicted_points": 10.6},
])

WEEKLY_MODEL = pd.DataFrame([
    {"name": "J.Gibbs", "position": "RB", "team": "DET", "opponent": "NO",
     "home_away": "home", "predicted_points": 21.4,
     "prediction_ci80_lower": 12.1, "prediction_ci80_upper": 30.8},
])


def _projections(weekly=PRORATED, mode="season_prorated"):
    return load_projections(2026, 1, board=BOARD, weekly=weekly,
                            meta={"mode": mode})


def _espn(**overrides):
    row = {"player_id": 4429795, "name": "Jahmyr Gibbs", "position": "RB",
           "team": "DET", "lineup_slot": "RB", "injury_status": "ACTIVE",
           "percent_owned": 99.9, "projected_avg_points": 21.5}
    row.update(overrides)
    return row


CROSSWALK = {"4429795": "00-0036223"}


def test_espn_id_is_the_join(): 
    out = join_players([_espn()], _projections(), CROSSWALK)

    row = out.iloc[0]
    assert row["match_method"] == "espn_id"
    assert row["player_id"] == "00-0036223"
    assert row["week_points"] == 15.9
    assert row["opponent"] == "NO"


def test_name_carries_a_player_the_id_map_misses():
    """Suffixes and ESPN's own team codes both have to normalise."""
    out = join_players([_espn(player_id=999, name="Kenneth Walker III",
                              team="SEA")], _projections(), CROSSWALK)

    row = out.iloc[0]
    assert row["match_method"] == "name_team_pos"
    assert row["player_id"] == "00-0000003"


@pytest.mark.parametrize("espn_team,board_team", [("LAR", "LA"), ("WSH", "WAS")])
def test_espn_team_codes_normalise(espn_team, board_team):
    board = BOARD.assign(team=BOARD["team"].replace({"SEA": board_team}))
    out = join_players([_espn(player_id=999, name="Kenneth Walker III",
                              team=espn_team)],
                       _projections(weekly=pd.DataFrame()).assign(
                           team=board["team"]),
                       CROSSWALK)

    assert out.iloc[0]["match_method"] == "name_team_pos"


def test_two_players_of_the_same_name_are_told_apart_by_team():
    """A.Smith DAL and A.Smith NYJ are different receivers, and the key knows."""
    out = join_players([_espn(player_id=999, name="Adam Smith",
                              position="WR", team="NYJ")],
                       _projections(), {})

    assert out.iloc[0]["player_id"] == "00-0000002"


def test_an_ambiguous_name_resolves_to_nothing():
    """Same name, same position, same team. Guessing would be worse."""
    twin = BOARD.iloc[[1]].assign(player_id="00-0000009", season_total=12.0)
    out = join_players([_espn(player_id=999, name="Adam Smith",
                              position="WR", team="DAL")],
                       load_projections(2026, 1,
                                        board=pd.concat([BOARD, twin],
                                                        ignore_index=True),
                                        weekly=PRORATED, meta={}), {})

    row = out.iloc[0]
    assert row["match_method"] is None
    assert row["unmatched_reason"] == "name is not unique on the board"
    assert pd.isna(row.get("season_total"))


def test_kickers_and_defences_are_named_as_unmodelled():
    """This project models QB/RB/WR/TE. The league starts K and D/ST too."""
    out = join_players([_espn(player_id=1, name="Cameron Dicker", position="K"),
                        _espn(player_id=2, name="Jaguars D/ST",
                              position="D/ST")], _projections(), {})

    assert list(out["unmatched_reason"]) == ["position not modelled"] * 2


def test_an_unmatched_player_keeps_his_row_with_no_number():
    out = join_players([_espn(player_id=555, name="Tank Dell",
                              position="WR", team="HOU")], _projections(), {})

    row = out.iloc[0]
    assert row["espn_name"] == "Tank Dell"
    assert row["unmatched_reason"] == "not on the board"
    # Null, not zero: "we have no projection" is not "we project nothing".
    assert pd.isna(row.get("season_total"))
    assert row["espn_projected_avg"] == 21.5


def test_a_traded_player_resolves_once():
    """The board carries a row per team he played for; one player, one row."""
    board = pd.concat([BOARD, BOARD.iloc[[0]].assign(team="KC",
                                                     season_total=100.0)],
                      ignore_index=True)
    out = join_players([_espn()],
                       load_projections(2026, 1, board=board,
                                        weekly=PRORATED, meta={}),
                       CROSSWALK)

    assert len(out) == 1
    assert out.iloc[0]["season_total"] == 270.3


def test_prorated_mode_is_labelled_and_carries_no_interval():
    out = join_players([_espn()], _projections(), CROSSWALK)

    row = out.iloc[0]
    assert row["projection_mode"] == "season_prorated"
    assert pd.isna(row["week_ci_low"]) and pd.isna(row["week_ci_high"])


def test_the_weekly_model_is_picked_up_with_no_code_change():
    """After week 1 the same file holds real weekly numbers and intervals."""
    out = join_players([_espn()],
                       _projections(weekly=WEEKLY_MODEL, mode="weekly_model"),
                       CROSSWALK)

    row = out.iloc[0]
    assert row["projection_mode"] == "weekly_model"
    assert row["week_points"] == 21.4
    assert (row["week_ci_low"], row["week_ci_high"]) == (12.1, 30.8)


def test_a_bye_is_not_a_missing_projection():
    """generate_weekly_data drops bye players from the week's file."""
    projections = _projections()
    walker = projections[projections["name"] == "K.Walker"].iloc[0]
    gibbs_week = projections[projections["name"] == "J.Gibbs"].iloc[0]

    assert not walker["on_bye"] and not gibbs_week["on_bye"]

    bye = _projections(weekly=PRORATED[PRORATED["name"] == "K.Walker"])
    gibbs = bye[bye["name"] == "J.Gibbs"].iloc[0]
    assert gibbs["on_bye"]
    assert pd.isna(gibbs["week_points"])
    assert pd.notna(gibbs["season_total"])


def test_match_report_counts_and_names_the_gaps():
    players = [_espn(), _espn(player_id=1, name="Cameron Dicker", position="K"),
               _espn(player_id=555, name="Tank Dell", position="WR", team="HOU")]
    report = match_report(join_players(players, _projections(), CROSSWALK))

    assert report == {
        "players": 3, "matched": 1, "by_method": {"espn_id": 1},
        "by_reason": {"position not modelled": 1, "not on the board": 1},
        "unmatched": [
            {"espn_name": "Cameron Dicker", "position": "K", "nfl_team": "DET",
             "unmatched_reason": "position not modelled"},
            {"espn_name": "Tank Dell", "position": "WR", "nfl_team": "HOU",
             "unmatched_reason": "not on the board"},
        ],
    }


def test_snapshot_reads_the_week_from_the_league_not_the_calendar():
    snap = Snapshot(path=None, info={"current_week": 3},
                    manifest={"season": 2026},
                    matchups=[{"week": 3, "team_id": 1,
                               "opponent_name": "Fig Newtons"}],
                    rosters=[{"team_name": "Baby Back Gibbs", "team_id": 1,
                              "roster": [{"name": "Jahmyr Gibbs"}]}])

    assert (snap.week, snap.season) == (3, 2026)
    assert snap.opponent_for(1)["opponent_name"] == "Fig Newtons"
    assert snap.rostered()[0]["fantasy_team"] == "Baby Back Gibbs"
    assert snap.opponent_for(1, week=9) == {}


VETERAN = pd.DataFrame([{
    "player_id": "00-0036223", "name": "J.Gibbs", "team": "DET",
    "position": "RB", "season_total": 170.0, "season_ppg": 10.0,
    "floor": 120.0, "ceiling": 220.0, "risk_score": 40,
    "support_class": "starter", "source": "preseason_model",
    "prev_season_games": 16, "expected_games": 10.0}])

VETERAN_WEEK = pd.DataFrame([{"name": "J.Gibbs", "position": "RB",
                              "team": "DET", "opponent": "NO",
                              "home_away": "home", "predicted_points": 10.0}])


def test_a_veteran_is_moved_off_the_season_divisor():
    """total/17 averages in games he is expected to miss; the week he is
    being started in is not one of them."""
    out = load_projections(2026, 1, board=VETERAN, weekly=VETERAN_WEEK,
                           meta={"mode": "season_prorated"})

    row = out.iloc[0]
    # published 10.0, per game 170/10 = 17.0, half way = 13.5
    assert row["week_points"] == 13.5
    assert row["week_points_basis"] == "0.5 of the way to per-game"
    assert VETERAN_PER_GAME_WEIGHT == 0.5


def test_a_first_year_player_keeps_the_published_number():
    """The two divisors nearly agree for him, and moving further overshoots."""
    board = VETERAN.assign(prev_season_games=None)
    out = load_projections(2026, 1, board=board, weekly=VETERAN_WEEK,
                           meta={"mode": "season_prorated"})

    assert out.iloc[0]["week_points"] == 10.0
    assert out.iloc[0]["week_points_basis"] == "published"


def test_the_real_weekly_model_is_never_rebased():
    """Once games are played the number is a weekly prediction, not a pace."""
    out = load_projections(2026, 1, board=VETERAN, weekly=VETERAN_WEEK,
                           meta={"mode": "weekly_model"})

    assert out.iloc[0]["week_points"] == 10.0
    assert out.iloc[0]["week_points_basis"] == "published"


def test_a_player_with_no_games_estimate_is_left_alone():
    board = VETERAN.assign(expected_games=None)
    out = load_projections(2026, 1, board=board, weekly=VETERAN_WEEK,
                           meta={"mode": "season_prorated"})

    assert out.iloc[0]["week_points"] == 10.0

"""The five sections of the league report.

Two properties matter more than any single number here. A comparison must
never cross pricing scales -- this project's per-week numbers run at 73-85% of
ESPN's, so subtracting one from the other measures the scale, not the player.
And a player nobody can price must never be assigned a zero, which would rank
him last instead of unknown.
"""
import pandas as pd
import pytest

from src.integrations.league_insights import (
    availability, build_report, eligible, find_team, horizon_weeks,
    first_year_starters, injury_watch, lineup_changes, lineup_value,
    optimal_lineup, pace_band, price, propose_trades, starting_slots,
    streaming_targets, tough_calls, waiver_targets, win_probability,
)
from src.integrations.league_join import Snapshot, load_projections

SETTINGS = {"position_slot_counts": {"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                                     "RB/WR/TE": 1, "D/ST": 1, "K": 1,
                                     "BE": 7, "IR": 2, "OP": 0}}


def _p(name, position, points, **kw):
    row = {"index": kw.pop("index", abs(hash(name)) % 10000), "name": name,
           "position": position, "points": points, "points_source": "model",
           "available": points is not None, "unavailable_reason": None,
           "nfl_team": "DET", "lineup_slot": "BE", "injury_status": "ACTIVE",
           "percent_owned": 50.0, "espn_projected_avg": points,
           "model_season_total": None, "model_season_ppg": points,
           "floor": None, "ceiling": None, "week_ci": [None, None],
           "opponent": "NO", "matched": True, "fantasy_team": "Mine"}
    row.update(kw)
    return row


def test_starting_slots_drops_the_bench():
    slots = starting_slots(SETTINGS)

    assert sorted(slots) == ["D/ST", "K", "QB", "RB", "RB", "RB/WR/TE",
                             "TE", "WR", "WR"]


@pytest.mark.parametrize("slot,position,ok", [
    ("RB", "RB", True), ("RB", "WR", False),
    ("RB/WR/TE", "TE", True), ("RB/WR/TE", "QB", False),
    # "D/ST" is one position, not a D-or-ST flex.
    ("D/ST", "D/ST", True), ("D/ST", "D", False),
])
def test_slot_eligibility(slot, position, ok):
    assert eligible(slot, position) is ok


def test_price_prefers_this_project_and_labels_the_fallback():
    assert price({"week_points": 15.0, "espn_projected_avg": 20.0}) == (15.0, "model")
    assert price({"week_points": None, "espn_projected_avg": 9.9}) == (9.9, "espn")
    assert price({"week_points": None, "espn_projected_avg": None}) == (None, None)


@pytest.mark.parametrize("row,expected", [
    ({"injury_status": "OUT"}, (False, "OUT")),
    ({"injury_status": "INJURY_RESERVE"}, (False, "INJURY_RESERVE")),
    ({"injury_status": "ACTIVE", "on_bye": True}, (False, "BYE")),
    # A decision for the manager, not for the optimiser.
    ({"injury_status": "QUESTIONABLE"}, (True, None)),
])
def test_availability(row, expected):
    assert availability(row) == expected


def test_lineup_fills_narrow_slots_before_the_flex():
    players = [_p("RB1", "RB", 20.0), _p("RB2", "RB", 18.0),
               _p("RB3", "RB", 17.0), _p("WR1", "WR", 16.0),
               _p("WR2", "WR", 15.0), _p("TE1", "TE", 8.0),
               _p("QB1", "QB", 22.0)]

    starters, bench = optimal_lineup(players, starting_slots(SETTINGS))

    flex = next(s for s in starters if s["slot"] == "RB/WR/TE")
    assert flex["name"] == "RB3"          # best player the narrow slots left
    assert {s["name"] for s in starters if s["slot"] == "RB"} == {"RB1", "RB2"}
    assert bench == []


def test_a_bye_or_an_injury_keeps_a_player_out_of_the_lineup():
    players = [_p("RB1", "RB", 20.0, available=False,
                  unavailable_reason="BYE"),
               _p("RB2", "RB", 5.0), _p("RB3", "RB", 4.0)]

    starters, bench = optimal_lineup(players, ["RB"])

    assert [s["name"] for s in starters] == ["RB2"]
    assert {b["name"] for b in bench} == {"RB1", "RB3"}


def test_an_unpriced_player_never_starts():
    """No projection is not a projection of zero -- but it is not a lineup."""
    players = [_p("Unknown", "RB", None, available=False), _p("RB2", "RB", 3.0)]

    starters, _ = optimal_lineup(players, ["RB"])

    assert [s["name"] for s in starters] == ["RB2"]


def test_tough_calls_are_the_close_ones_only():
    starters = [dict(_p("Starter", "WR", 11.0), slot="WR"),
                dict(_p("Flex", "RB", 10.0), slot="RB/WR/TE")]
    bench = [_p("Close", "WR", 9.5), _p("Miles off", "WR", 2.0)]

    calls = tough_calls(starters, bench)

    assert [c["benched"]["name"] for c in calls] == ["Close"]
    # Compared against the weakest starter he could legally replace.
    assert calls[0]["slot"] == "RB/WR/TE"
    assert calls[0]["gap"] == 0.5


def test_tough_calls_never_compare_across_pricing_scales():
    starters = [dict(_p("Kicker", "K", 9.0, points_source="espn"), slot="K")]
    bench = [_p("Other kicker", "K", 8.5, points_source="model")]

    assert tough_calls(starters, bench) == []


def test_waiver_targets_beat_the_man_who_would_be_dropped():
    roster = [_p("Keep", "RB", 12.0), _p("Drop", "RB", 3.0)]
    free_agents = [_p("Better", "RB", 6.0), _p("Worse", "RB", 2.0),
                   _p("Marginal", "RB", 3.5)]

    targets = waiver_targets(free_agents, roster, starting_slots(SETTINGS))

    assert [t["name"] for t in targets] == ["Better"]
    assert targets[0]["instead_of"] == {"name": "Drop", "points": 3.0}


def test_waiver_targets_stay_on_one_scale():
    roster = [_p("Drop", "RB", 3.0)]
    free_agents = [_p("ESPN priced", "RB", 30.0, points_source="espn")]

    assert waiver_targets(free_agents, roster, starting_slots(SETTINGS)) == []


def _snapshot(**kw):
    roster = [{"player_id": 1, "name": "Jahmyr Gibbs", "position": "RB",
               "team": "DET", "lineup_slot": "RB", "injury_status": "ACTIVE",
               "projected_avg_points": 21.5, "percent_owned": 99.9}]
    base = dict(
        path="snap",
        info={"current_week": 1, "name": "Test League"},
        settings=SETTINGS,
        rosters=[{"team_id": 1, "team_name": "Mine", "wins": 0, "losses": 0,
                  "roster": roster},
                 {"team_id": 2, "team_name": "Theirs", "wins": 0, "losses": 0,
                  "roster": []}],
        free_agents=[],
        matchups=[{"week": 1, "team_id": 1, "opponent_id": 2,
                   "opponent_name": "Theirs"}],
        manifest={"season": 2026},
    )
    base.update(kw)
    return Snapshot(**base)


BOARD = pd.DataFrame([{
    "player_id": "00-0036223", "name": "J.Gibbs", "team": "DET",
    "position": "RB", "season_total": 270.3, "season_ppg": 15.9,
    "floor": 200.0, "ceiling": 320.0, "risk_score": 40,
    "support_class": "starter", "source": "preseason_model",
    "prev_season_games": 16}])

WEEKLY = pd.DataFrame([{"name": "J.Gibbs", "position": "RB", "team": "DET",
                        "opponent": "NO", "home_away": "home",
                        "predicted_points": 15.9}])


def _projections(mode="season_prorated"):
    return load_projections(2026, 1, board=BOARD, weekly=WEEKLY,
                            meta={"mode": mode})


def test_build_report_states_its_matchup_and_its_limits():
    report = build_report(_snapshot(), _projections(), team=1,
                          crosswalk={"1": "00-0036223"})

    assert report["team"]["name"] == "Mine"
    assert report["matchup"]["opponent"] == "Theirs"
    assert report["matchup"]["projected_total"] == 15.9
    # An empty opponent roster projects nothing, and the edge says so.
    assert report["matchup"]["edge"] == 15.9
    assert report["projection_mode"] == "season_prorated"
    assert any("season total over 17" in c for c in report["caveats"])


def test_the_prorated_caveat_goes_away_in_weekly_mode():
    report = build_report(_snapshot(), _projections(mode="weekly_model"),
                          team=1, crosswalk={"1": "00-0036223"})

    assert report["projection_mode"] == "weekly_model"
    assert not any("season total over 17" in c for c in report["caveats"])


def test_find_team_by_id_or_by_part_of_the_name():
    snap = _snapshot()

    assert find_team(snap, 1)["team_name"] == "Mine"
    assert find_team(snap, "their")["team_name"] == "Theirs"
    with pytest.raises(ValueError, match="no team matches"):
        find_team(snap, "nobody")


SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "RB/WR/TE"]


def _tradeable(name, position, ours, espn, team, **kw):
    return _p(name, position, ours, fantasy_team=team, model_season_ppg=ours,
              espn_projected_avg=espn, **kw)


def test_lineup_value_counts_only_what_starts():
    """A fourth running back is worth what he starts for, which is nothing."""
    roster = [_tradeable(f"RB{i}", "RB", pts, pts, "Mine")
              for i, pts in enumerate([20.0, 15.0, 10.0, 9.0])]

    assert lineup_value(roster, ["RB", "RB"]) == 35.0
    assert lineup_value(roster[:3], ["RB", "RB"]) == 35.0


def test_a_trade_is_proposed_when_the_two_valuations_disagree():
    """We rate their WR above ours; ESPN rates ours above theirs. Both gain."""
    league = [
        _tradeable("My WR", "WR", 10.0, 18.0, "Mine"),
        _tradeable("My QB", "QB", 20.0, 20.0, "Mine"),
        _tradeable("Their WR", "WR", 16.0, 12.0, "Theirs"),
        _tradeable("Their QB", "QB", 20.0, 20.0, "Theirs"),
    ]

    proposals = propose_trades(league, "Mine", ["QB", "WR"], horizon_weeks=14)

    assert len(proposals) == 1
    deal = proposals[0]
    assert deal["with"] == "Theirs"
    assert (deal["give"]["name"], deal["get"]["name"]) == ("My WR", "Their WR")
    assert deal["our_gain_per_week"] == 6.0
    assert deal["our_gain_over_horizon"] == 84.0
    assert deal["their_gain_per_week_espn"] == 6.0
    assert deal["lineup_change"]["in"] == [{"name": "Their WR", "slot": "WR"}]
    assert deal["lineup_change"]["out"] == [
        {"name": "My WR", "slot": "WR", "reason": "traded"}]


def test_no_trade_when_both_sides_value_players_the_same():
    """Agreement means somebody has to lose, so nothing gets proposed."""
    league = [
        _tradeable("My WR", "WR", 10.0, 10.0, "Mine"),
        _tradeable("Their WR", "WR", 16.0, 16.0, "Theirs"),
    ]

    assert propose_trades(league, "Mine", ["WR"], horizon_weeks=14) == []


def test_espn_priced_players_are_never_traded():
    """Their number is on a different scale; swapping one in invents a gain."""
    league = [
        _tradeable("My WR", "WR", 10.0, 10.0, "Mine"),
        _tradeable("Their K", "K", 30.0, 30.0, "Theirs", points_source="espn"),
        _tradeable("Their WR", "WR", 30.0, 4.0, "Theirs", points_source="espn"),
    ]

    assert propose_trades(league, "Mine", ["WR"], horizon_weeks=14) == []


def test_an_injured_player_is_not_traded_for_as_if_he_plays():
    """OUT means he contributes nothing to the lineup he would move into."""
    league = [
        _tradeable("My WR", "WR", 10.0, 18.0, "Mine"),
        _tradeable("Their WR", "WR", 16.0, 12.0, "Theirs",
                   injury_status="OUT"),
    ]

    assert propose_trades(league, "Mine", ["WR"], horizon_weeks=14) == []


def test_horizon_is_the_rest_of_the_fantasy_regular_season():
    snap = _snapshot(settings=dict(SETTINGS, reg_season_count=14))

    assert horizon_weeks(snap, 1) == 14
    assert horizon_weeks(snap, 12) == 3
    assert horizon_weeks(snap, 18) == 1        # never zero or negative


def test_the_report_carries_proposals_and_their_horizon():
    report = build_report(_snapshot(), _projections(), team=1,
                          crosswalk={"1": "00-0036223"}, horizon=9)

    assert report["trades"]["horizon_weeks"] == 9
    assert report["trades"]["proposals"] == []   # the other roster is empty


def test_a_pure_scale_difference_produces_no_proposal():
    """ESPN at exactly 1.25x on every player is a disagreement about nothing.

    The orderings are identical, so no swap can improve both lineups. This is
    the trap the old raw-difference ranking fell into, kept pinned on the path
    that replaced it.
    """
    league = [_tradeable(f"WR{i}", "WR", pts, pts * 1.25,
                         "Mine" if i < 2 else "Theirs")
              for i, pts in enumerate([18.0, 12.0, 15.0, 9.0])]

    assert propose_trades(league, "Mine", ["WR", "WR"], horizon_weeks=14) == []


def test_the_player_you_trade_away_does_not_go_to_your_bench():
    """He is on another roster. Calling that "benched" reads as a mistake."""
    league = [
        _tradeable("My WR", "WR", 10.0, 18.0, "Mine"),
        _tradeable("Their WR", "WR", 16.0, 12.0, "Theirs"),
    ]

    deal = propose_trades(league, "Mine", ["WR"], horizon_weeks=14)[0]

    assert deal["lineup_change"]["out"] == [
        {"name": "My WR", "slot": "WR", "reason": "traded"}]


def test_a_bench_move_is_still_called_a_bench_move():
    """Only the departing player is "traded"; anyone else displaced is not."""
    league = [
        _tradeable("Starter", "WR", 9.0, 20.0, "Mine"),
        _tradeable("Also starting", "WR", 8.0, 8.0, "Mine"),
        _tradeable("Their WR", "WR", 20.0, 9.0, "Theirs"),
    ]

    deal = propose_trades(league, "Mine", ["WR"], horizon_weeks=14)[0]
    out = {o["name"]: o["reason"] for o in deal["lineup_change"]["out"]}

    assert out["Starter"] == "traded"


def test_the_lineup_says_where_it_disagrees_with_espn():
    """The lineup here is built from projections, not read from ESPN."""
    roster = [{"player_id": 1, "name": "Benched but better", "position": "RB",
               "team": "DET", "lineup_slot": "BE", "injury_status": "ACTIVE",
               "projected_avg_points": 20.0, "percent_owned": 50.0},
              {"player_id": 2, "name": "Starting but worse", "position": "RB",
               "team": "DET", "lineup_slot": "RB", "injury_status": "ACTIVE",
               "projected_avg_points": 4.0, "percent_owned": 50.0}]
    snap = _snapshot(settings={"position_slot_counts": {"RB": 1, "BE": 1}},
                     rosters=[{"team_id": 1, "team_name": "Mine",
                               "roster": roster}])

    report = build_report(snap, _projections(), team=1, crosswalk={})

    assert [c["action"] for c in report["lineup_changes"]] == ["START", "SIT"]
    assert report["starters"][0]["name"] == "Benched but better"


def test_no_disagreement_is_reported_as_no_changes():
    report = build_report(_snapshot(), _projections(), team=1,
                          crosswalk={"1": "00-0036223"})

    assert report["lineup_changes"] == []
    assert lineup_changes(report["starters"], report["bench"]) == []


def test_streaming_compares_against_the_man_you_are_starting():
    """You drop the kicker for a better kicker; the bench is not the baseline."""
    starters = [dict(_p("My K", "K", 8.0, points_source="espn"), slot="K")]
    free_agents = [_p("Better K", "K", 10.0, points_source="espn"),
                   _p("Worse K", "K", 7.0, points_source="espn"),
                   _p("A receiver", "WR", 20.0)]

    out = streaming_targets(free_agents, starters)

    assert [t["available"]["name"] for t in out] == ["Better K"]
    assert out[0]["gain"] == 2.0
    assert out[0]["current"]["name"] == "My K"


def test_streaming_only_covers_the_positions_you_stream():
    """A better RB is a waiver add, not a stream, and belongs in that section."""
    starters = [dict(_p("My RB", "RB", 8.0), slot="RB")]
    free_agents = [_p("Better RB", "RB", 14.0)]

    assert streaming_targets(free_agents, starters) == []


def test_streaming_stays_on_one_pricing_scale():
    starters = [dict(_p("My K", "K", 8.0, points_source="espn"), slot="K")]
    free_agents = [_p("Model priced", "K", 30.0, points_source="model")]

    assert streaming_targets(free_agents, starters) == []


def test_a_questionable_starter_is_paired_with_who_would_replace_him():
    starters = [dict(_p("Hurt", "WR", 13.0, injury_status="QUESTIONABLE"),
                     slot="WR")]
    bench = [_p("Next man", "WR", 9.0), _p("Worse", "WR", 4.0)]

    watch = injury_watch(starters, bench, ["WR"])

    assert watch[0]["name"] == "Hurt"
    assert watch[0]["replacement"] == {"name": "Next man", "points": 9.0,
                                       "cost": 4.0}


def test_a_healthy_starter_is_not_on_the_watch_list():
    starters = [dict(_p("Fine", "WR", 13.0), slot="WR")]

    assert injury_watch(starters, [_p("Bench", "WR", 9.0)], ["WR"]) == []


def test_no_eligible_replacement_says_so_rather_than_inventing_one():
    starters = [dict(_p("Hurt K", "K", 9.0, injury_status="QUESTIONABLE",
                        points_source="espn"), slot="K")]
    bench = [_p("A receiver", "WR", 20.0)]

    assert injury_watch(starters, bench, ["K"])[0]["replacement"] is None


def test_the_report_itemises_the_opponent_lineup():
    """The total says how big the hill is; the rows say where it is."""
    snap = _snapshot()
    snap.rosters[1]["roster"] = [
        {"player_id": 2, "name": "Jahmyr Gibbs", "position": "RB",
         "team": "DET", "lineup_slot": "RB", "injury_status": "ACTIVE",
         "projected_avg_points": 21.5, "percent_owned": 99.9}]

    report = build_report(snap, _projections(), team=1, crosswalk={})

    assert [p["name"] for p in report["opponent_starters"]] == ["Jahmyr Gibbs"]
    assert report["matchup"]["opponent_projected_total"] > 0


def _priced(name, points, mae=4.0, **kw):
    return _p(name, "RB", points, measured_mae=mae, **kw)


def test_win_probability_is_the_margin_over_the_combined_spread():
    mine = [_priced("A", 20.0), _priced("B", 20.0)]
    theirs = [_priced("C", 20.0), _priced("D", 20.0)]

    assert win_probability(mine, theirs) == 0.5

    ahead = [_priced("A", 30.0), _priced("B", 30.0)]
    assert win_probability(ahead, theirs) > 0.9


def test_no_measured_error_means_no_probability_rather_than_a_coin_flip():
    """K and D/ST have no measured error in this project. Saying 50% would be
    inventing one."""
    mine = [_p("Kicker", "K", 9.0, measured_mae=None)]

    assert win_probability(mine, [_p("Their K", "K", 8.0,
                                     measured_mae=None)]) is None


def test_unmeasured_players_add_points_but_no_spread():
    mine = [_priced("A", 20.0), _p("Kicker", "K", 10.0, measured_mae=None)]
    theirs = [_priced("C", 20.0)]

    # The kicker's 10 points move the margin, so this must beat an even match.
    assert win_probability(mine, theirs) > 0.9


def test_the_pace_band_is_the_season_band_over_seventeen():
    assert pace_band({"floor": 170.0, "ceiling": 340.0}) == [10.0, 20.0]
    assert pace_band({"floor": None, "ceiling": 340.0}) is None


def test_first_year_starters_are_the_ones_with_no_history():
    starters = [_p("Rookie", "WR", 9.0, prev_season_games=None, matched=True),
                _p("Veteran", "WR", 9.0, prev_season_games=16, matched=True),
                # Unmatched: no board row at all, so nothing is known either way.
                _p("Unknown", "WR", 9.0, prev_season_games=None, matched=False)]

    assert first_year_starters(starters) == ["Rookie"]


def test_a_rookie_starter_earns_the_measured_underprojection_caveat():
    snap = _snapshot()
    board = BOARD.assign(prev_season_games=None)
    projections = load_projections(2026, 1, board=board, weekly=WEEKLY,
                                   meta={"mode": "season_prorated"})

    report = build_report(snap, projections, team=1,
                          crosswalk={"1": "00-0036223"})

    assert any("no NFL history" in c and "1.7 points a week low" in c
               for c in report["caveats"])

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
    availability, build_report, eligible, find_team, optimal_lineup, price,
    starting_slots, tough_calls, trade_candidates, waiver_targets,
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


def test_a_pure_scale_difference_is_not_a_trade():
    """The whole point of standardising.

    ESPN at exactly 1.25x this project on every player is a disagreement about
    nothing: the ordering is identical. Raw subtraction would call every one
    of them a sell.
    """
    league = [_p(f"RB{i}", "RB", pts, fantasy_team=("Mine" if i < 3 else "Theirs"),
                 espn_projected_avg=pts * 1.25, model_season_ppg=pts)
              for i, pts in enumerate([20.0, 15.0, 12.0, 9.0, 6.0, 3.0])]

    trades = trade_candidates(league, "Mine")

    assert trades["buy_low"] == [] and trades["sell_high"] == []


def test_a_real_disagreement_survives_standardising():
    league = [_p(f"RB{i}", "RB", pts, fantasy_team="Theirs",
                 espn_projected_avg=espn, model_season_ppg=pts)
              for i, (pts, espn) in enumerate(
                  [(20.0, 25.0), (15.0, 18.75), (12.0, 15.0),
                   (11.0, 4.0), (6.0, 7.5), (3.0, 3.75)])]

    buys = trade_candidates(league, "Mine")["buy_low"]

    assert buys[0]["name"] == "RB3"
    assert buys[0]["model_rank"] < buys[0]["espn_rank"]


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

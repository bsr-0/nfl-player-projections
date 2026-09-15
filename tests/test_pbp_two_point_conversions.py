"""The PBP fallback must score two-point conversions.

nflverse zeroes yards/TDs/receptions on two-point plays (they are not
official passing/rushing/receiving stats), so those 2 points reach
fantasy_points only if counted separately. The weekly-data path does that
via passing_/rushing_/receiving_2pt_conversions; the PBP fallback declared
the column, defaulted it to 0, and never populated it -- so every
conversion in a PBP-sourced season scored zero. 2025 is exactly such a
season: nflverse weekly 404s for it, so the fallback is the only source.

Verified against real 2024 data when this landed: PBP-derived credits
matched nflverse official weekly exactly (100 player-week rows, 102
credits, no discrepancy in either direction).
"""
import numpy as np
import pandas as pd
import pytest

from src.data.pbp_stats_aggregator import PBPStatsAggregator


def _play(**kw):
    base = {
        "season": 2025, "week": 1, "posteam": "AAA", "defteam": "BBB",
        "home_team": "AAA", "play_type": "pass", "two_point_attempt": 0,
        "two_point_conv_result": None, "passer_player_id": None,
        "rusher_player_id": None, "receiver_player_id": None,
        "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 0.0,
        "complete_pass": 0, "pass_touchdown": 0, "rush_touchdown": 0,
        "interception": 0, "play_id": 1,
    }
    base.update(kw)
    return base


@pytest.fixture
def agg():
    return PBPStatsAggregator()


def test_conversion_pass_credits_both_passer_and_receiver(agg):
    pbp = pd.DataFrame([
        _play(two_point_attempt=1, two_point_conv_result="success",
              passer_player_id="QB1", receiver_player_id="WR1"),
    ])
    out = agg.aggregate_two_point_conversions(pbp).set_index("player_id")
    assert out.loc["QB1", "two_point_conversions"] == 1
    assert out.loc["WR1", "two_point_conversions"] == 1


def test_conversion_run_credits_the_rusher(agg):
    pbp = pd.DataFrame([
        _play(play_type="run", two_point_attempt=1,
              two_point_conv_result="success", rusher_player_id="RB1"),
    ])
    out = agg.aggregate_two_point_conversions(pbp)
    assert out.set_index("player_id").loc["RB1", "two_point_conversions"] == 1


def test_failed_conversions_are_not_credited(agg):
    pbp = pd.DataFrame([
        _play(two_point_attempt=1, two_point_conv_result="failure",
              passer_player_id="QB1", receiver_player_id="WR1"),
    ])
    assert agg.aggregate_two_point_conversions(pbp).empty


def test_multiple_conversions_in_a_week_accumulate(agg):
    pbp = pd.DataFrame([
        _play(two_point_attempt=1, two_point_conv_result="success",
              passer_player_id="QB1", receiver_player_id="WR1"),
        _play(two_point_attempt=1, two_point_conv_result="success",
              passer_player_id="QB1", receiver_player_id="TE1"),
    ])
    out = agg.aggregate_two_point_conversions(pbp).set_index("player_id")
    assert out.loc["QB1", "two_point_conversions"] == 2
    assert out.loc["WR1", "two_point_conversions"] == 1


def test_missing_two_point_columns_is_not_an_error(agg):
    pbp = pd.DataFrame([{"season": 2025, "week": 1, "play_type": "pass"}])
    assert agg.aggregate_two_point_conversions(pbp).empty


def test_conversion_plays_do_not_count_as_attempts_or_targets(agg):
    """A conversion is not an official attempt or target -- nflverse already
    zeroes its yards and completions, but it still carries play_type 'pass'
    and a play row, so counting plays folded it into the attempt totals."""
    plays = pd.DataFrame([
        _play(passer_player_id="QB1", receiver_player_id="WR1"),
        _play(two_point_attempt=1, two_point_conv_result="success",
              passer_player_id="QB1", receiver_player_id="WR1"),
    ])
    kept = agg._exclude_two_point_plays(plays)
    assert len(kept) == 1
    assert (kept["two_point_attempt"] == 0).all()


def test_two_points_reach_fantasy_points(agg):
    scored = agg.calculate_fantasy_points(pd.DataFrame([
        {"player_id": "WR1", "receiving_yards": 10.0, "receptions": 1,
         "two_point_conversions": 1},
        {"player_id": "WR2", "receiving_yards": 10.0, "receptions": 1,
         "two_point_conversions": 0},
    ]))
    delta = scored.loc[0, "fantasy_points"] - scored.loc[1, "fantasy_points"]
    assert delta == pytest.approx(2.0)

"""Tests for PBP advanced aggregation."""
import pandas as pd

from src.data.pbp_stats_aggregator import PBPStatsAggregator


def test_pbp_advanced_aggregation():
    pbp = pd.DataFrame([
        {
            "season": 2024,
            "week": 1,
            "play_type": "pass",
            "posteam": "KC",
            "passer_player_id": "QB1",
            "passer_player_name": "QB One",
            "receiver_player_id": "WR1",
            "receiver_player_name": "WR One",
            "passing_yards": 10,
            "receiving_yards": 10,
            "pass_touchdown": 0,
            "interception": 0,
            "complete_pass": 1,
            "play_id": 1,
            "air_yards": 8,
            "epa": 2.0,
            "wpa": 0.05,
            "success": 1,
            "score_differential": 3,
            "qtr": 2,
            "ydstogo": 5,
            "yardline_100": 30,
            "game_seconds_remaining": 100,
            "wp": 0.5,
            "down": 3,
        },
        {
            "season": 2024,
            "week": 1,
            "play_type": "run",
            "posteam": "KC",
            "rusher_player_id": "RB1",
            "rusher_player_name": "RB One",
            "rushing_yards": 5,
            "rush_touchdown": 0,
            "play_id": 2,
            "epa": -1.0,
            "wpa": -0.02,
            "success": 0,
            "score_differential": 3,
            "qtr": 2,
            "ydstogo": 1,
            "yardline_100": 4,
            "game_seconds_remaining": 110,
            "wp": 0.5,
            "down": 1,
        },
    ])

    agg = PBPStatsAggregator()
    stats = agg.aggregate_all_stats(pbp_df=pbp, include_advanced=True)

    # QB metrics
    qb = stats[stats["player_id"] == "QB1"].iloc[0]
    assert qb["pass_epa"] == 2.0
    assert qb["pass_wpa"] == 0.05
    assert qb["pass_success_rate"] == 1.0
    assert qb["pass_plays"] == 1

    # WR metrics
    wr = stats[stats["player_id"] == "WR1"].iloc[0]
    assert wr["recv_epa"] == 2.0
    assert wr["recv_wpa"] == 0.05
    assert wr["recv_success_rate"] == 1.0
    assert wr["neutral_targets"] == 1
    assert wr["third_down_targets"] == 1
    assert wr["two_minute_targets"] == 1
    assert wr["high_leverage_touches"] == 1

    # RB metrics
    rb = stats[stats["player_id"] == "RB1"].iloc[0]
    assert rb["rush_epa"] == -1.0
    assert rb["rush_wpa"] == -0.02
    assert rb["rush_success_rate"] == 0.0
    assert rb["short_yardage_rushes"] == 1
    assert rb["goal_line_touches"] == 1
    assert rb["high_leverage_touches"] == 1

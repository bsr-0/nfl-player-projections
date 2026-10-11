import numpy as np
import pandas as pd

from src.evaluation import weekly_replay as W


def _rows():
    return pd.DataFrame([
        {"player_id": "a", "name": "A.A", "team": "KC", "position": "QB", "predicted_points": 18.0,
         "prediction_ci80_lower": 8.0, "prediction_ci80_upper": 28.0},
        {"player_id": "b", "name": "B.B", "team": "KC", "position": "WR", "predicted_points": 9.0,
         "prediction_ci80_lower": 3.0, "prediction_ci80_upper": 15.0},
        {"player_id": "c", "name": "C.C", "team": "KC", "position": "WR", "predicted_points": 6.0,
         "prediction_ci80_lower": 1.0, "prediction_ci80_upper": 11.0},     # never played
        {"player_id": "k", "name": "K.K", "team": "KC", "position": "K", "predicted_points": 7.0,
         "prediction_ci80_lower": 1.0, "prediction_ci80_upper": 9.0},      # not a skill position
    ])


def _acts():
    return pd.DataFrame([
        {"player_id": "a", "name": "A.A", "team": "KC", "position": "QB", "actual_points": 20.0},
        {"player_id": "b", "name": "B.B", "team": "KC", "position": "WR", "actual_points": 2.0},
        {"player_id": "z", "name": "Z.Z", "team": "KC", "position": "WR", "actual_points": 11.0},   # no projection
    ])


def test_score_replay_scores_players_with_both_and_counts_the_rest():
    scored, unprojected, s = W.score_replay(_rows(), _acts())
    assert scored["player_id"].tolist() == ["a", "b"]
    assert s["n"] == 2 and s["mae"] == 4.5 and s["bias"] == 2.5      # projected minus actual: -2 and +7
    assert s["n_projected"] == 3 and s["n_no_stat_line"] == 1         # K dropped; C projected, no stat line
    assert s["n_unprojected"] == 1 and s["n_unprojected_5plus"] == 1
    assert unprojected["player_id"].tolist() == ["z"]
    assert s["coverage80"] == 0.5 and s["n_with_range"] == 2          # b's 2.0 is below its 3.0 floor
    assert set(s["by_position"]) == {"QB", "WR"}


def test_compare_with_published_uses_only_players_in_both():
    scored, _, _ = W.score_replay(_rows(), _acts())
    pub = pd.DataFrame([{"player_id": "a", "predicted_points": 14.0, "actual_points": 20.0},
                        {"player_id": "q", "predicted_points": 3.0, "actual_points": 3.0}])
    c = W.compare_with_published(scored, pub)
    assert c["n_common"] == 1 and c["mae_published"] == 6.0 and c["mae_replay"] == 2.0
    assert c["bias_published"] == -6.0 and c["bias_replay"] == -2.0 and c["mean_abs_change"] == 4.0
    assert W.compare_with_published(scored, pub.iloc[0:0]) is None

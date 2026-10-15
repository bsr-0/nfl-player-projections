"""WR feature-candidate harness: as-of features and the paired comparison, on synthetic rows only."""
import numpy as np
import pandas as pd
import pytest

from scripts import wr_feature_candidates_2025 as w


def _stats():
    rows = []
    for season, weeks in ((2023, [1, 2, 3]), (2024, [1, 2, 4])):          # 2024 has a bye at 3
        for wk, t in zip(weeks, (4, 8, 12)):
            rows.append(dict(player_id="a", season=season, week=wk, fantasy_points=float(t), targets=t, receptions=t // 2,
                             receiving_yards=10 * t, neutral_targets=t - 1, high_leverage_touches=t // 4))
    return pd.DataFrame(rows)


def test_season_to_date_is_an_expanding_mean_that_resets_each_season():
    out = w.season_to_date(_stats()).set_index(["season", "week"])
    assert out.loc[(2024, 1), "targets_s2d"] == 4 and out.loc[(2024, 2), "targets_s2d"] == 6
    assert out.loc[(2024, 4), "targets_s2d"] == pytest.approx(8) and out.loc[(2024, 4), "games_s2d"] == 3
    assert out.loc[(2023, 3), "receiving_yards_s2d"] == pytest.approx(80)       # not mixed with 2024
    assert out.loc[(2024, 2), "ppg_s2d"] == 6


def test_a_row_never_depends_on_a_later_game():
    full = w.season_to_date(_stats())
    cut = w.season_to_date(_stats().query("not (season == 2024 and week == 4)"))
    merged = cut.merge(full, on=w.KEY, suffixes=("_cut", "_full"))
    for f in list(w.SOURCE) + ["games_s2d", "ppg_s2d"]:
        assert (merged[f + "_cut"] == merged[f + "_full"]).all()


def test_duplicate_stat_rows_keep_one_per_player_week():
    s = pd.concat([_stats(), _stats().iloc[[0]]])
    assert len(w.season_to_date(s)) == len(_stats())


def test_paired_interval_brackets_the_direct_mean_difference():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.1, 1.0, 400)
    week, player = np.repeat(np.arange(1, 9), 50), np.tile(np.arange(50), 8)
    from src.evaluation.selection_gates import _weight_batches
    W = np.vstack(list(_weight_batches(week, player, 500, 1)))
    r = w.paired(d, W)
    assert r["est"] == pytest.approx(d.mean()) and r["lo"] < r["est"] < r["hi"]
    assert r["upper_95_one_sided"] < r["hi"]


def test_score_window_pairs_a_better_arm_and_sees_the_form_lag():
    rng = np.random.default_rng(3)
    n = 600
    d = pd.DataFrame({"player_id": [f"p{i % 60}" for i in range(n)], "season": 2025,
                      "target_week": np.repeat(np.arange(1, 11), 60), "gkey": [f"g{i // 6}" for i in range(n)]})
    truth = rng.gamma(3, 3, n)
    d["actual_points"] = truth + rng.normal(0, 4, n)
    d["games_s2d"] = 4
    d["C0"] = truth * 0.8 + rng.normal(0, 1, n)
    d["ppg_s2d"] = truth
    d["C1"] = truth + rng.normal(0, 1, n)                      # closer to the signal
    out = w.score_window(d, (1, 10), ["C0", "C1"])
    assert out["arms"]["C1"]["mae_diff_vs_C0"]["est"] < 0 and out["arms"]["C1"]["mae"] < out["arms"]["C0"]["mae"]
    assert abs(out["arms"]["C1"]["form_lag_slope"]) < abs(out["arms"]["C0"]["form_lag_slope"])
    assert out["arms"]["C0"]["mae_diff_vs_C0"] is None

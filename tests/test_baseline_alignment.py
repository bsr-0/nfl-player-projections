"""compare_model_to_baselines must not depend on the caller's row order.

The baselines sort internally; positional_rank_baseline even comes back
positionally aligned to its own sorted order (a merge resets the index),
while the comparison is positional against the caller's order. An unsorted
frame therefore scored the model against scrambled baselines -- the
serving-path backtest reported a fictitious +38% over trailing averages
(2026-09-11) where the internally sorted routine said +2.5%.
"""
import numpy as np
import pandas as pd

from src.evaluation.baselines import compare_model_to_baselines, trailing_average_baseline


def _season_frame(seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for season in (2024, 2025):
        for pid in range(60):
            base = rng.uniform(3, 20)
            for week in range(1, 19):
                rows.append({"player_id": f"P{pid:02d}", "position": ["QB", "RB", "WR", "TE"][pid % 4],
                             "season": season, "week": week, "team": "AAA",
                             "fantasy_points": max(0.0, base + rng.normal(0, 4))})
    return pd.DataFrame(rows)


def test_row_order_does_not_change_the_comparison():
    df = _season_frame()
    preds = trailing_average_baseline(df, n_weeks=3) * 1.05 + 0.5   # some model, aligned to df.index
    scored = df["season"] == 2025
    preds = preds.where(scored)                                       # only the test season is predicted

    sorted_result = compare_model_to_baselines(df, preds, target_col="fantasy_points")

    shuffled = df.sample(frac=1.0, random_state=7)
    shuffled_result = compare_model_to_baselines(shuffled, preds.loc[shuffled.index], target_col="fantasy_points")

    assert sorted_result.keys() == shuffled_result.keys()
    for name in sorted_result:
        for k in ("baseline_rmse", "model_rmse", "n_compared"):
            assert sorted_result[name][k] == shuffled_result[name][k], (name, k)


def test_a_model_equal_to_a_baseline_shows_no_improvement_over_it():
    """Alignment sanity: if the model IS the trailing-3g average, its
    improvement over that baseline must be ~0, in any row order."""
    df = _season_frame(seed=3)
    preds = trailing_average_baseline(df, n_weeks=3).where(df["season"] == 2025)
    shuffled = df.sample(frac=1.0, random_state=11)

    result = compare_model_to_baselines(shuffled, preds.loc[shuffled.index], target_col="fantasy_points")

    assert abs(result["trailing_3g_avg"]["rmse_improvement_pct"]) < 1e-6
    assert result["trailing_3g_avg"]["n_compared"] == int(preds.notna().sum())


def test_rows_without_a_prediction_are_history_only():
    df = _season_frame(seed=5)
    preds = pd.Series(np.nan, index=df.index)
    preds[df["season"] == 2025] = 8.0
    result = compare_model_to_baselines(df, preds, target_col="fantasy_points")
    # 2025 rows only; the season-to-date average is undefined at week 1, so
    # those 60 player-rows are excluded from this particular comparison.
    predicted_2025 = int((df["season"] == 2025).sum())
    assert result["season_avg"]["n_compared"] == predicted_2025 - 60
    assert result["prior_season_rank"]["n_compared"] == predicted_2025

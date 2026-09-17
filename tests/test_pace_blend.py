"""The served weekly number is the ensemble shrunk toward the Step 8 pace.

    served = w * weekly + (1 - w) * pace,   w = g / (g + kappa)

kappa=3 chosen out-of-sample on the 2025 walk-forward (config.settings).
These pin the algebra and the edge cases: g comes from the frame the
features were built from (so a backtest as_of week w counts only games
before w), a player without a pace row is served unblended, intervals move
with the centre, and n_weeks scales the pace.
"""
import numpy as np
import pandas as pd
import pytest

import config.settings as settings
from src.predict import NFLPredictor


@pytest.fixture
def predictor(monkeypatch):
    monkeypatch.setattr(settings, "PACE_BLEND_KAPPA", 3.0)
    monkeypatch.setattr(settings, "PACE_BLEND_CI_SCALE", 1.5)
    monkeypatch.setattr(NFLPredictor, "_PACE_TABLE_CACHE", pd.DataFrame({
        "player_id": ["vet", "vet", "rookie"],
        "season": [2024.0, 2025.0, 2025.0],
        "step8_pace": [20.0, 10.0, 6.0],
    }))
    return NFLPredictor.__new__(NFLPredictor)


def _history(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "week"])


def _results(**pts):
    return pd.DataFrame({
        "player_id": list(pts), "predicted_points": list(pts.values()),
        "prediction_ci80_lower": [v - 2 for v in pts.values()],
        "prediction_ci80_upper": [v + 2 for v in pts.values()],
    })


def test_weight_follows_games_played_this_season(predictor):
    hist = _history([("vet", 2024, w) for w in range(1, 18)] + [("vet", 2025, 1), ("vet", 2025, 2), ("vet", 2025, 3)])
    out = predictor._blend_toward_season_pace(_results(vet=16.0), hist, 2025, n_weeks=1)
    # g = 3 (2024 games do not count) -> w = 0.5
    assert out.loc[0, "games_played_season"] == 3
    assert out.loc[0, "pace_weight"] == pytest.approx(0.5)
    assert out.loc[0, "predicted_points"] == pytest.approx(0.5 * 16 + 0.5 * 10)
    assert out.loc[0, "predicted_points_model"] == 16.0
    assert out.loc[0, "pace_prior"] == 10.0, "the 2025 pace, not 2024's"


def test_week_one_is_the_pace_exactly(predictor):
    hist = _history([("rookie", 2024, 17)])   # nothing played in 2025
    out = predictor._blend_toward_season_pace(_results(rookie=13.0), hist, 2025, n_weeks=1)
    assert out.loc[0, "pace_weight"] == 1.0
    assert out.loc[0, "predicted_points"] == 6.0


def test_no_pace_row_means_unblended(predictor):
    hist = _history([("nobody", 2025, 1)])
    out = predictor._blend_toward_season_pace(_results(nobody=9.0), hist, 2025, n_weeks=1)
    assert out.loc[0, "pace_weight"] == 0.0
    assert out.loc[0, "predicted_points"] == 9.0
    assert np.isnan(out.loc[0, "pace_prior"])


def test_intervals_recentre_and_rescale_on_blended_rows(predictor):
    hist = _history([("vet", 2025, 1)])         # g = 1 -> w = 0.25
    out = predictor._blend_toward_season_pace(_results(vet=20.0), hist, 2025, n_weeks=1)
    centre = 0.25 * 20 + 0.75 * 10
    assert out.loc[0, "predicted_points"] == pytest.approx(centre)
    # half-width 2 from the weekly model, times PACE_BLEND_CI_SCALE (1.5 here)
    assert out.loc[0, "prediction_ci80_lower"] == pytest.approx(centre - 3)
    assert out.loc[0, "prediction_ci80_upper"] == pytest.approx(centre + 3)


def test_unblended_rows_keep_the_weekly_interval(predictor):
    out = predictor._blend_toward_season_pace(_results(nobody=9.0), _history([]), 2025, n_weeks=1)
    assert out.loc[0, "prediction_ci80_lower"] == 7.0
    assert out.loc[0, "prediction_ci80_upper"] == 11.0


def test_multi_week_scales_the_pace(predictor):
    hist = _history([])
    out = predictor._blend_toward_season_pace(_results(vet=40.0), hist, 2025, n_weeks=4)
    assert out.loc[0, "predicted_points"] == 40.0, "g=0 -> 4 weeks of pace = 4 * 10"


def test_missing_table_is_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(NFLPredictor, "_PACE_TABLE_CACHE", None)
    monkeypatch.setattr(settings, "STEP8_PACE_TABLE", tmp_path / "absent.csv")
    p = NFLPredictor.__new__(NFLPredictor)
    with pytest.raises(FileNotFoundError, match="build_step8_pace_table"):
        p._blend_toward_season_pace(_results(vet=1.0), _history([]), 2025, 1)

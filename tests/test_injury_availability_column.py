"""Injury availability is reported beside the prediction, never folded into
it, and comes from the REAL practice-report status for the target week --
not from injury_prob_combined.

predicted_points is points-given-the-player-plays: what the models are
trained on and what the serving-path walk-forward measures. Multiplying it
by (1 - injury_prob_combined) -- a synthetic, uncalibrated injury-HAZARD
estimate that never fell below ~0.75-0.85 availability for anyone, healthy
players included -- was a uniform ~19% haircut on every served number
(2026-09-11). injury_score already carries the real signal (Out=0.0 ..
Probable=0.85 .. no report/healthy=1.0) and is what a veteran statistician
would use: the actual report, not a re-predicted proxy for it.
"""
import numpy as np
import pandas as pd

from src.predict import NFLPredictor


def _frames():
    results = pd.DataFrame({
        "player_id": ["A", "B", "C", "D"],
        "predicted_points": [20.0, 10.0, 5.0, 8.0],
        "predicted_ppg": [20.0, 10.0, 5.0, 8.0],
        "predicted_utilization": [80.0, 50.0, 30.0, 40.0],
        "prediction_ci80_lower": [12.0, 6.0, 2.0, 4.0],
        "prediction_ci80_upper": [28.0, 14.0, 8.0, 12.0],
    })
    latest = pd.DataFrame({
        "player_id": ["A", "B", "C", "D"],
        "injury_score": [0.0, 1.0, np.nan, 0.5],       # A=Out, B=healthy, C=no report, D=Questionable
        "injury_prob_combined": [0.16, 0.18, 0.15, 0.20],  # synthetic hazard: uniformly ~0.15-0.2 regardless
    })
    return results, latest


def test_prediction_columns_are_not_discounted():
    results, latest = _frames()
    before = results.copy()
    out = NFLPredictor._attach_injury_availability(results, latest)
    for col in ("predicted_points", "predicted_ppg", "predicted_utilization",
                "prediction_ci80_lower", "prediction_ci80_upper"):
        pd.testing.assert_series_equal(out[col], before[col])


def test_availability_comes_from_injury_score_not_the_hazard_estimate():
    results, latest = _frames()
    out = NFLPredictor._attach_injury_availability(results, latest)
    # A is actually ruled Out this week -> fully zeroed, despite a LOW hazard estimate (0.16).
    assert out.loc[0, "injury_adjustment"] == 0.0
    assert out.loc[0, "expected_points"] == 0.0
    # B is healthy -> full value, despite the hazard estimate being nonzero (0.18).
    assert out.loc[1, "injury_adjustment"] == 1.0
    assert out.loc[1, "expected_points"] == 10.0
    # C has no report for the target week -> defaults to healthy, not discounted.
    assert out.loc[2, "injury_adjustment"] == 1.0
    # D is Questionable (0.5) -> half credit.
    assert out.loc[3, "injury_adjustment"] == 0.5
    assert out.loc[3, "expected_points"] == 4.0
    # The hazard estimate is still exposed, unmodified, as information.
    assert out["injury_prob_combined"].tolist() == [0.16, 0.18, 0.15, 0.20]


def test_missing_injury_score_is_a_passthrough_for_availability():
    results, latest = _frames()
    out = NFLPredictor._attach_injury_availability(results, latest.drop(columns=["injury_score"]))
    assert "expected_points" not in out.columns and "injury_adjustment" not in out.columns
    assert "injury_prob_combined" in out.columns   # still exposed independently


def test_missing_injury_prob_combined_is_a_passthrough_for_that_column():
    results, latest = _frames()
    out = NFLPredictor._attach_injury_availability(results, latest.drop(columns=["injury_prob_combined"]))
    assert "injury_prob_combined" not in out.columns
    assert "expected_points" in out.columns   # availability still computed from injury_score

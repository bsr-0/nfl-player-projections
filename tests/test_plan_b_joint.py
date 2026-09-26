"""The grouped model must conserve share mass and respect seasonal cutoffs."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plan_b_joint_backtester import KEYS, run_backtest
from src.models.team_hierarchical.joint_other import GROUP, JointOtherShareModel


def _panel():
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 5):
            for team_i in range(4):
                for rank, position in enumerate(("RB", "WR", "TE"), 1):
                    prior = (0.25, 0.42, 0.15)[rank - 1]
                    actual = prior + (0.02 if week % 2 else -0.02) * (2 - rank)
                    rows.append({"player_id": f"p{team_i}_{rank}", "team": f"T{team_i}",
                                 "position": position, "season": season, "week": week,
                                 "slot": f"{position}1", "depth_chart_rank": rank,
                                 "roster_snap_share_s2d": 0.6 - rank * 0.1,
                                 "is_cold_start": int(week == 1),
                                 "share_of_team_targets": actual,
                                 "share_of_team_targets_roll3": prior if week > 1 else np.nan,
                                 "share_of_team_targets_s2d": prior,
                                 "team_targets": 30.0,
                                 "team_targets_roll3": 29.0,
                                 "team_targets_s2d": 28.0})
    return pd.DataFrame(rows)


def test_model_allocates_one_with_explicit_other_and_no_clipping():
    panel = _panel()
    model = JointOtherShareModel("targets").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023]
    predicted, prior, other = model.predict(test)
    assert model.fit_diagnostics_["converged"]
    assert np.isfinite(predicted).all() and (predicted > 0).all() and (predicted < 1).all()
    assert np.isfinite(prior).all() and (prior > 0).all()
    group, n = model._groups(test)
    np.testing.assert_allclose(np.bincount(group, weights=predicted, minlength=n) + other, 1)
    assert (other > 0).all()


def test_future_labels_cannot_change_prior_fold_predictions():
    panel = _panel()
    rows, report = run_backtest(panel, "targets", [2022, 2023], n_bootstrap=100)
    changed = panel.copy()
    changed.loc[changed.season == 2023, "share_of_team_targets"] = 0.1
    again, _ = run_backtest(changed, "targets", [2022, 2023], n_bootstrap=100)
    for arm in ("rolling3", "normalized_prior", "joint_other"):
        np.testing.assert_allclose(rows[arm], again[arm], atol=1e-12)
    pd.testing.assert_frame_equal(rows[KEYS], again[KEYS])
    assert report["status"] == "complete"


def test_zero_total_week_is_excluded_from_fit_without_dropping_test_rows():
    panel = _panel()
    mask = (panel.season == 2018) & (panel.week == 1)
    panel.loc[mask, "team_targets"] = 0
    panel.loc[mask, "share_of_team_targets"] = 0
    model = JointOtherShareModel("targets").fit(panel[panel.season < 2022])
    assert model.fit_diagnostics_["zero_total_rows_excluded_from_fit"] == int(mask.sum())
    rows, _ = run_backtest(panel, "targets", [2022], n_bootstrap=100)
    assert len(rows) == int((panel.season == 2022).sum())


@pytest.mark.parametrize("problem", ["duplicate", "nonfinite", "mass", "team_total"])
def test_bad_grouped_inputs_fail_loudly(problem):
    panel = _panel()
    if problem == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]], ignore_index=True)
    elif problem == "nonfinite":
        panel.loc[0, "roster_snap_share_s2d"] = np.inf
    elif problem == "mass":
        panel.loc[0, "share_of_team_targets"] = 1.0
    else:
        panel.loc[0, "team_targets"] = 31
    with pytest.raises((ValueError, RuntimeError)):
        JointOtherShareModel("targets").fit(panel)


def test_prediction_ignores_current_week_labels_and_team_totals():
    panel = _panel()
    model = JointOtherShareModel("targets").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023].copy()
    before = model.predict(test)[0]
    test["share_of_team_targets"] = 0.0
    test["team_targets"] = 0.0
    np.testing.assert_allclose(before, model.predict(test)[0])

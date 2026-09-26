"""Checks for the smooth-absolute-error Plan B joint allocation arm."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plan_b_joint_mae_backtester import run_backtest
from src.models.team_hierarchical.joint_other_mae import JointOtherMAEModel


def _panel():
    rows = []
    for season in range(2018, 2024):
        for week in range(1, 5):
            for team in ("A", "B"):
                for rank, position in enumerate(("RB", "WR", "TE"), 1):
                    actual = (0.30, 0.35, 0.00)[rank - 1]
                    rows.append({"player_id": f"{team}{rank}", "team": team,
                                 "position": position, "season": season, "week": week,
                                 "slot": f"{position}1", "depth_chart_rank": rank,
                                 "roster_snap_share_s2d": 0.6 - rank * 0.1,
                                 "is_cold_start": int(week == 1),
                                 "share_of_team_targets": actual,
                                 "share_of_team_targets_roll3": (0.28, 0.34, 0.01)[rank - 1],
                                 "share_of_team_targets_s2d": (0.29, 0.33, 0.01)[rank - 1],
                                 "team_targets": 30.0,
                                 "team_targets_roll3": 29.0,
                                 "team_targets_s2d": 28.0})
    return pd.DataFrame(rows)


def test_analytic_gradient_matches_central_difference():
    panel = _panel()
    model = JointOtherMAEModel("targets")
    prepared = model._prepare(panel, fit=True)
    y = panel[model.label].to_numpy(float)
    group, n_groups = model._groups(panel)
    other_y = 1 - np.bincount(group, weights=y, minlength=n_groups)
    rng = np.random.default_rng(13)
    coef = rng.normal(0, 0.15, prepared[1].shape[1] + prepared[2].shape[1])
    loss, gradient = model._objective(prepared, y, other_y, coef)
    assert np.isfinite(loss)
    numerical = np.zeros_like(coef)
    for i in range(len(coef)):
        shifted = coef.copy()
        shifted[i] += 1e-6
        higher = model._objective(prepared, y, other_y, shifted)[0]
        shifted[i] -= 2e-6
        lower = model._objective(prepared, y, other_y, shifted)[0]
        numerical[i] = (higher - lower) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)


def test_mass_conservation_and_zero_player_rows():
    panel = _panel()
    model = JointOtherMAEModel("targets").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023]
    player, _, other = model.predict(test)
    group, n_groups = model._groups(test)
    np.testing.assert_allclose(np.bincount(group, weights=player, minlength=n_groups) + other, 1)
    assert model.fit_diagnostics_["converged"]
    assert (player >= 0).all() and (player <= 1).all()


def test_future_labels_cannot_change_earlier_fold():
    panel = _panel()
    before, _ = run_backtest(panel, "targets", [2022, 2023], n_bootstrap=100)
    changed = panel.copy()
    changed.loc[changed.season == 2023, "share_of_team_targets"] = 0.1
    after, _ = run_backtest(changed, "targets", [2022, 2023], n_bootstrap=100)
    early = before.season.eq(2022)
    for arm in ("rolling3", "normalized_prior", "joint_other_mae"):
        np.testing.assert_allclose(before.loc[early, arm], after.loc[early, arm], atol=1e-12)


@pytest.mark.parametrize("width", [0, -1, float("nan"), float("inf")])
def test_bad_smoothing_width_fails(width):
    with pytest.raises(ValueError):
        JointOtherMAEModel("targets", smooth_width=width)

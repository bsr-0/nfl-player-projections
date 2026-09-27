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


def test_unseen_slot_falls_back_to_same_position_rank1_not_a_silent_zero_vector():
    """Uncapping (2026-09-25) means a rare, deep-bench slot like WR3 can
    legitimately appear at prediction time without ever appearing in a
    shorter training window. Before the fix, an unseen slot silently got an
    all-zero one-hot row -- i.e. whichever slot is the lexicographically-first
    reference category. In this panel's vocabulary (sorted: RB1, TE1, WR1),
    that reference category is RB1, the WRONG position entirely for a WR row
    -- exactly the bug: the pre-fix behavior would have made an unseen WR3
    row predict identically to an RB1 row, not a WR1 row. The fix must fall
    back to the row's own position's rank-1 slot (WR1) instead, and count it."""
    panel = _panel()  # training only ever has RB1/WR1/TE1 -- no WR3 ever
    model = JointOtherShareModel("targets").fit(panel[panel.season < 2023])
    test = panel[panel.season == 2023].copy()
    unseen_row = test[(test.team == "T0") & (test.position == "WR")].iloc[[0]].copy()
    unseen_row["player_id"] = "unseen_wr3"
    unseen_row["slot"] = "WR3"  # never present in any training row
    combined = pd.concat([test, unseen_row], ignore_index=True)

    predicted, _, _ = model.predict(combined)
    assert model.prediction_diagnostics_["fallback_slot_rows"] == 1

    wr1_idx = combined.index[(combined.team == "T0") & (combined.slot == "WR1")
                              & (combined.week == unseen_row.week.iloc[0])
                              & (combined.season == unseen_row.season.iloc[0])][0]
    rb1_idx = combined.index[(combined.team == "T0") & (combined.slot == "RB1")
                              & (combined.week == unseen_row.week.iloc[0])
                              & (combined.season == unseen_row.season.iloc[0])][0]
    fallback_idx = combined.index[combined.player_id == "unseen_wr3"][0]
    # Same team-week, same numeric features as the WR1 row it was copied from
    # -- an unseen WR3 falling back to WR1's fitted coefficients must predict
    # identically to an actual WR1 row with the same features, NOT identically
    # to RB1 (which is what the pre-fix all-zero-vector bug would have done).
    assert predicted[fallback_idx] == pytest.approx(predicted[wr1_idx])
    assert predicted[fallback_idx] != pytest.approx(predicted[rb1_idx])


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

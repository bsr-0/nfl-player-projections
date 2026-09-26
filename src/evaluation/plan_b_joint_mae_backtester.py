"""Past-season-only evaluation of the MAE-fitted joint Plan B share model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.paired_ppr_comparison import paired_week_interval
from src.evaluation.team_hierarchical_backtester import KEYS, validate_panel
from src.models.team_hierarchical.joint_other import GROUP, JointOtherShareModel
from src.models.team_hierarchical.joint_other_mae import JointOtherMAEModel

ARMS = ("rolling3", "normalized_prior", "joint_other_mae")


def _metrics(rows: pd.DataFrame) -> dict:
    return {arm: {"n": len(rows), "mae": float(np.abs(rows[arm] - rows.actual_share).mean())}
            for arm in ARMS}


def run_backtest(panel: pd.DataFrame, target: str, seasons: list[int], *,
                 epsilon: float = 0.001, penalty: float = 0.00001,
                 smooth_width: float = 0.002, n_bootstrap: int = 2000,
                 seed: int = 42) -> tuple[pd.DataFrame, dict]:
    meta = validate_panel(panel, target, seasons)
    if type(n_bootstrap) is not int or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer >= 100")
    label = f"share_of_team_{target}"
    total = f"team_{target}"
    if total not in panel:
        raise ValueError(f"missing team-total label {total}")
    team_total = pd.to_numeric(panel[total], errors="raise").to_numpy(float)
    if not np.isfinite(team_total).all() or (team_total < 0).any():
        raise ValueError("invalid team-total labels")
    if panel.groupby(GROUP)[total].nunique().gt(1).any():
        raise ValueError("team totals differ within a team-week")
    if (panel.groupby(GROUP)[label].sum() > 1 + 1e-8).any():
        raise ValueError("represented shares exceed full-team mass")
    panel = panel.sort_values(KEYS).reset_index(drop=True)
    outputs = []
    for fold in meta["folds"]:
        season = fold["test_season"]
        train = panel[panel.season < season].reset_index(drop=True)
        test = panel[panel.season == season].reset_index(drop=True)
        model = JointOtherMAEModel(target, epsilon=epsilon, penalty=penalty,
                                   smooth_width=smooth_width).fit(train)
        candidate, prior, other = model.predict(test)
        out = test[KEYS + ["slot"]].copy()
        out["train_end_season"] = int(train.season.max())
        out["actual_share"] = test[label].to_numpy(float)
        out["rolling3"] = test[f"{label}_roll3"].fillna(0).to_numpy(float)
        out["normalized_prior"] = prior
        out["joint_other_mae"] = candidate
        out["predicted_other_mass"] = other[JointOtherShareModel._groups(test)[0]]
        if not np.isfinite(out[list(ARMS) + ["predicted_other_mass"]].to_numpy(float)).all():
            raise ValueError(f"nonfinite prediction for {season}")
        if ((out[list(ARMS) + ["predicted_other_mass"]] < 0)
                | (out[list(ARMS) + ["predicted_other_mass"]] > 1)).any().any():
            raise ValueError(f"out-of-bounds prediction for {season}")
        fold["fit"] = model.fit_diagnostics_
        fold["first_week_rolling3_null_to_zero_rows"] = int(test[f"{label}_roll3"].isna().sum())
        fold["n_team_weeks"] = int(test.groupby(GROUP).ngroups)
        outputs.append(out)
    rows = pd.concat(outputs, ignore_index=True)
    expected = panel.loc[panel.season.isin(seasons), KEYS].sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(rows[KEYS].sort_values(KEYS).reset_index(drop=True), expected)
    mass = rows.groupby(GROUP)[["joint_other_mae", "predicted_other_mass", "actual_share"]].agg(
        {"joint_other_mae": "sum", "predicted_other_mass": "first", "actual_share": "sum"})
    if not np.allclose(mass.joint_other_mae + mass.predicted_other_mass, 1.0, atol=1e-10):
        raise AssertionError("team-week allocation does not conserve mass")
    meta["pooled"] = _metrics(rows)
    for col in ("season", "position"):
        meta[f"by_{col}"] = {str(k): _metrics(v) for k, v in rows.groupby(col)}
    meta["by_actual_zero"] = {str(k): _metrics(v) for k, v in rows.groupby(rows.actual_share.eq(0))}
    meta["comparisons"] = {}
    for control in ("rolling3", "normalized_prior"):
        paired = rows[["season", "week"]].copy()
        paired["delta_abs_error"] = (np.abs(rows.joint_other_mae - rows.actual_share)
                                     - np.abs(rows[control] - rows.actual_share))
        meta["comparisons"][f"joint_other_mae_minus_{control}"] = paired_week_interval(
            paired, n_bootstrap, seed)
    meta.update(status="complete", model="joint_softmax_smooth_player_mae_with_other_bucket",
                epsilon=epsilon, penalty=penalty, smooth_width=smooth_width,
                represented_team_mass={"actual_mean": float(mass.actual_share.mean()),
                                       "candidate_mean": float(mass.joint_other_mae.mean()),
                                       "candidate_mass_mae": float(np.abs(mass.joint_other_mae - mass.actual_share).mean()),
                                       "other_mean": float(mass.predicted_other_mass.mean())},
                limitations=["Capped roster share evaluation; no full-population or PPR result.",
                             "2023-2025 seasons were inspected previously; any later run is iterative confirmation.",
                             "This model does not estimate team volume or served fantasy points."])
    return rows, meta

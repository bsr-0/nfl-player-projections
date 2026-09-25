"""Checked, raw eight-component PPR truth for Plan A evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.team_allocation.features import FULL_PPR_VOLUME_COLS, filter_population
from src.utils.helpers import calculate_fantasy_points_df

TARGETS = ["rushing_yards", "receiving_yards", *FULL_PPR_VOLUME_COLS]
IDENTITY = ["player_id", "season", "week", "team", "position"]
KEY = [*IDENTITY, "fold"]
TEAM_KEY = ["fold", "team", "season", "week"]


def checked_truth(raw: pd.DataFrame, oof_keys: pd.DataFrame) -> pd.DataFrame:
    """Return one raw PPR label per OOF row, never reconstructed from shares."""
    required = set(IDENTITY + TARGETS)
    if required.difference(raw.columns) or set(KEY).difference(oof_keys.columns):
        raise ValueError("raw truth or OOF keys missing required columns")
    if raw.duplicated(IDENTITY).any() or oof_keys.duplicated(KEY).any():
        raise ValueError("duplicate raw truth or OOF player-week key")
    truth = oof_keys[KEY].merge(raw[IDENTITY + TARGETS], on=IDENTITY, how="left", validate="one_to_one", indicator=True)
    if not truth._merge.eq("both").all():
        raise ValueError("held-out player-week missing raw truth")
    values = truth[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("raw PPR component is missing or nonfinite")
    truth = truth.drop(columns="_merge")
    truth[TARGETS] = values
    truth["actual_ppr"] = calculate_fantasy_points_df(truth[TARGETS])
    truth["negative_yards"] = truth[["rushing_yards", "receiving_yards"]].lt(0).any(axis=1)
    truth["off_position_stat"] = (
        (truth.position.eq("QB") & truth[["receiving_yards", "receptions", "receiving_tds"]].ne(0).any(axis=1))
        | (truth.position.ne("QB") & truth[["passing_yards", "passing_tds", "interceptions"]].ne(0).any(axis=1))
    )
    return truth


def validate_inputs(allocation_results: dict, team_results: dict, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fail closed on saved OOF lineage and return its canonical truth frame."""
    canonical = allocation_results["rushing_yards"]["predictions"]
    canonical = canonical[canonical.arm.eq("rolling3")]
    truth = checked_truth(raw, canonical[KEY])
    if not len(truth):
        raise ValueError("no held-out rows")
    audit = {"rows": len(truth), "folds": {}, "targets": {}}
    for fold, part in truth.groupby("fold"):
        if part.season.nunique() != 1:
            raise ValueError(f"fold {fold} spans multiple seasons")
        audit["folds"][str(int(fold))] = {"season": int(part.season.iloc[0]), "rows": len(part)}
    for target in TARGETS:
        expected = filter_population(truth, target)[KEY]
        pred = allocation_results[target]["predictions"]
        if pred[KEY + ["arm", "actual_share"]].isna().any().any() or pred.duplicated(KEY + ["arm"]).any():
            raise ValueError(f"{target}: missing identity/label/arm or duplicate allocation")
        arms = pred.arm.unique()
        for arm in arms:
            rows = pred[pred.arm.eq(arm)]
            merged = expected.merge(rows[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
            if not merged._merge.eq("both").all() or not np.isfinite(rows.predicted_share.to_numpy(float)).all():
                raise ValueError(f"{target}/{arm}: missing, extra, or nonfinite allocation prediction")
            labels = rows[KEY + ["actual_share"]].merge(
                filter_population(raw, target)[IDENTITY + [f"share_of_team_{target}"]],
                on=IDENTITY, how="left", validate="one_to_one",
            )
            if not np.allclose(labels.actual_share, labels[f"share_of_team_{target}"], rtol=0, atol=1e-9):
                raise ValueError(f"{target}/{arm}: saved share labels differ from current table")
        totals = team_results[target]["predictions"]
        expected_team = truth[TEAM_KEY].drop_duplicates()
        if totals.duplicated(TEAM_KEY).any():
            raise ValueError(f"{target}: duplicate team-week forecast")
        merged = expected_team.merge(totals[TEAM_KEY], on=TEAM_KEY, how="outer", indicator=True, validate="one_to_one")
        if not merged._merge.eq("both").all():
            raise ValueError(f"{target}: team-week forecast population differs")
        actual = totals[TEAM_KEY + ["actual"]].merge(
            raw[["team", "season", "week", f"team_{target}"]].drop_duplicates(["team", "season", "week"]),
            on=["team", "season", "week"], how="left", validate="one_to_one",
        )
        if not np.allclose(actual.actual, actual[f"team_{target}"], rtol=0, atol=1e-9):
            raise ValueError(f"{target}: saved team totals differ from current table")
        for arm in ("ridge", "xgb", "blend"):
            if arm in totals and not np.isfinite(totals[arm].to_numpy(float)).all():
                raise ValueError(f"{target}/{arm}: nonfinite team forecast")
        missing_roll = totals.rolling3.isna()
        if (missing_roll & totals.week.ne(1)).any():
            raise ValueError(f"{target}: unexpected rolling3 team-total null")
        audit["targets"][target] = {"allocation_arms": len(arms), "allocation_rows_per_arm": len(expected),
                                    "team_weeks": len(totals), "rolling3_week1_null_team_weeks": int(missing_roll.sum())}
    audit["negative_yard_rows"] = int(truth.negative_yards.sum())
    audit["off_position_stat_rows"] = int(truth.off_position_stat.sum())
    old = truth[KEY].copy()
    indexed = raw.set_index(IDENTITY)
    for target in TARGETS:
        values = indexed.reindex(pd.MultiIndex.from_frame(old[IDENTITY]))
        old[target] = (values[f"share_of_team_{target}"].to_numpy(float)
                       * values[f"team_{target}"].to_numpy(float))
        if target in {"receiving_yards", "receptions", "receiving_tds"}:
            old.loc[old.position.eq("QB"), target] = 0.0
        elif target in {"passing_yards", "passing_tds", "interceptions"}:
            old.loc[old.position.ne("QB"), target] = 0.0
    former = calculate_fantasy_points_df(old[TARGETS])
    delta = truth.actual_ppr.to_numpy(float) - former.to_numpy(float)
    audit["former_label_discrepancy"] = {
        "changed_rows": int(np.count_nonzero(delta)),
        "mean_absolute_points": float(np.abs(delta).mean()),
        "max_absolute_points": float(np.abs(delta).max()),
    }
    return truth, audit

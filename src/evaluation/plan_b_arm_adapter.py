"""Reshape Plan B's joint-MAE predictions into Plan A's full-PPR arm schema.

Plan A's `joint_ppr_selector.py`/`ppr_truth.py` are already arm-agnostic --
any candidate arm just needs to be a long-format row set
(player_id, season, week, team, position, is_cold_start, fold, arm,
actual_share, predicted_share) with the same population and labels as every
other arm for that target. This module only reshapes and validates; it does
not fit anything and does not modify `full_ppr_allocation_backtester.py` or
`joint_ppr_selector.py`.

Fold numbering must come from instantiating `SeasonAwareTimeSeriesSplit`
itself, never hand-derived arithmetic -- a second implementation of "which
season is fold N" would silently drift from Plan A's real one the moment
`--seasons`/`--n-test-seasons`/`cv_gap_seasons` changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.position_models import SeasonAwareTimeSeriesSplit
from src.models.team_allocation.features import filter_population

KEY = ["player_id", "season", "week", "team", "position", "fold"]
PLAN_B_ROW_COLS = ["player_id", "season", "week", "team", "position", "is_cold_start"]


def season_to_fold_map(seasons: list[int], n_test_seasons: int, gap_seasons: int) -> dict[int, int]:
    """The season -> fold mapping SeasonAwareTimeSeriesSplit itself would
    produce for this (seasons, n_test_seasons, gap_seasons) config -- derived
    by actually running the splitter, not by reproducing its index math."""
    unique_seasons = np.array(sorted({int(s) for s in seasons}))
    splitter = SeasonAwareTimeSeriesSplit(n_splits=n_test_seasons, seasons=unique_seasons,
                                          gap_seasons=gap_seasons, strict=True)
    mapping: dict[int, int] = {}
    for fold, (_, test_idx) in enumerate(splitter.split(np.empty(len(unique_seasons)))):
        test_seasons = set(unique_seasons[test_idx].tolist())
        if len(test_seasons) != 1:
            raise ValueError(f"fold {fold} spans multiple seasons: {sorted(test_seasons)}")
        season = test_seasons.pop()
        if season in mapping:
            raise ValueError(f"season {season} assigned to more than one fold")
        mapping[int(season)] = fold
    return mapping


def reshape_plan_b_predictions(predictions: pd.DataFrame, arm_name: str,
                               season_fold_map: dict[int, int], *,
                               source_column: str = "joint_other_mae") -> pd.DataFrame:
    """Wide Plan B `predictions.csv` (from plan_b_joint_mae_backtester.run_backtest)
    -> Plan A's long arm schema. `source_column` is the wide column holding
    Plan B's candidate share prediction; `arm_name` is the label attached to
    it (usually the same string, kept separate so a caller can rename
    without touching the source data)."""
    required = set(PLAN_B_ROW_COLS + ["actual_share", source_column])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    unmapped = set(predictions.season.unique()) - set(season_fold_map)
    if unmapped:
        raise ValueError(f"seasons {sorted(unmapped)} have no fold in season_fold_map -- "
                         "refusing to silently drop them")
    out = predictions[PLAN_B_ROW_COLS].copy()
    out["fold"] = predictions["season"].map(season_fold_map).astype(int)
    out["arm"] = arm_name
    out["actual_share"] = pd.to_numeric(predictions["actual_share"], errors="raise").to_numpy(float)
    out["predicted_share"] = pd.to_numeric(predictions[source_column], errors="raise").to_numpy(float)
    if not np.isfinite(out["predicted_share"]).all():
        raise ValueError("nonfinite predicted_share after reshape")
    if out.duplicated(KEY).any():
        raise ValueError("duplicate (player_id, season, week, team, position, fold) after reshape")
    return out.reset_index(drop=True)


def attach_plan_b_arm(allocation_result: dict, plan_b_rows: pd.DataFrame, target: str,
                      truth: pd.DataFrame, folds: set[int]) -> dict:
    """Validate Plan B's reshaped rows exactly match the truth population for
    the given folds, and that their actual_share labels agree with the
    existing arms' rows for those same keys, before concatenating. Fails
    loud here rather than relying solely on ppr_truth.validate_inputs
    downstream, which would raise a less specific error."""
    folds = {int(f) for f in folds}
    existing = allocation_result["predictions"]
    rows = plan_b_rows[plan_b_rows.fold.isin(folds)].copy()
    if rows.empty:
        raise ValueError(f"{target}: no Plan B rows for folds {sorted(folds)}")
    if rows.duplicated(KEY).any():
        raise ValueError(f"{target}: duplicate Plan B arm rows")
    expected = filter_population(truth[truth.fold.isin(folds)], target)[KEY].drop_duplicates()
    merged = expected.merge(rows[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
    if not merged._merge.eq("both").all():
        only_truth = int(merged._merge.eq("left_only").sum())
        only_plan_b = int(merged._merge.eq("right_only").sum())
        raise ValueError(f"{target}: Plan B arm population differs from truth for folds {sorted(folds)} "
                         f"({only_truth} rows only in truth, {only_plan_b} rows only in Plan B)")
    reference = existing[existing.arm.eq("rolling3") & existing.fold.isin(folds)]
    if reference.empty:
        raise ValueError(f"{target}: no existing rolling3 rows to cross-check labels against")
    check = reference[KEY + ["actual_share"]].merge(
        rows[KEY + ["actual_share"]], on=KEY, suffixes=("_existing", "_plan_b"), validate="one_to_one")
    if len(check) != len(rows):
        raise ValueError(f"{target}: Plan B rows do not align one-to-one with the existing rolling3 population")
    if not np.allclose(check.actual_share_existing.to_numpy(float),
                       check.actual_share_plan_b.to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError(f"{target}: Plan B actual_share disagrees with the existing arms' labels")
    # existing may carry extra reporting-only columns (e.g. role_tier) that
    # Plan B's rows don't have -- pd.concat fills those with NaN for the new
    # arm's rows, which is fine: neither validate_inputs nor the selector's
    # own gating/coordinate-descent logic reads role_tier.
    combined = pd.concat([existing, rows], ignore_index=True, sort=False)
    return {**allocation_result, "predictions": combined}

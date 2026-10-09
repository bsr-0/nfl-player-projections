#!/usr/bin/env python3
"""Freeze a prefit common population, then compare full-PPR target-game rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.full_ppr_head_to_head import FULL_SCORING_DEFINITION, scoring_weights
from src.evaluation.paired_ppr_comparison import (
    KEY, assert_identical_key_sets, check_keys, file_sha256, finite_columns,
    paired_week_interval, read_rows,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _prediction_input(path: Path, label: str) -> tuple[pd.DataFrame, dict]:
    meta = _read_json(path)
    for field, expected in (("schema_version", 1), ("key_semantics", "target_game"),
                            ("scoring_definition", FULL_SCORING_DEFINITION),
                            ("scoring_weights", scoring_weights()),
                            ("prediction_column", "predicted_ppr"),
                            ("actual_column", "actual_ppr")):
        if meta.get(field) != expected:
            raise ValueError(f"{label}: incompatible {field}")
    if meta.get("label") != label:
        raise ValueError(f"{label}: incorrect model label")
    if not isinstance(meta.get("provenance"), dict) or not meta["provenance"]:
        raise ValueError(f"{label}: missing producer provenance")
    folds = meta.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{label}: missing fold lineage")
    for fold in folds:
        season, trained = fold.get("test_season"), fold.get("trained_through_season")
        selected = fold.get("selection_through_season")
        if (type(season) is not int or type(trained) is not int or trained >= season
                or "selection_through_season" not in fold
                or (selected is not None and (type(selected) is not int or selected >= season))):
            raise ValueError(f"{label}: ineligible fold lineage")
    rows_path = path.parent / meta["rows_file"]
    if file_sha256(rows_path) != meta.get("rows_sha256"):
        raise ValueError(f"{label}: row hash mismatch")
    rows = read_rows(rows_path)
    check_keys(rows, label)
    finite_columns(rows, ["predicted_ppr", "actual_ppr"], label)
    if set(rows.season.astype(int)) != {fold["test_season"] for fold in folds}:
        raise ValueError(f"{label}: prediction seasons differ from folds")
    if label == "plan_a_zero_extra_components" and meta.get("prediction_extension") != {
            "fumbles_lost": 0, "two_point_conversions": 0}:
        raise ValueError("Plan A lacks its declared zero-extra full-PPR extension")
    return rows, meta


def _expected_production_keys(mapping: pd.DataFrame) -> pd.DataFrame:
    required = {"player_id", "target_season", "target_week", "target_team", "target_position"}
    if required - set(mapping):
        raise ValueError(f"target map missing {sorted(required - set(mapping))}")
    eligible = mapping.loc[mapping.target_week.notna(),
                           ["player_id", "target_season", "target_week", "target_team",
                            "target_position"]].rename(columns={
                                "target_season": "season", "target_week": "week",
                                "target_team": "team", "target_position": "position"}).copy()
    eligible[["season", "week"]] = eligible[["season", "week"]].astype(int)
    check_keys(eligible, "prefit production target map")
    return eligible


def _assert_exact_keys(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    check_keys(left, f"{label} left")
    check_keys(right, f"{label} right")
    joint = left[KEY].merge(right[KEY], on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not joint._merge.eq("both").all():
        raise ValueError(f"{label}: exact target-game keys differ: {joint._merge.value_counts().to_dict()}")


def predeclare(plan_a_manifest: Path, preflight_manifest: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    plan_a, plan_meta = _prediction_input(plan_a_manifest, "plan_a_zero_extra_components")
    preflight = _read_json(preflight_manifest)
    if preflight.get("status") != "prepared":
        raise ValueError("production preflight is incomplete")
    map_path = preflight_manifest.parent / preflight["inputs"]["mapping"]["file"]
    if file_sha256(map_path) != preflight["inputs"]["mapping"]["sha256"]:
        raise ValueError("production target map changed since preflight")
    target = _expected_production_keys(read_rows(map_path))
    test_season = int(preflight["test_season"])
    if not target.season.eq(test_season).all() or test_season not in set(plan_a.season):
        raise ValueError("production and Plan A have no common held-out season")
    selected = plan_a.loc[plan_a.season.eq(test_season), KEY]
    matched = target[KEY].merge(selected, on=KEY, how="left", validate="one_to_one", indicator=True)
    population = matched.loc[matched._merge.eq("both"), KEY].sort_values(KEY).reset_index(drop=True)
    check_keys(population, "prefit matched population")
    excluded = matched.loc[matched._merge.ne("both"), KEY].copy()
    identity = ["player_id", "season", "week"]
    alternate = excluded[identity].merge(selected[identity], on=identity,
                                         how="left", validate="one_to_one", indicator=True)
    excluded["reason"] = np.where(excluded.week.gt(18), "postseason_outside_plan_a",
                                  np.where(alternate._merge.eq("both"),
                                           "team_or_position_mismatch", "absent_from_plan_a"))
    if len(population) + len(excluded) != len(target):
        raise ValueError("prefit population coverage does not reconcile")
    output_dir.mkdir(parents=True)
    population_path = output_dir / "population_keys.csv"
    excluded_path = output_dir / "excluded_production_keys.csv"
    population.to_csv(population_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    manifest = {
        "schema_version": 1, "status": "prefit_frozen",
        "test_season": test_season, "eligible_production_target_rows": len(target),
        "matched_rows": len(population), "plan_a_season_rows": len(selected),
        "excluded_reasons": {str(k): int(v) for k, v in excluded.reason.value_counts().items()},
        "population_file": population_path.name, "population_sha256": file_sha256(population_path),
        "excluded_file": excluded_path.name, "excluded_sha256": file_sha256(excluded_path),
        "plan_a_manifest": str(plan_a_manifest.resolve()),
        "plan_a_manifest_sha256": file_sha256(plan_a_manifest),
        "plan_a_rows_sha256": plan_meta["rows_sha256"],
        "preflight_manifest": str(preflight_manifest.resolve()),
        "preflight_manifest_sha256": file_sha256(preflight_manifest),
        "target_map_sha256": file_sha256(map_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def compare(population_manifest: Path, served_manifest: Path, output_dir: Path,
            n_bootstrap: int = 2000) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}")
    declared = _read_json(population_manifest)
    if declared.get("status") != "prefit_frozen":
        raise ValueError("population was not frozen before fitting")
    for field, hash_field in (("plan_a_manifest", "plan_a_manifest_sha256"),
                              ("preflight_manifest", "preflight_manifest_sha256")):
        if file_sha256(Path(declared[field])) != declared[hash_field]:
            raise ValueError(f"frozen {field} changed")
    population_path = population_manifest.parent / declared["population_file"]
    if file_sha256(population_path) != declared["population_sha256"]:
        raise ValueError("frozen population keys changed")
    population = read_rows(population_path)
    check_keys(population, "frozen population")
    if len(population) != declared["matched_rows"]:
        raise ValueError("frozen population row count changed")
    plan_a, plan_meta = _prediction_input(Path(declared["plan_a_manifest"]),
                                          "plan_a_zero_extra_components")
    served, served_meta = _prediction_input(served_manifest, "served_model_seasonal_fold")
    served_lineage = served_meta["provenance"]
    if (served_lineage.get("source_preflight_sha256") != declared["preflight_manifest_sha256"]
            or served_lineage.get("source_preflight") != declared["preflight_manifest"]):
        raise ValueError("served fold was trained from a different preflight")
    weights = served_lineage.get("model_artifact_sha256")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("served fold lacks persisted model-weight hashes")
    for relative, digest in weights.items():
        path = served_manifest.parent / relative
        if not path.resolve().is_relative_to(served_manifest.parent.resolve()) or file_sha256(path) != digest:
            raise ValueError(f"served model-weight hash mismatch: {relative}")
    if served_meta["folds"] != [{"test_season": declared["test_season"],
                                 "trained_through_season": declared["test_season"] - 1,
                                 "selection_through_season": None}]:
        raise ValueError("served fold does not hold out the declared season")
    preflight = _read_json(Path(declared["preflight_manifest"]))
    map_path = Path(declared["preflight_manifest"]).parent / preflight["inputs"]["mapping"]["file"]
    if file_sha256(map_path) != declared["target_map_sha256"]:
        raise ValueError("prefit production target map changed")
    expected = _expected_production_keys(read_rows(map_path))
    _assert_exact_keys(served, expected, "served predictions versus prefit eligible targets")
    derived = expected[KEY].merge(plan_a[KEY], on=KEY, how="inner", validate="one_to_one")
    _assert_exact_keys(population, derived, "frozen versus derived matched population")
    truth_path = Path(declared["plan_a_manifest"]).parent / "full_truth.csv"
    if file_sha256(truth_path) != plan_meta["provenance"]["full_truth_sha256"]:
        raise ValueError("full-PPR truth hash differs from Plan A lineage")
    truth = read_rows(truth_path)
    check_keys(truth, "full-PPR truth")
    _assert_exact_keys(truth, plan_a, "Plan A full truth versus predictions")
    cols = KEY + ["actual_full_ppr", "draft_rookie", "returning_player",
                  "first_observed_season", "snap_segment", "negative_yards",
                  "off_position_stat", "extra_component_points"]
    rows = population[KEY].merge(truth[cols], on=KEY, how="left", validate="one_to_one", indicator=True)
    if not rows._merge.eq("both").all():
        raise ValueError("declared population is missing full-PPR truth")
    rows = rows.drop(columns="_merge")
    # Every arm must forecast exactly the declared population: no gaps, no extras.
    assert_identical_key_sets({
        "population": population[KEY],
        **{label: arm[KEY].merge(population[KEY], on=KEY, how="inner")
           for label, arm in (("plan_a", plan_a), ("served", served))},
    })
    for label, arm in (("plan_a", plan_a), ("served", served)):
        subset = arm[KEY + ["predicted_ppr", "actual_ppr"]].rename(columns={
            "predicted_ppr": f"{label}_prediction", "actual_ppr": f"{label}_recorded_actual"})
        rows = rows.merge(subset, on=KEY, how="left", validate="one_to_one", indicator=True)
        if not rows._merge.eq("both").all():
            raise ValueError(f"{label} missing an expected target-game prediction")
        rows = rows.drop(columns="_merge")
        finite_columns(rows, [f"{label}_prediction", f"{label}_recorded_actual"], label)
        if not np.allclose(rows[f"{label}_recorded_actual"], rows.actual_full_ppr,
                           rtol=0, atol=1e-9):
            raise ValueError(f"{label} actuals differ from frozen raw full-PPR truth")
        rows[f"{label}_abs_error"] = (rows[f"{label}_prediction"] - rows.actual_full_ppr).abs()
    rows["delta_abs_error"] = rows.served_abs_error - rows.plan_a_abs_error
    rows = rows.drop(columns=["plan_a_recorded_actual", "served_recorded_actual"])

    def metrics(part: pd.DataFrame) -> dict:
        return {"n": len(part), "plan_a_mae": float(part.plan_a_abs_error.mean()),
                "served_mae": float(part.served_abs_error.mean()),
                "delta_served_minus_plan_a": float(part.delta_abs_error.mean()),
                "paired_interval": paired_week_interval(part, n_bootstrap=n_bootstrap)}

    segments = {
        "total": rows,
        "confirmed_draft_rookies": rows[rows.draft_rookie],
        "returning_players": rows[rows.returning_player],
        "first_observed_season": rows[rows.first_observed_season],
        "nonzero_offensive_snaps": rows[rows.snap_segment.eq("nonzero")],
        "zero_offensive_snaps": rows[rows.snap_segment.eq("zero")],
        "unknown_offensive_snaps": rows[rows.snap_segment.eq("unknown")],
    }
    if (len(segments["returning_players"]) + len(segments["first_observed_season"]) != len(rows)
            or sum(len(segments[name]) for name in (
                "nonzero_offensive_snaps", "zero_offensive_snaps",
                "unknown_offensive_snaps")) != len(rows)):
        raise ValueError("cohort assignments do not partition the matched population")
    report = {
        "status": "complete", "scoring_definition": FULL_SCORING_DEFINITION,
        "delta_definition": "served absolute error minus Plan A absolute error; negative favors served",
        "population": {"test_season": declared["test_season"], "matched_rows": len(rows),
                       "production_eligible_rows": declared["eligible_production_target_rows"],
                       "plan_a_season_rows": declared["plan_a_season_rows"],
                       "excluded_production_reasons": declared["excluded_reasons"],
                       "matched_rows_with_nonzero_extra_component_points": int(
                           rows.extra_component_points.ne(0).sum()),
                       "matched_mean_absolute_extra_component_points": float(
                           rows.extra_component_points.abs().mean()),
                       "plan_a_all_2025_mae_unpaired_context": float(
                           (plan_a.loc[plan_a.season.eq(declared["test_season"]), "predicted_ppr"]
                            - plan_a.loc[plan_a.season.eq(declared["test_season"]), "actual_ppr"])
                           .abs().mean())},
        "segments": {name: metrics(part) if not part.empty else {"n": 0, "status": "empty"}
                     for name, part in segments.items()},
        "by_position": {str(pos): metrics(part) for pos, part in rows.groupby("position")},
        "by_position_and_cohort": {
            f"{pos}/{name}": metrics(part)
            for pos, position_rows in rows.groupby("position")
            for name, part in (("confirmed_draft_rookies", position_rows[position_rows.draft_rookie]),
                               ("returning_players", position_rows[position_rows.returning_player]),
                               ("nonzero_offensive_snaps", position_rows[
                                   position_rows.snap_segment.eq("nonzero")]))
            if not part.empty},
        "limitations": [
            "2025 is one held-out season of the served architecture, not the current 2026 fitted weights.",
            "The matched production population concentrates on players with recorded stats and is substantially harder for Plan A than its full canonical 2025 panel.",
            "Snap cohorts use observed postgame snaps and are retrospective evaluation slices.",
            "Confirmed draft rookies require a matching draft pick; first-observed players are reported separately.",
            "Segment intervals are exploratory and unadjusted for multiple comparisons.",
            "Paired week blocks retain within-week dependence; they do not capture shared training uncertainty.",
        ],
        "input_sha256": {"population_manifest": file_sha256(population_manifest),
                         "plan_a_manifest": declared["plan_a_manifest_sha256"],
                         "served_manifest": file_sha256(served_manifest),
                         "full_truth": file_sha256(truth_path)},
    }
    if not np.isclose(report["segments"]["total"]["plan_a_mae"],
                      rows.plan_a_abs_error.mean(), rtol=0, atol=1e-12):
        raise ValueError("pooled MAE does not match row-level errors")
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "paired_rows.csv"
    rows.to_csv(rows_path, index=False, float_format="%.17g")
    saved = read_rows(rows_path)
    for arm in ("plan_a", "served"):
        if not np.isclose(np.abs(saved[f"{arm}_prediction"] - saved.actual_full_ppr).mean(),
                          report["segments"]["total"][f"{arm}_mae"], rtol=0, atol=1e-12):
            raise ValueError(f"saved {arm} rows disagree with pooled report")
    report["paired_rows_sha256"] = file_sha256(rows_path)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    frozen = sub.add_parser("predeclare")
    frozen.add_argument("--plan-a-manifest", type=Path, required=True)
    frozen.add_argument("--preflight-manifest", type=Path, required=True)
    frozen.add_argument("--output-dir", type=Path, required=True)
    compared = sub.add_parser("compare")
    compared.add_argument("--population-manifest", type=Path, required=True)
    compared.add_argument("--served-manifest", type=Path, required=True)
    compared.add_argument("--output-dir", type=Path, required=True)
    compared.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    try:
        result = (predeclare(args.plan_a_manifest, args.preflight_manifest, args.output_dir)
                  if args.command == "predeclare" else
                  compare(args.population_manifest, args.served_manifest, args.output_dir,
                          n_bootstrap=args.n_bootstrap))
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        parser.exit(2, f"full-PPR comparison stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

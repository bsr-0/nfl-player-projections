"""Fail-closed, read-only comparison of predictions for the same target games.

This module does not fit models, infer a target week, or convert a prediction
trained on a different scoring target. See docs/PPR_HEAD_TO_HEAD.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import SCORING
from src.evaluation.ppr_truth import IDENTITY, TARGETS, checked_truth

SCORING_DEFINITION = "raw_eight_component_ppr_excludes_fumbles_and_two_point_conversions"
KEY = list(IDENTITY)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> pd.DataFrame:
    # In particular, preserve player IDs with leading zeroes and CSV floats.
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"unsupported row file type: {path}; use CSV or Parquet")
    return pd.read_csv(path, dtype={"player_id": str, "team": str, "position": str},
                       float_precision="round_trip")


def check_keys(frame: pd.DataFrame, name: str) -> None:
    missing = set(KEY) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"{name}: empty input or missing key columns: {sorted(missing)}")
    if frame[KEY].isna().any().any():
        raise ValueError(f"{name}: null target-game key")
    for col in ("player_id", "team", "position"):
        if not frame[col].map(lambda value: isinstance(value, str) and bool(value.strip())
                              and value == value.strip()).all():
            raise ValueError(f"{name}: {col} must contain nonblank, unpadded strings")
    for col, low, high in (("season", 1900, 2200), ("week", 1, 22)):
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        if not (np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
                and ((values >= low) & (values <= high)).all()):
            raise ValueError(f"{name}: invalid integer {col}")
    if not frame.position.isin(["QB", "RB", "WR", "TE"]).all():
        raise ValueError(f"{name}: unsupported position")
    # Even differing team/position values cannot create two forecasts of the
    # same player's game. They must be resolved by the producer, not merged.
    if frame.duplicated(["player_id", "season", "week"]).any():
        raise ValueError(f"{name}: duplicate player target-game key")


def finite_columns(frame: pd.DataFrame, columns: list[str], name: str) -> np.ndarray:
    if set(columns) - set(frame.columns):
        raise ValueError(f"{name}: missing required columns {sorted(set(columns) - set(frame.columns))}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: missing or nonfinite values in {columns}")
    return values


def check_manifest(manifest: dict, frame: pd.DataFrame, name: str) -> None:
    expected = {"schema_version": 1, "key_semantics": "target_game",
                "scoring_definition": SCORING_DEFINITION,
                "scoring_weights": {target: SCORING[target] for target in TARGETS}}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"{name}: incompatible {field}; expected {value!r}")
    if not isinstance(manifest.get("label"), str) or not manifest["label"].strip():
        raise ValueError(f"{name}: label is required")
    if not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]:
        raise ValueError(f"{name}: producer provenance is required")
    folds = manifest.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError(f"{name}: fold training/selection cutoffs are required")
    seasons = []
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError(f"{name}: invalid fold record")
        season, trained = fold.get("test_season"), fold.get("trained_through_season")
        selected = fold.get("selection_through_season")
        if (type(season) is not int or type(trained) is not int or trained >= season
                or "selection_through_season" not in fold
                or (selected is not None and (type(selected) is not int or selected >= season))):
            raise ValueError(f"{name}: training or arm selection reaches the held-out season")
        seasons.append(season)
    if len(set(seasons)) != len(seasons) or set(seasons) != set(frame.season):
        raise ValueError(f"{name}: fold seasons do not exactly cover prediction seasons")
    origin = {"origin_season", "origin_week"} & set(frame.columns)
    if origin:
        if len(origin) != 2:
            raise ValueError(f"{name}: both origin_season and origin_week are required")
        values = finite_columns(frame, ["origin_season", "origin_week"], name)
        if (not np.equal(values, np.floor(values)).all()
                or not ((values[:, 1] >= 1) & (values[:, 1] <= 22)).all()
                or not ((values[:, 0] < frame.season)
                        | ((values[:, 0] == frame.season) & (values[:, 1] < frame.week))).all()):
            raise ValueError(f"{name}: forecast origin must precede its target game")


def load_predictions(manifest_path: Path) -> tuple[pd.DataFrame, dict, dict]:
    before = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rows_file = manifest.get("rows_file")
    if not isinstance(rows_file, str) or not rows_file:
        raise ValueError(f"{manifest_path}: rows_file is required")
    path = (manifest_path.parent / rows_file).resolve()
    digest = file_sha256(path)
    if digest != manifest.get("rows_sha256"):
        raise ValueError(f"{path}: prediction file hash does not match its manifest")
    frame = read_rows(path)
    if file_sha256(path) != digest or file_sha256(manifest_path) != before:
        raise ValueError("prediction inputs changed while being read; wait for the export to finish")
    check_keys(frame, manifest_path.name)
    check_manifest(manifest, frame, manifest_path.name)
    columns = [manifest.get("prediction_column"), manifest.get("actual_column")]
    if any(not isinstance(col, str) for col in columns) or len(set(columns)) != 2:
        raise ValueError("manifest must name distinct prediction_column and actual_column")
    values = finite_columns(frame, columns, manifest_path.name)
    normalized = frame[KEY].copy()
    normalized[["prediction", "recorded_actual"]] = values
    return normalized, manifest, {"manifest": str(manifest_path.resolve()),
                                  "manifest_sha256": before, "rows": str(path), "rows_sha256": digest}


def _select(frame: pd.DataFrame, population: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = population[KEY].merge(frame, on=KEY, how="left", validate="one_to_one", indicator=True)
    missing = merged._merge.ne("both")
    if missing.any():
        sample = merged.loc[missing, KEY].head(3).to_dict("records")
        raise ValueError(f"{name}: missing {int(missing.sum())} expected target-game rows; examples: {sample}")
    excluded = frame[KEY].merge(population[KEY], on=KEY, how="left", indicator=True)
    excluded = excluded.loc[excluded._merge.eq("left_only"), KEY]
    return merged.drop(columns="_merge"), excluded


def pair_predictions(baseline: pd.DataFrame, candidate: pd.DataFrame, raw: pd.DataFrame,
                     population: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Check all inputs before selecting a declared population; never intersect silently."""
    for name, frame in (("baseline", baseline), ("candidate", candidate), ("raw truth", raw)):
        check_keys(frame, name)
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        finite_columns(frame, ["prediction", "recorded_actual"], name)
    finite_columns(raw, TARGETS, "raw truth")
    mode = "exact"
    if population is None:
        union = baseline[KEY].merge(candidate[KEY], on=KEY, how="outer", indicator=True, validate="one_to_one")
        if not union._merge.eq("both").all():
            counts = union._merge.value_counts().to_dict()
            raise ValueError(f"candidate/baseline key sets differ: {counts}; supply a predeclared population file")
        population = baseline[KEY]
    else:
        mode = "explicit_population"
        check_keys(population, "population")
    population = population[KEY].sort_values(KEY).reset_index(drop=True)
    base, excluded_base = _select(baseline, population, "baseline")
    cand, excluded_cand = _select(candidate, population, "candidate")
    # Reuse P1's scorer. Here each calendar season is the unambiguous fold ID.
    keys = population.assign(fold=population.season)
    truth = checked_truth(raw, keys)
    joined = truth[KEY + ["actual_ppr", "negative_yards", "off_position_stat"]].copy()
    for name, frame in (("baseline", base), ("candidate", cand)):
        if not np.allclose(frame.recorded_actual.to_numpy(float), joined.actual_ppr.to_numpy(float),
                           rtol=0, atol=1e-9):
            raise ValueError(f"{name}: recorded actual labels differ from raw target-game eight-component PPR")
        joined[f"predicted_ppr_{name}"] = frame.prediction.to_numpy(float)
        joined[f"{name}_abs_error"] = np.abs(joined[f"predicted_ppr_{name}"] - joined.actual_ppr)
    joined["delta_abs_error"] = joined.candidate_abs_error - joined.baseline_abs_error
    excluded = {"baseline": excluded_base, "candidate": excluded_cand}
    audit = {"population_mode": mode, "matched_rows": len(joined), "raw_truth_rows": len(raw),
             "raw_truth_rows_outside_population": len(raw) - len(joined)}
    for name, frame in (("baseline", baseline), ("candidate", candidate)):
        audit[name] = {"input_rows": len(frame), "excluded_rows": len(excluded[name]),
                       "excluded_by_season": {str(k): int(v) for k, v in excluded[name].groupby("season").size().items()},
                       "excluded_by_position": {str(k): int(v) for k, v in excluded[name].groupby("position").size().items()}}
    return joined, audit, excluded


def paired_week_interval(rows: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> dict:
    """Resample whole calendar weeks within seasons, using identical draws for both arms.

    Sum errors/counts before taking a mean: unequal block sizes must not turn
    the player-week MAE into an unweighted average of weekly MAEs.
    """
    if type(n_bootstrap) is not int or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer >= 100")
    result = {"method": "paired_calendar_week_blocks_stratified_by_season",
              "point_estimate": float(rows.delta_abs_error.mean()), "n_bootstrap": n_bootstrap,
              "seed": seed, "ci_low": None, "ci_high": None, "significant_improvement": False}
    blocks = rows.groupby(["season", "week"], sort=True).delta_abs_error.agg(["sum", "count"])
    result["blocks_by_season"] = {str(s): len(g) for s, g in blocks.groupby(level=0)}
    if any(n < 2 for n in result["blocks_by_season"].values()):
        result["status"] = "insufficient_week_blocks"
        return result
    rng = np.random.default_rng(seed)
    sums, counts = np.zeros(n_bootstrap), np.zeros(n_bootstrap)
    for _, group in blocks.groupby(level=0, sort=True):
        draws = rng.integers(0, len(group), size=(n_bootstrap, len(group)))
        sums += group["sum"].to_numpy()[draws].sum(axis=1)
        counts += group["count"].to_numpy()[draws].sum(axis=1)
    lo, hi = np.quantile(sums / counts, [0.025, 0.975])
    result.update(status="ok", ci_low=float(lo), ci_high=float(hi), significant_improvement=bool(hi < 0))
    return result


def comparison_report(rows: pd.DataFrame, n_bootstrap: int = 2000, seed: int = 42) -> dict:
    def metrics(part):
        return {"n": len(part), "baseline_mae": float(part.baseline_abs_error.mean()),
                "candidate_mae": float(part.candidate_abs_error.mean()),
                "delta_candidate_minus_baseline": float(part.delta_abs_error.mean()),
                "paired_interval": paired_week_interval(part, n_bootstrap, seed)}

    result = {"scoring_definition": SCORING_DEFINITION,
              "scoring_weights": {target: SCORING[target] for target in TARGETS},
              "pooled": metrics(rows)}
    for col in ("season", "position", "negative_yards", "off_position_stat"):
        result[f"by_{col}"] = {str(value): metrics(part) for value, part in rows.groupby(col, sort=True)}
    result["by_season_position"] = {f"{season}/{pos}": metrics(part)
                                     for (season, pos), part in rows.groupby(["season", "position"], sort=True)}
    result["limitations"] = [
        "Retrospective evaluation on declared rows; this is not a fresh prospective holdout or a promotion decision.",
        "Week blocks preserve within-week dependence; serial dependence across weeks and shared training data remain.",
        "Manifest training and selection cutoffs are producer declarations, not independently recovered from model weights.",
        "Segment intervals are exploratory and are not adjusted for multiple comparisons.",
    ]
    return result

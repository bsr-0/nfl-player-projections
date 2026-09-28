#!/usr/bin/env python3
"""Nested walk-forward evaluation of calibrated player-simulation draws.

This is deliberately a research-only runner. It reads one independently
verified production OOF panel and never alters served projections. For every
outer season, residual-pool complexity is selected using only earlier inner
folds; the selected pool is then fit on all prior seasons and scored once on
the outer season.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable
import uuid

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_oof_panel import verify  # noqa: E402
from src.models.oof_capture import ZERO_FLOOR, cluster_bootstrap_distribution, target_game_panel  # noqa: E402
from src.models.player_correlation import fit_sparse_role_residual_correlation  # noqa: E402
from src.models.residual_calibration import (  # noqa: E402
    EmpiricalResidualCalibration,
    fit_empirical_residual_calibration,
    independent_residual_matrix,
    induce_role_rank_dependence,
    role_keys_from_panel_game,
    save_calibration,
)
from src.models.simulation_evaluation import empirical_crps, evaluate_joint_player_draws  # noqa: E402


CANDIDATE_MIN_STRATUM_ROWS = (25, 50, 100)
DEFAULT_MIN_STRATUM_ROWS = 50
COVERAGE_80_LOWER = 0.75
COVERAGE_80_UPPER = 0.85
IDENTITY_COLUMNS = ("season", "week", "game_id", "player_id")
N_BOOTSTRAP = 1_000


def _seed(seed: int, value: str) -> int:
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2**63 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_panel(panel: pd.DataFrame) -> None:
    required = {
        "player_id", "season", "week", "game_id", "team", "position",
        "predicted_points", "actual_points", "residual", "is_cold_start",
        "home_team", "away_team",
    }
    if missing := required - set(panel.columns):
        raise ValueError(f"verified OOF panel lacks simulation columns: {sorted(missing)}")
    if panel.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("OOF panel has duplicate player-week rows")
    for column in ("season", "week", "predicted_points", "actual_points", "residual"):
        values = pd.to_numeric(panel[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"OOF panel has null or nonfinite {column}")
    if not np.allclose(panel["residual"].to_numpy(float),
                       panel["predicted_points"].to_numpy(float) - panel["actual_points"].to_numpy(float),
                       atol=1e-12, rtol=0):
        raise ValueError("OOF residuals are not predicted_points - actual_points")


def _fit_role_model(history: pd.DataFrame):
    by_game = []
    for _, game in history.groupby("game_id", sort=True):
        roles = role_keys_from_panel_game(game)
        by_game.append(dict(zip(roles, game["residual"].astype(float))))
    counts = pd.Series([role for row in by_game for role in row]).value_counts()
    role_keys = sorted(counts[counts >= 2].index.tolist())
    if len(role_keys) < 2:
        return None
    matrix = np.full((len(by_game), len(role_keys)), np.nan)
    for i, row in enumerate(by_game):
        for j, role in enumerate(role_keys):
            if role in row:
                matrix[i, j] = row[role]
    return fit_sparse_role_residual_correlation(matrix, role_keys, shrinkage=.25, min_pair_rows=2)


def _draw_mode(game: pd.DataFrame, *, artifact: EmpiricalResidualCalibration,
               role_model, n_draws: int, seed: int) -> dict[str, np.ndarray]:
    predicted = game["predicted_points"].to_numpy(float)
    independent = independent_residual_matrix(game, artifact, n_draws=n_draws, seed=seed)
    shared = independent.copy()
    if role_model is not None:
        roles = role_keys_from_panel_game(game)
        available = [i for i, role in enumerate(roles) if role in role_model.role_keys]
        if len(available) >= 2:
            subset_roles = tuple(roles[i] for i in available)
            shared[:, available] = induce_role_rank_dependence(
                independent[:, available], role_keys=subset_roles,
                correlation_model=role_model, seed=seed + 1)
    return {
        "production_marginal": np.broadcast_to(predicted, (n_draws, len(game))).copy(),
        "calibrated_independent": predicted + independent,
        "calibrated_role_correlation": predicted + shared,
    }


def _draw_rows(mode: str, game: pd.DataFrame, draws: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    identities = game[list(IDENTITY_COLUMNS)].to_dict("records")
    metadata = game[["team", "position", "predicted_points", "is_cold_start"]].to_dict("records")
    for draw in range(draws.shape[0]):
        for index, (identity, fields) in enumerate(zip(identities, metadata)):
            rows.append({"mode": mode, "draw": draw, **identity, **fields,
                         "fantasy_points": float(draws[draw, index])})
    return rows


def _marginal_player_metrics(draws: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    actual_required = set(IDENTITY_COLUMNS) | {"actual_points", "position", "predicted_points", "is_cold_start"}
    if missing := actual_required - set(actuals.columns):
        raise ValueError(f"actuals lack marginal-metric columns: {sorted(missing)}")
    if actuals.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("actual player-game rows are not unique")
    actual_by_key = actuals.set_index(list(IDENTITY_COLUMNS), verify_integrity=True)
    rows = []
    for key, group in draws.groupby(list(IDENTITY_COLUMNS), sort=True):
        if key not in actual_by_key.index:
            raise ValueError(f"simulated player-game has no actual row: {key}")
        values = group["fantasy_points"].to_numpy(float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"invalid simulation draws for player-game: {key}")
        actual = actual_by_key.loc[key]
        truth = float(actual.actual_points)
        mean = float(values.mean())
        row = dict(zip(IDENTITY_COLUMNS, key))
        row.update({
            "team": str(actual.team) if "team" in actual.index else None,
            "position": str(actual.position),
            "predicted_points": float(actual.predicted_points),
            "is_cold_start": bool(actual.is_cold_start),
            "actual_points": truth,
            "mean": mean,
            "mae": abs(mean - truth),
            "bias": mean - truth,
            "crps": empirical_crps(values, truth),
            "coverage_50": float(np.quantile(values, .25) <= truth <= np.quantile(values, .75)),
            "coverage_80": float(np.quantile(values, .10) <= truth <= np.quantile(values, .90)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_marginal(rows: pd.DataFrame) -> dict:
    return {
        "rows": int(len(rows)), "mae": float(rows["mae"].mean()),
        "mean_bias": float(rows["bias"].mean()), "crps": float(rows["crps"].mean()),
        "coverage_50": float(rows["coverage_50"].mean()),
        "coverage_80": float(rows["coverage_80"].mean()),
    }


def _independent_metrics(heldout: pd.DataFrame, artifact: EmpiricalResidualCalibration,
                         *, n_draws: int, seed: int, context: str) -> pd.DataFrame:
    output = []
    for game_id, game in heldout.groupby("game_id", sort=True):
        residuals = independent_residual_matrix(
            game, artifact, n_draws=n_draws, seed=_seed(seed, f"{context}:{game_id}"))
        output.extend(_draw_rows("calibrated_independent", game,
                                 game["predicted_points"].to_numpy(float) + residuals))
    return _marginal_player_metrics(pd.DataFrame(output), heldout)


def select_candidate(panel: pd.DataFrame, *, outer_season: int, n_draws: int,
                     seed: int, candidates: Iterable[int] = CANDIDATE_MIN_STRATUM_ROWS) -> dict:
    """Choose residual-pool support using strictly earlier nested folds."""
    candidates = tuple(sorted({int(value) for value in candidates}))
    if not candidates or any(value < 2 for value in candidates):
        raise ValueError("candidate minimum stratum sizes must be integers >= 2")
    history = panel.loc[panel["season"].astype(int) < int(outer_season)].copy()
    inner_seasons = sorted(int(value) for value in history["season"].unique())
    eligible_inner = [season for season in inner_seasons if (history["season"] < season).any()]
    candidate_rows = []
    if not eligible_inner:
        return {
            "outer_season": int(outer_season),
            "selected_min_stratum_rows": DEFAULT_MIN_STRATUM_ROWS,
            "selection_reason": "default_insufficient_inner_history",
            "eligible_inner_seasons": [], "candidates": [],
        }
    for minimum in candidates:
        per_inner = []
        for inner_season in eligible_inner:
            train = history.loc[history["season"] < inner_season].copy()
            test = history.loc[history["season"] == inner_season].copy()
            artifact = fit_empirical_residual_calibration(train, min_stratum_rows=minimum)
            metrics = _aggregate_marginal(_independent_metrics(
                test, artifact, n_draws=n_draws, seed=seed,
                context=f"inner:{outer_season}:{minimum}:{inner_season}"))
            metrics["test_season"] = int(inner_season)
            per_inner.append(metrics)
        combined = pd.DataFrame(per_inner)
        coverage = float(np.average(combined["coverage_80"], weights=combined["rows"]))
        candidate_rows.append({
            "min_stratum_rows": minimum, "inner_seasons": per_inner,
            "rows": int(combined["rows"].sum()),
            "crps": float(np.average(combined["crps"], weights=combined["rows"])),
            "coverage_80": coverage, "coverage_80_error": abs(coverage - .80),
            "coverage_eligible": bool(COVERAGE_80_LOWER <= coverage <= COVERAGE_80_UPPER),
        })
    eligible = [row for row in candidate_rows if row["coverage_eligible"]]
    if eligible:
        selected = min(eligible, key=lambda row: (row["crps"], row["min_stratum_rows"]))
        reason = "lowest_crps_within_80pct_coverage_band"
    else:
        selected = min(candidate_rows, key=lambda row: (
            row["coverage_80_error"], row["crps"], row["min_stratum_rows"]))
        reason = "closest_80pct_coverage_then_lowest_crps"
    return {
        "outer_season": int(outer_season),
        "selected_min_stratum_rows": int(selected["min_stratum_rows"]),
        "selection_reason": reason, "eligible_inner_seasons": eligible_inner,
        "candidates": candidate_rows,
    }


def _paired_cluster_interval(values: np.ndarray, clusters: np.ndarray, *, n_boot: int,
                             seed: int) -> dict:
    values, clusters = np.asarray(values, dtype=float), np.asarray(clusters)
    draws = cluster_bootstrap_distribution(values, clusters, statistic=np.mean,
                                          n_boot=n_boot, seed=seed)
    if not len(draws):
        return {"point_estimate": float(values.mean()), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_bootstrap": int(n_boot),
                "status": "insufficient_clusters", "significant_improvement": False}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point_estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "n_bootstrap": int(n_boot), "status": "complete",
            "significant_improvement": bool(high < 0)}


def _marginal_segments(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["predicted_activity"] = np.where(frame["predicted_points"] > ZERO_FLOOR,
                                            "predicted_active", "predicted_near_zero")
    frame["experience"] = np.where(frame["is_cold_start"], "cold_start", "returning")
    frame["actual_activity"] = np.where(frame["actual_points"] > ZERO_FLOOR,
                                         "nonzero_actual", "near_zero_actual")
    def add_entry(segment: str, value: str, mode: str, part: pd.DataFrame) -> dict:
        entry = {"segment": segment, "value": value, "mode": mode,
                 **_aggregate_marginal(part)}
        # Whole-player bootstrap: repeated weeks for the same player are not
        # independent evidence about calibration quality.
        for metric in ("mae", "bias", "crps", "coverage_50", "coverage_80"):
            draws = cluster_bootstrap_distribution(
                part[metric].to_numpy(float), part["player_id"].to_numpy(), statistic=np.mean,
                n_boot=N_BOOTSTRAP, seed=_seed(0, f"segment:{segment}:{value}:{mode}:{metric}"))
            entry[f"{metric}_ci_low"] = float(np.percentile(draws, 2.5)) if len(draws) else float("nan")
            entry[f"{metric}_ci_high"] = float(np.percentile(draws, 97.5)) if len(draws) else float("nan")
        return entry

    output = []
    for mode, part in frame.groupby("mode", sort=True):
        output.append(add_entry("pooled", "all", mode, part))
    for segment, group_column in (("position", "position"), ("predicted_activity", "predicted_activity"),
                                  ("experience", "experience"), ("actual_activity", "actual_activity")):
        for value, group in frame.groupby(group_column, sort=True):
            for mode, part in group.groupby("mode", sort=True):
                output.append(add_entry(segment, str(value), mode, part))
    return pd.DataFrame(output)


def _comparison_report(marginal: pd.DataFrame, joint_by_game: pd.DataFrame,
                       *, n_bootstrap: int, seed: int) -> dict:
    result = {"marginal_vs_production": {}, "marginal_vs_independent": {},
              "joint_vs_independent": {}}
    production = marginal.loc[marginal["mode"] == "production_marginal"].set_index(list(IDENTITY_COLUMNS))
    for mode in ("calibrated_independent", "calibrated_role_correlation"):
        candidate = marginal.loc[marginal["mode"] == mode].set_index(list(IDENTITY_COLUMNS))
        if not candidate.index.equals(production.index):
            raise ValueError(f"{mode} marginal keys do not match production")
        result["marginal_vs_production"][mode] = {
            metric: _paired_cluster_interval(
                candidate[metric].to_numpy(float) - production[metric].to_numpy(float),
                candidate.index.get_level_values("player_id").to_numpy(), n_boot=n_bootstrap,
                seed=_seed(seed, f"player:{mode}:{metric}"))
            for metric in ("mae", "crps", "bias", "coverage_50", "coverage_80")
        }
    independent_marginal = marginal.loc[marginal["mode"] == "calibrated_independent"].set_index(
        list(IDENTITY_COLUMNS))
    correlated_marginal = marginal.loc[marginal["mode"] == "calibrated_role_correlation"].set_index(
        list(IDENTITY_COLUMNS))
    if not correlated_marginal.index.equals(independent_marginal.index):
        raise ValueError("role-correlated marginal keys do not match calibrated-independent")
    result["marginal_vs_independent"]["calibrated_role_correlation"] = {
        metric: _paired_cluster_interval(
            correlated_marginal[metric].to_numpy(float) - independent_marginal[metric].to_numpy(float),
            correlated_marginal.index.get_level_values("player_id").to_numpy(), n_boot=n_bootstrap,
            seed=_seed(seed, f"player:correlated-v-independent:{metric}"))
        for metric in ("mae", "crps", "bias", "coverage_50", "coverage_80")
    }
    independent = joint_by_game.loc[joint_by_game["mode"] == "calibrated_independent"].set_index("game_id")
    correlated = joint_by_game.loc[joint_by_game["mode"] == "calibrated_role_correlation"].set_index("game_id")
    if not correlated.index.equals(independent.index):
        raise ValueError("role-correlated game keys do not match calibrated-independent")
    result["joint_vs_independent"]["calibrated_role_correlation"] = {
        metric: _paired_cluster_interval(
            correlated[metric].to_numpy(float) - independent[metric].to_numpy(float),
            correlated.index.to_numpy(), n_boot=n_bootstrap,
            seed=_seed(seed, f"game:correlated:{metric}"))
        for metric in ("energy_score", "variogram_score_p05")
    }
    return result


def _promotion_gate(mode_reports: dict, comparisons: dict) -> dict:
    """Apply the predeclared research gate without promoting any artifact."""
    independent = mode_reports["calibrated_independent"]
    correlated = mode_reports["calibrated_role_correlation"]
    independent_delta = comparisons["marginal_vs_production"]["calibrated_independent"]
    independent_eligible = bool(
        independent_delta["crps"]["significant_improvement"]
        and COVERAGE_80_LOWER <= independent["coverage_80"] <= COVERAGE_80_UPPER)
    correlated_delta = comparisons["marginal_vs_independent"]["calibrated_role_correlation"]
    crps_preserved = abs(correlated_delta["crps"]["point_estimate"]) <= .01 * max(independent["crps"], 1e-12)
    coverage_preserved = abs(correlated["coverage_80"] - independent["coverage_80"]) <= .01 * max(independent["coverage_80"], 1e-12)
    joint_delta = comparisons["joint_vs_independent"]["calibrated_role_correlation"]
    joint_improved = bool(joint_delta["energy_score"]["significant_improvement"] or
                          joint_delta["variogram_score_p05"]["significant_improvement"])
    return {
        "calibrated_independent": {
            "eligible_for_later_comparison": independent_eligible,
            "crps_interval_excludes_zero_in_favor": independent_delta["crps"]["significant_improvement"],
            "coverage_80_in_gate": bool(COVERAGE_80_LOWER <= independent["coverage_80"] <= COVERAGE_80_UPPER),
        },
        "calibrated_role_correlation": {
            "eligible_for_later_comparison": bool(independent_eligible and crps_preserved and coverage_preserved and joint_improved),
            "independent_gate_passed": independent_eligible,
            "marginal_crps_preserved_within_1pct": crps_preserved,
            "coverage_80_preserved_within_1pct": coverage_preserved,
            "energy_or_variogram_interval_excludes_zero_in_favor": joint_improved,
        },
        "promotion": "none; this report is a research gate only",
    }


def run(panel: pd.DataFrame, *, draws: int, seed: int,
        candidates: Iterable[int] = CANDIDATE_MIN_STRATUM_ROWS,
        n_bootstrap: int = N_BOOTSTRAP, return_details: bool = False):
    """Run the leakage-safe nested comparison without writing artifacts."""
    if draws < 2:
        raise ValueError("draws must be at least two")
    _validate_panel(panel)
    output, excluded, selections = [], [], []
    seasons = sorted(int(value) for value in panel["season"].unique())
    for season in seasons:
        history = panel.loc[panel["season"] < season].copy()
        heldout = panel.loc[panel["season"] == season].copy()
        if history.empty:
            excluded.append({"season": season, "reason": "no_prior_oof_residuals", "rows": int(len(heldout))})
            continue
        selection = select_candidate(panel, outer_season=season, n_draws=draws, seed=seed,
                                     candidates=candidates)
        artifact = fit_empirical_residual_calibration(
            history, min_stratum_rows=selection["selected_min_stratum_rows"])
        selection["fit_rows"] = int(len(history))
        selection["artifact_pool_counts"] = artifact.diagnostics["pool_counts"]
        selections.append(selection)
        role_model = _fit_role_model(history)
        for game_id, game in heldout.groupby("game_id", sort=True):
            matrices = _draw_mode(game, artifact=artifact, role_model=role_model,
                                  n_draws=draws, seed=_seed(seed, f"outer:{season}:{game_id}"))
            for mode, matrix in matrices.items():
                output.extend(_draw_rows(mode, game, matrix))
    if not output:
        raise ValueError("no held-out season has prior OOF residuals")
    draws_frame = pd.DataFrame(output)
    scored_keys = draws_frame[list(IDENTITY_COLUMNS)].drop_duplicates()
    actuals = panel.merge(scored_keys, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    expected_keys = set(map(tuple, scored_keys.to_numpy()))
    metric_frames, joint_frames, mode_reports = [], [], {}
    for mode, mode_draws in draws_frame.groupby("mode", sort=True):
        keys = set(map(tuple, mode_draws[list(IDENTITY_COLUMNS)].drop_duplicates().to_numpy()))
        if keys != expected_keys:
            raise ValueError(f"{mode} does not cover the exact common player-game population")
        player_metrics = _marginal_player_metrics(mode_draws, actuals)
        player_metrics.insert(0, "mode", mode)
        metric_frames.append(player_metrics)
        joint = evaluate_joint_player_draws(mode_draws, actuals.rename(columns={"actual_points": "fantasy_points"}))
        game_metrics = pd.DataFrame(joint.pop("by_game"))
        game_metrics.insert(0, "mode", mode)
        joint_frames.append(game_metrics)
        mode_reports[mode] = {**_aggregate_marginal(player_metrics),
                              "energy_score": joint["mean_energy_score"],
                              "variogram_score_p05": joint["mean_variogram_score_p05"],
                              "games_scored": joint["games_scored"]}
    marginal = pd.concat(metric_frames, ignore_index=True)
    joint_by_game = pd.concat(joint_frames, ignore_index=True)
    comparisons = _comparison_report(marginal, joint_by_game, n_bootstrap=n_bootstrap, seed=seed)
    report = {
        "status": "complete", "draws_per_player": int(draws), "modes": mode_reports,
        "excluded_seasons": excluded, "outer_season_selection": selections,
        "scoring_population": {"rows": int(len(actuals)), "games": int(actuals["game_id"].nunique()),
                               "seasons": sorted(int(s) for s in actuals["season"].unique())},
        "calibration": {"candidate_min_stratum_rows": list(sorted({int(x) for x in candidates})),
                        "selection_strata": ["position", "predicted_activity", "experience"],
                        "actual_activity_is_diagnostic_only": True, "environment": "not used",
                        "coverage_80_gate": [COVERAGE_80_LOWER, COVERAGE_80_UPPER]},
        "comparisons": comparisons,
        "promotion_gate": _promotion_gate(mode_reports, comparisons),
    }
    if return_details:
        return report, draws_frame, actuals, marginal, joint_by_game
    return report, draws_frame, actuals


def _write_run(output_dir: Path, *, report: dict, draws: pd.DataFrame, actuals: pd.DataFrame,
               marginal: pd.DataFrame, joint_by_game: pd.DataFrame, oof_run_dir: Path,
               oof_verification: dict, panel: pd.DataFrame, seed: int, n_bootstrap: int) -> None:
    if output_dir.exists():
        raise ValueError(f"output directory exists: {output_dir}")
    final_output_dir = output_dir
    output_dir = final_output_dir.with_name(f".{final_output_dir.name}.staging-{uuid.uuid4().hex}")
    output_dir.mkdir(parents=True)
    draws_path, actuals_path = output_dir / "player_draws.parquet", output_dir / "actuals.parquet"
    marginal_path, joint_path = output_dir / "marginal_player_metrics.csv", output_dir / "joint_game_metrics.csv"
    segments_path = output_dir / "marginal_segment_report.csv"
    draws.to_parquet(draws_path, index=False)
    actuals.to_parquet(actuals_path, index=False)
    marginal.to_csv(marginal_path, index=False)
    joint_by_game.to_csv(joint_path, index=False)
    _marginal_segments(marginal).to_csv(segments_path, index=False)
    selection_path = output_dir / "calibration_selection.json"
    selection_path.write_text(json.dumps(report["outer_season_selection"], indent=2, sort_keys=True) + "\n")
    artifact_paths = []
    for selection in report["outer_season_selection"]:
        season = int(selection["outer_season"])
        history = panel.loc[panel["season"] < season].copy()
        artifact = fit_empirical_residual_calibration(
            history, min_stratum_rows=int(selection["selected_min_stratum_rows"]))
        path = output_dir / "outer_folds" / str(season) / "residual_calibration.json"
        save_calibration(artifact, path)
        artifact_paths.append(path)
    report["oof_verification"] = oof_verification
    report["reporting"] = {"player_cluster_bootstrap": int(n_bootstrap),
                           "game_cluster_bootstrap": int(n_bootstrap), "seed": int(seed)}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_run_dir": str(oof_run_dir.resolve()),
        "source_oof_panel_sha256": oof_verification["panel_sha256"],
        "draws_per_player": int(report["draws_per_player"]), "seed": int(seed),
        "n_rows": int(report["scoring_population"]["rows"]),
        "n_games": int(report["scoring_population"]["games"]),
        "scoring_seasons": report["scoring_population"]["seasons"],
        "files": {str(path.relative_to(output_dir)): _sha256(path) for path in
                  (draws_path, actuals_path, marginal_path, joint_path, segments_path,
                   selection_path, report_path, *artifact_paths)},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Publish only after every row-level file, fitted artifact, report, and
    # manifest hash exists. A failed run leaves an obviously temporary
    # staging directory rather than a misleading immutable result directory.
    output_dir.rename(final_output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    if args.draws < 2 or args.n_bootstrap < 100:
        parser.error("--draws must be >= 2 and --n-bootstrap must be >= 100")
    verification = verify(args.oof_run_dir)
    # Simulate the game each actual came from, not the forecast origin row.
    panel = target_game_panel(pd.read_parquet(args.oof_run_dir / "panel.parquet"))
    report, draws_frame, actuals, marginal, joint_by_game = run(
        panel, draws=args.draws, seed=args.seed, n_bootstrap=args.n_bootstrap, return_details=True)
    _write_run(args.output_dir, report=report, draws=draws_frame, actuals=actuals,
               marginal=marginal, joint_by_game=joint_by_game, oof_run_dir=args.oof_run_dir,
               oof_verification=verification, panel=panel, seed=args.seed, n_bootstrap=args.n_bootstrap)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

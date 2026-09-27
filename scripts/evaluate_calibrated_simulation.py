#!/usr/bin/env python3
"""Rolling-origin evaluation of calibrated, dependent player-simulation draws.

Research-only: reads one independently verified production OOF panel and never
alters served projections. Every draw is centred on the served prediction, so
point MAE cannot change; what is measured is the quality of the predictive
distribution (marginal CRPS/coverage) and of its dependence (variogram score,
calibration of stack/team/game sums).

Protocol (all causal):
- Week W of season S is simulated from artifacts fitted on every panel row
  with (season, week) < (S, W). The season-S model's earlier-week residuals
  are known before week W kicks off, so this is legitimate and adapts in-season.
- Only weeks of seasons that have at least one earlier panel season are scored.
- The marginal family is re-selected every week from rolling evidence on
  strictly earlier weeks, and moves off the legacy default only when a
  week-clustered bootstrap says the improvement is real.
- Modes share random numbers: both independent arms use the same seed per
  game, and every dependent arm reorders the same independent draws, so joint
  deltas measure dependence alone.
- Primary comparisons are fixed in PRIMARY_COMPARISONS before any run and
  corrected with Holm; everything else is secondary.
- ``--confirm-season`` withholds the panel's latest season from development
  runs entirely; ``--run-confirmation`` scores it once, after design is frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
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
from src.models.player_correlation import (  # noqa: E402
    fit_factor_copula,
    fit_sparse_role_residual_correlation,
    pooled_pair_correlations,
)
from src.models.residual_calibration import (  # noqa: E402
    ANALOG_FAMILIES,
    fit_empirical_residual_calibration,
    fit_prediction_analog_calibration,
    independent_residual_matrix,
    induce_role_rank_dependence,
    normal_scores,
    role_keys_from_panel_game,
    save_calibration,
)
from src.models.simulation_evaluation import (  # noqa: E402
    empirical_crps,
    energy_score,
    scaled_variogram_score,
    variogram_score,
)
from src.utils.multiple_comparisons import bootstrap_two_sided_p_value, holm_bonferroni  # noqa: E402


LEGACY_MIN_STRATUM_ROWS = (25, 50, 100)
ANALOG_K = (100, 250, 500)
DEFAULT_MIN_STRATUM_ROWS = 50
COVERAGE_80_LOWER = 0.75
COVERAGE_80_UPPER = 0.85
IDENTITY_COLUMNS = ("season", "week", "game_id", "player_id")
DRAW_METADATA_COLUMNS = ("team", "position", "predicted_points", "is_cold_start")
N_BOOTSTRAP = 1_000
SELECTION_BOOTSTRAP = 200
ALPHA = 0.05
STACK_PASS_CATCHERS = 2

PRODUCTION = "production_marginal"
LEGACY_INDEPENDENT = "calibrated_independent_legacy"
INDEPENDENT = "calibrated_independent"
TEAM_FACTOR = "calibrated_team_factor"
ROLE_FACTOR = "calibrated_role_factor"
LEGACY_ROLE = "calibrated_role_correlation"
MODES = (PRODUCTION, LEGACY_INDEPENDENT, INDEPENDENT, TEAM_FACTOR, ROLE_FACTOR, LEGACY_ROLE)
FACTOR_MODES = (TEAM_FACTOR, ROLE_FACTOR)

# Pre-registered primary comparisons: (table, metric, candidate, baseline).
# Lower is better for every metric listed. Anything not here is secondary
# and must not drive a promotion decision.
PRIMARY_COMPARISONS = (
    ("marginal", "crps", INDEPENDENT, LEGACY_INDEPENDENT),
    ("joint", "variogram_score_p05_scaled", TEAM_FACTOR, INDEPENDENT),
    ("sum:stack", "crps", TEAM_FACTOR, INDEPENDENT),
    ("joint", "variogram_score_p05_scaled", ROLE_FACTOR, INDEPENDENT),
    ("sum:stack", "crps", ROLE_FACTOR, INDEPENDENT),
    ("joint", "variogram_score_p05_scaled", ROLE_FACTOR, TEAM_FACTOR),
    ("sum:stack", "crps", ROLE_FACTOR, TEAM_FACTOR),
)
PANEL_SCOPE_CAVEAT = (
    "Scored rows are the player-games present in the OOF panel. Players ruled "
    "inactive before kickoff are not in it, so these results describe "
    "calibration conditional on the player appearing, not production "
    "calibration when availability is itself uncertain.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _seed(seed: int, value: str) -> int:
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2**63 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _week_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["season"].astype(int).to_numpy() * 100 + frame["week"].astype(int).to_numpy()


def _week_cluster(frame: pd.DataFrame) -> pd.Series:
    return frame["season"].astype(int).astype(str) + "-" + frame["week"].astype(int).map("{:02d}".format)


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
    weeks = panel["week"].astype(int)
    if ((weeks < 1) | (weeks > 99)).any():
        raise ValueError("OOF panel weeks must lie in 1..99")
    if not np.allclose(panel["residual"].to_numpy(float),
                       panel["predicted_points"].to_numpy(float) - panel["actual_points"].to_numpy(float),
                       atol=1e-12, rtol=0):
        raise ValueError("OOF residuals are not predicted_points - actual_points")
    per_game = panel.groupby("game_id")[["season", "week"]].nunique()
    if (per_game > 1).any().any():
        raise ValueError("a game_id spans more than one season-week")


def _attach_roles(panel: pd.DataFrame) -> pd.Series:
    """Role keys from prediction-time fields only, computed once per game."""
    roles = pd.Series(index=panel.index, dtype=object)
    for _, game in panel.groupby("game_id", sort=True):
        roles.loc[game.index] = role_keys_from_panel_game(game)
    return roles


# ---------------------------------------------------------------------------
# Marginal candidates and causal selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    family: str  # "legacy" (stratified pools) or an analog family
    size: int    # min_stratum_rows for legacy, k for analog

    @property
    def id(self) -> str:
        return f"{self.family}:{self.size}"


DEFAULT_CANDIDATE = Candidate("legacy", DEFAULT_MIN_STRATUM_ROWS)


def candidate_grid(legacy_sizes: Iterable[int] = LEGACY_MIN_STRATUM_ROWS,
                   analog_k: Iterable[int] = ANALOG_K) -> tuple[Candidate, ...]:
    legacy = sorted({int(value) for value in legacy_sizes})
    analog = sorted({int(value) for value in analog_k})
    if any(value < 2 for value in legacy + analog):
        raise ValueError("candidate sizes must be integers >= 2")
    if DEFAULT_MIN_STRATUM_ROWS not in legacy:
        raise ValueError(f"legacy candidates must include the default {DEFAULT_MIN_STRATUM_ROWS}")
    return tuple([Candidate("legacy", value) for value in legacy]
                 + [Candidate(family, value) for family in ANALOG_FAMILIES for value in analog])


def fit_candidate(candidate: Candidate, history: pd.DataFrame):
    if candidate.family == "legacy":
        return fit_empirical_residual_calibration(history, min_stratum_rows=candidate.size)
    return fit_prediction_analog_calibration(history, family=candidate.family, k=candidate.size)


def support_metrics(calibration, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exact CRPS and 80% coverage of each row's full support (no sampling).

    Equal to ``empirical_crps(predicted + support, actual)``: CRPS is shift
    invariant in its spread term, so only the sorted deviations are needed.
    """
    crps = np.empty(len(rows))
    covered = np.empty(len(rows))
    for index, row in enumerate(rows.itertuples(index=False)):
        support = calibration.sorted_support_deviations(
            position=row.position, predicted_points=float(row.predicted_points),
            is_cold_start=bool(row.is_cold_start))
        observed = float(row.actual_points) - float(row.predicted_points)
        n = len(support)
        crps[index] = (np.abs(support - observed).mean()
                       - np.dot(2 * np.arange(n) - n + 1, support) / (n * n))
        low, high = np.quantile(support, [.10, .90])
        covered[index] = float(low <= observed <= high)
    return crps, covered


def select_candidate(evidence: pd.DataFrame, *, target_week_id: int,
                     candidates: tuple[Candidate, ...], n_boot: int = SELECTION_BOOTSTRAP,
                     seed: int = 0) -> dict:
    """Choose a marginal family using only weeks strictly before the target.

    Among candidates whose rolling 80% coverage sits in the gate band, take
    the lowest CRPS -- but leave the legacy default only when the paired,
    week-clustered bootstrap CI of (best - default) lies wholly below zero.
    """
    ids = [candidate.id for candidate in candidates]
    if DEFAULT_CANDIDATE.id not in ids:
        raise ValueError("candidate set must include the default")
    prior = evidence.loc[(evidence["week_id"] < target_week_id) & evidence["candidate"].isin(ids)]
    base = {"target_week_id": int(target_week_id), "evidence_weeks": int(prior["week_id"].nunique()),
            "evidence_rows": int((prior["candidate"] == DEFAULT_CANDIDATE.id).sum())}
    if prior.empty:
        return {**base, "candidate": DEFAULT_CANDIDATE.id, "reason": "no_prior_evidence", "summary": []}
    wide = prior.pivot_table(index=["week_id", "player_id"], columns="candidate",
                             values=["crps", "coverage_80"], aggfunc="first")
    if wide.isna().any().any():
        raise ValueError("candidate evidence does not cover identical rows")
    summary = []
    for position, candidate_id in enumerate(ids):
        coverage = float(wide[("coverage_80", candidate_id)].mean())
        summary.append({"candidate": candidate_id, "order": position,
                        "crps": float(wide[("crps", candidate_id)].mean()), "coverage_80": coverage,
                        "in_band": bool(COVERAGE_80_LOWER <= coverage <= COVERAGE_80_UPPER)})
    by_id = {entry["candidate"]: entry for entry in summary}
    in_band = [entry for entry in summary if entry["in_band"]]
    default = by_id[DEFAULT_CANDIDATE.id]
    if not default["in_band"]:
        if in_band:
            chosen = min(in_band, key=lambda e: (e["crps"], e["order"]))
            return {**base, "candidate": chosen["candidate"], "reason": "default_outside_coverage_band",
                    "summary": summary}
        chosen = min(summary, key=lambda e: (abs(e["coverage_80"] - .8), e["crps"], e["order"]))
        return {**base, "candidate": chosen["candidate"], "reason": "no_candidate_in_coverage_band",
                "summary": summary}
    best = min(in_band, key=lambda e: (e["crps"], e["order"]))
    if best["candidate"] == DEFAULT_CANDIDATE.id:
        return {**base, "candidate": DEFAULT_CANDIDATE.id, "reason": "default_is_best", "summary": summary}
    delta = (wide[("crps", best["candidate"])] - wide[("crps", DEFAULT_CANDIDATE.id)]).to_numpy(float)
    clusters = wide.index.get_level_values("week_id").to_numpy()
    draws = cluster_bootstrap_distribution(delta, clusters, statistic=np.mean, n_boot=n_boot,
                                           seed=_seed(seed, f"select:{target_week_id}:{','.join(ids)}"))
    if not len(draws):
        return {**base, "candidate": DEFAULT_CANDIDATE.id, "reason": "insufficient_evidence_weeks",
                "summary": summary}
    high = float(np.percentile(draws, 97.5))
    if high < 0:
        return {**base, "candidate": best["candidate"], "reason": "significant_improvement_over_default",
                "delta_ci_high": high, "summary": summary}
    return {**base, "candidate": DEFAULT_CANDIDATE.id, "reason": "improvement_not_significant",
            "best_rejected": best["candidate"], "delta_ci_high": high, "summary": summary}


# ---------------------------------------------------------------------------
# Dependence
# ---------------------------------------------------------------------------

def _fit_role_model(history: pd.DataFrame):
    """Legacy arm, unchanged: per-role Pearson on raw residuals, min_pair_rows=2."""
    by_game = [dict(zip(game["role_key"], game["residual"].astype(float)))
               for _, game in history.groupby("game_id", sort=True)]
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


def _dependent(independent: np.ndarray, roles: tuple[str, ...], model, seed: int) -> np.ndarray:
    output = independent.copy()
    if model is None:
        return output
    available = [i for i, role in enumerate(roles) if model.covers(role)]
    if len(available) >= 2:
        output[:, available] = induce_role_rank_dependence(
            independent[:, available], role_keys=tuple(roles[i] for i in available),
            correlation_model=model, seed=seed)
    return output


# ---------------------------------------------------------------------------
# Metrics (one implementation, used by both the run and the verifier)
# ---------------------------------------------------------------------------

def _player_metrics(values: np.ndarray, truth: float) -> dict:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("invalid simulation draws for a player-game")
    mean = float(values.mean())
    return {
        "mean": mean, "mae": abs(mean - truth), "bias": mean - truth,
        "crps": empirical_crps(values, truth),
        "coverage_50": float(np.quantile(values, .25) <= truth <= np.quantile(values, .75)),
        "coverage_80": float(np.quantile(values, .10) <= truth <= np.quantile(values, .90)),
    }


def _metric_identity(actual) -> dict:
    return {
        "team": str(actual.team), "position": str(actual.position),
        "predicted_points": float(actual.predicted_points), "is_cold_start": bool(actual.is_cold_start),
        "actual_points": float(actual.actual_points),
    }


def _marginal_player_metrics(draws: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Per-player metrics from a long draw frame (the verifier's path)."""
    actual_required = set(IDENTITY_COLUMNS) | {"actual_points", "team", "position",
                                               "predicted_points", "is_cold_start"}
    if missing := actual_required - set(actuals.columns):
        raise ValueError(f"actuals lack marginal-metric columns: {sorted(missing)}")
    if actuals.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("actual player-game rows are not unique")
    actual_by_key = actuals.set_index(list(IDENTITY_COLUMNS), verify_integrity=True)
    rows = []
    for key, group in draws.groupby(list(IDENTITY_COLUMNS), sort=True):
        if key not in actual_by_key.index:
            raise ValueError(f"simulated player-game has no actual row: {key}")
        actual = actual_by_key.loc[key]
        values = group.sort_values("draw")["fantasy_points"].to_numpy(float)
        rows.append({**dict(zip(IDENTITY_COLUMNS, key)), **_metric_identity(actual),
                     **_player_metrics(values, float(actual.actual_points))})
    return pd.DataFrame(rows)


def _marginal_from_matrix(rows: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    output = []
    for index, actual in enumerate(rows.itertuples(index=False)):
        output.append({**{column: getattr(actual, column) for column in IDENTITY_COLUMNS},
                       **_metric_identity(actual),
                       **_player_metrics(matrix[:, index], float(actual.actual_points))})
    frame = pd.DataFrame(output)
    return frame.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _game_view(game_rows: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Canonical player order for every joint calculation: sorted player_id."""
    order = np.argsort(game_rows["player_id"].astype(str).to_numpy(), kind="mergesort")
    return game_rows.iloc[order].reset_index(drop=True), matrix[:, order]


def _joint_game_row(game_rows: pd.DataFrame, matrix: np.ndarray) -> dict:
    rows, samples = _game_view(game_rows, matrix)
    observation = rows["actual_points"].to_numpy(float)
    return {"game_id": rows["game_id"].iloc[0], "dimensions": int(len(rows)),
            "energy_score": energy_score(samples, observation),
            "variogram_score_p05": variogram_score(samples, observation, p=.5),
            "variogram_score_p05_scaled": scaled_variogram_score(
                samples, observation, rows["predicted_points"].to_numpy(float), p=.5)}


def _sum_groups(rows: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(kind, side, member indices) for stack, team_total and game_total.

    stack = the team's highest-projected QB plus its two highest-projected
    WR/TE (the standard QB+2 stack); ties break on player_id.
    """
    groups = []
    ranked = rows.assign(_order=np.arange(len(rows))).sort_values(
        ["predicted_points", "player_id"], ascending=[False, True], kind="mergesort")
    for side, team in (("home", rows["home_team"].iloc[0]), ("away", rows["away_team"].iloc[0])):
        on_team = ranked[ranked["team"] == team]
        if on_team.empty:
            continue
        groups.append(("team_total", side, np.sort(on_team["_order"].to_numpy())))
        quarterbacks = on_team[on_team["position"].astype(str).str.upper() == "QB"]
        catchers = on_team[on_team["position"].astype(str).str.upper().isin(["WR", "TE"])]
        if len(quarterbacks) and len(catchers) >= STACK_PASS_CATCHERS:
            members = np.concatenate([quarterbacks["_order"].to_numpy()[:1],
                                      catchers["_order"].to_numpy()[:STACK_PASS_CATCHERS]])
            groups.append(("stack", side, np.sort(members)))
    groups.append(("game_total", "both", np.arange(len(rows))))
    return groups


def _sum_rows(game_rows: pd.DataFrame, matrix: np.ndarray) -> list[dict]:
    rows, samples = _game_view(game_rows, matrix)
    actual = rows["actual_points"].to_numpy(float)
    output = []
    for kind, side, members in _sum_groups(rows):
        totals = samples[:, members].sum(axis=1)
        truth = float(actual[members].sum())
        output.append({
            "season": int(rows["season"].iloc[0]), "week": int(rows["week"].iloc[0]),
            "game_id": rows["game_id"].iloc[0], "sum_kind": kind, "side": side,
            "members": int(len(members)), "actual_sum": truth, "mean_sum": float(totals.mean()),
            "crps": empirical_crps(totals, truth),
            "coverage_80": float(np.quantile(totals, .10) <= truth <= np.quantile(totals, .90)),
        })
    return output


def _aggregate_marginal(rows: pd.DataFrame) -> dict:
    return {
        "rows": int(len(rows)), "mae": float(rows["mae"].mean()),
        "mean_bias": float(rows["bias"].mean()), "crps": float(rows["crps"].mean()),
        "coverage_50": float(rows["coverage_50"].mean()),
        "coverage_80": float(rows["coverage_80"].mean()),
    }


def _marginal_segments(rows: pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy()
    frame["predicted_activity"] = np.where(frame["predicted_points"] > ZERO_FLOOR,
                                            "predicted_active", "predicted_near_zero")
    frame["experience"] = np.where(frame["is_cold_start"], "cold_start", "returning")
    frame["actual_activity"] = np.where(frame["actual_points"] > ZERO_FLOOR,
                                         "nonzero_actual", "near_zero_actual")

    def add_entry(segment: str, value: str, mode: str, part: pd.DataFrame) -> dict:
        entry = {"segment": segment, "value": value, "mode": mode, **_aggregate_marginal(part)}
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
    for segment in ("position", "predicted_activity", "experience", "actual_activity"):
        for value, group in frame.groupby(segment, sort=True):
            for mode, part in group.groupby("mode", sort=True):
                output.append(add_entry(segment, str(value), mode, part))
    return pd.DataFrame(output)


# ---------------------------------------------------------------------------
# Draw storage
# ---------------------------------------------------------------------------

@dataclass
class SimulationDraws:
    """Scored rows plus one (draws x rows) matrix per mode, same column order."""
    rows: pd.DataFrame
    matrices: dict[str, np.ndarray]

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(self.matrices)

    def long_frame(self, mode: str) -> pd.DataFrame:
        matrix = self.matrices[mode]
        n_draws, n_rows = matrix.shape
        frame = {"mode": np.full(n_rows * n_draws, mode, dtype=object),
                 "draw": np.tile(np.arange(n_draws), n_rows)}
        for column in (*IDENTITY_COLUMNS, *DRAW_METADATA_COLUMNS):
            frame[column] = np.repeat(self.rows[column].to_numpy(), n_draws)
        frame["fantasy_points"] = matrix.T.reshape(-1)
        return pd.DataFrame(frame)


# ---------------------------------------------------------------------------
# Comparisons and gates
# ---------------------------------------------------------------------------

def _table(tables: dict[str, pd.DataFrame], name: str) -> tuple[pd.DataFrame, list[str]]:
    if name == "marginal":
        return tables["marginal"], list(IDENTITY_COLUMNS)
    if name == "joint":
        return tables["joint"], ["game_id"]
    if name.startswith("sum:"):
        kind = name.split(":", 1)[1]
        return tables["sums"].loc[tables["sums"]["sum_kind"] == kind], ["game_id", "sum_kind", "side"]
    raise ValueError(f"unknown comparison table {name!r}")


def paired_comparison(tables: dict[str, pd.DataFrame], table: str, metric: str, candidate: str,
                      baseline: str, *, cluster: str = "week_cluster", n_boot: int, seed: int) -> dict:
    frame, keys = _table(tables, table)
    left = frame.loc[frame["mode"] == candidate].set_index(keys).sort_index()
    right = frame.loc[frame["mode"] == baseline].set_index(keys).sort_index()
    if not left.index.equals(right.index):
        raise ValueError(f"{candidate} and {baseline} {table} rows are not identical")
    delta = left[metric].to_numpy(float) - right[metric].to_numpy(float)
    entry = {"table": table, "metric": metric, "candidate": candidate, "baseline": baseline,
             "cluster": cluster, "n": int(len(delta)),
             "point_estimate": float(delta.mean()) if len(delta) else float("nan")}
    clusters = (left.index.get_level_values(cluster) if cluster in keys else left[cluster]).to_numpy()
    draws = cluster_bootstrap_distribution(
        delta, clusters, statistic=np.mean, n_boot=n_boot,
        seed=_seed(seed, f"{table}:{metric}:{candidate}:{baseline}:{cluster}"))
    if not len(draws):
        return {**entry, "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"),
                "status": "insufficient_clusters"}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {**entry, "ci_low": float(low), "ci_high": float(high),
            "p_value": bootstrap_two_sided_p_value(draws), "status": "complete"}


def _comparisons(tables: dict[str, pd.DataFrame], *, n_boot: int, seed: int) -> dict:
    primary = [paired_comparison(tables, *spec, n_boot=n_boot, seed=seed) for spec in PRIMARY_COMPARISONS]
    significant = holm_bonferroni([entry["p_value"] for entry in primary], alpha=ALPHA)
    for entry, flag in zip(primary, significant):
        entry["significant_holm"] = bool(flag)
        entry["favours_candidate"] = bool(flag and entry["point_estimate"] < 0)
        entry["favours_baseline"] = bool(flag and entry["point_estimate"] > 0)
    secondary = []
    for mode in MODES[1:]:
        for metric in ("crps", "coverage_80"):
            secondary.append(paired_comparison(tables, "marginal", metric, mode, PRODUCTION,
                                               n_boot=n_boot, seed=seed))
    secondary.append(paired_comparison(tables, "marginal", "crps", INDEPENDENT, LEGACY_INDEPENDENT,
                                       cluster="player_id", n_boot=n_boot, seed=seed))
    for mode in (TEAM_FACTOR, ROLE_FACTOR, LEGACY_ROLE):
        for metric in ("energy_score", "variogram_score_p05"):
            secondary.append(paired_comparison(tables, "joint", metric, mode, INDEPENDENT,
                                               n_boot=n_boot, seed=seed))
        for kind in ("team_total", "game_total"):
            secondary.append(paired_comparison(tables, f"sum:{kind}", "crps", mode, INDEPENDENT,
                                               n_boot=n_boot, seed=seed))
    for table, metric in (("joint", "variogram_score_p05_scaled"), ("sum:stack", "crps")):
        secondary.append(paired_comparison(tables, table, metric, LEGACY_ROLE, INDEPENDENT,
                                           n_boot=n_boot, seed=seed))
    return {"primary": primary, "secondary": secondary,
            "note": "primary = pre-registered, Holm-corrected; secondary = uncorrected, descriptive"}


def _promotion_gate(mode_reports: dict, comparisons: dict) -> dict:
    """Apply the pre-registered research gate without promoting any artifact."""
    primary = comparisons["primary"]

    def find(candidate, baseline, table):
        return next(e for e in primary if e["candidate"] == candidate and e["baseline"] == baseline
                    and e["table"] == table)

    marginal = next(e for e in comparisons["secondary"] if e["table"] == "marginal"
                    and e["metric"] == "crps" and e["candidate"] == INDEPENDENT
                    and e["baseline"] == PRODUCTION)
    coverage = mode_reports[INDEPENDENT]["coverage_80"]
    independent_ok = bool(marginal["status"] == "complete" and marginal["ci_high"] < 0
                          and COVERAGE_80_LOWER <= coverage <= COVERAGE_80_UPPER)
    gate = {INDEPENDENT: {
        "eligible_for_later_comparison": independent_ok,
        "coverage_80_in_band": bool(COVERAGE_80_LOWER <= coverage <= COVERAGE_80_UPPER),
        "improves_on_legacy_holm": find(INDEPENDENT, LEGACY_INDEPENDENT, "marginal")["favours_candidate"],
    }}
    for mode in FACTOR_MODES:
        entries = [find(mode, INDEPENDENT, "joint"), find(mode, INDEPENDENT, "sum:stack")]
        improves = any(e["favours_candidate"] for e in entries)
        worsens = any(e["favours_baseline"] for e in entries)
        gate[mode] = {"eligible_for_later_comparison": bool(independent_ok and improves and not worsens),
                      "improves_joint_primary_holm": improves, "worsens_joint_primary_holm": worsens}
    gate["role_structure_over_team_factor"] = any(
        find(ROLE_FACTOR, TEAM_FACTOR, table)["favours_candidate"] for table in ("joint", "sum:stack"))
    gate[LEGACY_ROLE] = "comparison_only; not a promotion candidate"
    gate["promotion"] = "none; this report is a research gate only"
    return gate


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRun:
    report: dict
    draws: SimulationDraws
    actuals: pd.DataFrame
    marginal: pd.DataFrame
    joint_by_game: pd.DataFrame
    sums: pd.DataFrame
    evidence: pd.DataFrame
    correlation_calibration: pd.DataFrame
    calibrations: dict = field(default_factory=dict)
    factor_models: dict = field(default_factory=dict)


def _phase(panel: pd.DataFrame, confirm_season, run_confirmation: bool) -> tuple[pd.DataFrame, str]:
    if confirm_season is None:
        if run_confirmation:
            raise ValueError("run_confirmation requires confirm_season")
        return panel, "unrestricted"
    confirm_season = int(confirm_season)
    if confirm_season != int(panel["season"].max()):
        raise ValueError("the confirmation season must be the panel's latest season")
    if run_confirmation:
        return panel, "confirmation"
    return panel.loc[panel["season"].astype(int) != confirm_season].copy(), "development"


def minimum_bootstrap_for_holm(n_tests: int = len(PRIMARY_COMPARISONS), alpha: float = ALPHA) -> int:
    """Smallest n_boot whose smallest attainable p-value, 2/(n+1), is below
    Holm's first threshold alpha/m. Fewer resamples make every primary
    comparison unrejectable no matter how large the effect."""
    return int(np.floor(2 * n_tests / alpha))


def run(panel: pd.DataFrame, *, draws: int, seed: int,
        candidates: tuple[Candidate, ...] | None = None, n_bootstrap: int = N_BOOTSTRAP,
        selection_bootstrap: int = SELECTION_BOOTSTRAP, confirm_season: int | None = None,
        run_confirmation: bool = False) -> EvaluationRun:
    """Run the causal rolling comparison without writing artifacts."""
    if draws < 2:
        raise ValueError("draws must be at least two")
    if n_bootstrap < minimum_bootstrap_for_holm():
        raise ValueError(f"n_bootstrap={n_bootstrap} cannot reach Holm significance for "
                         f"{len(PRIMARY_COMPARISONS)} primaries; use >= {minimum_bootstrap_for_holm()}")
    candidates = candidates or candidate_grid()
    legacy_candidates = tuple(c for c in candidates if c.family == "legacy")
    _validate_panel(panel)
    panel, phase = _phase(panel.reset_index(drop=True), confirm_season, run_confirmation)
    panel = panel.reset_index(drop=True)
    panel["role_key"] = _attach_roles(panel)
    week_ids = _week_ids(panel)
    first_season = int(panel["season"].min())
    eligible = sorted({int(w) for w in week_ids[panel["season"].astype(int).to_numpy() > first_season]})
    scored_weeks = [w for w in eligible if phase != "confirmation" or w // 100 == int(confirm_season)]
    if not scored_weeks:
        raise ValueError("no week has an earlier panel season to learn from")

    evidence_parts, selections, scored_parts = [], [], []
    matrix_parts = {mode: [] for mode in MODES}
    implied: dict[tuple, list[float]] = {}
    calibrations, factor_models = {}, {}
    for target in eligible:
        history = panel.loc[week_ids < target]
        current = panel.loc[week_ids == target]
        fits = {candidate.id: fit_candidate(candidate, history) for candidate in candidates}
        week_evidence = []
        for candidate in candidates:
            crps, covered = support_metrics(fits[candidate.id], current)
            week_evidence.append(pd.DataFrame({
                "candidate": candidate.id, "week_id": target, "season": current["season"].to_numpy(),
                "week": current["week"].to_numpy(), "player_id": current["player_id"].to_numpy(),
                "crps": crps, "coverage_80": covered}))
        if target in scored_weeks:
            prior = pd.concat(evidence_parts, ignore_index=True) if evidence_parts else pd.DataFrame(
                columns=["candidate", "week_id", "player_id", "crps", "coverage_80"])
            full = select_candidate(prior, target_week_id=target, candidates=candidates,
                                    n_boot=selection_bootstrap, seed=seed)
            legacy = select_candidate(prior, target_week_id=target, candidates=legacy_candidates,
                                      n_boot=selection_bootstrap, seed=seed)
            calibration, legacy_calibration = fits[full["candidate"]], fits[legacy["candidate"]]
            pairs = pooled_pair_correlations(normal_scores(history, calibration),
                                             history["role_key"].tolist(), history["game_id"].to_numpy())
            models = {TEAM_FACTOR: fit_factor_copula(pairs, structure="team"),
                      ROLE_FACTOR: fit_factor_copula(pairs, structure="role"),
                      LEGACY_ROLE: _fit_role_model(history)}
            week_parts = []
            for game_id, game in current.groupby("game_id", sort=True):
                game_seed = _seed(seed, f"game:{game_id}")
                predicted = game["predicted_points"].to_numpy(float)
                independent = independent_residual_matrix(game, calibration, n_draws=draws, seed=game_seed)
                roles = tuple(game["role_key"])
                matrix_parts[PRODUCTION].append(np.broadcast_to(predicted, (draws, len(game))).copy())
                matrix_parts[LEGACY_INDEPENDENT].append(predicted + independent_residual_matrix(
                    game, legacy_calibration, n_draws=draws, seed=game_seed))
                matrix_parts[INDEPENDENT].append(predicted + independent)
                for mode in (TEAM_FACTOR, ROLE_FACTOR, LEGACY_ROLE):
                    matrix_parts[mode].append(predicted + _dependent(
                        independent, roles, models[mode], game_seed + 1))
                week_parts.append(game.assign(normal_score=normal_scores(game, calibration)))
            scored_parts.extend(week_parts)
            week_rows = pd.concat(week_parts)
            observed = pooled_pair_correlations(week_rows["normal_score"].to_numpy(),
                                                week_rows["role_key"].tolist(), week_rows["game_id"].to_numpy())
            for entry in observed.itertuples(index=False):
                key = (entry.role_a, entry.role_b, entry.relation)
                acc = implied.setdefault(key, [0.0, 0.0, 0.0])
                acc[0] += entry.n
                acc[1] += entry.n * models[TEAM_FACTOR].implied_correlation(*key)
                acc[2] += entry.n * models[ROLE_FACTOR].implied_correlation(*key)
            season, week = divmod(target, 100)
            selections.append({"season": season, "week": week, "full": full, "legacy": legacy,
                               "factor_status": {m: models[m].diagnostics.get("status") for m in FACTOR_MODES}})
            calibrations[season] = {"week": week, "full": calibration, "legacy": legacy_calibration}
            factor_models.setdefault(season, {})[week] = {m: models[m].to_dict() for m in FACTOR_MODES}
        evidence_parts.append(pd.concat(week_evidence, ignore_index=True))

    scored = pd.concat(scored_parts, ignore_index=True)
    matrices = {mode: np.concatenate(parts, axis=1) for mode, parts in matrix_parts.items()}
    if scored.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("scored player-games are not unique")
    draws_store = SimulationDraws(scored, matrices)
    actuals = scored.drop(columns=["normal_score"])

    marginal_frames, joint_frames, sum_frames, mode_reports = [], [], [], {}
    floors = panel.groupby(panel["position"].astype(str).str.upper())["actual_points"].min()
    row_floor = scored["position"].astype(str).str.upper().map(floors).to_numpy(float)
    game_blocks = [np.asarray(idx) for _, idx in sorted(scored.groupby("game_id").indices.items())]
    for mode in MODES:
        matrix = matrices[mode]
        marginal = _marginal_from_matrix(scored, matrix)
        marginal.insert(0, "mode", mode)
        joint_rows, sum_rows = [], []
        for idx in game_blocks:
            game_rows = scored.iloc[idx]
            joint_rows.append(_joint_game_row(game_rows, matrix[:, idx]))
            sum_rows.extend(_sum_rows(game_rows, matrix[:, idx]))
        joint = pd.DataFrame(joint_rows)
        joint.insert(0, "mode", mode)
        sums = pd.DataFrame(sum_rows)
        sums.insert(0, "mode", mode)
        marginal_frames.append(marginal)
        joint_frames.append(joint)
        sum_frames.append(sums)
        report = {**_aggregate_marginal(marginal),
                  "energy_score": float(joint["energy_score"].mean()),
                  "variogram_score_p05": float(joint["variogram_score_p05"].mean()),
                  "variogram_score_p05_scaled": float(joint["variogram_score_p05_scaled"].mean()),
                  "games_scored": int(len(joint)),
                  "support_violation_rate": float((matrix < row_floor).mean())}
        for kind in ("stack", "team_total", "game_total"):
            part = sums.loc[sums["sum_kind"] == kind]
            report[f"{kind}_crps"] = float(part["crps"].mean()) if len(part) else float("nan")
            report[f"{kind}_coverage_80"] = float(part["coverage_80"].mean()) if len(part) else float("nan")
        mode_reports[mode] = report
    marginal = pd.concat(marginal_frames, ignore_index=True)
    joint_by_game = pd.concat(joint_frames, ignore_index=True)
    sums = pd.concat(sum_frames, ignore_index=True)

    game_weeks = scored.drop_duplicates("game_id").set_index("game_id")
    week_cluster = _week_cluster(game_weeks)
    tables = {
        "marginal": marginal.assign(week_cluster=_week_cluster(marginal)),
        "joint": joint_by_game.assign(week_cluster=joint_by_game["game_id"].map(week_cluster)),
        "sums": sums.assign(week_cluster=_week_cluster(sums)),
    }
    comparisons = _comparisons(tables, n_boot=n_bootstrap, seed=seed)
    correlation_calibration = pd.DataFrame([
        {"role_a": a, "role_b": b, "relation": relation, "n": int(values[0]),
         "team_factor_implied": values[1] / values[0], "role_factor_implied": values[2] / values[0]}
        for (a, b, relation), values in sorted(implied.items()) if values[0] > 0])
    heldout = pooled_pair_correlations(scored["normal_score"].to_numpy(), scored["role_key"].tolist(),
                                       scored["game_id"].to_numpy())
    if len(correlation_calibration):
        correlation_calibration = correlation_calibration.merge(
            heldout.rename(columns={"correlation": "heldout_correlation", "n": "heldout_n"}),
            on=["role_a", "role_b", "relation"], how="left", validate="one_to_one")
    evidence = pd.concat(evidence_parts, ignore_index=True)
    zero_share = (scored.assign(zero=scored["actual_points"] <= ZERO_FLOOR)
                  .groupby("position")["zero"].mean())
    report = {
        "status": "complete", "phase": phase, "draws_per_player": int(draws),
        "confirm_season": None if confirm_season is None else int(confirm_season),
        "modes": mode_reports,
        "scoring_population": {"rows": int(len(scored)), "games": int(scored["game_id"].nunique()),
                               "weeks": int(len(scored_weeks)),
                               "seasons": sorted(int(s) for s in scored["season"].unique())},
        "selection": selections,
        "selected_candidate_counts": pd.Series(
            [entry["full"]["candidate"] for entry in selections]).value_counts().to_dict(),
        "calibration": {"candidates": [c.id for c in candidates], "default": DEFAULT_CANDIDATE.id,
                        "coverage_80_gate": [COVERAGE_80_LOWER, COVERAGE_80_UPPER],
                        "selection_bootstrap": int(selection_bootstrap),
                        "refit": "weekly, all rows with (season, week) < target"},
        "comparisons": comparisons,
        "promotion_gate": _promotion_gate(mode_reports, comparisons),
        "panel_scope": {"caveat": PANEL_SCOPE_CAVEAT,
                        "near_zero_actual_share_by_position": {str(k): float(v) for k, v in zero_share.items()}},
    }
    return EvaluationRun(report, draws_store, actuals, marginal, joint_by_game, sums, evidence,
                         correlation_calibration, calibrations, factor_models)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(value, digits=3) -> str:
    return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def summary_markdown(report: dict, correlation_calibration: pd.DataFrame) -> str:
    population = report["scoring_population"]
    lines = [f"# Calibrated simulation backtest ({report['phase']})", "",
             f"Scored {population['rows']} player-games, {population['games']} games, "
             f"{population['weeks']} weeks, seasons {population['seasons']}; "
             f"{report['draws_per_player']} draws per player. Lower is better except coverage "
             f"(target 0.50 / 0.80). Every calibrated support is centred on the served prediction, "
             f"so MAE differences from production_marginal are Monte Carlo noise in the draw mean.",
             "", "## Modes", "",
             "| mode | MAE | CRPS | cov50 | cov80 | energy | variogram | scaled variogram | stack CRPS "
             "| stack cov80 | team CRPS | game CRPS | below support |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, m in report["modes"].items():
        lines.append(f"| {mode} | {_fmt(m['mae'])} | {_fmt(m['crps'])} | {_fmt(m['coverage_50'])} | "
                     f"{_fmt(m['coverage_80'])} | {_fmt(m['energy_score'])} | {_fmt(m['variogram_score_p05'])} | "
                     f"{_fmt(m['variogram_score_p05_scaled'], 4)} | "
                     f"{_fmt(m['stack_crps'])} | {_fmt(m['stack_coverage_80'])} | {_fmt(m['team_total_crps'])} | "
                     f"{_fmt(m['game_total_crps'])} | {_fmt(m['support_violation_rate'], 4)} |")
    lines += ["", "## Pre-registered primary comparisons",
              "", f"Week-block bootstrap, Holm-corrected at alpha={ALPHA}. Negative delta favours the candidate.",
              "", "| candidate | baseline | table | metric | delta | 95% CI | p | Holm |", "|---|---|---|---|---|---|---|---|"]
    for e in report["comparisons"]["primary"]:
        lines.append(f"| {e['candidate']} | {e['baseline']} | {e['table']} | {e['metric']} | "
                     f"{_fmt(e['point_estimate'], 4)} | [{_fmt(e['ci_low'], 4)}, {_fmt(e['ci_high'], 4)}] | "
                     f"{_fmt(e['p_value'], 4)} | {'significant' if e['significant_holm'] else '-'} |")
    lines += ["", "## Gate", "", "```", json.dumps(report["promotion_gate"], indent=2), "```", "",
              "## Marginal family selected per week", ""]
    for candidate, count in sorted(report["selected_candidate_counts"].items()):
        lines.append(f"- {candidate}: {count} weeks")
    if len(correlation_calibration):
        lines += ["", "## Dependence calibration (held-out normal scores vs model-implied, largest classes)", "",
                  "| roles | relation | n | held-out | team factor | role factor |", "|---|---|---|---|---|---|"]
        top = correlation_calibration.sort_values("n", ascending=False, kind="mergesort").head(15)
        for r in top.itertuples(index=False):
            lines.append(f"| {r.role_a}-{r.role_b} | {r.relation} | {r.n} | {_fmt(r.heldout_correlation)} | "
                         f"{_fmt(r.team_factor_implied)} | {_fmt(r.role_factor_implied)} |")
    lines += ["", "## Caveat", "", report["panel_scope"]["caveat"], ""]
    return "\n".join(lines)


def _write_run(output_dir: Path, result: EvaluationRun, *, oof_run_dir: Path,
               oof_verification: dict, seed: int, n_bootstrap: int, config: dict) -> None:
    if output_dir.exists():
        raise ValueError(f"output directory exists: {output_dir}")
    final_output_dir = output_dir
    output_dir = final_output_dir.with_name(f".{final_output_dir.name}.staging-{uuid.uuid4().hex}")
    (output_dir / "player_draws").mkdir(parents=True)
    written = []

    def track(path: Path) -> Path:
        written.append(path)
        return path

    for mode in result.draws.modes:
        # One mode at a time keeps peak memory to a single long frame.
        result.draws.long_frame(mode).to_parquet(track(output_dir / "player_draws" / f"{mode}.parquet"),
                                                 index=False)
    result.actuals.to_parquet(track(output_dir / "actuals.parquet"), index=False)
    result.marginal.to_csv(track(output_dir / "marginal_player_metrics.csv"), index=False)
    result.joint_by_game.to_csv(track(output_dir / "joint_game_metrics.csv"), index=False)
    result.sums.to_csv(track(output_dir / "joint_sum_metrics.csv"), index=False)
    _marginal_segments(result.marginal).to_csv(track(output_dir / "marginal_segment_report.csv"), index=False)
    result.evidence.to_csv(track(output_dir / "candidate_evidence.csv"), index=False)
    result.correlation_calibration.to_csv(track(output_dir / "correlation_calibration.csv"), index=False)
    track(output_dir / "selection_log.json").write_text(
        json.dumps(result.report["selection"], indent=2, sort_keys=True) + "\n")
    for season, entry in sorted(result.calibrations.items()):
        folder = output_dir / "outer_folds" / str(season)
        save_calibration(entry["full"], track(folder / f"calibration_full_week{entry['week']:02d}.json"))
        save_calibration(entry["legacy"], track(folder / f"calibration_legacy_week{entry['week']:02d}.json"))
    for season, weeks in sorted(result.factor_models.items()):
        path = track(output_dir / "outer_folds" / str(season) / "factor_models.json")
        path.write_text(json.dumps({str(w): models for w, models in sorted(weeks.items())},
                                   indent=2, sort_keys=True) + "\n")
    report = dict(result.report)
    report["oof_verification"] = oof_verification
    report["reporting"] = {"week_block_bootstrap": int(n_bootstrap), "seed": int(seed)}
    report["config"] = config
    track(output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    track(output_dir / "summary.md").write_text(summary_markdown(report, result.correlation_calibration))
    manifest = {
        "schema_version": 2, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_run_dir": str(oof_run_dir.resolve()),
        "source_oof_panel_sha256": oof_verification["panel_sha256"],
        "draws_per_player": int(report["draws_per_player"]), "seed": int(seed),
        "phase": report["phase"], "modes": list(result.draws.modes),
        "config": config, "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
        "n_rows": int(report["scoring_population"]["rows"]),
        "n_games": int(report["scoring_population"]["games"]),
        "scoring_seasons": report["scoring_population"]["seasons"],
        "files": {str(path.relative_to(output_dir)): _sha256(path) for path in written},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Publish only after every row-level file, fitted artifact, report, and
    # manifest hash exists. A failed run leaves an obviously temporary
    # staging directory rather than a misleading immutable result directory.
    output_dir.rename(final_output_dir)


def run_config(args, candidates: tuple[Candidate, ...]) -> dict:
    from src.models.position_models import _git_commit
    return {"git_commit": _git_commit(), "draws": int(args.draws), "seed": int(args.seed),
            "n_bootstrap": int(args.n_bootstrap), "selection_bootstrap": SELECTION_BOOTSTRAP,
            "candidates": [c.id for c in candidates], "confirm_season": args.confirm_season,
            "run_confirmation": bool(args.run_confirmation),
            "primary_comparisons": [list(spec) for spec in PRIMARY_COMPARISONS],
            "coverage_80_gate": [COVERAGE_80_LOWER, COVERAGE_80_UPPER], "alpha": ALPHA}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--legacy-min-stratum-rows", type=int, nargs="+", default=list(LEGACY_MIN_STRATUM_ROWS))
    parser.add_argument("--analog-k", type=int, nargs="+", default=list(ANALOG_K))
    parser.add_argument("--confirm-season", type=int, default=None,
                        help="withhold this (latest) season from development runs")
    parser.add_argument("--run-confirmation", action="store_true",
                        help="score the withheld season once; use only after design is frozen")
    args = parser.parse_args()
    if args.draws < 2 or args.n_bootstrap < minimum_bootstrap_for_holm():
        parser.error(f"--draws must be >= 2 and --n-bootstrap >= {minimum_bootstrap_for_holm()}")
    if args.run_confirmation and args.confirm_season is None:
        parser.error("--run-confirmation requires --confirm-season")
    candidates = candidate_grid(args.legacy_min_stratum_rows, args.analog_k)
    config = run_config(args, candidates)
    verification = verify(args.oof_run_dir)
    # Simulate the game each actual came from, not the forecast origin row.
    panel = target_game_panel(pd.read_parquet(args.oof_run_dir / "panel.parquet"))
    result = run(panel, draws=args.draws, seed=args.seed, candidates=candidates,
                 n_bootstrap=args.n_bootstrap, confirm_season=args.confirm_season,
                 run_confirmation=args.run_confirmation)
    _write_run(args.output_dir, result, oof_run_dir=args.oof_run_dir, oof_verification=verification,
               seed=args.seed, n_bootstrap=args.n_bootstrap, config=config)
    from scripts.verify_calibrated_simulation import verify as verify_run
    verification_result = verify_run(args.output_dir)
    (args.output_dir / "verification.json").write_text(json.dumps(verification_result, indent=2, sort_keys=True) + "\n")
    print((args.output_dir / "summary.md").read_text())
    print(f"verified: {verification_result['status']} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

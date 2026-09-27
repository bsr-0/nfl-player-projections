"""Calibrated player simulation: one implementation for backtest and production.

A player's draws are the served prediction plus deviations from a calibrated
marginal (``residual_calibration``), reordered into game-level dependence by a
role-factor Gaussian copula (``player_correlation.FactorCopulaModel``).
``scripts/evaluate_calibrated_simulation.py`` measures this against
alternatives; ``scripts/fit_simulation_artifacts.py`` fits it for serving and
``scripts/generate_simulation_data.py`` serves it. All three call the helpers
here so the served draws are exactly the backtest's ``calibrated_role_factor``
draws for the same artifacts and seed.

Scope: fitted on OOF player-games where the player appeared, so draws are
conditional on playing; availability is not modelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable
import uuid

import numpy as np
import pandas as pd

from src.models.oof_capture import cluster_bootstrap_distribution
from src.models.player_correlation import FactorCopulaModel, fit_factor_copula, pooled_pair_correlations
from src.models.residual_calibration import (
    ANALOG_FAMILIES,
    fit_empirical_residual_calibration,
    fit_prediction_analog_calibration,
    independent_residual_matrix,
    induce_role_rank_dependence,
    load_calibration,
    normal_scores,
    role_keys_from_panel_game,
    save_calibration,
)
from src.utils.atomic_io import atomic_write_json

LEGACY_MIN_STRATUM_ROWS = (25, 50, 100)
ANALOG_K = (100, 250, 500)
DEFAULT_MIN_STRATUM_ROWS = 50
COVERAGE_80_LOWER = 0.75
COVERAGE_80_UPPER = 0.85
SELECTION_BOOTSTRAP = 200
STACK_PASS_CATCHERS = 2
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "calibrated_simulation_artifacts"
LATEST_POINTER = "latest.json"
SCOPE_CAVEAT = ("Draws are conditional on the player appearing: the calibration is fitted "
                "on OOF player-games that happened, and availability is not modelled.")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def derived_seed(seed: int, value: str) -> int:
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2**63 - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def week_ids(frame: pd.DataFrame) -> np.ndarray:
    return frame["season"].astype(int).to_numpy() * 100 + frame["week"].astype(int).to_numpy()


def validate_panel(panel: pd.DataFrame) -> None:
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


def attach_roles(rows: pd.DataFrame) -> pd.Series:
    """Role keys from prediction-time fields only, computed once per game."""
    roles = pd.Series(index=rows.index, dtype=object)
    for _, game in rows.groupby("game_id", sort=True):
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

    @classmethod
    def parse(cls, candidate_id: str) -> "Candidate":
        family, _, size = str(candidate_id).partition(":")
        if family not in ("legacy", *ANALOG_FAMILIES) or not size.isdigit():
            raise ValueError(f"not a candidate id: {candidate_id!r}")
        return cls(family, int(size))


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


def evidence_for_week(history: pd.DataFrame, current: pd.DataFrame, candidates: tuple[Candidate, ...],
                      target_week_id: int) -> tuple[dict, pd.DataFrame]:
    """Fit every candidate on ``history`` and score it on ``current``'s rows."""
    fits = {candidate.id: fit_candidate(candidate, history) for candidate in candidates}
    parts = []
    for candidate in candidates:
        crps, covered = support_metrics(fits[candidate.id], current)
        parts.append(pd.DataFrame({
            "candidate": candidate.id, "week_id": int(target_week_id),
            "season": current["season"].to_numpy(), "week": current["week"].to_numpy(),
            "player_id": current["player_id"].to_numpy(), "crps": crps, "coverage_80": covered}))
    return fits, pd.concat(parts, ignore_index=True)


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
                                           seed=derived_seed(seed, f"select:{target_week_id}:{','.join(ids)}"))
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
# Dependence and game draws
# ---------------------------------------------------------------------------

def normal_score_pairs(history: pd.DataFrame, calibration) -> pd.DataFrame:
    """Pooled same-game pair correlations of normal scores under ``calibration``."""
    return pooled_pair_correlations(normal_scores(history, calibration),
                                    history["role_key"].tolist(), history["game_id"].to_numpy())


def fit_role_factor(history: pd.DataFrame, calibration) -> FactorCopulaModel:
    """Role-factor copula on normal scores under ``calibration``'s marginal."""
    return fit_factor_copula(normal_score_pairs(history, calibration), structure="role")


def apply_dependence(independent: np.ndarray, roles: tuple[str, ...], model, seed: int) -> np.ndarray:
    output = independent.copy()
    if model is None:
        return output
    available = [i for i, role in enumerate(roles) if model.covers(role)]
    if len(available) >= 2:
        output[:, available] = induce_role_rank_dependence(
            independent[:, available], role_keys=tuple(roles[i] for i in available),
            correlation_model=model, seed=seed)
    return output


def game_seed(seed: int, game_id: str) -> int:
    return derived_seed(seed, f"game:{game_id}")


def simulate_game(rows: pd.DataFrame, *, calibration, dependence, n_draws: int, seed: int) -> np.ndarray:
    """(n_draws x players) fantasy points for one game, columns in ``rows`` order.

    The same two steps, seeds included, that produce the backtest's
    independent and role-factor arms.
    """
    required = {"game_id", "role_key", "position", "predicted_points", "is_cold_start"}
    if missing := required - set(rows.columns):
        raise ValueError(f"game rows lack simulation columns: {sorted(missing)}")
    game_ids = rows["game_id"].unique()
    if len(game_ids) != 1:
        raise ValueError("simulate_game takes the rows of exactly one game")
    predicted = rows["predicted_points"].to_numpy(float)
    if not np.isfinite(predicted).all():
        raise ValueError("served predictions must be finite")
    base_seed = game_seed(seed, str(game_ids[0]))
    independent = independent_residual_matrix(rows, calibration, n_draws=n_draws, seed=base_seed)
    return predicted + apply_dependence(independent, tuple(rows["role_key"]), dependence, base_seed + 1)


def sum_groups(rows: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
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


# ---------------------------------------------------------------------------
# Production artifacts
# ---------------------------------------------------------------------------

@dataclass
class SimulationArtifacts:
    calibration: object
    dependence: FactorCopulaModel
    selection: dict
    known_player_ids: frozenset
    fitted_through: tuple[int, int]
    n_rows: int
    manifest: dict | None = None


def fit_simulation_artifacts(panel: pd.DataFrame, *, candidates: tuple[Candidate, ...] | None = None,
                             selection_bootstrap: int = SELECTION_BOOTSTRAP,
                             seed: int = 0) -> SimulationArtifacts:
    """Fit the serving calibration and copula on every panel row.

    Same rule as the backtest at its next week: build rolling evidence over
    every week that has an earlier panel season, select the marginal family
    for the week after the panel's last, and fit it and the role-factor copula
    on all rows. ``panel`` must already be keyed to the target game
    (``oof_capture.target_game_panel``).
    """
    candidates = candidates or candidate_grid()
    validate_panel(panel)
    panel = panel.reset_index(drop=True).copy()
    panel["role_key"] = attach_roles(panel)
    ids = week_ids(panel)
    first_season = int(panel["season"].min())
    eligible = sorted({int(w) for w in ids[panel["season"].astype(int).to_numpy() > first_season]})
    evidence = [evidence_for_week(panel.loc[ids < target], panel.loc[ids == target], candidates, target)[1]
                for target in eligible]
    evidence_frame = (pd.concat(evidence, ignore_index=True) if evidence else
                      pd.DataFrame(columns=["candidate", "week_id", "player_id", "crps", "coverage_80"]))
    last = int(ids.max())
    selection = select_candidate(evidence_frame, target_week_id=last + 1, candidates=candidates,
                                 n_boot=selection_bootstrap, seed=seed)
    calibration = fit_candidate(Candidate.parse(selection["candidate"]), panel)
    dependence = fit_role_factor(panel, calibration)
    return SimulationArtifacts(calibration, dependence, selection,
                               frozenset(panel["player_id"].astype(str)), divmod(last, 100), int(len(panel)))


def save_simulation_artifacts(artifacts: SimulationArtifacts, root: str | Path, *,
                              provenance: dict) -> Path:
    """Write ``root/<run_id>/`` atomically, then point ``root/latest.json`` at it."""
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final = root / run_id
    if final.exists():
        raise ValueError(f"artifact directory exists: {final}")
    staging = root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    save_calibration(artifacts.calibration, staging / "calibration.json")
    (staging / "dependence.json").write_text(
        json.dumps(artifacts.dependence.to_dict(), indent=2, sort_keys=True) + "\n")
    manifest = {
        "kind": ARTIFACT_KIND, "schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fitted_through": {"season": artifacts.fitted_through[0], "week": artifacts.fitted_through[1]},
        "n_rows": artifacts.n_rows, "marginal_candidate": artifacts.selection["candidate"],
        "selection": artifacts.selection, "dependence": "role_factor",
        "known_player_ids": sorted(artifacts.known_player_ids), "provenance": provenance,
        "scope_caveat": SCOPE_CAVEAT,
        "files": {name: sha256_file(staging / name) for name in ("calibration.json", "dependence.json")},
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    staging.rename(final)
    atomic_write_json({"run_id": run_id}, root / LATEST_POINTER)
    return final


def resolve_artifacts_dir(path: str | Path) -> Path:
    """An artifact run directory, or a root whose latest.json names one."""
    path = Path(path)
    if (path / "manifest.json").exists():
        return path
    pointer = path / LATEST_POINTER
    if not pointer.exists():
        raise FileNotFoundError(
            f"no simulation artifacts at {path}; fit them with scripts/fit_simulation_artifacts.py")
    run_dir = path / json.loads(pointer.read_text())["run_id"]
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"{pointer} names a missing artifact directory: {run_dir}")
    return run_dir


def load_simulation_artifacts(path: str | Path) -> SimulationArtifacts:
    """Load and verify; any mismatch fails closed."""
    run_dir = resolve_artifacts_dir(path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("kind") != ARTIFACT_KIND or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported simulation artifact manifest")
    for name in ("calibration.json", "dependence.json"):
        if sha256_file(run_dir / name) != manifest["files"].get(name):
            raise ValueError(f"simulation artifact hash mismatch: {name}")
    calibration = load_calibration(run_dir / "calibration.json")
    candidate = Candidate.parse(manifest["marginal_candidate"])
    expected = (getattr(calibration, "min_stratum_rows", None) if candidate.family == "legacy"
                else getattr(calibration, "k", None))
    if expected != candidate.size or (candidate.family != "legacy"
                                      and getattr(calibration, "family", None) != candidate.family):
        raise ValueError("calibration artifact does not match the manifest's selected candidate")
    dependence = FactorCopulaModel.from_dict(json.loads((run_dir / "dependence.json").read_text()))
    if dependence.structure != "role":
        raise ValueError("serving dependence must be the role-factor copula")
    through = manifest["fitted_through"]
    return SimulationArtifacts(calibration, dependence, manifest["selection"],
                               frozenset(manifest["known_player_ids"]),
                               (int(through["season"]), int(through["week"])), int(manifest["n_rows"]),
                               manifest)

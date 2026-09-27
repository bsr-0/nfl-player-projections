"""Evaluation for multivariate game/player simulation ensembles.

Inputs must be generated before the evaluated games occurred and realized player
panels must include every simulated player, including verified zero-production
outcomes. This module never reads a database; callers enforce chronology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_JOINT_SCORE_DRAWS = 250

def _thin_draws(samples: np.ndarray, max_draws: int = MAX_JOINT_SCORE_DRAWS) -> np.ndarray:
    if max_draws < 2:
        raise ValueError("max_draws must be at least two")
    if len(samples) <= max_draws:
        return samples
    return samples[np.linspace(0, len(samples) - 1, max_draws, dtype=int)]

def energy_score(samples: np.ndarray, observation: np.ndarray) -> float:
    """Empirical multivariate energy score; lower is better."""
    samples = _thin_draws(np.asarray(samples, dtype=float))
    observation = np.asarray(observation, dtype=float)
    if samples.ndim != 2 or observation.ndim != 1 or samples.shape[1] != len(observation):
        raise ValueError("samples must be draws by dimensions matching observation")
    if len(samples) < 2 or not np.isfinite(samples).all() or not np.isfinite(observation).all():
        raise ValueError("need at least two finite simulation draws")
    distance_to_truth = np.linalg.norm(samples - observation, axis=1).mean()
    pairwise = np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2).mean()
    return float(distance_to_truth - .5 * pairwise)

def variogram_score(samples: np.ndarray, observation: np.ndarray, p: float = .5) -> float:
    """Variogram score, sensitive to simulated dependence; lower is better."""
    samples = _thin_draws(np.asarray(samples, dtype=float))
    observation = np.asarray(observation, dtype=float)
    if p <= 0:
        raise ValueError("p must be positive")
    if samples.ndim != 2 or samples.shape[1] != len(observation):
        raise ValueError("samples must be draws by dimensions matching observation")
    if samples.shape[1] < 2:
        return 0.0
    observed = np.abs(observation[:, None] - observation[None, :]) ** p
    simulated = np.abs(samples[:, :, None] - samples[:, None, :]) ** p
    expected = simulated.mean(axis=0)
    upper = np.triu_indices(samples.shape[1], k=1)
    return float(np.mean((observed[upper] - expected[upper]) ** 2))

# Served predictions below this many points are treated as this many when
# scaling, so near-zero projections do not blow up the scaled score.
VARIOGRAM_SCALE_FLOOR = 1.0

def scaled_variogram_score(samples: np.ndarray, observation: np.ndarray, scale: np.ndarray,
                           p: float = .5) -> float:
    """Variogram score after dividing each dimension by a fixed scale.

    Raw fantasy-point variograms are dominated by the highest-variance pairs,
    which dilutes the dependence signal (measured on a known-dependence
    synthetic panel: z = -4.5 raw vs -6.7 scaled for the true model against
    independent draws). The scale must NOT come from the forecast being scored
    or propriety is lost; callers pass the served point prediction, which is
    identical for every simulation mode compared.
    """
    scale = np.maximum(np.asarray(scale, dtype=float), VARIOGRAM_SCALE_FLOOR)
    if scale.ndim != 1 or not np.isfinite(scale).all():
        raise ValueError("scale must be a finite vector")
    return variogram_score(np.asarray(samples, dtype=float) / scale,
                           np.asarray(observation, dtype=float) / scale, p=p)

def empirical_crps(samples: np.ndarray, observation: float) -> float:
    """Univariate ensemble CRPS, used for player-marginal evaluation."""
    values = np.asarray(samples, dtype=float).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(observation):
        raise ValueError("need at least two finite draws and a finite observation")
    # The direct pairwise matrix is O(n²) per player and makes a 1,000-draw
    # production panel impractical.  For sorted samples, the upper-triangle
    # sum of pairwise absolute differences is sum((2*i-n+1) * x[i]), giving
    # exactly the same ensemble CRPS in O(n log n).  No thinning: this
    # estimator is biased upward by E|X-X'|/(2n), so thinning to a fixed n
    # penalises wide distributions more than narrow ones and would tilt any
    # comparison between marginal families.
    ordered = np.sort(values)
    n = len(ordered)
    upper_pair_sum = np.dot(2 * np.arange(n) - n + 1, ordered)
    return float(np.abs(values - observation).mean() - upper_pair_sum / (n * n))

def _require_actual_coverage(simulated_ids: set[str], actual_frame: pd.DataFrame,
                             game_id: str) -> None:
    actual_ids = set(actual_frame["player_id"])
    missing = sorted(simulated_ids - actual_ids)
    if missing:
        raise ValueError(
            f"game {game_id} lacks actual outcomes for simulated players: {missing[:10]}")

def marginal_calibration(summary: pd.DataFrame, actuals: pd.DataFrame) -> dict:
    """Coverage and mean bias, with strict no-silent-drop actual alignment."""
    required_summary = {"game_id", "player_id", "mean", "p10", "p25", "p75", "p90"}
    required_actual = {"game_id", "player_id", "fantasy_points"}
    missing = (required_summary - set(summary)) | (required_actual - set(actuals))
    if missing:
        raise ValueError(f"missing calibration columns: {sorted(missing)}")
    frame = summary.merge(actuals[list(required_actual)], on=["game_id", "player_id"], how="left")
    if frame["fantasy_points"].isna().any():
        raise ValueError("marginal calibration has simulated players without actual outcomes")
    actual = frame["fantasy_points"].to_numpy(float)
    return {
        "n": int(len(frame)),
        "mean_bias": float((frame["mean"].to_numpy(float) - actual).mean()),
        "mae_of_mean": float(np.abs(frame["mean"].to_numpy(float) - actual).mean()),
        "coverage_50": float(((actual >= frame["p25"]) & (actual <= frame["p75"])).mean()),
        "coverage_80": float(((actual >= frame["p10"]) & (actual <= frame["p90"])).mean()),
    }

def evaluate_joint_player_draws(player_draws: pd.DataFrame,
                                actuals: pd.DataFrame, *, scale_column: str | None = None) -> dict:
    """Score each game as a joint player outcome, then average games equally.

    With ``scale_column`` (a per-player column of ``player_draws`` fixed across
    draws, e.g. the served prediction) each game also gets
    ``variogram_score_p05_scaled``.
    """
    required_draws = {"game_id", "draw", "player_id", "fantasy_points"}
    if scale_column is not None:
        required_draws.add(scale_column)
    required_actuals = {"game_id", "player_id", "fantasy_points"}
    missing = (required_draws - set(player_draws)) | (required_actuals - set(actuals))
    if missing:
        raise ValueError(f"missing joint-evaluation columns: {sorted(missing)}")
    scores, marginal_scores, pit_values = [], [], []
    for game_id, game_draws in player_draws.groupby("game_id", sort=True):
        actual_game = actuals[actuals["game_id"] == game_id].drop_duplicates("player_id")
        simulated_ids = set(game_draws["player_id"])
        _require_actual_coverage(simulated_ids, actual_game, str(game_id))
        actual_game = actual_game.set_index("player_id")
        players = sorted(simulated_ids)
        matrix = (game_draws.pivot(index="draw", columns="player_id", values="fantasy_points")
                  .reindex(columns=players))
        if matrix.isna().any().any() or len(matrix) < 2:
            raise ValueError(f"game {game_id} has incomplete player draw matrix")
        observation = actual_game.loc[players, "fantasy_points"].to_numpy(float)
        samples = matrix.to_numpy(float)
        entry = {
            "game_id": game_id, "dimensions": len(players),
            "energy_score": energy_score(samples, observation),
            "variogram_score_p05": variogram_score(samples, observation, p=.5),
        }
        if scale_column is not None:
            scale = game_draws.groupby("player_id")[scale_column].agg(["min", "max"]).reindex(players)
            if not np.allclose(scale["min"], scale["max"]):
                raise ValueError(f"game {game_id} has a {scale_column} that varies across draws")
            entry["variogram_score_p05_scaled"] = scaled_variogram_score(
                samples, observation, scale["min"].to_numpy(float), p=.5)
        scores.append(entry)
        for index, actual in enumerate(observation):
            values = samples[:, index]
            marginal_scores.append(empirical_crps(values, actual))
            pit_values.append(float(np.mean(values <= actual)))
    if not scores:
        raise ValueError("no games available for joint simulation evaluation")
    frame = pd.DataFrame(scores)
    return {
        "games_scored": int(len(frame)),
        "mean_energy_score": float(frame["energy_score"].mean()),
        "mean_variogram_score_p05": float(frame["variogram_score_p05"].mean()),
        **({"mean_variogram_score_p05_scaled": float(frame["variogram_score_p05_scaled"].mean())}
           if "variogram_score_p05_scaled" in frame else {}),
        "mean_marginal_crps": float(np.mean(marginal_scores)),
        "mean_pit": float(np.mean(pit_values)),
        "by_game": frame.to_dict(orient="records"),
    }

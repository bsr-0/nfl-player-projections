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

def empirical_crps(samples: np.ndarray, observation: float) -> float:
    """Univariate ensemble CRPS, used for player-marginal evaluation."""
    values = _thin_draws(np.asarray(samples, dtype=float).reshape(-1, 1)).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all() or not np.isfinite(observation):
        raise ValueError("need at least two finite draws and a finite observation")
    return float(np.abs(values - observation).mean()
                 - .5 * np.abs(values[:, None] - values[None, :]).mean())

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
                                actuals: pd.DataFrame) -> dict:
    """Score each game as a joint player outcome, then average games equally."""
    required_draws = {"game_id", "draw", "player_id", "fantasy_points"}
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
        scores.append({
            "game_id": game_id, "dimensions": len(players),
            "energy_score": energy_score(samples, observation),
            "variogram_score_p05": variogram_score(samples, observation, p=.5),
        })
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
        "mean_marginal_crps": float(np.mean(marginal_scores)),
        "mean_pit": float(np.mean(pit_values)),
        "by_game": frame.to_dict(orient="records"),
    }

def game_draw_calibration(game_draws: pd.DataFrame,
                          actuals: pd.DataFrame) -> dict:
    """Evaluate simulated win/total/margin against realized game outcomes."""
    required_draws = {"game_id", "draw", "home_won", "simulated_total", "simulated_margin"}
    required_actuals = {"game_id", "home_score", "away_score"}
    missing = (required_draws - set(game_draws)) | (required_actuals - set(actuals))
    if missing:
        raise ValueError(f"missing game-calibration columns: {sorted(missing)}")
    summaries = (game_draws.groupby("game_id").agg(
        home_win_prob=("home_won", "mean"),
        total_mean=("simulated_total", "mean"),
        margin_mean=("simulated_margin", "mean"),
    ).reset_index())
    actual = actuals.copy()
    actual["home_won_actual"] = (actual["home_score"] > actual["away_score"]).astype(float)
    actual["total_actual"] = actual["home_score"] + actual["away_score"]
    actual["margin_actual"] = actual["home_score"] - actual["away_score"]
    frame = summaries.merge(actual[["game_id", "home_won_actual", "total_actual", "margin_actual"]],
                            on="game_id", how="left")
    if frame[["home_won_actual", "total_actual", "margin_actual"]].isna().any().any():
        raise ValueError("game calibration has simulated games without actual results")
    return {
        "n": int(len(frame)),
        "home_win_brier": float(np.mean((frame["home_win_prob"] - frame["home_won_actual"]) ** 2)),
        "total_mae": float(np.mean(np.abs(frame["total_mean"] - frame["total_actual"]))),
        "margin_mae": float(np.mean(np.abs(frame["margin_mean"] - frame["margin_actual"]))),
    }

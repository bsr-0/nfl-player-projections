"""Versioned output contract for calibrated player simulations (game-sim-v2).

Draws are held as one (draws x players) matrix per game, columns in the order
of that game's rows in ``players``; long tables are built only for Parquet.
Summaries are always recomputed from the draws during validation, so a stored
summary can never disagree with the draws it describes.

v2 replaces v1 (the game-script simulator's score/plays/pass-attempt draws,
now deleted). There are no simulated scores: ``served_*`` game fields are the
game models' predictions passed through unchanged, and the game-level
distributions are sums of the players' fantasy points.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from src.models.calibrated_simulation import sum_groups

SIMULATION_SCHEMA_VERSION = "game-sim-v2"
POSITIONS = ("QB", "RB", "WR", "TE")
GAME_COLUMNS = ("game_id", "season", "week", "home_team", "away_team",
                "served_home_win_prob", "served_margin", "served_total")
PLAYER_COLUMNS = ("game_id", "player_id", "team", "position", "served_prediction",
                  "role_key", "is_cold_start")


@dataclass
class SimulationPayload:
    seed: int
    model_config: dict
    games: pd.DataFrame
    players: pd.DataFrame
    draws: dict[str, np.ndarray] = field(repr=False)

    @property
    def n_draws(self) -> int:
        return int(next(iter(self.draws.values())).shape[0])

    def game_players(self, game_id: str) -> pd.DataFrame:
        return self.players.loc[self.players["game_id"] == game_id]


def _nonempty_strings(values: pd.Series, label: str) -> None:
    if not values.map(lambda v: isinstance(v, str) and bool(v)).all():
        raise ValueError(f"{label} must be non-empty strings")


def validate_payload(payload: SimulationPayload) -> None:
    games, players = payload.games, payload.players
    if isinstance(payload.seed, bool) or int(payload.seed) != payload.seed:
        raise ValueError("seed must be an integer")
    if not isinstance(payload.model_config, dict):
        raise ValueError("model_config must be a dict")
    for frame, columns, label in ((games, GAME_COLUMNS, "games"), (players, PLAYER_COLUMNS, "players")):
        if missing := set(columns) - set(frame.columns):
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
    if games.empty or players.empty:
        raise ValueError("payload needs at least one game and one player")
    _nonempty_strings(games["game_id"], "game_id")
    if games["game_id"].duplicated().any():
        raise ValueError("duplicate game_id")
    if (games["home_team"] == games["away_team"]).any():
        raise ValueError("a game cannot have the same home and away team")
    for column in ("served_home_win_prob", "served_margin", "served_total"):
        present = games[column].dropna()
        if not np.isfinite(present.to_numpy(float)).all():
            raise ValueError(f"{column} must be finite when present")
    _nonempty_strings(players["player_id"], "player_id")
    _nonempty_strings(players["team"], "team")
    if not players["position"].isin(POSITIONS).all():
        raise ValueError(f"position must be one of {POSITIONS}")
    if players.duplicated(["game_id", "player_id"]).any():
        raise ValueError("duplicate player within a game")
    if not np.isfinite(players["served_prediction"].to_numpy(float)).all():
        raise ValueError("served_prediction must be finite")
    sides = players.merge(games[["game_id", "home_team", "away_team"]], on="game_id", how="left",
                          validate="many_to_one")
    if sides["home_team"].isna().any():
        raise ValueError("player references an absent game")
    if not ((sides["team"] == sides["home_team"]) | (sides["team"] == sides["away_team"])).all():
        raise ValueError("player team is not playing in its game")
    if set(payload.draws) != set(players["game_id"]):
        raise ValueError("draws must cover exactly the games that have players")
    n_draws = None
    for game_id, matrix in payload.draws.items():
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] != len(payload.game_players(game_id)):
            raise ValueError(f"draw matrix for {game_id} does not match its players")
        if not np.isfinite(matrix).all():
            raise ValueError(f"draw matrix for {game_id} is not finite")
        n_draws = matrix.shape[0] if n_draws is None else n_draws
        if matrix.shape[0] != n_draws or n_draws < 2:
            raise ValueError("every game needs the same number (>= 2) of draws")


def summarize_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "draw_count": int(len(values)), "mean": float(values.mean()),
        "median": float(np.quantile(values, .50)),
        "p10": float(np.quantile(values, .10)), "p25": float(np.quantile(values, .25)),
        "p75": float(np.quantile(values, .75)), "p90": float(np.quantile(values, .90)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "max": float(values.max()),
    }


def player_summary(payload: SimulationPayload) -> list[dict]:
    validate_payload(payload)
    output = []
    for game_id in sorted(payload.draws):
        rows = payload.game_players(game_id)
        matrix = payload.draws[game_id]
        for column, row in enumerate(rows.itertuples(index=False)):
            output.append({"game_id": game_id, "player_id": row.player_id, "team": row.team,
                           "position": row.position, "served_prediction": float(row.served_prediction),
                           **summarize_values(matrix[:, column])})
    return sorted(output, key=lambda entry: (entry["game_id"], entry["player_id"]))


def _optional(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def game_summary(payload: SimulationPayload) -> list[dict]:
    validate_payload(payload)
    output = []
    for game in payload.games.sort_values("game_id").itertuples(index=False):
        entry = {"game_id": game.game_id, "season": int(game.season), "week": int(game.week),
                 "home_team": game.home_team, "away_team": game.away_team,
                 "served_home_win_prob": _optional(game.served_home_win_prob),
                 "served_margin": _optional(game.served_margin),
                 "served_total": _optional(game.served_total), "fantasy_sums": []}
        if game.game_id in payload.draws:
            rows = payload.game_players(game.game_id).assign(
                home_team=game.home_team, away_team=game.away_team,
                predicted_points=lambda f: f["served_prediction"])
            matrix = payload.draws[game.game_id]
            for kind, side, members in sum_groups(rows.reset_index(drop=True)):
                totals = matrix[:, members].sum(axis=1)
                entry["fantasy_sums"].append({
                    "kind": kind, "side": side,
                    "player_ids": rows["player_id"].iloc[members].tolist(),
                    "mean": float(totals.mean()), "p10": float(np.quantile(totals, .10)),
                    "p50": float(np.quantile(totals, .50)), "p90": float(np.quantile(totals, .90)),
                })
        output.append(entry)
    return output


def draws_frame(payload: SimulationPayload) -> pd.DataFrame:
    """Long (game_id, draw, player_id, fantasy_points) table for local analysis."""
    validate_payload(payload)
    parts = []
    for game_id in sorted(payload.draws):
        matrix = payload.draws[game_id]
        ids = payload.game_players(game_id)["player_id"].to_numpy()
        n_draws, n_players = matrix.shape
        parts.append(pd.DataFrame({
            "game_id": game_id, "draw": np.tile(np.arange(n_draws), n_players),
            "player_id": np.repeat(ids, n_draws), "fantasy_points": matrix.T.reshape(-1)}))
    return pd.concat(parts, ignore_index=True)

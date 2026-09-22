"""Adapters from current serving DataFrames into game-simulation contracts.

This module intentionally does not read SQLite or model artifacts.  Call it
immediately after the existing game and player serving paths have produced
their DataFrames.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import pandas as pd

from src.models.game_simulation import (
    GameScriptInput,
    PlayerSimulationInput,
    TeamVolumeBaseline,
)

_Z_80 = 1.281551565545

def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

def _game_key(row: pd.Series) -> str:
    return f"{int(row.season)}_{int(row.week)}_{row.home_team}_{row.away_team}"

def game_inputs_from_predictions(
    game_predictions: pd.DataFrame,
    *,
    model: str = "ridge",
    team_volume: Mapping[str, TeamVolumeBaseline] | None = None,
) -> dict[str, GameScriptInput]:
    """Adapt build_prediction_rows-style game predictions.

    The selected model must have home_win_prob_MODEL, predicted_margin_MODEL,
    and predicted_total_MODEL columns.  Margin follows the existing convention:
    positive means the home team is projected to win by that many points.
    """
    required = {"season", "week", "home_team", "away_team",
                f"home_win_prob_{model}", f"predicted_margin_{model}",
                f"predicted_total_{model}"}
    _require(game_predictions, required, "game_predictions")
    volume = team_volume or {}
    out = {}
    for row in game_predictions.itertuples(index=False):
        key = _game_key(row)
        if key in out:
            raise ValueError(f"duplicate game prediction: {key}")
        home_prob = float(getattr(row, f"home_win_prob_{model}"))
        margin = float(getattr(row, f"predicted_margin_{model}"))
        total = float(getattr(row, f"predicted_total_{model}"))
        if not all(math.isfinite(x) for x in (home_prob, margin, total)):
            raise ValueError(f"non-finite prediction for {key}")
        out[key] = GameScriptInput(
            game_id=key, home_team=str(row.home_team), away_team=str(row.away_team),
            home_win_prob=home_prob, predicted_margin=margin, predicted_total=total,
            home=volume.get(str(row.home_team), TeamVolumeBaseline()),
            away=volume.get(str(row.away_team), TeamVolumeBaseline()),
        )
    return out

def player_inputs_from_predictions(
    player_predictions: pd.DataFrame,
    games: Mapping[str, GameScriptInput],
) -> dict[str, list[PlayerSimulationInput]]:
    """Adapt NFLPredictor weekly rows to their matching scheduled game.

    Intervals supply a per-player normal-scale uncertainty estimate.  If the
    interval is unavailable, the conservative position-agnostic 8-point
    fallback is retained. Participation stays 1.0 until the availability model
    is explicitly connected.
    """
    _require(player_predictions, {"player_id", "team", "opponent", "position", "predicted_points"},
             "player_predictions")
    by_pair = {
        frozenset((game.home_team, game.away_team)): game_id
        for game_id, game in games.items()
    }
    out = {game_id: [] for game_id in games}
    for row in player_predictions.itertuples(index=False):
        team, opponent = str(row.team), str(row.opponent)
        game_id = by_pair.get(frozenset((team, opponent)))
        if game_id is None:
            continue  # bye, stale roster team, or a game outside supplied predictions
        position = str(row.position).upper()
        if position not in {"QB", "RB", "WR", "TE"}:
            continue
        mean = float(row.predicted_points)
        if not math.isfinite(mean):
            continue
        sd = 8.0
        if hasattr(row, "prediction_ci80_lower") and hasattr(row, "prediction_ci80_upper"):
            lower, upper = float(row.prediction_ci80_lower), float(row.prediction_ci80_upper)
            if math.isfinite(lower) and math.isfinite(upper) and upper >= lower:
                sd = max(0.01, (upper - lower) / (2 * _Z_80))
        out[game_id].append(PlayerSimulationInput(
            player_id=str(row.player_id), team=team, position=position,
            mean_fantasy_points=max(0.0, mean), sd_fantasy_points=sd,
        ))
    return {game_id: players for game_id, players in out.items() if players}

def simulation_inputs_from_predictions(
    game_predictions: pd.DataFrame, player_predictions: pd.DataFrame, **kwargs
) -> dict[str, tuple[GameScriptInput, list[PlayerSimulationInput]]]:
    """Create complete, simulation-ready inputs keyed by game ID."""
    games = game_inputs_from_predictions(game_predictions, **kwargs)
    players = player_inputs_from_predictions(player_predictions, games)
    return {game_id: (game, players[game_id]) for game_id, game in games.items()
            if game_id in players}

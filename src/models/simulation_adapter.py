"""Adapters from current serving DataFrames into game-simulation contracts.

This module intentionally does not read SQLite or model artifacts. Call it
immediately after the existing game and player serving paths have produced
their DataFrames.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import pandas as pd

from src.models.game_simulation import GameScriptInput, PlayerSimulationInput, TeamVolumeBaseline

_Z_80 = 1.281551565545

def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

def _game_key(row) -> str:
    return f"{int(row.season)}_{int(row.week)}_{row.home_team}_{row.away_team}"

def game_inputs_from_predictions(
    game_predictions: pd.DataFrame,
    *,
    outcome_model: str = "logistic",
    margin_total_model: str = "ridge",
    team_volume: Mapping[str, TeamVolumeBaseline] | None = None,
) -> dict[str, GameScriptInput]:
    """Adapt build_prediction_rows-style game predictions.

    The current serving path has different classifier and regressor families:
    home_win_prob_LOGISTIC (or xgb/rf) and predicted_margin_RIDGE /
    predicted_total_RIDGE (or xgb/rf). Positive margin means the home team is
    projected to win by that many points.
    """
    probability = f"home_win_prob_{outcome_model}"
    margin = f"predicted_margin_{margin_total_model}"
    total = f"predicted_total_{margin_total_model}"
    _require(game_predictions, {"season", "week", "home_team", "away_team",
                                probability, margin, total}, "game_predictions")
    volume, out = team_volume or {}, {}
    for row in game_predictions.itertuples(index=False):
        key = _game_key(row)
        if key in out:
            raise ValueError(f"duplicate game prediction: {key}")
        values = (float(getattr(row, probability)), float(getattr(row, margin)), float(getattr(row, total)))
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite prediction for {key}")
        home_prob, expected_margin, expected_total = values
        out[key] = GameScriptInput(
            game_id=key, home_team=str(row.home_team), away_team=str(row.away_team),
            home_win_prob=home_prob, predicted_margin=expected_margin, predicted_total=expected_total,
            home=volume.get(str(row.home_team), TeamVolumeBaseline()),
            away=volume.get(str(row.away_team), TeamVolumeBaseline()),
        )
    return out

def _optional_unit_interval(row, candidates: tuple[str, ...]) -> float | None:
    for field in candidates:
        if not hasattr(row, field):
            continue
        value = getattr(row, field)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            return value
    return None

def player_inputs_from_predictions(
    player_predictions: pd.DataFrame,
    games: Mapping[str, GameScriptInput],
) -> dict[str, list[PlayerSimulationInput]]:
    """Adapt NFLPredictor weekly rows to their matching scheduled game.

    Intervals supply a per-player normal-scale uncertainty estimate. If an
    interval is unavailable, the conservative 8-point fallback is retained.
    Participation stays 1.0 until the availability model is connected.
    """
    _require(player_predictions, {"player_id", "team", "opponent", "position", "predicted_points"},
             "player_predictions")
    by_pair = {frozenset((game.home_team, game.away_team)): game_id
               for game_id, game in games.items()}
    out = {game_id: [] for game_id in games}
    for row in player_predictions.itertuples(index=False):
        team, opponent = str(row.team), str(row.opponent)
        game_id = by_pair.get(frozenset((team, opponent)))
        if game_id is None:
            continue
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
        share_fields = (
            ("rush_share", "usage_share") if position == "RB"
            else ("pass_share", "usage_share") if position == "QB"
            else ("target_share", "usage_share")
        )
        usage_share = _optional_unit_interval(row, share_fields)
        participation_prob = _optional_unit_interval(
            row, ("participation_prob", "will_play_probability")) or 1.0
        out[game_id].append(PlayerSimulationInput(
            player_id=str(row.player_id), team=team, position=position,
            mean_fantasy_points=max(0.0, mean), sd_fantasy_points=sd,
            usage_share=usage_share, participation_prob=participation_prob,
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

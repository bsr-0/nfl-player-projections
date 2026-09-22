"""Versioned output contract for game/player Monte Carlo simulations.

The contract is deliberately plain dictionaries so it can serialize to JSON or
Parquet without adding a runtime schema dependency.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import numpy as np

SIMULATION_SCHEMA_VERSION = "game-sim-v1"
_REQUIRED_GAME = {"game_id", "draw", "home_score", "away_score", "home_plays",
                  "away_plays", "home_pass_attempts", "away_pass_attempts"}
_REQUIRED_PLAYER = {"game_id", "draw", "player_id", "team", "position", "active",
                    "fantasy_points"}

def _finite(value, field, row):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value

def _validate_required(row: Mapping, required: set[str], label: str):
    missing = required - set(row)
    if missing:
        raise ValueError(f"{label} missing required fields: {sorted(missing)}")

def validate_game_draws(rows: Iterable[Mapping]) -> list[dict]:
    """Validate and normalize game-level rows."""
    normalized = []
    for row in rows:
        _validate_required(row, _REQUIRED_GAME, "game draw")
        item = dict(row)
        for field in ("home_score", "away_score", "home_plays", "away_plays",
                      "home_pass_attempts", "away_pass_attempts"):
            item[field] = _finite(item[field], field, row)
        if not isinstance(item["game_id"], str) or not item["game_id"]:
            raise ValueError("game_id must be a non-empty string")
        if int(item["draw"]) != item["draw"] or item["draw"] < 0:
            raise ValueError("draw must be a nonnegative integer")
        for field in ("home_score", "away_score", "home_plays", "away_plays",
                      "home_pass_attempts", "away_pass_attempts"):
            if item[field] < 0:
                raise ValueError(f"{field} must be nonnegative")
        if item["home_pass_attempts"] > item["home_plays"] or item["away_pass_attempts"] > item["away_plays"]:
            raise ValueError("pass attempts cannot exceed plays")
        item["draw"] = int(item["draw"])
        normalized.append(item)
    return normalized

def validate_player_draws(rows: Iterable[Mapping]) -> list[dict]:
    """Validate and normalize player-level rows."""
    normalized = []
    for row in rows:
        _validate_required(row, _REQUIRED_PLAYER, "player draw")
        item = dict(row)
        for field in ("fantasy_points",):
            item[field] = _finite(item[field], field, row)
        if item["fantasy_points"] < 0:
            raise ValueError("fantasy_points must be nonnegative")
        if not isinstance(item["game_id"], str) or not item["game_id"]:
            raise ValueError("game_id must be a non-empty string")
        if not isinstance(item["player_id"], str) or not item["player_id"]:
            raise ValueError("player_id must be a non-empty string")
        if not isinstance(item["team"], str) or not item["team"]:
            raise ValueError("team must be a non-empty string")
        if str(item["position"]).upper() not in {"QB", "RB", "WR", "TE"}:
            raise ValueError("position must be QB, RB, WR, or TE")
        if not isinstance(item["active"], (bool, np.bool_)):
            raise ValueError("active must be boolean")
        if int(item["draw"]) != item["draw"] or item["draw"] < 0:
            raise ValueError("draw must be a nonnegative integer")
        item["draw"] = int(item["draw"])
        item["position"] = str(item["position"]).upper()
        item["active"] = bool(item["active"])
        normalized.append(item)
    return normalized

def summarize_player_draws(rows: Iterable[Mapping]) -> list[dict]:
    """Aggregate player draws into UI/DFS-ready distribution summaries."""
    draws = validate_player_draws(rows)
    groups = {}
    for row in draws:
        key = (row["game_id"], row["player_id"], row["team"], row["position"])
        groups.setdefault(key, []).append(row)
    output = []
    for (game_id, player_id, team, position), group in sorted(groups.items()):
        points = np.asarray([row["fantasy_points"] for row in group], dtype=float)
        active = np.asarray([row["active"] for row in group], dtype=float)
        output.append({
            "game_id": game_id, "player_id": player_id, "team": team, "position": position,
            "draw_count": int(len(points)),
            "mean": float(points.mean()), "median": float(np.quantile(points, .50)),
            "p10": float(np.quantile(points, .10)), "p25": float(np.quantile(points, .25)),
            "p75": float(np.quantile(points, .75)), "p90": float(np.quantile(points, .90)),
            "std": float(points.std(ddof=1)) if len(points) > 1 else 0.0,
            "prob_zero": float(np.mean(points == 0.0)),
            "prob_active": float(active.mean()),
            "max": float(points.max()),
        })
    return output

def build_simulation_payload(
    game_draws: Iterable[Mapping],
    player_draws: Iterable[Mapping],
    *,
    seed: int,
    model_config: Mapping | None = None,
) -> dict:
    """Build the complete versioned payload used by JSON/Parquet adapters."""
    games = validate_game_draws(game_draws)
    players = validate_player_draws(player_draws)
    game_ids = sorted({row["game_id"] for row in games})
    player_game_ids = {row["game_id"] for row in players}
    if not player_game_ids <= set(game_ids):
        raise ValueError("player draw references a game absent from game draws")
    draws_by_game = {}
    for row in games:
        draws_by_game.setdefault(row["game_id"], set()).add(row["draw"])
    for row in players:
        if row["draw"] not in draws_by_game[row["game_id"]]:
            raise ValueError("player draw references an absent game draw")
    return {
        "schema_version": SIMULATION_SCHEMA_VERSION,
        "seed": int(seed),
        "game_count": len(game_ids),
        "game_ids": game_ids,
        "model_config": dict(model_config or {}),
        "game_draws": games,
        "player_draws": players,
        "player_summary": summarize_player_draws(players),
    }

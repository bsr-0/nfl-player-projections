"""Serialization for game-sim-v2 payloads.

The site JSON carries summaries only; raw Monte Carlo draws never go to
docs/data (a 1,000-draw slate is hundreds of thousands of player rows).
Parquet tables are for local analysis and scoring against actuals.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from src.models.simulation_schema import (
    SIMULATION_SCHEMA_VERSION,
    SimulationPayload,
    draws_frame,
    game_summary,
    player_summary,
    validate_payload,
)
from src.utils.atomic_io import atomic_write_json

SITE_SCHEMA_VERSION = "game-sim-site-v2"


def build_site_payload(payload: SimulationPayload) -> dict:
    validate_payload(payload)
    games = sorted(payload.games["game_id"])
    return {
        "schema_version": SITE_SCHEMA_VERSION, "source_schema_version": SIMULATION_SCHEMA_VERSION,
        "seed": int(payload.seed), "draws_per_player": payload.n_draws,
        "game_count": len(games), "game_ids": games, "model_config": payload.model_config,
        "game_summary": game_summary(payload), "player_summary": player_summary(payload),
    }


def write_simulation_site_payload(payload: SimulationPayload, *, json_path: str | Path) -> Path:
    """Atomically write compact site summaries, never raw simulation draws."""
    site = build_site_payload(payload)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(site, json_path)
    return json_path


def write_simulation_parquet(payload: SimulationPayload, *, parquet_dir: str | Path,
                             stem: str) -> list[Path]:
    """Write games, players and long draws; all files appear together or not at all."""
    validate_payload(payload)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".simulation-", dir=parquet_dir))
    staged = []
    try:
        for name, frame in (("games", payload.games), ("players", payload.players),
                            ("player_draws", draws_frame(payload))):
            destination = parquet_dir / f"{stem}_{name}.parquet"
            temporary = temporary_dir / destination.name
            frame.to_parquet(temporary, index=False)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return [destination for _, destination in staged]

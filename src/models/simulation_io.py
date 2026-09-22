"""Serialization helpers for versioned simulation payloads."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.models.simulation_schema import build_simulation_payload

def write_simulation_payload(
    payload: Mapping,
    *,
    json_path: str | Path,
    parquet_dir: str | Path | None = None,
) -> list[Path]:
    """Write one validated JSON payload and optional normalized Parquet tables.

    JSON is the site-facing artifact. Parquet is an analysis artifact and is
    opt-in because pandas requires an installed Parquet engine.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    written = [json_path]
    if parquet_dir is None:
        return written
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    try:
        tables = {
            "game_draws": pd.DataFrame(payload["game_draws"]),
            "player_draws": pd.DataFrame(payload["player_draws"]),
            "player_summary": pd.DataFrame(payload["player_summary"]),
        }
        for name, frame in tables.items():
            path = parquet_dir / f"{json_path.stem}_{name}.parquet"
            frame.to_parquet(path, index=False)
            written.append(path)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requested but no Parquet engine is installed; "
            "install pyarrow or omit parquet_dir"
        ) from exc
    return written

def build_and_write_simulation_payload(
    game_draws,
    player_draws,
    *,
    seed: int,
    json_path: str | Path,
    model_config: Mapping | None = None,
    parquet_dir: str | Path | None = None,
) -> list[Path]:
    payload = build_simulation_payload(
        game_draws, player_draws, seed=seed, model_config=model_config)
    return write_simulation_payload(payload, json_path=json_path, parquet_dir=parquet_dir)

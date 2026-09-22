"""Serialization helpers for versioned simulation payloads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import pandas as pd

from src.models.simulation_schema import (
    build_simulation_payload,
    validate_simulation_payload,
)

def _atomic_json_write(payload: Mapping, path: Path) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def write_simulation_payload(
    payload: Mapping,
    *,
    json_path: str | Path,
    parquet_dir: str | Path | None = None,
) -> list[Path]:
    """Validate, then atomically write JSON and optional Parquet tables."""
    payload = validate_simulation_payload(payload)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = []
    if parquet_dir is not None:
        parquet_dir = Path(parquet_dir)
        parquet_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=".simulation-", dir=parquet_dir))
        try:
            tables = {
                "game_draws": pd.DataFrame(payload["game_draws"]),
                "player_draws": pd.DataFrame(payload["player_draws"]),
                "player_summary": pd.DataFrame(payload["player_summary"]),
            }
            for name, frame in tables.items():
                destination = parquet_dir / f"{json_path.stem}_{name}.parquet"
                temporary = temporary_dir / destination.name
                try:
                    frame.to_parquet(temporary, index=False)
                except (ImportError, ValueError) as exc:
                    raise RuntimeError(
                        "Parquet output requested but no compatible Parquet engine "
                        "is installed; install pyarrow or omit parquet_dir"
                    ) from exc
                parquet_files.append((temporary, destination))
            for temporary, destination in parquet_files:
                os.replace(temporary, destination)
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)

    _atomic_json_write(payload, json_path)
    return [json_path] + [destination for _, destination in parquet_files]

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
    return write_simulation_payload(
        payload, json_path=json_path, parquet_dir=parquet_dir)

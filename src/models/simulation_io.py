"""Serialization helpers for versioned simulation payloads."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np
import pandas as pd

from src.models.simulation_schema import build_simulation_payload, validate_simulation_payload
from src.utils.atomic_io import atomic_write_json

SITE_SCHEMA_VERSION = "game-sim-site-v1"

# Kept as a module-local name because callers below (and their tests) refer
# to it; the implementation now lives in src/utils/atomic_io.py, shared with
# the three other copies of this pattern that had grown up elsewhere.
def _atomic_json_write(payload: Mapping, path: Path) -> None:
    atomic_write_json(payload, path)

def _game_summary(game_draws: list[Mapping]) -> list[dict]:
    grouped = {}
    for row in game_draws:
        grouped.setdefault(row["game_id"], []).append(row)
    summary = []
    for game_id, rows in sorted(grouped.items()):
        totals = np.asarray([row["simulated_total"] for row in rows], dtype=float)
        margins = np.asarray([row["simulated_margin"] for row in rows], dtype=float)
        wins = np.asarray([row["home_won"] for row in rows], dtype=float)
        summary.append({
            "game_id": game_id, "draw_count": len(rows),
            "home_win_prob_simulated": float(wins.mean()),
            "total_mean": float(totals.mean()),
            "total_p10": float(np.quantile(totals, .10)),
            "total_p50": float(np.quantile(totals, .50)),
            "total_p90": float(np.quantile(totals, .90)),
            "home_margin_mean": float(margins.mean()),
            "home_margin_p10": float(np.quantile(margins, .10)),
            "home_margin_p50": float(np.quantile(margins, .50)),
            "home_margin_p90": float(np.quantile(margins, .90)),
        })
    return summary

def build_site_payload(payload: Mapping) -> dict:
    """Build the compact artifact served from docs/data.

    Raw Monte Carlo draws deliberately stay out of GitHub Pages output; the
    site consumes game and player summaries only.
    """
    payload = validate_simulation_payload(payload)
    required_game_fields = {"simulated_total", "simulated_margin", "home_won"}
    if not all(required_game_fields <= set(row) for row in payload["game_draws"]):
        raise ValueError("game draws need simulated_total, simulated_margin, and home_won for site output")
    return {
        "schema_version": SITE_SCHEMA_VERSION,
        "source_schema_version": payload["schema_version"],
        "seed": payload["seed"], "game_count": payload["game_count"],
        "game_ids": payload["game_ids"], "model_config": payload["model_config"],
        "game_summary": _game_summary(payload["game_draws"]),
        "player_summary": payload["player_summary"],
    }

def write_simulation_payload(payload: Mapping, *, json_path: str | Path) -> list[Path]:
    """Validate and atomically write the complete research payload."""
    payload = validate_simulation_payload(payload)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(payload, json_path)
    return [json_path]

def write_simulation_parquet(payload: Mapping, *, parquet_dir: str | Path,
                             stem: str) -> list[Path]:
    """Write raw draw tables for local analysis; never target docs/data."""
    payload = validate_simulation_payload(payload)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".simulation-", dir=parquet_dir))
    parquet_files = []
    try:
        tables = {
            "game_draws": pd.DataFrame(payload["game_draws"]),
            "player_draws": pd.DataFrame(payload["player_draws"]),
            "player_summary": pd.DataFrame(payload["player_summary"]),
        }
        for name, frame in tables.items():
            destination = parquet_dir / f"{stem}_{name}.parquet"
            temporary = temporary_dir / destination.name
            try:
                frame.to_parquet(temporary, index=False)
            except (ImportError, ValueError) as exc:
                raise RuntimeError(
                    "Parquet output requested but no compatible Parquet engine "
                    "is installed; install pyarrow or omit Parquet output"
                ) from exc
            parquet_files.append((temporary, destination))
        for temporary, destination in parquet_files:
            os.replace(temporary, destination)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return [destination for _, destination in parquet_files]

def write_simulation_site_payload(payload: Mapping, *, json_path: str | Path) -> list[Path]:
    """Atomically write compact site summaries, never raw simulation draws."""
    site_payload = build_site_payload(payload)
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(site_payload, json_path)
    return [json_path]

def build_and_write_simulation_payload(
    game_draws, player_draws, *, seed: int, json_path: str | Path,
    model_config: Mapping | None = None,
) -> list[Path]:
    payload = build_simulation_payload(
        game_draws, player_draws, seed=seed, model_config=model_config)
    return write_simulation_payload(payload, json_path=json_path)

def build_and_write_site_simulation_payload(
    game_draws, player_draws, *, seed: int, site_json_path: str | Path,
    model_config: Mapping | None = None, parquet_dir: str | Path | None = None,
    parquet_stem: str | None = None,
) -> list[Path]:
    payload = build_simulation_payload(
        game_draws, player_draws, seed=seed, model_config=model_config)
    written = write_simulation_site_payload(payload, json_path=site_json_path)
    if parquet_dir is not None:
        written.extend(write_simulation_parquet(
            payload, parquet_dir=parquet_dir,
            stem=parquet_stem or Path(site_json_path).stem))
    return written

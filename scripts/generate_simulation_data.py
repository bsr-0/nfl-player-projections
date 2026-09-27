#!/usr/bin/env python3
"""Generate calibrated-copula simulation summaries from the serving JSON files.

Downstream of generate_weekly_data.py and generate_game_predictions_data.py; it
does not access SQLite or retrain anything. Each player's draws are the served
prediction plus a calibrated marginal deviation, reordered into game-level
dependence by the role-factor copula -- exactly the backtest's
``calibrated_role_factor`` arm (src/models/calibrated_simulation.py). Artifacts
come from scripts/fit_simulation_artifacts.py; without them this fails rather
than falling back to anything.

Weeks at or before the artifacts' fitted-through week are refused: their own
outcomes shaped the calibration.

Usage:
    python scripts/generate_simulation_data.py --season 2026
    python scripts/generate_simulation_data.py --season 2026 --week 2 --draws 2000
    python scripts/generate_simulation_data.py --season 2026 --write-parquet
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.calibrated_simulation import (  # noqa: E402
    SCOPE_CAVEAT,
    SimulationArtifacts,
    attach_roles,
    load_simulation_artifacts,
    simulate_game,
)
from src.models.simulation_io import write_simulation_parquet, write_simulation_site_payload  # noqa: E402
from src.models.simulation_schema import POSITIONS, SimulationPayload  # noqa: E402

DOCS_DATA = PROJECT_ROOT / "docs" / "data"
PARQUET_DATA = PROJECT_ROOT / "data" / "simulations"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "data" / "models" / "simulation"
# Served game-model heads passed through (never simulated), as before.
SERVED_GAME_FIELDS = {"served_home_win_prob": "home_win_prob_logistic",
                      "served_margin": "predicted_margin_ridge",
                      "served_total": "predicted_total_ridge"}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.DataFrame(json.loads(path.read_text()))


def _weeks(season: int, requested: int | None) -> list[int]:
    if requested is not None:
        return [requested]
    values = []
    for path in DOCS_DATA.glob(f"game_predictions_{season}_wk*.json"):
        try:
            values.append(int(path.stem.rsplit("wk", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))


def build_week_inputs(games: pd.DataFrame, players: pd.DataFrame, *, season: int, week: int,
                      known_player_ids: frozenset) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Serving rows -> (games, simulation rows, exclusion report). Outcomes are dropped first."""
    if missing := {"home_team", "away_team"} - set(games.columns):
        raise ValueError(f"game predictions lack {sorted(missing)}")
    games = games.copy()
    games["home_team"], games["away_team"] = games["home_team"].astype(str), games["away_team"].astype(str)
    teams = pd.concat([games["home_team"], games["away_team"]])
    if teams.duplicated().any():
        raise ValueError(f"teams scheduled twice in {season} week {week}: {sorted(teams[teams.duplicated()])}")
    games["game_id"] = [f"{season}_{week:02d}_{a}_{h}" for h, a in zip(games["home_team"], games["away_team"])]
    games["season"], games["week"] = season, week
    for served, source in SERVED_GAME_FIELDS.items():
        games[served] = pd.to_numeric(games[source], errors="coerce") if source in games else np.nan
    game_by_team = {}
    for game in games.itertuples(index=False):
        game_by_team[game.home_team] = (game.game_id, game.home_team, game.away_team, game.away_team)
        game_by_team[game.away_team] = (game.game_id, game.home_team, game.away_team, game.home_team)

    outcome_columns = [c for c in players.columns if c == "actual_points" or str(c).startswith("actual_")]
    players = players.drop(columns=outcome_columns)
    if missing := {"player_id", "team", "opponent", "position", "predicted_points"} - set(players.columns):
        raise ValueError(f"weekly predictions lack {sorted(missing)}")
    players = players.assign(player_id=players["player_id"].astype(str), team=players["team"].astype(str),
                             opponent=players["opponent"].astype(str),
                             position=players["position"].astype(str).str.upper())
    if players["player_id"].duplicated().any():
        raise ValueError("weekly predictions contain duplicate player_ids")
    report = {"offered": int(len(players)), "dropped_outcome_columns": outcome_columns}
    supported = players["position"].isin(POSITIONS)
    report["excluded_unsupported_position"] = int((~supported).sum())
    players = players.loc[supported]
    scheduled = players["team"].isin(game_by_team)
    report["excluded_team_not_scheduled"] = int((~scheduled).sum())
    players = players.loc[scheduled].copy()
    info = players["team"].map(game_by_team)
    players["game_id"] = info.map(lambda v: v[0])
    players["home_team"] = info.map(lambda v: v[1])
    players["away_team"] = info.map(lambda v: v[2])
    wrong_opponent = players["opponent"] != info.map(lambda v: v[3])
    if wrong_opponent.any():
        raise ValueError(f"{int(wrong_opponent.sum())} players' opponent disagrees with the schedule")
    predicted = pd.to_numeric(players["predicted_points"], errors="coerce")
    if predicted.isna().any() or not np.isfinite(predicted.to_numpy(float)).all():
        raise ValueError("served predicted_points must be finite for every simulated player")
    players["predicted_points"] = predicted.astype(float)
    players["season"], players["week"] = season, week
    players["is_cold_start"] = ~players["player_id"].isin(known_player_ids)
    players = players.sort_values(["game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    players["role_key"] = attach_roles(players)
    report["simulated"] = int(len(players))
    return games, players, report


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def simulate_week(games: pd.DataFrame, rows: pd.DataFrame, artifacts: SimulationArtifacts, *,
                  draws: int, seed: int, model_config: dict) -> SimulationPayload:
    matrices, ordered = {}, []
    for game_id, game_rows in rows.groupby("game_id", sort=True):
        matrices[game_id] = simulate_game(game_rows, calibration=artifacts.calibration,
                                          dependence=artifacts.dependence, n_draws=draws, seed=seed)
        ordered.append(game_rows)
    players = pd.concat(ordered, ignore_index=True).rename(columns={"predicted_points": "served_prediction"})
    payload = SimulationPayload(
        seed=int(seed), model_config=_json_safe(model_config),
        games=games[["game_id", "season", "week", "home_team", "away_team", *SERVED_GAME_FIELDS]]
        .reset_index(drop=True),
        players=players[["game_id", "player_id", "team", "position", "served_prediction", "role_key",
                         "is_cold_start"]],
        draws=matrices)
    return payload


def generate_week(season: int, week: int, draws: int, seed: int, write_parquet: bool,
                  artifacts: SimulationArtifacts) -> Path | None:
    game_path = DOCS_DATA / f"game_predictions_{season}_wk{week}.json"
    player_path = DOCS_DATA / f"weekly_{season}_wk{week}.json"
    if not game_path.exists() or not player_path.exists():
        print(f"  wk{week}: missing game/player input, skipped")
        return None
    if (season, week) <= artifacts.fitted_through:
        print(f"  wk{week}: artifacts were fitted on data through {artifacts.fitted_through}, "
              f"which includes this week's outcomes; refused")
        return None
    games, rows, report = build_week_inputs(_read(game_path), _read(player_path), season=season,
                                            week=week, known_player_ids=artifacts.known_player_ids)
    if rows.empty:
        print(f"  wk{week}: no scheduled QB/RB/WR/TE rows, skipped")
        return None
    manifest = artifacts.manifest or {}
    provenance = manifest.get("provenance", {})
    payload = simulate_week(games, rows, artifacts, draws=draws, seed=seed, model_config={
        "simulation_status": "calibrated_copula", "season": season, "week": week, "draws": draws,
        "marginal_candidate": manifest.get("marginal_candidate"),
        "selection_reason": artifacts.selection.get("reason"),
        "dependence": "role_factor", "dependence_fit_status": artifacts.dependence.diagnostics.get("status"),
        "dependence_evidence": provenance.get("dependence_evidence"),
        "artifacts_run_id": manifest.get("run_id"),
        "artifacts_fitted_through": list(artifacts.fitted_through),
        "source_oof_panel_sha256": provenance.get("source_oof_panel_sha256"),
        "served_game_fields": SERVED_GAME_FIELDS, "scope_caveat": SCOPE_CAVEAT,
        "player_rows": report,
    })
    output = DOCS_DATA / f"simulation_{season}_wk{week}.json"
    write_simulation_site_payload(payload, json_path=output)
    if write_parquet:
        write_simulation_parquet(payload, parquet_dir=PARQUET_DATA / str(season), stem=output.stem)
    print(f"  wk{week}: {len(payload.draws)} games, {report['simulated']} players "
          f"({report['offered'] - report['simulated']} excluded) -> {output.name}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS,
                        help="artifact run directory, or a root containing latest.json")
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("--draws must be at least 2")
    try:
        artifacts = load_simulation_artifacts(args.artifacts)
    except (FileNotFoundError, ValueError) as exc:
        print(f"simulation artifacts unavailable: {exc}")
        return 1
    weeks = _weeks(args.season, args.week)
    if not weeks:
        print(f"no game prediction files found for {args.season}")
        return 1
    written = [generate_week(args.season, week, args.draws, args.seed, args.write_parquet, artifacts)
               for week in weeks]
    return 0 if any(written) else 1


if __name__ == "__main__":
    raise SystemExit(main())

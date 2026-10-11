"""Saved game-outcome models: a refit-only path, a manifest, and a staleness check.

The win/loss, margin and total models are trained once by
scripts/train_game_outcome_model.py and scripts/train_game_margin_model.py,
which also run (and, with --tune-optuna, tune) a walk-forward backtest. That
is slow and is not needed to take in a new week of results. This module refits
the nine saved models on the games completed so far with the hyperparameters
already recorded in the metadata sidecars, so a new week costs seconds, and
records what each saved set was fitted on in data/models/game_models_manifest.json.

Two sets:

    production        every completed game; what the site serves for games
                      that have not kicked off. Lives in MODELS_DIR.
    walkforward_<S>   seasons before S only; scores the games of season S out
                      of sample, never published (src/evaluation/game_backtest.py).
                      Lives in MODELS_DIR/game_walkforward/<S>/.

The joblib files are gitignored (*.joblib); the manifest is tracked, so a fresh
checkout can see which games the models behind a published number had seen.
Sidecar metadata (backtest, tuned parameters) is left as the training scripts
wrote it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DB_PATH, MODELS_DIR
from src.models.game_outcome.features import build_game_outcome_rows, build_margin_total_rows, feature_columns
from src.models.game_outcome.models import (
    GameMarginRidgeModel,
    GameOutcomeLogisticModel,
    GameOutcomeRFModel,
    GameOutcomeXGBModel,
    GameRegressionRFModel,
    GameRegressionXGBModel,
    load_model,
    save_model,
)

MANIFEST_NAME = "game_models_manifest.json"
WALKFORWARD_DIR = "game_walkforward"
SCHEMA_VERSION = 1

# (target, arm key in the sidecar's tuned_params, artifact file, class, site column, kind)
ARTIFACTS = [
    ("outcome", "logistic", "game_outcome_logistic.joblib", GameOutcomeLogisticModel, "home_win_prob_logistic", "proba"),
    ("outcome", "xgboost", "game_outcome_xgb.joblib", GameOutcomeXGBModel, "home_win_prob_xgb", "proba"),
    ("outcome", "random_forest", "game_outcome_rf.joblib", GameOutcomeRFModel, "home_win_prob_rf", "proba"),
    ("margin", "ridge", "game_margin_ridge.joblib", GameMarginRidgeModel, "predicted_margin_ridge", "point"),
    ("margin", "xgboost", "game_margin_xgb.joblib", GameRegressionXGBModel, "predicted_margin_xgb", "point"),
    ("margin", "random_forest", "game_margin_rf.joblib", GameRegressionRFModel, "predicted_margin_rf", "point"),
    ("total", "ridge", "game_total_ridge.joblib", GameMarginRidgeModel, "predicted_total_ridge", "point"),
    ("total", "xgboost", "game_total_xgb.joblib", GameRegressionXGBModel, "predicted_total_xgb", "point"),
    ("total", "random_forest", "game_total_rf.joblib", GameRegressionRFModel, "predicted_total_rf", "point"),
]
SIDECARS = {t: f"game_{t}_model_metadata.json" for t in ("outcome", "margin", "total")}
LABELS = {"outcome": "home_win", "margin": "home_margin", "total": "game_total"}


def walkforward_name(season: int) -> str:
    return f"walkforward_{int(season)}"


def set_dir(name: str, models_dir: Path = MODELS_DIR) -> Path:
    if name == "production":
        return Path(models_dir)
    if name.startswith("walkforward_"):
        return Path(models_dir) / WALKFORWARD_DIR / name.split("_", 1)[1]
    raise ValueError(f"unknown model set {name!r}")


def tuned_params(models_dir: Path = MODELS_DIR) -> dict[str, dict]:
    """Per-target hyperparameters recorded by the training scripts (empty if never tuned)."""
    out = {}
    for target, filename in SIDECARS.items():
        path = Path(models_dir) / filename
        out[target] = json.loads(path.read_text()).get("tuned_params", {}) if path.exists() else {}
    return out


def completed_through(con: sqlite3.Connection, before_season: Optional[int] = None) -> Optional[tuple[int, int]]:
    """(season, week) of the latest completed, non-tie game, optionally before a season."""
    rows = build_game_outcome_rows(con=con)
    if before_season is not None:
        rows = rows[rows["season"] < before_season]
    if rows.empty:
        return None
    last = rows.sort_values(["season", "week"]).iloc[-1]
    return int(last["season"]), int(last["week"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_sha256(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parents[3]).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def read_manifest(models_dir: Path = MODELS_DIR) -> dict:
    path = Path(models_dir) / MANIFEST_NAME
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "sets": {}}
    return json.loads(path.read_text())


def fit_set(name: str, *, con: sqlite3.Connection, models_dir: Path = MODELS_DIR,
            before_season: Optional[int] = None, params: Optional[dict] = None) -> dict:
    """Fit all nine models on the completed games (before `before_season` if given), save, record.

    Returns the manifest entry. Nothing is tuned and no backtest runs: the
    hyperparameters are the ones the sidecars already hold.
    """
    params = params if params is not None else tuned_params(models_dir)
    frames = {"outcome": build_game_outcome_rows(con=con), "margin": build_margin_total_rows(con=con)}
    if before_season is not None:
        frames = {t: f[f["season"] < before_season].reset_index(drop=True) for t, f in frames.items()}
    feats = feature_columns(frames["outcome"])
    if feature_columns(frames["margin"]) != feats:
        raise ValueError("outcome and margin/total frames disagree on feature columns")
    if frames["outcome"].empty:
        raise ValueError("no completed games to fit on")

    out_dir = set_dir(name, models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for target, arm, filename, cls, _, _ in ARTIFACTS:
        df = frames["outcome" if target == "outcome" else "margin"]
        model = cls(**params.get(target, {}).get(arm, {})).fit(df[feats], df[LABELS[target]].to_numpy())
        save_model(model, out_dir / filename)
        artifacts[filename] = _sha256(out_dir / filename)

    through = frames["outcome"].sort_values(["season", "week"]).iloc[-1]
    entry = {
        "dir": str(out_dir.relative_to(Path(models_dir).parent.parent)) if Path(models_dir).parent.parent in out_dir.parents else str(out_dir),
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": {"season": int(through["season"]), "week": int(through["week"])},
        "trained_before_season": None if before_season is None else int(before_season),
        "n_games": int(len(frames["outcome"])),
        "n_features": len(feats),
        "feature_columns_sha256": hashlib.sha256(json.dumps(feats).encode()).hexdigest(),
        "hyperparameters_from": {t: SIDECARS[t] for t in SIDECARS},
        "hyperparameters_sha256": params_sha256(params),
        "artifacts_sha256": artifacts,
        "git_commit": _git_commit(),
    }
    manifest = read_manifest(models_dir)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest.setdefault("sets", {})[name] = entry
    (Path(models_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return entry


def check_set(name: str, *, con: sqlite3.Connection, models_dir: Path = MODELS_DIR,
              before_season: Optional[int] = None) -> tuple[bool, str]:
    """Is the saved set usable as is? (ok, reason).

    Production is stale once any game completes after `trained_through`. A
    walk-forward set depends on nothing that moves in-season, so it is stale
    only when its files, feature list or training cut-off no longer match.
    """
    entry = read_manifest(models_dir).get("sets", {}).get(name)
    if entry is None:
        return False, f"no manifest entry for {name!r}"
    out_dir = set_dir(name, models_dir)
    for filename, sha in entry["artifacts_sha256"].items():
        path = out_dir / filename
        if not path.exists():
            return False, f"{path} is missing"
        if _sha256(path) != sha:
            return False, f"{path.name} does not match the manifest (changed since it was fitted)"
    if before_season is not None:
        if entry.get("trained_before_season") != int(before_season):
            return False, f"fitted before season {entry.get('trained_before_season')}, wanted {before_season}"
    latest = completed_through(con, before_season=before_season)
    got = (entry["trained_through"]["season"], entry["trained_through"]["week"])
    if latest is not None and latest > got:
        return False, f"fitted through {got[0]} wk{got[1]}; games are complete through {latest[0]} wk{latest[1]}"
    if entry.get("hyperparameters_sha256") != params_sha256(tuned_params(models_dir)):
        return False, "the hyperparameters in the metadata sidecars changed since the models were fitted"
    cols = feature_columns(build_game_outcome_rows(con=con))
    if hashlib.sha256(json.dumps(cols).encode()).hexdigest() != entry["feature_columns_sha256"]:
        return False, "the feature list changed since the models were fitted"
    return True, f"fitted through {got[0]} wk{got[1]} on {entry['n_games']} games"


def ensure_set(name: str, *, con: sqlite3.Connection, models_dir: Path = MODELS_DIR,
               before_season: Optional[int] = None, refit: bool = True) -> tuple[dict, str]:
    """Return the manifest entry for a current set, refitting first if it is stale (and `refit`)."""
    ok, why = check_set(name, con=con, models_dir=models_dir, before_season=before_season)
    if not ok:
        if not refit:
            return read_manifest(models_dir).get("sets", {}).get(name, {}), f"STALE: {why}"
        entry = fit_set(name, con=con, models_dir=models_dir, before_season=before_season)
        return entry, f"refit ({why}) -> through {entry['trained_through']['season']} wk{entry['trained_through']['week']}"
    return read_manifest(models_dir)["sets"][name], why


def load_set(name: str, models_dir: Path = MODELS_DIR) -> dict:
    """{site column: (model, kind)} for the saved set."""
    out_dir = set_dir(name, models_dir)
    return {column: (load_model(out_dir / filename), kind) for _, _, filename, _, column, kind in ARTIFACTS}


def predict_set(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Every model's prediction for the feature frame X, one column per model."""
    X = X.apply(pd.to_numeric, errors="coerce")
    cols = {}
    for column, (model, kind) in models.items():
        cols[column] = model.predict_proba(X)[:, 1] if kind == "proba" else np.asarray(model.predict(X), dtype=float)
    return pd.DataFrame(cols, index=X.index)


def open_db() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))

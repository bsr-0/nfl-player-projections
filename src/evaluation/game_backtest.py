"""Season-pure backtest of the game-outcome models on a season's completed games.

The site's honest record (published_record.py) only holds what was committed
before kickoff. Games nobody published a prediction for -- every week-1 game
(the game page did not exist), a game whose file was never committed -- would
otherwise stay blank. This module scores them, and every other completed game
of the season, with models fitted on EARLIER SEASONS ONLY (the registry's
walk-forward set). The season's own results never reach the fit, so each
prediction is out of sample; the team form, Elo and rest features of a game
look only at games strictly before it (_lagged_form, _team_elo_pre).

It is a backtest, not a publication: rows are tagged ``source="backtest"`` and
the page labels them so. A game with a published prediction keeps its published
row; the backtest rows for the same games feed the summary only.

Remaining differences from a prediction made at the time, stated on the page:
the spread and total lines are the ones now stored for the game (the closing
line in the schedule feed), and the Elo/form inputs are rebuilt from the
database as it stands today.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR
from src.evaluation.published_record import game_metrics, kickoff_times
from src.models.game_outcome import registry
from src.models.game_outcome.features import build_game_outcome_rows, feature_columns

ID = ["season", "week", "home_team", "away_team"]


def complete_weeks(games: pd.DataFrame) -> list[int]:
    """Weeks in which every scheduled game has a final score."""
    done = games.groupby("week")["final"].all()
    return sorted(int(w) for w in done[done].index)


def season_backtest(season: int, con: sqlite3.Connection, *, models_dir: Path = MODELS_DIR,
                    refit: bool = True) -> tuple[pd.DataFrame, dict]:
    """Backtest rows (one per completed game of every complete week) and a description.

    The description carries what the models were fitted on, so the page and the
    audit trail can say it: fitted_before_season, n_games, trained_through.
    """
    name = registry.walkforward_name(season)
    entry, note = registry.ensure_set(name, con=con, models_dir=models_dir, before_season=season, refit=refit)
    if not entry or "STALE" in note:
        raise RuntimeError(f"walk-forward game models for {season} are not usable: {note}")
    models = registry.load_set(name, models_dir)

    rows = build_game_outcome_rows(con=con)
    rows = rows[rows["season"] == season].reset_index(drop=True)
    if rows.empty:
        return pd.DataFrame(), {"note": note}
    preds = registry.predict_set(models, rows[feature_columns(rows)])
    out = pd.concat([rows[ID + ["spread_line", "total_line"]], preds], axis=1)

    games = kickoff_times(season, con)
    out = out.merge(games[["week", "home_team", "away_team", "kickoff", "home_score", "away_score", "final"]],
                    on=["week", "home_team", "away_team"], how="left")
    out["final"] = out["final"].fillna(False).astype(bool)
    keep = complete_weeks(games)
    out = out[out["week"].isin(keep)].sort_values(["week", "kickoff"]).reset_index(drop=True)
    info = {
        "note": note,
        "fitted_before_season": entry["trained_before_season"],
        "trained_through": entry["trained_through"],
        "n_training_games": entry["n_games"],
        "fitted_at": entry["fitted_at"],
        "weeks": keep,
    }
    return out, info


def backtest_summary(rows: pd.DataFrame) -> dict:
    """Per-week and season metrics, in the shape published_record.game_metrics uses."""
    if rows.empty:
        return {"n": 0}
    return {
        "season": game_metrics(rows),
        "weeks": {str(int(w)): game_metrics(g) for w, g in rows.groupby("week")},
    }


def unpublished_rows(backtest: pd.DataFrame, published_keys: set[tuple]) -> pd.DataFrame:
    """Backtest rows for games with no published prediction, tagged ``source="backtest"``."""
    if backtest.empty:
        return backtest
    key = list(zip(backtest["week"], backtest["home_team"], backtest["away_team"]))
    mask = np.array([k not in published_keys for k in key])
    return backtest[mask].assign(source="backtest")

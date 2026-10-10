"""Emit game-outcome/margin/total predictions for docs/game_predictions.html.

Serves the phase-1/2 models in src/models/game_outcome/ (see GAPS.md's
2026-09-18/19 entries) -- win/loss probability, predicted spread margin,
predicted over/under total -- via the same build_prediction_rows()/
load_model() serving path as scripts/predict_upcoming_games.py.

Two kinds of rows per week file (`source`):
- "live": a game that has not kicked off, predicted now.
- "published": a game that has kicked off. Its row is the one committed to
  this repository before its kickoff (src/evaluation/published_record.py),
  never a re-run: the saved models are trained once on all history, so
  re-scoring a played game would be in-sample and overstate accuracy. Once
  the game is final the row carries the score, and the meta file carries
  the week's and the season's accuracy of what was published, beside the
  market line it was published against.
The models' walk-forward backtest metadata stays the long-run accuracy claim.

Usage:
    python scripts/generate_game_predictions_data.py
    python scripts/generate_game_predictions_data.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import MODELS_DIR

OUT_DIR = Path("docs/data")

# (output column, artifact filename, "proba" for classifiers or "point" for
# regressors) -- same list as scripts/predict_upcoming_games.py, kept as an
# independent copy rather than a shared import since that script prints a
# console table and has no reason to depend on this one (or vice versa).
MODEL_SPECS = [
    ("home_win_prob_logistic", "game_outcome_logistic.joblib", "proba"),
    ("home_win_prob_xgb", "game_outcome_xgb.joblib", "proba"),
    ("home_win_prob_rf", "game_outcome_rf.joblib", "proba"),
    ("predicted_margin_ridge", "game_margin_ridge.joblib", "point"),
    ("predicted_margin_xgb", "game_margin_xgb.joblib", "point"),
    ("predicted_margin_rf", "game_margin_rf.joblib", "point"),
    ("predicted_total_ridge", "game_total_ridge.joblib", "point"),
    ("predicted_total_xgb", "game_total_xgb.joblib", "point"),
    ("predicted_total_rf", "game_total_rf.joblib", "point"),
]


def _load_if_exists(path: Path):
    from src.models.game_outcome.models import load_model
    if not path.exists():
        return None
    return load_model(path)


def measured_accuracy() -> dict | None:
    """Honest, held-out walk-forward backtest numbers from each model's own
    metadata sidecar -- see module docstring for why this page never shows
    a per-game predicted-vs-actual retrospective the way weekly.html does.
    Returns None (page says nothing) if a metadata file is missing rather
    than asserting a stale/guessed number.
    """
    def _pooled(filename: str) -> dict | None:
        path = MODELS_DIR / filename
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return raw.get("backtest", {}).get("pooled")

    outcome = _pooled("game_outcome_model_metadata.json")
    margin = _pooled("game_margin_model_metadata.json")
    total = _pooled("game_total_model_metadata.json")
    if not (outcome or margin or total):
        return None

    def _round(d, keys):
        # NaN is a real, legitimate value here (e.g. MarketLineBaseline's
        # ats_accuracy/ou_accuracy is NaN by construction -- it predicts the
        # line exactly, so it never "picks" a side). Python's json.dumps
        # emits bare NaN, which is NOT valid JSON per spec -- browsers'
        # JSON.parse rejects it outright (found 2026-09-19 testing this page
        # with Playwright; Python's own permissive json.loads had masked it
        # in local dev). Convert to None -> JSON null instead.
        if not d:
            return None
        out = {}
        for k in keys:
            if k not in d:
                continue
            v = float(d[k])
            out[k] = None if v != v else round(v, 4)  # v != v is the NaN check
        return out

    out = {}
    if outcome:
        out["win_loss"] = {
            arm: _round(m, ["accuracy", "log_loss", "roc_auc"])
            for arm, m in outcome.items()
        }
    if margin:
        out["margin"] = {
            arm: _round(m, ["mae", "rmse", "ats_accuracy"])
            for arm, m in margin.items()
        }
    if total:
        out["total"] = {
            arm: _round(m, ["mae", "rmse", "ou_accuracy"])
            for arm, m in total.items()
        }
    return out


def _clean(df: pd.DataFrame) -> list:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(float).round(3)
    return json.loads(out.to_json(orient="records"))


def build_predictions(season: int, games: pd.DataFrame, now: pd.Timestamp) -> dict[int, pd.DataFrame]:
    """Live predictions for every game that has not kicked off, by week."""
    from src.models.game_outcome.features import build_prediction_rows, feature_columns

    rows = build_prediction_rows(season)
    if rows.empty:
        return {}
    upcoming = games.loc[games["kickoff"] > now, ["week", "home_team", "away_team"]]
    rows = rows.merge(upcoming, on=["week", "home_team", "away_team"], how="inner")
    if rows.empty:
        return {}

    feat_cols = feature_columns(rows)
    # Keep the site-generation serving path identical to the CLI path:
    # schedule odds may arrive from SQLite as object strings, while the saved
    # models were trained on numeric feature columns.
    X = rows[feat_cols].apply(pd.to_numeric, errors="coerce")
    out = rows[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()

    missing = []
    for out_col, filename, kind in MODEL_SPECS:
        model = _load_if_exists(MODELS_DIR / filename)
        if model is None:
            missing.append(filename)
            continue
        if kind == "proba":
            out[out_col] = model.predict_proba(X)[:, 1]
        else:
            out[out_col] = model.predict(X)
    if missing:
        print(f"  warning: missing model artifact(s), skipping: {missing}")
    # season/week are implied by the filename (same convention as
    # weekly_{season}_wk{N}.json) -- dropped rather than carried per row.
    return {int(wk): d.drop(columns=["season", "week"]).assign(source="live")
            for wk, d in out.groupby("week")}


PUBLISHED_KEEP = ["published_at", "published_commit", "kickoff", "final", "home_score", "away_score"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()

    from src.utils.nfl_calendar import get_current_nfl_season
    season = args.season or get_current_nfl_season()

    import sqlite3
    from config.settings import DB_PATH
    from src.evaluation.published_record import game_week, kickoff_times, season_game_summary

    now = pd.Timestamp.now(tz="UTC")
    with sqlite3.connect(DB_PATH) as con:
        games = kickoff_times(season, con)

    print(f"Generating game predictions for {season}: live for unplayed games, "
          "published-before-kickoff rows for the rest...")
    live = build_predictions(season, games, now)

    weeks = sorted(int(w) for w in games["week"].unique())
    files, counts, results, published_frames = {}, {}, {}, []
    for wk in weeks:
        published, summary = game_week(season, wk, games, now)
        if summary:
            results[str(wk)] = summary
        parts = []
        if not published.empty:
            keep = [c for c in published.columns
                    if c in PUBLISHED_KEEP or c.startswith(("home_win_prob_", "predicted_"))
                    or c in ("home_team", "away_team", "spread_line", "total_line")]
            published = published[keep].assign(source="published")
            published_frames.append(published)
            parts.append(published)
        if wk in live:
            parts.append(live[wk])
        if parts:
            files[wk] = pd.concat(parts, ignore_index=True)

    if not files:
        print("No games to write -- nothing written.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(f"game_predictions_{season}_wk*.json"):
        stale.unlink()
    for wk, df in files.items():
        (OUT_DIR / f"game_predictions_{season}_wk{wk}.json").write_text(json.dumps(_clean(df)))
        counts[str(wk)] = len(df)
        n_pub = int((df["source"] == "published").sum())
        print(f"  wk{wk}: {len(df)} games ({n_pub} published before kickoff)")
    written = sorted(files)
    not_published = {w: r["games_not_published"] for w, r in results.items() if r.get("games_not_published")}

    meta = {
        "season": int(season),
        "weeks": written,
        "counts": counts,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured": measured_accuracy(),
        # Played games: the predictions as committed before kickoff, scored.
        "week_results": results,
        "track_record": season_game_summary(published_frames),
        "games_not_published": not_published,
        "model": (
            "phase-1 win/loss classifiers (logistic/xgboost/random_forest) + "
            "phase-2 margin/total regressors (ridge/xgboost/random_forest), "
            "src/models/game_outcome/ -- see GAPS.md 2026-09-18/19 entries"
        ),
    }
    (OUT_DIR / "game_predictions_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote weeks {written[0]}-{written[-1]} for {season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

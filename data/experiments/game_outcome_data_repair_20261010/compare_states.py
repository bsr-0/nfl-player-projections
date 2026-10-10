"""Week 5 game predictions under three database states (copies of backups)."""
import shutil, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path("/Users/benrosen/Documents/prediction_models/nfl-player-projections"); sys.path.insert(0, str(ROOT))
SP = Path(sys.argv[1])
import pandas as pd
from config.settings import MODELS_DIR
from src.models.game_outcome.features import build_prediction_rows, feature_columns
from src.models.game_outcome.models import load_model
SPECS = [("p_logistic", "game_outcome_logistic.joblib", "proba"), ("p_xgb", "game_outcome_xgb.joblib", "proba"),
         ("p_rf", "game_outcome_rf.joblib", "proba"), ("margin_ridge", "game_margin_ridge.joblib", "point"),
         ("margin_xgb", "game_margin_xgb.joblib", "point"), ("total_ridge", "game_total_ridge.joblib", "point"),
         ("total_xgb", "game_total_xgb.joblib", "point")]
MODELS = {n: load_model(MODELS_DIR / f) for n, f, _ in SPECS}

def predict(con, season, week):
    rows = build_prediction_rows(season, week=week, con=con)
    X = rows[feature_columns(rows)].apply(pd.to_numeric, errors="coerce")
    out = rows[["season", "week", "home_team", "away_team", "spread_line", "total_line"]].copy()
    for n, f, kind in SPECS:
        out[n] = MODELS[n].predict_proba(X)[:, 1] if kind == "proba" else MODELS[n].predict(X)
    return out, rows

def open_copy(src, name):
    dst = SP / f"{name}.db"
    if not dst.exists():
        shutil.copyfile(src, dst)
        c = sqlite3.connect(str(dst)); c.execute("PRAGMA journal_mode=DELETE"); c.commit(); c.close()
    return sqlite3.connect(str(dst))

if __name__ == "__main__":
    B = ROOT / "data/backups"
    states = {"before_fixes": B / "nfl_data_pre_schedule_scores_20261009_234228.db",
              "scores_only": B / "nfl_data_pre_team_pbp_20261009_234609.db",
              "both_fixed": B / "nfl_data_pre_job_run_20261010_000934.db"}
    res, frames = {}, {}
    for name, path in states.items():
        con = open_copy(path, name)
        out, rows = predict(con, 2026, 5)
        res[name], frames[name] = out, rows
        con.close()
        print(name, "games", len(out), flush=True)
    pd.to_pickle({"res": res, "frames": frames}, SP / "week5_states.pkl")

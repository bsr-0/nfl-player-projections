"""Marginal CRPS/coverage of the calibrated simulation on 2026 target weeks 1-4."""
import pickle, sqlite3, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path("/Users/benrosen/Documents/prediction_models/nfl-player-projections"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.models.calibrated_simulation import game_seed
from src.models.residual_calibration import load_calibration, independent_residual_matrix, role_keys_from_panel_game
from src.models.simulation_evaluation import empirical_crps
from src.utils.helpers import calculate_fantasy_points_df

SP = Path(sys.argv[1]); DRAWS = 1000
cal = load_calibration(ROOT / "data/experiments/calibrated_simulation_backtest_20261007_raw_actuals/outer_folds/2025/calibration_full_week22.json")
con = sqlite3.connect(f"file:{ROOT / 'data/nfl_data.db'}?mode=ro", uri=True)
sched = pd.read_sql("SELECT season, week, home_team, away_team FROM schedule WHERE season = 2026 AND week <= 4", con)
stats = pd.read_sql("SELECT * FROM player_weekly_stats WHERE season = 2026 AND week <= 4", con)
hist = pd.read_sql("SELECT player_id, MIN(season * 100 + week) AS first_wk FROM player_weekly_stats WHERE season >= 2023 GROUP BY player_id", con)
stats["actual_points"] = calculate_fantasy_points_df(stats)
frames = []
for w in (1, 2, 3, 4):
    r = pickle.load(open(SP / f"w{w}.pkl", "rb"))["results"]
    r = r[["player_id", "team", "position", "predicted_points", "predicted_points_model"]].drop_duplicates("player_id").assign(week=w)
    frames.append(r)
p = pd.concat(frames)
p = p.merge(stats[["player_id", "week", "actual_points"]], on=["player_id", "week"], how="inner")   # played the target game
# game identity from the schedule: home/away and an id in the panel's format
g = pd.concat([sched.assign(team=sched.home_team), sched.assign(team=sched.away_team)])[["week", "team", "home_team", "away_team"]]
p = p.merge(g, on=["week", "team"], how="inner")
p["game_id"] = p.apply(lambda r: f"2026_{int(r.week):02d}_{r.away_team}_{r.home_team}", axis=1)
p = p.merge(hist, on="player_id", how="left")
p["is_cold_start"] = p.first_wk.isna() | (p.first_wk >= 202600 + p.week)       # no appearance since 2023 before this week
p = p[p.position.isin(["QB", "RB", "WR", "TE"])].reset_index(drop=True)
print("player-games scored:", len(p), "| by week:", p.groupby("week").size().to_dict(), "| cold start share:", round(p.is_cold_start.mean(), 3))

def score(col):
    rows = []
    for gid, game in p.groupby("game_id", sort=True):
        game = game.assign(predicted_points=game[col])
        seed = game_seed(42, gid)
        resid = independent_residual_matrix(game, cal, n_draws=DRAWS, seed=seed)
        for j, row in enumerate(game.itertuples(index=False)):
            d = row.predicted_points + resid[:, j]; t = row.actual_points
            rows.append({"week": row.week, "position": row.position, "is_cold_start": row.is_cold_start, "player_id": row.player_id,
                         "mae": abs(d.mean() - t), "crps": empirical_crps(d, t),
                         "cov50": float(np.quantile(d, .25) <= t <= np.quantile(d, .75)),
                         "cov80": float(np.quantile(d, .10) <= t <= np.quantile(d, .90))})
    return pd.DataFrame(rows)

out = {"unblended (model)": score("predicted_points_model"), "blend (served)": score("predicted_points")}
pd.to_pickle({"scores": out, "frame": p}, SP / "sim_2026_scores.pkl")
for name, s in out.items():
    print(f"\n{name}: n={len(s)}  MAE {s.mae.mean():.3f}  CRPS {s.crps.mean():.3f}  cov50 {s.cov50.mean():.3f}  cov80 {s.cov80.mean():.3f}")
    print(s.groupby("week")[["mae", "crps", "cov50", "cov80"]].mean().round(3).to_string())
    print(s.groupby("position")[["mae", "crps", "cov50", "cov80"]].mean().round(3).to_string())

"""Stack / team / game-total CRPS, independent vs role-factor dependence, on 2026 weeks 1-4."""
import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path("/Users/benrosen/Documents/prediction_models/nfl-player-projections"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from src.models.calibrated_simulation import apply_dependence, game_seed, sum_groups
from src.models.residual_calibration import load_calibration, independent_residual_matrix, role_keys_from_panel_game
from src.models.player_correlation import FactorCopulaModel
from src.models.simulation_evaluation import empirical_crps
SP = Path(sys.argv[1]); DRAWS = 1000
B = ROOT / "data/experiments/calibrated_simulation_backtest_20261007_raw_actuals/outer_folds/2025"
cal = load_calibration(B / "calibration_full_week22.json")
fm = json.loads((B / "factor_models.json").read_text())["22"]
models = {"independent": None, "team_factor": FactorCopulaModel.from_dict(fm["calibrated_team_factor"]),
          "role_factor": FactorCopulaModel.from_dict(fm["calibrated_role_factor"])}
d = pd.read_pickle(SP / "sim_2026_scores.pkl")["frame"]
out = []
for center in ("predicted_points_model", "predicted_points"):
    for gid, game in d.groupby("game_id", sort=True):
        game = game.assign(predicted_points=game[center]).reset_index(drop=True)
        game["role_key"] = role_keys_from_panel_game(game)
        base = game_seed(42, gid)
        indep = independent_residual_matrix(game, cal, n_draws=DRAWS, seed=base)
        pred = game.predicted_points.to_numpy(float)
        for mode, model in models.items():
            draws = pred + apply_dependence(indep, tuple(game.role_key), model, base + 1)
            actual = game.actual_points.to_numpy(float)
            for kind, side, idx in sum_groups(game):
                tot = draws[:, idx].sum(axis=1); t = actual[idx].sum()
                out.append({"center": center, "mode": mode, "kind": kind, "game_id": gid, "week": int(game.week.iloc[0]),
                            "crps": empirical_crps(tot, t), "cov80": float(np.quantile(tot, .1) <= t <= np.quantile(tot, .9)),
                            "cov50": float(np.quantile(tot, .25) <= t <= np.quantile(tot, .75))})
r = pd.DataFrame(out); r.to_pickle(SP / "sim_2026_sums.pkl")
for center, g in r.groupby("center"):
    print(f"\ncentred on {center}")
    print(g.groupby(["kind", "mode"])[["crps", "cov50", "cov80"]].mean().round(3).to_string())
    for kind in ("stack", "team_total", "game_total"):
        a = g[(g.kind == kind) & (g["mode"] == "independent")].set_index(["game_id"]).crps
        b = g[(g.kind == kind) & (g["mode"] == "role_factor")].set_index(["game_id"]).crps
        a = g[(g.kind == kind) & (g["mode"] == "independent")].reset_index(drop=True).crps
        b = g[(g.kind == kind) & (g["mode"] == "role_factor")].reset_index(drop=True).crps
        diff = (b - a).to_numpy(); se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"  {kind:10s} role_factor - independent CRPS = {diff.mean():+.3f} (se {se:.3f}, n={len(diff)})")

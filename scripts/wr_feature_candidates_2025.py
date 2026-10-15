#!/usr/bin/env python3
"""WR season-to-date volume features, tested on held-out 2025 (docs/PREREG_WR_FEATURES_2025.md).

    prepare   prepared train frame (2006-2024) and the as-of 2025 WR test frame, cached
    fit       refit the WR 1-week model for C0 (baseline), C1, C2 and predict 2025 as of each week
    score     --window dev|confirm: paired comparison vs C0 (confirm refuses until the dev
              report is committed)

Design. The WR model is refit on seasons through 2024 with the served fold's tuned
hyperparameters held fixed (no retuning). The trainer's own feature selection
(correlation, stability, VIF) is bypassed: every arm gets an explicit column list, so
a candidate can't be dropped silently for being correlated with the 3-game averages.
Everything else follows ModelTrainer.train_all_positions: median fill on the training
prefix, rows without a target dropped, recency weights, season labels for CV.

Nothing under data/models is read or written: fits run with MODELS_DIR redirected, and
the live artifacts are hashed before and after.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

EXP = ROOT / "data/experiments/full_ppr_head_to_head_20260930"
FOLD = EXP / "served_2025_fold_v6"
PREFLIGHT = EXP / "served_2025_preflight_v5"
IMPORTANCE = ROOT / "data/experiments/feature_group_importance_2025"
OUT = ROOT / "data/experiments/wr_features_2025"
KEY = ["player_id", "season", "week"]
B, SEED = 10_000, 42

# stat column behind each candidate feature (expanding mean through the origin game, same season)
SOURCE = {"targets_s2d": "targets", "receptions_s2d": "receptions", "receiving_yards_s2d": "receiving_yards",
          "neutral_targets_s2d": "neutral_targets", "high_leverage_touches_s2d": "high_leverage_touches"}
C1 = ["targets_s2d", "receptions_s2d", "receiving_yards_s2d"]
ARMS = {"C0": [], "C1": C1, "C2": C1 + ["neutral_targets_s2d", "high_leverage_touches_s2d"]}
DEV_WEEKS, CONFIRM_WEEKS = (1, 5), (6, 18)


# --- features --------------------------------------------------------------------------------------

def season_to_date(stats: pd.DataFrame) -> pd.DataFrame:
    """Per player-season, the mean of each source stat over games up to and including this week.

    Uses only games at or before the row's week, so a row's value never depends on a later game.
    Also returns `games_s2d` and `ppg_s2d` (fantasy points), used for evaluation only.
    """
    s = stats.sort_values(KEY).drop_duplicates(KEY, keep="last").reset_index(drop=True)
    g = s.groupby(["player_id", "season"], sort=False)
    out = s[KEY].copy()
    out["games_s2d"] = g.cumcount() + 1
    for feature, col in {**SOURCE, "ppg_s2d": "fantasy_points"}.items():
        if col in s.columns:
            out[feature] = g[col].cumsum() / out["games_s2d"]
    return out


# --- data ------------------------------------------------------------------------------------------

def load_stats() -> pd.DataFrame:
    from config.settings import DB_PATH
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cols = ["fantasy_points", *sorted(set(SOURCE.values()))]
    return pd.read_sql(f"SELECT player_id, season, week, {', '.join(cols)} FROM player_weekly_stats "
                       "WHERE week <= 18", con)


def live_hashes() -> dict[str, str]:
    from config.settings import MODELS_DIR
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(MODELS_DIR).glob("*"))
            if p.is_file() and p.suffix in {".joblib", ".json"}}


def prepare() -> None:
    from config.settings import POSITIONS
    from src.models.feature_preparation import _prepare_training_data
    from src.utils.models_dir import redirect_models_dir
    OUT.mkdir(parents=True, exist_ok=True)
    before = live_hashes()
    train = pd.read_parquet(PREFLIGHT / "raw_train.parquet")
    test = pd.read_parquet(PREFLIGHT / "raw_test.parquet")
    work = OUT / "models_scratch"
    work.mkdir(exist_ok=True)
    with redirect_models_dir(work):
        ptrain, ptest, _ = _prepare_training_data(train, test, list(POSITIONS), tune_hyperparameters=False,
                                                  n_trials=1, fit_models=False)
    if live_hashes() != before:
        raise SystemExit("live model files changed during prepare")
    ptrain[ptrain.position == "WR"].to_parquet(OUT / "train_wr.parquet", index=False)
    ptest[ptest.position == "WR"].to_parquet(OUT / "test_wr_onepass.parquet", index=False)
    # The one-pass test frame should match the cached one the importance run used. Float noise is fine; a
    # model input that moves by more than 1e-3 is not.
    cached = pd.read_parquet(IMPORTANCE / "prepared_test.parquet")
    a = ptest[ptest.position == "WR"].drop_duplicates(KEY).set_index(KEY)
    b = cached[cached.position == "WR"].drop_duplicates(KEY).set_index(KEY)
    common = a.index.intersection(b.index)
    inputs = [c for c in base_features() if c in a.columns and c in b.columns]
    worst = {c: float(np.nanmax(np.abs(a.loc[common, c].to_numpy(float) - b.loc[common, c].to_numpy(float))))
             for c in inputs}
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:3]
    print(f"prepared WR train rows {int((ptrain.position == 'WR').sum())}; {len(common)} test rows compared on "
          f"{len(inputs)} model inputs; largest differences {top}")
    if max(worst.values()) > 1e-3:
        raise SystemExit("a model input differs from the importance run's frame")


# --- fitting ---------------------------------------------------------------------------------------

def base_features() -> list[str]:
    import joblib
    m = joblib.load(FOLD / "models/multiweek_wr.joblib")
    return list(next(iter(m["unique_models"].values()))["model"].feature_names)


@contextlib.contextmanager
def fixed_params():
    """PositionModel(tune_hyperparameters=False) falls back to _get_default_params; make that the fold's tuned set."""
    import joblib
    from src.models.position_models import PositionModel
    tuned = next(iter(joblib.load(FOLD / "models/multiweek_wr.joblib")["unique_models"].values()))["model"].best_params
    original = PositionModel._get_default_params
    if set(tuned) != set(original(PositionModel("WR"))):
        raise SystemExit("tuned parameter families differ from the model's defaults")
    PositionModel._get_default_params = lambda self: copy.deepcopy(tuned)
    try:
        yield tuned
    finally:
        PositionModel._get_default_params = original


def training_matrix(pos: pd.DataFrame, columns: list[str]):
    """ModelTrainer's single-position path without its feature selection."""
    from src.models.position_models import VALIDATION_PCT
    from config.settings import MODEL_CONFIG
    pos = pos.sort_values(["season", "week"]).reset_index(drop=True)
    X = pos[columns].replace([np.inf, -np.inf], np.nan)
    split = int(len(X) * (1 - VALIDATION_PCT))
    medians = {}
    for col in X.columns:
        if X[col].isna().any():
            med = X[col].iloc[:split].median()
            medians[col] = float(med) if pd.notna(med) else 0.0
            X[col] = X[col].fillna(medians[col])
    y = pos["target_1w"]
    keep = ~y.isna()
    X, y, seasons = X[keep], y[keep], pos.loc[keep, "season"].to_numpy()
    sw = None
    halflife = MODEL_CONFIG.get("recency_decay_halflife")
    if halflife and seasons.max() > seasons.min():
        decay = np.power(0.5, (seasons.max() - seasons.astype(float)) / float(halflife))
        sw = decay / decay.max()
    return X, y, seasons, sw, medians


def fit(arms: list[str]) -> None:
    from src.models.position_models import MultiWeekModel
    from src.utils.models_dir import redirect_models_dir
    base = base_features()
    stats = season_to_date(load_stats())
    train = pd.read_parquet(OUT / "train_wr.parquet").merge(stats[KEY + list(SOURCE)], on=KEY, how="left")
    test = pd.read_parquet(IMPORTANCE / "prepared_test_asof.parquet")
    test = test[test.position == "WR"].reset_index(drop=True).merge(stats[KEY + list(SOURCE) + ["games_s2d", "ppg_s2d"]],
                                                                  on=KEY, how="left")
    cap = pd.read_parquet(FOLD / "raw_capture.parquet")
    cap = cap[(cap.position == "WR") & (cap.target_week <= 18)]
    cap["gkey"] = (cap.season.astype(str) + "wk" + cap.target_week.astype(str) + " "
                   + cap[["target_team", "target_opponent"]].apply(lambda r: "|".join(sorted(r)), axis=1))
    frame = test.merge(cap[KEY + ["target_week", "actual_points", "predicted_points", "gkey"]], on=KEY, how="inner")
    print(f"WR test rows {len(frame)} in {frame.gkey.nunique()} games; train rows {len(train)}")
    before = live_hashes()
    preds = frame[KEY + ["target_week", "actual_points", "gkey", "games_s2d", "ppg_s2d"]].copy()
    preds["saved_fold"] = frame["predicted_points"].to_numpy(float)
    info = {}
    with redirect_models_dir(OUT / "models_scratch"), fixed_params():
        for arm in arms:
            cols = base + ARMS[arm]
            X, y, seasons, sw, medians = training_matrix(train, cols)
            model = MultiWeekModel("WR").fit(X, {1: y}, tune_hyperparameters=False, sample_weight=sw, seasons=seasons)
            Xt = frame[cols].copy()
            for col in ARMS[arm]:                       # new features: the training medians, never the zero-fill
                Xt[col] = Xt[col].fillna(medians.get(col, 0.0))
            preds[arm] = np.asarray(model.predict(Xt, n_weeks=1), float)
            info[arm] = {"features": len(cols), "train_rows": int(len(X)),
                         "mae_vs_actual": float(np.abs(preds[arm] - preds.actual_points).mean())}
            print(arm, info[arm], flush=True)
    if live_hashes() != before:
        raise SystemExit("live model files changed during fit")
    preds.to_parquet(OUT / "predictions.parquet", index=False)
    (OUT / "fit_info.json").write_text(json.dumps({
        "arms": info, "saved_fold_mae": float(np.abs(preds.saved_fold - preds.actual_points).mean()),
        "rows": int(len(preds)), "live_artifacts_unchanged": True}, indent=2) + "\n")


# --- scoring ---------------------------------------------------------------------------------------

def paired(d: np.ndarray, W: np.ndarray) -> dict:
    b = (W @ d) / W.sum(1)
    return {"est": float(d.mean()), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975)),
            "upper_95_one_sided": float(np.quantile(b, .95))}


def slope(x: np.ndarray, r: np.ndarray) -> float | None:
    m = np.isfinite(x) & np.isfinite(r)
    return float(np.polyfit(x[m], r[m], 1)[0]) if m.sum() >= 10 else None


def spearman_by_week(df: pd.DataFrame, col: str) -> pd.Series:
    from scipy.stats import spearmanr
    return pd.Series({w: spearmanr(g[col], g.actual_points).statistic for w, g in df.groupby("target_week") if len(g) >= 5})


def pair_accuracy(df: pd.DataFrame, col: str) -> float:
    hit = tot = 0.0
    for _, g in df.groupby("target_week"):
        p, a = g[col].to_numpy(float), g.actual_points.to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        k = p[i] != p[j]
        lead = np.where(p[i] > p[j], a[i] - a[j], a[j] - a[i])[k]
        hit += (lead > 0).sum() + 0.5 * (lead == 0).sum()
        tot += k.sum()
    return float(hit / tot)


def score_window(preds: pd.DataFrame, weeks: tuple[int, int], arms: list[str]) -> dict:
    from src.evaluation.selection_gates import _weight_batches, assign_tiers
    d = preds[preds.target_week.between(*weeks)].reset_index(drop=True)
    y = d.actual_points.to_numpy(float)
    W = np.vstack(list(_weight_batches(d.target_week.to_numpy(), d.player_id.to_numpy(), B, SEED)))
    tiers = assign_tiers(pd.DataFrame({"season": d.season, "week": d.target_week, "position": "WR",
                                       "player_id": d.player_id, "projection": d["C0"]}), "projection")
    gap = d.ppg_s2d.to_numpy(float) - d["C0"].to_numpy(float)
    lag_rows = (d.games_s2d >= 3).to_numpy()
    out = {"weeks": list(weeks), "rows": int(len(d)), "games": int(d.gkey.nunique()), "arms": {}}
    base_err = d["C0"].to_numpy(float) - y
    for arm in arms:
        p = d[arm].to_numpy(float)
        e = p - y
        rho, rho0 = spearman_by_week(d, arm), spearman_by_week(d, "C0")
        dr = (rho - rho0).dropna()
        out["arms"][arm] = {
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())), "lean": float(e.mean()),
            "lean_by_tier": {t: float(e[tiers == t].mean()) for t in ("1-12", "13-36", "37+") if (tiers == t).any()},
            "mae_diff_vs_C0": paired(np.abs(e) - np.abs(base_err), W) if arm != "C0" else None,
            "rmse_diff_vs_C0": float(np.sqrt((e ** 2).mean()) - np.sqrt((base_err ** 2).mean())),
            "spearman": float(rho.mean()), "spearman_diff_vs_C0": float(dr.mean()) if len(dr) else None,
            "spearman_diff_upper95": (float(dr.mean() + 1.645 * dr.std(ddof=1) / np.sqrt(len(dr))) if len(dr) > 2 else None),
            "pair_accuracy": pair_accuracy(d, arm),
            "form_lag_slope": slope(gap[lag_rows], (y - p)[lag_rows]),
            "mae_by_week": {int(w): float(np.abs(e)[(d.target_week == w).to_numpy()].mean())
                            for w in sorted(d.target_week.unique())},
        }
    return out


def committed(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return tracked and clean


def score(window: str) -> None:
    preds = pd.read_parquet(OUT / "predictions.parquet")
    arms = [a for a in ARMS if a in preds.columns]
    dev_report = OUT / "report_dev.json"
    if window == "confirm":
        if not dev_report.exists() or not committed(dev_report):
            raise SystemExit("commit report_dev.json before the confirmation window is scored")
        target = OUT / "report_confirm.json"
    else:
        target = dev_report
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    res = score_window(preds, DEV_WEEKS if window == "dev" else CONFIRM_WEEKS, arms)
    res["window"] = window
    res["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.write_text(json.dumps(res, indent=2) + "\n")
    for arm, r in res["arms"].items():
        dd = r["mae_diff_vs_C0"]
        print(f"{window} {arm}: MAE {r['mae']:.3f} RMSE {r['rmse']:.3f} lean {r['lean']:+.2f} rho {r['spearman']:.3f} "
              f"pairs {r['pair_accuracy']:.4f} lag {r['form_lag_slope']}"
              + (f" | d {dd['est']:+.3f} ({dd['lo']:+.3f},{dd['hi']:+.3f})" if dd else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    f = sub.add_parser("fit")
    f.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    s = sub.add_parser("score")
    s.add_argument("--window", choices=["dev", "confirm"], required=True)
    a = ap.parse_args()
    {"prepare": prepare, "fit": lambda: fit(a.arms), "score": lambda: score(a.window)}[a.cmd]()


if __name__ == "__main__":
    main()

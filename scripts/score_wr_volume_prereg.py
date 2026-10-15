#!/usr/bin/env python3
"""Score docs/PREREG_WR_VOLUME_2026.md: C1' (WR volume features) vs the identical baseline C0'.

    check   the shadow weeks written so far (reads no outcome)
    score   once, after week 13's stats are complete: joins each valid week's shadow forecasts to raw
            ten-component full PPR and applies the pre-registered rule. Refuses to run early or to
            overwrite a report.

Reads only: the shadow `forecasts.csv` files (each checked against its manifest sha256),
the forward runner's frozen WR lists, and `player_weekly_stats`.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH
from scripts import wr_volume_shadow as shadow
from scripts.score_no_prior_prereg import actual_points, stats_complete
from src.evaluation.paired_ppr_comparison import file_sha256
from src.evaluation.selection_gates import ALPHA, _weight_batches

MARGIN, B, SEED, MIN_WEEKS = 0.03, 10_000, 42, 5
LAG_MIN_GAMES = 3
FORWARD = ROOT / "data/experiments/selection_forward_2026"
OUTPUT = shadow.OUT / "report.json"


def _spearman_by_week(df: pd.DataFrame, col: str) -> pd.Series:
    from scipy.stats import spearmanr
    return pd.Series({w: spearmanr(g[col], g.actual).statistic for w, g in df.groupby("week") if len(g) >= 5})


def _one_sided_upper(x: np.ndarray) -> float:
    """Mean plus the t-based one-sided 95% margin over week means (the rule's Spearman interval)."""
    from scipy.stats import t as student_t
    if len(x) < 3:
        return float("nan")
    return float(x.mean() + student_t.ppf(1 - ALPHA, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def pair_accuracy(df: pd.DataFrame, col: str) -> float:
    hit = tot = 0.0
    for _, g in df.groupby("week"):
        p, a = g[col].to_numpy(float), g.actual.to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        k = p[i] != p[j]
        lead = np.where(p[i] > p[j], a[i] - a[j], a[j] - a[i])[k]
        hit += (lead > 0).sum() + 0.5 * (lead == 0).sum()
        tot += k.sum()
    return float(hit / tot)


def slope(gap: np.ndarray, resid: np.ndarray, rows: np.ndarray) -> float | None:
    return float(np.polyfit(gap[rows], resid[rows], 1)[0]) if rows.sum() >= 10 else None


def score(frame: pd.DataFrame, *, B: int = B, seed: int = SEED) -> dict:
    """frame: one row per scored WR player-week with actual, c0, c1, week, player_id, games_before, ppg_before."""
    from src.evaluation.selection_gates import assign_tiers
    frame = frame.sort_values(["week", "player_id"]).reset_index(drop=True)
    W = np.vstack(list(_weight_batches(frame.week.to_numpy(), frame.player_id.to_numpy(), B, seed)))
    y = frame.actual.to_numpy(float)
    e0, e1 = frame.c0.to_numpy(float) - y, frame.c1.to_numpy(float) - y
    d = np.abs(e1) - np.abs(e0)
    boot = (W @ d) / W.sum(1)
    rmse_d = np.sqrt((W @ e1 ** 2) / W.sum(1)) - np.sqrt((W @ e0 ** 2) / W.sum(1))
    rho0, rho1 = _spearman_by_week(frame, "c0"), _spearman_by_week(frame, "c1")
    drho = (rho1 - rho0).dropna().to_numpy()                       # positive: C1' ranks better
    gap = (frame.ppg_before - frame.c0).to_numpy(float)
    lag = (frame.games_before >= LAG_MIN_GAMES).to_numpy() & np.isfinite(gap)
    s0, s1 = slope(gap, -e0, lag), slope(gap, -e1, lag)
    tiers = assign_tiers(frame.assign(season=shadow.SEASON, position="WR", projection=frame.c0), "projection")
    out = {
        "n_rows": int(len(frame)), "weeks": sorted(int(w) for w in frame.week.unique()), "B": B, "seed": seed,
        "margin": MARGIN, "mae": {"c0": float(np.abs(e0).mean()), "c1": float(np.abs(e1).mean())},
        "rule_1": {"mean_d": float(d.mean()), "upper_95_one_sided": float(np.quantile(boot, 1 - ALPHA)),
                   "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]},
        "rule_2": {"rmse_diff": float(np.sqrt((e1 ** 2).mean()) - np.sqrt((e0 ** 2).mean())),
                   "rmse_diff_lower_95_one_sided": float(np.quantile(rmse_d, ALPHA)),
                   "rmse_diff_upper_95_one_sided": float(np.quantile(rmse_d, 1 - ALPHA)),
                   "spearman_diff": float(drho.mean()) if len(drho) else None,
                   "spearman_diff_upper_95_one_sided": _one_sided_upper(drho) if len(drho) > 2 else None},
        "rule_3": {"lag_slope_c0": s0, "lag_slope_c1": s1},
        "report_only": {"lean": {"c0": float(e0.mean()), "c1": float(e1.mean())},
                        "lean_by_tier": {t: [float(e0[tiers == t].mean()), float(e1[tiers == t].mean())]
                                         for t in ("1-12", "13-36", "37+") if (tiers == t).any()},
                        "pair_accuracy": {"c0": pair_accuracy(frame, "c0"), "c1": pair_accuracy(frame, "c1")},
                        "mae_by_week": {int(w): [float(np.abs(e0)[(frame.week == w).to_numpy()].mean()),
                                                 float(np.abs(e1)[(frame.week == w).to_numpy()].mean())]
                                        for w in sorted(frame.week.unique())},
                        "superiority_upper_95_below_0": bool(np.quantile(boot, 1 - ALPHA) < 0)},
    }
    r1 = out["rule_1"]["mean_d"] < 0 and out["rule_1"]["upper_95_one_sided"] < MARGIN
    # "Not significantly worse": the one-sided 95% interval must not show a worsening. For RMSE a worsening is a
    # positive difference, so its lower bound must not be above 0; for Spearman a worsening is a negative
    # difference, so its upper bound must not be below 0.
    rmse_ok = out["rule_2"]["rmse_diff_lower_95_one_sided"] <= 0
    rho_upper = None if len(drho) <= 2 else _one_sided_upper(drho)
    rho_ok = rho_upper is not None and rho_upper >= 0
    out["rule_2"]["rmse_ok"], out["rule_2"]["spearman_ok"] = bool(rmse_ok), bool(rho_ok)
    r3 = s0 is not None and s1 is not None and abs(s1) < abs(s0)
    out["passes"] = {"rule_1": bool(r1), "rule_2": bool(rmse_ok and rho_ok), "rule_3": bool(r3),
                     "all": bool(r1 and rmse_ok and rho_ok and r3)}
    return out


def build_frame(valid: dict[int, str], con: sqlite3.Connection) -> pd.DataFrame:
    fc = pd.concat([pd.read_csv(p, dtype={"player_id": str}) for p in valid.values()], ignore_index=True)
    lists = pd.concat([pd.read_csv(FORWARD / f"week_{w:02d}/list.csv", dtype={"player_id": str}) for w in valid],
                      ignore_index=True)
    fc = fc.merge(lists[["player_id", "week"]], on=["player_id", "week"], how="inner", validate="one_to_one")
    stats = pd.read_sql("SELECT * FROM player_weekly_stats WHERE season = ? AND week <= ?", con,
                        params=(shadow.SEASON, max(shadow.WINDOW)))
    act = actual_points(stats[stats.week.isin(list(valid))])
    frame = fc.merge(act, on=["player_id", "season", "week"], how="inner", validate="one_to_one")
    st = stats[["player_id", "week", "fantasy_points"]].sort_values(["player_id", "week"])
    by = {p: (g.week.to_numpy(), np.cumsum(g.fantasy_points.to_numpy(float))) for p, g in st.groupby("player_id")}
    games, ppg = [], []
    for p, w in zip(frame.player_id, frame.week):
        weeks, cum = by.get(p, (np.array([]), np.array([])))
        i = int(np.searchsorted(weeks, w))
        games.append(i)
        ppg.append(cum[i - 1] / i if i else np.nan)
    return frame.assign(games_before=games, ppg_before=ppg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "score"])
    ap.add_argument("--output", type=Path, default=OUTPUT)
    a = ap.parse_args()
    status = shadow.check()
    if a.cmd == "check":
        print(json.dumps(status, indent=2))
        return
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    if not stats_complete(con, max(shadow.WINDOW)):
        raise SystemExit(f"week {max(shadow.WINDOW)} stats are not complete; the window is scored once, after it")
    valid = {w: str(shadow.OUT / f"week_{w:02d}/forecasts.csv") for w in status["valid_weeks"]}
    report = ({"void": True, "reason": f"only {len(valid)} valid weeks", "excluded": status["excluded"]}
              if len(valid) < MIN_WEEKS else score(build_frame(valid, con)) | {"void": False, "excluded": status["excluded"]})
    report["script_sha256"] = file_sha256(Path(__file__))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("void", "passes", "excluded")}, indent=2))


if __name__ == "__main__":
    main()

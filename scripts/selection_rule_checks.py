#!/usr/bin/env python3
"""Pre-registration checks for docs/PRODUCTION_SELECTION_RULE.md.

Read-only. Uses the 2025 matched-set paired rows (served vs Plan A) and the
rolling-3 projections only to measure the *inference procedure*:

1. Synthetic coverage of week-, player- and two-way-cluster bootstraps,
   across true differences, shared weekly shocks and seeds.
2. No-difference feasibility of the guardrails under rule B (fail when the
   one-sided 95% lower bound of the worsening is above 0, or the point
   worsening exceeds the backstop), per cell and family-wise.
3. Power: the same gates against injected regressions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

from src.evaluation.paired_ppr_comparison import KEY, assert_identical_key_sets, file_sha256

PAIRED = ROOT / "data/experiments/full_ppr_head_to_head_20260930/comparison_2025/paired_rows.csv"
ROLL3 = ROOT / "data/experiments/full_ppr_raw_truth_20260924/ppr_oof_rows.csv"
# Point-estimate backstop: 2x the original non-inferiority margins.
BACKSTOP = {"mae": 0.10, "rmse": 0.10, "bias": 0.10, "spearman": 0.06}
POSITIONS = ["QB", "RB", "WR", "TE"]
TIERINGS = ["tier_roll3", "tier_served"]


# ---------------------------------------------------------------- data
def load(paired: Path, roll3: Path) -> pd.DataFrame:
    df = pd.read_csv(paired, dtype={"player_id": str})
    r3 = pd.read_csv(roll3, dtype={"player_id": str})
    r3 = r3[r3.season.isin(df.season.unique())]
    r3 = r3.merge(df[["player_id", "season", "week"]], on=["player_id", "season", "week"])
    assert_identical_key_sets({
        "served": df[KEY], "plan_a": df[KEY], "rolling3": r3[KEY],
    })
    df = df.merge(r3[KEY + ["predicted_ppr_baseline"]].rename(
        columns={"predicted_ppr_baseline": "roll3"}), on=KEY, validate="one_to_one")
    for col in ("served_prediction", "plan_a_prediction", "roll3", "actual_full_ppr"):
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise ValueError(f"nonfinite {col}")
    for name, col in (("tier_roll3", "roll3"), ("tier_served", "served_prediction")):
        rank = (df.sort_values(["season", "week", "position", col, "player_id"],
                               ascending=[True, True, True, False, True])
                  .groupby(["season", "week", "position"]).cumcount() + 1).reindex(df.index)
        df[name] = np.where(rank <= 12, "1-12", np.where(rank <= 36, "13-36", "37+"))
    return df.reset_index(drop=True)


class Panel:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.y = df.actual_full_ppr.to_numpy(float)
        self.sv = df.served_prediction.to_numpy(float)
        self.pa = df.plan_a_prediction.to_numpy(float)
        self.week = df.week.to_numpy()
        _, self.wk = np.unique(self.week, return_inverse=True)
        _, self.pl = np.unique(df.player_id.to_numpy(), return_inverse=True)
        _, self.game = np.unique(df.week.astype(str) + "|" + df.team, return_inverse=True)
        _, self.grp = np.unique(df.week.astype(str) + "|" + df.position, return_inverse=True)
        self.W, self.P, self.G = self.wk.max() + 1, self.pl.max() + 1, self.grp.max() + 1
        self.grp_wk = np.zeros(self.G, int)
        self.grp_wk[self.grp] = self.wk

    def cells(self) -> dict[str, tuple[str, np.ndarray | None]]:
        pos = self.df.position.to_numpy()
        out = {"RMSE": ("rmse", None)}
        for p in POSITIONS:
            out[f"MAE {p}"] = ("mae", pos == p)
        for t in TIERINGS:
            for tier in ("1-12", "13-36"):
                out[f"MAE {t} {tier}"] = ("mae", self.df[t].to_numpy() == tier)
        out["|bias| overall"] = ("bias", None)
        for t in TIERINGS:
            for tier in ("1-12", "13-36", "37+"):
                out[f"|bias| {t} {tier}"] = ("bias", self.df[t].to_numpy() == tier)
        out["Spearman"] = ("spearman", None)
        return out


# ---------------------------------------------------------------- bootstraps
def cluster_matrix(rng, clusters: np.ndarray, B: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample only the clusters present; returns (B x C weights, remapped index)."""
    present, idx = np.unique(clusters, return_inverse=True)
    c = len(present)
    return rng.multinomial(c, np.full(c, 1 / c), size=B).astype(float), idx


def cluster_mean(x, idx, M):
    c = M.shape[1]
    return (M @ np.bincount(idx, x, c)) / (M @ np.bincount(idx, minlength=c).astype(float))


def twoway_mean(rng, x, wk, pl, B):
    Mw, wi = cluster_matrix(rng, wk, B)
    Mp, pi = cluster_matrix(rng, pl, B)
    w = Mw[:, wi] * Mp[:, pi]
    return (w @ x) / w.sum(1)


def twoway_weights(rng, wk, pl, B):
    """B x n row weights of the pigeonhole (week x player) bootstrap."""
    Mw, wi = cluster_matrix(rng, wk, B)
    Mp, pi = cluster_matrix(rng, pl, B)
    return Mw[:, wi] * Mp[:, pi]


def worsening(kind, cand, base, y, mask, Wt):
    """Point and bootstrap of (candidate worse) for one cell; Wt is B x n row weights."""
    m = np.ones_like(y) if mask is None else mask.astype(float)
    ec, eb = cand - y, base - y

    def agg(v):
        return Wt @ (v * m), (v * m).sum()
    n_b, n = agg(np.ones_like(y))
    if kind == "mae":
        a_b, a = agg(np.abs(ec) - np.abs(eb))
        return a / n, a_b / n_b
    if kind == "rmse":
        (qc_b, qc), (qb_b, qb) = agg(ec ** 2), agg(eb ** 2)
        return np.sqrt(qc / n) - np.sqrt(qb / n), np.sqrt(qc_b / n_b) - np.sqrt(qb_b / n_b)
    if kind == "bias":
        (sc_b, sc), (sb_b, sb) = agg(ec), agg(eb)
        return abs(sc / n) - abs(sb / n), np.abs(sc_b / n_b) - np.abs(sb_b / n_b)
    raise ValueError(kind)


def spearman_worsening(panel, cand, base, rows):
    """Mean within position-week Spearman, base minus candidate.

    Returns the point and the one-sided 95% lower bound from a t-interval over
    week means (few clusters: a week bootstrap undercovers here).
    """
    d = np.full(panel.G, np.nan)
    for g in np.unique(panel.grp[rows]):
        m = rows & (panel.grp == g)
        if m.sum() >= 3:
            d[g] = (spearmanr(base[m], panel.y[m]).statistic
                    - spearmanr(cand[m], panel.y[m]).statistic)
    ok = np.isfinite(d)
    wk = panel.grp_wk[ok]
    k = np.unique(wk)
    means = np.array([d[ok][wk == w].mean() for w in k])
    se = means.std(ddof=1) / np.sqrt(len(k))
    return d[ok].mean(), means.mean() - student_t.ppf(0.95, len(k) - 1) * se


# ---------------------------------------------------------------- 1. synthetic coverage
def variance_components(p: Panel) -> dict:
    d = np.abs(p.sv - p.y) - np.abs(p.pa - p.y)
    r = d - d.mean()
    out = {}
    for name, idx in (("week", p.wk), ("game", p.game), ("player", p.pl)):
        k = idx.max() + 1
        n = np.bincount(idx, minlength=k)
        mean = np.bincount(idx, r, k) / np.maximum(n, 1)
        within = ((r - mean[idx]) ** 2).sum() / (len(r) - k)
        between = np.average(mean[n > 0] ** 2, weights=n[n > 0])
        out[name] = float(np.sqrt(max(between - within / np.average(n[n > 0], weights=n[n > 0]), 0)))
        r = r - mean[idx]
    out["resid_pool"] = r
    out["resid"] = float(r.std())
    return out


def synthetic_coverage(p: Panel, vc: dict, B: int, reps: int, seeds, mus, week_mults) -> list[dict]:
    rows = []
    for mult in week_mults:
        for mu in mus:
            hits = {k: 0 for k in ("player_2s", "player_ub", "player_lb", "week_2s", "twoway_2s")}
            n_tw = n = 0
            for seed in seeds:
                rng = np.random.default_rng([seed, int(mu * 100), int(mult * 10)])
                for rep in range(reps):
                    x = (mu + rng.normal(0, vc["week"] * mult, p.W)[p.wk]
                         + rng.normal(0, vc["game"], p.game.max() + 1)[p.game]
                         + rng.normal(0, vc["player"], p.P)[p.pl]
                         + rng.choice(vc["resid_pool"], len(p.y)))
                    M, idx = cluster_matrix(rng, p.pl, B)
                    b = cluster_mean(x, idx, M)
                    lo, lb, ub, hi = np.quantile(b, [0.025, 0.05, 0.95, 0.975])
                    hits["player_2s"] += lo <= mu <= hi
                    hits["player_ub"] += mu <= ub
                    hits["player_lb"] += mu >= lb
                    M, idx = cluster_matrix(rng, p.wk, B)
                    lo, hi = np.quantile(cluster_mean(x, idx, M), [0.025, 0.975])
                    hits["week_2s"] += lo <= mu <= hi
                    if rep < reps // 4:
                        lo, hi = np.quantile(twoway_mean(rng, x, p.wk, p.pl, 500), [0.025, 0.975])
                        hits["twoway_2s"] += lo <= mu <= hi
                        n_tw += 1
                    n += 1
            rows.append({"week_sd_multiplier": mult, "week_sd": vc["week"] * mult, "true_diff": mu,
                         "reps": n, "twoway_reps": n_tw,
                         **{k: (v / n_tw if k == "twoway_2s" else v / n) for k, v in hits.items()}})
            print(f"  week_sd x{mult} mu={mu:.2f}: player 2s={rows[-1]['player_2s']:.3f} "
                  f"ub={rows[-1]['player_ub']:.3f} lb={rows[-1]['player_lb']:.3f} | "
                  f"week 2s={rows[-1]['week_2s']:.3f} | twoway 2s={rows[-1]['twoway_2s']:.3f}", flush=True)
    return rows


# ---------------------------------------------------------------- 2/3. gates
def evaluate_gates(p: Panel, cand, base, rows, rng, B, scaled=None) -> dict[str, dict]:
    out = {}
    Wt = twoway_weights(rng, p.wk[rows], p.pl[rows], B)
    for name, (kind, mask) in p.cells().items():
        if kind == "spearman":
            point, lb = spearman_worsening(p, cand, base, rows)
        else:
            msk = None if mask is None else mask[rows]
            point, b = worsening(kind, cand[rows], base[rows], p.y[rows], msk, Wt)
            lb = np.quantile(b[np.isfinite(b)], 0.05)
        out[name] = {"point": float(point), "lb": float(lb), "lb_fail": bool(lb > 0),
                     "backstop_fail": bool(point > BACKSTOP[kind]),
                     "scaled_fail": bool(scaled is not None and point > scaled[name])}
    return out


RULES = {
    "any LB fail": lambda lb, bs, sc: lb >= 1,
    ">=2 LB fails": lambda lb, bs, sc: lb >= 2,
    ">=3 LB fails": lambda lb, bs, sc: lb >= 3,
    "any LB or fixed backstop": lambda lb, bs, sc: lb >= 1 or bs >= 1,
    "any scaled backstop": lambda lb, bs, sc: sc >= 1,
    ">=2 LB or any scaled backstop": lambda lb, bs, sc: lb >= 2 or sc >= 1,
    ">=3 LB or any scaled backstop": lambda lb, bs, sc: lb >= 3 or sc >= 1,
}


def family(verdicts: dict[str, dict]) -> dict:
    lb = sum(v["lb_fail"] for v in verdicts.values())
    bs = sum(v["backstop_fail"] for v in verdicts.values())
    sc = sum(v["scaled_fail"] for v in verdicts.values())
    return {"n_lb_fail": lb, **{k: f(lb, bs, sc) for k, f in RULES.items()}}


def run_scenario(p: Panel, rows, R, B, seed, inject=None, label="", scaled=None):
    rng = np.random.default_rng(seed)
    per_cell, fams = {}, []
    for _ in range(R):
        s = (rng.random(p.P) < 0.5)[p.pl]          # per-player swap: true difference 0
        cand, base = np.where(s, p.pa, p.sv), np.where(s, p.sv, p.pa)
        if inject is not None:
            cand = inject(cand, base, rng)
        v = evaluate_gates(p, cand, base, rows, rng, B, scaled)
        fams.append(family(v))
        for k, x in v.items():
            per_cell.setdefault(k, []).append(x)
    cells = {k: {"lb_fail_rate": float(np.mean([x["lb_fail"] for x in xs])),
                 "backstop_fail_rate": float(np.mean([x["backstop_fail"] for x in xs])),
                 "scaled_fail_rate": float(np.mean([x["scaled_fail"] for x in xs])),
                 "mean_point": float(np.mean([x["point"] for x in xs])),
                 "sd_point": float(np.std([x["point"] for x in xs], ddof=1))}
             for k, xs in per_cell.items()}
    fam = {k: float(np.mean([f[k] for f in fams])) for k in RULES}
    fam["n_lb_fail_distribution"] = np.bincount([f["n_lb_fail"] for f in fams]).tolist()
    print(f"\n== {label}  (R={R}, B={B})")
    print("   block rate: " + "  ".join(f"[{k}]={v:.2f}" for k, v in fam.items() if k in RULES))
    print(f"   #LB-failing cells distribution: {fam['n_lb_fail_distribution']}")
    for k, c in cells.items():
        print(f"   {k:28s} point={c['mean_point']:+.3f} sd={c['sd_point']:.3f} lb_fail={c['lb_fail_rate']:.2f} "
              f"fixed_bs={c['backstop_fail_rate']:.2f} scaled_bs={c['scaled_fail_rate']:.2f}")
    return {"label": label, "R": R, "B": B, "family": fam, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", type=Path, default=PAIRED)
    ap.add_argument("--rolling3", type=Path, default=ROLL3)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small R/B smoke run")
    a = ap.parse_args()
    q = a.quick
    p = Panel(load(a.paired, a.rolling3))
    vc = variance_components(p)
    print(f"rows={len(p.y)} weeks={p.W} players={p.P} | sd week={vc['week']:.3f} "
          f"game={vc['game']:.3f} player={vc['player']:.3f} resid={vc['resid']:.3f}")

    print("\n1. synthetic coverage (nominal 0.95)")
    cov = synthetic_coverage(p, vc, B=500 if q else 2000, reps=20 if q else 300,
                             seeds=(1, 2, 3), mus=(0.0, 0.10, 0.20), week_mults=(1, 2, 4))

    allr = np.ones(len(p.y), bool)
    fwd = (p.week >= 6) & (p.week < 14)
    R, B = (10, 300) if q else (200, 1000)
    sd_gap = float(np.std(p.pa - p.sv))
    scen = []
    # Pass 1: null runs set the scaled backstop, max(2 x margin, 3 x null sd of the point).
    null_full = run_scenario(p, allr, R, B, a.seed, label="null: 2025 matched set")
    null_fwd = run_scenario(p, fwd, R // 2, B, a.seed + 6, label="null: 8-week forward window, weeks 6-13")
    scaled = {}
    for key, null in (("2025", null_full), ("forward", null_fwd)):
        scaled[key] = {c: max(BACKSTOP[kind], 3 * null["cells"][c]["sd_point"])
                       for c, (kind, _) in p.cells().items()}
    # Pass 2: every scenario judged with the scaled backstop.
    scen.append(run_scenario(p, allr, R, B, a.seed + 1, label="null: 2025 matched set (fresh draws)",
                             scaled=scaled["2025"]))
    scen.append(run_scenario(p, fwd, R // 2, B, a.seed + 7,
                             label="null: 8-week forward window, weeks 6-13 (fresh draws)", scaled=scaled["forward"]))
    w916 = (p.week >= 9) & (p.week < 17)
    scen.append(run_scenario(p, w916, R // 2, B, a.seed + 9,
                             label="null: 8-week forward window, weeks 9-16", scaled=scaled["forward"]))

    def noise_on(mask_fn, scale):
        def f(cand, base, rng):
            m = mask_fn(rng)
            return cand + np.where(m, rng.normal(0, sd_gap * scale, len(cand)), 0.0)
        return f

    def shift_away(size):
        def f(cand, base, rng):  # push the candidate's bias away from zero
            return cand + size * np.sign(np.mean(cand - p.y) or 1.0)
        return f
    pos = p.df.position.to_numpy()
    top = p.df.tier_roll3.to_numpy() == "1-12"
    injections = {
        "random 25% of players, full-size noise": noise_on(lambda r: (r.random(p.P) < 0.25)[p.pl], 1.0),
        "all QB rows, full-size noise": noise_on(lambda r: pos == "QB", 1.0),
        "all QB rows, half-size noise": noise_on(lambda r: pos == "QB", 0.5),
        "top-12 (rolling-3) rows, half-size noise": noise_on(lambda r: top, 0.5),
        "all rows, quarter-size noise": noise_on(lambda r: np.ones(len(p.y), bool), 0.25),
        "bias pushed 0.3 further from zero": shift_away(0.3),
    }
    for name, f in injections.items():
        scen.append(run_scenario(p, allr, R // 2, B, a.seed + 100, inject=f,
                                 label=f"regression: {name}", scaled=scaled["2025"]))
        scen.append(run_scenario(p, fwd, R // 4, B, a.seed + 200, inject=f,
                                 label=f"regression, forward weeks 6-13: {name}", scaled=scaled["forward"]))

    rng = np.random.default_rng(a.seed)
    g1 = []
    for _ in range(30 if not q else 3):
        s_ = (rng.random(p.P) < 0.5)[p.pl]
        d = np.abs(np.where(s_, p.pa, p.sv) - p.y) - np.abs(np.where(s_, p.sv, p.pa) - p.y)
        Wt = twoway_weights(rng, p.wk, p.pl, B)
        b = (Wt @ d) / Wt.sum(1)
        g1.append(float(np.quantile(b, 0.95) - d.mean()))
    print(f"\nG1 pooled MAE, two-way bootstrap: one-sided 95% half-width median={np.median(g1):.3f}")

    report = {
        "inputs": {"paired_rows": str(a.paired.relative_to(ROOT)), "paired_rows_sha256": file_sha256(a.paired),
                   "rolling3": str(a.rolling3.relative_to(ROOT)), "rolling3_sha256": file_sha256(a.rolling3)},
        "script_sha256": file_sha256(Path(__file__)),
        "seed": a.seed, "quick": q, "backstop": BACKSTOP,
        "variance_components": {k: v for k, v in vc.items() if k != "resid_pool"},
        "prediction_gap_sd": sd_gap,
        "synthetic_coverage": cov, "scaled_backstop": scaled,
        "g1_twoway_one_sided_halfwidth_median": float(np.median(g1)),
        "null_pass1": [null_full, null_fwd], "scenarios": scen,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()

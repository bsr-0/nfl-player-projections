"""Gates of docs/PRODUCTION_SELECTION_RULE.md, applied to paired target-game rows.

Read-only and deterministic for a given seed. The caller supplies one row per
target game with the actual, the incumbent's prediction, each candidate's
prediction, and the rolling-3 projection. Every model must cover exactly the
same keys: the caller asserts that with
`paired_ppr_comparison.assert_identical_key_sets` before building the frame,
and `evaluate` re-checks the frame itself.

Inference (see the rule):
* two-way (week x player) pigeonhole bootstrap, shared by every cell and
  candidate so the comparisons are paired;
* G1: one-sided superiority, Holm-corrected across candidates, plus a point
  improvement of at least MIN_EFFECT;
* G2-G6 guardrails: a cell fails when the one-sided 95% lower bound of its
  worsening is above 0; any single failing cell fails the stage. Spearman uses
  a one-sided t-interval over week means. No point backstop and no threshold
  derived from a null spread.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

ALPHA = 0.05
MIN_EFFECT = 0.10
COVERAGE_RANGE = (0.75, 0.85)
POSITIONS = ("QB", "RB", "WR", "TE")
TIERS = ("1-12", "13-36", "37+")


@dataclass(frozen=True)
class Columns:
    actual: str = "actual"
    incumbent: str = "incumbent"
    rolling3: str = "rolling3"


def assign_tiers(frame: pd.DataFrame, projection: str) -> np.ndarray:
    """Rank within (season, week, position), highest first, ties by player_id."""
    order = frame.sort_values(["season", "week", "position", projection, "player_id"],
                              ascending=[True, True, True, False, True])
    rank = (order.groupby(["season", "week", "position"]).cumcount() + 1).reindex(frame.index)
    return np.where(rank <= 12, "1-12", np.where(rank <= 36, "13-36", "37+"))


def guardrail_cells(frame: pd.DataFrame) -> dict[str, tuple[str, np.ndarray | None]]:
    """The 17 guardrail cells: name -> (kind, row mask or None for all rows)."""
    pos = frame.position.to_numpy()
    cells: dict[str, tuple[str, np.ndarray | None]] = {"G2 RMSE": ("rmse", None)}
    for p in POSITIONS:
        cells[f"G3 MAE {p}"] = ("mae", pos == p)
    for tiering in ("tier_rolling3", "tier_incumbent"):
        for tier in ("1-12", "13-36"):
            cells[f"G4 MAE {tiering} {tier}"] = ("mae", frame[tiering].to_numpy() == tier)
    cells["G5 |bias| overall"] = ("bias", None)
    for tiering in ("tier_rolling3", "tier_incumbent"):
        for tier in TIERS:
            cells[f"G5 |bias| {tiering} {tier}"] = ("bias", frame[tiering].to_numpy() == tier)
    cells["G6 Spearman"] = ("spearman", None)
    return cells


def _weight_batches(week: np.ndarray, player: np.ndarray, B: int, seed: int, batch: int = 500):
    """Yield (b x n) pigeonhole weights; the sequence depends only on the seed."""
    _, wi = np.unique(week, return_inverse=True)
    _, pi = np.unique(player, return_inverse=True)
    nw, npl = wi.max() + 1, pi.max() + 1
    rng = np.random.default_rng(seed)
    done = 0
    while done < B:
        b = min(batch, B - done)
        mw = rng.multinomial(nw, np.full(nw, 1 / nw), size=b).astype(float)
        mp = rng.multinomial(npl, np.full(npl, 1 / npl), size=b).astype(float)
        yield mw[:, wi] * mp[:, pi]
        done += b


def _cell_value(kind: str, ec: np.ndarray, eb: np.ndarray, w) -> np.ndarray | float:
    """Worsening (candidate minus incumbent; positive is worse) for weights w (b x n) or rows (n,)."""
    if w.ndim == 1:
        w = w[None, :]
        squeeze = True
    else:
        squeeze = False
    n = w.sum(1)
    if kind == "mae":
        v = (w @ (np.abs(ec) - np.abs(eb))) / n
    elif kind == "rmse":
        v = np.sqrt((w @ ec ** 2) / n) - np.sqrt((w @ eb ** 2) / n)
    elif kind == "bias":
        v = np.abs((w @ ec) / n) - np.abs((w @ eb) / n)
    else:
        raise ValueError(kind)
    return float(v[0]) if squeeze else v


def spearman_worsening(frame: pd.DataFrame, cand: np.ndarray, inc: np.ndarray, actual: np.ndarray):
    """Mean within position-week Spearman, incumbent minus candidate, and its one-sided 95% LB."""
    groups = frame.groupby(["week", "position"]).indices
    rows = []
    for (week, _), idx in groups.items():
        if len(idx) >= 3:
            rc = spearmanr(cand[idx], actual[idx]).statistic
            ri = spearmanr(inc[idx], actual[idx]).statistic
            if np.isfinite(rc) and np.isfinite(ri):
                rows.append((week, ri - rc))
    d = pd.DataFrame(rows, columns=["week", "d"])
    means = d.groupby("week").d.mean().to_numpy()
    if len(means) < 2:
        raise ValueError("Spearman needs at least two weeks")
    se = means.std(ddof=1) / np.sqrt(len(means))
    return float(d.d.mean()), float(means.mean() - student_t.ppf(1 - ALPHA, len(means) - 1) * se)


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, bool]:
    """Holm step-down: reject in ascending p order until the first non-rejection."""
    order = sorted(pvalues, key=pvalues.get)
    k = len(order)
    out = {name: False for name in pvalues}
    for i, name in enumerate(order):
        if pvalues[name] <= alpha / (k - i):
            out[name] = True
        else:
            break
    return out


def evaluate(frame: pd.DataFrame, candidates: list[str], *, cols: Columns = Columns(),
             B: int = 10_000, seed: int = 42, intervals: dict[str, tuple[str, str]] | None = None,
             holm_family: list[str] | None = None) -> dict:
    """Apply G1-G9 to every candidate. Returns per-candidate verdicts and the cells behind them.

    `holm_family` names every candidate in the pre-registered Holm family; it
    defaults to `candidates` and must contain them. `intervals` maps a model
    column to its (lower, upper) 80% interval columns, for G7.
    """
    need = ["player_id", "season", "week", "position", cols.actual, cols.incumbent, cols.rolling3, *candidates]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("duplicate player target-game key")
    if frame[need[4:]].isna().any().any() or not np.isfinite(frame[need[4:]].to_numpy(float)).all():
        raise ValueError("every model must predict every target game: missing or nonfinite values")
    family = list(holm_family or candidates)
    if set(family) != set(candidates):
        raise ValueError("Holm needs every family member's p-value; evaluate the whole family together")

    frame = frame.reset_index(drop=True).copy()
    frame["tier_rolling3"] = assign_tiers(frame, cols.rolling3)
    frame["tier_incumbent"] = assign_tiers(frame, cols.incumbent)
    y = frame[cols.actual].to_numpy(float)
    inc = frame[cols.incumbent].to_numpy(float)
    eb = inc - y
    cells = guardrail_cells(frame)
    empty = [name for name, (_, m) in cells.items() if m is not None and not m.any()]
    if empty:
        raise ValueError(f"guardrail cells with no rows: {empty}")
    errs = {c: frame[c].to_numpy(float) - y for c in candidates}

    # One shared bootstrap: G1 and every guardrail cell, every candidate.
    boot = {c: {"G1": []} | {name: [] for name, (k, _) in cells.items() if k != "spearman"}
            for c in candidates}
    masks = {name: (np.ones(len(y)) if m is None else m.astype(float))
             for name, (k, m) in cells.items() if k != "spearman"}
    for w in _weight_batches(frame.week.to_numpy(), frame.player_id.to_numpy(), B, seed):
        for c in candidates:
            boot[c]["G1"].append(_cell_value("mae", errs[c], eb, w))
            for name, m in masks.items():
                boot[c][name].append(_cell_value(cells[name][0], errs[c], eb, w * m))

    out: dict = {"n_rows": len(frame), "B": B, "seed": seed, "alpha": ALPHA,
                 "min_effect": MIN_EFFECT, "holm_family": family, "candidates": {}}
    pvals = {}
    for c in candidates:
        ec = errs[c]
        ones = np.ones(len(y))
        g1_boot = np.concatenate(boot[c]["G1"])
        g1_point = _cell_value("mae", ec, eb, ones)
        # One-sided H0: candidate is not better (difference >= 0).
        pvals[c] = float((np.sum(g1_boot >= 0) + 1) / (len(g1_boot) + 1))
        guard = {}
        for name, (kind, m) in cells.items():
            if kind == "spearman":
                point, lb = spearman_worsening(frame, frame[c].to_numpy(float), inc, y)
            else:
                point = _cell_value(kind, ec, eb, masks[name])
                values = np.concatenate(boot[c][name])
                if not np.isfinite(values).all():
                    # A NaN bound compares False against 0 and would pass silently.
                    raise ValueError(f"{name}: a bootstrap replicate left the cell empty")
                lb = float(np.quantile(values, ALPHA))
            guard[name] = {"point_worsening": point, "lower_bound": lb, "fails": bool(lb > 0)}
        roll3_mae = float(np.mean(np.abs(frame[cols.rolling3].to_numpy(float) - y)))
        cand_mae = float(np.mean(np.abs(ec)))
        early = frame.week.to_numpy() <= 3
        g7 = None
        if intervals and c in intervals:
            lo, up = (frame[k].to_numpy(float) for k in intervals[c])
            cov = float(np.mean((y >= lo) & (y <= up)))
            g7 = {"coverage80": cov, "passes": COVERAGE_RANGE[0] <= cov <= COVERAGE_RANGE[1]}
        out["candidates"][c] = {
            "mae": cand_mae, "incumbent_mae": float(np.mean(np.abs(eb))),
            "G1": {"point_difference": g1_point, "p_one_sided": pvals[c],
                   "upper_bound_95": float(np.quantile(g1_boot, 1 - ALPHA))},
            "guardrails": guard,
            "guardrail_failures": [k for k, v in guard.items() if v["fails"]],
            "G7": g7,
            "G8_weeks_1_3_report": ({"n": int(early.sum()),
                                     "mae_difference": float(np.mean(np.abs(ec[early]) - np.abs(eb[early])))}
                                    if early.any() else None),
            "G9": {"rolling3_mae": roll3_mae, "passes": cand_mae < roll3_mae},
        }
    rejected = holm(pvals)
    for c in candidates:
        r = out["candidates"][c]
        r["G1"]["holm_rejects"] = rejected[c]
        r["G1"]["passes"] = bool(rejected[c] and -r["G1"]["point_difference"] >= MIN_EFFECT)
        r["guardrails_pass"] = not r["guardrail_failures"]
        r["stage_passes"] = bool(r["G1"]["passes"] and r["G9"]["passes"] and r["guardrails_pass"]
                                 and (r["G7"] is None or r["G7"]["passes"]))
    return out

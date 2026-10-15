#!/usr/bin/env python3
"""Score docs/PREREG_NO_PRIOR_2026.md: the weekly model without the pace prior.

    check   verify every window week written so far against its manifest
            (hash, written before the deadline, not a dry run). Reads no
            outcome, so it can run any week.
    score   once, after week 13's stats are complete: join the window's
            forecasts to raw ten-component full PPR and apply P1, the
            guardrails and S1. Refuses to run early or to overwrite a report.

Inputs, and nothing else: each window week's forecasts.csv (checked against
its manifest sha256), player_weekly_stats, and the pace table pinned in the
forward-test lineage (checked against the lineage sha256).

Candidates (fixed in the pre-registration):
    no_prior_scaled           served_unblended x K
    no_prior_scaled_qb_blend  QB: incumbent_live_blend; others: served_unblended x K
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
from src.evaluation.paired_ppr_comparison import file_sha256
from src.evaluation.selection_gates import ALPHA, _weight_batches, evaluate, holm
from src.utils.helpers import calculate_fantasy_points_df

K = 0.9218
MARGIN = 0.10
SEASON = 2026
WINDOW = range(6, 14)
MAX_EXCLUDED_WEEKS = 1
S1_MIN_GAMES, S1_MIN_GAP = 3, 3.0
CANDIDATES = ["no_prior_scaled", "no_prior_scaled_qb_blend"]
POSITIONS = ("QB", "RB", "WR", "TE")
COMPONENTS = ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds", "receptions",
              "receiving_yards", "receiving_tds", "fumbles_lost", "two_point_conversions"]
FORWARD = ROOT / "data/experiments/selection_forward_2026"
PACE = "data/step8_pace_by_season.csv"
OUTPUT = ROOT / "data/experiments/no_prior_prereg_2026/report.json"


# --- inputs ----------------------------------------------------------------------------------------

def window_weeks(forward: Path = FORWARD) -> tuple[dict[int, Path], dict[int, str]]:
    """Valid window weeks -> forecasts.csv, and excluded weeks -> reason."""
    valid, excluded = {}, {}
    for week in WINDOW:
        d = forward / f"week_{week:02d}"
        if not (d / "manifest.json").exists() or not (d / "forecasts.csv").exists():
            excluded[week] = "not written"
            continue
        m = json.loads((d / "manifest.json").read_text())
        if m.get("dry_run") or m.get("season") != SEASON or m.get("week") != week:
            excluded[week] = "manifest is a dry run or names another week"
        elif not m.get("before_deadline"):
            excluded[week] = "written after the deadline"
        elif file_sha256(d / "forecasts.csv") != m.get("forecasts_sha256"):
            excluded[week] = "forecasts differ from the manifest sha256"
        else:
            valid[week] = d / "forecasts.csv"
    return valid, excluded


def candidate_forecasts(fc: pd.DataFrame) -> pd.DataFrame:
    """Add the two pre-registered candidates to a forecasts frame."""
    out = fc.copy()
    out["no_prior_scaled"] = out.served_unblended * K
    out["no_prior_scaled_qb_blend"] = np.where(out.position.eq("QB"), out.incumbent_live_blend,
                                               out.no_prior_scaled)
    return out


def actual_points(stats: pd.DataFrame) -> pd.DataFrame:
    """Raw ten-component full PPR per stat row, checked against the stored fantasy_points."""
    if stats.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("more than one stat row for a player-week")
    if stats[COMPONENTS].isna().any().any():
        raise ValueError("a scoring component is null")
    out = stats[["player_id", "season", "week"]].copy()
    out["actual"] = calculate_fantasy_points_df(stats[COMPONENTS]).to_numpy(float)
    gap = np.abs(out.actual - pd.to_numeric(stats.fantasy_points, errors="coerce").to_numpy(float))
    if not (gap <= 0.05).all():
        raise ValueError("stored fantasy_points differ from raw ten-component PPR beyond rounding")
    return out


def form_context(frame: pd.DataFrame, stats: pd.DataFrame, pace: pd.DataFrame) -> pd.DataFrame:
    """Games played this season before the week, and season-to-date PPG minus the pace prior."""
    st = stats[["player_id", "week", "fantasy_points"]].sort_values(["player_id", "week"])
    by_player = {p: (g.week.to_numpy(), np.cumsum(g.fantasy_points.to_numpy(float)))
                 for p, g in st.groupby("player_id")}
    games, ppg = [], []
    for p, w in zip(frame.player_id, frame.week):
        weeks, cum = by_player.get(p, (np.array([]), np.array([])))
        i = int(np.searchsorted(weeks, w))           # games strictly before week w
        games.append(i)
        ppg.append(cum[i - 1] / i if i else np.nan)
    out = frame.copy()
    out["games_before"] = games
    prior = pace.drop_duplicates("player_id").set_index("player_id").step8_pace
    out["form_gap"] = np.asarray(ppg) - out.player_id.map(prior).to_numpy(float)
    return out


# --- scoring ---------------------------------------------------------------------------------------

def _boot_means(W: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    w = W if mask is None else W * mask
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ v) / w.sum(1)


def score(frame: pd.DataFrame, *, B: int = 10_000, seed: int = 42) -> dict:
    """P1, guardrails, S1 and the report-only measures. `frame` holds one row per scored
    player-week with actual, incumbent, rolling3, the candidates, games_before and form_gap."""
    frame = frame.sort_values(["season", "week", "player_id"]).reset_index(drop=True)
    gates = evaluate(frame, CANDIDATES, B=B, seed=seed)
    # The same draws evaluate() uses: the sequence depends only on seed and row order.
    W = np.vstack(list(_weight_batches(frame.week.to_numpy(), frame.player_id.to_numpy(), B, seed)))
    y = frame.actual.to_numpy(float)
    inc = frame.incumbent.to_numpy(float)
    s1_mask = ((frame.games_before >= S1_MIN_GAMES) & (frame.form_gap >= S1_MIN_GAP)).to_numpy(float)
    lag_rows = ((frame.games_before >= S1_MIN_GAMES) & frame.form_gap.notna()).to_numpy()

    out = {"n_rows": len(frame), "weeks": sorted(int(w) for w in frame.week.unique()), "B": B, "seed": seed,
           "alpha": ALPHA, "margin": MARGIN, "K": K, "candidates": {}}
    pvals = {}
    for c in CANDIDATES:
        p = frame[c].to_numpy(float)
        d = np.abs(p - y) - np.abs(inc - y)
        boot = _boot_means(W, d)
        pvals[c] = float((np.sum(boot >= MARGIN) + 1) / (B + 1))
        n_s1 = int(s1_mask.sum())
        if n_s1:
            cut = np.abs(np.mean((inc - y)[s1_mask == 1])) - np.abs(np.mean((p - y)[s1_mask == 1]))
            cut_boot = (np.abs(_boot_means(W, inc - y, s1_mask)) - np.abs(_boot_means(W, p - y, s1_mask)))
            s1 = {"n": n_s1, "lean_incumbent": float(np.mean((inc - y)[s1_mask == 1])),
                  "lean_candidate": float(np.mean((p - y)[s1_mask == 1])), "abs_lean_cut": float(cut),
                  "lower_bound_95": float(np.nanquantile(cut_boot, ALPHA)),
                  "empty_draws": int(np.isnan(cut_boot).sum()), "passes": bool(cut > 0)}
        else:
            s1 = {"n": 0, "passes": False}
        g = gates["candidates"][c]
        weeks = frame.week.to_numpy()
        out["candidates"][c] = {
            "P1": {"mae": float(np.abs(p - y).mean()), "incumbent_mae": float(np.abs(inc - y).mean()),
                   "point_difference": float(d.mean()), "upper_bound_95": float(np.quantile(boot, 1 - ALPHA)),
                   "p_noninferiority": pvals[c]},
            "guardrails": g["guardrails"], "guardrail_failures": g["guardrail_failures"],
            "S1": s1,
            "report_only": {
                "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "incumbent_rmse": float(np.sqrt(np.mean((inc - y) ** 2))),
                "mae_by_position": {pos: [float(np.abs(p - y)[m].mean()), float(np.abs(inc - y)[m].mean())]
                                    for pos in POSITIONS if (m := frame.position.eq(pos).to_numpy()).any()},
                "mae_weeks_6_9": _pair(p, inc, y, weeks <= 9), "mae_weeks_10_13": _pair(p, inc, y, weeks >= 10),
                "form_lag_slope": {"candidate": _slope(frame.form_gap, y - p, lag_rows),
                                   "incumbent": _slope(frame.form_gap, y - inc, lag_rows)},
                "pair_ordering": {"candidate": pair_ordering(frame, c), "incumbent": pair_ordering(frame, "incumbent")},
                "superiority_G1_report": g["G1"], "G9_beats_rolling3": g["G9"],
            },
        }
    rejected = holm(pvals)
    for c in CANDIDATES:
        r = out["candidates"][c]
        r["P1"]["holm_rejects"] = rejected[c]
        r["eligible"] = bool(rejected[c] and not r["guardrail_failures"] and r["S1"]["passes"])
    eligible = [c for c in CANDIDATES if out["candidates"][c]["eligible"]]
    out["decision"] = eligible[0] if eligible else "incumbent_live_blend stays"   # CANDIDATES lists the simpler first
    return out


def _pair(p, inc, y, m) -> list[float] | None:
    return [float(np.abs(p - y)[m].mean()), float(np.abs(inc - y)[m].mean())] if m.any() else None


def _slope(x: pd.Series, r: np.ndarray, m: np.ndarray) -> float | None:
    return float(np.polyfit(x.to_numpy(float)[m], r[m], 1)[0]) if m.sum() >= 3 else None


def pair_ordering(frame: pd.DataFrame, col: str) -> float:
    """Share of same week and position pairs the higher forecast ordered correctly (ties in actual = half)."""
    hit = tot = 0.0
    for _, g in frame.groupby(["week", "position"]):
        p, a = g[col].to_numpy(float), g.actual.to_numpy(float)
        i, j = np.triu_indices(len(g), 1)
        k = p[i] != p[j]
        lead = np.where(p[i] > p[j], a[i] - a[j], a[j] - a[i])[k]
        hit += (lead > 0).sum() + 0.5 * (lead == 0).sum()
        tot += k.sum()
    return float(hit / tot) if tot else float("nan")


# --- commands --------------------------------------------------------------------------------------

def stats_complete(con: sqlite3.Connection, week: int) -> bool:
    teams = con.execute("SELECT COUNT(*) * 2 FROM schedule WHERE season = ? AND week = ?", (SEASON, week)).fetchone()[0]
    have = con.execute("SELECT COUNT(DISTINCT team) FROM player_weekly_stats WHERE season = ? AND week = ?",
                       (SEASON, week)).fetchone()[0]
    return bool(teams) and have >= teams


def build_frame(valid: dict[int, Path], con: sqlite3.Connection) -> pd.DataFrame:
    lineage = json.loads((FORWARD / "lineage.json").read_text())
    if file_sha256(ROOT / PACE) != lineage["served"]["files_sha256"][PACE]:
        raise SystemExit("pace table differs from the forward-test lineage")
    pace = pd.read_csv(ROOT / PACE, dtype={"player_id": str})
    pace = pace[pace.season == SEASON]
    fc = pd.concat([pd.read_csv(p, dtype={"player_id": str}) for p in valid.values()], ignore_index=True)
    fc = candidate_forecasts(fc[fc.position.isin(POSITIONS)])
    stats = pd.read_sql("SELECT * FROM player_weekly_stats WHERE season = ? AND week <= ?", con,
                        params=(SEASON, max(WINDOW)))
    act = actual_points(stats[stats.week.isin(list(valid))])
    frame = fc.merge(act, on=["player_id", "season", "week"], how="inner", validate="one_to_one")
    frame = frame.rename(columns={"incumbent_live_blend": "incumbent"})
    return form_context(frame, stats, pace)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "score"])
    ap.add_argument("--output", type=Path, default=OUTPUT)
    a = ap.parse_args()
    valid, excluded = window_weeks()
    if a.cmd == "check":
        print(json.dumps({"valid_weeks": sorted(valid), "excluded": excluded}, indent=2))
        return
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    if not stats_complete(con, max(WINDOW)):
        raise SystemExit(f"week {max(WINDOW)} stats are not complete; the window is scored once, after it")
    if len(excluded) > MAX_EXCLUDED_WEEKS:
        report = {"void": True, "excluded": excluded}
    else:
        report = score(build_frame(valid, con)) | {"void": False, "excluded": excluded}
    report["script_sha256"] = file_sha256(Path(__file__))
    report["forecasts_sha256"] = {int(w): file_sha256(p) for w, p in valid.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("void", "excluded")} | {"decision": report.get("decision")}, indent=2))


if __name__ == "__main__":
    main()

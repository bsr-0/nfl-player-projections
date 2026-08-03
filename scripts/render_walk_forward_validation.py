#!/usr/bin/env python3
"""Render walk-forward validation markdown for the GitHub Actions summary.

Reads the cached leakage-free predictions in
``data/backtest_results/ts_backtest_{season}_*_predictions.csv`` and emits:

1. A condensed markdown summary to ``$GITHUB_STEP_SUMMARY`` (or stdout if
   that env var is unset).  Each season gets a section, each week gets a
   collapsible <details> block with the top-N players by predicted points.
2. A full markdown report to ``walk-forward-validation.md`` (every player,
   every week — uploaded as a workflow artifact).

The cached CSVs are the output of ``scripts/run_ts_backtest.py``, which runs
the expanding-window leakage-safe backtester in
``src/evaluation/ts_backtester.py``.  This script does not retrain — it
reads what's already on disk.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"
SEASONS = list(range(2018, 2026))
TOP_N_PER_WEEK_SUMMARY = 25
ARTIFACT_PATH = Path(os.environ.get("WALK_FORWARD_OUT", "walk-forward-validation.md"))


def latest_predictions_csv(season: int) -> Optional[Path]:
    """Pick the newest ts_backtest_{season}_*_predictions.csv by filename
    timestamp (filename mtime is unreliable — a recent batch ``touch``
    flattened all mtimes to a single second)."""
    candidates = [
        c for c in RESULTS_DIR.glob(f"ts_backtest_{season}_*_predictions.csv")
        if "_conformal" not in c.name
    ]
    if not candidates:
        return None

    def ts_key(p: Path) -> str:
        # ts_backtest_{season}_{YYYYMMDD}_{HHMMSS}_predictions.csv
        # parts: ['ts', 'backtest', '{season}', '{YYYYMMDD}', '{HHMMSS}', 'predictions']
        parts = p.stem.split("_")
        if len(parts) >= 5:
            return parts[3] + parts[4]
        return p.name

    return max(candidates, key=ts_key)


def load_rows(csv_path: Path, season: int) -> List[Dict]:
    rows: List[Dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("season", -1)) != season:
                    continue
            except (TypeError, ValueError):
                continue
            if str(r.get("is_active", "1")).lower() in ("0", "false"):
                continue
            pred = r.get("predicted")
            actual = r.get("actual")
            if not pred or actual in (None, "", "nan"):
                continue
            try:
                rows.append({
                    "week": int(r["week"]),
                    "name": (r.get("name") or "").strip(),
                    "position": (r.get("position") or "").strip(),
                    "team": (r.get("team") or "").strip(),
                    "predicted": float(pred),
                    "actual": float(actual),
                })
            except (TypeError, ValueError):
                continue
    return rows


def metrics(rows: List[Dict]) -> Dict[str, Optional[float]]:
    if not rows:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    n = len(rows)
    errs = [r["actual"] - r["predicted"] for r in rows]
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    mean_actual = sum(r["actual"] for r in rows) / n
    ss_res = sum(e * e for e in errs)
    ss_tot = sum((r["actual"] - mean_actual) ** 2 for r in rows) or 1.0
    r2 = 1 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


def fmt_metrics(m: Dict[str, Optional[float]]) -> str:
    if not m["n"]:
        return "n=0"
    return f"n={m['n']} · MAE={m['mae']:.2f} · RMSE={m['rmse']:.2f} · R²={m['r2']:.3f}"


def render_week(week: int, rows: List[Dict], top_n: Optional[int]) -> str:
    rows_sorted = sorted(rows, key=lambda r: r["predicted"], reverse=True)
    display = rows_sorted if top_n is None else rows_sorted[:top_n]
    m = metrics(rows)  # metrics over the full week, not just displayed
    n_total = len(rows_sorted)
    truncated = top_n is not None and n_total > top_n
    suffix = f" · showing top {top_n} of {n_total}" if truncated else ""

    lines = [
        f"<details><summary><b>Week {week}</b> · {fmt_metrics(m)}{suffix}</summary>",
        "",
        "| Player | Pos | Team | Predicted | Actual | Residual |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in display:
        resid = r["actual"] - r["predicted"]
        lines.append(
            f"| {r['name']} | {r['position']} | {r['team']} | "
            f"{r['predicted']:.2f} | {r['actual']:.2f} | {resid:+.2f} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def render_season(season: int, rows: List[Dict], top_n_per_week: Optional[int]) -> str:
    by_week: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_week[r["week"]].append(r)
    weeks = sorted(by_week.keys())
    overall = metrics(rows)

    by_pos: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_pos[r["position"]].append(r)

    out = [f"## Season {season}", ""]
    if not rows:
        out.append("_No predictions found for this season._")
        return "\n".join(out)

    out.append(f"Weeks {weeks[0]}–{weeks[-1]} · {fmt_metrics(overall)}")
    out.append("")
    out.append("**By position**")
    out.append("")
    out.append("| Position | n | MAE | RMSE | R² |")
    out.append("|---|---:|---:|---:|---:|")
    for pos in ["QB", "RB", "WR", "TE"]:
        if pos in by_pos:
            m = metrics(by_pos[pos])
            out.append(
                f"| {pos} | {m['n']} | {m['mae']:.2f} | "
                f"{m['rmse']:.2f} | {m['r2']:.3f} |"
            )
    out.append("")
    for wk in weeks:
        out.append(render_week(wk, by_week[wk], top_n=top_n_per_week))
    return "\n".join(out)


def render(top_n_per_week: Optional[int], header_note: str) -> str:
    parts = ["# Walk-forward Validation", "", header_note, ""]

    # Headline table
    parts.append("## Headline")
    parts.append("")
    parts.append("| Season | n | MAE | RMSE | R² |")
    parts.append("|---|---:|---:|---:|---:|")
    season_rows: Dict[int, List[Dict]] = {}
    for season in SEASONS:
        csv_path = latest_predictions_csv(season)
        if not csv_path:
            parts.append(f"| {season} | _no predictions on disk_ | | | |")
            continue
        rows = load_rows(csv_path, season)
        season_rows[season] = rows
        m = metrics(rows)
        if m["n"]:
            parts.append(
                f"| {season} | {m['n']} | {m['mae']:.2f} | "
                f"{m['rmse']:.2f} | {m['r2']:.3f} |"
            )
        else:
            parts.append(f"| {season} | 0 | | | |")
    parts.append("")

    for season in SEASONS:
        if season not in season_rows:
            continue
        parts.append(render_season(season, season_rows[season], top_n_per_week))

    return "\n".join(parts)


def main() -> int:
    summary_note = (
        "Per-season expanding-window predictions vs actuals. Each row is "
        "what the model predicted using only data available **before** that "
        "week (leakage-safe pipeline, `src/evaluation/ts_backtester.py`). "
        f"Showing top {TOP_N_PER_WEEK_SUMMARY} by predicted per week here; "
        "full per-player tables are in the `walk-forward-validation` "
        "artifact attached to this run."
    )
    full_note = (
        "Every player-week prediction from the leakage-free expanding-window "
        "backtest, 2018–2025. Unfiltered."
    )

    summary_md = render(top_n_per_week=TOP_N_PER_WEEK_SUMMARY, header_note=summary_note)
    full_md = render(top_n_per_week=None, header_note=full_note)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        Path(step_summary).write_text(summary_md)
        print(f"Wrote step summary: {step_summary} ({len(summary_md):,} bytes)",
              file=sys.stderr)
    else:
        sys.stdout.write(summary_md)

    ARTIFACT_PATH.write_text(full_md)
    print(f"Wrote artifact: {ARTIFACT_PATH} ({len(full_md):,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

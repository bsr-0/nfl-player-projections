#!/usr/bin/env python3
"""Generate data files for the static 2025 validation / 2026 projections page.

Reads the latest 2025 walk-forward backtest, data/model_performance.json, and
the current data/players_{POS}.json projection files, and writes two trimmed
JSON files consumed by docs/index.html:

    docs/data/backtest_2025.json
    docs/data/projections_2026.json

Usage:
    python scripts/generate_results_page.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_DIR = DATA_DIR / "backtest_results"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"


def copy_board_data() -> None:
    """Copy players_{POS}.json into docs/data/.

    docs/draft.html fetches these, and GitHub Pages publishes only the docs/
    tree -- a relative ../data/ path resolves outside the publish root and
    404s in production while working fine locally. Copying is deliberate over
    rewriting the path: data/players_*.json stays the canonical artifact that
    scripts read, and docs/data/ is the published mirror.
    """
    import shutil
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for pos in POSITIONS:
        src = DATA_DIR / f"players_{pos}.json"
        if src.exists():
            shutil.copy2(src, DOCS_DATA_DIR / src.name)
            print(f"Copied {src.name} -> docs/data/")


POSITIONS = ["QB", "RB", "WR", "TE"]


def latest_backtest_2025() -> Path:
    candidates = sorted(BACKTEST_DIR.glob("ts_backtest_2025_*.json"))
    candidates = [c for c in candidates if not c.name.endswith("_predictions.csv")]
    if not candidates:
        raise FileNotFoundError("No ts_backtest_2025_*.json files found")
    return candidates[-1]


def build_backtest_summary() -> dict:
    path = latest_backtest_2025()
    raw = json.loads(path.read_text())

    dq = raw.get("decision_quality", {})
    weekly = dq.get("weekly_results", [])

    return {
        "source_file": path.name,
        "backtest_date": raw.get("backtest_date"),
        "season": raw.get("season"),
        "n_predictions": raw.get("n_predictions"),
        "overall": raw.get("metrics", {}),
        "by_position": raw.get("by_position", {}),
        "baselines": raw.get("baselines", {}),
        "decision_quality": {
            "n_weeks": dq.get("n_weeks"),
            "avg_model_score": dq.get("avg_model_score"),
            "avg_projection_error": dq.get("avg_projection_error"),
            "vs_oracle": dq.get("vs_oracle", {}),
            "vs_hindsight": dq.get("vs_hindsight", {}),
            "vs_replacement": dq.get("vs_replacement", {}),
            "weekly_results": [
                {
                    "week": w.get("week"),
                    "model_actual": w.get("model_actual"),
                    "oracle_actual": w.get("oracle_actual"),
                    "hindsight_actual": w.get("hindsight_actual"),
                    "replacement_actual": w.get("replacement_actual"),
                }
                for w in weekly
            ],
        },
    }


def build_prediction_accuracy() -> dict:
    perf_path = DATA_DIR / "model_performance.json"
    perf = json.loads(perf_path.read_text())
    players = perf.get("per_player_season_totals", [])
    qualified = [p for p in players if (p.get("games") or 0) >= 8]

    over = sorted(qualified, key=lambda p: p["error"])[:15]  # most negative = model underestimated
    under = sorted(qualified, key=lambda p: -p["error"])[:15]  # most positive = model overestimated

    return {
        "aggregate_metrics": perf.get("aggregate_metrics", {}),
        "by_position": perf.get("by_position", {}),
        "biggest_underestimates": over,
        "biggest_overestimates": under,
    }


def build_2026_projections() -> list[dict]:
    combined = []
    for pos in POSITIONS:
        path = DATA_DIR / f"players_{pos}.json"
        if not path.exists():
            continue
        players = json.loads(path.read_text())
        for p in players:
            combined.append({
                "name": p.get("name"),
                "team": p.get("team"),
                "position": p.get("position", pos),
                "adp": p.get("adp"),
                "projection_total": p.get("projection_points_total"),
                "projection_ppg": p.get("projection_points_per_game"),
                "floor": p.get("projection_floor"),
                "ceiling": p.get("projection_ceiling"),
                "risk_score": p.get("risk_score"),
                "prev_season_ppg": p.get("prev_season_ppg"),
            })
    _attach_vor(combined)
    combined.sort(key=lambda p: p["vor"] if p["vor"] is not None else -1e9, reverse=True)
    return combined


# Standard 12-team league with 1 QB / 2 RB / 3 WR / 1 TE. These defaults must
# match docs/draft.html's initial control values, or the two pages disagree
# about who is best at first load.
VOR_TEAMS = 12
VOR_STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


def _attach_vor(rows: list[dict]) -> None:
    """Value over replacement, computed once in the data.

    Ranking a draft board by raw projected points puts six quarterbacks in the
    top ten. That is accurate scoring and bad advice: you start one QB, so the
    13th-best is nearly as good as the 5th, while the 25th-best RB is far
    worse. Replacement level is the last startable player at each position.

    Computed HERE rather than in each page so the projections table and the
    draft board cannot drift apart -- two copies of the same logic kept in sync
    by hand is the failure mode that produced the is_power5 and TE-architecture
    bugs. docs/draft.html recomputes it client-side only because its league
    settings are adjustable; at the defaults above it must agree with this.
    """
    repl: dict[str, float] = {}
    for pos, n_start in VOR_STARTERS.items():
        vals = sorted((r["projection_total"] for r in rows
                       if r["position"] == pos and r["projection_total"] is not None),
                      reverse=True)
        if not vals:
            repl[pos] = 0.0
            continue
        idx = min(max(VOR_TEAMS * n_start - 1, 0), len(vals) - 1)
        repl[pos] = vals[idx]
    for r in rows:
        tot = r["projection_total"]
        r["vor"] = None if tot is None else round(tot - repl.get(r["position"], 0.0), 1)
    return None


def main() -> None:
    copy_board_data()
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    backtest_summary = build_backtest_summary()
    backtest_summary["prediction_accuracy"] = build_prediction_accuracy()
    (DOCS_DATA_DIR / "backtest_2025.json").write_text(
        json.dumps(backtest_summary, indent=2)
    )

    projections = build_2026_projections()
    (DOCS_DATA_DIR / "projections_2026.json").write_text(
        json.dumps(projections, indent=2)
    )

    print(f"Wrote {DOCS_DATA_DIR / 'backtest_2025.json'} "
          f"({backtest_summary['source_file']})")
    print(f"Wrote {DOCS_DATA_DIR / 'projections_2026.json'} "
          f"({len(projections)} players)")


if __name__ == "__main__":
    main()

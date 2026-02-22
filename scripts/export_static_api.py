#!/usr/bin/env python3
"""
Export a lightweight static JSON snapshot for GitHub Pages.

Writes JSON files to frontend/public/api/ by default so Vite can bundle them.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _top_n_per_position(rows: List[Dict[str, Any]], n: int = 250) -> List[Dict[str, Any]]:
    by_pos: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pos = str(r.get("position") or "").upper()
        by_pos.setdefault(pos, []).append(r)

    out: List[Dict[str, Any]] = []
    for pos, items in by_pos.items():
        def sort_key(x: Dict[str, Any]) -> float:
            v = x.get("projection_1w")
            if v is None:
                v = x.get("predicted_points")
            try:
                return float(v) if v is not None else float("-inf")
            except Exception:
                return float("-inf")

        items = sorted(items, key=sort_key, reverse=True)
        out.extend(items[:n])
    return out


def _pick_latest_season(backtest_payload: Dict[str, Any]) -> Optional[int]:
    latest = backtest_payload.get("latest", {}) if backtest_payload else {}
    season = latest.get("season")
    if season is not None:
        return season
    files = backtest_payload.get("files", []) if backtest_payload else []
    seasons = [f.get("season") for f in files if f.get("season") is not None]
    return max(seasons) if seasons else None


def _pick_latest_ts_season(ts_payload: Dict[str, Any]) -> Optional[int]:
    seasons = ts_payload.get("available_seasons", []) if ts_payload else []
    seasons = [s for s in seasons if isinstance(s, int)]
    return max(seasons) if seasons else None


def export_static_api(out_dir: Path, eda_max_rows: int = 2000) -> None:
    from api.main import (
        get_hero,
        get_data_pipeline,
        get_training_years,
        get_eda,
        get_model_config,
        get_advanced_results,
        get_backtest,
        get_ts_backtest,
        _get_predictions_impl,
        get_backtest_season,
        get_ts_backtest_season,
        get_ts_backtest_predictions,
    )
    from src.app_data import get_utilization_weights_merged

    hero = get_hero()
    data_pipeline = get_data_pipeline()
    training_years = get_training_years()
    eda = get_eda(max_rows=eda_max_rows)
    utilization_weights = get_utilization_weights_merged()
    model_config = get_model_config()
    advanced_results = get_advanced_results()
    backtest = get_backtest()
    ts_backtest = get_ts_backtest()
    predictions = _get_predictions_impl(horizon="all")

    if predictions and predictions.get("rows"):
        predictions["rows"] = _top_n_per_position(predictions["rows"], n=250)

    _write_json(out_dir / "hero.json", hero)
    _write_json(out_dir / "data-pipeline.json", data_pipeline)
    _write_json(out_dir / "training-years.json", training_years)
    _write_json(out_dir / "eda.json", eda)
    _write_json(out_dir / "utilization-weights.json", utilization_weights)
    _write_json(out_dir / "model-config.json", model_config)
    _write_json(out_dir / "advanced-results.json", advanced_results)
    _write_json(out_dir / "backtest.json", backtest)
    _write_json(out_dir / "predictions.json", predictions)
    _write_json(out_dir / "ts-backtest.json", ts_backtest)

    latest_backtest_season = _pick_latest_season(backtest)
    if latest_backtest_season is not None:
        _write_json(
            out_dir / f"backtest-{latest_backtest_season}.json",
            get_backtest_season(latest_backtest_season),
        )

    latest_ts_season = _pick_latest_ts_season(ts_backtest)
    if latest_ts_season is not None:
        _write_json(
            out_dir / f"ts-backtest-{latest_ts_season}.json",
            get_ts_backtest_season(latest_ts_season),
        )
        ts_preds = get_ts_backtest_predictions(latest_ts_season, position=None, week=None)
        rows = ts_preds.get("rows", []) if isinstance(ts_preds, dict) else []
        if isinstance(rows, list) and len(rows) > 2000:
            ts_preds["rows"] = rows[:2000]
        _write_json(
            out_dir / f"ts-backtest-{latest_ts_season}-predictions.json",
            ts_preds,
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export static JSON snapshot for GH Pages")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "frontend" / "public" / "api"),
        help="Output directory for JSON files (default: frontend/public/api)",
    )
    parser.add_argument("--eda-max-rows", type=int, default=2000, help="Max rows for EDA sample")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    export_static_api(out_dir, eda_max_rows=args.eda_max_rows)
    print(f"Static API snapshot written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

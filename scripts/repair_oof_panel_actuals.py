"""Replace winsorized actuals in an OOF panel with raw target-game points.

Until 2026-10-07 `_prepare_training_data` clipped TEST targets to per-position
training 1st/99th percentiles, so every OOF panel's `actual_points` (and its
residuals) stopped at those bounds. Predictions never depended on that clip,
so the panel is repairable without refitting: look up each row's target game
(player_id, season, target_week) in the raw weekly stats and rescore.

The repair is strict. Within each fold (test season) and position the clip
bounds are the captured actuals' own min/max, so a row may differ from raw
only when it sits at one of those extremes with the raw value beyond it. Any
other difference means the panel is not what this assumes and the repair
stops. The source panel is never modified; a new run directory is written
beside it with provenance pointing back to it.

    python scripts/repair_oof_panel_actuals.py \\
        --panel-dir data/experiments/oof_panels/<run_id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.oof_capture import ZERO_FLOOR, segment_report  # noqa: E402
from src.utils.atomic_io import atomic_write_json, atomic_write_parquet  # noqa: E402

TOL = 1e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_target_points(panel: pd.DataFrame, db_path: Path) -> np.ndarray:
    """Raw fantasy_points of each row's target game, read-only from the DB."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        seasons = sorted(int(s) for s in panel.season.unique())
        stats = pd.read_sql(
            "SELECT player_id, season, week, fantasy_points FROM player_weekly_stats "
            f"WHERE season IN ({','.join('?' * len(seasons))})", con, params=seasons)
    finally:
        con.close()
    if stats.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("player_weekly_stats has duplicate player-season-week rows")
    keys = panel[["player_id", "season", "target_week"]].rename(columns={"target_week": "week"})
    keys = keys.astype({"player_id": str, "season": int, "week": int})
    stats = stats.astype({"player_id": str, "season": int, "week": int})
    joined = keys.merge(stats, on=["player_id", "season", "week"], how="left", validate="many_to_one")
    if len(joined) != len(panel) or joined.fantasy_points.isna().any():
        raise ValueError(f"{int(joined.fantasy_points.isna().sum())} target games lack raw stats")
    return joined.fantasy_points.to_numpy(float)


def repaired(panel: pd.DataFrame, raw: np.ndarray) -> tuple[pd.DataFrame, dict]:
    captured = panel.actual_points.to_numpy(float)
    group = panel.groupby(["season", "position"]).actual_points
    lo = group.transform("min").to_numpy(float)
    hi = group.transform("max").to_numpy(float)
    differs = np.abs(captured - raw) > TOL
    at_high = (np.abs(captured - hi) <= TOL) & (raw > captured)
    at_low = (np.abs(captured - lo) <= TOL) & (raw < captured)
    unexplained = differs & ~(at_high | at_low)
    if unexplained.any():
        sample = panel.loc[unexplained, ["player_id", "season", "week", "target_week",
                                         "position", "actual_points"]].head(5)
        raise ValueError(f"{int(unexplained.sum())} rows differ from raw without being "
                         f"clipped:\n{sample.assign(raw=raw[unexplained][:5])}")
    out = panel.copy()
    out["actual_points"] = raw
    out["residual"] = out.predicted_points.to_numpy(float) - raw
    by_group = (panel.assign(clipped=differs).groupby(["season", "position"]).clipped.sum()
                .astype(int).reset_index().to_dict("records"))
    summary = {"rows": int(len(panel)), "rows_repaired": int(differs.sum()),
               "repaired_high": int((differs & at_high).sum()),
               "repaired_low": int((differs & at_low).sum()),
               "max_abs_change": float(np.abs(captured - raw).max()),
               "by_season_position": by_group}
    return out, summary


def write_segment_reports(panel: pd.DataFrame, run_dir: Path) -> None:
    """Recompute the segment reports train.py writes beside a panel; the old
    ones scored clipped actuals. verify_oof_panel.py rechecks every one."""
    frame = panel.copy()
    frame["all_rows"] = "all"
    frame["actual_activity"] = np.where(frame["actual_points"] > ZERO_FLOOR,
                                        "nonzero_actual", "near_zero_actual")
    specs = [("segment_report.csv", ("season",)),
             ("total_segment_report.csv", ("all_rows",)),
             ("position_segment_report.csv", ("position",)),
             ("experience_segment_report.csv", ("position", "is_cold_start", "week_bucket")),
             ("position_season_segment_report.csv", ("position", "season")),
             ("actual_activity_segment_report.csv", ("actual_activity",))]
    if "is_cold_start_career" in panel.columns and panel["is_cold_start_career"].notna().any():
        specs.append(("career_segment_report.csv", ("position", "is_cold_start_career")))
    for filename, by in specs:
        segment_report(frame, by=by, cluster="player_id").to_csv(run_dir / filename, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "nfl_data.db")
    args = parser.parse_args()
    source = args.panel_dir
    manifest = json.loads((source / "manifest.json").read_text())
    panel_path = source / "panel.parquet"
    if _sha256(panel_path) != manifest["panel_sha256"]:
        raise SystemExit("source panel hash does not match its manifest")
    if manifest.get("actuals_repair"):
        raise SystemExit("source panel was already repaired")
    panel = pd.read_parquet(panel_path)
    try:
        fixed, summary = repaired(panel, raw_target_points(panel, args.db))
    except ValueError as exc:
        raise SystemExit(f"repair stopped: {exc}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = f"{manifest['label']}_raw_actuals"
    run_dir = source.parent / f"{stamp}_{manifest.get('git_commit') or 'nogit'}_{label}"
    run_dir.mkdir(parents=True)
    out_path = atomic_write_parquet(fixed, run_dir / "panel.parquet")
    if (source / "coverage.json").exists():
        shutil.copy(source / "coverage.json", run_dir / "coverage.json")
    new_manifest = dict(manifest)
    new_manifest.update({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": label, "panel_sha256": _sha256(out_path),
        "actuals_repair": {
            "reason": "test targets were winsorized before 2026-10-07; actual_points "
                      "and residual rescored from raw target-game fantasy_points",
            "source_run": source.name, "source_panel_sha256": manifest["panel_sha256"],
            "raw_source": "player_weekly_stats.fantasy_points (read-only)",
            **summary,
        },
    })
    write_segment_reports(fixed, run_dir)
    atomic_write_json(new_manifest, run_dir / "manifest.json")
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_oof_panel", ROOT / "scripts" / "verify_oof_panel.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify(run_dir)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

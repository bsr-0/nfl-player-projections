"""Emit per-week projections for docs/weekly.html.

ONE PATH for every week, before and after kickoff: the weekly ensemble via
predict(as_of=(season, week)), which shrinks its number toward the Step 8
season pace by games played this season (src/predict.py
_blend_toward_season_pace, config.settings.PACE_BLEND_KAPPA):

    served = w * weekly + (1 - w) * pace,   w = g / (g + 3)

Before any game is played g = 0 for everyone and the served number IS the
season total / 17 the page used to publish as a separate "season_prorated"
mode. That mode existed because the weekly model cannot differentiate weeks
before kickoff (opponent stats and betting lines do not exist yet; weeks 1
and 2 were byte-identical for all 795 players, 2026-08-31) and its week-1
number was worse than the pace (2025 walk-forward, 0 games played: MAE 4.47
vs 4.00). The blend keeps that property at g = 0 and glides into the weekly
model as games accumulate; the mode switch is gone.

Measured on the 2025 serving-path walk-forward (scripts/
run_pace_blend_experiment.py, 2026-09-17): MAE 4.467 -> 4.354 out-of-fold,
paired-bootstrap CI95 on the difference [-0.156, -0.063], gains in every
games-played bucket and every position. `/ 17` itself was the measured
cold-start choice (2026-09-05): on 208 no-history players over 2021-2025,
MAE 3.70, bias -0.18, beating four alternatives -- unchanged here, since at
g = 0 the blend reproduces it exactly for every player with a Step 8 row.

Rows carry an 80% interval from the weekly model's conformal width, shifted
to the blended centre (conservative: the blend's errors are smaller), plus
`pace_weight` so a reader can see how much of a number is pace.

Usage:
    python scripts/generate_weekly_data.py
    python scripts/generate_weekly_data.py --weeks 1 6
"""
import argparse
import json
import sqlite3
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH, PACE_BLEND_KAPPA

OUT_DIR = Path("docs/data")

BACKTEST_DIR = Path("data/backtest_results")


def measured_accuracy() -> dict | None:
    """Per-position accuracy read from the latest TRUSTED, full-season
    serving-path backtest -- the artifact that scores the model this page
    actually serves, week by week, against what those players really did.

    This used to be a hardcoded dict (QB 7.09 / RB 4.43 / WR 4.03 / TE 3.00,
    with a -1.2 to -1.6 point "runs low" bias) copied from a three-week spot
    check whose artifact no longer exists -- AUDIT_REPORT.md #16. Two things
    made those numbers wrong to keep publishing: they were never
    reproducible, and the low bias they described was largely the blanket
    injury discount that used to be multiplied into every projection, which
    is gone (availability is reported separately now). Returns None when no
    trusted artifact exists, so the page can say nothing rather than assert
    a stale number.
    """
    for path in sorted(BACKTEST_DIR.glob("backtest_*_*.json"), reverse=True):
        if "UNTRUSTED" in path.name or "PARTIAL" in path.name:
            continue
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if raw.get("backtest_path") != "serving_path_as_of_walk_forward":
            continue
        if raw.get("trust", {}).get("trusted", True) is False or raw.get("partial_season"):
            continue
        by_pos = raw.get("by_position") or {}
        out = {}
        for pos, m in by_pos.items():
            if m.get("mae") is None or m.get("avg_predicted") is None or m.get("avg_actual") is None:
                continue
            out[pos] = {
                "mae": round(float(m["mae"]), 2),
                "bias": round(float(m["avg_predicted"]) - float(m["avg_actual"]), 2),
            }
        if out:
            out["_source"] = {
                "file": path.name,
                "season": raw.get("season"),
                "weeks": len(raw.get("weeks_evaluated") or []),
                "model_type": raw.get("model_type"),
                "feature_version": raw.get("feature_version"),
            }
            return out
    return None


def baseline_standing() -> dict | None:
    """How the served model actually compares to the naive baselines, from
    the same trusted artifact.

    The fix order's instruction was "until it wins, keep serving Step 8 and
    SAY SO" -- this is the say-so, published rather than left in a JSON file
    nobody reads. The honest current answer is a narrow win: the model beats
    a trailing-3-game average by a few percent and season-average by less,
    which is not the same as being decisively better, and the page should
    not imply otherwise.
    """
    for path in sorted(BACKTEST_DIR.glob("backtest_*_*.json"), reverse=True):
        if "UNTRUSTED" in path.name or "PARTIAL" in path.name:
            continue
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if raw.get("backtest_path") != "serving_path_as_of_walk_forward":
            continue
        if raw.get("trust", {}).get("trusted", True) is False or raw.get("partial_season"):
            continue
        strong = raw.get("strong_baseline_comparison") or {}
        if not strong:
            continue
        criteria = raw.get("success_criteria") or {}
        return {
            "season": raw.get("season"),
            "rmse_improvement_pct": {
                k: v.get("rmse_improvement_pct") for k, v in strong.items()
                if isinstance(v, dict)
            },
            "beats_every_baseline": all(
                (v.get("rmse_improvement_pct") or 0) > 0 for v in strong.values()
                if isinstance(v, dict)
            ),
            "beat_all_baselines_by_20_pct": criteria.get("beat_all_baselines_by_20_pct"),
            "model_has_real_edge": criteria.get("model_has_real_edge"),
        }
    return None

KEEP = ["player_id", "name", "position", "team", "opponent", "home_away",
        "predicted_points", "prediction_ci80_lower", "prediction_ci80_upper",
        "injury_adjustment",
        "predicted_points_model", "pace_prior", "pace_weight", "games_played_season",
        "actual_points"]


def completed_games(season: int) -> int:
    con = sqlite3.connect(DB_PATH)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM player_weekly_stats WHERE season = ?",
            (season,)).fetchone()[0]
    finally:
        con.close()


def actual_points(season: int, week: int) -> pd.Series:
    """Real PPR points for a played week, indexed by player_id; empty if the
    week has no rows yet. Attached to that week's file so the page can show
    projected vs. actual once the games are in."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT player_id, fantasy_points FROM player_weekly_stats "
            "WHERE season = ? AND week = ? AND fantasy_points IS NOT NULL",
            con, params=(int(season), int(week)))
    finally:
        con.close()
    return df.set_index("player_id")["fantasy_points"]


def week_result(df: pd.DataFrame) -> dict | None:
    """Accuracy of THIS week's served numbers against what happened: mean
    absolute miss, signed bias, and how often the 80% range held. Only rows
    with an actual -- a projected player who did not record a stat line is
    not scored, matching the backtester."""
    if "actual_points" not in df.columns:
        return None
    d = df[df["actual_points"].notna() & df["predicted_points"].notna()]
    if len(d) < 20:
        return None
    err = d["predicted_points"] - d["actual_points"]
    out = {"n": int(len(d)), "mae": round(float(err.abs().mean()), 2),
           "bias": round(float(err.mean()), 2)}
    if {"prediction_ci80_lower", "prediction_ci80_upper"} <= set(d.columns):
        inside = ((d["actual_points"] >= d["prediction_ci80_lower"])
                  & (d["actual_points"] <= d["prediction_ci80_upper"]))
        out["coverage80"] = round(float(inside.mean()), 3)
    return out


def schedule_by_week(season: int) -> dict:
    """{week: {team: (opponent, home|away)}} from the real schedule."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT week, home_team, away_team FROM schedule WHERE season = ?",
            con, params=(season,))
    finally:
        con.close()
    out = {}
    for _, r in df.iterrows():
        wk = int(r["week"])
        out.setdefault(wk, {})
        h, a = str(r["home_team"]).strip(), str(r["away_team"]).strip()
        if h and a:
            out[wk][h] = (a, "home")
            out[wk][a] = (h, "away")
    return out


def _clean(df: pd.DataFrame) -> list:
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float).round(2)
    return json.loads(out.to_json(orient="records"))


def build_weekly_model(season, weeks, top_n):
    from src.predict import NFLPredictor
    p = NFLPredictor()
    if not p.initialize():
        print("no trained models; run `python -m src.models.train` first")
        return [], {}
    written, counts, results = [], {}, {}
    for wk in weeks:
        df = p.predict(n_weeks=1, position=None, top_n=top_n, as_of=(season, wk))
        if df.empty:
            continue
        if "opponent" in df.columns:   # no opponent = bye, no game to project
            df = df[df["opponent"].astype(str).str.strip().ne("")]
        if df.empty:
            continue
        actual = actual_points(season, wk)
        if len(actual):
            df["actual_points"] = df["player_id"].map(actual)
            res = week_result(df)
            if res:
                results[str(wk)] = res
        df = df.sort_values("predicted_points", ascending=False)
        (OUT_DIR / f"weekly_{season}_wk{wk}.json").write_text(json.dumps(_clean(df)))
        written.append(wk); counts[str(wk)] = len(df)
        played = f", played: MAE {results[str(wk)]['mae']}" if str(wk) in results else ""
        print(f"  wk{wk}: {len(df):4d} players, median "
              f"{df['predicted_points'].median():.1f}{played}", flush=True)
    return written, counts, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=[1, 18])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=1200)
    args = ap.parse_args()

    from src.predict import get_prediction_target_week
    season = args.season or get_prediction_target_week()[0]

    played = completed_games(season)
    mode = "weekly_blend"
    print(f"{season}: {played} completed player-game rows -> mode={mode}")

    sched = schedule_by_week(season)
    weeks = [w for w in range(args.weeks[0], args.weeks[1] + 1) if w in sched]
    if not weeks:
        print(f"no scheduled weeks in range for {season}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(f"weekly_{season}_wk*.json"):
        stale.unlink()

    written, counts, results = build_weekly_model(season, weeks, args.top_n)

    if not written:
        print("nothing written")
        return 1

    meta = {
        "season": int(season),
        "mode": mode,
        "weeks": written,
        "counts": counts,
        # Per played week: served-number accuracy against real results.
        "week_results": results,
        "completed_game_rows": int(played),
        "has_intervals": True,
        "pace_blend_kappa": PACE_BLEND_KAPPA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured": measured_accuracy(),
        "baseline_standing": baseline_standing(),
        "model": ("weekly ensemble (1w horizon), log1p+smearing calibration, "
                  "shrunk toward the Step 8 season pace by games played "
                  f"(w = g / (g + {PACE_BLEND_KAPPA:g}))"),
    }
    (OUT_DIR / "weekly_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmode={mode}, wrote weeks {written[0]}-{written[-1]} for {season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

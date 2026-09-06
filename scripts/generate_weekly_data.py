"""Emit per-week projections for docs/weekly.html.

TWO MODES, chosen automatically from whether the season has started.

  season_prorated  -- before any game is played. The weekly model cannot
                      differentiate weeks yet: opp_fpts_allowed comes from
                      team_defense_stats (no rows for an unplayed season) and
                      spread/implied_team_total come from betting lines that
                      are not posted. Verified 2026-08-31: weeks 1 and 2
                      produced byte-identical projections for all 795 players,
                      mean |diff| 0.000. So instead of dressing one number up
                      as eighteen forecasts, this mode publishes the SEASON
                      model's total divided by 17 -- a season pace, labelled as
                      one -- with each week's real opponent and byes removed.

  weekly_model     -- once games exist. Runs the weekly ensemble per week via
                      predict(as_of=(season, week)), which is the path
                      validated against real 2025 outcomes on 2026-08-31.

The mode is written into weekly_meta.json so the page can state which it is
showing. Prorated rows deliberately carry NO confidence interval: a season
floor/ceiling divided by 17 is a pace band, not a weekly outcome range, and
presenting it as the latter would overstate precision.

`/ 17` IS THE MEASURED CHOICE FOR COLD START, not a placeholder (2026-09-05).
Scored against real week-1 PPR over 2021-2025, 208 players with no NFL
history: MAE 3.70, bias -0.18, R2 +0.270. Four alternatives lost, each with a
paired-bootstrap CI entirely on the wrong side of zero -- team-room output x a
draft-round share (+0.58 MAE), that plus the week-1 opponent's defence
(+0.32), a half-and-half blend with the published number (+0.23), and
dividing by expected games instead of 17 (+0.77). See GAPS.md and
scripts/run_week1_coldstart_experiment.py. Anything replacing this has to beat
3.70 at -0.18 on cold start.

Usage:
    python scripts/generate_weekly_data.py
    python scripts/generate_weekly_data.py --weeks 1 6
    python scripts/generate_weekly_data.py --force-mode weekly_model
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

from config.settings import DB_PATH

OUT_DIR = Path("docs/data")
GAMES_PER_SEASON = 17

MEASURED = {
    "QB": {"mae": 7.09, "bias": -1.30},
    "RB": {"mae": 4.43, "bias": -1.62},
    "WR": {"mae": 4.03, "bias": -1.45},
    "TE": {"mae": 3.00, "bias": -1.22},
}

KEEP = ["name", "position", "team", "opponent", "home_away",
        "predicted_points", "prediction_ci80_lower", "prediction_ci80_upper"]


def completed_games(season: int) -> int:
    con = sqlite3.connect(DB_PATH)
    try:
        return con.execute(
            "SELECT COUNT(*) FROM player_weekly_stats WHERE season = ?",
            (season,)).fetchone()[0]
    finally:
        con.close()


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


def season_rows() -> pd.DataFrame:
    """Season-model projections from data/players_{POS}.json."""
    frames = []
    for pos in ("QB", "RB", "WR", "TE"):
        p = Path(f"data/players_{pos}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d if isinstance(d, list) else sum(d.values(), [])
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["position"] = pos
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[df["projection_points_total"].notna()]


def _clean(df: pd.DataFrame) -> list:
    cols = [c for c in KEEP if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float).round(2)
    return json.loads(out.to_json(orient="records"))


def build_prorated(season, weeks, sched):
    base = season_rows()
    if base.empty:
        print("no season projections in data/players_*.json")
        return [], {}
    base["predicted_points"] = base["projection_points_total"] / GAMES_PER_SEASON
    written, counts = [], {}
    for wk in weeks:
        smap = sched.get(wk, {})
        if not smap:
            continue
        df = base[base["team"].isin(smap)].copy()
        if df.empty:
            continue
        df["opponent"] = df["team"].map(lambda t: smap[t][0])
        df["home_away"] = df["team"].map(lambda t: smap[t][1])
        df = df.sort_values("predicted_points", ascending=False)
        (OUT_DIR / f"weekly_{season}_wk{wk}.json").write_text(json.dumps(_clean(df)))
        written.append(wk); counts[str(wk)] = len(df)
        print(f"  wk{wk}: {len(df):4d} players "
              f"({len(base) - len(df)} on bye), pace median "
              f"{df['predicted_points'].median():.1f}", flush=True)
    return written, counts


def build_weekly_model(season, weeks, top_n):
    from src.predict import NFLPredictor
    p = NFLPredictor()
    if not p.initialize():
        print("no trained models; run `python -m src.models.train` first")
        return [], {}
    written, counts = [], {}
    for wk in weeks:
        df = p.predict(n_weeks=1, position=None, top_n=top_n, as_of=(season, wk))
        if df.empty:
            continue
        if "opponent" in df.columns:   # no opponent = bye, no game to project
            df = df[df["opponent"].astype(str).str.strip().ne("")]
        if df.empty:
            continue
        df = df.sort_values("predicted_points", ascending=False)
        (OUT_DIR / f"weekly_{season}_wk{wk}.json").write_text(json.dumps(_clean(df)))
        written.append(wk); counts[str(wk)] = len(df)
        print(f"  wk{wk}: {len(df):4d} players, median "
              f"{df['predicted_points'].median():.1f}", flush=True)
    return written, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=[1, 18])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=1200)
    ap.add_argument("--force-mode", choices=["season_prorated", "weekly_model"],
                    default=None)
    args = ap.parse_args()

    from src.predict import get_prediction_target_week
    season = args.season or get_prediction_target_week()[0]

    played = completed_games(season)
    mode = args.force_mode or (
        "weekly_model" if played > 0 else "season_prorated")
    print(f"{season}: {played} completed player-game rows -> mode={mode}")

    sched = schedule_by_week(season)
    weeks = [w for w in range(args.weeks[0], args.weeks[1] + 1) if w in sched]
    if not weeks:
        print(f"no scheduled weeks in range for {season}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUT_DIR.glob(f"weekly_{season}_wk*.json"):
        stale.unlink()

    if mode == "season_prorated":
        written, counts = build_prorated(season, weeks, sched)
    else:
        written, counts = build_weekly_model(season, weeks, args.top_n)

    if not written:
        print("nothing written")
        return 1

    meta = {
        "season": int(season),
        "mode": mode,
        "weeks": written,
        "counts": counts,
        "games_per_season": GAMES_PER_SEASON,
        "completed_game_rows": int(played),
        "has_intervals": mode == "weekly_model",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured": MEASURED,
        "model": ("season model (Step 8) total / 17"
                  if mode == "season_prorated"
                  else "weekly ensemble (1w horizon), log1p+smearing calibration"),
    }
    (OUT_DIR / "weekly_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmode={mode}, wrote weeks {written[0]}-{written[-1]} for {season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

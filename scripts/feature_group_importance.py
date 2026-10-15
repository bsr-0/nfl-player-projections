#!/usr/bin/env python3
"""Grouped permutation importance of the served model on held-out 2025.

How much does the served model lean on each family of features? For each
family the 2025 test rows have that family's columns shuffled together (one
row permutation for the whole group, so correlated features move as a unit),
the held-out fold predicts again, and the change in MAE is the importance.

- Model: the served architecture's 2025 fold, trained through 2024
  (data/experiments/full_ppr_head_to_head_20260930/served_2025_fold_v6), on
  its frozen preflight inputs. Nothing is refit; no live artifact is read or
  written (MODELS_DIR is redirected to a scratch copy).
- Rows: that fold's captured next-game targets, regular season only.
- Features as of each week: week k's rows are rebuilt with every 2025 row
  after week k removed. The standard one-pass build lets five features see
  later weeks (the timing audit below records which), so importance is not
  measured on it.
- Uncertainty: the per-row change is averaged over R shuffles, then whole
  target games are resampled (95%).
- 2026 is never touched: this is reliance on a family in an earlier held-out
  season, not a selection test, and not a causal effect.

The served page applies a pace blend on top of this model; the blend is not
in this analysis.

Output: data/experiments/feature_group_importance_2025/importance.json

Timing audit (stored with the result): every row's target game comes after its
origin game; game-level inputs (betting line, venue, weather, injury report) are
matched to the origin or target game; and the features are rebuilt with all
2025 weeks after a cutoff removed, which must leave every earlier row unchanged.

Usage:
    python scripts/feature_group_importance.py [--repeats 5]
    python scripts/feature_group_importance.py --audit-only   # timing audit only
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import POSITIONS
from src.evaluation.weekly_review import mean_ci

EXP = ROOT / "data" / "experiments" / "full_ppr_head_to_head_20260930"
FOLD = EXP / "served_2025_fold_v6"
PREFLIGHT = EXP / "served_2025_preflight_v5"
OUT = ROOT / "data" / "experiments" / "feature_group_importance_2025"
KEY = ["player_id", "season", "week"]
SEED = 42

# First match wins. Correlated features sit in one family so a shuffle cannot
# be undone by a near-copy left in place.
GROUPS = [
    ("Availability", r"^(injury_score|availability_3yr)$"),
    ("Betting line", r"^(implied_team_total|spread)$"),
    ("Opponent", r"^opp_"),
    ("Weather & venue", r"^(wind_speed_mph|precipitation_flag|temperature_bucket|is_dome)$"),
    ("Usage", r"(share|snap|targets_roll|attempts_roll|receptions_roll|wopr|participation|depth_chart"
              r"|dest_team_pos)"),
    ("Team context", r"^(team_|coaching_|current_qb_)"),
    ("Player history & profile", r"(prior|prev_season|age_curve|career_year|contract|draft|combine|speed_score"
                                 r"|power5|rookie|undrafted)"),
    ("Recent production & efficiency", r"."),
]


def family(feature: str) -> str:
    return next(name for name, pat in GROUPS if re.search(pat, feature))


def _uniform_injury_dates() -> None:
    """nflverse now ships 2025-26 injuries without date_modified while older
    seasons keep it; concatenated, the column mixes Timestamps and NaN and
    pandas' to_datetime(utc=True) raises a bare AssertionError, which the
    pipeline catches and then skips every external feature. Parse it here per
    row so this run gets the features the fold was trained with. Only this
    process is patched; the repository code is unchanged."""
    from src.data import external_data as ed
    orig = ed.InjuryDataLoader.load_injuries
    if getattr(orig, "_uniform", False):
        return

    def load(self, seasons):
        df = orig(self, seasons)
        if "date_modified" in df.columns:
            iso = df["date_modified"].map(lambda x: x.isoformat() if hasattr(x, "isoformat") else None)
            df["date_modified"] = pd.to_datetime(iso, utc=True, errors="coerce", format="ISO8601")
        return df
    load._uniform = True
    ed.InjuryDataLoader.load_injuries = load


def prepare(work: Path) -> pd.DataFrame:
    """The fold's test frame after the training pipeline's own preprocessing."""
    from src.models.feature_preparation import _prepare_training_data
    from src.utils.models_dir import redirect_models_dir
    cached = OUT / "prepared_test.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    _uniform_injury_dates()
    train = pd.read_parquet(PREFLIGHT / "raw_train.parquet")
    test = pd.read_parquet(PREFLIGHT / "raw_test.parquet")
    with redirect_models_dir(work):
        _, test, _ = _prepare_training_data(train, test, list(POSITIONS), tune_hyperparameters=False,
                                            n_trials=1, fit_models=False)
    test.to_parquet(cached, index=False)
    return test


def predictor(work: Path):
    from src.models.position_models import MultiWeekModel
    models = {p: MultiWeekModel.load(p, filepath=work / f"multiweek_{p.lower()}.joblib") for p in POSITIONS}

    def predict(frame: pd.DataFrame, position: str) -> np.ndarray:
        m = models[position]
        base = m.models.get(1) or list(m.models.values())[0]
        frame = frame.copy()
        for fn in getattr(base, "feature_names", []):
            if fn not in frame.columns:
                frame[fn] = getattr(base, "feature_medians", {}).get(fn, 0)
        return np.asarray(m.predict(frame, n_weeks=1), dtype=float)

    features = {p: list(getattr(models[p].models.get(1) or list(models[p].models.values())[0],
                                "feature_names", [])) for p in POSITIONS}
    return predict, features


# ---------------------------------------------------------------------------
# Timing audit: was every input known before the target game kicked off?
# ---------------------------------------------------------------------------

def prepare_through(work: Path, cutoff: int) -> pd.DataFrame:
    """The prepared test frame when every 2025 row after week `cutoff` is
    removed BEFORE preprocessing. Rows at or before the cutoff must come out
    identical to the full-season run if no feature looks ahead."""
    from src.models.feature_preparation import _prepare_training_data
    from src.utils.models_dir import redirect_models_dir
    cached = OUT / f"prepared_test_through_wk{cutoff}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    _uniform_injury_dates()
    train = pd.read_parquet(PREFLIGHT / "raw_train.parquet")
    test = pd.read_parquet(PREFLIGHT / "raw_test.parquet")
    test = test[test["week"] <= cutoff].reset_index(drop=True)
    with redirect_models_dir(work):
        _, test, _ = _prepare_training_data(train, test, list(POSITIONS), tune_hyperparameters=False,
                                            n_trials=1, fit_models=False)
    test.to_parquet(cached, index=False)
    return test


def prepare_asof(work: Path, weeks) -> pd.DataFrame:
    """The held-out frame built as of each week: week k's rows come from a
    rebuild with every 2025 row after week k removed, so no feature can see a
    later game. The standard one-pass build lets a few features see later
    weeks (the truncation check finds them)."""
    cached = OUT / "prepared_test_asof.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    parts = []
    for k in weeks:
        cut = prepare_through(work, k)
        parts.append(cut[cut["week"] == k])
        print(f"as-of week {k}: {int((cut['week'] == k).sum())} rows")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cached, index=False)
    return out


def truncation_check(full: pd.DataFrame, cut: pd.DataFrame, cutoff: int, features: list) -> dict:
    """Compare every model feature on rows at or before the cutoff."""
    a = full[full["week"] <= cutoff].drop_duplicates(KEY).set_index(KEY)
    b = cut.drop_duplicates(KEY).set_index(KEY)
    common = a.index.intersection(b.index)
    changed = {}
    for fn in features:
        if fn not in a.columns or fn not in b.columns:
            continue
        x = pd.to_numeric(a.loc[common, fn], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(b.loc[common, fn], errors="coerce").to_numpy(dtype=float)
        diff = ~(np.isclose(x, y, rtol=1e-9, atol=1e-9) | (np.isnan(x) & np.isnan(y)))
        if diff.any():
            changed[fn] = {"rows": int(diff.sum()), "max_abs_diff": float(np.nanmax(np.abs(x - y)))}
    return {"cutoff_week": cutoff, "rows_compared": int(len(common)), "rows_missing": int(len(a) - len(common)),
            "features_compared": int(sum(fn in a.columns and fn in b.columns for fn in features)),
            "features_changed": changed}


def _team_games(sched: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: implied team total from the nflverse schedule
    (spread_line > 0 means the home team is favoured), and dome and wind from
    game_weather, the table the weather features are built from (wind is 0
    in a dome, as feature_engineering sets it)."""
    rows = []
    for g in sched.itertuples(index=False):
        total, spread = getattr(g, "total_line", np.nan), getattr(g, "spread_line", np.nan)
        for team, sign in ((g.home_team, 1), (g.away_team, -1)):
            rows.append({"season": g.season, "week": g.week, "team": team,
                         "implied": (total + sign * spread) / 2})
    wx = pd.concat([weather.rename(columns={"home_team": "team"}).drop(columns="away_team"),
                    weather.rename(columns={"away_team": "team"}).drop(columns="home_team")])
    wx = wx.drop_duplicates(["season", "week", "team"])
    wx["dome"] = wx["is_dome"].fillna(0).astype(float)
    wx["wind"] = wx["wind_mph"].where(wx["dome"] == 0, 0.0).fillna(0.0)
    return pd.DataFrame(rows).merge(wx[["season", "week", "team", "dome", "wind"]],
                                    on=["season", "week", "team"], how="left")


def alignment_check(test: pd.DataFrame, cap: pd.DataFrame) -> dict:
    """Game-level inputs (betting line, venue, weather, injury report) are
    joined by game. Which game: the origin game, already played when the
    target game kicks off, or the target game itself?"""
    import sqlite3
    import nfl_data_py as nfl
    from config.settings import DB_PATH
    from src.data.external_data import InjuryDataLoader
    with sqlite3.connect(DB_PATH) as con:
        weather = pd.read_sql("SELECT season, week, home_team, away_team, is_dome, wind_mph FROM game_weather "
                              "WHERE season = 2025", con)
    sched = _team_games(nfl.import_schedules([2025]), weather)
    rows = cap.merge(test.drop_duplicates(KEY)[KEY + [c for c in ("implied_team_total", "is_dome", "wind_speed_mph",
                                                                     "injury_score") if c in test.columns]],
                     on=KEY, how="inner")
    o = rows.merge(sched, on=["season", "week", "team"], how="left")
    t = rows.merge(sched.rename(columns={"week": "target_week", "team": "target_team"}),
                   on=["season", "target_week", "target_team"], how="left")
    out = {}

    def share(feature, ov, tv, row, tol=0.01):
        ok = row.notna() & ov.notna() & tv.notna() & ((ov - tv).abs() > tol)   # rows where the games differ
        if not ok.any():
            return {"rows_where_games_differ": 0}
        return {"rows_where_games_differ": int(ok.sum()),
                "matches_origin_game": round(float(((row - ov).abs() <= tol)[ok].mean()), 4),
                "matches_target_game": round(float(((row - tv).abs() <= tol)[ok].mean()), 4)}
    if "implied_team_total" in rows:
        out["Betting line (implied_team_total)"] = share("implied", o["implied"], t["implied"],
                                                         rows["implied_team_total"].astype(float))
    if "is_dome" in rows:
        out["Weather & venue (is_dome)"] = share("dome", o["dome"].astype(float), t["dome"].astype(float),
                                                 rows["is_dome"].astype(float), tol=0.5)
    if "wind_speed_mph" in rows:
        out["Weather & venue (wind_speed_mph)"] = share("wind", o["wind"].astype(float), t["wind"].astype(float),
                                                        rows["wind_speed_mph"].astype(float), tol=0.5)
    if "injury_score" in rows:
        loader = InjuryDataLoader()
        st = loader.get_player_injury_status(loader.load_injuries([2025]))[["player_id", "season", "week",
                                                                            "injury_score"]]
        st = st.drop_duplicates(["player_id", "season", "week"])
        oi = rows.merge(st, on=["player_id", "season", "week"], how="left")["injury_score_y"].fillna(1.0)
        ti = rows.merge(st.rename(columns={"week": "target_week"}), on=["player_id", "season", "target_week"],
                        how="left")["injury_score_y"].fillna(1.0)
        out["Availability (injury_score)"] = share("injury", oi, ti, rows["injury_score"].astype(float))
    return out


def audit(work: Path, test: pd.DataFrame, cap: pd.DataFrame, features: dict, cutoffs,
          asof: pd.DataFrame | None = None) -> dict:
    """`test` is the standard one-pass build (what the fold was scored on);
    the truncation check is run on it, and records which features look
    ahead. Importance itself uses `asof`, which cannot."""
    allf = sorted({fn for fs in features.values() for fn in fs})
    train_max = max(int(s) for s in str(cap["train_seasons"].iloc[0]).split(","))
    later = (cap["target_week"] > cap["week"])
    res = {"train_seasons_through": train_max, "test_season": int(cap["season"].max()),
           "rows": int(len(cap)), "target_after_origin_rows": int(later.sum()),
           "alignment": alignment_check(asof if asof is not None else test, cap),
           "truncation": [truncation_check(test, prepare_through(work, k), k, allf) for k in cutoffs],
           "importance_built_as_of": asof is not None,
           "asof_rows": int(len(asof)) if asof is not None else None,
           "not_verifiable": [
               "The frozen 2025 inputs were read from the database on 2026-10-07, after the season; stat "
               "corrections published after each kickoff cannot be ruled out (no as-of-kickoff archive exists).",
               "2025 injury reports carry no modification time in the current nflverse feed, so report timing "
               "within a week cannot be checked; the injury feature is joined to the origin week (above)."]}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--audit-only", action="store_true",
                    help="rerun the timing audit and keep the existing importance numbers")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[6, 12])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / "models_scratch"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(FOLD / "models", work)

    test = prepare(work)
    predict, features = predictor(work)
    cap = pd.read_parquet(FOLD / "raw_capture.parquet")
    cap = cap[cap["target_week"] <= 18]
    cap["gkey"] = (cap["season"].astype(str) + "wk" + cap["target_week"].astype(str) + " "
                   + cap[["target_team", "target_opponent"]].apply(lambda r: "|".join(sorted(r)), axis=1))
    asof = prepare_asof(work, sorted(int(w) for w in cap["week"].unique()))
    if args.audit_only:
        result = json.loads((OUT / "importance.json").read_text())
        result["timing_audit"] = audit(work, test, cap, features, args.cutoffs, asof)
        (OUT / "importance.json").write_text(json.dumps(result, indent=2) + "\n")
        shutil.rmtree(work)
        print(json.dumps(result["timing_audit"], indent=1))
        return 0
    rng = np.random.default_rng(SEED)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
              "fold": str(FOLD.relative_to(ROOT)), "preflight": str(PREFLIGHT.relative_to(ROOT)),
              "repeats": args.repeats, "seed": SEED, "features": "as of each week", "groups": {}, "positions": {}}
    for pos in POSITIONS:
        keys = cap[cap["position"] == pos][KEY + ["actual_points", "predicted_points", "gkey"]]
        keys = keys.rename(columns={"predicted_points": "captured"})
        one_pass = test[test["position"] == pos].merge(keys, on=KEY, how="inner")
        check = float(np.nanmax(np.abs(predict(one_pass, pos) - one_pass["captured"].to_numpy())))
        frame = asof[asof["position"] == pos].reset_index(drop=True).merge(keys, on=KEY, how="inner")
        base = predict(frame, pos)
        moved = np.abs(base - frame["captured"].to_numpy())
        y = frame["actual_points"].to_numpy(dtype=float)
        base_err = np.abs(base - y)
        fams: dict[str, list] = {}
        for fn in features[pos]:
            fams.setdefault(family(fn), []).append(fn)
        out = {"n": int(len(frame)), "games": int(frame["gkey"].nunique()),
               "base_mae": round(float(base_err.mean()), 3),
               "saved_fold_mae": round(float(np.abs(frame["captured"] - frame["actual_points"]).mean()), 3),
               "one_pass_reproduction_max_abs_diff": round(check, 6),
               "asof_rows_moved": int((moved > 1e-6).sum()), "asof_max_move": round(float(moved.max()), 3),
               "families": {}}
        print(f"{pos}: n={len(frame)} as-of MAE {base_err.mean():.3f} (saved fold {out['saved_fold_mae']:.3f}; "
              f"one-pass rebuild reproduces it to {check:.1e}; {out['asof_rows_moved']} rows moved, max "
              f"{out['asof_max_move']:.2f})")
        for name, cols in fams.items():
            cols = [c for c in cols if c in frame.columns]
            gain = np.zeros(len(frame))
            for _ in range(args.repeats):
                perm = rng.permutation(len(frame))
                shuffled = frame.copy()
                shuffled[cols] = frame[cols].to_numpy()[perm]
                gain += np.abs(predict(shuffled, pos) - y) - base_err
            gain /= args.repeats
            ci = mean_ci(gain, frame["gkey"].to_numpy())
            out["families"][name] = {**ci, "features": len(cols), "members": sorted(cols)}
            result["groups"][name] = sorted(set(result["groups"].get(name, [])) | set(cols))
            print(f"   {name:<32} {ci['est']:+.3f} ({ci['lo']:+.3f} to {ci['hi']:+.3f}), {len(cols)} features")
        result["positions"][pos] = out
    result["timing_audit"] = audit(work, test, cap, features, args.cutoffs, asof)
    (OUT / "importance.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.rmtree(work)
    print(f"Wrote {(OUT / 'importance.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What the site published before each game, against what happened.

The weekly and game pages are regenerated after games are played, so the
files on disk for a past week are not what anyone saw before kickoff. Git
history is: every committed version of ``docs/data/*_{season}_wk{N}.json``
carries its commit time. For each game this module takes the LAST version
committed strictly before that game's kickoff, which is the out-of-sample
prediction the project actually made, and scores it against the result.

Rules, all conservative:
- Kickoff is the nflverse game time (US Eastern); when unavailable, the
  start of the game day in Eastern time, which can only exclude snapshots.
- A game with no version committed before its kickoff has no published
  prediction. It is listed, never filled in from a later version.
- Commit time is the timestamp. It is the moment the prediction existed in
  the repository; the public site may have deployed it later, never earlier.
- A projected player with no stat line in a final game is not scored
  (matching the backtester); a player who recorded a stat line without a
  published projection is counted separately, never silently dropped.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_DATA = "docs/data"
EASTERN = "America/New_York"
POSITIONS = ("QB", "RB", "WR", "TE")


# ---------------------------------------------------------------------------
# Git snapshots
# ---------------------------------------------------------------------------

def git_versions(relpath: str, repo: Path = PROJECT_ROOT) -> list[tuple[str, int]]:
    """(commit, committer unix time) for every committed version, newest first."""
    out = subprocess.run(["git", "log", "--format=%H %ct", "--", relpath], cwd=repo,
                         capture_output=True, text=True, check=True).stdout.split()
    return [(out[i], int(out[i + 1])) for i in range(0, len(out), 2)]


@lru_cache(maxsize=256)
def _json_at(repo: str, commit: str, relpath: str):
    proc = subprocess.run(["git", "show", f"{commit}:{relpath}"], cwd=repo, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def json_at(commit: str, relpath: str, repo: Path = PROJECT_ROOT):
    return _json_at(str(repo), commit, relpath)


def snapshot_before(versions: list[tuple[str, int]], kickoff_ts: float) -> tuple[str, int] | None:
    """The newest version committed strictly before kickoff, or None."""
    before = [(c, t) for c, t in versions if t < kickoff_ts]
    return max(before, key=lambda v: v[1]) if before else None


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="minutes")


# ---------------------------------------------------------------------------
# Kickoffs and results
# ---------------------------------------------------------------------------

def kickoff_times(season: int, con: sqlite3.Connection) -> pd.DataFrame:
    """One row per scheduled game: week, home_team, away_team, kickoff (UTC),
    kickoff_source, home_score, away_score (local schedule)."""
    local = pd.read_sql("SELECT week, home_team, away_team, game_time, home_score, away_score "
                        "FROM schedule WHERE season = ?", con, params=[int(season)])
    day = pd.to_datetime(local["game_time"].astype(str).str[:10], errors="coerce")
    local["kickoff"] = day.dt.tz_localize(EASTERN).dt.tz_convert("UTC")
    local["kickoff_source"] = "game day 00:00 ET"
    try:
        from src.data.nfl_data_loader import _fetch_schedules
        up = _fetch_schedules([int(season)])
        up = up[up["gameday"].notna() & up["gametime"].notna()]
        exact = pd.to_datetime(up["gameday"].astype(str) + " " + up["gametime"].astype(str), errors="coerce")
        up = up.assign(_kick=exact.dt.tz_localize(EASTERN).dt.tz_convert("UTC"))[
            ["week", "home_team", "away_team", "_kick"]].dropna()
        local = local.merge(up, on=["week", "home_team", "away_team"], how="left")
        has = local["_kick"].notna()
        local.loc[has, "kickoff"] = local.loc[has, "_kick"]
        local.loc[has, "kickoff_source"] = "nflverse game time"
        local = local.drop(columns="_kick")
    except Exception:  # network or schema: keep the conservative day start
        pass
    local["final"] = local["home_score"].notna() & local["away_score"].notna()
    return local.drop(columns="game_time")


def player_actuals(con: sqlite3.Connection, season: int, week: int) -> pd.DataFrame:
    """Stat lines for a week: player_id, name, team, position, actual_points."""
    return pd.read_sql(
        "SELECT s.player_id, p.name, s.team, p.position, s.fantasy_points AS actual_points "
        "FROM player_weekly_stats s JOIN players p ON p.player_id = s.player_id "
        "WHERE s.season = ? AND s.week = ? AND s.fantasy_points IS NOT NULL",
        con, params=[int(season), int(week)])


def _short_name(name) -> str:
    """'Bi.Robinson' / 'B.Robinson' -> 'b.robinson': the display name's first
    initial and surname, which survives the database later disambiguating
    initials ('B.' -> 'Bi.')."""
    text = str(name or "").strip().lower()
    first, _, last = text.partition(".")
    return f"{first[:1]}.{last.strip()}" if last else text


def collapse_duplicate_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop repeated listings of one player with the same projection.

    Early files without player_id sometimes listed a player twice; scoring
    both would count one miss twice. Repeats with DIFFERENT projections are
    kept (and stay unmatched below), since which one was meant is unknowable.
    """
    if "player_id" in rows.columns and rows["player_id"].notna().all():
        return rows, 0
    key = ["name", "team", "position", "predicted_points"]
    dup = rows.duplicated(key, keep="first")
    return rows[~dup].reset_index(drop=True), int(dup.sum())


def attach_actuals(rows: pd.DataFrame, actuals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join actual points onto published rows; return (rows, stat lines never matched).

    By player_id when the published rows carry it. Early-season files did not,
    so otherwise by (name, team, position), then by (first initial + surname,
    team, position); each only where the key is unique on both sides (an
    ambiguous name is left unmatched, not guessed).
    """
    rows = rows.copy()
    rows["actual_points"] = np.nan
    matched_ids: set = set()
    if "player_id" in rows.columns and rows["player_id"].notna().any():
        by_id = actuals.drop_duplicates("player_id").set_index("player_id")["actual_points"]
        has = rows["player_id"].notna()
        rows.loc[has, "actual_points"] = rows.loc[has, "player_id"].map(by_id)
        matched_ids |= set(rows.loc[has & rows["actual_points"].notna(), "player_id"])
    no_id = rows["player_id"].isna() if "player_id" in rows.columns else pd.Series(True, index=rows.index)
    a_all = actuals.assign(_short=actuals["name"].map(_short_name))
    r_all = rows.assign(_short=rows["name"].map(_short_name)) if "name" in rows.columns else None
    for key in (["name", "team", "position"], ["_short", "team", "position"]):
        need = rows["actual_points"].isna() & no_id
        if r_all is None or not need.any():
            break
        a = a_all[~a_all["player_id"].isin(matched_ids)]
        a = a[~a.duplicated(key, keep=False)]
        r_unique = ~r_all.duplicated(key, keep=False)
        lookup = a.set_index(key)
        for i in rows.index[need & r_unique]:
            k = tuple(r_all.loc[i, key])
            if k in lookup.index:
                rows.loc[i, "actual_points"] = float(lookup.loc[k, "actual_points"])
                matched_ids.add(lookup.loc[k, "player_id"])
    unmatched = actuals[~actuals["player_id"].isin(matched_ids)]
    return rows, unmatched


# ---------------------------------------------------------------------------
# Published rows per game
# ---------------------------------------------------------------------------

@dataclass
class WeekRecord:
    week: int
    rows: pd.DataFrame                      # published rows of kicked-off games
    snapshots: list = field(default_factory=list)
    not_published: list = field(default_factory=list)   # "AWY@HOM" kicked off with no prior version


def published_rows(kind: str, season: int, week: int, games: pd.DataFrame, now: pd.Timestamp,
                   repo: Path = PROJECT_ROOT) -> WeekRecord:
    """Published rows for every game of `week` that kicked off before `now`.

    kind "weekly": player rows whose team plays in the game; kind "game":
    the game's own row.
    """
    relpath = f"{SITE_DATA}/{'weekly' if kind == 'weekly' else 'game_predictions'}_{season}_wk{week}.json"
    meta_path = f"{SITE_DATA}/{'weekly_meta' if kind == 'weekly' else 'game_predictions_meta'}.json"
    versions = git_versions(relpath, repo)
    parts, snaps, missing = [], {}, []
    for g in games[(games["week"] == week) & (games["kickoff"] <= now)].itertuples(index=False):
        label = f"{g.away_team}@{g.home_team}"
        snap = snapshot_before(versions, g.kickoff.timestamp())
        data = json_at(snap[0], relpath, repo) if snap else None
        if not data:
            missing.append(label)
            continue
        frame = pd.DataFrame(data)
        if kind == "weekly":
            mine = frame[frame["team"].isin([g.home_team, g.away_team])]
        else:
            mine = frame[(frame["home_team"] == g.home_team) & (frame["away_team"] == g.away_team)]
        if mine.empty:
            missing.append(label)
            continue
        commit, ts = snap
        if commit not in snaps:
            meta = json_at(commit, meta_path, repo) or {}
            snaps[commit] = {"commit": commit[:8], "published_at": _iso(ts),
                             "mode": meta.get("mode"), "games": []}
        snaps[commit]["games"].append(label)
        parts.append(mine.assign(published_at=_iso(ts), published_commit=commit[:8],
                                 published_mode=snaps[commit]["mode"], game=label,
                                 kickoff=g.kickoff.isoformat(timespec="minutes"), final=bool(g.final),
                                 home_score=g.home_score, away_score=g.away_score))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return WeekRecord(week, rows, sorted(snaps.values(), key=lambda s: s["published_at"]), missing)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _r(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def player_metrics(rows: pd.DataFrame) -> dict | None:
    """Error of published projections on players with a stat line."""
    d = rows[rows["actual_points"].notna() & rows["predicted_points"].notna()]
    if d.empty:
        return None
    err = d["predicted_points"] - d["actual_points"]
    out = {"n": int(len(d)), "mae": _r(err.abs().mean(), 2), "bias": _r(err.mean(), 2),
           "rmse": _r(np.sqrt((err ** 2).mean()), 2)}
    if {"prediction_ci80_lower", "prediction_ci80_upper"} <= set(d.columns):
        c = d[d["prediction_ci80_lower"].notna() & d["prediction_ci80_upper"].notna()]
        if len(c):
            inside = (c["actual_points"] >= c["prediction_ci80_lower"]) & (c["actual_points"] <= c["prediction_ci80_upper"])
            out["coverage80"] = _r(inside.mean())
            out["n_with_range"] = int(len(c))
    return out


def player_week_summary(rec: WeekRecord, unprojected: pd.DataFrame) -> dict | None:
    if rec.rows.empty and not rec.not_published:
        return None
    final = rec.rows[rec.rows["final"]] if not rec.rows.empty else rec.rows
    m = player_metrics(final) if not final.empty else None
    out = {"published_basis": "last version committed before each game's kickoff",
           "snapshots": rec.snapshots, "games_not_published": rec.not_published,
           "n_published_final": int(len(final)),
           "n_no_stat_line": int(final["actual_points"].isna().sum()) if not final.empty else 0,
           "n_unprojected": int(len(unprojected)),
           "n_unprojected_5plus": int((unprojected["actual_points"] >= 5).sum()) if len(unprojected) else 0,
           "games_final": int(final["game"].nunique()) if not final.empty else 0}
    if m:
        out.update(m)
        out["by_position"] = {p: player_metrics(final[final["position"] == p])
                              for p in POSITIONS if (final["position"] == p).any()}
    return out


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def game_metrics(rows: pd.DataFrame) -> dict | None:
    """Accuracy of published game predictions on final games, beside the market."""
    d = rows[_final_mask(rows)].copy() if not rows.empty else rows
    if d.empty:
        return None
    hs, as_ = d["home_score"].astype(float), d["away_score"].astype(float)
    margin, total = hs - as_, hs + as_
    decided = margin != 0
    home_win = (margin > 0).astype(float)
    out = {"n": int(len(d)), "n_ties": int((~decided).sum()), "win_loss": {}, "margin": {}, "total": {}}
    for arm in ("logistic", "xgb", "rf"):
        col = f"home_win_prob_{arm}"
        if col in d and d[col].notna().any():
            p = d.loc[decided, col].astype(float)
            out["win_loss"][arm] = {"correct": int(((p >= .5) == (home_win[decided] == 1)).sum()),
                                    "n": int(decided.sum()), "brier": _r(_brier(p, home_win[decided]))}
    line = d["spread_line"].astype(float) if "spread_line" in d else pd.Series(np.nan, index=d.index)
    fav = decided & line.notna() & (line != 0)
    out["win_loss"]["market_favorite"] = {"correct": int(((line[fav] > 0) == (home_win[fav] == 1)).sum()),
                                          "n": int(fav.sum())}
    for arm in ("ridge", "xgb", "rf"):
        mcol, tcol = f"predicted_margin_{arm}", f"predicted_total_{arm}"
        if mcol in d and d[mcol].notna().any():
            pm = d[mcol].astype(float)
            ats = line.notna() & (pm != line) & (margin != line)
            win = (np.sign(pm - line) == np.sign(margin - line)) & ats
            out["margin"][arm] = {"mae": _r((pm - margin).abs().mean(), 2),
                                  "ats_won": int(win.sum()), "ats_lost": int((ats & ~win).sum()),
                                  "ats_push": int((line.notna() & (margin == line)).sum())}
        if tcol in d and d[tcol].notna().any():
            pt = d[tcol].astype(float)
            tl = d["total_line"].astype(float) if "total_line" in d else pd.Series(np.nan, index=d.index)
            ou = tl.notna() & (pt != tl) & (total != tl)
            win = (np.sign(pt - tl) == np.sign(total - tl)) & ou
            out["total"][arm] = {"mae": _r((pt - total).abs().mean(), 2),
                                 "ou_won": int(win.sum()), "ou_lost": int((ou & ~win).sum()),
                                 "ou_push": int((tl.notna() & (total == tl)).sum())}
    if line.notna().any():
        out["margin"]["market_line"] = {"mae": _r((line[line.notna()] - margin[line.notna()]).abs().mean(), 2)}
    if "total_line" in d and d["total_line"].notna().any():
        tl = d["total_line"].astype(float)
        out["total"]["market_line"] = {"mae": _r((tl[tl.notna()] - total[tl.notna()]).abs().mean(), 2)}
    return out


def game_week_summary(rec: WeekRecord) -> dict | None:
    if rec.rows.empty and not rec.not_published:
        return None
    out = {"published_basis": "last version committed before each game's kickoff",
           "snapshots": rec.snapshots, "games_not_published": rec.not_published,
           "games_final": int(rec.rows["final"].sum()) if not rec.rows.empty else 0}
    m = game_metrics(rec.rows) if not rec.rows.empty else None
    if m:
        out.update(m)
    return out


# ---------------------------------------------------------------------------
# One call per week, for the site generators
# ---------------------------------------------------------------------------

def player_week(con: sqlite3.Connection, season: int, week: int, games: pd.DataFrame,
                now: pd.Timestamp, repo: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict | None]:
    """Published player rows of every kicked-off game in `week` (with actual
    points for final games) and the week's honesty summary."""
    rec = published_rows("weekly", season, week, games, now, repo)
    dropped = 0
    unprojected = pd.DataFrame(columns=["player_id", "name", "team", "position", "actual_points"])
    if not rec.rows.empty:
        rec.rows, dropped = collapse_duplicate_rows(rec.rows)
        final_teams = set(games.loc[(games["week"] == week) & games["final"] & (games["kickoff"] <= now),
                                    ["home_team", "away_team"]].to_numpy().ravel())
        acts = player_actuals(con, season, week)
        acts = acts[acts["team"].isin(final_teams) & acts["position"].isin(POSITIONS)]
        rec.rows, unprojected = attach_actuals(rec.rows, acts)
        rec.rows.loc[~rec.rows["final"], "actual_points"] = np.nan
    summary = player_week_summary(rec, unprojected)
    if summary is not None:
        summary["n_duplicate_rows_dropped"] = dropped
        summary["unprojected_top"] = (unprojected.sort_values("actual_points", ascending=False)
                                      .head(10)[["name", "team", "position", "actual_points"]]
                                      .round(1).to_dict("records"))
    return rec.rows, summary


def game_week(season: int, week: int, games: pd.DataFrame, now: pd.Timestamp,
              repo: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict | None]:
    """Published game rows of every kicked-off game in `week` (scores for final
    games) and the week's honesty summary."""
    rec = published_rows("game", season, week, games, now, repo)
    if not rec.rows.empty:
        rec.rows.loc[~rec.rows["final"], ["home_score", "away_score"]] = np.nan
    return rec.rows, game_week_summary(rec)


def _final_mask(rows: pd.DataFrame) -> pd.Series:
    """`final` as a clean boolean (it may come back from JSON as 1.0/null)."""
    return rows["final"].map(lambda v: bool(v) if v == v and v is not None else False).astype(bool)


def season_player_summary(frames: list[pd.DataFrame]) -> dict | None:
    rows = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    if rows.empty:
        return None
    final = rows[_final_mask(rows)]
    m = player_metrics(final)
    if m:
        m["by_position"] = {p: player_metrics(final[final["position"] == p])
                            for p in POSITIONS if (final["position"] == p).any()}
    return m


def season_game_summary(frames: list[pd.DataFrame]) -> dict | None:
    rows = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    return game_metrics(rows) if not rows.empty else None

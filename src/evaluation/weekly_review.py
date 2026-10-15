"""The live-test review: how the served PPR projections did, scored honestly.

Production means what the site served before kickoff: ``predicted_points``
and its 80% range. What is scored is the version committed before each game's
kickoff (``published_record``), never the files on disk, which are regenerated
after games are played.

Population. The accuracy measures cover projected players with a stat line in
player_weekly_stats (zero and negative scores included). Every other published
projection is counted by ``accounting`` and classified from roster and snap
evidence; none is ever scored as zero or mixed into the accuracy measures.

Conventions. error = projected - actual, so a negative mean signed error means
projections ran low. The baseline difference is |model error| - |last-3
error| on the same player-weeks; negative favours the model.

Uncertainty. Players in one game share shocks, so every interval resamples
whole games (``mean_ci``): MAE, mean signed error, the paired difference and
range coverage alike, one method throughout, so the same rows always show the
same interval. With 16 games a week, intervals are approximate.

Only whitelisted columns survive ``player_rows``. The published files also
carry forecasts that are blinded candidates under
docs/PRODUCTION_SELECTION_RULE.md (the 2026 forward window); this module drops
them on read and never scores them. tests/test_weekly_review.py holds it to
that.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from src.evaluation.published_record import player_week

POSITIONS = ("QB", "RB", "WR", "TE")
PLAYER_COLS = ["week", "game", "kickoff", "published_commit", "published_at", "published_mode", "player_id", "name",
               "team", "position", "predicted_points", "prediction_ci80_lower",
               "prediction_ci80_upper", "actual_points", "final"]
B, SEED = 2000, 42
# NFL seasons completed before this one (nflverse years_exp): 0 = rookie.
EXPERIENCE = [("Rookie", 0, 0), ("Year 2\u20133", 1, 2), ("Year 4+", 3, 99)]
CAL_EDGES = [0, 5, 10, 15, 20, 25, np.inf]
TIER_EDGES = [0, 5, 10, 15, np.inf]
ERR_EDGES = np.arange(-27, 28, 3)
# What each published mode was, in words (docs/data/weekly_meta.json "mode").
MODES = {"season_prorated": "Season model \u00f7 games",
         "weekly_model": "Weekly model",
         "weekly_blend": "Weekly + pace blend"}
# Weekly-roster statuses that mean the player was not on the game-day active
# roster (nflverse weekly_rosters): with no offensive snap, a verified DNP.
NOT_ACTIVE = {"INA": "inactive", "RES": "reserve list", "DEV": "practice squad", "CUT": "released",
              "RET": "retired", "EXE": "exempt"}
VERIFIED_DNP, PLAYED_NO_STAT, UNRESOLVED = ("Verified did not play", "Played, no stat recorded", "Unresolved")
HIGH_PROJECTION = 10.0
STARTER_SNAPS = 0.6


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _keep(frame: pd.DataFrame, cols: list) -> pd.DataFrame:
    return frame[[c for c in cols if c in frame.columns]].copy()


def player_rows(con, season, weeks, games, now) -> tuple[pd.DataFrame, dict]:
    """Published player rows of every kicked-off game, whitelisted, plus the
    per-week summaries (for unprojected-player counts)."""
    parts, summaries = [], {}
    for wk in weeks:
        rows, summary = player_week(con, season, wk, games, now)
        if summary:
            summaries[wk] = summary
        if not rows.empty:
            parts.append(_keep(rows.assign(week=wk), PLAYER_COLS))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PLAYER_COLS)
    rows["final"] = rows["final"].astype(bool)
    rows = fill_ids(rows, con, season)
    rows = one_row_per_player_week(rows, con, season)
    # A game is one cluster for the bootstrap; division rivals meet twice a
    # season, so the label alone is not enough.
    rows["gkey"] = "wk" + rows["week"].astype(str) + " " + rows["game"].astype(str)
    return rows, summaries


def collect(con: sqlite3.Connection, season: int, weeks: list, games: pd.DataFrame, now) -> tuple[pd.DataFrame, dict]:
    """Every published row for `weeks`, with the baseline, experience and
    participation context the reviews use. Both review pages load through
    here, so they score identical rows."""
    rows, summaries = player_rows(con, season, weeks, games, now)
    rows = add_rolling3(rows, con, season)
    rows = add_experience(rows, con, season)
    rows = add_context(rows, con, season)
    return rows, summaries


def one_row_per_player_week(rows: pd.DataFrame, con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """A player traded or signed away between two snapshots of the same week
    is listed under both teams' games. Keep the row whose team matches that
    week's roster (weekly_rosters); with no roster row, keep the later
    snapshot. The number dropped is kept in rows.attrs."""
    ids = rows["player_id"].notna()
    dup = ids & rows.duplicated(["week", "player_id"], keep=False)
    if not dup.any():
        rows.attrs["moved_players_dropped"] = 0
        return rows
    ro = pd.read_sql("SELECT player_id, week, team AS roster_team FROM weekly_rosters WHERE season = ?",
                     con, params=[int(season)]).drop_duplicates(["player_id", "week"], keep="last")
    d = rows[dup].reset_index().merge(ro, on=["player_id", "week"], how="left")
    d["_match"] = d["team"] == d["roster_team"]
    d = d.sort_values(["_match", "published_at"], ascending=[False, False])
    keep = set(d.drop_duplicates(["week", "player_id"])["index"])
    drop = [i for i in d["index"] if i not in keep]
    out = rows.drop(index=drop)
    out.attrs["moved_players_dropped"] = len(drop)
    return out


SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV", "V"}


def short_name(name: str) -> str:
    """"Amon-Ra St. Brown" -> "A.St. Brown", the form week 1 was published in."""
    parts = [x for x in str(name).split() if x not in SUFFIXES]
    return f"{parts[0][0]}.{' '.join(parts[1:])}" if len(parts) > 1 else str(name)


def fill_ids(rows: pd.DataFrame, con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Week 1 was published without player ids, under short names. Recover the
    id from this season's roster where short name, position and team point to
    exactly one player; anything ambiguous stays without an id."""
    missing = rows["player_id"].isna() if "player_id" in rows else pd.Series(True, index=rows.index)
    if not missing.any():
        return rows
    ro = pd.read_sql("SELECT DISTINCT player_id, player_name, position, team FROM rosters WHERE season = ?",
                     con, params=[int(season)])
    ro["key"] = ro["player_name"].map(short_name) + "|" + ro["position"] + "|" + ro["team"]
    ro = ro.drop_duplicates(["player_id", "key"])
    ro = ro[~ro["key"].duplicated(keep=False)].set_index("key")["player_id"]
    key = rows["name"].astype(str) + "|" + rows["position"].astype(str) + "|" + rows["team"].astype(str)
    rows = rows.copy()
    rows.loc[missing, "player_id"] = key[missing].map(ro)
    return rows


def _attach(rows: pd.DataFrame, right: pd.DataFrame, on: list) -> pd.DataFrame:
    """Left-join `right` on `on` without ever matching a missing player id
    (pandas joins NaN to NaN) and without adding rows (a duplicated key on the
    right would score one projection twice)."""
    right = right.dropna(subset=["player_id"]).drop_duplicates(on, keep="last")
    out = rows.merge(right, on=on, how="left")
    out.loc[out["player_id"].isna(), [c for c in right.columns if c not in on]] = np.nan
    if len(out) != len(rows):
        raise ValueError(f"join on {on} changed the row count: {len(rows)} -> {len(out)}")
    return out


def scored(rows: pd.DataFrame) -> pd.DataFrame:
    """Final games, players with a stat line and a projection."""
    return rows[rows["final"] & rows["actual_points"].notna()
                & rows["predicted_points"].notna()].copy()


def add_rolling3(rows: pd.DataFrame, con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Each player's mean over his previous three games (crossing seasons),
    the naive reference. Only rows with a player id get one."""
    st = pd.read_sql("SELECT player_id, season, week, fantasy_points FROM player_weekly_stats "
                     "WHERE season <= ? AND fantasy_points IS NOT NULL", con, params=[int(season)])
    st = st.sort_values(["player_id", "season", "week"]).drop_duplicates(["player_id", "season", "week"])
    st["rolling3"] = (st.groupby("player_id")["fantasy_points"]
                      .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
    st = st[st["season"] == season][["player_id", "week", "rolling3"]]
    return _attach(rows.drop(columns=["rolling3"], errors="ignore"), st, ["player_id", "week"])


def add_experience(rows: pd.DataFrame, con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """years_exp from this season's roster; players missing from it stay NaN."""
    ex = pd.read_sql("SELECT player_id, MAX(years_exp) AS years_exp FROM rosters "
                     "WHERE season = ? GROUP BY player_id", con, params=[int(season)])
    return _attach(rows.drop(columns=["years_exp"], errors="ignore"), ex, ["player_id"])


def add_context(rows: pd.DataFrame, con: sqlite3.Connection, season: int) -> pd.DataFrame:
    """What else is known about each player-week:
    - snap_prior3: mean offensive snap share over the previous three games
      with a snap row (crossing seasons), known before kickoff;
    - snap_pct, participation_state, roster_status, has_snap_row and
      has_stats_row: this game, from canonical_player_weeks (known only
      afterwards; used to classify players with no stat line and to audit
      the stat join);
    - injury_status: the game-status designation on the injury report.
    Missing wherever the tables have no row (week 5 snaps load on Tuesday)."""
    can = pd.read_sql("SELECT player_id, season, week, participation_state, offense_pct, status AS roster_status, "
                      "has_snap_row, has_stats_row FROM canonical_player_weeks WHERE season BETWEEN ? AND ?",
                      con, params=[int(season) - 1, int(season)])
    can = can.drop_duplicates(["player_id", "season", "week"])
    cur = (can[can["season"] == season][["player_id", "week", "participation_state", "offense_pct",
                                         "roster_status", "has_snap_row", "has_stats_row"]]
           .rename(columns={"offense_pct": "snap_pct"}))
    snaps = can.dropna(subset=["offense_pct"]).assign(t=lambda d: d["season"] * 100 + d["week"])
    snaps = snaps.sort_values(["player_id", "t"])
    snaps["snap_prior3"] = snaps.groupby("player_id")["offense_pct"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    inj = pd.read_sql("SELECT player_id, week, report_status AS injury_status FROM player_injuries "
                      "WHERE season = ? AND COALESCE(report_status, '') != ''", con, params=[int(season)])
    out = rows.drop(columns=[c for c in ("participation_state", "snap_pct", "snap_prior3", "injury_status",
                                         "roster_status", "has_snap_row", "has_stats_row") if c in rows.columns])
    out = _attach(out, cur, ["player_id", "week"])
    out = _attach(out, inj, ["player_id", "week"])
    out["_t"] = int(season) * 100 + out["week"]
    has = out["player_id"].notna()
    # Strictly before this game: the latest earlier snap row's 3-game mean.
    prior = pd.merge_asof(out.loc[has, ["player_id", "_t"]].reset_index().sort_values("_t"),
                          snaps[["player_id", "t", "snap_prior3"]].rename(columns={"t": "_t"}).sort_values("_t"),
                          on="_t", by="player_id", direction="backward", allow_exact_matches=False)
    out["snap_prior3"] = prior.set_index("index")["snap_prior3"].reindex(out.index)
    return out.drop(columns="_t")


# ---------------------------------------------------------------------------
# Statistics: one interval method for every mean on the page
# ---------------------------------------------------------------------------

def mean_ci(values, clusters, b: int = B, seed: int = SEED) -> dict:
    """Mean of `values` with a 95% percentile interval from resampling whole
    games (clusters) with replacement, B times. Every interval in the report
    comes from here, so the same rows always get the same interval. Computed
    from per-game sums, which gives the same answer as concatenating the
    resampled games. Fewer than two games: no interval."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = ~np.isnan(v)
    v, c = v[ok], c[ok]
    if not len(v):
        return {"est": None, "lo": None, "hi": None, "n": 0, "games": 0}
    codes, uniq = pd.factorize(c)
    k = len(uniq)
    out = {"est": float(v.mean()), "lo": None, "hi": None, "n": int(len(v)), "games": int(k)}
    if k < 2:
        return out
    sums, cnts = np.bincount(codes, weights=v, minlength=k), np.bincount(codes, minlength=k)
    w = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=b)
    sims = (w @ sums) / np.maximum(w @ cnts, 1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return {**out, "lo": float(lo), "hi": float(hi)}


def error(d: pd.DataFrame) -> pd.Series:
    """Signed error, projected minus actual: negative = projected low."""
    return d["predicted_points"] - d["actual_points"]


def with_range(d: pd.DataFrame) -> pd.DataFrame:
    return d.dropna(subset=["prediction_ci80_lower", "prediction_ci80_upper"])


def paired(d: pd.DataFrame) -> pd.DataFrame:
    """Player-weeks that have both the model's projection and a last-3 baseline."""
    return d.dropna(subset=["rolling3"])


def stats_for(d: pd.DataFrame, label: str | None = None) -> dict:
    """Every measure the report shows for one group of scored player-weeks.
    Headline, version table, failure panel and appendix all call this, so a
    group's numbers and intervals agree wherever they appear."""
    g = d["gkey"]
    err = error(d)
    r = with_range(d)
    p = paired(d)
    p_err = error(p).abs()
    base_err = (p["rolling3"] - p["actual_points"]).abs()
    return {
        "label": label, "n": int(len(d)), "games": int(g.nunique()),
        "mae": mean_ci(err.abs(), g),
        "bias": mean_ci(err, g),
        "n_range": int(len(r)),
        "cov80": mean_ci(((r["actual_points"] >= r["prediction_ci80_lower"])
                          & (r["actual_points"] <= r["prediction_ci80_upper"])).astype(float), r["gkey"]),
        "above": mean_ci((r["actual_points"] > r["prediction_ci80_upper"]).astype(float), r["gkey"]),
        "below": mean_ci((r["actual_points"] < r["prediction_ci80_lower"]).astype(float), r["gkey"]),
        "width": float((r["prediction_ci80_upper"] - r["prediction_ci80_lower"]).mean()) if len(r) else None,
        "n_paired": int(len(p)),
        "model_mae_paired": mean_ci(p_err, p["gkey"]),
        "base_mae": mean_ci(base_err, p["gkey"]),
        "vs_rolling3": mean_ci(p_err - base_err, p["gkey"]),
    }


def _with_gkey(rows: pd.DataFrame) -> pd.DataFrame:
    if "gkey" in rows.columns:
        return rows
    return rows.assign(gkey="wk" + rows["week"].astype(str) + " " + rows["game"].astype(str))


# ---------------------------------------------------------------------------
# Every published projection
# ---------------------------------------------------------------------------

def classify_no_stat(u: pd.DataFrame) -> pd.DataFrame:
    """For projected players with no stat line, what the evidence says.
    - Played, no stat recorded: offensive snaps on the snap sheet.
    - Verified did not play: zero offensive snaps on the snap sheet, or not on
      the game-day active roster (inactive, reserve, practice squad,
      released, retired, exempt) with no snap.
    - Unresolved: anything else -- active with no snap row, no roster or snap
      record, or no player id. Never assumed to be a zero."""
    state = u.get("participation_state", pd.Series(index=u.index, dtype=object))
    status = u.get("roster_status", pd.Series(index=u.index, dtype=object))
    played = state.eq("confirmed_played")
    off = state.eq("confirmed_zero_snaps") | (status.isin(list(NOT_ACTIVE)) & ~played)
    out = pd.DataFrame(index=u.index)
    out["class"] = np.where(played, PLAYED_NO_STAT, np.where(off, VERIFIED_DNP, UNRESOLVED))
    reason = np.where(u["player_id"].isna(), "no player id (week 1 names)",
                      np.where(state.isna() & status.isna(), "no roster or snap record",
                               np.where(status.eq("ACT"), "active, no snap row", "other")))
    out["reason"] = np.where(out["class"] == UNRESOLVED, reason, None)
    return out


def accounting(rows: pd.DataFrame, weeks: list, summaries: dict) -> dict:
    """Every projection published for a final game in `weeks`: scored, or
    counted here by what happened. Nothing here enters an accuracy measure."""
    r = rows[rows["week"].isin(weeks) & rows["final"] & rows["predicted_points"].notna()]
    has_stat = r["actual_points"].notna()
    u = r[~has_stat]
    cls = classify_no_stat(u)
    hi = u["predicted_points"] >= 5
    classes = []
    for label in (VERIFIED_DNP, PLAYED_NO_STAT, UNRESOLVED):
        m = cls["class"] == label
        row = {"label": label, "n": int(m.sum()), "projected5": int((m & hi).sum())}
        if label == UNRESOLVED:
            row["reasons"] = {k: int(v) for k, v in cls.loc[m, "reason"].value_counts().items()}
        if label == VERIFIED_DNP:
            status = u.get("roster_status", pd.Series(index=u.index, dtype=object))
            state = u.get("participation_state", pd.Series(index=u.index, dtype=object))
            st = status[m].map(NOT_ACTIVE)
            st = st.where(~state[m].eq("confirmed_zero_snaps"), "zero offensive snaps")
            row["reasons"] = {k: int(v) for k, v in st.fillna("other").value_counts().items()}
        classes.append(row)
    listed = u.get("injury_status", pd.Series(index=u.index, dtype=object)).isin(["Out", "Doubtful"])
    return {"weeks": list(weeks), "published": int(len(r)), "scored": int(has_stat.sum()),
            "no_stat_line": int(len(u)), "no_stat_projected5": int(hi.sum()),
            "no_stat_projected5_listed_out": int((hi & listed).sum()),
            "classes": classes,
            "unprojected": sum((summaries.get(w) or {}).get("n_unprojected", 0) for w in weeks),
            "unprojected_5plus": sum((summaries.get(w) or {}).get("n_unprojected_5plus", 0) for w in weeks)}


def test_set(rows: pd.DataFrame, games: pd.DataFrame, weeks: list) -> list[dict]:
    """Week by week: the model and commit each game's projections came from,
    and whether every scheduled game has player results yet."""
    out = []
    for w in weeks:
        r = rows[rows["week"] == w]
        fin = r[r["final"] & r["predicted_points"].notna()]
        sc = scored(r)
        sched = int((games["week"] == w).sum())
        snaps = []
        if "published_commit" in r.columns and not r.empty:
            for (commit, at, mode), d in r.groupby(["published_commit", "published_at", "published_mode"],
                                                    dropna=False):
                snaps.append({"commit": commit, "published_at": at, "mode": mode,
                              "model": MODES.get(mode, mode), "games": int(d["game"].nunique())})
        n_games = int(sc["gkey"].nunique()) if len(sc) else 0
        out.append({"week": int(w), "games": sched, "games_scored": n_games,
                    "published": int(len(fin)), "scored": int(len(sc)),
                    "no_stat_line": int(fin["actual_points"].isna().sum()),
                    "with_range": int(sc["prediction_ci80_lower"].notna().sum()),
                    "snapshots": sorted(snaps, key=lambda x: str(x["published_at"])),
                    "complete": bool(sched) and n_games == sched})
    return out


def version_table(rows: pd.DataFrame, ts: list) -> dict:
    """Accuracy by week and served model version, so a headline that pools
    different models says so; weeks still being scored are flagged."""
    s = scored(_with_gkey(rows))
    complete = [w["week"] for w in ts if w["complete"]]
    out = []
    for w in ts:
        d = s[s["week"] == w["week"]]
        for mode, dm in d.groupby("published_mode", sort=False):
            out.append({**stats_for(dm, MODES.get(mode, mode)), "week": w["week"], "mode": mode,
                        "complete": w["complete"], "games_scheduled": w["games"]})
    pooled = []
    for mode in MODES:
        dm = s[s["week"].isin(complete) & (s["published_mode"] == mode)]
        if len(dm):
            pooled.append({**stats_for(dm, MODES[mode]), "mode": mode,
                           "weeks": sorted(int(x) for x in dm["week"].unique())})
    return {"rows": sorted(out, key=lambda x: (x["week"], str(x["mode"]) != "weekly_model")),
            "pooled": pooled, "complete_weeks": complete,
            "versions_in_headline": sorted({x["mode"] for x in out if x["complete"]})}


# ---------------------------------------------------------------------------
# The player section
# ---------------------------------------------------------------------------

def player_section(rows: pd.DataFrame, week: int, summaries: dict, weeks: list | None = None) -> dict:
    """Measures for `week`, or pooled over `weeks` (the season view passes
    complete weeks only)."""
    rows = _with_gkey(rows)
    s = scored(rows)
    in_scope = list(weeks) if weeks else [week]
    wk = s[s["week"].isin(in_scope)]
    head = stats_for(wk, "All")
    by_pos = [stats_for(wk[wk["position"] == p], p) for p in POSITIONS if (wk["position"] == p).any()]
    out = {"week": week, "weeks": in_scope, "n": head["n"], "games_final": head["games"], "headline": head,
           "accounting": accounting(rows, in_scope, summaries), "by_position": by_pos,
           "failure_panel": failure_panel(wk), "segments": segments(wk, by_pos), "splits": splits(wk),
           "calibration": calibration(wk), "distribution": distribution(paired(wk))}
    no_range = wk[wk["prediction_ci80_lower"].isna()]
    out["ranges"] = {"rows": [x for x in by_pos + [head] if x["n_range"]], "n_without": int(len(no_range)),
                     "weeks_without": sorted(int(w) for w in no_range["week"].unique())}
    out["by_week"] = []
    for w, d in s.groupby("week"):
        kicked = int(rows.loc[rows["week"] == w, "gkey"].nunique())
        out["by_week"].append({**stats_for(d, f"Week {int(w)}"), "week": int(w),
                               "partial": int(d["gkey"].nunique()) < kicked})
    out["points"] = wk[["position", "predicted_points", "actual_points", "prediction_ci80_lower",
                        "prediction_ci80_upper", "name"]].to_dict("records")
    return out


def failure_panel(wk: pd.DataFrame) -> list[dict]:
    """The four places the evaluation points at, as one compact panel."""
    groups = [("QB", wk["position"] == "QB"), ("TE", wk["position"] == "TE"),
              (f"Projected {HIGH_PROJECTION:g}+ points", wk["predicted_points"] >= HIGH_PROJECTION)]
    if "snap_prior3" in wk.columns:
        groups.append((f"Likely starters ({STARTER_SNAPS:.0%}+ of snaps, previous 3 games)",
                       wk["snap_prior3"] >= STARTER_SNAPS))
    return [stats_for(wk[m], label) for label, m in groups if m.any()]


def segments(wk: pd.DataFrame, by_pos: list | None = None) -> list[dict]:
    """Where the error sits: position, projected points and NFL experience."""
    pos = by_pos if by_pos is not None else [stats_for(wk[wk["position"] == p], p) for p in POSITIONS
                                             if (wk["position"] == p).any()]
    tiers = []
    for lo, hi in zip(TIER_EDGES[:-1], TIER_EDGES[1:]):
        m = (wk["predicted_points"] >= lo) & (wk["predicted_points"] < hi)
        if m.any():
            tiers.append(stats_for(wk[m], f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"{lo:g}+"))
    groups = [("Position", pos), ("Projected points", tiers)]
    if "years_exp" in wk.columns:
        groups.append(("Experience", [stats_for(wk[wk["years_exp"].between(lo, hi)], lab)
                                      for lab, lo, hi in EXPERIENCE if wk["years_exp"].between(lo, hi).any()]))
    return [{"group": g, "rows": r} for g, r in groups if r]


def splits(wk: pd.DataFrame) -> list[dict]:
    """Fuller splits for the appendix: usage going in, a role change on the
    day, and the injury report."""
    total = float(error(wk).abs().sum())
    share = lambda d: float(error(d).abs().sum() / total) if total else None
    groups = []
    if "snap_prior3" in wk.columns:
        sp = wk["snap_prior3"]
        groups.append(("Snap share, previous 3 games (known before kickoff)", [
            (lab, m) for lab, m in [("60% or more", sp >= 0.6), ("25–60%", (sp >= 0.25) & (sp < 0.6)),
                                    ("Under 25%", sp < 0.25), ("No snap history", sp.isna())]]))
        change = wk["snap_pct"] - sp
        groups.append(("This game's snap share vs previous 3 (known after)", [
            (lab, m) for lab, m in [("Down 20+ points", change <= -0.2), ("Within 20 points", change.abs() < 0.2),
                                    ("Up 20+ points", change >= 0.2), ("Not known", change.isna())]]))
    if "injury_status" in wk.columns:
        st = wk["injury_status"].fillna("Not listed")
        groups.append(("Injury report", [(lab, st == lab) for lab in ("Questionable", "Doubtful", "Out", "Not listed")]))
    return [{"group": g, "rows": [{**stats_for(wk[m], lab), "share": share(wk[m])} for lab, m in rows if m.any()]}
            for g, rows in groups]


def calibration(wk: pd.DataFrame) -> list[dict]:
    """Mean actual against mean projection, in bands of projected points."""
    out = []
    for lo, hi in zip(CAL_EDGES[:-1], CAL_EDGES[1:]):
        d = wk[(wk["predicted_points"] >= lo) & (wk["predicted_points"] < hi)]
        if len(d) < 5:
            continue
        out.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "n": int(len(d)),
                    "projected": float(d["predicted_points"].mean()),
                    "actual": mean_ci(d["actual_points"], d["gkey"])})
    return out


def distribution(p: pd.DataFrame) -> dict:
    """Signed misses (projected minus actual) for the model and the last-3
    average on the same player-weeks, as shares per 3-point bin; tails
    clipped into the end bins."""
    if p.empty:
        return {}
    m = error(p).clip(ERR_EDGES[0], ERR_EDGES[-1] - 1e-9)
    r = (p["rolling3"] - p["actual_points"]).clip(ERR_EDGES[0], ERR_EDGES[-1] - 1e-9)
    hist = lambda x: (np.histogram(x, bins=ERR_EDGES)[0] / len(x)).round(4).tolist()
    out = {"n": int(len(p)), "edges": ERR_EDGES.tolist(), "model": hist(m), "rolling3": hist(r)}
    for k in (3, 5, 10):
        out[f"model_within{k}"] = float((m.abs() <= k).mean())
        out[f"rolling3_within{k}"] = float((r.abs() <= k).mean())
    return out


# ---------------------------------------------------------------------------
# Claims: every sentence is decided by an interval
# ---------------------------------------------------------------------------

def _side(ci: dict, ref: float = 0.0) -> str | None:
    """'below' / 'above' when the 95% interval excludes `ref`, else None."""
    if ci["lo"] is None:
        return None
    return "below" if ci["hi"] < ref else "above" if ci["lo"] > ref else None


def _pts(x: float, d: int = 2) -> str:
    return f"{abs(x):.{d}f}"


def conclusion(ps: dict) -> list[str]:
    """The main conclusion in two sentences, each claim decided by its
    interval. Sentence one: the three headline measures. Sentence two: the
    weak spots the failure panel shows, only where an interval says so."""
    h = ps["headline"]
    if not h["n"]:
        return ["No final game has stat lines yet, so nothing is scored."]
    if h["mae"]["lo"] is None:
        return [f"Results are in for only {h['games']} game so far ({h['n']} players): too few games to put an "
                "interval on anything yet."]
    d, b, c = h["vs_rolling3"], h["bias"], h["cov80"]
    if _side(d) == "below":
        base = (f"the projections missed by {_pts(d['est'])} points less than a last-three-game average on the "
                f"same player-weeks (95% interval {_pts(d['hi'])} to {_pts(d['lo'])} less)")
    elif _side(d) == "above":
        base = (f"the projections missed by {_pts(d['est'])} points more than a last-three-game average on the "
                f"same player-weeks (95% interval {_pts(d['lo'])} to {_pts(d['hi'])} more)")
    else:
        base = (f"the projections could not be told apart from a last-three-game average on the same "
                f"player-weeks (difference {d['est']:+.2f} points, 95% interval {d['lo']:+.2f} to {d['hi']:+.2f})")
    lean = {"below": f"ran low by {_pts(b['est'], 1)} points on average",
            "above": f"ran high by {_pts(b['est'], 1)} points on average"}.get(
        _side(b), f"showed no clear lean ({b['est']:+.1f} points)")
    cov = ""
    if c["est"] is not None and c["lo"] is not None:
        cov = {"below": "too narrow", "above": "too wide"}.get(_side(c, 0.80), "in line with the 80% target")
        cov = f", and their 80% ranges held {c['est']:.0%} of outcomes, {cov}"
    span = (f"weeks {ps['weeks'][0]}–{ps['weeks'][-1]}" if len(ps["weeks"]) > 1 else f"week {ps['weeks'][0]}")
    out = [f"Over {span}, {base}; they {lean}{cov}."]
    weak = []
    for x in ps["failure_panel"]:
        if x["label"] == "QB" and _side(x["bias"]) == "below":
            weak.append(f"QB projections ran {_pts(x['bias']['est'], 1)} points low")
        if x["label"] == "TE" and _side(x["cov80"], 0.80) == "below":
            weak.append(f"TE ranges held only {x['cov80']['est']:.0%}, and {x['above']['est']:.0%} of TE outcomes "
                        "landed above the range (a well-set range puts about 10% there)")
    if weak:
        out.append("Weak spots: " + "; ".join(weak) + ".")
    return out


# ---------------------------------------------------------------------------
# Audit: the checks the generator runs on its own numbers
# ---------------------------------------------------------------------------

def _intervals(obj, path=""):
    """Every {est, lo, hi} interval inside a nested result."""
    if isinstance(obj, dict):
        if {"est", "lo", "hi"} <= set(obj):
            yield path, obj
        for k, v in obj.items():
            yield from _intervals(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _intervals(v, f"{path}[{i}]")


def audit(rows: pd.DataFrame, ps: dict, ts: list | None = None) -> list[dict]:
    """Checks on the scoring joins, timestamps, denominators, signs and
    intervals. Each returns ok plus the counts behind it."""
    rows = _with_gkey(rows)
    scope = rows[rows["week"].isin(ps["weeks"]) & rows["final"] & rows["predicted_points"].notna()]
    sc = scope[scope["actual_points"].notna()]
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    if {"published_at", "kickoff"} <= set(scope.columns):
        lead = (pd.to_datetime(scope["kickoff"], utc=True) - pd.to_datetime(scope["published_at"], utc=True))
        late = int((lead <= pd.Timedelta(0)).sum())
        add("Every projection was committed before its game's kickoff", late == 0 and lead.notna().all(),
            f"{late} of {len(scope):,} at or after kickoff; shortest lead "
            f"{lead.min().total_seconds() / 3600:.1f} hours" if len(scope) else "no rows")
    ids = scope.dropna(subset=["player_id"])
    dup = int(ids.duplicated(["week", "player_id"]).sum())
    add("One projection per player per week", dup == 0,
        f"{dup} duplicates; {int(scope['player_id'].isna().sum())} rows without a player id "
        f"({int(sc['player_id'].isna().sum())} of them scored, joined by name)")
    teams = sc["game"].astype(str).str.split("@")
    wrong = int(sum(t not in g for t, g in zip(sc["team"], teams)))
    add("Each scored player's team played in that game", wrong == 0, f"{wrong} mismatches in {len(sc):,}")
    if "has_stats_row" in scope.columns:
        known = scope["has_stats_row"].notna()
        a = int((known & scope["actual_points"].notna() & scope["has_stats_row"].eq(0)).sum())
        b_ = int((known & scope["actual_points"].isna() & scope["has_stats_row"].eq(1)).sum())
        add("Stat-line join agrees with the participation table", a == 0 and b_ == 0,
            f"{a} scored without a stats row there, {b_} unscored with one; {int((~known).sum())} rows not "
            "in that table")
    acc = ps["accounting"]
    parts = acc["scored"] + sum(c["n"] for c in acc["classes"])
    add("Every published projection is accounted for",
        acc["published"] == acc["scored"] + acc["no_stat_line"] == parts and acc["scored"] == ps["n"],
        f"{acc['published']:,} published = {acc['scored']:,} scored + " +
        " + ".join(f"{c['n']:,} {c['label'].lower()}" for c in acc["classes"]))
    if ts is not None:
        complete = {w["week"] for w in ts if w["complete"]}
        add("Headline uses complete weeks only", set(ps["weeks"]) <= complete,
            f"headline weeks {ps['weeks']}; complete {sorted(complete)}")
    h = ps["headline"]
    add("Model and baseline compared on identical player-weeks",
        h["model_mae_paired"]["n"] == h["base_mae"]["n"] == h["vs_rolling3"]["n"] == h["n_paired"],
        f"{h['n_paired']:,} player-weeks with a last-3 history, of {h['n']:,} scored")
    gap = abs(h["model_mae_paired"]["est"] - h["base_mae"]["est"] - h["vs_rolling3"]["est"]) if h["n_paired"] else 0
    add("Paired difference equals the difference of the two MAEs", gap < 1e-9, f"gap {gap:.1e}")
    bias = float(error(sc).mean()) if len(sc) else 0.0
    add("Mean signed error is projected minus actual", abs(bias - (h["bias"]["est"] or 0)) < 1e-9,
        f"recomputed {bias:+.3f}; negative = projected low")
    pos = {x["label"]: x for x in ps["by_position"]}
    table = {x["label"]: x for x in ps["ranges"]["rows"]}
    same = (all(pos[x["label"]]["cov80"] == x["cov80"] for g in ps["segments"] if g["group"] == "Position"
                for x in g["rows"])
            and all(table[k]["cov80"] == pos[k]["cov80"] for k in table if k in pos)
            and ("All" not in table or table["All"]["cov80"] == h["cov80"]))
    add("Same rows show the same interval in every section", same,
        "position coverage (chart vs table) and overall coverage (headline vs table)")
    bad = [p for p, ci in _intervals(ps) if ci["lo"] is not None and not ci["lo"] - 1e-9 <= ci["est"] <= ci["hi"] + 1e-9]
    n_ci = sum(1 for _, ci in _intervals(ps) if ci["lo"] is not None)
    add("Every interval contains its estimate", not bad, f"{len(bad)} of {n_ci:,} outside" +
        (f": {', '.join(bad[:3])}" if bad else ""))
    return checks

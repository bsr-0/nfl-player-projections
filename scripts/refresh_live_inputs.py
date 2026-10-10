#!/usr/bin/env python3
"""Load the nflverse tables the live model reads but the weekly refresh does not.

For 2026 these had no rows: weekly_pfr, ngs_passing/receiving/rushing and
snap_counts, and seasonal_pfr had nothing after 2024 (the *_prior features of
2026 read 2025). Together they feed 14 of the served model's 100 features, which
were zero or missing on live rows and populated in training and replays
(GAPS.md, 2026-10-09).

Each table is a direct dump of an nflverse release, so this only appends rows
whose key is absent (existing rows are never changed). Before writing, the same
fetch is repeated for a season that is already stored and must reproduce it
exactly; a table that does not is skipped.

    python scripts/refresh_live_inputs.py --seasons 2026                 # dry run
    python scripts/refresh_live_inputs.py --seasons 2026 --write
    python scripts/refresh_live_inputs.py --validate-only

Run it weekly after the stats refresh (it is idempotent). Only weeks already in
player_weekly_stats are loaded, never the week-0 season aggregates. Seasonal PFR
is a prior-season table: --seasons 2026 loads 2025, not 2026.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import DB_PATH

NGS_KEY = ["season", "season_type", "week", "player_gsis_id"]
KEYS = {
    "weekly_pfr": ["season", "week", "pfr_player_id", "stat_type"],
    "ngs_passing": NGS_KEY, "ngs_receiving": NGS_KEY, "ngs_rushing": NGS_KEY,
    "snap_counts": ["season", "week", "pfr_player_id", "team", "game_id"],
    "seasonal_pfr": ["season", "pfr_player_id", "stat_type"],
}
# Validation season per table: stored, complete, and the same release format.
VALIDATE = {"weekly_pfr": 2025, "ngs_passing": 2025, "ngs_receiving": 2025, "ngs_rushing": 2025,
            "snap_counts": 2025, "seasonal_pfr": 2024}


def stored_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]


def fetch(table: str, season: int, columns: list[str]) -> pd.DataFrame:
    """The release for one season, shaped exactly like the stored table."""
    import nfl_data_py as nfl
    if table == "weekly_pfr":
        parts = []
        for stat_type in ("pass", "rush", "rec"):
            d = nfl.import_weekly_pfr(stat_type, [season])
            parts.append(d.assign(stat_type=stat_type, position=np.nan))
        df = pd.concat(parts, ignore_index=True)
    elif table.startswith("ngs_"):
        df = nfl.import_ngs_data(table.split("_", 1)[1], [season])
    elif table == "snap_counts":
        df = nfl.import_snap_counts([season])
    elif table == "seasonal_pfr":
        df = seasonal_pfr_frame(season)
    else:
        raise ValueError(table)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: upstream lacks {missing}")
    return df[columns].reset_index(drop=True)


def seasonal_pfr_frame(season: int) -> pd.DataFrame:
    """Upstream seasonal PFR in the stored table's shape (one frame, three stat types)."""
    import nfl_data_py as nfl
    p = nfl.import_seasonal_pfr("pass", [season])
    r = nfl.import_seasonal_pfr("rush", [season])
    c = nfl.import_seasonal_pfr("rec", [season])
    return pd.concat([
        pd.DataFrame({"season": p.season, "pfr_player_id": p.pfr_id, "player_name": p.player,
                      "team": p.team, "stat_type": "pass", "drop_pct": p.drop_pct,
                      "bad_throw_pct": p.bad_throw_pct, "pocket_time": p.pocket_time}),
        pd.DataFrame({"season": r.season, "pfr_player_id": r.pfr_id, "player_name": r.player,
                      "team": r.tm, "stat_type": "rush", "yards_before_contact_avg": r.ybc_att,
                      "yards_after_contact_avg": r.yac_att,
                      "broken_tackles_per_att": (r.brk_tkl / r.att.replace(0, np.nan)).astype(float)}),
        pd.DataFrame({"season": c.season, "pfr_player_id": c.pfr_id, "player_name": c.player,
                      "team": c.tm, "stat_type": "rec", "rec_drop_pct": c.drop_percent,
                      "rec_broken_tackles": c.brk_tkl}),
    ], ignore_index=True)


def _key_frame(df: pd.DataFrame, key: list[str]) -> pd.Series:
    return df[key].astype(object).where(df[key].notna(), "<NA>").astype(str).agg("|".join, axis=1)


def played_weeks_only(con: sqlite3.Connection, fetched: pd.DataFrame, season: int) -> pd.DataFrame:
    """Keep weeks whose stats are loaded, and never the week-0 season aggregates.

    The aggregates are unused by the model (poisoning them changes no prediction,
    data/experiments/serving_leakage_audit) and run to "now". Tables without a
    week column are returned unchanged.
    """
    if "week" not in fetched.columns:
        return fetched
    last = con.execute("SELECT MAX(week) FROM player_weekly_stats WHERE season = ?", (season,)).fetchone()[0]
    return fetched[(fetched.week >= 1) & (fetched.week <= (last or 0))]


def new_rows(stored: pd.DataFrame, fetched: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Fetched rows whose key is not stored. Duplicate keys in the fetch are kept once."""
    have = set(_key_frame(stored, key)) if len(stored) else set()
    k = _key_frame(fetched, key)
    return fetched[~k.isin(have) & ~k.duplicated()].reset_index(drop=True)


def compare(stored: pd.DataFrame, fetched: pd.DataFrame, key: list[str]) -> dict:
    """How well a re-fetch reproduces what is stored (shared keys only).

    A cell is `revised` when both sides hold a value and they differ (nflverse
    recomputes some releases after the fact), and a `null_mismatch` when exactly
    one side is missing, which would mean the column mapping is wrong.
    """
    s = stored.assign(_k=_key_frame(stored, key)).drop_duplicates("_k").set_index("_k")
    f = fetched.assign(_k=_key_frame(fetched, key)).drop_duplicates("_k").set_index("_k")
    both = s.index.intersection(f.index)
    revised, null_mismatch = {}, {}
    for c in [c for c in stored.columns if c in fetched.columns and c not in key]:
        a, b = s.loc[both, c], f.loc[both, c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a, b = a.astype(float), b.astype(float)
            same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            a, b = a.astype(object), b.astype(object)
            same = (a.astype(str).where(a.notna(), "") == b.astype(str).where(b.notna(), "")).to_numpy()
        one_missing = (a.isna().to_numpy() ^ b.isna().to_numpy())
        if one_missing.any():
            null_mismatch[c] = int(one_missing.sum())
        changed = ~np.asarray(same) & ~one_missing
        if changed.any():
            revised[c] = int(changed.sum())
    return {"stored": len(s), "fetched": len(f), "shared": len(both),
            "only_stored": len(s.index.difference(f.index)), "only_fetched": len(f.index.difference(s.index)),
            "revised": revised, "null_mismatch": null_mismatch}


def validate(con: sqlite3.Connection, table: str) -> dict:
    season = VALIDATE[table]
    cols = stored_columns(con, table)
    stored = pd.read_sql(f"SELECT {','.join(cols)} FROM {table} WHERE season = ?", con, params=[season])
    result = compare(stored, fetch(table, season, cols), KEYS[table])
    # What must hold: nearly every stored row is reproduced and no cell is missing on
    # one side only. Value revisions are reported, not failures: nflverse recomputes
    # some releases (NGS), and a new row is appended once and never revised here.
    result["ok"] = not result["null_mismatch"] and result["shared"] >= 0.99 * max(result["stored"], 1)
    result["season"] = season
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[])
    ap.add_argument("--tables", nargs="+", default=list(KEYS), choices=list(KEYS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    if not a.seasons and not a.validate_only:
        ap.error("give --seasons or --validate-only")

    plan, failed = {}, []
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        for table in a.tables:
            v = validate(con, table)
            print(f"validate {table} {v['season']}: stored {v['stored']}, re-fetched {v['fetched']}, shared {v['shared']}, "
                  f"only stored {v['only_stored']}, only fetched {v['only_fetched']}, "
                  f"null mismatches {sum(v['null_mismatch'].values())}, revised cells {sum(v['revised'].values())} "
                  f"-> {'OK' if v['ok'] else 'FAILED'}")
            if not v["ok"]:
                failed.append(table)
                continue
            if a.validate_only:
                continue
            cols = stored_columns(con, table)
            # Seasonal PFR holds finished seasons: 2026 reads 2025, and an in-season
            # partial aggregate would never be revised once its key exists.
            seasons = sorted({s - 1 for s in a.seasons}) if table == "seasonal_pfr" else a.seasons
            for season in seasons:
                stored = pd.read_sql(f"SELECT {','.join(KEYS[table])} FROM {table} WHERE season = ?", con, params=[season])
                try:
                    fetched = fetch(table, season, cols)
                except Exception as e:  # not published yet is a normal state
                    print(f"  {table} {season}: not available ({type(e).__name__}: {str(e)[:70]})")
                    continue
                fetched = played_weeks_only(con, fetched, season)
                add = new_rows(stored, fetched, KEYS[table])
                weeks = sorted(add.week.dropna().astype(int).unique().tolist()) if "week" in add else []
                print(f"  {table} {season}: upstream {len(fetched)}, stored {len(stored)}, to add {len(add)}"
                      + (f", weeks {weeks}" if weeks else ""))
                if len(add):
                    plan[(table, season)] = add
    if failed:
        print(f"skipped (did not reproduce stored data): {failed}")
    if not a.write or a.validate_only or not plan:
        print("dry run: nothing written" if not a.write else "nothing to write")
        return
    backup = DB_PATH.parent / "backups" / f"nfl_data_pre_live_inputs_{datetime.now():%Y%m%d_%H%M%S}.db"
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con, sqlite3.connect(str(backup)) as dst:
        con.backup(dst)
        print(f"backup: {backup}")
        for (table, season), add in plan.items():
            add.to_sql(table, con, if_exists="append", index=False)
            print(f"  wrote {len(add)} rows to {table} ({season})")
        con.commit()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        for (table, season) in plan:
            cols = stored_columns(con, table)
            stored = pd.read_sql(f"SELECT {','.join(KEYS[table])} FROM {table} WHERE season = ?", con, params=[season])
            left = new_rows(stored, played_weeks_only(con, fetch(table, season, cols), season), KEYS[table])
            print(f"after write: {table} {season} still missing {len(left)}")


if __name__ == "__main__":
    main()

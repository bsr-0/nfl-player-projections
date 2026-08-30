"""Backfill the seasonal_pfr columns that were only ever populated for 2024.

Measured 2026-08-29: in `seasonal_pfr`, `broken_tackles_per_att` and
`rec_drop_pct` were non-null for 2024 ONLY (152 and 494 rows). Because
`_add_seasonal_pfr_features` shifts season +1 (year N predicts year N+1), that
single populated year landed on target season 2025 -- which is exactly why
`rb_broken_tackles_prior` and `recv_drop_pct_season_prior` read ~98% zero for
2019-2024 and real for 2025.

The data was always available upstream. nfl_data_py.import_seasonal_pfr carries
`brk_tkl`/`att` (rush) and `drop_percent` (rec) at ~100% coverage for every
season from 2018. Nothing in this repo ever wrote them; the table was populated
ad hoc.

Usage:
    python scripts/backfill_seasonal_pfr.py --dry-run
    python scripts/backfill_seasonal_pfr.py --start 2018 --end 2024
"""
import argparse
import shutil
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import DB_PATH

# Sanity bounds: a backfill must not replace one fabrication with another.
BOUNDS = {
    "broken_tackles_per_att": (0.0, 1.0),
    "rec_drop_pct": (0.0, 100.0),
}


def backup_db() -> Path:
    dest = DB_PATH.with_name(
        f"{DB_PATH.stem}.prepfr_{datetime.now():%Y%m%d_%H%M%S}.db")
    shutil.copy2(DB_PATH, dest)
    return dest


def build_frames(years):
    import nfl_data_py as nfl

    rush = nfl.import_seasonal_pfr("rush", years)
    rush["broken_tackles_per_att"] = (
        rush["brk_tkl"] / rush["att"].replace(0, pd.NA)
    ).astype(float)
    rush = rush[["season", "pfr_id", "broken_tackles_per_att"]].rename(
        columns={"pfr_id": "pfr_player_id"})
    rush["stat_type"] = "rush"

    rec = nfl.import_seasonal_pfr("rec", years)
    drop_col = "drop_percent" if "drop_percent" in rec.columns else "drop_pct"
    rec["rec_drop_pct"] = rec[drop_col].astype(float)
    rec = rec[["season", "pfr_id", "rec_drop_pct"]].rename(
        columns={"pfr_id": "pfr_player_id"})
    rec["stat_type"] = "rec"

    return rush, rec


def check_bounds(df, col, season_label):
    if col not in df.columns:
        return []
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return [f"{season_label} {col}: no values"]
    lo, hi = BOUNDS[col]
    if s.min() < lo or s.max() > hi:
        return [f"{season_label} {col}: [{s.min():.3f}, {s.max():.3f}] "
                f"outside [{lo}, {hi}]"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    years = list(range(args.start, args.end + 1))
    rush, rec = build_frames(years)

    problems = (check_bounds(rush, "broken_tackles_per_att", "rush")
                + check_bounds(rec, "rec_drop_pct", "rec"))
    if problems:
        print("BOUNDS VIOLATION, refusing to write:")
        for p in problems:
            print("   ", p)
        return 1

    print(f"upstream rows: rush={len(rush)}, rec={len(rec)}")
    for name, df, col in (("rush", rush, "broken_tackles_per_att"),
                          ("rec", rec, "rec_drop_pct")):
        cov = 100 * df.groupby("season")[col].apply(lambda s: s.notna().mean())
        print(f"  {name} {col} % non-null by season: "
              + ", ".join(f"{int(y)}={v:.0f}%" for y, v in cov.items()))

    if args.dry_run:
        con = sqlite3.connect(DB_PATH)
        for stat_type, df, col in (("rush", rush, "broken_tackles_per_att"),
                                   ("rec", rec, "rec_drop_pct")):
            n = con.execute(
                f"SELECT COUNT(*) FROM seasonal_pfr WHERE stat_type=? "
                f"AND season BETWEEN ? AND ? AND {col} IS NULL",
                (stat_type, args.start, args.end)).fetchone()[0]
            print(f"  would fill up to {n} NULL {col} rows ({stat_type})")
        con.close()
        print("\nDRY RUN -- nothing written")
        return 0

    dest = backup_db()
    print(f"\nDB backed up -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")

    con = sqlite3.connect(DB_PATH)
    total = 0
    for stat_type, df, col in (("rush", rush, "broken_tackles_per_att"),
                               ("rec", rec, "rec_drop_pct")):
        cur = con.cursor()
        n = 0
        for _, r in df.iterrows():
            if pd.isna(r[col]):
                continue
            cur.execute(
                f"UPDATE seasonal_pfr SET {col} = ? "
                f"WHERE pfr_player_id = ? AND season = ? AND stat_type = ?",
                (float(r[col]), r["pfr_player_id"], int(r["season"]), stat_type))
            n += cur.rowcount
        con.commit()
        print(f"  {stat_type}: {n} rows updated ({col})")
        total += n

    print(f"\ndone: {total} rows updated")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

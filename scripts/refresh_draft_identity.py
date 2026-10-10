#!/usr/bin/env python3
"""Replace stale draft-feed ids in `draft_picks_v2` with the official GSIS ids.

Before a rookie plays, the draft feed identifies him by a placeholder
(`MEN516487`) or not at all, and `draft_picks_v2` stores that as `player_id`.
nflverse swaps in the official GSIS id (`00-0041562`) once he appears in game
data. Everything that joins draft capital to a player does so on `player_id`
(the training and serving frames, `_merge_draft_capital`, the rookie priors), so
a debuted rookie whose row still holds the placeholder has no draft round, pick
or capital at all. For the 2026 class that was 49 players with stats.

Only the `player_id` of a row whose id is not already a GSIS id is changed:

  * rows are matched on (draft_season, draft_round, draft_pick), unique on both
    sides, and a match whose position or PFR id disagrees is left alone;
  * an existing GSIS id is never overwritten (a different one is reported);
  * an id that another pick already carries is not assigned twice.

    python scripts/refresh_draft_identity.py --seasons 2026            # dry run
    python scripts/refresh_draft_identity.py --seasons 2026 --write

Run it weekly after the stats refresh (idempotent): ids for players who debut
later arrive as nflverse publishes them.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import DB_PATH

KEY = ["draft_season", "draft_round", "draft_pick"]
GSIS = re.compile(r"^00-\d{7}$")


def is_gsis(value) -> bool:
    return isinstance(value, str) and bool(GSIS.match(value))


def plan_updates(local: pd.DataFrame, upstream: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The rows to change and why every other row stays.

    `local` needs rowid, KEY, player_id, position, pfr_player_id; `upstream` KEY,
    gsis_id, position, pfr_player_id. Returns (updates with rowid/old/new, counts).
    """
    m = local.merge(upstream[KEY + ["gsis_id", "position", "pfr_player_id"]], on=KEY, how="left",
                    suffixes=("", "_up"), indicator="upstream_match")
    if len(m) != len(local):
        raise ValueError("(draft_season, draft_round, draft_pick) is not unique upstream")
    carried = m.loc[m.player_id.map(is_gsis), "player_id"]
    skipped = {"no_upstream_row": 0, "identity_mismatch": 0, "no_gsis_upstream": 0,
               "already_gsis": 0, "conflict": 0, "id_in_use": 0}
    rows = []
    claimed = set(carried)
    for r in m.itertuples(index=False):
        if r.upstream_match == "left_only":
            skipped["no_upstream_row"] += 1
        elif (pd.notna(r.position) and pd.notna(r.position_up) and r.position != r.position_up) or \
                (pd.notna(r.pfr_player_id) and pd.notna(r.pfr_player_id_up) and r.pfr_player_id != r.pfr_player_id_up):
            skipped["identity_mismatch"] += 1
        elif not is_gsis(r.gsis_id):
            skipped["no_gsis_upstream"] += 1
        elif is_gsis(r.player_id):
            skipped["already_gsis" if r.player_id == r.gsis_id else "conflict"] += 1
        elif r.gsis_id in claimed:
            skipped["id_in_use"] += 1
        else:
            claimed.add(r.gsis_id)
            rows.append((r.rowid, r.player_id, r.gsis_id, r.position, r.draft_round, r.draft_pick))
    return (pd.DataFrame(rows, columns=["rowid", "old_id", "new_id", "position", "draft_round", "draft_pick"]),
            skipped)


def apply_updates(con: sqlite3.Connection, updates: pd.DataFrame) -> int:
    """Set the ids. The WHERE clause keeps a GSIS id from ever being overwritten."""
    before = con.total_changes
    con.executemany(
        "UPDATE draft_picks_v2 SET player_id = ? WHERE rowid = ? "
        "AND (player_id IS NULL OR player_id NOT LIKE '00-%')",
        [(r.new_id, int(r.rowid)) for r in updates.itertuples()])
    return con.total_changes - before


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", required=True, help="draft classes to refresh")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    import nfl_data_py as nfl
    up = nfl.import_draft_picks(a.seasons).rename(
        columns={"season": "draft_season", "round": "draft_round", "pick": "draft_pick"})
    if up.empty or up.duplicated(KEY).any():
        raise SystemExit("upstream draft picks are empty or not unique on (season, round, pick)")

    marks = ",".join("?" * len(a.seasons))
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        local = pd.read_sql(f"SELECT rowid, * FROM draft_picks_v2 WHERE draft_season IN ({marks})",
                            con, params=a.seasons)
    if local.duplicated(KEY).any():
        raise SystemExit("draft_picks_v2 is not unique on (season, round, pick); an update could fan out")
    updates, skipped = plan_updates(local, up)

    for season in a.seasons:
        n = int((local.draft_season == season).sum())
        have = int(local[local.draft_season == season].player_id.map(is_gsis).sum())
        print(f"{season}: local {n} picks, {have} with a GSIS id; upstream {int((up.draft_season == season).sum())} "
              f"picks, {int(up[up.draft_season == season].gsis_id.map(is_gsis).sum())} with one")
    skill = updates[updates.position.isin(["QB", "RB", "WR", "TE"])]
    print(f"to update: {len(updates)} ids ({len(skill)} QB/RB/WR/TE); left alone: {skipped}")
    if skipped["conflict"] or skipped["identity_mismatch"]:
        print("  note: conflicts and identity mismatches are reported, never changed")
    if not a.write or updates.empty:
        print("dry run: nothing written" if not a.write else "nothing to write")
        return

    backup = DB_PATH.parent / "backups" / f"nfl_data_pre_draft_identity_{datetime.now():%Y%m%d_%H%M%S}.db"
    backup.parent.mkdir(exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con, sqlite3.connect(str(backup)) as dst:
        con.backup(dst)
        print(f"backup: {backup}")
        changed = apply_updates(con, updates)
        con.commit()
    print(f"wrote {changed} ids")
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as con:
        after = pd.read_sql(f"SELECT rowid, * FROM draft_picks_v2 WHERE draft_season IN ({marks})",
                            con, params=a.seasons)
    left, _ = plan_updates(after, up)
    print(f"after write: {len(left)} ids still to update; row count {len(after)} (was {len(local)})")


if __name__ == "__main__":
    main()

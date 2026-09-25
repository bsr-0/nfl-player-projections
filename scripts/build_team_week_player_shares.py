#!/usr/bin/env python3
"""Build the team_week_player_shares table -- Plan A's shared prerequisite.

See docs/TEAM_LEVEL_ALLOCATION_MODELS.md for the full plan. This script
produces, for each QB/RB/WR/TE player-week in the audited
`canonical_player_weeks` panel, that player's SAME-WEEK share of team volume
(targets, rushing attempts, receiving yards, rushing yards) plus LAGGED
share-history features (season-to-date and rolling-3 mean of the player's own
share in PRIOR weeks).

Same-week share columns are LABELS, not features -- exactly like
`home_win` in src/models/game_outcome/labels.py, they must never be fed into
a model that predicts them. Only the `_s2d`/`_roll3` lagged columns are
features. See src/models/game_outcome/features.py's `_lagged_form` for the
prior art this mirrors (shift(1) before .expanding()/.rolling(), prior-season
fallback for cold starts) -- kept as an independent copy here (player-keyed,
not team-keyed) rather than a shared import, same rationale used elsewhere in
this repo for the game-outcome module's own copies.

Team-total denominator note: team_X is the SUM of player_X across every
player_id in this table's own (team, season, week) group, using 0 for any
player with no player_weekly_stats row that week. This is NOT the same kind
of decision as leaving `fantasy_points` NULL in build_canonical_player_weeks.py
(where an unknown outcome must not be silently zero-filled) -- here, team_X
and player_X are computed from literally the same source rows, so "no stats
row this week" and "recorded 0 targets this week" are indistinguishable and
both correctly contribute 0 to the team total. There is no unknown/zero
ambiguity to preserve, unlike fantasy_points.

Usage:
    python scripts/build_team_week_player_shares.py --dry-run
    python scripts/build_team_week_player_shares.py --write
    python scripts/build_team_week_player_shares.py --write --seasons 2013 2026
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DB_PATH, POSITIONS
from src.models.team_allocation.features import ALL_VOLUME_COLS, OPPORTUNITY_COLS, ROLL_WINDOW, TABLE_NAME

MIN_PRIOR_GAMES_FOR_FORM = 3

# VOLUME_COLS/ROLL_WINDOW/TABLE_NAME are imported from
# src/models/team_allocation/features.py (single source of truth -- that
# module's feature_columns() must exclude exactly these same names, so it
# owns the definitions rather than each module keeping its own copy). Each
# share is player_X / team_X where team_X is this table's own group-sum of
# the same column -- see module docstring for why 0-fill on a missing stats
# row is correct here specifically (unlike fantasy_points elsewhere in this
# repo).


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_population(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    """QB/RB/WR/TE player-weeks from the audited canonical panel -- this is
    the population every share is computed over, not a raw stats pull."""
    if not _table_exists(conn, "canonical_player_weeks"):
        raise ValueError(
            "canonical_player_weeks table not found -- run "
            "scripts/build_canonical_player_weeks.py --write first"
        )
    df = pd.read_sql(
        """
        SELECT player_id, season, week, team, position
        FROM canonical_player_weeks
        WHERE season BETWEEN ? AND ? AND position IN ({})
        """.format(",".join("?" * len(POSITIONS))),
        conn,
        params=[lo, hi, *POSITIONS],
    )
    return df


def load_volumes(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    cols = pd.read_sql("PRAGMA table_info(player_weekly_stats)", conn)["name"].tolist()
    keep = ["player_id", "season", "week"] + [
        c for c in [*ALL_VOLUME_COLS, *OPPORTUNITY_COLS] if c in cols
    ]
    df = pd.read_sql(
        f"SELECT {','.join(keep)} FROM player_weekly_stats WHERE season BETWEEN ? AND ?",
        conn, params=(lo, hi),
    )
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "week", *ALL_VOLUME_COLS])
    df = df.drop_duplicates(["player_id", "season", "week"], keep="last")
    for c in ALL_VOLUME_COLS:
        if c not in df.columns:
            df[c] = 0
    for c in OPPORTUNITY_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[["player_id", "season", "week", *ALL_VOLUME_COLS, *OPPORTUNITY_COLS]]


def _safe_share(numer: pd.Series, denom: pd.Series) -> pd.Series:
    return (numer / denom.replace(0, pd.NA)).fillna(0.0)


def _lagged_team_totals(panel: pd.DataFrame) -> pd.DataFrame:
    """Team-level lagged season-to-date / rolling-3 mean of each volume
    column's TEAM TOTAL (not any single player's share) -- the cheapest
    legitimate "predicted team total" input for Plan A's reconstruction
    step (renormalized share x predicted team total, see
    src/models/team_allocation/reconstruct.py). Same shift(1)-before-
    aggregating discipline as the player-level share lags above: the value
    attached to a given (team, season, week) uses only that team's own
    games strictly before it. One row per (team, season, week); callers
    merge this back onto the per-player panel by that key.

    No prior-season cold-start fallback here (unlike the player share lags)
    -- kept out to keep this first cut simple; a team's first few tracked
    weeks legitimately have no team-total history yet, and NaN is the
    correct signal for "can't reconstruct yet," not something to
    zero-fill or approximate.
    """
    team_cols = [f"team_{c}" for c in ALL_VOLUME_COLS]
    team_week = panel[["team", "season", "week", *team_cols]].drop_duplicates(["team", "season", "week"])
    team_week = team_week.sort_values(["team", "season", "week"]).reset_index(drop=True)
    grp = team_week.groupby(["team", "season"], group_keys=False)
    out = team_week[["team", "season", "week"]].copy()
    for col in team_cols:
        out[f"{col}_s2d"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
        out[f"{col}_roll{ROLL_WINDOW}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )
    return out


def build_shares(conn: sqlite3.Connection, lo: int, hi: int) -> pd.DataFrame:
    pop = load_population(conn, lo, hi)
    if pop.empty:
        return pop
    vol = load_volumes(conn, lo, hi)

    panel = pop.merge(vol, on=["player_id", "season", "week"], how="left")
    for c in ALL_VOLUME_COLS:
        panel[c] = panel[c].fillna(0.0)
    for c in OPPORTUNITY_COLS:
        panel[c] = panel[c].fillna(0.0)

    # "Share of team volume" is only a coherent concept for nonnegative
    # volume, but a player's own recorded receiving_yards/rushing_yards can
    # be genuinely negative for a single game (e.g. a screen pass stuffed
    # for a loss) -- found running this against the real database: 670
    # negative receiving_yards rows and 1,927 negative rushing_yards rows,
    # producing 671 team-weeks where a share fell outside [0, 1] (a
    # teammate's share pulled above 1 by a negative contributor shrinking
    # the team total, or that player's own share going negative). Floor at
    # 0 for this share/team-total computation only -- the raw, unfloored
    # `panel[c]` column (kept for audit/reporting, never used as a model
    # feature -- see feature_columns()'s exclusion of VOLUME_COLS) and
    # fantasy-point scoring elsewhere in the repo are untouched. A
    # net-negative week genuinely captured none of the team's positive
    # offensive output, so 0 is the correct share, not a fabricated value.
    for c in ALL_VOLUME_COLS:
        panel[f"_share_basis_{c}"] = panel[c].clip(lower=0.0)
    n_floored = {c: int((panel[c] < 0).sum()) for c in ALL_VOLUME_COLS}
    if any(n_floored.values()):
        print(f"floored negative volume for share computation only: {n_floored}")

    team_totals = (
        panel.groupby(["team", "season", "week"])[[f"_share_basis_{c}" for c in ALL_VOLUME_COLS]]
        .transform("sum")
        .rename(columns={f"_share_basis_{c}": f"team_{c}" for c in ALL_VOLUME_COLS})
    )
    panel = pd.concat([panel, team_totals], axis=1)

    share_cols = []
    for c in ALL_VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        panel[share_col] = _safe_share(panel[f"_share_basis_{c}"], panel[f"team_{c}"])
        share_cols.append(share_col)
    panel = panel.drop(columns=[f"_share_basis_{c}" for c in ALL_VOLUME_COLS])

    panel = panel.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grp = panel.groupby(["player_id", "season"], group_keys=False)
    panel["n_prior_games"] = grp.cumcount()

    for col in share_cols:
        panel[f"{col}_s2d"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
        panel[f"{col}_roll{ROLL_WINDOW}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )

    prior_season = panel.groupby(["player_id", "season"])[share_cols].mean().reset_index()
    prior_season["season"] = prior_season["season"] + 1
    prior_season = prior_season.rename(columns={c: f"{c}_prior_season" for c in share_cols})
    panel = panel.merge(prior_season, on=["player_id", "season"], how="left")

    panel["is_cold_start"] = (panel["n_prior_games"] < MIN_PRIOR_GAMES_FOR_FORM).astype(int)
    cold = panel["is_cold_start"].astype(bool)
    for col in share_cols:
        prior_col = f"{col}_prior_season"
        panel.loc[cold, f"{col}_s2d"] = panel.loc[cold, f"{col}_s2d"].where(
            panel.loc[cold, prior_col].isna(), panel.loc[cold, prior_col]
        )
        panel.loc[cold, f"{col}_roll{ROLL_WINDOW}"] = panel.loc[cold, f"{col}_roll{ROLL_WINDOW}"].where(
            panel.loc[cold, prior_col].isna(), panel.loc[cold, prior_col]
        )

    drop_cols = [f"{c}_prior_season" for c in share_cols] + ["n_prior_games"]
    panel = panel.drop(columns=drop_cols)

    # Lag role/opportunity history.  Same-week opportunity values are retained
    # for auditability but explicitly excluded by feature_columns(); only the
    # shifted player history and shifted team-opportunity history are usable.
    opp_group = panel.groupby(["player_id", "season"], group_keys=False)
    team_keys = ["team", "season", "week"]
    team_group = panel.groupby(team_keys)
    team_opp = team_group[OPPORTUNITY_COLS].transform("sum")
    team_opp = team_opp.rename(columns={c: f"team_{c}" for c in OPPORTUNITY_COLS})
    panel = pd.concat([panel, team_opp], axis=1)

    # Player history and same-week opportunity shares are vectorized before
    # constructing team history; avoid repeated many-column merges here (the
    # table spans millions of player-weeks in a full rebuild).
    for col in OPPORTUNITY_COLS:
        panel[f"{col}_s2d"] = opp_group[col].transform(lambda s: s.shift(1).expanding().mean())
        panel[f"{col}_roll{ROLL_WINDOW}"] = opp_group[col].transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )
        panel[f"share_of_team_{col}"] = _safe_share(panel[col], panel[f"team_{col}"])
        share_group = panel.groupby(["player_id", "season"], group_keys=False)[f"share_of_team_{col}"]
        panel[f"share_of_team_{col}_s2d"] = share_group.transform(lambda s: s.shift(1).expanding().mean())
        panel[f"share_of_team_{col}_roll{ROLL_WINDOW}"] = share_group.transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )

    team_week = panel[team_keys + [f"team_{c}" for c in OPPORTUNITY_COLS]].drop_duplicates(team_keys)
    team_week = team_week.sort_values(team_keys).reset_index(drop=True)
    team_hist = team_week[team_keys].copy()
    for col in OPPORTUNITY_COLS:
        team_col = f"team_{col}"
        tg = team_week.groupby(["team", "season"], group_keys=False)[team_col]
        team_hist[f"{team_col}_s2d"] = tg.transform(lambda s: s.shift(1).expanding().mean())
        team_hist[f"{team_col}_roll{ROLL_WINDOW}"] = tg.transform(
            lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
        )
    panel = panel.merge(team_hist, on=team_keys, how="left", validate="many_to_one")

    team_totals_lagged = _lagged_team_totals(panel)
    panel = panel.merge(team_totals_lagged, on=["team", "season", "week"], how="left")

    return panel.reset_index(drop=True)


def validate_shares(panel: pd.DataFrame) -> None:
    if panel.empty:
        raise ValueError("team_week_player_shares panel is empty")
    if panel.duplicated(["player_id", "season", "week"]).any():
        raise ValueError("duplicate player_id/season/week rows")
    for c in ALL_VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        if (panel[share_col] < -1e-9).any() or (panel[share_col] > 1 + 1e-9).any():
            raise ValueError(f"{share_col} outside [0, 1]")
    # Same-week shares within a team must sum to <= 1 (+ float tolerance) --
    # a share > the whole team's total is a join or double-count bug.
    for c in ALL_VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        totals = panel.groupby(["team", "season", "week"])[share_col].sum()
        bad = totals[totals > 1 + 1e-6]
        if not bad.empty:
            raise ValueError(f"{share_col} sums to > 1 for {len(bad)} team-week(s)")


def audit_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Season/position coverage summary -- printed before any downstream
    metric is trusted, so a silent population discontinuity (a season
    half-missing, a position's cold-start rate spiking) is visible rather
    than baked silently into a walk-forward MAE. Mirrors
    build_canonical_player_weeks.py's audit_panel() summary, same rationale
    (Phase 2's acceptance criterion 3: "no unexplained season/position
    discontinuity").
    """
    summary = (
        panel.groupby(["season", "position"], dropna=False)
        .agg(
            player_weeks=("player_id", "size"),
            players=("player_id", "nunique"),
            cold_start=("is_cold_start", "sum"),
        )
        .reset_index()
    )
    summary["cold_start_rate"] = (summary["cold_start"] / summary["player_weeks"]).round(3)
    return summary


def audit_team_week_sums(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-volume-column team-week share-sum diagnostics.

    `validate_shares` already hard-fails any team-week summing to > 1, but a
    sum well UNDER 1 is not an error -- it can be legitimate (a bye-week
    artifact should not appear at all since canonical_player_weeks only has
    rows for games actually played, but a real data issue -- e.g. a trade
    landing a player on two teams' rosters the same week, or a team-code
    normalization gap) would show up as a suspiciously low sum rather than a
    crash. This is a report to read, not a gate to pass.
    """
    rows = []
    for c in ALL_VOLUME_COLS:
        share_col = f"share_of_team_{c}"
        totals = panel.groupby(["team", "season", "week"])[share_col].sum()
        rows.append({
            "share_col": share_col,
            "n_team_weeks": int(len(totals)),
            "min_sum": float(totals.min()),
            "mean_sum": float(totals.mean()),
            "max_sum": float(totals.max()),
            "n_team_weeks_sum_lt_0.5": int((totals < 0.5).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("LO", "HI"), default=[2013, 2026])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--audit-csv", type=Path,
        default=Path("data") / "experiments" / "team_week_player_shares_audit.csv",
    )
    args = ap.parse_args()
    lo, hi = sorted(args.seasons)

    conn = sqlite3.connect(str(args.db))
    panel = build_shares(conn, lo, hi)
    validate_shares(panel)

    print(f"team_week_player_shares: {len(panel):,} player-weeks, seasons {lo}-{hi}")
    print(f"cold-start rows: {int(panel['is_cold_start'].sum()):,}")

    coverage = audit_coverage(panel)
    print("\nseason/position coverage:")
    print(coverage.to_string(index=False))
    if args.audit_csv:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.audit_csv, index=False)
        print(f"wrote coverage audit CSV: {args.audit_csv}")

    sums = audit_team_week_sums(panel)
    print("\nteam-week share-sum diagnostics (validate_shares already hard-fails > 1;\n"
          "a low min_sum or nonzero n_team_weeks_sum_lt_0.5 here is worth reading,\n"
          "not necessarily a bug):")
    print(sums.to_string(index=False))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.csv, index=False)
        print(f"wrote CSV: {args.csv}")

    if args.write and not args.dry_run:
        panel.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE_NAME}_key "
            f"ON {TABLE_NAME}(player_id, season, week)"
        )
        conn.commit()
        print(f"wrote SQLite table: {TABLE_NAME}")
    else:
        print("dry-run: database unchanged")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

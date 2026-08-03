#!/usr/bin/env python3
"""
Snake-draft simulator: ModelBot (uses our walk-forward projections
aggregated to season totals) vs ADPBot (uses FantasyPros ECR as
draft rank).

Implements Step 2 of the 2026-04-23 council's Critical Next Steps
(``council-transcript-20260423-051434.md``).

Success signal: "simulator runs a 2024 draft end-to-end, outputs
rosters + projected season points for your-bot vs ADP-bot."

Design:
- 12 teams, 15 rounds, snake order (round 1: 1→12, round 2: 12→1, …).
- One ModelBot at ``--model-slot`` (default 6); the other 11 teams
  are ADPBots that pick the top-ranked available player.
- Roster slots for scoring: QB 1, RB 2, WR 2, TE 1, FLEX(RB/WR/TE) 1
  → 7 starters + 8 bench = 15 roster spots.  Soft per-position caps
  are enforced during drafting so bots don't over-stack.
- After the draft, each team is scored by summing season-total
  *actual* fantasy points over its 7 best starters (QB, best 2 RBs,
  best 2 WRs, best TE, best remaining RB/WR/TE for FLEX).  Preseason
  projection and retrospective actual are both reported.

Name matching between the ADP board and the walk-forward predictions
CSV uses the same last-name + position fallback as
``scripts/start_sit_prototype.py``.  Unmatched ADP players are
skipped (and counted) — intentional, silent auto-matching hides
data issues in a prototype.

Usage:
    python scripts/snake_draft_sim.py --season 2024
    python scripts/snake_draft_sim.py --season 2024 --model-slot 1
    python scripts/snake_draft_sim.py \\
        --season 2024 \\
        --predictions-csv data/backtest_results/ts_backtest_2024_20260423_055841_predictions.csv \\
        --json /tmp/draft_2024.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
POSITION_CAPS: Dict[str, int] = {"QB": 3, "RB": 8, "WR": 8, "TE": 3}
TEAMS = 10
ROUNDS = 15

# --------------------------------------------------------------------
# Name utilities (match scripts/start_sit_prototype.py)
# --------------------------------------------------------------------

def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _last_token(name: str) -> str:
    if "." in name and " " not in name:
        return name.split(".")[-1]
    parts = name.split()
    while parts and parts[-1].lower().rstrip(".") in _NAME_SUFFIXES:
        parts.pop()
    return parts[-1] if parts else name


def _first_initial(name: str) -> str:
    """Return the first character of the name, normalized.  Handles
    both 'C.Lamb' (first letter = 'c') and 'Christian McCaffrey'
    (first letter = 'c')."""
    n = (name or "").strip()
    return n[0].lower() if n else ""


def _normalize_full_name(name: str) -> str:
    parts = []
    for token in (name or "").replace(".", " ").split():
        norm = _normalize(token)
        if norm and norm not in _NAME_SUFFIXES:
            parts.append(norm)
    return " ".join(parts)


_TEAM_ALIASES = {
    "JAC": "JAX",
    "LA": "LAR",
    "LAR": "LAR",
}


def _canonical_team(team: str) -> str:
    return _TEAM_ALIASES.get((team or "").strip().upper(), (team or "").strip().upper())


@lru_cache(maxsize=1)
def _load_full_name_player_id_lookup() -> Dict[Tuple[str, str, str], str]:
    """Return {(full_name, position, team): player_id} from roster sources.

    The draft ADP feed uses full names while historical projections often use
    abbreviated names. This lookup lets us keep exact player_id matches when
    two players otherwise share the same initial/last-name signature.
    """
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    lookup: Dict[Tuple[str, str, str], str] = {}

    queries = [
        (
            """
            SELECT player_name, position, team, player_id
              FROM weekly_rosters_v2
             WHERE player_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
             ORDER BY season DESC, week DESC
            """
        ),
        (
            """
            SELECT full_name AS player_name, position, team, player_id
              FROM weekly_rosters
             WHERE full_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
             ORDER BY season DESC, week DESC
            """
        ),
        (
            """
            SELECT player_name, position, team, player_id
              FROM rosters
             WHERE player_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
             ORDER BY season DESC
            """
        ),
    ]

    for query in queries:
        try:
            for name, position, team, player_id in conn.execute(query):
                key = (
                    (name or "").strip().lower(),
                    (position or "").strip().upper(),
                    _canonical_team(team),
                )
                if key not in lookup and player_id:
                    lookup[key] = str(player_id)
        except sqlite3.OperationalError:
            continue

    conn.close()
    return lookup


def _resolve_player_id_from_full_name(name: str, position: str, team: str) -> str:
    lookup = _load_full_name_player_id_lookup()
    return lookup.get(
        (
            (name or "").strip().lower(),
            (position or "").strip().upper(),
            _canonical_team(team),
        ),
        "",
    )


@lru_cache(maxsize=1)
def _load_player_id_name_aliases() -> Dict[str, set[str]]:
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    aliases: Dict[str, set[str]] = {}
    queries = [
        (
            """
            SELECT player_name, player_id
              FROM weekly_rosters_v2
             WHERE player_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
            """
        ),
        (
            """
            SELECT full_name AS player_name, player_id
              FROM weekly_rosters
             WHERE full_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
            """
        ),
        (
            """
            SELECT player_name, player_id
              FROM rosters
             WHERE player_name IS NOT NULL
               AND position IN ('QB', 'RB', 'WR', 'TE')
            """
        ),
    ]
    for query in queries:
        try:
            for name, player_id in conn.execute(query):
                pid = str(player_id or "")
                norm_name = _normalize_full_name(name)
                if pid and norm_name:
                    aliases.setdefault(pid, set()).add(norm_name)
        except sqlite3.OperationalError:
            continue

    conn.close()
    return aliases


def _prediction_matches_adp_name(pred: Dict, adp_name: str) -> bool:
    if not pred:
        return False
    if " " not in (adp_name or "").strip():
        return True
    player_id = str(pred.get("player_id") or "")
    if not player_id:
        return True
    aliases = _load_player_id_name_aliases().get(player_id)
    if not aliases:
        return True
    return _normalize_full_name(adp_name) in aliases


def _build_pred_index(predictions: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict]:
    """Build (first_initial, normalized_last, position) → record index
    for fast match.  First-initial + last-name disambiguates common
    surnames (e.g. Bijan Robinson vs Keilan Robinson, both RB)."""
    idx: Dict[Tuple[str, str, str], Dict] = {}
    for _, r in predictions.iterrows():
        key = (
            _first_initial(r["name"]),
            _normalize(_last_token(r["name"])),
            r["position"],
        )
        # Keep the record with the most weeks (i.e. the more
        # established player) when two players share (initial, last,
        # position).  Breaks ties by higher pred_total.
        existing = idx.get(key)
        if existing is None:
            idx[key] = r.to_dict()
        else:
            new_weeks = int(r.get("weeks", 0) or 0)
            old_weeks = int(existing.get("weeks", 0) or 0)
            if new_weeks > old_weeks or (
                new_weeks == old_weeks
                and float(r.get("pred_total", 0) or 0) > float(existing.get("pred_total", 0) or 0)
            ):
                idx[key] = r.to_dict()
    return idx


def _build_pred_team_fallback_index(
    predictions: pd.DataFrame,
) -> Dict[Tuple[str, str, str], Dict]:
    """Build a secondary index that ignores source position but keeps team.

    This rescues players whose predictions artifact has a bad position label
    while still requiring the same team and normalized identity.
    """
    idx: Dict[Tuple[str, str, str], Dict] = {}
    for _, r in predictions.iterrows():
        key = (
            _first_initial(r["name"]),
            _normalize(_last_token(r["name"])),
            str(r.get("team") or ""),
        )
        existing = idx.get(key)
        if existing is None:
            idx[key] = r.to_dict()
        else:
            new_weeks = int(r.get("weeks", 0) or 0)
            old_weeks = int(existing.get("weeks", 0) or 0)
            if new_weeks > old_weeks or (
                new_weeks == old_weeks
                and float(r.get("pred_total", 0) or 0) > float(existing.get("pred_total", 0) or 0)
            ):
                idx[key] = r.to_dict()
    return idx


def _build_pred_name_fallback_index(
    predictions: pd.DataFrame,
) -> Dict[Tuple[str, str], Dict]:
    """Build a last-resort index on normalized identity only.

    We only keep keys that resolve to a single underlying player record;
    ambiguous identities are discarded instead of guessing.
    """
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for _, r in predictions.iterrows():
        key = (
            _first_initial(r["name"]),
            _normalize(_last_token(r["name"])),
        )
        grouped.setdefault(key, []).append(r.to_dict())

    idx: Dict[Tuple[str, str], Dict] = {}
    for key, rows in grouped.items():
        teams = {str(r.get("team") or "") for r in rows}
        positions = {str(r.get("position") or "") for r in rows}
        if len(rows) == 1 or (len(teams) == 1 and len(positions) == 1):
            idx[key] = max(
                rows,
                key=lambda r: (
                    int(r.get("weeks", 0) or 0),
                    float(r.get("pred_total", 0) or 0.0),
                ),
            )
    return idx


# --------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------

def load_adp_board(season: int, before_date: str = None) -> pd.DataFrame:
    """Return the latest pre-draft ADP snapshot for ``season`` from
    the ``adp_history`` table (``page_type='redraft-overall'``)."""
    from src.utils.database import DatabaseManager
    db = DatabaseManager()
    with db._get_connection() as conn:
        # Pick the latest scrape_date ≤ before_date (default Sep 10
        # of the season, i.e. just after Week 1 kickoff).
        cutoff = before_date or f"{season}-09-10"
        row = conn.execute(
            """
            SELECT MAX(scrape_date) FROM adp_history
             WHERE season = ? AND page_type = 'redraft-overall'
               AND scrape_date <= ?
            """,
            (season, cutoff),
        ).fetchone()
        scrape_date = row[0]
        if scrape_date is None:
            raise RuntimeError(
                f"No redraft-overall ADP for season={season} before {cutoff}. "
                "Run scripts/backfill_adp.py first."
            )
        board = pd.read_sql(
            """
            SELECT player_name AS name, position, team, ecr, sd, best, worst
              FROM adp_history
             WHERE season = ? AND scrape_date = ?
               AND page_type = 'redraft-overall'
               AND position IN ('QB', 'RB', 'WR', 'TE')
             ORDER BY ecr ASC
            """,
            conn,
            params=(season, scrape_date),
        )
    return board.assign(scrape_date=scrape_date)


def load_model_projections(
    predictions_csv: Path, ranking: str = "season_sum", season: int = None
) -> pd.DataFrame:
    """Aggregate a walk-forward predictions CSV to season totals per
    player.  Only ``is_active=1`` rows contribute, matching how the
    paper trade locks are scored.

    ``ranking`` controls what we expose as ``model_rank_value`` (the
    score ModelBot ranks with):
      - ``season_sum`` (default): sum of weekly predictions across the
        season.  Causal per-week but embeds within-season learning.
      - ``week1``: the model's week-1 prediction alone.  Week 1's
        features only use pre-season info, so this is a genuine
        pre-draft forecast.  We rank by a per-week *rate* rather
        than a total — positions are ranked against their own pool so
        QB vs RB scale differences don't matter.
      - ``prior_season``: ignore this model entirely; rank by the
        player's prior-season total actual fantasy points from
        ``player_weekly_stats``.  Naive "last year's points" baseline.

    ``pred_total`` and ``actual_total`` (used for post-draft scoring)
    are always the season-sum versions regardless of ranking mode."""
    df = pd.read_csv(predictions_csv)
    if "week" in df.columns:
        weeks = pd.to_numeric(df["week"], errors="coerce")
        df = df[weeks.between(1, 18)]
    if "is_active" in df.columns:
        df = df[df["is_active"] == 1]
    agg = (
        df.groupby(["player_id", "name", "position", "team"])
          .agg(pred_total=("predicted", "sum"),
               actual_total=("actual", "sum"),
               weeks=("week", "count"))
          .reset_index()
    )
    if ranking == "season_sum":
        agg["model_rank_value"] = agg["pred_total"]
    elif ranking == "vorp":
        agg["model_rank_value"] = _apply_vorp(agg, basis_col="pred_total")
    elif ranking == "week1":
        w1 = (
            df[df["week"] == 1]
              .groupby("player_id")["predicted"].first()
              .rename("model_rank_value")
              .reset_index()
        )
        agg = agg.merge(w1, on="player_id", how="left")
        agg["model_rank_value"] = agg["model_rank_value"].fillna(0.0)
    elif ranking == "prior_season":
        if season is None:
            raise ValueError("season required for ranking='prior_season'")
        from src.utils.database import DatabaseManager
        prior = season - 1
        with DatabaseManager()._get_connection() as conn:
            prior_df = pd.read_sql(
                """
                SELECT pws.player_id,
                       SUM(pws.fantasy_points) AS prior_total
                  FROM player_weekly_stats pws
                 WHERE pws.season = ?
                 GROUP BY pws.player_id
                """,
                conn,
                params=(prior,),
            )
        agg = agg.merge(prior_df, on="player_id", how="left")
        agg["model_rank_value"] = agg["prior_total"].fillna(0.0)
        agg.drop(columns=["prior_total"], inplace=True)
    else:
        raise ValueError(f"unknown ranking={ranking!r}")
    return agg


def _load_season_actual_totals(season: int) -> pd.DataFrame:
    """Load regular-season fantasy-point totals for a completed season."""
    from src.utils.database import DatabaseManager

    with DatabaseManager()._get_connection() as conn:
        return pd.read_sql(
            """
            SELECT pws.player_id,
                   p.name,
                   p.position,
                   SUM(pws.fantasy_points) AS actual_total,
                   COUNT(*) AS weeks
              FROM player_weekly_stats pws
              JOIN players p ON pws.player_id = p.player_id
             WHERE pws.season = ?
               AND pws.week <= 18
               AND p.position IN ('QB', 'RB', 'WR', 'TE')
             GROUP BY pws.player_id, p.name, p.position
            """,
            conn,
            params=(season,),
        )


def _attach_actual_totals(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach target-season actual totals using player_id, then name fallback."""
    if projections.empty:
        return projections.copy()

    out = projections.copy()
    if actuals.empty:
        out["actual_total"] = 0.0
        return out

    actual_rows = actuals.to_dict("records")
    by_player_id = {
        str(r.get("player_id") or ""): float(r.get("actual_total") or 0.0)
        for r in actual_rows
        if str(r.get("player_id") or "")
    }
    by_full_name = {}
    by_signature = {}
    for row in actual_rows:
        full_key = (
            _normalize_full_name(str(row.get("name") or "")),
            str(row.get("position") or ""),
        )
        sig_key = (
            _first_initial(str(row.get("name") or "")),
            _normalize(_last_token(str(row.get("name") or ""))),
            str(row.get("position") or ""),
        )
        existing_full = by_full_name.get(full_key)
        existing_sig = by_signature.get(sig_key)
        weeks = int(row.get("weeks", 0) or 0)
        total = float(row.get("actual_total", 0.0) or 0.0)
        if existing_full is None or (
            weeks,
            total,
        ) > (
            int(existing_full.get("weeks", 0) or 0),
            float(existing_full.get("actual_total", 0.0) or 0.0),
        ):
            by_full_name[full_key] = row
        if existing_sig is None or (
            weeks,
            total,
        ) > (
            int(existing_sig.get("weeks", 0) or 0),
            float(existing_sig.get("actual_total", 0.0) or 0.0),
        ):
            by_signature[sig_key] = row

    actual_totals: List[float] = []
    for _, row in out.iterrows():
        player_id = str(row.get("player_id") or "")
        actual_total = by_player_id.get(player_id)
        if actual_total is None:
            full_key = (
                _normalize_full_name(str(row.get("name") or "")),
                str(row.get("position") or ""),
            )
            match = by_full_name.get(full_key)
            if match is None:
                sig_key = (
                    _first_initial(str(row.get("name") or "")),
                    _normalize(_last_token(str(row.get("name") or ""))),
                    str(row.get("position") or ""),
                )
                match = by_signature.get(sig_key)
            actual_total = float(match.get("actual_total", 0.0) or 0.0) if match else 0.0
        actual_totals.append(float(actual_total))

    out["actual_total"] = actual_totals
    return out


def load_preseason_projections(
    season: int,
    adp_df: pd.DataFrame = None,
    projection_mode: str = "auto",
    actuals_season: int | None = None,
    projector_path: Path | None = None,
) -> pd.DataFrame:
    """Build preseason projections from prior-season stats.

    ``projection_mode``:
      - ``auto``: try the trained preseason projector, then fall back to PPG x 17
      - ``ml``: require the trained preseason projector
      - ``ppg17``: use prior-season fantasy points per game x 17

    If ``actuals_season`` is provided, overwrite ``actual_total`` with the
    realized totals from that season so the frame can drive a historical
    draft simulation instead of only a future preseason board.
    """
    from src.utils.database import DatabaseManager
    prior = season - 1
    with DatabaseManager()._get_connection() as conn:
        # Full feature query — matches PreseasonProjector._build_season_pairs() so
        # that predict() receives all the features it was trained on (not just ppg).
        # Missing features in a minimal query get filled with 0, which after
        # StandardScaler centering produces severely under-estimated projections.
        prior_df = pd.read_sql(
            """
            SELECT pws.player_id,
                   p.name,
                   p.position,
                   pws.team,
                   p.birth_date,
                   COUNT(*) AS games_played,
                   AVG(pws.fantasy_points) AS ppg,
                   SUM(pws.fantasy_points) AS pred_total,
                   SUM(pws.fantasy_points) AS actual_total,
                   COUNT(*) AS weeks,
                   AVG(COALESCE(pws.passing_yards, 0))  AS passing_yards_pg,
                   AVG(COALESCE(pws.passing_tds, 0))    AS passing_tds_pg,
                   AVG(COALESCE(pws.interceptions, 0))  AS interceptions_pg,
                   AVG(COALESCE(pws.rushing_yards, 0))  AS rushing_yards_pg,
                   AVG(COALESCE(pws.rushing_attempts, 0)) AS carries_pg,
                   AVG(COALESCE(pws.targets, 0))        AS targets_pg,
                   AVG(COALESCE(pws.receptions, 0))     AS receptions_pg,
                   AVG(COALESCE(pws.receiving_yards, 0)) AS receiving_yards_pg,
                   AVG(COALESCE(pws.air_yards, 0))      AS air_yards_pg,
                   AVG(CASE WHEN COALESCE(pws.passing_attempts, 0) > 0
                       THEN 100.0 * pws.passing_completions / pws.passing_attempts
                       ELSE 0 END) AS completion_pct,
                   AVG(COALESCE(us.snap_share, 0))   AS snap_share,
                   AVG(COALESCE(us.target_share, 0)) AS target_share,
                   AVG(COALESCE(us.rush_share, 0))   AS rush_share
              FROM player_weekly_stats pws
              JOIN players p ON pws.player_id = p.player_id
              LEFT JOIN utilization_scores us
                ON pws.player_id = us.player_id
               AND pws.season = us.season
               AND pws.week = us.week
             WHERE pws.season = ?
               AND p.position IN ('QB', 'RB', 'WR', 'TE')
             GROUP BY pws.player_id
            """,
            conn,
            params=(prior,),
        )
        prior_df["prior_season"] = prior
        prior_df["projection_season"] = season
    if prior_df.empty and adp_df is None:
        return prior_df

    if not prior_df.empty and actuals_season is not None:
        prior_df["actual_total"] = 0.0

    if not prior_df.empty:
        if projection_mode not in {"auto", "ml", "ppg17"}:
            raise ValueError(f"unknown projection_mode={projection_mode!r}")

        if projection_mode == "ppg17":
            prior_df["model_rank_value"] = prior_df["ppg"] * 17
            prior_df["pred_total"] = prior_df["model_rank_value"]
            prior_df = prior_df.drop(columns=["ppg"])
        else:
            # Try ML season-total projector first; optionally fall back to PPG × 17.
            try:
                from pathlib import Path as _Path
                _projector_path = (
                    _Path(projector_path)
                    if projector_path is not None
                    else _Path(__file__).resolve().parent.parent / "data" / "models" / "preseason_projector.json"
                )
                if _projector_path.exists():
                    from src.models.preseason_projector import PreseasonProjector
                    _proj = PreseasonProjector.load(_projector_path)
                    _ml_preds = []
                    for pos in ("QB", "RB", "WR", "TE"):
                        pos_mask = prior_df["position"] == pos
                        if pos_mask.any() and pos in _proj.models:
                            pos_df = prior_df[pos_mask].copy()
                            preds = _proj.predict(pos_df, pos)
                            _ml_preds.extend(zip(pos_df.index, preds))
                    if _ml_preds:
                        ml_series = {idx: val for idx, val in _ml_preds}
                        prior_df["model_rank_value"] = prior_df.index.map(
                            lambda i: ml_series.get(i, prior_df.loc[i, "ppg"] * 17)
                        )
                        prior_df["pred_total"] = prior_df["model_rank_value"]
                        prior_df = prior_df.drop(columns=["ppg"])
                    else:
                        raise ValueError("No ML predictions produced")
                else:
                    raise FileNotFoundError("preseason_projector.json not found")
            except Exception:
                if projection_mode == "ml":
                    raise
                prior_df["model_rank_value"] = prior_df["ppg"] * 17
                prior_df["pred_total"] = prior_df["model_rank_value"]
                prior_df = prior_df.drop(columns=["ppg"])

    # Add rookies/unmatched ADP players
    if adp_df is not None and not adp_df.empty:
        # Try to use draft capital data for rookies
        draft_capital = _load_draft_capital(season)

        def _ecr_to_ppg(ecr: float, pos: str) -> float:
            """Fallback: ECR rank to estimated PPG for undrafted players."""
            baselines = {"QB": 20.0, "RB": 14.0, "WR": 12.0, "TE": 10.0}
            base = baselines.get(pos, 10.0)
            return max(base * (50.0 / (ecr + 30.0)), 1.0)

        matched_names = set()
        if not prior_df.empty:
            matched_names = {
                (_first_initial(n), _normalize(_last_token(n)), p)
                for n, p in zip(prior_df["name"], prior_df["position"])
            }

        rookie_rows = []
        for _, r in adp_df.iterrows():
            key = (_first_initial(r["name"]), _normalize(_last_token(r["name"])), r["position"])
            if key not in matched_names:
                # Check if we have draft capital for this player
                dc = draft_capital.get(key)
                if dc:
                    proj = dc["proj"]
                else:
                    proj = _ecr_to_ppg(r["ecr"], r["position"]) * 17

                rookie_rows.append({
                    "player_id": (
                        _resolve_player_id_from_full_name(
                            r["name"], r["position"], r.get("team", "")
                        )
                        or f"rookie_{r['name']}_{r['position']}"
                    ),
                    "name": r["name"],
                    "position": r["position"],
                    "team": r.get("team", ""),
                    "pred_total": proj,
                    "actual_total": 0.0,
                    "weeks": 0,
                    "model_rank_value": proj,
                })

        if rookie_rows:
            rookies_df = pd.DataFrame(rookie_rows)
            prior_df = pd.concat([prior_df, rookies_df], ignore_index=True)

    if actuals_season is not None and not prior_df.empty:
        prior_df = _attach_actual_totals(
            prior_df,
            _load_season_actual_totals(actuals_season),
        )

    return prior_df


def _load_draft_capital(season: int) -> dict:
    """Load draft picks for a season and return {(initial, last, pos): {round, proj}}."""
    draft_path = Path(__file__).resolve().parent.parent / "data" / "draft_picks.parquet"
    if not draft_path.exists():
        return {}

    dp = pd.read_parquet(draft_path)
    cls = dp[(dp["season"] == season) & (dp["position"].isin(["QB", "RB", "WR", "TE"]))]
    if cls.empty:
        return {}

    # Build historical rookie curve
    curve = _build_rookie_curve(dp)

    result = {}
    for _, r in cls.iterrows():
        key = (_first_initial(r["pfr_player_name"]), _normalize(_last_token(r["pfr_player_name"])), r["position"])
        rnd = int(r["round"])
        proj = curve.get((r["position"], rnd), 50.0)
        result[key] = {"round": rnd, "proj": proj}
    return result


def _build_rookie_curve(dp: pd.DataFrame, seasons=range(2020, 2025)) -> dict:
    """Build (position, round) → avg rookie season FP from historical data."""
    from src.utils.database import DatabaseManager
    skill = dp[
        (dp["position"].isin(["QB", "RB", "WR", "TE"]))
        & (dp["season"].isin(seasons))
        & (dp["gsis_id"].notna())
    ]
    with DatabaseManager()._get_connection() as conn:
        curves = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            for rnd in range(1, 8):
                picks = skill[(skill["position"] == pos) & (skill["round"] == rnd)]
                fps = []
                for _, r in picks.iterrows():
                    row = conn.execute(
                        "SELECT SUM(fantasy_points) FROM player_weekly_stats "
                        "WHERE player_id=? AND season=?",
                        (r["gsis_id"], int(r["season"])),
                    ).fetchone()
                    if row[0] and row[0] > 0:
                        fps.append(row[0])
                curves[(pos, rnd)] = round(sum(fps) / len(fps), 1) if fps else 50.0
    return curves


def _compute_replacement_values(
    agg: pd.DataFrame,
    basis_col: str = "pred_total",
    teams: int = TEAMS,
    roster_slots: Dict[str, int] = ROSTER_SLOTS,
) -> Dict[str, float]:
    """Compute starter-based replacement values for the current league shape.

    ``basis_col`` is already in PPR fantasy-point units, so replacement levels
    inherit the scoring system automatically. FLEX scarcity is handled by first
    reserving the base starters at each position, then filling all FLEX starter
    slots with the best remaining RB/WR/TE projections across positions.
    """
    if agg.empty or "position" not in agg.columns:
        return {}

    frame = agg[["position", basis_col]].copy()
    frame[basis_col] = pd.to_numeric(frame[basis_col], errors="coerce").fillna(0.0)

    selected_indices = set()
    sorted_by_pos: Dict[str, pd.DataFrame] = {}

    for pos, count_per_team in roster_slots.items():
        if pos == "FLEX":
            continue
        pos_df = frame[frame["position"] == pos].sort_values(
            basis_col, ascending=False
        )
        sorted_by_pos[pos] = pos_df
        starter_count = max(teams * count_per_team, 0)
        if starter_count > 0:
            selected_indices.update(pos_df.head(starter_count).index.tolist())

    flex_slots = max(teams * roster_slots.get("FLEX", 0), 0)
    if flex_slots > 0:
        flex_pool = frame[
            frame["position"].isin(FLEX_ELIGIBLE) & ~frame.index.isin(selected_indices)
        ].sort_values(basis_col, ascending=False)
        selected_indices.update(flex_pool.head(flex_slots).index.tolist())

    replacements: Dict[str, float] = {}
    for pos, pos_df in sorted_by_pos.items():
        remaining = pos_df.loc[~pos_df.index.isin(selected_indices), basis_col]
        replacements[pos] = float(remaining.iloc[0]) if not remaining.empty else 0.0
    return replacements


def _apply_vorp(agg: pd.DataFrame, basis_col: str = "pred_total") -> pd.Series:
    """Compute Value Over Replacement Player per row.

    Replacement is derived from the active league setup: base starters plus a
    FLEX starter pool across RB/WR/TE. Positions outside the roster model keep
    their raw value.
    """
    vorp = pd.to_numeric(agg[basis_col], errors="coerce").fillna(0.0).copy()
    replacements = _compute_replacement_values(agg, basis_col=basis_col)
    for pos, replacement in replacements.items():
        pos_mask = agg["position"] == pos
        vorp.loc[pos_mask] = vorp.loc[pos_mask] - replacement
    return vorp


# --------------------------------------------------------------------
# Draft board & bots
# --------------------------------------------------------------------

@dataclass
class DraftPlayer:
    player_id: str
    name: str
    position: str
    team: str
    ecr: float
    pred_total: float
    actual_total: float
    is_modelable: bool  # True if matched to projections
    model_rank_value: float = 0.0  # ModelBot's sort key (mode-dependent)


@dataclass
class Team:
    name: str
    is_model_bot: bool
    slot: int  # 1..TEAMS
    roster: List[DraftPlayer] = field(default_factory=list)

    def pos_count(self, pos: str) -> int:
        return sum(1 for p in self.roster if p.position == pos)

    def can_add(self, pos: str) -> bool:
        return self.pos_count(pos) < POSITION_CAPS.get(pos, 99)

    def rank_key_for(self, player: DraftPlayer) -> float:
        # Lower is better for ECR; negate the model score for the
        # ModelBot so we can sort ascending universally.
        if self.is_model_bot:
            # Prefer higher model_rank_value.  If unmodelable, fall
            # back to ECR (worst-case still sane).
            if player.is_modelable:
                return -player.model_rank_value
            return 1000.0 + player.ecr
        return player.ecr


def build_draft_board(
    adp: pd.DataFrame, projections: pd.DataFrame
) -> List[DraftPlayer]:
    """Merge ADP rows with model projections on (first-initial,
    normalized last name, position).  Returns the draft board sorted
    by ADP (lowest first)."""
    pred_idx = _build_pred_index(projections)
    pred_team_fallback_idx = _build_pred_team_fallback_index(projections)
    pred_name_fallback_idx = _build_pred_name_fallback_index(projections)
    pred_player_id_idx = {}
    if "player_id" in projections.columns:
        for _, r in projections.iterrows():
            pid = str(r.get("player_id") or "")
            if pid:
                pred_player_id_idx[pid] = r.to_dict()
    board: List[DraftPlayer] = []
    for _, r in adp.iterrows():
        pred = None
        resolved_player_id = _resolve_player_id_from_full_name(
            r["name"], r["position"], r.get("team") or ""
        )
        if resolved_player_id:
            pred = pred_player_id_idx.get(resolved_player_id)
        key = (
            _first_initial(r["name"]),
            _normalize(_last_token(r["name"])),
            r["position"],
        )
        if pred is None:
            pred = pred_idx.get(key)
            if pred is not None and not _prediction_matches_adp_name(pred, r["name"]):
                pred = None
        if pred is None:
            fallback_key = (
                _first_initial(r["name"]),
                _normalize(_last_token(r["name"])),
                str(r.get("team") or ""),
            )
            pred = pred_team_fallback_idx.get(fallback_key)
            if pred is not None and not _prediction_matches_adp_name(pred, r["name"]):
                pred = None
        if pred is None:
            pred = pred_name_fallback_idx.get((
                _first_initial(r["name"]),
                _normalize(_last_token(r["name"])),
            ))
            if pred is not None and not _prediction_matches_adp_name(pred, r["name"]):
                pred = None
        board.append(DraftPlayer(
            player_id=str(pred.get("player_id")) if pred else "",
            name=r["name"],
            position=r["position"],
            team=r.get("team") or "",
            ecr=float(r["ecr"]),
            pred_total=float(pred["pred_total"]) if pred else 0.0,
            actual_total=float(pred["actual_total"]) if pred else 0.0,
            is_modelable=pred is not None,
            model_rank_value=(
                float(pred["model_rank_value"])
                if pred and "model_rank_value" in pred
                else (float(pred["pred_total"]) if pred else 0.0)
            ),
        ))
    board.sort(key=lambda p: p.ecr)
    return board


def snake_pick_order(teams: int = TEAMS, rounds: int = ROUNDS) -> List[int]:
    """Return a flat list of ``teams * rounds`` 1-indexed team slots
    in snake order."""
    order: List[int] = []
    for r in range(rounds):
        if r % 2 == 0:
            order.extend(range(1, teams + 1))
        else:
            order.extend(range(teams, 0, -1))
    return order


def run_draft(
    board: List[DraftPlayer], model_slot: int = 6,
    model_pick_fn=None,
) -> List[Team]:
    """Execute the draft. Returns the 12 Team rosters in slot order.

    If ``model_pick_fn`` is provided, it is called for the model bot's
    picks instead of the default ECR/model-rank selector:
        model_pick_fn(team, available, pick_number) -> DraftPlayer
    """
    teams = [
        Team(name=f"Team{i}", is_model_bot=(i == model_slot), slot=i)
        for i in range(1, TEAMS + 1)
    ]
    available = list(board)
    order = snake_pick_order(TEAMS, ROUNDS)
    for pick_idx, slot in enumerate(order):
        t = teams[slot - 1]
        if t.is_model_bot and model_pick_fn is not None:
            pick = model_pick_fn(t, available, pick_idx + 1)
        else:
            pick = _select_pick(t, available)
        if pick is None:
            continue
        available.remove(pick)
        t.roster.append(pick)
    return teams


def _select_pick(t: Team, available: List[DraftPlayer]) -> Optional[DraftPlayer]:
    eligible = [p for p in available if t.can_add(p.position)]
    if not eligible:
        # Saturated every cap — let the team take the best remaining.
        eligible = available
    if not eligible:
        return None
    return min(eligible, key=t.rank_key_for)


# --------------------------------------------------------------------
# Post-draft scoring
# --------------------------------------------------------------------

def score_team(t: Team, use: str = "actual_total") -> Tuple[float, List[DraftPlayer]]:
    """Return (starters_total, starters_list) picking optimal lineup
    from the drafted roster.  ``use`` is either ``'actual_total'`` or
    ``'pred_total'``."""
    def val(p: DraftPlayer) -> float:
        return getattr(p, use)

    starters_ids: set = set()
    starters: List[DraftPlayer] = []

    def top_n(pos: str, n: int) -> List[DraftPlayer]:
        cands = sorted(
            (p for p in t.roster
             if p.position == pos and id(p) not in starters_ids),
            key=val, reverse=True,
        )
        return cands[:n]

    for pos, n in (("QB", 1), ("RB", 2), ("WR", 2), ("TE", 1)):
        for pick in top_n(pos, n):
            starters.append(pick)
            starters_ids.add(id(pick))

    flex_pool = sorted(
        (p for p in t.roster
         if p.position in FLEX_ELIGIBLE and id(p) not in starters_ids),
        key=val, reverse=True,
    )
    if flex_pool:
        starters.append(flex_pool[0])
        starters_ids.add(id(flex_pool[0]))

    total = sum(val(p) for p in starters)
    return total, starters


# --------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------

def summarize(teams: List[Team]) -> Dict:
    rows = []
    for t in teams:
        pred_total, _pred_starters = score_team(t, "pred_total")
        actual_total, actual_starters = score_team(t, "actual_total")
        rows.append({
            "slot": t.slot,
            "bot": "ModelBot" if t.is_model_bot else "ADPBot",
            "pred_starter_total": round(pred_total, 1),
            "actual_starter_total": round(actual_total, 1),
            "roster_names": [
                f"{p.name} ({p.position})" for p in t.roster
            ],
            "starter_names": [
                f"{p.name} ({p.position})" for p in actual_starters
            ],
        })
    rows_sorted = sorted(rows, key=lambda r: r["actual_starter_total"], reverse=True)
    model_row = next(r for r in rows if r["bot"] == "ModelBot")
    model_rank = next(
        i + 1 for i, r in enumerate(rows_sorted) if r["bot"] == "ModelBot"
    )
    return {
        "teams": rows,
        "ranked": rows_sorted,
        "model_rank_of_12": model_rank,
        "model_actual_total": model_row["actual_starter_total"],
        "adp_mean_actual_total": round(
            sum(r["actual_starter_total"] for r in rows if r["bot"] == "ADPBot")
            / max(1, sum(1 for r in rows if r["bot"] == "ADPBot")),
            1,
        ),
    }


CAVEATS = {
    "season_sum": (
        "CAVEAT: ModelBot ranks by the sum of per-week walk-forward "
        "predictions across the season. Each weekly prediction is "
        "causal (pre-week features only), but summing across the "
        "season embeds within-season learning a real draft-time "
        "forecast does not have. Structural check only."
    ),
    "week1": (
        "NOTE: ModelBot ranks by its week-1 prediction alone. The "
        "week-1 walk-forward model is trained on data up to end of "
        "the prior season, so this IS a genuine pre-draft forecast. "
        "n=1 draft is a directional signal, not a lift number."
    ),
    "prior_season": (
        "NOTE: ModelBot ranks by the player's prior-season total "
        "fantasy points (from player_weekly_stats). This is a naive "
        "baseline that bypasses the model entirely — useful to "
        "confirm whether a simple 'last year's points' strategy "
        "beats ADP without any ML."
    ),
    "vorp": (
        "NOTE: ModelBot ranks by VORP — the season_sum projection "
        "minus a flex-aware replacement baseline derived from the "
        f"{TEAMS}-team lineup ({ROSTER_SLOTS['QB']} QB, {ROSTER_SLOTS['RB']} RB, "
        f"{ROSTER_SLOTS['WR']} WR, {ROSTER_SLOTS['TE']} TE, "
        f"{ROSTER_SLOTS['FLEX']} FLEX in PPR). Cross-position: the best "
        "RB with VORP=12 outranks the best QB with VORP=6. Same hindsight "
        "caveat as season_sum applies (uses summed per-week walk-forward "
        "predictions)."
    ),
}


def _print_report(summary: Dict, season: int, scrape_date: str,
                  ranking: str = "season_sum") -> None:
    print(f"=== Snake draft {season} (ADP: {scrape_date}, "
          f"ranking={ranking}) ===")
    print(f"  {'slot':>4}  {'bot':>8}  {'pred':>7}  {'actual':>7}")
    print("  " + "-" * 36)
    for r in summary["ranked"]:
        marker = "*" if r["bot"] == "ModelBot" else " "
        print(
            f"  {r['slot']:>4}  {r['bot']:>8}  "
            f"{r['pred_starter_total']:>7.1f}  "
            f"{r['actual_starter_total']:>7.1f} {marker}"
        )
    print()
    print(
        f"ModelBot finished {summary['model_rank_of_12']} of {TEAMS} "
        f"by actual starter total: "
        f"{summary['model_actual_total']:.1f} vs "
        f"ADP mean {summary['adp_mean_actual_total']:.1f}."
    )
    print()
    print(CAVEATS.get(ranking, CAVEATS["season_sum"]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--model-slot", type=int, default=6)
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Walk-forward predictions CSV. Default: latest "
             "data/backtest_results/ts_backtest_{season}_*_predictions.csv.",
    )
    ap.add_argument(
        "--adp-before",
        type=str,
        default=None,
        help="Only use ADP scrapes on or before this YYYY-MM-DD. "
             "Default: {season}-09-10.",
    )
    ap.add_argument(
        "--ranking",
        choices=["season_sum", "week1", "prior_season", "vorp"],
        default="season_sum",
        help="How ModelBot ranks players. 'season_sum' (default) uses "
             "the hindsight walk-forward aggregate. 'week1' uses the "
             "walk-forward week-1 prediction (genuinely pre-season). "
             "'prior_season' ranks by prior-season actual FP (naive "
             "baseline, ignores the model). 'vorp' applies Value Over "
             "Replacement Player on the season_sum projection using "
             "the configured PPR starter lineup plus FLEX.",
    )
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not (1 <= args.model_slot <= TEAMS):
        print(f"--model-slot must be 1..{TEAMS}", file=sys.stderr)
        return 2

    csv_path = args.predictions_csv
    if csv_path is None:
        matches = sorted(
            (PROJECT_ROOT / "data" / "backtest_results").glob(
                f"ts_backtest_{args.season}_*_predictions.csv"
            )
        )
        if not matches:
            print(
                f"No predictions CSV for season {args.season} in "
                f"data/backtest_results/. Run the walk-forward backtest first.",
                file=sys.stderr,
            )
            return 1
        csv_path = matches[-1]

    print(f"ADP season:       {args.season}")
    print(f"Model preds file: {csv_path}")
    print(f"Model slot:       {args.model_slot}")
    print(f"Ranking mode:     {args.ranking}")

    adp = load_adp_board(args.season, args.adp_before)
    scrape_date = adp["scrape_date"].iloc[0]
    print(f"ADP scrape date:  {scrape_date}  ({len(adp)} rows)")

    projections = load_model_projections(
        csv_path, ranking=args.ranking, season=args.season,
    )
    board = build_draft_board(adp, projections)
    unmatched = sum(1 for p in board if not p.is_modelable)
    print(f"Draft board:      {len(board)} players; "
          f"{unmatched} unmatched to model (fallback to ECR).")
    print()

    teams = run_draft(board, model_slot=args.model_slot)
    summary = summarize(teams)
    _print_report(summary, args.season, scrape_date, args.ranking)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {
                "season": args.season,
                "scrape_date": scrape_date,
                "predictions_csv": str(csv_path),
                "model_slot": args.model_slot,
                "ranking": args.ranking,
                "summary": summary,
            },
            indent=2,
        ))
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

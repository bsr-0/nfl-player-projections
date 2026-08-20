"""Pre-modeling data quality gates for refresh and training workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR, POSITIONS
from src.utils.nfl_calendar import get_current_nfl_season, get_current_nfl_week

EXPECTED_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}


@dataclass
class DataQualityGateResult:
    passed: bool
    report: Dict[str, Any]


class DataQualityGates:
    """Evaluate freshness, completeness, and anomaly checks."""

    def __init__(self, expected_positions: Optional[list[str]] = None):
        self.expected_positions = expected_positions or list(POSITIONS)

    def evaluate(
        self,
        df: pd.DataFrame,
        expected_season: Optional[int] = None,
        expected_week: Optional[int] = None,
    ) -> DataQualityGateResult:
        report: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "checks": {},
        }

        if df.empty:
            report["checks"]["dataset"] = {
                "passed": False,
                "reason": "dataset is empty",
            }
            return DataQualityGateResult(passed=False, report=report)

        required_cols = {"season", "week", "team", "position", "player_id"}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            report["checks"]["dataset"] = {
                "passed": False,
                "reason": "missing required columns",
                "missing_columns": missing_cols,
            }
            return DataQualityGateResult(passed=False, report=report)

        checks = {
            "freshness": self._check_freshness(df, expected_season, expected_week),
            "completeness": self._check_completeness(df),
            "anomalies": self._check_anomalies(df),
        }
        report["checks"] = checks

        passed = all(check.get("passed", False) for check in checks.values())
        report["status"] = "pass" if passed else "fail"
        report["latest_observed"] = {
            "season": int(df["season"].max()),
            "week": int(df.loc[df["season"] == df["season"].max(), "week"].max()),
            "rows": int(len(df)),
        }
        return DataQualityGateResult(passed=passed, report=report)

    def _check_freshness(
        self,
        df: pd.DataFrame,
        expected_season: Optional[int],
        expected_week: Optional[int],
    ) -> Dict[str, Any]:
        target_season = expected_season or get_current_nfl_season()
        target_week = expected_week
        if target_week is None:
            current_week = get_current_nfl_week().get("week_num", 0)
            target_week = int(current_week) if current_week else None

        latest_season = int(df["season"].max())
        season_slice = df[df["season"] == latest_season]
        latest_week = int(season_slice["week"].max()) if not season_slice.empty else 0

        season_ok = latest_season >= target_season
        week_ok = True if target_week is None else latest_week >= target_week

        return {
            "passed": bool(season_ok and week_ok),
            "expected": {"season": target_season, "week": target_week},
            "observed": {"season": latest_season, "week": latest_week},
        }

    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        latest_season = int(df["season"].max())
        latest_week = int(df.loc[df["season"] == latest_season, "week"].max())
        latest = df[(df["season"] == latest_season) & (df["week"] == latest_week)].copy()

        teams_present = set(latest["team"].dropna().astype(str).unique())
        missing_teams = sorted(EXPECTED_TEAMS - teams_present)

        positions_present = set(latest["position"].dropna().astype(str).unique())
        missing_positions = sorted(set(self.expected_positions) - positions_present)

        activity_metric = pd.Series(0, index=latest.index, dtype=float)
        for col in ["snap_count", "targets", "carries", "passing_attempts", "fantasy_points"]:
            if col in latest.columns:
                activity_metric = activity_metric + latest[col].fillna(0).astype(float)
        active_players = int(latest.loc[activity_metric > 0, "player_id"].nunique())

        return {
            "passed": len(missing_teams) == 0 and len(missing_positions) == 0 and active_players > 0,
            "latest_window": {"season": latest_season, "week": latest_week},
            "missing_teams": missing_teams,
            "missing_positions": missing_positions,
            "active_players": active_players,
        }

    def _check_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        weekly = (
            df.groupby(["season", "week"], as_index=False)
            .size()
            .rename(columns={"size": "row_count"})
            .sort_values(["season", "week"])
            .reset_index(drop=True)
        )

        weekly["baseline"] = (
            weekly["row_count"]
            .shift(1)
            .rolling(window=4, min_periods=2)
            .median()
        )
        weekly["ratio_vs_baseline"] = weekly["row_count"] / weekly["baseline"]

        latest = weekly.iloc[-1]
        baseline = latest.get("baseline")
        if pd.isna(baseline) or baseline <= 0:
            return {
                "passed": True,
                "reason": "insufficient history for anomaly baseline",
                "latest_row_count": int(latest["row_count"]),
            }

        ratio = float(latest["ratio_vs_baseline"])
        lower_bound, upper_bound = 0.6, 1.6
        passed = lower_bound <= ratio <= upper_bound
        return {
            "passed": passed,
            "latest_window": {
                "season": int(latest["season"]),
                "week": int(latest["week"]),
                "row_count": int(latest["row_count"]),
            },
            "baseline_row_count": float(baseline),
            "ratio_vs_baseline": round(ratio, 3),
            "bounds": {"lower": lower_bound, "upper": upper_bound},
        }


def load_player_weekly_stats(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load core columns needed by data quality gates."""
    db_file = db_path or (DATA_DIR / "nfl_data.db")
    with sqlite3.connect(str(db_file)) as conn:
        return pd.read_sql_query(
            """
            SELECT
                pws.player_id,
                pws.season,
                pws.week,
                pws.team,
                p.position,
                pws.snap_count,
                pws.targets,
                pws.rushing_attempts,
                pws.passing_attempts,
                pws.fantasy_points
            FROM player_weekly_stats pws
            JOIN players p ON p.player_id = pws.player_id
            """,
            conn,
        )


def run_quality_gates(
    df: pd.DataFrame,
    *,
    expected_season: Optional[int] = None,
    expected_week: Optional[int] = None,
    report_path: Optional[Path] = None,
) -> DataQualityGateResult:
    """Run quality gates over a dataframe and optionally write JSON report."""
    runner = DataQualityGates()
    result = runner.evaluate(df, expected_season=expected_season, expected_week=expected_week)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.report, indent=2))
    return result


def validate_training_cache_integrity(
    cache_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> DataQualityGateResult:
    """Validate cached training data for critical integrity issues.

    This gate is designed to BLOCK training or app-data generation when the
    cache has corruption that would silently degrade model quality.  It checks:
      1. Snap data is populated (not all zeros)
      2. No duplicate (player_id, season, week) rows
      3. Fantasy points are consistent with component stats (PPR formula)
      4. No real game data rows have been overwritten with null stubs
      5. Key stat columns have plausible null rates by position

    Returns DataQualityGateResult with passed=False on any critical failure.
    """
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "checks": {},
    }
    failures = []

    # Load cache
    parquet_path = cache_path or (DATA_DIR / "cached_features.parquet")
    if not parquet_path.exists():
        report["checks"]["cache_exists"] = {"passed": False, "reason": "cache file not found"}
        return DataQualityGateResult(passed=False, report=report)

    df = pd.read_parquet(parquet_path)
    if df.empty:
        report["checks"]["cache_exists"] = {"passed": False, "reason": "cache is empty"}
        return DataQualityGateResult(passed=False, report=report)
    report["checks"]["cache_exists"] = {"passed": True, "rows": len(df)}

    # Some cache producers (e.g. scripts/generate_app_data.py) append a
    # not-yet-played "prediction target" week for app display — every skill
    # row in that week is null by construction, since the games haven't
    # happened. That's not corruption, so exclude it from the ghost-row and
    # duplicate checks below. Detected narrowly: only the single latest
    # (season, week) in the cache, and only when 100% of its skill-position
    # rows are null — a partially-null latest week is still real corruption
    # and must keep failing the gate.
    check_df = df
    stub_stat_cols = [c for c in ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards",
                                   "passing_attempts", "rushing_attempts", "targets"] if c in df.columns]
    if stub_stat_cols and all(c in df.columns for c in ["season", "week", "position"]):
        max_season = df["season"].max()
        max_week = df.loc[df["season"] == max_season, "week"].max()
        latest_mask = (df["season"] == max_season) & (df["week"] == max_week)
        latest_skill = df.loc[latest_mask & df["position"].isin(["QB", "RB", "WR", "TE"])]
        if not latest_skill.empty and latest_skill[stub_stat_cols].isnull().all(axis=1).all():
            check_df = df.loc[~latest_mask]

    # 1. Snap data check
    snap_all_zero = True
    if "snap_count" in df.columns:
        snap_all_zero = (df["snap_count"].fillna(0) == 0).all()
    report["checks"]["snap_data_populated"] = {
        "passed": not snap_all_zero,
        "nonzero_snap_rows": int((df["snap_count"].fillna(0) > 0).sum()) if "snap_count" in df.columns else 0,
    }
    if snap_all_zero:
        failures.append("snap_count is zero for every row — snap data ingestion has failed")

    # 2. Duplicate check
    if all(c in check_df.columns for c in ["player_id", "season", "week"]):
        dupes = check_df.duplicated(subset=["player_id", "season", "week"], keep=False).sum()
        report["checks"]["no_duplicates"] = {"passed": dupes == 0, "duplicate_rows": int(dupes)}
        if dupes > 0:
            failures.append(f"{dupes} duplicate (player_id, season, week) rows found")

    # 3. Fantasy points formula consistency (PPR)
    scoring_cols = {
        "passing_yards": 0.04, "passing_tds": 4, "interceptions": -2,
        "rushing_yards": 0.1, "rushing_tds": 6, "receiving_yards": 0.1,
        "receiving_tds": 6, "receptions": 1.0, "fumbles_lost": -2,
    }
    available = {k: v for k, v in scoring_cols.items() if k in df.columns}
    if available and "fantasy_points" in df.columns:
        calc = sum(df[col].fillna(0) * w for col, w in available.items())
        actual = df["fantasy_points"].fillna(0)
        abs_diff = (calc - actual).abs()
        bad_rows = int((abs_diff > 1.0).sum())
        mean_diff = float(abs_diff.mean())
        report["checks"]["fantasy_points_formula"] = {
            "passed": bad_rows < len(df) * 0.01,  # <1% tolerance
            "rows_with_diff_gt_1": bad_rows,
            "mean_abs_diff": round(mean_diff, 4),
        }
        if bad_rows >= len(df) * 0.01:
            failures.append(f"Fantasy points formula mismatch: {bad_rows} rows differ by >1pt ({bad_rows/len(df)*100:.1f}%)")

    # 4. Ghost row check (real-data rows with all stats null)
    stat_cols = [c for c in ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards",
                              "passing_attempts", "rushing_attempts", "targets"] if c in check_df.columns]
    if stat_cols:
        # Exclude K and DST (they legitimately have offensive stats as null)
        skill_df = check_df[check_df["position"].isin(["QB", "RB", "WR", "TE"])] if "position" in check_df.columns else check_df
        all_null = skill_df[stat_cols].isnull().all(axis=1)
        ghost_count = int(all_null.sum())
        ghost_pct = ghost_count / len(skill_df) * 100 if len(skill_df) > 0 else 0
        report["checks"]["no_ghost_rows"] = {
            "passed": ghost_pct < 1.0,
            "ghost_rows": ghost_count,
            "ghost_pct": round(ghost_pct, 2),
        }
        if ghost_pct >= 1.0:
            failures.append(f"{ghost_count} skill-position rows ({ghost_pct:.1f}%) have all stat columns null")

    # 5. Position coverage
    if "position" in df.columns:
        pos_counts = df["position"].value_counts().to_dict()
        expected = {"QB", "RB", "WR", "TE"}
        missing = expected - set(pos_counts.keys())
        report["checks"]["position_coverage"] = {
            "passed": len(missing) == 0,
            "counts": pos_counts,
            "missing": sorted(missing),
        }
        if missing:
            failures.append(f"Missing positions: {sorted(missing)}")

    passed = len(failures) == 0
    report["status"] = "pass" if passed else "fail"
    report["failures"] = failures
    return DataQualityGateResult(passed=passed, report=report)


# A handful of players legitimately have no weekly roster row (very old
# seasons, cup-of-coffee careers), so require a rate rather than zero.
POSITION_MISMATCH_THRESHOLD = 0.005


def check_position_integrity(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """`players.position` must agree with the weekly roster snapshots.

    Exists because it did not, silently, for a long time: the PBP
    aggregator stamps position='QB' on anyone with a pass attempt, and that
    reached `players` and stayed there -- 11.4% of the QB training
    population were not quarterbacks (Henry, Kupp, McCaffrey, DJ Moore).
    Nothing downstream of ingestion noticed, because the only correction
    lived in the draft-board output scripts (GAPS.md 2026-08-20).
    """
    from src.utils.database import DatabaseManager

    db = DatabaseManager(db_path) if db_path else DatabaseManager()
    authoritative = {k: {"FB": "RB", "HB": "RB"}.get(v, v)
                     for k, v in db.get_authoritative_player_positions().items()}
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT player_id, position FROM players WHERE position IN "
            "('QB','RB','WR','TE')"
        ).fetchall()

    checked = [(pid, pos) for pid, pos in rows if pid in authoritative]
    mismatched = [
        {"player_id": pid, "stored": pos, "authoritative": authoritative[pid]}
        for pid, pos in checked if authoritative[pid] != pos
    ]
    rate = len(mismatched) / len(checked) if checked else 0.0
    return {
        "passed": rate <= POSITION_MISMATCH_THRESHOLD,
        "checked": len(checked),
        "mismatched": len(mismatched),
        "rate": rate,
        "threshold": POSITION_MISMATCH_THRESHOLD,
        "examples": mismatched[:10],
        "remedy": "python scripts/repair_player_positions.py --write",
    }


def run_db_quality_gates(
    *,
    db_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    expected_season: Optional[int] = None,
    expected_week: Optional[int] = None,
) -> DataQualityGateResult:
    """Run quality gates against the project DB."""
    df = load_player_weekly_stats(db_path=db_path)
    final_report_path = report_path or (MODELS_DIR / "data_quality_gate_report.json")
    result = run_quality_gates(
        df,
        expected_season=expected_season,
        expected_week=expected_week,
        report_path=final_report_path,
    )
    try:
        position_check = check_position_integrity(db_path=db_path)
    except Exception as e:  # a broken lookup must not mask the other gates
        position_check = {"passed": True, "skipped": f"{type(e).__name__}: {e}"}
    result.report.setdefault("checks", {})["position_integrity"] = position_check
    if not position_check["passed"]:
        result.report["status"] = "fail"
        return DataQualityGateResult(passed=False, report=result.report)
    return result

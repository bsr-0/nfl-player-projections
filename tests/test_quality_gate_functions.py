"""Coverage for the quality-gate functions that had none.

`tests/test_data_quality_gates.py` covers `_check_completeness`,
`_check_anomalies`, and `_truncate_to_complete_window`. The four functions
exercised here -- `_check_freshness`, `validate_training_cache_integrity`,
`check_position_integrity`, `check_feature_season_continuity` -- had zero
tests: the original `tests/test_quality_gates.py` was removed in a past
"Delete tests directory" commit and never restored. These are the gates that
are supposed to BLOCK a bad retrain, so an untested gate is a gate that can
silently stop blocking.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from config.settings import SCORING
from src.data.quality_gates import (
    KNOWN_MISSINGNESS_BOUNDARIES,
    POSITION_MISMATCH_THRESHOLD,
    DataQualityGates,
    check_feature_season_continuity,
    check_position_integrity,
    validate_training_cache_integrity,
)


# --------------------------------------------------------------------------
# _check_freshness
# --------------------------------------------------------------------------

def _stats(pairs):
    return pd.DataFrame([{"season": s, "week": w, "fantasy_points": 1.0} for s, w in pairs])


def test_freshness_passes_when_data_reaches_the_expected_week():
    gates = DataQualityGates()
    result = gates._check_freshness(_stats([(2026, 1), (2026, 2), (2026, 3)]),
                                    expected_season=2026, expected_week=3)
    assert result["passed"]
    assert result["observed"] == {"season": 2026, "week": 3}


def test_freshness_fails_on_a_stale_week():
    gates = DataQualityGates()
    result = gates._check_freshness(_stats([(2026, 1), (2026, 2)]),
                                    expected_season=2026, expected_week=5)
    assert not result["passed"]
    assert result["observed"]["week"] == 2 and result["expected"]["week"] == 5


def test_freshness_fails_on_a_stale_season():
    gates = DataQualityGates()
    result = gates._check_freshness(_stats([(2024, 18)]),
                                    expected_season=2026, expected_week=1)
    assert not result["passed"]


def test_freshness_reads_latest_week_within_the_latest_season_only():
    """A prior season's week 18 must not satisfy a current-season week check.

    `latest_week` is computed inside the latest-season slice; taking a global
    MAX(week) would let last season's full schedule mask a current season
    that has only played one week.
    """
    gates = DataQualityGates()
    result = gates._check_freshness(_stats([(2025, 18), (2026, 1)]),
                                    expected_season=2026, expected_week=4)
    assert result["observed"] == {"season": 2026, "week": 1}
    assert not result["passed"]


def test_freshness_ignores_week_when_no_expected_week_is_resolvable():
    gates = DataQualityGates()
    result = gates._check_freshness(_stats([(2026, 1)]),
                                    expected_season=2026, expected_week=None)
    # expected_week=None falls through to the live calendar; whatever it
    # returns, the season check alone must be able to pass.
    assert result["expected"]["season"] == 2026
    if result["expected"]["week"] is None:
        assert result["passed"]


# --------------------------------------------------------------------------
# validate_training_cache_integrity
# --------------------------------------------------------------------------

def _cache_frame(n=200, season=2025):
    """A clean cache: consistent PPR points, populated snaps, no duplicates."""
    rows = []
    for i in range(n):
        receptions = i % 7
        receiving_yards = 10.0 * (i % 9)
        rows.append({
            "player_id": f"p{i}", "season": season, "week": 1 + (i % 4),
            "position": ["QB", "RB", "WR", "TE"][i % 4],
            "snap_count": 30 + (i % 10),
            "receptions": receptions, "receiving_yards": receiving_yards,
            "passing_yards": 0.0, "passing_tds": 0, "interceptions": 0,
            "rushing_yards": 0.0, "rushing_tds": 0, "receiving_tds": 0,
            "fumbles_lost": 0, "two_point_conversions": 0,
            "rushing_attempts": 0, "passing_attempts": 0, "targets": receptions,
        })
    df = pd.DataFrame(rows)
    df["fantasy_points"] = sum(df[c] * w for c, w in SCORING.items() if c in df.columns)
    return df


STUB_STAT_COLS = ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards",
                  "passing_attempts", "rushing_attempts", "targets"]


def _stub_week(n=40, week=99):
    """A not-yet-played "prediction target" week as generate_app_data.py
    appends it: every stat null, including every scoring component.
    """
    stub = _cache_frame(n=n).assign(week=week)
    stub["player_id"] = [f"u{i}" for i in range(n)]
    stub[sorted(set(STUB_STAT_COLS) | set(SCORING))] = np.nan
    return stub


def _write(tmp_path, df, name="cached_features.parquet"):
    path = tmp_path / name
    df.to_parquet(path)
    return path


def test_cache_gate_passes_on_a_clean_cache(tmp_path):
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, _cache_frame()))
    assert result.passed, result.report.get("failures")
    assert result.report["status"] == "pass"


def test_cache_gate_fails_when_the_file_is_absent(tmp_path):
    result = validate_training_cache_integrity(cache_path=tmp_path / "nope.parquet")
    assert not result.passed
    assert result.report["checks"]["cache_exists"]["reason"] == "cache file not found"


def test_cache_gate_fails_on_an_empty_cache(tmp_path):
    empty = _cache_frame().iloc[:0]
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, empty))
    assert not result.passed
    assert result.report["checks"]["cache_exists"]["reason"] == "cache is empty"


def test_cache_gate_fails_when_snap_ingestion_produced_all_zeros(tmp_path):
    df = _cache_frame()
    df["snap_count"] = 0
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert not result.report["checks"]["snap_data_populated"]["passed"]
    assert any("snap_count is zero" in f for f in result.report["failures"])


def test_cache_gate_fails_on_duplicate_player_weeks(tmp_path):
    df = pd.concat([_cache_frame(), _cache_frame().iloc[:5]], ignore_index=True)
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert result.report["checks"]["no_duplicates"]["duplicate_rows"] == 10


def test_cache_gate_catches_a_fantasy_points_formula_mismatch(tmp_path):
    df = _cache_frame()
    df["fantasy_points"] = df["fantasy_points"] + 9.0
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert not result.report["checks"]["fantasy_points_formula"]["passed"]


def test_cache_gate_scores_two_point_conversions(tmp_path):
    """Regression for AUDIT_REPORT.md #24: an inline scoring copy here once
    omitted two_point_conversions, so a 2PC-only discrepancy was unflaggable.
    """
    assert "two_point_conversions" in SCORING
    df = _cache_frame()
    df["two_point_conversions"] = 1  # points column left stale -> 2pt/row short
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert not result.report["checks"]["fantasy_points_formula"]["passed"]


def test_cache_gate_fails_on_ghost_rows(tmp_path):
    """Real-data skill rows whose every stat column is null are overwritten
    data, not a bye week -- above 1% that must block training.
    """
    df = _cache_frame()
    stat_cols = ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards",
                 "passing_attempts", "rushing_attempts", "targets"]
    df.loc[df.index[:20], stat_cols] = np.nan
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert result.report["checks"]["no_ghost_rows"]["ghost_rows"] == 20


def test_cache_gate_exempts_an_all_null_prediction_target_week(tmp_path):
    """generate_app_data.py appends the upcoming, not-yet-played week; every
    skill row in it is null by construction. That is not corruption.
    """
    result = validate_training_cache_integrity(
        cache_path=_write(tmp_path, pd.concat([_cache_frame(), _stub_week()], ignore_index=True)))
    assert result.passed, result.report.get("failures")


def test_cache_gate_formula_check_is_not_stub_exempt(tmp_path):
    """Deliberate asymmetry, pinned so it isn't changed by accident: the
    duplicate and ghost-row checks run on the stub-exempted `check_df`, but
    the PPR formula check runs on the full `df`. A stub week only survives it
    because *every* scoring component is null alongside fantasy_points. Leave
    one component populated and the gate fires -- correctly, since a row with
    5 receptions and null points is not a not-yet-played game.
    """
    stub = _stub_week()
    stub["receptions"] = 5.0
    result = validate_training_cache_integrity(
        cache_path=_write(tmp_path, pd.concat([_cache_frame(), stub], ignore_index=True)))
    assert not result.passed
    assert not result.report["checks"]["fantasy_points_formula"]["passed"]
    # ...while the checks that ARE stub-exempt still pass on the same frame.
    assert result.report["checks"]["no_ghost_rows"]["passed"]


def test_cache_gate_still_fails_a_partially_null_latest_week(tmp_path):
    """The stub exemption is deliberately narrow: only a 100%-null latest
    week is a prediction target. A half-null one is real corruption.
    """
    upcoming = _cache_frame(n=40).assign(week=99)
    upcoming["player_id"] = [f"u{i}" for i in range(len(upcoming))]
    # 35 of 40 null, not all -- so the stub exemption must not apply.
    upcoming.loc[upcoming.index[:35], sorted(set(STUB_STAT_COLS) | set(SCORING))] = np.nan
    result = validate_training_cache_integrity(
        cache_path=_write(tmp_path, pd.concat([_cache_frame(), upcoming], ignore_index=True)))
    assert not result.passed
    assert result.report["checks"]["no_ghost_rows"]["ghost_rows"] == 35


def test_cache_gate_fails_when_a_position_is_missing_entirely(tmp_path):
    df = _cache_frame()
    df = df[df["position"] != "TE"]
    result = validate_training_cache_integrity(cache_path=_write(tmp_path, df))
    assert not result.passed
    assert result.report["checks"]["position_coverage"]["missing"] == ["TE"]


# --------------------------------------------------------------------------
# check_position_integrity
# --------------------------------------------------------------------------

def _position_db(tmp_path, players, roster):
    """Minimal DB with the two tables check_position_integrity reads."""
    path = tmp_path / "pos.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE players (player_id TEXT PRIMARY KEY, position TEXT)")
    conn.executemany("INSERT INTO players VALUES (?, ?)", players)
    conn.execute(
        "CREATE TABLE weekly_rosters_v2 ("
        "player_id TEXT, season INTEGER, week INTEGER, position TEXT)"
    )
    conn.executemany(
        "INSERT INTO weekly_rosters_v2 VALUES (?, ?, ?, ?)",
        [(pid, 2025, 1, pos) for pid, pos in roster],
    )
    conn.commit()
    conn.close()
    return path


def test_position_integrity_passes_when_players_match_the_roster(tmp_path):
    pairs = [(f"p{i}", ["QB", "RB", "WR", "TE"][i % 4]) for i in range(40)]
    result = check_position_integrity(db_path=_position_db(tmp_path, pairs, pairs))
    assert result["passed"]
    assert result["checked"] == 40 and result["mismatched"] == 0


def test_position_integrity_fails_the_pbp_aggregator_qb_stamp(tmp_path):
    """The bug this gate exists for: the PBP aggregator stamps position='QB'
    on anyone with a pass attempt, so RBs/WRs entered the QB population.
    """
    roster = [(f"p{i}", ["QB", "RB", "WR", "TE"][i % 4]) for i in range(40)]
    players = list(roster)
    for i in (1, 5, 9, 13):  # four RBs restamped as QBs
        players[i] = (f"p{i}", "QB")
    result = check_position_integrity(db_path=_position_db(tmp_path, players, roster))
    assert not result["passed"]
    assert result["mismatched"] == 4
    assert result["rate"] > POSITION_MISMATCH_THRESHOLD
    assert {e["stored"] for e in result["examples"]} == {"QB"}
    assert "repair_player_positions" in result["remedy"]


def test_position_integrity_treats_fb_and_hb_as_rb(tmp_path):
    """Roster feeds use FB/HB; `players` stores RB. That is a labelling
    convention, not a mismatch, and must not consume the error budget.
    """
    players = [("a", "RB"), ("b", "RB"), ("c", "WR")]
    roster = [("a", "FB"), ("b", "HB"), ("c", "WR")]
    result = check_position_integrity(db_path=_position_db(tmp_path, players, roster))
    assert result["passed"] and result["mismatched"] == 0


def test_position_integrity_ignores_players_with_no_roster_history(tmp_path):
    """Old seasons and cup-of-coffee careers have no roster row; they are
    unknowable rather than wrong, so they leave the denominator alone.
    """
    players = [("a", "QB"), ("b", "RB"), ("ghost", "WR")]
    roster = [("a", "QB"), ("b", "RB")]
    result = check_position_integrity(db_path=_position_db(tmp_path, players, roster))
    assert result["checked"] == 2 and result["passed"]


# --------------------------------------------------------------------------
# check_feature_season_continuity
# --------------------------------------------------------------------------

def _panel(seasons, values_by_season, col="feat", n=100, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        mean = values_by_season[season]
        vals = np.full(n, np.nan) if mean is None else rng.normal(mean, 1.0, n)
        frames.append(pd.DataFrame({"season": season, col: vals}))
    return pd.concat(frames, ignore_index=True)


def test_continuity_passes_on_a_gradual_league_trend():
    """A real trend moves gradually; an ingestion break moves in one step.
    Five seasons drifting well inside one pooled sd must not fire.
    """
    df = _panel([2021, 2022, 2023, 2024, 2025],
                {2021: 10.0, 2022: 10.2, 2023: 10.4, 2024: 10.6, 2025: 10.8})
    result = check_feature_season_continuity(df, ["feat"])
    assert result["passed"], result["examples"]
    assert result["checked_features"] == 1


def test_continuity_catches_a_level_break_between_adjacent_seasons():
    """The team_plays_roll3_mean shape: ~52 for years, then a step."""
    df = _panel([2022, 2023, 2024], {2022: 52.0, 2023: 52.1, 2024: 3.0})
    result = check_feature_season_continuity(df, ["feat"])
    assert not result["passed"]
    top = result["examples"][0]
    assert (top["from_season"], top["to_season"]) == (2023, 2024)
    assert top["sd_shift"] > 1.0


def test_continuity_catches_a_missingness_cliff_with_a_flat_mean():
    """A column going fully populated -> fully NaN never moves its mean, so
    a level-only check would miss it entirely.
    """
    df = _panel([2023, 2024], {2023: 5.0, 2024: None})
    result = check_feature_season_continuity(df, ["feat"])
    assert not result["passed"]
    assert result["examples"][0]["missingness_jump"] == pytest.approx(1.0)


def test_continuity_exempts_documented_source_start_boundaries():
    """FTN charting genuinely starts partway through the panel. A gate that
    always fails is a gate nobody reads -- the NaN boundary is exempt.
    """
    col = next(iter(KNOWN_MISSINGNESS_BOUNDARIES))
    df = _panel([2021, 2022], {2021: None, 2022: 0.4}, col=col)
    assert check_feature_season_continuity(df, [col])["passed"]


def test_continuity_exempts_caller_supplied_boundaries_too():
    df = _panel([2021, 2022], {2021: None, 2022: 0.4})
    assert not check_feature_season_continuity(df, ["feat"])["passed"]
    assert check_feature_season_continuity(df, ["feat"], known_boundaries=["feat"])["passed"]


def test_continuity_still_checks_levels_within_an_exempt_columns_observed_seasons():
    """Only the NaN boundary is exempt; a level break among the observed
    seasons of an exempt column must still fire.
    """
    col = next(iter(KNOWN_MISSINGNESS_BOUNDARIES))
    df = _panel([2021, 2022, 2023], {2021: None, 2022: 0.4, 2023: 40.0}, col=col)
    result = check_feature_season_continuity(df, [col])
    assert not result["passed"]
    assert (result["examples"][0]["from_season"], result["examples"][0]["to_season"]) == (2022, 2023)


def test_continuity_skips_missing_and_non_numeric_columns():
    df = _panel([2023, 2024], {2023: 1.0, 2024: 1.1})
    df["label"] = "x"
    result = check_feature_season_continuity(df, ["feat", "label", "not_a_column"])
    assert result["passed"]
    assert result["checked_features"] == 2  # counts present columns, numeric or not


def test_continuity_threshold_is_configurable():
    df = _panel([2023, 2024], {2023: 10.0, 2024: 11.5})
    assert check_feature_season_continuity(df, ["feat"], threshold=0.5)["violations"] == 1
    assert check_feature_season_continuity(df, ["feat"], threshold=5.0)["violations"] == 0

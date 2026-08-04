"""Tests that validate backtest infrastructure correctness.

These tests verify that the backtest pipeline measures model quality
accurately. A backtest that uses leaked features or corrupted inputs
produces misleading metrics — the tests here catch those failures.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.leakage import is_leakage_feature, filter_feature_columns

DATA_DIR = Path(__file__).parent.parent / "data"
def _latest_backtest_json() -> Path:
    """Return the most recent backtest_2025_*.json, or a default path."""
    candidates = sorted(
        (DATA_DIR / "backtest_results").glob("backtest_2025_*.json"),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else DATA_DIR / "backtest_results" / "backtest_2025.json"


def _backtest_matches_current_models(backtest_path: Path) -> bool:
    """Check if the backtest was generated with the current feature version."""
    metadata_path = DATA_DIR / "models" / "model_metadata.json"
    if not metadata_path.exists() or not backtest_path.exists():
        return True  # Can't verify; assume it matches
    meta = json.loads(metadata_path.read_text())
    model_date = meta.get("training_date", "")[:10]
    # Backtest filename encodes date: backtest_2025_YYYYMMDD.json
    backtest_date = backtest_path.stem.split("_")[-1][:8]
    if len(backtest_date) == 8:
        backtest_date_fmt = f"{backtest_date[:4]}-{backtest_date[4:6]}-{backtest_date[6:]}"
        return backtest_date_fmt >= model_date[:10]
    return True


BACKTEST_2025 = _latest_backtest_json()
BOUNDS_PATH = DATA_DIR / "utilization_percentile_bounds.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_backtest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_bounds(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Root Cause 1: Percentile bounds must never be zero-width
# ---------------------------------------------------------------------------

class TestPercentileBoundsValidity:
    """Verify that persisted percentile bounds are usable."""

    def test_percentile_bounds_no_zero_width(self):
        """No bound should have lo == hi; zero-width bounds cause rank-based
        fallback that leaks test-set information."""
        bounds = _load_bounds(BOUNDS_PATH)
        zero_width = []
        for key, value in bounds.items():
            if key == "__meta__":
                continue
            lo, hi = value
            if lo == hi:
                zero_width.append((key, lo, hi))
        assert zero_width == [], (
            f"Found {len(zero_width)} zero-width percentile bounds "
            f"(lo == hi). These cause rank-based fallback that leaks "
            f"test-set information:\n"
            + "\n".join(f"  {k}: [{lo}, {hi}]" for k, lo, hi in zero_width)
        )

    def test_percentile_bounds_have_reasonable_spread(self):
        """Bounds where hi - lo < 1.0 are effectively constant and provide
        no discriminative power."""
        bounds = _load_bounds(BOUNDS_PATH)
        narrow = []
        for key, value in bounds.items():
            if key == "__meta__":
                continue
            lo, hi = value
            if hi - lo < 1.0:
                narrow.append((key, lo, hi))
        assert narrow == [], (
            f"Found {len(narrow)} near-zero-width bounds (spread < 1.0):\n"
            + "\n".join(f"  {k}: [{lo}, {hi}]" for k, lo, hi in narrow)
        )


# ---------------------------------------------------------------------------
# Root Cause 2: Backtester must apply leakage filters
# ---------------------------------------------------------------------------

class TestBacktesterLeakageFiltering:
    """Verify that the backtester excludes leakage-prone columns."""

    def test_backtester_excludes_known_leakage_columns(self):
        """The ts_backtester feature selection must exclude columns that
        the training pipeline filters out via filter_feature_columns().

        We simulate the backtester's full feature selection pipeline
        (exclude set + filter_feature_columns) and verify no leakage
        columns survive."""
        # Columns that might appear in the data and ARE leakage
        leakage_columns = [
            "utilization_score",
            "predicted_points",
            "predicted_utilization",
            "baseline_trailing_avg",
            "target_1w",
            "target_util_1w",
            "projection_fp",
            "actual_for_backtest",
        ]
        # Columns that are legitimate features
        safe_columns = [
            "snap_share_pct",
            "rush_share_pct",
            "rolling_3w_fantasy_points",
            "lag_1_fantasy_points",
            "opponent_defense_rank",
        ]
        all_cols = safe_columns + leakage_columns

        # Step 1: backtester exclude set (ts_backtester.py:322-328)
        backtester_exclude = {
            "player_id", "name", "position", "team", "opponent",
            "season", "week", "home_away", "created_at", "id",
            "game_date", "fantasy_points",
        }
        backtester_exclude |= {c for c in all_cols if c.startswith("target_")}
        backtester_features = [c for c in all_cols if c not in backtester_exclude]

        # Step 2: leakage filter (ts_backtester.py now calls this)
        backtester_features = filter_feature_columns(backtester_features)

        # After both steps, no leakage columns should remain
        leaked = [c for c in backtester_features if is_leakage_feature(c)]
        assert leaked == [], (
            f"Backtester feature selection allows {len(leaked)} leakage "
            f"columns after filtering:\n"
            + "\n".join(f"  - {c}" for c in leaked)
        )

    def test_backtester_imports_leakage_filter(self):
        """Verify that ts_backtester.py imports filter_feature_columns.
        This is a structural check to prevent regression."""
        import importlib
        ts_backtester = importlib.import_module("src.evaluation.ts_backtester")
        assert hasattr(ts_backtester, "filter_feature_columns"), (
            "ts_backtester.py must import filter_feature_columns from "
            "src.utils.leakage to match the training pipeline."
        )

    def test_filter_feature_columns_removes_leakage(self):
        """Sanity check: filter_feature_columns() correctly removes
        known leakage patterns."""
        cols = [
            "snap_share_pct",
            "utilization_score",
            "predicted_points",
            "baseline_trailing_avg",
            "rolling_3w_fantasy_points",
        ]
        filtered = filter_feature_columns(cols)
        assert "utilization_score" not in filtered
        assert "predicted_points" not in filtered
        assert "baseline_trailing_avg" not in filtered
        assert "snap_share_pct" in filtered
        assert "rolling_3w_fantasy_points" in filtered


# ---------------------------------------------------------------------------
# Root Cause 4: Systematic prediction bias
# ---------------------------------------------------------------------------

class TestPredictionBias:
    """Verify that model predictions are calibrated (no systematic bias)."""

    @pytest.fixture()
    def backtest_2025(self):
        if not BACKTEST_2025.exists():
            pytest.skip("2025 backtest results not available")
        if not _backtest_matches_current_models(BACKTEST_2025):
            pytest.skip(
                f"Backtest {BACKTEST_2025.name} predates current models — "
                f"re-run backtest with trained models to validate bias"
            )
        return _load_backtest(BACKTEST_2025)

    def test_prediction_bias_within_tolerance(self, backtest_2025):
        """Average predicted should be within 15% of average actual.
        A larger gap indicates systematic calibration failure."""
        metrics = backtest_2025["metrics"]
        avg_pred = metrics["avg_predicted"]
        avg_actual = metrics["avg_actual"]
        bias_pct = abs(avg_pred - avg_actual) / avg_actual * 100
        assert bias_pct < 15, (
            f"Systematic prediction bias of {bias_pct:.1f}% "
            f"(predicted {avg_pred:.2f} vs actual {avg_actual:.2f}). "
            f"Must be < 15%."
        )

    def test_per_position_r2_positive(self, backtest_2025):
        """R² must be positive for every position. Negative R² means the
        model is worse than predicting the mean."""
        by_position = backtest_2025.get("by_position", {})
        negative_r2 = []
        for pos, metrics in by_position.items():
            r2 = metrics.get("r2", 0)
            if r2 < 0:
                negative_r2.append((pos, r2))
        assert negative_r2 == [], (
            f"Negative R² for {len(negative_r2)} position(s) "
            f"(worse than predicting the mean):\n"
            + "\n".join(f"  {pos}: R² = {r2:.3f}" for pos, r2 in negative_r2)
        )

    def test_overall_r2_positive(self, backtest_2025):
        """Overall R² must be positive."""
        r2 = backtest_2025["metrics"]["r2"]
        assert r2 > 0, (
            f"Overall R² = {r2:.3f} is negative. Model predictions are "
            f"worse than predicting the mean for every player."
        )

    def test_per_position_bias_within_tolerance(self, backtest_2025):
        """Per-position bias should not exceed 25%."""
        by_position = backtest_2025.get("by_position", {})
        biased = []
        for pos, metrics in by_position.items():
            avg_pred = metrics.get("avg_predicted", 0)
            avg_actual = metrics.get("avg_actual", 0)
            if avg_actual > 0:
                bias_pct = abs(avg_pred - avg_actual) / avg_actual * 100
                if bias_pct > 25:
                    biased.append((pos, avg_pred, avg_actual, bias_pct))
        assert biased == [], (
            f"Per-position bias exceeds 25% for {len(biased)} position(s):\n"
            + "\n".join(
                f"  {pos}: predicted {pred:.2f} vs actual {act:.2f} ({pct:.1f}%)"
                for pos, pred, act, pct in biased
            )
        )


# ---------------------------------------------------------------------------
# Walk-forward bias regression (council 2026-04-14, Step 2)
# ---------------------------------------------------------------------------
# Replaces the safety function of the calibration gate (deleted in commit
# 8f4f717).  The gate blended predictions toward a trailing-average baseline
# whenever they deviated >50% — that worked as a circuit breaker against the
# +44% pre-leakage-fix bias, but it also suppressed legitimate breakout
# predictions.  This regression test enforces the same bias floor without
# touching the predictions: if a future pipeline change reintroduces
# systematic over- or under-prediction, the test fails at PR time.

WALK_FORWARD_DIR = DATA_DIR / "backtest_results"
WALK_FORWARD_BIAS_TOLERANCE_PCT = 10.0


def _latest_walk_forward_predictions():
    """Return Path of the most recent ts_backtest_*_predictions.csv, or None."""
    if not WALK_FORWARD_DIR.exists():
        return None
    candidates = sorted(
        WALK_FORWARD_DIR.glob("ts_backtest_*_predictions.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class TestWalkForwardBiasRegression:
    """Pin per-position bias on the most recent walk-forward backtest.

    Tolerance: ±10% per position (council verdict, 2026-04-14).
    Reference: CRITICAL_LIMITATION.md — current per-position bias is in
    [−5.3%, +7.5%], well within tolerance.  The pre-leakage-fix February
    backtest showed +44% bias overall and would fail this test, which is
    exactly the regression mode we want to catch.

    Provenance: council-transcript-20260414-034617.md (Step 2).
    """

    @pytest.fixture(scope="class")
    def predictions_df(self):
        path = _latest_walk_forward_predictions()
        if path is None:
            pytest.skip("No walk-forward backtest predictions found")
        df = pd.read_csv(path)
        required = {"position", "predicted", "actual"}
        missing = required - set(df.columns)
        if missing:
            pytest.skip(f"Walk-forward CSV missing required columns: {missing}")
        # Drop phantom rows emitted by --emit-inactive-predictions: those have
        # predicted values but actual is NaN (the player was on the
        # cumulative-active roster but didn't play this week).  The bias
        # regression test only makes sense on rows where the model's
        # prediction can be compared to a real observed fantasy-point value.
        df = df.dropna(subset=["actual"])
        if "is_active" in df.columns:
            df = df[df["is_active"].fillna(1).astype(int) == 1]
        return df

    def test_overall_bias_within_tolerance(self, predictions_df):
        """Overall mean prediction must be within ±10% of overall mean actual."""
        avg_pred = predictions_df["predicted"].mean()
        avg_actual = predictions_df["actual"].mean()
        assert avg_actual > 0, "Cannot compute bias: avg_actual is zero or negative"
        bias_pct = (avg_pred - avg_actual) / avg_actual * 100
        assert abs(bias_pct) <= WALK_FORWARD_BIAS_TOLERANCE_PCT, (
            f"Overall walk-forward bias is {bias_pct:+.1f}% "
            f"(predicted {avg_pred:.2f} vs actual {avg_actual:.2f}, "
            f"tolerance ±{WALK_FORWARD_BIAS_TOLERANCE_PCT:.0f}%). "
            f"This is the regression the calibration gate was deleted to "
            f"replace — see council-transcript-20260414-034617.md."
        )

    def test_per_position_bias_within_tolerance(self, predictions_df):
        """Each position's mean prediction must be within ±10% of mean actual."""
        offenders = []
        for pos, group in predictions_df.groupby("position"):
            avg_pred = group["predicted"].mean()
            avg_actual = group["actual"].mean()
            if avg_actual <= 0:
                continue
            bias_pct = (avg_pred - avg_actual) / avg_actual * 100
            if abs(bias_pct) > WALK_FORWARD_BIAS_TOLERANCE_PCT:
                offenders.append((pos, avg_pred, avg_actual, bias_pct, len(group)))

        assert not offenders, (
            f"{len(offenders)} position(s) exceed "
            f"±{WALK_FORWARD_BIAS_TOLERANCE_PCT:.0f}% bias tolerance on the "
            f"walk-forward backtest ({_latest_walk_forward_predictions().name}):\n"
            + "\n".join(
                f"  {pos}: {bias:+.1f}% (predicted {pred:.2f} vs actual {act:.2f}, n={n})"
                for pos, pred, act, bias, n in offenders
            )
            + "\n\nThis is the regression the calibration gate was deleted "
            "to replace — see council-transcript-20260414-034617.md."
        )


# ---------------------------------------------------------------------------
# Quality gate — CI-enforceable, uses committed data/models/backtest_metrics.json
# ---------------------------------------------------------------------------
# This gate runs in CI (no live data required). It enforces regression floors
# against the committed measured metrics. Thresholds are conservative now;
# tighten them as model quality improves and backtest_metrics.json is updated.

QUALITY_GATE_PATH = DATA_DIR / "models" / "backtest_metrics.json"


class TestQualityGate:
    """Enforce model quality floors using committed backtest_metrics.json.

    Unlike TestPredictionBias (which skips in CI because backtest data is
    gitignored), this class always runs — the metrics file is committed.
    Thresholds are regression guards, not quality targets. Update
    backtest_metrics.json and tighten thresholds after each retrain.
    """

    @pytest.fixture()
    def gate(self):
        if not QUALITY_GATE_PATH.exists():
            pytest.skip("data/models/backtest_metrics.json not found")
        return json.loads(QUALITY_GATE_PATH.read_text())

    def test_overall_r2_above_floor(self, gate):
        r2 = gate["measured"]["overall"]["r2"]
        floor = gate["thresholds"]["min_overall_r2"]
        assert r2 >= floor, (
            f"Overall R² = {r2:.3f} is below regression floor {floor}. "
            f"Update data/models/backtest_metrics.json after retraining."
        )

    def test_per_position_r2_above_floor(self, gate):
        floor = gate["thresholds"]["min_position_r2"]
        bad = [
            (pos, m["r2"])
            for pos, m in gate["measured"]["by_position"].items()
            if m["r2"] < floor
        ]
        assert bad == [], (
            f"Positions below R² floor {floor}: "
            + ", ".join(f"{pos}={r2:.3f}" for pos, r2 in bad)
        )

    def test_overall_correlation_above_floor(self, gate):
        corr = gate["measured"]["overall"]["correlation"]
        floor = gate["thresholds"]["min_overall_correlation"]
        assert corr >= floor, (
            f"Overall correlation = {corr:.3f} is below floor {floor}."
        )

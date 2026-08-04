"""Tests for confidence-interval behavior in backtester."""

from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.backtester import ModelBacktester, check_success_criteria


def test_confidence_interval_custom_column_names():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "position": ["RB", "RB", "WR", "WR"],
        "predicted_points": [10.0, 14.0, 9.0, 13.0],
        "fantasy_points": [12.0, 13.0, 8.5, 15.0],
    })
    out = backtester.calculate_confidence_intervals(
        df,
        pred_col="predicted_points",
        actual_col="fantasy_points",
        confidence=0.8,
        lower_col="prediction_ci80_lower",
        upper_col="prediction_ci80_upper",
    )
    assert "prediction_ci80_lower" in out.columns
    assert "prediction_ci80_upper" in out.columns
    assert (out["prediction_ci80_lower"] >= 0).all()


def test_success_criteria_prefers_explicit_ci_coverage():
    payload = {
        "metrics": {
            "spearman_rho": 0.7,
            "mape": 20.0,
            "within_7_pts_pct": 75.0,
            "within_10_pts_pct": 82.0,
            "tier_classification_accuracy": 0.8,
            "std_predicted": 7.0,
            "std_actual": 8.0,
            "mae_rmse_ratio": 0.77,
            "mae_rmse_healthy": True,
        },
        "by_week": {"1": {"rmse": 7.0}, "2": {"rmse": 7.4}},
        "by_position": {},
        "multiple_baseline_comparison": {
            "model_beats_all_internal_by_20_pct": True,
            "model_beats_all_internal_by_25_pct": True,
            "baseline_season_avg": {"improvement_pct": 26.0},
            "status": {"model_has_real_edge": False},
        },
        "confidence_band_coverage_10pt": 90.0,
    }
    sc = check_success_criteria(payload)
    assert sc["confidence_band_coverage_10pt"] == 90.0
    assert sc["confidence_band_target_882"] is True


def test_compare_to_multiple_baselines_returns_expected_keys():
    backtester = ModelBacktester()
    df = pd.DataFrame({
        "player_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4", "p5", "p5"],
        "position": ["RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE"],
        "season": [2024] * 10,
        "week": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "fantasy_points": [10, 13, 9, 8, 14, 16, 6, 7, 11, 12],
        "predicted_points": [10.5, 12.5, 9.5, 8.5, 13.5, 15.0, 6.5, 7.0, 10.5, 11.5],
    })
    out = backtester.compare_to_multiple_baselines(df)
    assert "model" in out
    assert "baseline_persistence" in out
    assert "baseline_season_avg" in out
    assert "baseline_position_avg" in out
    assert "model_beats_all_internal_by_20_pct" in out or "model_beats_all_by_20_pct" in out
    assert "status" in out


def test_backtest_lineup_decisions_perfect_model():
    """Model that predicts perfectly should beat replacement and hindsight opponents."""
    backtester = ModelBacktester()
    rows = []
    for week in range(1, 6):
        # 3 QBs, 4 RBs, 4 WRs, 3 TEs per week
        for i, (pos, n) in enumerate([("QB", 3), ("RB", 4), ("WR", 4), ("TE", 3)]):
            for j in range(n):
                pts = float(10 + j * 3 + week * 0.5)
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts,  # perfect predictions
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(df)
    assert "error" not in result
    # Perfect model picks the actual best players each week, so it matches
    # the oracle lineup. vs_replacement and vs_hindsight should be wins.
    assert result["vs_replacement"]["win_rate"] == 1.0
    assert result["n_weeks"] == 5
    assert result["avg_model_score"] > 0
    # Three tiers present with p-values
    assert "vs_oracle" in result
    assert "vs_hindsight" in result
    assert "vs_replacement" in result
    # Binomial test: 5/5 wins vs replacement, p should be significant
    assert result["vs_replacement"]["p_value"] < 0.05
    assert result["vs_replacement"]["significant_at_05"] is True
    # Top-level p_value is the hindsight tier
    assert "p_value" in result
    assert "significant_at_05" in result


def test_backtest_lineup_decisions_three_opponent_tiers():
    """Verify all three opponent tiers are present and correctly ordered."""
    backtester = ModelBacktester()
    rows = []
    np.random.seed(42)
    for week in range(1, 8):
        for pos, n in [("QB", 5), ("RB", 8), ("WR", 8), ("TE", 5)]:
            for j in range(n):
                pts = float(10 + j * 2 + np.random.normal(0, 3))
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts + np.random.normal(0, 2),
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(df)
    assert "error" not in result

    # Oracle is hardest to beat, replacement is easiest
    assert result["vs_oracle"]["win_rate"] <= result["vs_replacement"]["win_rate"]

    # Weekly results have all keys
    for wr in result["weekly_results"]:
        assert "won_vs_oracle" in wr
        assert "won_vs_hindsight" in wr
        assert "won_vs_replacement" in wr
        assert "vs_oracle_margin" in wr
        assert "vs_hindsight_margin" in wr
        assert "vs_replacement_margin" in wr

    # Primary metric (win_rate) is the hindsight rate
    assert result["win_rate"] == result["vs_hindsight"]["win_rate"]
    assert result["wins"] + result["losses"] == result["n_weeks"]

    # Each tier has a p-value from binomial test
    for tier_key in ("vs_oracle", "vs_hindsight", "vs_replacement"):
        tier = result[tier_key]
        assert "p_value" in tier
        assert "significant_at_05" in tier
        assert 0.0 <= tier["p_value"] <= 1.0


def test_backtest_lineup_decisions_missing_column():
    backtester = ModelBacktester()
    df = pd.DataFrame({"x": [1, 2]})
    result = backtester.backtest_lineup_decisions(df)
    assert "error" in result


def test_backtest_lineup_decisions_custom_roster_slots():
    backtester = ModelBacktester()
    rows = []
    for week in range(1, 4):
        for pos, n in [("QB", 3), ("RB", 5), ("WR", 5), ("TE", 3)]:
            for j in range(n):
                pts = float(10 + j * 3)
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": pts,
                    "predicted_points": pts,
                })
    df = pd.DataFrame(rows)
    result = backtester.backtest_lineup_decisions(
        df, roster_slots={"QB": 1, "RB": 3, "WR": 3, "TE": 1}
    )
    assert "error" not in result
    assert result["roster_slots"] == {"QB": 1, "RB": 3, "WR": 3, "TE": 1}


def test_success_criteria_includes_lineup_win_rate():
    """check_success_criteria should pick up lineup decision results with tiers."""
    payload = {
        "metrics": {
            "spearman_rho": 0.7,
            "mape": 20.0,
            "within_7_pts_pct": 75.0,
            "within_10_pts_pct": 82.0,
            "tier_classification_accuracy": 0.8,
            "std_predicted": 7.0,
            "std_actual": 8.0,
            "mae_rmse_ratio": 0.77,
            "mae_rmse_healthy": True,
        },
        "by_week": {"1": {"rmse": 7.0}, "2": {"rmse": 7.4}},
        "by_position": {},
        "multiple_baseline_comparison": {
            "model_beats_all_internal_by_20_pct": True,
            "model_beats_all_internal_by_25_pct": True,
            "baseline_season_avg": {"improvement_pct": 26.0},
            "status": {"model_has_real_edge": True},
        },
        "lineup_decisions": {
            "win_rate": 0.65,
            "wins": 13,
            "losses": 7,
            "n_weeks": 20,
            "avg_margin": 4.5,
            "p_value": 0.0207,
            "significant_at_05": True,
            "vs_oracle": {"win_rate": 0.10, "wins": 2, "losses": 18, "p_value": 1.0, "significant_at_05": False},
            "vs_hindsight": {"win_rate": 0.65, "wins": 13, "losses": 7, "p_value": 0.0207, "significant_at_05": True},
            "vs_replacement": {"win_rate": 0.90, "wins": 18, "losses": 2, "p_value": 0.0002, "significant_at_05": True},
        },
    }
    sc = check_success_criteria(payload)
    assert sc["lineup_win_rate"] == 0.65
    assert sc["lineup_win_rate_significant"] is True
    assert sc["lineup_p_value"] == 0.0207
    assert sc["lineup_wins"] == 13
    assert sc["lineup_losses"] == 7
    # Tier data preserved
    assert sc["lineup_vs_oracle"]["win_rate"] == 0.10
    assert sc["lineup_vs_replacement"]["win_rate"] == 0.90


def _build_lineup_fixture(weeks: int = 6, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, weeks + 1):
        for pos, n in [("QB", 4), ("RB", 6), ("WR", 6), ("TE", 4)]:
            for j in range(n):
                actual = float(8 + j * 2.5 + rng.normal(0, 3))
                predicted = actual + float(rng.normal(0, 2))
                rows.append({
                    "player_id": f"{pos}{j}",
                    "position": pos,
                    "week": week,
                    "fantasy_points": actual,
                    "predicted_points": predicted,
                })
    return pd.DataFrame(rows)


def test_backtest_lineup_decisions_returns_roi():
    """ROI must appear on every tier and satisfy roi = payout * win_rate - 1."""
    backtester = ModelBacktester()
    result = backtester.backtest_lineup_decisions(_build_lineup_fixture())
    assert "error" not in result
    assert result["payout_multiplier"] == 1.8
    # break_even_win_rate is rounded to 4dp in the returned dict.
    assert abs(result["break_even_win_rate"] - (1.0 / 1.8)) < 1e-3

    # Each returned win_rate is rounded to 4dp, but roi is computed from the
    # raw (un-rounded) ratio — allow one ULP of drift when recomputing.
    for tier_key in ("vs_oracle", "vs_hindsight", "vs_replacement"):
        tier = result[tier_key]
        assert "roi" in tier, f"{tier_key} missing roi"
        expected = round(1.8 * tier["win_rate"] - 1.0, 4)
        assert abs(tier["roi"] - expected) < 2e-4, (
            f"{tier_key} roi={tier['roi']} expected≈{expected}"
        )


def test_backtest_lineup_decisions_returns_weekly_series():
    """weekly_results exposes per-week cumulative win rates, monotone in n."""
    backtester = ModelBacktester()
    result = backtester.backtest_lineup_decisions(_build_lineup_fixture(weeks=8))
    weekly = result["weekly_results"]
    assert len(weekly) == 8

    for tier in ("oracle", "hindsight", "replacement"):
        key = f"cumulative_win_rate_vs_{tier}"
        values = [w[key] for w in weekly]
        assert all(0.0 <= v <= 1.0 for v in values), f"{key} out of [0,1]: {values}"
        # Final cumulative equals the aggregate tier win_rate.
        assert abs(values[-1] - result[f"vs_{tier}"]["win_rate"]) < 1e-9

    # Each step is bounded by ±1/n from the previous (one flip per week).
    for tier in ("oracle", "hindsight", "replacement"):
        key = f"cumulative_win_rate_vs_{tier}"
        for i in range(1, len(weekly)):
            delta = abs(weekly[i][key] - weekly[i - 1][key])
            assert delta <= 1.0 / (i + 1) + 1e-9, (
                f"{key} jumped {delta} at week index {i}"
            )


def test_backtest_lineup_decisions_payout_multiplier_configurable():
    """Changing payout_multiplier shifts ROI but leaves win_rate / p_value alone."""
    backtester = ModelBacktester()
    df = _build_lineup_fixture()
    base = backtester.backtest_lineup_decisions(df, payout_multiplier=1.8)
    alt = backtester.backtest_lineup_decisions(df, payout_multiplier=2.0)

    assert base["vs_hindsight"]["win_rate"] == alt["vs_hindsight"]["win_rate"]
    assert base["vs_hindsight"]["p_value"] == alt["vs_hindsight"]["p_value"]
    assert base["payout_multiplier"] == 1.8
    assert alt["payout_multiplier"] == 2.0

    for tier_key in ("vs_oracle", "vs_hindsight", "vs_replacement"):
        wr = base[tier_key]["win_rate"]
        assert abs(base[tier_key]["roi"] - round(1.8 * wr - 1.0, 4)) < 2e-4
        assert abs(alt[tier_key]["roi"] - round(2.0 * wr - 1.0, 4)) < 2e-4


def test_compare_to_expert_consensus_with_csv(tmp_path):
    backtester = ModelBacktester()
    preds = pd.DataFrame({
        "name": [f"Player{i}" for i in range(12)],
        "fantasy_points": np.linspace(8, 20, 12),
        "predicted_points": np.linspace(8.5, 19.5, 12),
    })
    expert_path = tmp_path / "expert.csv"
    expert = pd.DataFrame({
        "player_name": [f"Player{i}" for i in range(12)],
        "proj_points": np.linspace(7.0, 18.0, 12),
    })
    expert.to_csv(expert_path, index=False)
    out = backtester.compare_to_expert_consensus(preds, str(expert_path))
    assert "model_rmse" in out
    assert "expert_rmse" in out
    assert "model_vs_expert_pct" in out

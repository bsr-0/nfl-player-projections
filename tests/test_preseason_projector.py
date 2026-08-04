from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.preseason_projector import (
    BASE_FEATURES_BY_POSITION,
    CALIBRATION_FEATURES_BY_POSITION,
    MarketAnchorCurve,
    PreseasonProjector,
    UpstreamCalibrator,
)


def _rb_rows() -> pd.DataFrame:
    rows = []
    player_num = 0
    for curr_season, season_bump in [(2024, 0.0), (2025, 6.0), (2026, 12.0)]:
        for group, carries, targets, snap, ppg, total_base, rookie in [
            ("starter", 18.0, 4.5, 0.69, 16.0, 255.0, 0.0),
            ("committee", 11.0, 3.0, 0.47, 11.5, 176.0, 0.0),
            ("backup", 5.0, 2.0, 0.24, 10.5, 102.0, 1.0),
            ("rotational", 7.5, 2.5, 0.36, 9.2, 128.0, 0.0),
        ]:
            for replica in range(8):
                player_num += 1
                rows.append(
                    {
                        "player_id": f"rb_{curr_season}_{group}_{replica}",
                        "player_name": f"RB {curr_season} {group} {replica}",
                        "position": "RB",
                        "projection_season": curr_season,
                        "curr_season": curr_season,
                        "prior_season": curr_season - 1,
                        "birth_date": "1998-01-01",
                        "years_exp": 0 if rookie else 4,
                        "ppg": ppg + 0.18 * replica,
                        "games_played": 14 + (replica % 3),
                        "snap_share": snap + 0.01 * replica,
                        "carries_pg": carries + 0.35 * replica,
                        "targets_pg": targets + 0.08 * replica,
                        "receptions_pg": 2.2 + 0.05 * replica,
                        "rushing_yards_pg": carries * 4.4 + replica,
                        "receiving_yards_pg": targets * 7.2 + 1.5 * replica,
                        "rush_share": min(0.85, snap + 0.08),
                        "target_share": min(0.22, 0.08 + targets / 60.0),
                        "season_total": total_base + season_bump + 1.7 * replica,
                        "preseason_ecr": (
                            16 + replica + (curr_season - 2024) * 3
                            if group == "starter"
                            else 78 + replica + (curr_season - 2024) * 5
                            if group == "committee"
                            else 170 + replica + (curr_season - 2024) * 9
                            if group == "backup"
                            else 118 + replica + (curr_season - 2024) * 6
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _wr_rows() -> pd.DataFrame:
    rows = []
    for curr_season, season_bump in [(2024, 0.0), (2025, 4.0), (2026, 8.0)]:
        for group, targets, snap, ppg, total_base, rookie in [
            ("starter", 9.1, 0.83, 17.2, 252.0, 0.0),
            ("committee", 6.2, 0.67, 13.0, 188.0, 0.0),
            ("backup", 3.6, 0.39, 11.8, 118.0, 1.0),
            ("rotational", 4.8, 0.52, 10.2, 142.0, 0.0),
        ]:
            for replica in range(8):
                rows.append(
                    {
                        "player_id": f"wr_{curr_season}_{group}_{replica}",
                        "player_name": f"WR {curr_season} {group} {replica}",
                        "position": "WR",
                        "projection_season": curr_season,
                        "curr_season": curr_season,
                        "prior_season": curr_season - 1,
                        "birth_date": "1999-01-01",
                        "years_exp": 0 if rookie else 3,
                        "ppg": ppg + 0.2 * replica,
                        "games_played": 13 + (replica % 4),
                        "snap_share": snap + 0.01 * replica,
                        "targets_pg": targets + 0.12 * replica,
                        "receptions_pg": 3.7 + 0.11 * replica,
                        "receiving_yards_pg": 54.0 + 2.8 * replica + targets * 5.5,
                        "air_yards_pg": 88.0 + 4.1 * replica,
                        "target_share": min(0.34, 0.09 + targets / 40.0),
                        "season_total": total_base + season_bump + 1.8 * replica,
                        "preseason_ecr": (
                            12 + replica + (curr_season - 2024) * 2
                            if group == "starter"
                            else 65 + replica + (curr_season - 2024) * 4
                            if group == "committee"
                            else 165 + replica + (curr_season - 2024) * 8
                            if group == "backup"
                            else 108 + replica + (curr_season - 2024) * 5
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _qb_rows() -> pd.DataFrame:
    rows = []
    for curr_season, season_bump in [(2024, 0.0), (2025, 5.0), (2026, 10.0)]:
        for group, pass_yds, ppg, total_base in [
            ("starter", 274.0, 19.2, 285.0),
            ("backup", 162.0, 11.5, 158.0),
        ]:
            for replica in range(10):
                rows.append(
                    {
                        "player_id": f"qb_{curr_season}_{group}_{replica}",
                        "player_name": f"QB {curr_season} {group} {replica}",
                        "position": "QB",
                        "projection_season": curr_season,
                        "curr_season": curr_season,
                        "prior_season": curr_season - 1,
                        "birth_date": "1996-01-01",
                        "years_exp": 5,
                        "ppg": ppg + 0.16 * replica,
                        "games_played": 14 + (replica % 3),
                        "snap_share": 0.95,
                        "passing_yards_pg": pass_yds + 4.0 * replica,
                        "passing_tds_pg": 1.7 + 0.06 * replica,
                        "interceptions_pg": 0.6,
                        "rushing_yards_pg": 22.0 + 1.2 * replica,
                        "completion_pct": 66.0 + 0.2 * replica,
                        "season_total": total_base + season_bump + 2.0 * replica,
                        "preseason_ecr": 15 + replica if group == "starter" else 145 + replica,
                    }
                )
    return pd.DataFrame(rows)


def _te_rows() -> pd.DataFrame:
    rows = []
    for curr_season, season_bump in [(2024, 0.0), (2025, 3.0), (2026, 6.0)]:
        for group, targets, snap, ppg, total_base in [
            ("starter", 7.2, 0.76, 13.5, 192.0),
            ("committee", 4.6, 0.58, 10.5, 140.0),
            ("backup", 2.4, 0.38, 7.8, 98.0),
        ]:
            for replica in range(8):
                rows.append(
                    {
                        "player_id": f"te_{curr_season}_{group}_{replica}",
                        "player_name": f"TE {curr_season} {group} {replica}",
                        "position": "TE",
                        "projection_season": curr_season,
                        "curr_season": curr_season,
                        "prior_season": curr_season - 1,
                        "birth_date": "1997-01-01",
                        "years_exp": 4,
                        "ppg": ppg + 0.17 * replica,
                        "games_played": 13 + (replica % 3),
                        "snap_share": snap + 0.01 * replica,
                        "targets_pg": targets + 0.11 * replica,
                        "receptions_pg": 2.8 + 0.07 * replica,
                        "receiving_yards_pg": 39.0 + 2.2 * replica + targets * 4.6,
                        "target_share": min(0.28, 0.08 + targets / 35.0),
                        "season_total": total_base + season_bump + 1.5 * replica,
                        "preseason_ecr": 28 + replica if group == "starter" else 95 + replica if group == "committee" else 182 + replica,
                    }
                )
    return pd.DataFrame(rows)


def _training_pairs() -> pd.DataFrame:
    return pd.concat([_qb_rows(), _rb_rows(), _wr_rows(), _te_rows()], ignore_index=True)


def test_prepare_feature_frame_derives_interactions_and_support_features():
    df = pd.DataFrame(
        [
            {
                "position": "RB",
                "birth_date": "2002-01-04",
                "projection_season": 2026,
                "ppg": 14.5,
                "games_played": 15,
                "snap_share": 0.33,
                "carries_pg": 8.0,
                "targets_pg": 2.1,
                "years_exp": 1,
            }
        ]
    )

    out = PreseasonProjector._prepare_feature_frame(df)

    assert out.loc[0, "ppg_x_carries_pg"] == 116.0
    assert out.loc[0, "rookie_or_low_experience"] == 1.0
    assert out.loc[0, "support_class"] in {"committee", "backup", "rotational"}
    assert 0.05 <= out.loc[0, "confidence_score"] <= 1.0
    assert out.loc[0, "low_information_score"] == 1.0 - out.loc[0, "confidence_score"]


def test_market_anchor_curve_predicts_from_preseason_ecr():
    curve = MarketAnchorCurve(
        position="RB",
        intercept=420.0,
        coef_log_ecr=-42.0,
        coef_inv_sqrt_ecr=60.0,
        sample_size=40,
    )
    ecr = pd.Series([5.0, 25.0, np.nan, 100.0])

    pred = curve.predict_series(ecr)

    assert pred.iloc[0] > pred.iloc[1] > pred.iloc[3]
    assert np.isnan(pred.iloc[2])


def test_upstream_calibrator_is_bounded_and_market_aware():
    prepared = pd.DataFrame(
        {
            "confidence_score": [0.15, 0.90],
            "low_information_score": [0.85, 0.10],
            "rookie_or_low_experience": [1.0, 0.0],
            "support_class_starter": [0.0, 1.0],
            "support_class_committee": [0.0, 0.0],
            "support_class_backup": [1.0, 0.0],
            "support_class_rotational": [0.0, 0.0],
            "games_played": [8.0, 16.0],
            "snap_share": [0.24, 0.82],
            "carries_pg": [4.0, 17.0],
            "targets_pg": [2.0, 4.6],
            "ppg_x_carries_pg": [44.0, 272.0],
            "low_volume_efficiency_flag": [1.0, 0.0],
            "market_anchor": [120.0, 240.0],
            "market_gap": [0.0, 0.0],
            "raw_pred_x_confidence": [0.0, 0.0],
            "market_gap_x_low_information": [0.0, 0.0],
        }
    )
    raw = np.array([60.0, 230.0])
    prepared["raw_pred"] = raw
    prepared["market_gap"] = prepared["market_anchor"] - raw
    prepared["raw_pred_x_confidence"] = raw * prepared["confidence_score"]
    prepared["market_gap_x_low_information"] = prepared["market_gap"] * prepared["low_information_score"]

    calibrator = UpstreamCalibrator(
        position="RB",
        features=[f for f in CALIBRATION_FEATURES_BY_POSITION["RB"] if f in prepared.columns],
        coef=[0.0] * len([f for f in CALIBRATION_FEATURES_BY_POSITION["RB"] if f in prepared.columns]),
        intercept=500.0,
        scaler_mean=[0.0] * len([f for f in CALIBRATION_FEATURES_BY_POSITION["RB"] if f in prepared.columns]),
        scaler_scale=[1.0] * len([f for f in CALIBRATION_FEATURES_BY_POSITION["RB"] if f in prepared.columns]),
        max_adjustment_share=0.30,
        market_weight_cap=0.36,
        sample_size=20,
        train_mae_before=20.0,
        train_mae_after=15.0,
    )

    pred = calibrator.calibrate(prepared, raw)

    assert pred[0] > raw[0]
    assert pred[0] < 120.0
    assert pred[1] <= raw[1] + 20.0


def test_upstream_calibrator_handles_scalar_market_anchor():
    prepared = pd.DataFrame(
        {
            "confidence_score": [0.35, 0.85],
            "low_information_score": [0.65, 0.15],
            "rookie_or_low_experience": [1.0, 0.0],
            "support_class_starter": [0.0, 1.0],
            "support_class_committee": [1.0, 0.0],
            "support_class_backup": [0.0, 0.0],
            "support_class_rotational": [0.0, 0.0],
            "games_played": [9.0, 16.0],
            "snap_share": [0.32, 0.78],
            "carries_pg": [6.0, 15.0],
            "targets_pg": [2.5, 4.0],
            "ppg_x_carries_pg": [54.0, 225.0],
            "low_volume_efficiency_flag": [1.0, 0.0],
            "raw_pred_x_confidence": [0.0, 0.0],
            "market_gap_x_low_information": [0.0, 0.0],
        }
    )
    raw = np.array([80.0, 210.0])
    features = [f for f in CALIBRATION_FEATURES_BY_POSITION["RB"] if f in prepared.columns or f == "market_anchor"]
    calibrator = UpstreamCalibrator(
        position="RB",
        features=features,
        coef=[0.0] * len(features),
        intercept=190.0,
        scaler_mean=[0.0] * len(features),
        scaler_scale=[1.0] * len(features),
        max_adjustment_share=0.20,
        market_weight_cap=0.20,
        sample_size=20,
        train_mae_before=20.0,
        train_mae_after=15.0,
    )

    out = calibrator.calibrate(prepared.assign(market_anchor=np.float64(180.0)), raw)

    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_fit_learns_position_specific_models_and_calibration():
    projector = PreseasonProjector().fit(_training_pairs())

    assert projector.variant_name in {
        "ridge_baseline",
        "position_specific_ridge",
        "position_specific_ridge_rb_construction",
        "hybrid_legacy_rb_position_specific",
        "position_specific_ridge_plus_calibrator",
    }
    assert set(projector.models) == {"QB", "RB", "WR", "TE"}
    assert "selection_report" in projector.audit_report
    assert projector.get_selection_report()["selected_variant"] == projector.variant_name
    assert projector.audit_report["overall"]["pred_mae"] <= projector.audit_report["overall"]["base_mae"] + 3.0


def test_predict_keeps_public_contract_and_can_use_market_when_present():
    projector = PreseasonProjector().fit(_training_pairs())
    players = pd.DataFrame(
        [
            {
                "player_id": "rb_test_1",
                "player_name": "RB Test 1",
                "position": "RB",
                "projection_season": 2027,
                "birth_date": "1998-01-01",
                "years_exp": 4,
                "ppg": 16.5,
                "games_played": 16,
                "snap_share": 0.69,
                "carries_pg": 18.0,
                "targets_pg": 4.2,
                "receptions_pg": 3.0,
                "rushing_yards_pg": 82.0,
                "receiving_yards_pg": 28.0,
                "rush_share": 0.70,
                "target_share": 0.14,
                "preseason_ecr": 18.0,
            },
            {
                "player_id": "rb_test_2",
                "player_name": "RB Test 2",
                "position": "RB",
                "projection_season": 2027,
                "birth_date": "2003-01-01",
                "years_exp": 1,
                "ppg": 10.7,
                "games_played": 11,
                "snap_share": 0.26,
                "carries_pg": 5.4,
                "targets_pg": 2.1,
                "receptions_pg": 1.6,
                "rushing_yards_pg": 31.0,
                "receiving_yards_pg": 13.0,
                "rush_share": 0.24,
                "target_share": 0.07,
                "preseason_ecr": 170.0,
            },
        ]
    )

    pred = projector.predict(players, "RB")
    details = projector.predict_with_details(players, "RB")

    assert pred.shape == (2,)
    assert np.all(pred >= 0.0)
    assert list(details.columns) == [
        "base_pred",
        "pred",
        "market_anchor",
        "confidence_score",
        "support_class",
    ]
    assert details.loc[players.index[0], "market_anchor"] > details.loc[players.index[1], "market_anchor"]


def test_save_and_load_round_trip_new_schema(tmp_path):
    projector = PreseasonProjector().fit(_training_pairs())
    model_path = tmp_path / "preseason_projector.json"
    players = _training_pairs().query("position == 'WR'").head(6).copy()

    before = projector.predict(players, "WR")
    projector.save(model_path)
    raw = json.loads(model_path.read_text())
    after = PreseasonProjector.load(model_path).predict(players, "WR")

    assert raw["schema_version"] == 2
    assert "base_outcome_model" in raw["positions"]["WR"]
    assert "upstream_calibrator" in raw["positions"]["WR"]
    assert np.allclose(before, after)


def test_load_legacy_schema_remains_supported(tmp_path):
    legacy = {
        "positions": {
            "RB": {
                "features": ["ppg", "games_played"],
                "coef": [10.0, 2.0],
                "intercept": 5.0,
                "scaler_mean": [0.0, 0.0],
                "scaler_scale": [1.0, 1.0],
            }
        },
        "veteran_elite_calibration": {
            "RB": {
                "position": "RB",
                "factor": 1.1,
                "age_threshold": 29.0,
                "elite_ppg_threshold": 15.0,
                "sample_size": 14,
                "mean_error_before": -8.0,
                "mean_error_after": -2.0,
                "mae_before": 18.0,
                "mae_after": 12.0,
                "median_actual_to_pred_ratio": 1.1,
            }
        },
        "fragile_role_calibration": {},
        "bias_audit": {},
        "fragile_role_audit": {},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))

    projector = PreseasonProjector.load(path)
    players = pd.DataFrame(
        [
            {
                "position": "RB",
                "projection_season": 2026,
                "birth_date": "1994-01-01",
                "ppg": 16.0,
                "games_played": 16,
            },
            {
                "position": "RB",
                "projection_season": 2026,
                "birth_date": "2001-01-01",
                "ppg": 16.0,
                "games_played": 16,
            },
        ]
    )

    pred = projector.predict(players, "RB")

    assert projector.variant_name == "legacy_ridge_with_cohort_patches"
    assert pred[0] > pred[1]


def test_upstream_audit_contains_market_alignment_fields():
    projector = PreseasonProjector().fit(_training_pairs())
    audit = projector.get_upstream_audit_report()

    assert "overall" in audit
    assert "pred_market_mae" in audit["overall"]
    assert "pred_large_divergence_share" in audit["overall"]
    assert "pred_rb_wr_gap_excess" in audit["overall"]
    assert "cohort_market_error" in audit

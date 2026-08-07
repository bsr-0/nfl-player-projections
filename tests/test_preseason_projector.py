from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.preseason_projector import (
    BASE_FEATURES_BY_POSITION,
    PreseasonProjector,
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
    # confidence_score feeds generate_draft_data.py's floor/ceiling sizing
    # (kept even though the base Ridge model doesn't consume it directly).
    assert 0.05 <= out.loc[0, "confidence_score"] <= 1.0


def test_fit_learns_position_specific_models():
    projector = PreseasonProjector().fit(_training_pairs())

    assert set(projector.models) == {"QB", "RB", "WR", "TE"}
    for pos in ("QB", "RB", "WR", "TE"):
        assert set(projector.feature_names[pos]).issubset(BASE_FEATURES_BY_POSITION[pos])
    assert "overall" in projector.audit_report
    assert "mae" in projector.audit_report["overall"]


def test_predict_keeps_public_contract():
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
            },
        ]
    )

    pred = projector.predict(players, "RB")
    details = projector.predict_with_details(players, "RB")

    assert pred.shape == (2,)
    assert np.all(pred >= 0.0)
    assert list(details.columns) == ["pred", "confidence_score", "support_class"]
    assert details.loc[players.index[0], "pred"] > details.loc[players.index[1], "pred"]


def test_save_and_load_round_trip(tmp_path):
    projector = PreseasonProjector().fit(_training_pairs())
    model_path = tmp_path / "preseason_projector.json"
    players = _training_pairs().query("position == 'WR'").head(6).copy()

    before = projector.predict(players, "WR")
    projector.save(model_path)
    raw = json.loads(model_path.read_text())
    after = PreseasonProjector.load(model_path).predict(players, "WR")

    assert raw["schema_version"] == 3
    assert "coef" in raw["positions"]["WR"]
    assert np.allclose(before, after)


def test_load_legacy_schema_remains_supported(tmp_path):
    """Old (pre-2026-08-07) artifacts nested the base model under
    positions[pos]["base_outcome_model"] and carried now-removed
    calibration keys -- load() should still read the base model and
    silently ignore the calibration keys."""
    legacy = {
        "positions": {
            "RB": {
                "base_outcome_model": {
                    "features": ["ppg", "games_played"],
                    "coef": [10.0, 2.0],
                    "intercept": 5.0,
                    "scaler_mean": [0.0, 0.0],
                    "scaler_scale": [1.0, 1.0],
                },
                "upstream_calibrator": {"position": "RB"},
            }
        },
        "legacy_veteran_elite_calibration": {},
        "legacy_fragile_role_calibration": {},
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
                "ppg": 18.0,
                "games_played": 16,
            },
            {
                "position": "RB",
                "projection_season": 2026,
                "birth_date": "2001-01-01",
                "ppg": 12.0,
                "games_played": 16,
            },
        ]
    )

    pred = projector.predict(players, "RB")

    assert set(projector.models) == {"RB"}
    assert pred[0] > pred[1]


def test_audit_report_contains_outcome_fields():
    projector = PreseasonProjector().fit(_training_pairs())
    audit = projector.audit_report

    assert "overall" in audit
    assert "mae" in audit["overall"]
    assert "bias" in audit["overall"]
    assert "by_position" in audit

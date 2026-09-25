import sqlite3

import numpy as np
import pandas as pd
import pytest

from scripts.build_team_week_player_shares import build_shares
from src.evaluation.joint_ppr_selector import PPR_RECONSTRUCTION_TARGETS, _ppr_for_choices
from src.evaluation.full_ppr_reconstruction import evaluate_reconstruction
from src.evaluation.ppr_truth import IDENTITY, KEY, checked_truth, validate_inputs
from src.utils.helpers import calculate_fantasy_points_df


def _panel():
    con = sqlite3.connect(":memory:")
    players = [("q", "QB"), ("r", "RB"), ("w", "WR")]
    pd.DataFrame([{"player_id": pid, "position": pos, "team": "AAA", "season": 2025, "week": 1}
                  for pid, pos in players]).to_sql("canonical_player_weeks", con, index=False)
    pd.DataFrame([
        {"player_id": "q", "season": 2025, "week": 1, "rushing_yards": 0, "receiving_yards": 4,
         "receptions": 1, "receiving_tds": 0, "rushing_tds": 0, "passing_yards": 100,
         "passing_tds": 1, "interceptions": 0},
        {"player_id": "r", "season": 2025, "week": 1, "rushing_yards": -5, "receiving_yards": 0,
         "receptions": 0, "receiving_tds": 0, "rushing_tds": 0, "passing_yards": 0,
         "passing_tds": 0, "interceptions": 0},
        {"player_id": "w", "season": 2025, "week": 1, "rushing_yards": 0, "receiving_yards": 20,
         "receptions": 2, "receiving_tds": 0, "rushing_tds": 0, "passing_yards": 10,
         "passing_tds": 0, "interceptions": 0},
    ]).to_sql("player_weekly_stats", con, index=False)
    panel = build_shares(con, 2025, 2025)
    con.close()
    return panel


def test_raw_truth_keeps_negative_yards_and_off_position_stats():
    panel = _panel()
    keys = panel[IDENTITY].assign(fold=0)
    truth = checked_truth(panel, keys)
    expected = calculate_fantasy_points_df(panel[PPR_RECONSTRUCTION_TARGETS])
    joined = truth.merge(panel[IDENTITY].assign(expected=expected), on=IDENTITY, validate="one_to_one")
    assert np.allclose(joined.actual_ppr, joined.expected)
    assert truth.set_index("player_id").loc["r", "actual_ppr"] == -0.5
    assert truth.set_index("player_id").loc["q", "off_position_stat"]
    assert truth.set_index("player_id").loc["w", "off_position_stat"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nan", "inf"])
def test_truth_fails_closed(mutation):
    raw = _panel()
    keys = raw[IDENTITY].assign(fold=0)
    if mutation == "missing":
        raw = raw.iloc[1:]
    elif mutation == "duplicate":
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    else:
        raw.loc[0, "rushing_yards"] = np.nan if mutation == "nan" else np.inf
    with pytest.raises(ValueError):
        checked_truth(raw, keys)


def test_scorer_rejects_missing_expected_prediction():
    raw = _panel()
    truth = checked_truth(raw, raw[IDENTITY].assign(fold=0))
    allocation, totals = {}, {}
    for target in PPR_RECONSTRUCTION_TARGETS:
        if target in {"receiving_yards", "receptions", "receiving_tds"}:
            pop = raw[raw.position.ne("QB")]
        elif target in {"passing_yards", "passing_tds", "interceptions"}:
            pop = raw[raw.position.eq("QB")]
        else:
            pop = raw
        rows = pop[IDENTITY].assign(fold=0, arm="rolling3", actual_share=0.0, predicted_share=0.0)
        if target == "rushing_yards":
            rows = rows.iloc[1:]
        allocation[target] = {"predictions": rows}
        totals[target] = {"predictions": pd.DataFrame({"fold": [0], "team": ["AAA"],
            "season": [2025], "week": [1], "actual": [1.0], "rolling3": [1.0]})}
    arms = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, "rolling3")
    with pytest.raises(ValueError, match="missing expected rushing_yards"):
        _ppr_for_choices(allocation, totals, arms, arms, {0}, truth)


def test_legacy_evaluator_uses_all_eight_raw_components():
    raw = _panel()
    allocation, totals = {}, {}
    for target in PPR_RECONSTRUCTION_TARGETS:
        if target in {"receiving_yards", "receptions", "receiving_tds"}:
            pop = raw[raw.position.ne("QB")]
        elif target in {"passing_yards", "passing_tds", "interceptions"}:
            pop = raw[raw.position.eq("QB")]
        else:
            pop = raw
        template = pop[IDENTITY].assign(fold=0, actual_share=pop[f"share_of_team_{target}"].to_numpy(),
                                         predicted_share=0.0)
        allocation[target] = {"predictions": pd.concat(
            [template.assign(arm=arm) for arm in ("rolling3", "rolling3_renorm", "selected")],
            ignore_index=True)}
        totals[target] = {"predictions": pd.DataFrame({
            "fold": [0], "team": ["AAA"], "season": [2025], "week": [1],
            "actual": [raw[f"team_{target}"].iloc[0]],
            "rolling3": [1.0], "ridge": [1.0], "xgb": [1.0], "blend": [1.0],
        })}
    arms = dict.fromkeys(PPR_RECONSTRUCTION_TARGETS, "selected")
    report = evaluate_reconstruction(allocation, totals, arms=arms, raw_rows=raw)
    assert report["n"] == 3
    assert report["candidate_mae"] == pytest.approx(
        calculate_fantasy_points_df(raw[PPR_RECONSTRUCTION_TARGETS]).abs().mean()
    )


def test_unexpected_rolling_team_total_null_fails_preflight():
    raw = _panel().copy()
    raw["week"] = 2
    allocation, totals = {}, {}
    for target in PPR_RECONSTRUCTION_TARGETS:
        if target in {"receiving_yards", "receptions", "receiving_tds"}:
            pop = raw[raw.position.ne("QB")]
        elif target in {"passing_yards", "passing_tds", "interceptions"}:
            pop = raw[raw.position.eq("QB")]
        else:
            pop = raw
        allocation[target] = {"predictions": pop[IDENTITY].assign(
            fold=0, arm="rolling3", actual_share=pop[f"share_of_team_{target}"].to_numpy(),
            predicted_share=0.0)}
        totals[target] = {"predictions": pd.DataFrame({
            "fold": [0], "team": ["AAA"], "season": [2025], "week": [2],
            "actual": [raw[f"team_{target}"].iloc[0]],
            "rolling3": [np.nan], "ridge": [1.0], "xgb": [1.0], "blend": [1.0],
        })}
    with pytest.raises(ValueError, match="unexpected rolling3 team-total null"):
        validate_inputs(allocation, totals, raw)

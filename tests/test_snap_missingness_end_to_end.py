"""End-to-end: an unknown snap row must reach the model as an imputed value.

Traces real rows through every stage that can destroy missingness:

    snap_count/team_snaps -> snap_share_pct -> calculate_all_scores' blanket
    fill -> roll3 mean -> roll3 known -> persisted imputation -> the
    backtester's X.fillna(0) -> ComponentPredictor._prepare_array

Unit tests of each stage all passed while the pipeline as a whole still
zeroed the value: calculate_all_scores fills NaN before feature engineering
builds the rolling feature, so the missingness was gone before the
imputation could act. Only a trace across stages catches that.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.utilization_score import (
    SNAP_KNOWN_COL,
    SNAP_ROLL3_COL,
    UtilizationScoreCalculator,
    apply_snap_imputation,
    fit_snap_imputation,
    load_snap_imputation,
    save_snap_imputation,
)
from src.features.feature_engineering import FeatureEngineer


KNOWN, UNKNOWN = "known_player", "unknown_player"


def _raw_frame():
    """Two WRs over four weeks. One always measured; one never measured."""
    rows = []
    for week in range(1, 5):
        rows.append({
            "player_id": KNOWN, "season": 2016, "week": week, "position": "WR",
            "snap_count": 45.0, "team_snaps": 60.0,
            "targets": 6, "receptions": 4, "receiving_yards": 55,
            "receiving_tds": 0, "rushing_attempts": 0, "rushing_yards": 0,
            "rushing_tds": 0,
        })
        rows.append({
            "player_id": UNKNOWN, "season": 2016, "week": week, "position": "WR",
            "snap_count": np.nan, "team_snaps": np.nan,
            "targets": 3, "receptions": 2, "receiving_yards": 25,
            "receiving_tds": 0, "rushing_attempts": 0, "rushing_yards": 0,
            "rushing_tds": 0,
        })
    return pd.DataFrame(rows).sort_values(["player_id", "week"]).reset_index(drop=True)


def _through_utilization(df):
    return UtilizationScoreCalculator().calculate_all_scores(df.copy(), pd.DataFrame())


def _through_feature_engineering(df):
    """The two derived columns, built exactly as the roll loop builds them."""
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    df[SNAP_ROLL3_COL] = (
        df.groupby("player_id")["snap_share_pct"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    return FeatureEngineer.__new__(FeatureEngineer)._add_snap_roll3_known(df, 3)


def test_missingness_survives_the_utilization_blanket_fill():
    """The stage that silently defeated the whole change."""
    out = _through_utilization(_raw_frame())
    unknown = out[out.player_id == UNKNOWN]

    assert unknown["snap_share_pct"].isna().all(), (
        "snap_share_pct was zeroed before the rolling feature could record "
        "that it was unknown"
    )
    assert (out[out.player_id == KNOWN]["snap_share_pct"] > 0).all()


def test_known_indicator_reflects_reality_after_the_full_chain():
    df = _through_feature_engineering(_through_utilization(_raw_frame()))

    known_rows = df[(df.player_id == KNOWN) & (df.week > 1)]
    unknown_rows = df[(df.player_id == UNKNOWN) & (df.week > 1)]

    assert (known_rows[SNAP_KNOWN_COL] == 1.0).all()
    assert (unknown_rows[SNAP_KNOWN_COL] == 0.0).all(), (
        "an unmeasured player must not read as fully known"
    )


def test_unknown_row_reaches_the_model_as_the_imputed_value(tmp_path):
    """The invariant: the final model input for an unknown row is the
    persisted median, NOT the old fabricated zero."""
    df = _through_feature_engineering(_through_utilization(_raw_frame()))

    # production: fit on train rows, persist, reload, apply
    train = df[df.player_id == KNOWN]
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path,
                         metadata={"train_seasons": [2016]})
    imputed = apply_snap_imputation(df, load_snap_imputation(path))

    unknown = imputed[imputed.player_id == UNKNOWN]
    expected = train[SNAP_ROLL3_COL].median()

    assert unknown[SNAP_ROLL3_COL].notna().all()
    assert unknown[SNAP_ROLL3_COL].iloc[-1] == pytest.approx(expected)
    assert unknown[SNAP_ROLL3_COL].iloc[-1] != 0.0, "reverted to fabricated zero"


def test_value_survives_the_downstream_fillna_and_prepare_array(tmp_path):
    """Two more stages that convert NaN to 0. Because the value is already
    imputed, both must be no-ops for it."""
    from src.models.component_predictor import ComponentPredictor

    df = _through_feature_engineering(_through_utilization(_raw_frame()))
    train = df[df.player_id == KNOWN]
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(train), path,
                         metadata={"train_seasons": [2016]})
    df = apply_snap_imputation(df, load_snap_imputation(path))

    cols = [SNAP_ROLL3_COL, SNAP_KNOWN_COL]
    X = df[cols].fillna(0)                       # ts_backtester stage
    prepared = ComponentPredictor("WR")._prepare_array(X.to_numpy(dtype=float))

    unknown_idx = df.index[df.player_id == UNKNOWN]
    final = prepared[[df.index.get_loc(i) for i in unknown_idx], 0]

    assert (final != 0.0).all(), "a downstream fill undid the imputation"
    assert final == pytest.approx(train[SNAP_ROLL3_COL].median())


def test_the_old_behaviour_would_have_produced_zero():
    """Guards the comparison itself: under mode 'zero' the unknown row's
    feature really is 0.0, so the assertions above are meaningful."""
    from src.features import utilization_score as us

    original = us.SNAP_MISSINGNESS_MODE
    try:
        us.set_snap_missingness_mode("zero")
        df = _through_feature_engineering(_through_utilization(_raw_frame()))
    finally:
        us.set_snap_missingness_mode(original)

    unknown = df[(df.player_id == UNKNOWN) & (df.week > 1)]
    assert (unknown[SNAP_ROLL3_COL] == 0.0).all()
    assert (unknown[SNAP_KNOWN_COL] == 1.0).all()   # indistinguishable from real


# --- the real pipeline, not composed stages -----------------------------

def test_real_create_features_preserves_snap_missingness():
    """Runs the ACTUAL FeatureEngineer.create_features.

    The composed-stage tests above pass A -> B -> C -> D. Production runs
    A -> B -> C -> X -> D, and twice now an X silently destroyed the
    missingness while every component test stayed green:
    _create_base_features recomputed snap_share_pct through safe_divide, and
    _impute_missing filled the rolling column with a frame-wide median. Only
    exercising the real path catches that class of bug.
    """
    from src.features.utilization_score import UtilizationScoreCalculator

    raw = _raw_frame()
    # widen to enough history for rolling windows
    extra = []
    for wk in range(5, 9):
        for pid, snaps, team in ((KNOWN, 45.0, 60.0), (UNKNOWN, np.nan, np.nan)):
            extra.append({
                "player_id": pid, "season": 2016, "week": wk, "position": "WR",
                "snap_count": snaps, "team_snaps": team,
                "targets": 5, "receptions": 3, "receiving_yards": 40,
                "receiving_tds": 0, "rushing_attempts": 0, "rushing_yards": 0,
                "rushing_tds": 0,
            })
    raw = pd.concat([raw, pd.DataFrame(extra)], ignore_index=True)

    scored = UtilizationScoreCalculator().calculate_all_scores(raw, pd.DataFrame())
    assert scored["snap_share_pct"].isna().any(), (
        "utilization stage already destroyed the missingness"
    )

    built = FeatureEngineer().create_features(scored, include_target=False)

    unknown = built[built.player_id == UNKNOWN]
    assert built[SNAP_ROLL3_COL].isna().any(), (
        "create_features filled the rolling snap column; the persisted "
        "imputation step no longer owns it"
    )
    assert (unknown[SNAP_KNOWN_COL] < 1.0).all(), (
        "an unmeasured player reads as fully known after the real pipeline"
    )


def test_real_pipeline_then_artifact_yields_the_persisted_value(tmp_path):
    """End of the chain: real feature construction, then the artifact."""
    from src.features.utilization_score import UtilizationScoreCalculator

    raw = _raw_frame()
    scored = UtilizationScoreCalculator().calculate_all_scores(raw, pd.DataFrame())
    built = FeatureEngineer().create_features(scored, include_target=False)

    known_rows = built[built[SNAP_ROLL3_COL].notna()]
    path = tmp_path / "snap_imputation.json"
    save_snap_imputation(fit_snap_imputation(known_rows), path,
                         metadata={"train_seasons": [2016]})
    values = load_snap_imputation(path)
    final = apply_snap_imputation(built, values)

    was_missing = built[SNAP_ROLL3_COL].isna()
    if was_missing.any():
        filled = final.loc[was_missing, SNAP_ROLL3_COL]
        assert filled.notna().all()
        assert (filled != 0.0).all(), "fell back to the fabricated zero"

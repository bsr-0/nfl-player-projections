"""The weekly serving path must run, and must return football-shaped numbers.

Two real defects reached main on 2026-08-30 with the full suite green, because
nothing exercised `NFLPredictor.predict_next_week()` end to end.

1. A regex deletion of an unrelated method swallowed the `@staticmethod`
   decorator on the method that followed it, so `_apply_snap_imputation` became
   an instance method and every prediction raised TypeError. The serving path
   was completely dead and 550 tests still passed.

2. Older and worse: `predict()` loaded ONE POSITION at a time and then computed
   team-relative features inside that frame, so every share denominator was
   position-local. With only RBs loaded, "share of team targets going to RBs"
   is 100% by construction:

       team_rb_target_share_roll3_mean   serve 100.000   train 18.395
       target_share_pct_roll3_mean       serve  25.417   train  5.829

   Fed to a model trained on real shares, that produced a median RB weekly
   projection of 42.3 points against an actual median of 5.5 -- every weekly
   prediction the system served was roughly 8x too high, invisibly.

These tests are deliberately loose. They are not accuracy checks; they are
"is the serving path alive and is it producing points rather than nonsense"
checks, which is exactly the gap both defects went through.
"""
import pytest

import pandas as pd

from config.settings import DB_PATH, MODELS_DIR


pytestmark = pytest.mark.skipif(
    not DB_PATH.exists() or not (MODELS_DIR / "feature_version.txt").exists(),
    reason="needs the local database and trained models",
)


# Real 2018-2025 weekly PPR medians are roughly QB 14.9, RB 5.5, WR 5.1,
# TE 2.6. The ceilings below sit far above those and far below the values the
# share-inflation bug produced (RB median 42.3), so they catch that class of
# failure without being sensitive to ordinary model drift.
MEDIAN_CEILING = {"QB": 30.0, "RB": 20.0, "WR": 20.0, "TE": 15.0}
MAX_CEILING = {"QB": 60.0, "RB": 45.0, "WR": 45.0, "TE": 40.0}


@pytest.fixture(scope="module")
def predictions():
    """One real prediction run, reused across tests (it is expensive)."""
    from src.predict import NFLPredictor

    p = NFLPredictor()
    if not p.initialize():
        pytest.skip("no trained models available")
    out = {}
    for pos in ("QB", "RB", "WR", "TE"):
        df = p.predict(n_weeks=1, position=pos, top_n=200)
        if not df.empty:
            out[pos] = df
    if not out:
        pytest.skip("serving path returned nothing for every position")
    return out


def test_serving_path_runs_at_all(predictions):
    """Regression for the swallowed decorator: this raised TypeError."""
    assert predictions, "predict() produced no output for any position"
    for pos, df in predictions.items():
        assert not df.empty, f"{pos}: empty prediction frame"
        assert "predicted_points" in df.columns, f"{pos}: no predicted_points"


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_predictions_are_on_the_fantasy_point_scale(predictions, pos):
    """Regression for the share-inflation bug (RB median was 42.3)."""
    if pos not in predictions:
        pytest.skip(f"no {pos} predictions")
    pts = pd.to_numeric(predictions[pos]["predicted_points"], errors="coerce").dropna()
    assert len(pts), f"{pos}: all predictions NaN"

    med = float(pts.median())
    assert 0.0 <= med <= MEDIAN_CEILING[pos], (
        f"{pos}: median weekly projection {med:.1f} is outside the plausible "
        f"band (0, {MEDIAN_CEILING[pos]}]. A median far above this means the "
        f"model is being fed features on a different scale than it trained on."
    )
    assert float(pts.max()) <= MAX_CEILING[pos], (
        f"{pos}: max weekly projection {pts.max():.1f} exceeds anything seen "
        f"in a real NFL week"
    )
    assert float(pts.min()) >= 0.0, f"{pos}: negative points projected"


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_predictions_discriminate_between_players(predictions, pos):
    """A constant output is the other signature of broken serving features.

    When the share denominators collapsed, the top ten RBs came back within
    0.58 points of each other.
    """
    if pos not in predictions:
        pytest.skip(f"no {pos} predictions")
    pts = pd.to_numeric(predictions[pos]["predicted_points"], errors="coerce").dropna()
    assert pts.std() > 0.5, (
        f"{pos}: predictions are nearly constant (std={pts.std():.3f}); the "
        f"model is not discriminating between players"
    )
    assert pts.nunique() > 10, f"{pos}: only {pts.nunique()} distinct values"


def test_team_share_features_are_not_position_local(predictions):
    """The specific mechanism, asserted directly.

    Loading one position at a time made every team-share denominator
    position-local. `team_rb_target_share` reached exactly 100% because the
    only players in the frame were running backs.
    """
    from src.predict import NFLPredictor

    p = NFLPredictor()
    if not p.initialize():
        pytest.skip("no trained models available")
    frame = p._prepare_features(p._load_player_data(None, min_games=1))
    latest = frame.groupby("player_id").last().reset_index()

    for col in ("team_rb_target_share_roll3_mean", "target_share_pct_roll3_mean"):
        if col not in latest.columns:
            continue
        med = pd.to_numeric(latest[col], errors="coerce").median()
        if pd.isna(med):
            continue
        assert med < 50.0, (
            f"{col} median is {med:.1f}; a team share near 100 means the frame "
            f"was filtered to one position before the share was computed"
        )

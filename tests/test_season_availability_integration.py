"""Integration contract for the season availability layer.

Seven requirements, one test each (or more), fixed by the directive that
authorised this layer:

  1. weekly predictions are unchanged by the availability layer
  2. season games use hist_shrunk
  3. season PPR = games x PPR/game
  4. the estimator is strictly causal
  5. players with no prior history have an explicit, documented fallback
  6. the evaluation reproduces the experiment's results
  7. no synthetic zero weeks are reintroduced
"""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.single_week_ppr.season_availability import (
    SHRINKAGE_K, SeasonAvailabilityEstimator, load_player_seasons, project_season_ppr,
)

REPO = Path(__file__).resolve().parent.parent


def _history(seasons=(2020, 2021, 2022)):
    rows = []
    for s in seasons:
        rows += [
            {"player_id": "durable", "position": "RB", "season": s,
             "games_played": 17, "possible_games": 17},
            {"player_id": "fragile", "position": "RB", "season": s,
             "games_played": 4, "possible_games": 17},
        ]
    return pd.DataFrame(rows)


def _targets():
    return pd.DataFrame({
        "player_id": ["durable", "fragile", "rookie"],
        "position": ["RB", "RB", "RB"],
        "possible_games": [17.0, 17.0, 17.0],
    })


# --- 1. weekly predictions are unchanged ----------------------------------

def test_weekly_path_does_not_import_the_availability_layer():
    """The weekly model must not consult availability at all. Enforced
    structurally so it cannot drift back in."""
    weekly_modules = [
        REPO / "src" / "models" / "single_week_ppr" / "evaluate.py",
        REPO / "src" / "models" / "single_week_ppr" / "architectures.py",
        REPO / "src" / "models" / "single_week_ppr" / "population.py",
        REPO / "src" / "models" / "single_week_ppr" / "opportunity.py",
    ]
    for path in weekly_modules:
        if not path.exists():
            continue
        source = path.read_text()
        assert "season_availability" not in source, (
            f"{path.name} references the season availability layer; weekly "
            "predictions must be unaffected by it")


def test_availability_layer_never_touches_a_weekly_prediction():
    """Its whole public surface operates on player-SEASON rows, never on
    weekly rows, so there is no path by which it can alter a weekly number."""
    for name, fn in [("predict_rate", SeasonAvailabilityEstimator.predict_rate),
                     ("predict_games", SeasonAvailabilityEstimator.predict_games),
                     ("project_season_ppr", project_season_ppr)]:
        params = set(inspect.signature(fn).parameters)
        assert "week" not in params, f"{name} takes a week argument"


# --- 2. season games use hist_shrunk --------------------------------------

def test_games_estimate_is_shrunk_toward_the_position_mean():
    est = SeasonAvailabilityEstimator().fit(_history(), before_season=2023)
    rates = est.predict_rate(_targets())
    position_mean = (17 + 4) / 2 / 17
    # Durable pulled DOWN toward the mean, fragile pulled UP toward it.
    assert rates.iloc[0] < 1.0
    assert rates.iloc[1] > 4 / 17
    assert rates.iloc[0] > position_mean > rates.iloc[1]


def test_shrinkage_strength_follows_games_observed():
    """One season of history earns less weight than five."""
    short = SeasonAvailabilityEstimator().fit(_history(seasons=(2022,)), before_season=2023)
    long = SeasonAvailabilityEstimator().fit(
        _history(seasons=(2018, 2019, 2020, 2021, 2022)), before_season=2023)
    t = _targets().iloc[[0]]
    assert long.predict_rate(t).iloc[0] > short.predict_rate(t).iloc[0]


def test_raw_per_player_rate_is_not_used():
    """hist_player over-corrects (WR season bias +12.36 vs +7.85); the
    adopted estimator must not reproduce it."""
    est = SeasonAvailabilityEstimator().fit(_history(), before_season=2023)
    assert est.predict_rate(_targets()).iloc[0] != pytest.approx(1.0)


# --- 3. season PPR = games x PPR/game -------------------------------------

def test_season_ppr_is_exactly_the_product():
    games = pd.Series([15.0, 8.0])
    rate = pd.Series([12.0, 6.0])
    assert project_season_ppr(games, rate).tolist() == [180.0, 48.0]


def test_no_separate_bias_correction_term_exists():
    """The bias gain comes from the product itself; a second correction
    would double-count it."""
    src = (REPO / "src" / "models" / "single_week_ppr" / "season_availability.py").read_text()
    for banned in ("bias_correction", "bias_offset", "calibration_factor"):
        assert banned not in src


# --- 4. strictly causal ---------------------------------------------------

def test_fit_drops_the_target_season_and_later():
    hist = _history(seasons=(2021, 2022, 2023, 2024))
    est = SeasonAvailabilityEstimator().fit(hist, before_season=2023)
    # Only 2021-2022 may contribute: 2 seasons x 17 possible.
    row = est.player_history_.set_index("player_id").loc["durable"]
    assert row["prior_possible"] == 34


def test_fit_refuses_when_no_prior_seasons_exist():
    with pytest.raises(ValueError, match="no seasons before"):
        SeasonAvailabilityEstimator().fit(_history(seasons=(2023,)), before_season=2023)


def test_fit_requires_the_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        SeasonAvailabilityEstimator().fit(
            pd.DataFrame({"player_id": ["a"], "season": [2020]}), before_season=2023)


# --- 5. no-history fallback is explicit -----------------------------------

def test_unknown_player_falls_back_to_the_position_mean():
    est = SeasonAvailabilityEstimator().fit(_history(), before_season=2023)
    rate = est.predict_rate(_targets()).iloc[2]
    assert rate == pytest.approx(est.position_rate_["RB"])


def test_has_history_flags_the_fallback_rows():
    est = SeasonAvailabilityEstimator().fit(_history(), before_season=2023)
    assert est.has_history(_targets()).tolist() == [True, True, False]


def test_fallback_is_not_a_claim_about_playing_at_all():
    """The boundary must stay documented: this estimator was never
    evaluated on players with zero games."""
    import importlib
    doc = importlib.import_module(SeasonAvailabilityEstimator.__module__).__doc__
    normalised = " ".join(doc.lower().split())
    assert "will this player play at all" in normalised
    assert "zero games" in normalised


def test_rates_stay_in_bounds():
    hist = _history()
    hist.loc[len(hist)] = {"player_id": "odd", "position": "RB", "season": 2022,
                           "games_played": 25, "possible_games": 17}
    est = SeasonAvailabilityEstimator().fit(hist, before_season=2023)
    t = pd.DataFrame({"player_id": ["odd"], "position": ["RB"], "possible_games": [17.0]})
    assert 0.0 <= est.predict_rate(t).iloc[0] <= 1.0


# --- 6. reproduces the experiment -----------------------------------------

@pytest.mark.parametrize("season", [2024, 2025])
def test_beats_the_constant_baseline_on_real_data(season):
    """The headline result: shrunk history beats a constant on games played.
    Reproduced through the production estimator, not the experiment script."""
    panel = load_player_seasons()
    target = panel[panel.season == season]
    if target.empty:
        pytest.skip("panel has no rows for this season")
    est = SeasonAvailabilityEstimator().fit(panel, before_season=season)
    pred = est.predict_games(target)
    const_rate = target["position"].map(est.position_rate_)
    const = const_rate * target["possible_games"]

    shrunk_mae = float((pred - target["games_played"]).abs().mean())
    const_mae = float((const - target["games_played"]).abs().mean())
    assert shrunk_mae < const_mae, (
        f"{season}: shrunk {shrunk_mae:.3f} did not beat constant {const_mae:.3f}")


# --- 7. no synthetic zero weeks -------------------------------------------

def test_panel_contains_no_fabricated_zero_games():
    """Every player-season must come from observed participation. A
    manufactured row would show up as games_played == 0."""
    panel = load_player_seasons()
    assert (panel["games_played"] > 0).all()
    assert (panel["possible_games"] > 0).all()


def test_module_does_not_construct_synthetic_rows():
    src = (REPO / "src" / "models" / "single_week_ppr" / "season_availability.py").read_text()
    for banned in ("synthetic", "reindex", "fill_missing_weeks"):
        assert banned not in src.replace("synthetic-week architecture", "")

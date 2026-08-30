"""The evaluation harness must train the same pipeline production trains.

`run_fold` is what every walk-forward comparison in this project measures. It
called `_prepare_training_data` WITHOUT `context_data` while `train_models()`
passed it, so from 2026-08-30 the two diverged: lookback features (rolling
windows, 3-year availability, prior-season stats) started cold at the training
window's first season in the harness and did not in production.

A harness that silently differs from production is the same class of defect as
the fabricated values this audit was chasing -- it does not error, it just
measures something other than what ships.

The second test is the one that matters more. Context exists to warm up
lookback windows with OLDER seasons; if it ever contained the held-out season,
the fold would be training on its own test data.
"""
import pytest

from config.settings import DB_PATH


pytestmark = pytest.mark.skipif(
    not DB_PATH.exists(), reason="needs the local database")


@pytest.fixture
def captured(monkeypatch):
    """Intercept _prepare_training_data and record how it was called."""
    import src.models.feature_preparation as fp

    seen = {}

    def _spy(train_data, test_data, positions, tune, n_trials,
             fast=False, context_data=None):
        seen["context_data"] = context_data
        seen["train_seasons"] = sorted(train_data["season"].unique())
        raise _Stop()

    class _Stop(Exception):
        pass

    seen["Stop"] = _Stop
    monkeypatch.setattr(fp, "_prepare_training_data", _spy)
    return seen


def _run(captured, test_season):
    from src.models.single_week_ppr.evaluate import run_fold
    try:
        run_fold("RB", test_season)
    except captured["Stop"]:
        pass
    except Exception as e:  # pragma: no cover - surfaces real breakage
        pytest.fail(f"run_fold raised before reaching the spy: {e!r}")
    return captured


def test_run_fold_passes_context_data(captured):
    seen = _run(captured, 2024)
    ctx = seen.get("context_data")

    assert ctx is not None, (
        "run_fold called _prepare_training_data without context_data; the "
        "harness is measuring a different pipeline from the one that ships"
    )
    assert not ctx.empty, "context_data is empty; lookback windows stay cold"


def test_context_never_contains_the_held_out_season(captured):
    test_season = 2024
    seen = _run(captured, test_season)
    ctx = seen.get("context_data")
    if ctx is None or ctx.empty:
        pytest.skip("no context to check")

    seasons = set(int(s) for s in ctx["season"].dropna().unique())
    assert test_season not in seasons, (
        f"context carries the held-out season {test_season}: the fold would "
        f"warm up its lookback features on its own test data"
    )
    assert max(seasons) < min(seen["train_seasons"]), (
        "context must lie strictly before the training window"
    )

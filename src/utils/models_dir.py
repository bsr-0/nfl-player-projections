"""Temporarily redirect MODELS_DIR everywhere it is bound.

Three call sites (walk-forward validation in `src/models/train.py`, the
LOYO backtest in `src/evaluation/backtester.py`, and
`src/models/single_week_ppr/evaluate.py`) all tried to keep throwaway
validation folds from overwriting real model artifacts by doing:

    settings.MODELS_DIR = Path(tmp)

That never worked. Nineteen modules bind the directory with
``from config.settings import MODELS_DIR``, which copies the *value* at
import time; rebinding the attribute on the ``config.settings`` module
leaves every one of those bindings still pointing at the real directory.
`src/models/position_models.py` is the consequential one -- its save path
(``MODELS_DIR / f"model_{position}_{n_weeks}w.joblib"`` and the multiweek
equivalent) writes exactly the artifacts `EnsemblePredictor.load_models()`
serves, so each walk-forward fold silently replaced the production models
with fold models. Observed 2026-09-24: a walk-forward run overwrote all
four positions' served artifacts between 04:55 and 07:09.

GAPS.md documents this same import-by-value defect being found and fixed
once before, for `single_week_ppr`'s Phase 2 code, via a snapshot/restore
guard. That approach does not transfer here -- ``*.joblib`` is gitignored,
so there is no tracked copy to restore from -- so this fixes the binding
itself instead.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import config.settings as settings

_OWN_PACKAGES = ("src.", "config.", "scripts.")


def _bound_modules() -> List[object]:
    """Modules holding their own MODELS_DIR binding from a `from` import."""
    found = []
    for module in list(sys.modules.values()):
        if module is None or module is settings:
            continue
        name = getattr(module, "__name__", "")
        if not name.startswith(_OWN_PACKAGES):
            continue
        if isinstance(getattr(module, "MODELS_DIR", None), Path):
            found.append(module)
    return found


class ProductionArtifactWriteError(RuntimeError):
    """A run that is not a full production retrain would rewrite data/models."""


def is_production_models_dir(path: Path | str) -> bool:
    return Path(path).resolve() == Path(settings.PRODUCTION_MODELS_DIR).resolve()


def assert_safe_models_dir_write(
    models_dir: Path | str,
    positions: Iterable[str],
    *,
    fit_models: bool,
    production_run: bool,
) -> None:
    """Refuse to write the served artifacts unless this run replaces all of them.

    The bounded scaler is one MinMaxScaler fit jointly on every training row,
    and utilization weights / percentile bounds are per position but share
    one file each -- so a run over a subset of positions rewrites what the
    positions it did NOT retrain are served with (a QB-only run left RB/WR/TE
    on a QB-fit scaler, default utilization weights and no percentile bounds;
    GAPS.md 2026-09-28). A fit_models=False run is worse: it rewrites the
    preprocessing and trains nothing to match it. Anything other than a full
    production retrain must run inside redirect_models_dir.
    """
    if not is_production_models_dir(models_dir):
        return
    if not production_run:
        raise ProductionArtifactWriteError(
            f"refusing to write {models_dir}: only a full production retrain "
            "(train_models without --walk-forward/--loyo) may replace the served "
            "artifacts. Wrap this call in redirect_models_dir(<temp dir>).")
    if not fit_models:
        raise ProductionArtifactWriteError(
            "refusing a fit_models=False production run: it would rewrite the "
            "served scaler/utilization artifacts without retraining the models "
            "that depend on them.")
    missing = sorted(set(settings.POSITIONS) - set(positions))
    if missing:
        raise ProductionArtifactWriteError(
            f"refusing a production retrain without {missing}: the bounded scaler "
            "and utilization weights/bounds are shared across positions, so a "
            "subset run desyncs the positions it skips. Retrain all of "
            f"{list(settings.POSITIONS)}, or experiment with --walk-forward.")


@contextmanager
def redirect_models_dir(target: Path | str) -> Iterator[Path]:
    """Point every MODELS_DIR binding at ``target`` for the duration.

    Both halves are needed. Rebinding ``settings.MODELS_DIR`` covers modules
    imported *after* the redirect starts; rewriting the already-imported
    bindings covers the modules that copied the value before it. On exit
    every binding is restored, including modules first imported inside the
    block -- those would otherwise be left pointing at a deleted temporary
    directory.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    original = settings.MODELS_DIR
    saved: List[Tuple[object, Path]] = [(m, m.MODELS_DIR) for m in _bound_modules()]

    settings.MODELS_DIR = target
    for module, _ in saved:
        module.MODELS_DIR = target
    try:
        yield target
    finally:
        settings.MODELS_DIR = original
        previous = {id(module): old for module, old in saved}
        # Re-scan rather than reusing `saved`: anything imported inside the
        # block bound `target` and must be sent back to the real directory.
        for module in _bound_modules():
            module.MODELS_DIR = previous.get(id(module), original)

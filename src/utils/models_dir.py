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
from typing import Iterator, List, Tuple

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

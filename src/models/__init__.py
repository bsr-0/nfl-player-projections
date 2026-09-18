"""ML models for prediction.

Public legacy exports are loaded lazily.  Eager imports made every independent
submodule require XGBoost and the full ensemble stack merely to be imported.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ensemble import EnsemblePredictor
    from .position_models import PositionModel

__all__ = ["PositionModel", "EnsemblePredictor"]


def __getattr__(name: str):
    if name == "PositionModel":
        from .position_models import PositionModel
        return PositionModel
    if name == "EnsemblePredictor":
        from .ensemble import EnsemblePredictor
        return EnsemblePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Turn model point forecasts into unambiguous market picks.

The project convention is that ``spread_line`` and a margin forecast are
both *home-team margins*: a positive number favours the home team.  Keeping
this small conversion in one place prevents the common sign inversion where
an away underdog is accidentally reported as the pick to cover.
"""
from __future__ import annotations

import math
from typing import Optional


def _finite(value: object) -> Optional[float]:
    """Return a usable float, treating None/NaN/infinity as unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def spread_pick(home_team: str, away_team: str, predicted_margin: object, spread_line: object) -> dict:
    """Return the ATS side and edge for a home-margin forecast.

    A positive edge means the model is higher on the home team than the
    market.  Exact matches deliberately produce no pick, rather than
    inventing conviction where the model has none.
    """
    forecast, line = _finite(predicted_margin), _finite(spread_line)
    if forecast is None or line is None:
        return {"pick": None, "edge": None}
    edge = forecast - line
    if edge == 0:
        return {"pick": None, "edge": edge}
    return {"pick": home_team if edge > 0 else away_team, "edge": edge}


def total_pick(predicted_total: object, total_line: object) -> dict:
    """Return ``OVER``/``UNDER`` and point edge for a total forecast."""
    forecast, line = _finite(predicted_total), _finite(total_line)
    if forecast is None or line is None:
        return {"pick": None, "edge": None}
    edge = forecast - line
    if edge == 0:
        return {"pick": None, "edge": edge}
    return {"pick": "OVER" if edge > 0 else "UNDER", "edge": edge}

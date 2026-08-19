"""Causal P(plays) estimators for the season projection's synthetic weeks.

Motivation (measured, not assumed). Phase 7's season-total bias tracks the
gap between assumed and realized availability almost perfectly:

    synthetic share | n  | bias   | assumed | realized
    none            | 39 | -28.1  | 0.833   | 0.897
    0-25%           | 41 | +14.4  | 0.812   | 0.834
    25-50%          | 41 | +43.3  | 0.710   | 0.568
    50-75%          | 49 | +54.1  | 0.547   | 0.365
    75-100%         | 68 | +43.5  | 0.407   | 0.157

The original estimator uses PRIOR SEASONS ONLY, so a QB benched or injured
in week 3 is still assigned his prior-season healthy rate for every
remaining week. Games already missed in weeks 1..W-1 are fully known when
projecting week W -- that is legitimate causal information the estimator
was discarding, and it was discarding it worst for exactly the players who
most need it.

Every estimator here is strictly causal: it may read the player's rows for
weeks < target_week in the current season, and any prior season, but never
week >= target_week. The formulation is deliberately NOT decided a priori --
they are compared empirically (scripts/run_availability_comparison.py) on
both the real-vs-synthetic bias split and season-level accuracy.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

POSITION_AVG_FALLBACK = 0.88

# Shrinkage strength: weight on current-season evidence is
# n_elapsed / (n_elapsed + SHRINKAGE_K). At k=4, one elapsed week gets 20%
# weight and 12 elapsed weeks get 75% -- encoding that a missed game after
# 2 games is much weaker evidence than a missed game after 14.
SHRINKAGE_K = 4.0

# Half-life (in weeks) for the recency-weighted variant: a game 4 weeks ago
# counts half as much as last week's.
RECENCY_HALFLIFE = 4.0


class PlayerAvailabilityHistory:
    """Precomputed per-player game log, so estimators are cheap to call
    per (player, week) inside the projection loop rather than re-querying.

    `weeks_by_player_season[(player_id, season)]` -> sorted array of weeks
    the player has a real row for. `team_weeks[(team, season)]` -> sorted
    array of that team's regular-season weeks.
    """

    def __init__(self, history: pd.DataFrame, team_weeks: Dict[tuple, Sequence[int]]):
        self.team_weeks = {k: np.asarray(sorted(v)) for k, v in team_weeks.items()}
        self._weeks: Dict[tuple, np.ndarray] = {}
        if history is not None and not history.empty:
            for (pid, season), g in history.groupby(["player_id", "season"]):
                self._weeks[(pid, int(season))] = np.asarray(sorted(g["week"].astype(int).unique()))

    def weeks_played(self, player_id: str, season: int) -> np.ndarray:
        return self._weeks.get((player_id, int(season)), np.asarray([], dtype=int))

    def seasons_for(self, player_id: str) -> list:
        return sorted({s for (p, s) in self._weeks if p == player_id})

    def team_regular_weeks(self, team: str, season: int) -> np.ndarray:
        return self.team_weeks.get((team, int(season)), np.asarray([], dtype=int))


def _prior_season_rate(hist: PlayerAvailabilityHistory, player_id: str, season: int,
                        team: str) -> Optional[float]:
    """Mean games-played rate across seasons strictly before `season`."""
    rates = []
    for s in hist.seasons_for(player_id):
        if s >= season:
            continue
        possible = len(hist.team_regular_weeks(team, s))
        if possible <= 0:
            possible = 17  # team unknown for that season; league-typical length
        played = len(hist.weeks_played(player_id, s))
        if possible > 0:
            rates.append(min(played / possible, 1.0))
    return float(np.mean(rates)) if rates else None


def _current_season_to_date(hist: PlayerAvailabilityHistory, player_id: str, season: int,
                             team: str, target_week: int) -> tuple:
    """(rate, n_elapsed) over team weeks STRICTLY BEFORE target_week.

    n_elapsed is the count of the team's games already played this season --
    the amount of in-season evidence available, which the shrinkage variants
    use to decide how much to trust `rate`.
    """
    team_wks = hist.team_regular_weeks(team, season)
    elapsed = team_wks[team_wks < target_week]
    n_elapsed = len(elapsed)
    if n_elapsed == 0:
        return None, 0
    played = hist.weeks_played(player_id, season)
    n_played = int(np.isin(played, elapsed).sum())
    return min(n_played / n_elapsed, 1.0), n_elapsed


# --- estimators ------------------------------------------------------------
# All share the signature (hist, player_id, season, team, target_week) -> float

def prior_season_only(hist, player_id, season, team, target_week) -> float:
    """The original behaviour (baseline). Ignores target_week entirely."""
    prior = _prior_season_rate(hist, player_id, season, team)
    return prior if prior is not None else POSITION_AVG_FALLBACK


def current_season_only(hist, player_id, season, team, target_week) -> float:
    """Pure in-season rate; falls back to prior (then league default) when
    no games have elapsed yet -- e.g. a week-1 projection."""
    rate, n = _current_season_to_date(hist, player_id, season, team, target_week)
    if rate is not None:
        return rate
    return prior_season_only(hist, player_id, season, team, target_week)


def simple_blend(hist, player_id, season, team, target_week) -> float:
    """Equal weight on prior-season and current-season-to-date."""
    cur, n = _current_season_to_date(hist, player_id, season, team, target_week)
    prior = _prior_season_rate(hist, player_id, season, team)
    if cur is None:
        return prior if prior is not None else POSITION_AVG_FALLBACK
    if prior is None:
        return cur
    return 0.5 * prior + 0.5 * cur


def shrinkage_blend(hist, player_id, season, team, target_week) -> float:
    """Weight current-season evidence by how much of it exists:
    w = n_elapsed / (n_elapsed + k). Encodes that 1 missed game after 2
    games is far weaker evidence than 1 missed game after 14."""
    cur, n = _current_season_to_date(hist, player_id, season, team, target_week)
    prior = _prior_season_rate(hist, player_id, season, team)
    base = prior if prior is not None else POSITION_AVG_FALLBACK
    if cur is None:
        return base
    w = n / (n + SHRINKAGE_K)
    return w * cur + (1.0 - w) * base


def recency_weighted(hist, player_id, season, team, target_week) -> float:
    """In-season availability with exponentially decaying weight on older
    weeks, then shrunk toward the prior by total evidence weight. A player
    who missed weeks 2-4 but has played 8-9 is treated as more available
    than a flat in-season rate would suggest."""
    team_wks = hist.team_regular_weeks(team, season)
    elapsed = team_wks[team_wks < target_week]
    prior = _prior_season_rate(hist, player_id, season, team)
    base = prior if prior is not None else POSITION_AVG_FALLBACK
    if len(elapsed) == 0:
        return base
    played = set(hist.weeks_played(player_id, season).tolist())
    decay = 0.5 ** ((target_week - elapsed) / RECENCY_HALFLIFE)
    played_mask = np.array([1.0 if w in played else 0.0 for w in elapsed])
    total_w = decay.sum()
    if total_w <= 0:
        return base
    cur = float((decay * played_mask).sum() / total_w)
    # Shrink by effective sample size (sum of weights), same spirit as above.
    w = total_w / (total_w + SHRINKAGE_K)
    return w * cur + (1.0 - w) * base


AVAILABILITY_ESTIMATORS: Dict[str, Callable] = {
    "prior_season_only": prior_season_only,      # current production behaviour
    "current_season_only": current_season_only,
    "simple_blend": simple_blend,
    "shrinkage_blend": shrinkage_blend,
    "recency_weighted": recency_weighted,
}

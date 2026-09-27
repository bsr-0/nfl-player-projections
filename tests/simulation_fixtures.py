"""Synthetic OOF panels with known marginals and known game dependence.

Shared by the simulation tests. Not collected by pytest (no ``test_`` prefix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

# Mean served prediction by (position, depth rank) for one team.
ROSTER = (("QB", 1, 18.0), ("RB", 1, 13.0), ("RB", 2, 6.0), ("WR", 1, 15.0),
          ("WR", 2, 10.0), ("WR", 3, 6.0), ("TE", 1, 8.0))
# Game, team-volume and game-script loadings. Script loads with opposite
# sign on RBs and pass catchers, so same-team RB1-WR1 is negatively
# correlated while QB1-WR1 is strongly positive.
DEPENDENT_LOADINGS = {
    "QB1": (0.30, 0.45, 0.45), "RB1": (0.20, 0.20, -0.50), "RB2": (0.15, 0.20, -0.30),
    "WR1": (0.25, 0.35, 0.45), "WR2": (0.20, 0.30, 0.35), "WR3": (0.15, 0.20, 0.25),
    "TE1": (0.20, 0.30, 0.30),
}
GAMMA_SHAPE = 2.0


def factor_scores(canonical_roles, sides, rng, loadings) -> np.ndarray:
    """One game's normal scores under the three-factor model."""
    game, script = rng.standard_normal(2)
    team = {"home": rng.standard_normal(), "away": rng.standard_normal()}
    scores = []
    for role, side in zip(canonical_roles, sides):
        g, v, s = loadings.get(role, (0.0, 0.0, 0.0)) if loadings else (0.0, 0.0, 0.0)
        signed_script = script if side == "home" else -script
        unique = np.sqrt(1 - g * g - v * v - s * s)
        scores.append(g * game + v * team[side] + s * signed_script + unique * rng.standard_normal())
    return np.asarray(scores)


def synthetic_panel(*, seasons=(2022, 2023, 2024), weeks=8, n_teams=8, loadings=None,
                    seed=0) -> pd.DataFrame:
    """Panel shaped like a verified OOF run with game context.

    Actuals are Gamma(shape=2, mean=prediction) outcomes linked through a
    Gaussian copula: heteroscedastic and right-skewed, mean-calibrated.
    ``loadings=None`` gives independent players (the null).
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            order = rng.permutation(teams)
            for home, away in zip(order[0::2], order[1::2]):
                game_id = f"{season}_{week:02d}_{away}_{home}"
                players = []
                for side, team, opponent in (("home", home, away), ("away", away, home)):
                    for position, rank, mean in ROSTER:
                        predicted = float(mean * rng.uniform(0.85, 1.15))
                        players.append((side, team, opponent, position, rank, predicted))
                roles = [f"{position}{rank}" for _, _, _, position, rank, _ in players]
                sides = [side for side, *_ in players]
                z = factor_scores(roles, sides, rng, loadings)
                for (side, team, opponent, position, rank, predicted), score in zip(players, z):
                    actual = float(gamma.ppf(norm.cdf(score), GAMMA_SHAPE, scale=predicted / GAMMA_SHAPE))
                    rows.append({
                        "player_id": f"{team}_{position}{rank}", "season": season, "week": week,
                        "game_id": game_id, "team": team, "opponent": opponent,
                        "home_team": home, "away_team": away, "position": position,
                        "predicted_points": predicted, "actual_points": actual,
                        "residual": predicted - actual, "is_cold_start": False,
                    })
    return pd.DataFrame(rows)

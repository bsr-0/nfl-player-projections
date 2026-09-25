"""Game-script simulation primitives.

This module is intentionally database-independent.  It consumes predictions
already produced by the game and player serving paths; learned usage shares,
availability, and role-correlation artifacts are separate inputs.
"""
from dataclasses import dataclass
from hashlib import blake2b
from math import sqrt
from typing import Iterable
import numpy as np

from src.models.player_correlation import (
    ResidualCorrelationModel,
    RoleResidualCorrelationModel,
)
from src.models.usage_allocation import draw_usage_shares

@dataclass(frozen=True)
class TeamVolumeBaseline:
    plays: float = 64.0
    pass_rate: float = 0.58

@dataclass(frozen=True)
class GameScriptInput:
    game_id: str
    home_team: str
    away_team: str
    home_win_prob: float
    predicted_margin: float
    predicted_total: float
    home: TeamVolumeBaseline = TeamVolumeBaseline()
    away: TeamVolumeBaseline = TeamVolumeBaseline()
    margin_sd: float = 10.0
    total_sd: float = 10.0
    # Optional: set by simulation_adapter so a home/away team pair can be
    # disambiguated across multiple weeks of schedule (see
    # simulation_adapter.player_inputs_from_predictions).
    season: int | None = None
    week: int | None = None

@dataclass(frozen=True)
class PlayerSimulationInput:
    player_id: str
    team: str
    position: str
    mean_fantasy_points: float
    sd_fantasy_points: float = 8.0
    usage_share: float | None = None
    participation_prob: float = 1.0

@dataclass(frozen=True)
class GameDraw:
    draw: int
    home_score: float
    away_score: float
    home_plays: int
    away_plays: int
    home_pass_attempts: int
    away_pass_attempts: int
    home_won: bool
    simulated_margin: float
    simulated_total: float

def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def _seed_for_game(seed: int, game_id: str) -> int:
    digest = blake2b(game_id.encode("utf-8"), digest_size=8).digest()
    return (int(seed) + int.from_bytes(digest, "little")) % (2**63 - 1)

def _score_state_pass_adjustment(score_diff: float) -> float:
    return float(np.clip(-0.0025 * score_diff, -0.10, 0.10))

def _draw_volume(rng, baseline, score_diff):
    plays = max(1, int(rng.poisson(max(1.0, baseline.plays))))
    rate = _clip_probability(baseline.pass_rate + _score_state_pass_adjustment(score_diff))
    return plays, int(rng.binomial(plays, rate))

def _split_score(total: float, margin: float) -> tuple[float, float]:
    """Convert total/margin to nonnegative scores while conserving total."""
    total = max(0.0, total)
    if margin >= total:
        return total, 0.0
    if margin <= -total:
        return 0.0, total
    return (total + margin) / 2.0, (total - margin) / 2.0

def _draw_margin(rng, home_win_prob: float, predicted_margin: float,
                 margin_sd: float) -> tuple[float, bool]:
    """Draw a signed margin matching supplied win probability and mean margin.

    The sign is sampled from home_win_prob. Conditional margin magnitudes use
    exponential distributions whose means are solved so the unconditional
    expected margin is predicted_margin. This is a coherent fallback until a
    joint score model is fitted from historical data.
    """
    probability = float(np.clip(home_win_prob, 0.001, 0.999))
    base_loss_margin = max(1.0, margin_sd * sqrt(2.0 / np.pi))
    if predicted_margin >= 0:
        away_margin_mean = base_loss_margin
        home_margin_mean = max(
            0.01, (predicted_margin + (1.0 - probability) * away_margin_mean) / probability)
    else:
        home_margin_mean = base_loss_margin
        away_margin_mean = max(
            0.01, (probability * home_margin_mean - predicted_margin) / (1.0 - probability))
    home_won = bool(rng.random() < probability)
    magnitude = rng.exponential(home_margin_mean if home_won else away_margin_mean)
    return (float(magnitude) if home_won else -float(magnitude)), home_won

def simulate_game_scripts(game: GameScriptInput, n_draws: int = 1000,
                          seed: int = 42) -> list[GameDraw]:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if not 0 <= game.home_win_prob <= 1:
        raise ValueError("home_win_prob must be in [0, 1]")
    if game.predicted_total < 0 or game.margin_sd < 0 or game.total_sd < 0:
        raise ValueError("totals and standard deviations must be nonnegative")
    rng = np.random.default_rng(_seed_for_game(seed, game.game_id))
    output = []
    for draw in range(n_draws):
        total = max(0.0, float(rng.normal(game.predicted_total, game.total_sd)))
        margin, home_won = _draw_margin(
            rng, game.home_win_prob, game.predicted_margin, game.margin_sd)
        home_score, away_score = _split_score(total, margin)
        realized_margin = home_score - away_score
        hp, hpa = _draw_volume(rng, game.home, realized_margin)
        ap, apa = _draw_volume(rng, game.away, -realized_margin)
        output.append(GameDraw(
            draw, home_score, away_score, hp, ap, hpa, apa,
            home_won, realized_margin, home_score + away_score))
    return output

def _opportunity_multiplier(player, team_plays, team_pass, baseline):
    """Volume-only fallback; usage allocation must be supplied separately."""
    passing = player.position in ("QB", "WR", "TE")
    actual = team_pass if passing else team_plays - team_pass
    expected = baseline.plays * (baseline.pass_rate if passing else 1.0 - baseline.pass_rate)
    return actual / max(1.0, expected)

def _opportunity_type(player: PlayerSimulationInput) -> str:
    """Which team opportunity pool this player's `usage_share` is a fraction of.

    QB and WR/TE both draw on the team's pass-attempt volume, but they are
    NOT the same pool: a starting QB's `pass_share` (the adapter's
    pass_share/usage_share field) is his share of the team's own dropbacks
    -- normally ~1.0 -- while a receiver's `target_share` is that
    receiver's share of the targets thrown on those same dropbacks. Pooling
    them together would sum a ~1.0 QB share with several ~0.1-0.3 receiver
    shares in one pool, which `draw_usage_shares` correctly rejects as
    exceeding one team's opportunity pool. Both pools still size themselves
    off the same pass-attempt volume (see `_is_pass_opportunity`); only
    their player-level share semantics differ.
    """
    if player.position == "RB":
        return "rush"
    if player.position == "QB":
        return "pass_dropbacks"
    return "pass_targets"

def _is_pass_opportunity(opportunity: str) -> bool:
    return opportunity in ("pass_dropbacks", "pass_targets")

def _draw_usage_multipliers(game: GameScriptInput,
                            players: list[PlayerSimulationInput],
                            script: GameDraw,
                            rng: np.random.Generator) -> dict[int, float]:
    """Allocate optional player shares inside each team's pass/rush pool.

    A player without a supplied `usage_share` is simply left out of the
    returned mapping -- `simulate_players` falls back to
    `_opportunity_multiplier` for them individually -- rather than the pool
    rejecting a partially-supplied group. Real serving data routinely has
    partial coverage (e.g. a rookie with no `target_share` yet alongside
    veterans that have one), so requiring every player in a pool to have a
    share made the common case an error.
    """
    grouped = {}
    for index, player in enumerate(players):
        grouped.setdefault((player.team, _opportunity_type(player)), []).append((index, player))
    multipliers = {}
    for (team, opportunity), entries in grouped.items():
        supplied_entries = [(index, player) for index, player in entries
                             if player.usage_share is not None]
        if not supplied_entries:
            continue
        shares = np.asarray([player.usage_share for _, player in supplied_entries], dtype=float)
        sampled = draw_usage_shares(shares, concentration=80.0, rng=rng)
        is_home = team == game.home_team
        baseline = game.home if is_home else game.away
        is_pass = _is_pass_opportunity(opportunity)
        actual = ((script.home_pass_attempts if is_home else script.away_pass_attempts)
                  if is_pass
                  else (script.home_plays - script.home_pass_attempts
                        if is_home else script.away_plays - script.away_pass_attempts))
        expected_pool = baseline.plays * (
            baseline.pass_rate if is_pass else 1.0 - baseline.pass_rate)
        for (index, _), base_share, draw_share in zip(supplied_entries, shares, sampled):
            multipliers[index] = (
                0.0 if base_share == 0
                else actual * draw_share / max(1e-6, expected_pool * base_share))
    return multipliers

def role_keys_for_players(game: GameScriptInput,
                          players: list[PlayerSimulationInput]) -> tuple[str, ...]:
    """Assign stable within-game roles such as home_WR1 and away_RB2."""
    grouped = {}
    for index, player in enumerate(players):
        side = "home" if player.team == game.home_team else "away"
        grouped.setdefault((side, player.position), []).append((index, player))
    assigned = {}
    for (side, position), entries in grouped.items():
        entries.sort(
            key=lambda item: (
                item[1].usage_share is not None,
                item[1].usage_share if item[1].usage_share is not None else -1.0,
                item[1].mean_fantasy_points,
                item[1].player_id,
            ),
            reverse=True,
        )
        for rank, (index, _) in enumerate(entries, start=1):
            assigned[index] = f"{side}_{position}{rank}"
    return tuple(assigned[index] for index in range(len(players)))

def simulate_players(game: GameScriptInput, players: Iterable[PlayerSimulationInput],
                     n_draws: int = 1000, seed: int = 42,
                     correlation_model: (ResidualCorrelationModel
                                         | RoleResidualCorrelationModel
                                         | None) = None) -> list[dict]:
    scripts = simulate_game_scripts(game, n_draws, seed)
    player_list = list(players)
    player_keys = tuple(player.player_id for player in player_list)
    marginal_sds = np.asarray(
        [max(0.01, player.sd_fantasy_points) for player in player_list], dtype=float)
    correlated_noise = None
    if isinstance(correlation_model, ResidualCorrelationModel):
        if correlation_model.player_keys != player_keys:
            raise ValueError("correlation model keys must exactly match simulation players")
        correlated_noise = correlation_model.sample_scaled_residuals(
            marginal_sds, n_draws, _seed_for_game(seed + 2, game.game_id))
    elif isinstance(correlation_model, RoleResidualCorrelationModel):
        correlated_noise = correlation_model.sample_scaled_residuals(
            role_keys_for_players(game, player_list), marginal_sds, n_draws,
            _seed_for_game(seed + 2, game.game_id))
    rng = np.random.default_rng(_seed_for_game(seed + 1, game.game_id))
    usage_rng = np.random.default_rng(_seed_for_game(seed + 3, game.game_id))
    rows = []
    for script in scripts:
        usage_multipliers = _draw_usage_multipliers(game, player_list, script, usage_rng)
        for index, player in enumerate(player_list):
            if player.team not in (game.home_team, game.away_team):
                raise ValueError("player is not in this game")
            is_home = player.team == game.home_team
            baseline = game.home if is_home else game.away
            team_plays = script.home_plays if is_home else script.away_plays
            team_pass = script.home_pass_attempts if is_home else script.away_pass_attempts
            active = rng.random() < _clip_probability(player.participation_prob)
            multiplier = usage_multipliers.get(
                index, _opportunity_multiplier(player, team_plays, team_pass, baseline))
            noise = (correlated_noise[script.draw, index]
                     if correlated_noise is not None
                     else rng.normal(0.0, marginal_sds[index]))
            value = 0.0 if not active else max(
                0.0, player.mean_fantasy_points * multiplier + noise)
            rows.append({"game_id": game.game_id, "draw": script.draw, "player_id": player.player_id,
                         "team": player.team, "position": player.position,
                         "home_score": script.home_score, "away_score": script.away_score,
                         "plays": team_plays, "pass_attempts": team_pass,
                         "fantasy_points": value, "active": active})
    return rows

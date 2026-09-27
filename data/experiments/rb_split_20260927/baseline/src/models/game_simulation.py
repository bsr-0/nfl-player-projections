"""Game-script simulation primitives.

This module is intentionally database-independent.  It consumes predictions
already produced by the game and player serving paths; learned usage shares,
availability, and role-correlation artifacts are separate inputs.
"""
from dataclasses import dataclass
from hashlib import blake2b
from math import erf, sqrt
from typing import Iterable
import logging
import numpy as np

from src.models.player_correlation import (
    ResidualCorrelationModel,
    RoleResidualCorrelationModel,
)
from src.models.usage_allocation import draw_usage_shares

logger = logging.getLogger(__name__)

# Chosen loosely: existing test fixtures' predicted_margin/margin_sd pairs
# (e.g. .75 win prob with a 3-point margin at margin_sd=10) imply a
# probability roughly 10-15 points away from the supplied home_win_prob and
# should not warn; a mismatch this large (e.g. a 1% win probability paired
# with a double-digit positive predicted home margin) means the two model
# heads are describing different games, not just disagreeing on degree.
MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD = 0.20

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

def implied_win_probability_from_margin(predicted_margin: float, margin_sd: float) -> float:
    """Win probability implied by treating margin ~ Normal(predicted_margin, margin_sd).

    An independent estimate of P(home wins), distinct from home_win_prob,
    used only to check the two inputs for mutual consistency -- see
    margin_win_probability_disagreement.

    margin_sd is treated as the SD of the margin variable itself, matching
    how _draw_margin above already uses it directly (`margin_sd * sqrt(2/pi)`
    is the mean-absolute-value of N(0, margin_sd)) -- not as a per-team score
    SD requiring a further sqrt(2) inflation. Keep this consistent with
    _draw_margin if either changes; the two disagreeing was a real bug
    caught by two independent implementations landing here at once
    (2026-09-26).
    """
    if margin_sd <= 0:
        if predicted_margin > 0:
            return 1.0
        if predicted_margin < 0:
            return 0.0
        return 0.5
    z = predicted_margin / (margin_sd * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))

def margin_win_probability_disagreement(game: GameScriptInput) -> float:
    """Absolute gap between game.home_win_prob and the margin's own implied probability.

    _draw_margin is internally self-consistent (the unconditional expected
    margin it produces always equals predicted_margin given home_win_prob),
    but nothing upstream guarantees home_win_prob and predicted_margin
    describe the same game consistently -- they can come from independently
    trained model heads. A large gap here means _draw_margin will solve for
    extreme, one-sided exponential means to reconcile them, producing a
    silently bimodal, unrealistic score distribution rather than a warning
    (docs/GAME_SIMULATION_CORRELATION_PLAN.md, "Code review findings
    (2026-09-26)").
    """
    return abs(game.home_win_prob - implied_win_probability_from_margin(
        game.predicted_margin, game.margin_sd))

def simulate_game_scripts(game: GameScriptInput, n_draws: int = 1000,
                          seed: int = 42) -> list[GameDraw]:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if not 0 <= game.home_win_prob <= 1:
        raise ValueError("home_win_prob must be in [0, 1]")
    if game.predicted_total < 0 or game.margin_sd < 0 or game.total_sd < 0:
        raise ValueError("totals and standard deviations must be nonnegative")
    disagreement = margin_win_probability_disagreement(game)
    if disagreement > MARGIN_WIN_PROBABILITY_DISAGREEMENT_THRESHOLD:
        logger.warning(
            "game %s: home_win_prob=%.3f disagrees with the margin-implied "
            "win probability=%.3f (predicted_margin=%.2f, margin_sd=%.2f) by "
            "%.3f -- _draw_margin will solve for an extreme, one-sided "
            "exponential mean to reconcile them",
            game.game_id, game.home_win_prob,
            implied_win_probability_from_margin(game.predicted_margin, game.margin_sd),
            game.predicted_margin, game.margin_sd, disagreement)
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

def _grouped_role_assignments(game: GameScriptInput,
                              players: list[PlayerSimulationInput]) -> dict:
    """Group players by (side, position) and rank each group.

    Shared by role_keys_for_players and role_assignment_diagnostics so the
    two can never drift apart on what counts as which role.

    Ranks on mean_fantasy_points/player_id only -- usage_share is
    deliberately NOT used here, even though it is often a better serving-time
    signal, because the role-keyed correlation model these keys are used to
    query (RoleResidualCorrelationModel) is fit from the historical OOF
    panel via role_keys_from_panel_game (src/models/residual_calibration.py),
    which has no usage_share column to rank on at all (oof_capture.py never
    captures it). A role label is only meaningful if it means the same thing
    at fit time and serve time; ranking on a signal the fitting side
    structurally cannot see would silently misapply one player's fitted
    correlation to a different player whenever usage_share reordered them
    relative to points (docs/GAME_SIMULATION_CORRELATION_PLAN.md, "Code
    review findings (2026-09-26)"). If usage_share is ever added to the OOF
    panel, both ranking functions should start using it together, not just
    this one.
    """
    grouped = {}
    for index, player in enumerate(players):
        side = "home" if player.team == game.home_team else "away"
        grouped.setdefault((side, player.position), []).append((index, player))
    for entries in grouped.values():
        entries.sort(
            key=lambda item: (item[1].mean_fantasy_points, item[1].player_id),
            reverse=True,
        )
    return grouped

def role_keys_for_players(game: GameScriptInput,
                          players: list[PlayerSimulationInput]) -> tuple[str, ...]:
    """Assign stable within-game roles such as home_WR1 and away_RB2."""
    grouped = _grouped_role_assignments(game, players)
    assigned = {}
    for (side, position), entries in grouped.items():
        for rank, (index, _) in enumerate(entries, start=1):
            assigned[index] = f"{side}_{position}{rank}"
    return tuple(assigned[index] for index in range(len(players)))

def role_assignment_diagnostics(game: GameScriptInput,
                                players: list[PlayerSimulationInput]) -> dict[str, bool]:
    """Flag which role keys were assigned via a genuine mean_fantasy_points tie.

    _grouped_role_assignments ranks on mean_fantasy_points then player_id --
    player_id carries no role information, so whenever two players in the
    same (side, position) group have equal mean_fantasy_points, which one
    lands in the higher-numbered role slot is arbitrary. A role like
    home_RB2, fit from many historical games, can therefore end up
    representing whichever depth player happened to tie-break ahead that
    week rather than a stable role, which dilutes any real correlation
    signal for that slot.

    Returns one entry per role key: True if that slot's occupant shared its
    exact mean_fantasy_points value with at least one other player in the
    same (side, position) group (rank order between them was decided by the
    player_id tiebreak, not by an actual difference in projected points).
    """
    grouped = _grouped_role_assignments(game, players)
    diagnostics = {}
    for (side, position), entries in grouped.items():
        points = [player.mean_fantasy_points for _, player in entries]
        tied_values = {value for value in points if points.count(value) > 1}
        for rank, (_, player) in enumerate(entries, start=1):
            diagnostics[f"{side}_{position}{rank}"] = player.mean_fantasy_points in tied_values
    return diagnostics

def simulate_players(game: GameScriptInput, players: Iterable[PlayerSimulationInput],
                     n_draws: int = 1000, seed: int = 42,
                     correlation_model: (ResidualCorrelationModel
                                         | RoleResidualCorrelationModel
                                         | None) = None,
                     residual_draws: np.ndarray | None = None,
                     apply_game_script: bool = True) -> list[dict]:
    scripts = simulate_game_scripts(game, n_draws, seed)
    player_list = list(players)
    player_keys = tuple(player.player_id for player in player_list)
    marginal_sds = np.asarray(
        [max(0.01, player.sd_fantasy_points) for player in player_list], dtype=float)
    empirical_noise = None
    if residual_draws is not None:
        empirical_noise = np.asarray(residual_draws, dtype=float)
        if empirical_noise.shape != (n_draws, len(player_list)) or not np.isfinite(empirical_noise).all():
            raise ValueError("residual_draws must be finite and shaped (n_draws, players)")
        if correlation_model is not None:
            raise ValueError("apply shared dependence to residual_draws before simulate_players")
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
            multiplier = (usage_multipliers.get(
                index, _opportunity_multiplier(player, team_plays, team_pass, baseline))
                if apply_game_script else 1.0)
            noise = (empirical_noise[script.draw, index]
                     if empirical_noise is not None else
                     correlated_noise[script.draw, index]
                     if correlated_noise is not None else rng.normal(0.0, marginal_sds[index]))
            raw_value = player.mean_fantasy_points * multiplier + noise
            # Empirical OOF residuals may legitimately yield negative PPR;
            # clipping them would shift their center away from the served
            # point forecast and invalidate marginal calibration.
            value = 0.0 if not active else (raw_value if empirical_noise is not None else max(0.0, raw_value))
            rows.append({"game_id": game.game_id, "draw": script.draw, "player_id": player.player_id,
                         "team": player.team, "position": player.position,
                         "home_score": script.home_score, "away_score": script.away_score,
                         "plays": team_plays, "pass_attempts": team_pass,
                         "fantasy_points": value, "active": active})
    return rows

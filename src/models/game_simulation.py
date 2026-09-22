"""Database-independent game-script simulation primitives."""
from dataclasses import dataclass
from typing import Iterable
import numpy as np

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

@dataclass(frozen=True)
class PlayerSimulationInput:
    player_id: str
    team: str
    position: str
    mean_fantasy_points: float
    sd_fantasy_points: float = 8.0
    usage_share: float = 0.0
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

def _clip_probability(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def _score_state_pass_adjustment(score_diff: float) -> float:
    return float(np.clip(-0.0025 * score_diff, -0.10, 0.10))

def _draw_volume(rng, baseline, score_diff):
    plays = max(1, int(rng.poisson(max(1.0, baseline.plays))))
    rate = _clip_probability(baseline.pass_rate + _score_state_pass_adjustment(score_diff))
    return plays, int(rng.binomial(plays, rate))

def simulate_game_scripts(game: GameScriptInput, n_draws: int = 1000,
                          seed: int = 42) -> list[GameDraw]:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if not 0 <= game.home_win_prob <= 1:
        raise ValueError("home_win_prob must be in [0, 1]")
    if game.predicted_total < 0 or game.margin_sd < 0 or game.total_sd < 0:
        raise ValueError("totals and standard deviations must be nonnegative")
    rng = np.random.default_rng(seed)
    output = []
    for draw in range(n_draws):
        total = max(0.0, rng.normal(game.predicted_total, game.total_sd))
        margin = rng.normal(game.predicted_margin, game.margin_sd)
        home_score = max(0.0, (total + margin) / 2.0)
        away_score = max(0.0, total - home_score)
        hp, hpa = _draw_volume(rng, game.home, home_score - away_score)
        ap, apa = _draw_volume(rng, game.away, away_score - home_score)
        output.append(GameDraw(draw, home_score, away_score, hp, ap, hpa, apa))
    return output

def _opportunity_multiplier(player, team_plays, team_pass, baseline):
    """Use volume deviation from baseline; a neutral draw preserves the mean."""
    passing = player.position in ("QB", "WR", "TE")
    actual = team_pass if passing else team_plays - team_pass
    expected = baseline.plays * (baseline.pass_rate if passing else 1.0 - baseline.pass_rate)
    return actual / max(1.0, expected)

def simulate_players(game: GameScriptInput, players: Iterable[PlayerSimulationInput],
                     n_draws: int = 1000, seed: int = 42) -> list[dict]:
    scripts = simulate_game_scripts(game, n_draws, seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for script in scripts:
        for player in players:
            if player.team not in (game.home_team, game.away_team):
                raise ValueError("player is not in this game")
            is_home = player.team == game.home_team
            baseline = game.home if is_home else game.away
            team_plays = script.home_plays if is_home else script.away_plays
            team_pass = script.home_pass_attempts if is_home else script.away_pass_attempts
            active = rng.random() < _clip_probability(player.participation_prob)
            multiplier = _opportunity_multiplier(player, team_plays, team_pass, baseline)
            value = 0.0 if not active else max(0.0, rng.normal(
                player.mean_fantasy_points * multiplier,
                max(0.01, player.sd_fantasy_points)))
            rows.append({"game_id": game.game_id, "draw": script.draw, "player_id": player.player_id,
                         "team": player.team, "position": player.position,
                         "home_score": script.home_score, "away_score": script.away_score,
                         "plays": team_plays, "pass_attempts": team_pass,
                         "fantasy_points": value, "active": active})
    return rows

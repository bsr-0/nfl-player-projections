"""Forward check of the game simulation (docs/SIMULATION_FORWARD_CHECK.md).

Scores what was published: for every final game of the window, the version of
``docs/data/simulation_{season}_wk{N}.json`` committed before that game's
kickoff. The site file holds summaries, not draws, so the draws are rebuilt
from the weekly and game files of the same commit with the same artifacts, seed
and draw count, and the rebuild is checked against the published per-player
summaries; a game whose rebuild does not match is excluded and listed, never
scored. The independent arm is the same marginal draws without the role-factor
reordering (calibrated_simulation.simulate_game's first step).

Two populations, fixed by the pre-registration:

    played      players with a stat line; stacks and totals over them only (the
                backtest's rule). Used for the correlation-layer verdict.
    published   the sums exactly as published (every simulated player; one who
                did not play scored 0) against the real team and game fantasy
                totals, which include players the simulation did not list.
                Player ranges are scored on players with a stat line. Used for
                the display gate.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.published_record import (
    POSITIONS, PROJECT_ROOT, git_versions, json_at, player_actuals, snapshot_before,
)

UNBLENDED_CENTRE = "predicted_points_model"
KINDS = ("player", "stack", "team_total", "game_total")


@dataclass
class Reconstructed:
    commit: str
    games: pd.DataFrame
    rows: pd.DataFrame                         # simulated players, simulate_week order
    draws: dict = field(default_factory=dict)  # game_id -> {"role_factor": m, "independent": m}
    max_abs_mean_error: float = 0.0


def interval_hits(samples: np.ndarray, observation: float) -> tuple[float, float]:
    q = np.quantile(samples, [.1, .25, .75, .9])
    return float(q[1] <= observation <= q[2]), float(q[0] <= observation <= q[3])


def reconstruct(commit: str, season: int, week: int, *, artifacts_root: Path,
                repo: Path = PROJECT_ROOT, tolerance: float = 1e-6) -> Reconstructed | None:
    """Rebuild the draws of the simulation file at `commit`; None if it is not the unblended-centred kind."""
    import scripts.generate_simulation_data as generate
    from src.models.calibrated_simulation import apply_dependence, game_seed, load_simulation_artifacts
    from src.models.residual_calibration import independent_residual_matrix

    sim = json_at(commit, f"docs/data/simulation_{season}_wk{week}.json", repo)
    if not sim or not str(sim.get("model_config", {}).get("centre", "")).startswith(UNBLENDED_CENTRE):
        return None
    cfg = sim["model_config"]
    artifacts = load_simulation_artifacts(Path(artifacts_root) / cfg["artifacts_run_id"])
    weekly = pd.DataFrame(json_at(commit, f"docs/data/weekly_{season}_wk{week}.json", repo))
    games = pd.DataFrame(json_at(commit, f"docs/data/game_predictions_{season}_wk{week}.json", repo))
    if "source" in games:
        games = games[games["source"] == "live"]
    if "source" in weekly:
        weekly = weekly[weekly["source"] != "published"]
    games, rows, _ = generate.build_week_inputs(games, weekly, season=season, week=week,
                                                known_player_ids=artifacts.known_player_ids)
    published = {(p["game_id"], p["player_id"]): p["mean"] for p in sim["player_summary"]}
    out = Reconstructed(commit=commit, games=games, rows=rows)
    seed, n_draws = int(sim["seed"]), int(sim["draws_per_player"])
    for game_id, g in rows.groupby("game_id", sort=True):
        pred = g["predicted_points"].to_numpy(float)
        base = game_seed(seed, str(game_id))
        indep = independent_residual_matrix(g, artifacts.calibration, n_draws=n_draws, seed=base)
        role = pred + apply_dependence(indep, tuple(g["role_key"]), artifacts.dependence, base + 1)
        means = np.array([published.get((game_id, pid), np.nan) for pid in g["player_id"]])
        err = float(np.nanmax(np.abs(role.mean(axis=0) - means))) if np.isfinite(means).any() else np.inf
        if not np.isfinite(means).all() or err > tolerance:
            raise ValueError(f"{commit[:8]} {game_id}: rebuilt draws do not match the published summary (max error {err})")
        out.max_abs_mean_error = max(out.max_abs_mean_error, err)
        out.draws[game_id] = {"role_factor": role, "independent": pred + indep}
    return out


def score_game(g: pd.DataFrame, draws: dict, actual: pd.Series, team_totals: dict) -> list[dict]:
    """Score one game's draws (both arms, both populations). `g` is in draw-column order."""
    from src.models.calibrated_simulation import sum_groups

    g = g.reset_index(drop=True)
    acts = g["player_id"].map(actual).to_numpy(float)
    played = np.isfinite(acts)
    game_id = g["game_id"].iloc[0]
    out = []
    for arm, m in draws.items():
        for j in np.flatnonzero(played):
            cov50, cov80 = interval_hits(m[:, j], acts[j])
            out.append(_row(game_id, arm, "played", "player", g["team"].iloc[j], m[:, j], acts[j], cov50, cov80))
        # played: membership and sums over players with a stat line only (the backtest's rule)
        sub = g[played].reset_index(drop=True)
        if not sub.empty:
            mp, ap = m[:, played], acts[played]
            for kind, side, idx in sum_groups(sub):
                team = sub["home_team"].iloc[0] if side == "home" else sub["away_team"].iloc[0] if side == "away" else "both"
                t = float(ap[idx].sum())
                out.append(_row(game_id, arm, "played", kind, team, mp[:, idx].sum(axis=1), t, *interval_hits(mp[:, idx].sum(axis=1), t)))
        # published: membership as published; non-players score 0; totals against the real team totals
        zero = np.where(played, acts, 0.0)
        for kind, side, idx in sum_groups(g):
            team = g["home_team"].iloc[0] if side == "home" else g["away_team"].iloc[0] if side == "away" else "both"
            if kind == "stack":
                t = float(zero[idx].sum())
            elif kind == "team_total":
                t = float(team_totals.get(team, np.nan))
            else:
                t = float(team_totals.get(g["home_team"].iloc[0], np.nan) + team_totals.get(g["away_team"].iloc[0], np.nan))
            if np.isfinite(t):
                s = m[:, idx].sum(axis=1)
                out.append(_row(game_id, arm, "published", kind, team, s, t, *interval_hits(s, t)))
        for j in np.flatnonzero(played):   # player ranges: same rows in both populations
            cov50, cov80 = interval_hits(m[:, j], acts[j])
            out.append(_row(game_id, arm, "published", "player", g["team"].iloc[j], m[:, j], acts[j], cov50, cov80))
    return out


def _row(game_id, arm, population, kind, team, samples, observation, cov50, cov80) -> dict:
    from src.models.simulation_evaluation import empirical_crps
    return {"game_id": game_id, "arm": arm, "population": population, "kind": kind, "team": team,
            "crps": empirical_crps(samples, observation), "cov50": cov50, "cov80": cov80,
            "median": float(np.median(samples)), "actual": float(observation)}


def score_weeks(season: int, weeks: list[int], *, con: sqlite3.Connection, artifacts_root: Path,
                repo: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, dict]:
    """Every final game of `weeks` with an unblended-centred simulation committed before its kickoff."""
    from src.evaluation.published_record import kickoff_times

    sched = kickoff_times(season, con)
    rows, report = [], {"scored_games": 0, "not_published": [], "other_centre": [], "max_abs_mean_error": 0.0}
    for week in weeks:
        wk = sched[(sched["week"] == week) & sched["final"]]
        versions = git_versions(f"docs/data/simulation_{season}_wk{week}.json", repo)
        acts = player_actuals(con, season, week)
        acts = acts[acts["position"].isin(POSITIONS)]
        actual = acts.drop_duplicates("player_id").set_index("player_id")["actual_points"]
        team_totals = acts.groupby("team")["actual_points"].sum().to_dict()
        cache: dict[str, Reconstructed | None] = {}
        for game in wk.itertuples(index=False):
            label = f"{game.away_team}@{game.home_team}"
            snap = snapshot_before(versions, game.kickoff.timestamp())
            if snap is None:
                report["not_published"].append(f"wk{week} {label}")
                continue
            if snap[0] not in cache:
                cache[snap[0]] = reconstruct(snap[0], season, week, artifacts_root=artifacts_root, repo=repo)
            rec = cache[snap[0]]
            game_id = f"{season}_{week:02d}_{game.away_team}_{game.home_team}"
            if rec is None or game_id not in rec.draws:
                report["other_centre" if rec is None else "not_published"].append(f"wk{week} {label}")
                continue
            report["max_abs_mean_error"] = max(report["max_abs_mean_error"], rec.max_abs_mean_error)
            g = rec.rows[rec.rows["game_id"] == game_id]
            for r in score_game(g, rec.draws[game_id], actual, team_totals):
                rows.append({**r, "week": week})
            report["scored_games"] += 1
    return pd.DataFrame(rows), report


def correlation_verdict(scores: pd.DataFrame) -> dict:
    """Pre-registered: stack CRPS, role factor minus independent, population 'played',
    averaged within each game, then across games; SE over games. KEEP iff mean + 1.96 SE < 0."""
    s = scores[(scores["population"] == "played") & (scores["kind"] == "stack")]
    wide = s.pivot_table(index=["game_id", "team"], columns="arm", values="crps").dropna()
    per_game = (wide["role_factor"] - wide["independent"]).groupby(level="game_id").mean()
    n = len(per_game)
    mean = float(per_game.mean()) if n else float("nan")
    se = float(per_game.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    upper = mean + 1.96 * se
    return {"n_games": n, "n_stacks": int(len(wide)), "mean_difference": round(mean, 4), "se": round(se, 4),
            "upper_95": round(upper, 4), "verdict": "KEEP" if n > 1 and upper < 0 else "REVERT"}


GATE = {"cov80": (0.72, 0.88), "cov50": (0.40, 0.60)}


def display_gate(scores: pd.DataFrame, arm: str = "role_factor") -> dict:
    """Pre-registered: population 'published'; a quantity may be shown iff both coverages are in band."""
    s = scores[(scores["population"] == "published") & (scores["arm"] == arm)]
    out = {}
    for kind in KINDS:
        k = s[s["kind"] == kind]
        if k.empty:
            out[kind] = {"n": 0, "show": False}
            continue
        c80, c50 = float(k["cov80"].mean()), float(k["cov50"].mean())
        ok = GATE["cov80"][0] <= c80 <= GATE["cov80"][1] and GATE["cov50"][0] <= c50 <= GATE["cov50"][1]
        out[kind] = {"n": int(len(k)), "cov80": round(c80, 3), "cov50": round(c50, 3),
                     "crps": round(float(k["crps"].mean()), 3), "show": bool(ok)}
    return out

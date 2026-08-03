#!/usr/bin/env python3
"""
Draft Advisor: Decision support for fantasy football drafts.

Not a draft replacement — a draft enhancement. Uses ADP as the baseline
and layers model insight on top. Three features:

1. **Spread**: Where our model disagrees with ADP (undervalued/overvalued players)
2. **Availability**: P(player available) at your draft slot picks
3. **VONA**: Value Over Next Available — opportunity cost of each pick

Usage:
    python scripts/draft_advisor.py --mode spread --season 2025
    python scripts/draft_advisor.py --mode availability --slot 8 --season 2025
    python scripts/draft_advisor.py --mode vona --slot 8 --season 2025
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.snake_draft_sim import (
    DraftPlayer,
    POSITION_CAPS,
    TEAMS,
    ROUNDS,
    build_draft_board,
    load_adp_board,
    load_model_projections,
    snake_pick_order,
)

RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"

# Asymmetric blend weights (empirically tuned on 2024+2025 backtests).
# Model is better at spotting overvalued players (fades) than finding
# hidden gems (sleepers), so trust model more on fades, defer to ADP
# on sleepers.  Flat 50/50 = 67%, asymmetric = 69%.
ADP_BLEND_FADE_MODEL_WEIGHT = 0.60   # model weight when model says LOWER than ADP
ADP_BLEND_SLEEPER_MODEL_WEIGHT = 0.30  # model weight when model says HIGHER than ADP


# ====================================================================
# Data helpers
# ====================================================================

def _latest_predictions_csv(season: int) -> Optional[Path]:
    """Find the most recent walk-forward predictions CSV for a season."""
    candidates = sorted(
        [c for c in RESULTS_DIR.glob(f"ts_backtest_{season}_*_predictions.csv")
         if "_conformal" not in c.name],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def _build_adp_yield_curve(board: List[DraftPlayer]) -> Dict[int, float]:
    """Build empirical ADP yield: for each ECR rank bucket, what's the
    median actual season total of players drafted at that rank?

    Returns {rank_bucket: median_actual_total} where rank_bucket is
    the ECR rounded to nearest 5 (e.g., ranks 1-5 -> bucket 3)."""
    data = [(int(round(p.ecr)), p.actual_total) for p in board if p.actual_total > 0]
    if not data:
        return {}
    df = pd.DataFrame(data, columns=["ecr_rank", "actual"])
    df["bucket"] = ((df["ecr_rank"] - 1) // 5) * 5 + 3  # center of 5-wide bucket
    return df.groupby("bucket")["actual"].median().to_dict()


def _adp_implied_total(ecr: float, yield_curve: Dict[int, float]) -> float:
    """Look up the expected actual total for a player at a given ECR."""
    bucket = int(((ecr - 1) // 5) * 5 + 3)
    # Try exact bucket, then neighbors
    for offset in [0, 5, -5, 10, -10]:
        val = yield_curve.get(bucket + offset)
        if val is not None:
            return val
    return 0.0


# ====================================================================
# Phase 1: Model-ADP Spread
# ====================================================================

@dataclass
class SpreadResult:
    player_id: str
    name: str
    position: str
    team: str
    ecr: float
    model_rank: int
    rank_spread: int  # positive = model says undervalued
    model_projection: float
    adp_implied: float
    blended_projection: float
    actual_total: float
    model_error: float
    adp_error: float
    blended_error: float
    model_wins: bool
    blended_wins: bool  # blended closer than both model and ADP


def compute_spread(board: List[DraftPlayer]) -> List[SpreadResult]:
    """Compute the rank spread between model and ADP for every player."""
    yield_curve = _build_adp_yield_curve(board)

    # Rank players by model projection (descending)
    modelable = [p for p in board if p.is_modelable and p.model_rank_value > 0]
    model_sorted = sorted(modelable, key=lambda p: -p.model_rank_value)
    model_rank_map = {p.name: i + 1 for i, p in enumerate(model_sorted)}

    # ADP rank is just position in board (already sorted by ECR)
    adp_rank_map = {p.name: i + 1 for i, p in enumerate(board)}

    results = []
    for p in board:
        if not p.is_modelable or p.model_rank_value <= 0:
            continue

        adp_rank = adp_rank_map[p.name]
        model_rank = model_rank_map.get(p.name, 999)
        rank_spread = adp_rank - model_rank  # positive = model likes more

        adp_implied = _adp_implied_total(p.ecr, yield_curve)
        model_proj = p.pred_total
        actual = p.actual_total

        # Asymmetric blend: trust model more on fades, defer to ADP on sleepers
        model_weight = (
            ADP_BLEND_FADE_MODEL_WEIGHT if rank_spread < 0
            else ADP_BLEND_SLEEPER_MODEL_WEIGHT
        )
        blended = model_weight * model_proj + (1 - model_weight) * adp_implied

        model_error = abs(model_proj - actual) if actual > 0 else float("inf")
        adp_error = abs(adp_implied - actual) if actual > 0 and adp_implied > 0 else float("inf")
        blended_error = abs(blended - actual) if actual > 0 and adp_implied > 0 else float("inf")
        model_wins = model_error < adp_error
        blended_wins = blended_error < adp_error

        results.append(SpreadResult(
            player_id=p.player_id,
            name=p.name,
            position=p.position,
            team=p.team,
            ecr=p.ecr,
            model_rank=model_rank,
            rank_spread=rank_spread,
            model_projection=model_proj,
            adp_implied=adp_implied,
            blended_projection=blended,
            actual_total=actual,
            model_error=model_error,
            adp_error=adp_error,
            blended_error=blended_error,
            model_wins=model_wins,
            blended_wins=blended_wins,
        ))

    results.sort(key=lambda r: -abs(r.rank_spread))
    return results


def validate_spread_direction(
    results: List[SpreadResult], min_spread: int = 10
) -> Dict:
    """For players where |rank_spread| > min_spread and actuals exist,
    how often is the model closer to reality than ADP?"""
    eligible = [
        r for r in results
        if abs(r.rank_spread) >= min_spread
        and r.actual_total > 0
        and r.adp_implied > 0
    ]
    if not eligible:
        return {"n": 0, "model_wins": 0, "accuracy": 0.0}

    wins = sum(1 for r in eligible if r.model_wins)
    blended_wins = sum(1 for r in eligible if r.blended_wins)
    return {
        "n": len(eligible),
        "model_wins": wins,
        "adp_wins": len(eligible) - wins,
        "accuracy": wins / len(eligible),
        "blended_wins": blended_wins,
        "blended_accuracy": blended_wins / len(eligible),
    }


# ====================================================================
# Phase 2: Availability Modeling
# ====================================================================

def player_availability(ecr: float, sd: float, pick: int) -> float:
    """P(player still available at overall pick number)."""
    sd = max(sd, 0.5)  # floor to avoid degenerate cases
    return float(1.0 - norm.cdf(pick, loc=ecr, scale=sd))


def draft_slot_availability(
    slot: int, board: List[DraftPlayer], adp_df: pd.DataFrame,
    teams: int = TEAMS, rounds: int = ROUNDS,
) -> pd.DataFrame:
    """For each of the user's picks in the snake, compute P(available)
    for every player on the board."""
    # Get user's pick numbers
    order = snake_pick_order(teams, rounds)
    my_picks = [i + 1 for i, s in enumerate(order) if s == slot]

    # Build sd lookup from ADP DataFrame
    sd_map = {}
    for _, r in adp_df.iterrows():
        sd_map[r["name"]] = float(r.get("sd") or 5.0)

    rows = []
    for p in board:
        if not p.is_modelable:
            continue
        sd = sd_map.get(p.name, 5.0)
        row = {
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "ecr": p.ecr,
            "sd": sd,
            "model_proj": p.pred_total,
        }
        for pick_num in my_picks:
            row[f"pick_{pick_num}"] = player_availability(p.ecr, sd, pick_num)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_sd_map(adp_df: pd.DataFrame) -> Dict[str, float]:
    """Build {player_name: sd} from ADP data, with NaN handling."""
    sd_map: Dict[str, float] = {}
    for _, r in adp_df.iterrows():
        sd_val = r.get("sd")
        sd_map[r["name"]] = max(float(sd_val) if pd.notna(sd_val) else 5.0, 0.5)
    return sd_map


def run_noisy_drafts(
    board: List[DraftPlayer],
    adp_df: pd.DataFrame,
    n_sims: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Run n_sims noisy drafts tracking player availability at each pick.

    Each sim: 12 ADPBots, each independently perceiving players via
    ecr + N(0, sd).  Position caps enforced.

    Returns array of shape (n_players, total_picks) with counts of
    how many sims each player was still available at each pick.
    """
    n_players = len(board)
    total_picks = TEAMS * ROUNDS
    order = snake_pick_order(TEAMS, ROUNDS)

    sd_map = _build_sd_map(adp_df)
    ecr_arr = np.array([p.ecr for p in board])
    sd_arr = np.array([sd_map.get(p.name, 5.0) for p in board])

    pos_to_idx = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    pos_idx_arr = np.array([pos_to_idx.get(p.position, 0) for p in board])
    pos_caps_arr = np.array([POSITION_CAPS[p] for p in ["QB", "RB", "WR", "TE"]])

    rng = np.random.default_rng(seed)
    avail_counts = np.zeros((n_players, total_picks), dtype=np.int32)

    for sim_idx in range(n_sims):
        # Per-bot noisy ECR: each bot independently perceives players
        noise = rng.normal(0, 1, size=(TEAMS, n_players)) * sd_arr
        noisy_ecr = ecr_arr + noise  # (TEAMS, n_players)

        available = np.ones(n_players, dtype=bool)
        team_pos = np.zeros((TEAMS, 4), dtype=np.int16)

        for pick_idx in range(total_picks):
            avail_counts[available, pick_idx] += 1

            bot_idx = order[pick_idx] - 1
            team_caps = team_pos[bot_idx]
            cap_ok = team_caps[pos_idx_arr] < pos_caps_arr[pos_idx_arr]
            eligible = available & cap_ok
            if not eligible.any():
                eligible = available.copy()
            if not eligible.any():
                continue

            masked = np.where(eligible, noisy_ecr[bot_idx], np.inf)
            best = int(np.argmin(masked))
            available[best] = False
            team_pos[bot_idx, pos_idx_arr[best]] += 1

        if (sim_idx + 1) % 200 == 0:
            print(f"    {sim_idx + 1}/{n_sims} sims complete...")

    return avail_counts


def validate_availability_calibration(
    board: List[DraftPlayer],
    adp_df: pd.DataFrame,
    n_sims: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float]:
    """Validate normal-CDF availability model against noisy draft sims.

    Samples (predicted, empirical) pairs for each player at picks near
    their ECR (the transition zone where availability matters), then
    bins into deciles and reports calibration gaps.

    Returns (calibration_df, max_abs_gap_pp).
    """
    print(f"  Running {n_sims} noisy draft simulations...")
    avail_counts = run_noisy_drafts(board, adp_df, n_sims=n_sims, seed=seed)

    sd_map = _build_sd_map(adp_df)
    total_picks = TEAMS * ROUNDS

    rows = []
    for p_idx, player in enumerate(board):
        sd = sd_map.get(player.name, 5.0)
        # Sample picks in [ecr - 3*sd, ecr + 3*sd] — the transition zone
        lo = max(1, int(player.ecr - 3 * sd))
        hi = min(total_picks, int(player.ecr + 3 * sd))
        for pick_num in range(lo, hi + 1):
            emp = avail_counts[p_idx, pick_num - 1] / n_sims
            pred = player_availability(player.ecr, sd, pick_num)
            rows.append({"predicted": pred, "empirical": emp})

    df = pd.DataFrame(rows)
    bins = np.arange(0, 1.1, 0.1)
    labels = [f"{int(b * 100)}-{int(bins[i + 1] * 100)}%"
              for i, b in enumerate(bins[:-1])]
    df["decile"] = pd.cut(df["predicted"], bins=bins, labels=labels,
                          include_lowest=True)

    cal = (
        df.groupby("decile", observed=True)
        .agg(
            mean_predicted=("predicted", "mean"),
            mean_empirical=("empirical", "mean"),
            n=("predicted", "count"),
        )
        .reset_index()
    )
    cal["gap_pp"] = ((cal["mean_predicted"] - cal["mean_empirical"]) * 100
                     ).round(1)
    max_gap = float(cal["gap_pp"].abs().max())

    return cal, max_gap


# ====================================================================
# Phase 3: VONA (Value Over Next Available)
# ====================================================================

def expected_best_available(
    position: str,
    pick: int,
    board: List[DraftPlayer],
    sd_map: Dict[str, float],
    min_avail: float = 0.10,
) -> Tuple[float, str]:
    """Expected projection of the best available player at a position
    at a given pick, weighted by availability probability."""
    candidates = []
    for p in board:
        if p.position != position or not p.is_modelable:
            continue
        sd = sd_map.get(p.name, 5.0)
        avail = player_availability(p.ecr, sd, pick)
        if avail >= min_avail:
            candidates.append((p.pred_total * avail, p.pred_total, avail, p.name))

    if not candidates:
        return 0.0, "(none)"

    # Weighted by availability: E[best] ≈ max of (proj * p(avail))
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]
    return best[1], best[3]  # (projection, name)


def compute_vona(
    board: List[DraftPlayer],
    adp_df: pd.DataFrame,
    slot: int,
    teams: int = TEAMS,
    rounds: int = ROUNDS,
) -> List[Dict]:
    """Compute VONA for each candidate at each of the user's picks."""
    order = snake_pick_order(teams, rounds)
    my_picks = [i + 1 for i, s in enumerate(order) if s == slot]

    sd_map = {}
    for _, r in adp_df.iterrows():
        sd_map[r["name"]] = float(r.get("sd") or 5.0)

    results = []
    positions = ["QB", "RB", "WR", "TE"]

    for round_idx, pick_num in enumerate(my_picks):
        if round_idx >= len(my_picks) - 1:
            break  # no next pick to compare against
        next_pick = my_picks[round_idx + 1]

        # E[best available] at next pick for each position
        next_best = {}
        for pos in positions:
            proj, name = expected_best_available(pos, next_pick, board, sd_map)
            next_best[pos] = {"proj": proj, "name": name}

        # E[best available] at current pick for each position (for opp cost)
        curr_best = {}
        for pos in positions:
            proj, name = expected_best_available(pos, pick_num, board, sd_map)
            curr_best[pos] = {"proj": proj, "name": name}

        # Score each candidate at this pick
        for p in board:
            if not p.is_modelable:
                continue
            sd = sd_map.get(p.name, 5.0)
            avail = player_availability(p.ecr, sd, pick_num)
            if avail < 0.10:
                continue

            pos = p.position
            vona = p.pred_total - next_best[pos]["proj"]

            # Opportunity cost: biggest drop-off at OTHER positions
            opp_cost = 0.0
            opp_cost_pos = ""
            for other_pos in positions:
                if other_pos == pos:
                    continue
                dropoff = curr_best[other_pos]["proj"] - next_best[other_pos]["proj"]
                if dropoff > opp_cost:
                    opp_cost = dropoff
                    opp_cost_pos = other_pos

            net_value = vona - opp_cost

            results.append({
                "round": round_idx + 1,
                "pick": pick_num,
                "name": p.name,
                "position": pos,
                "team": p.team,
                "ecr": p.ecr,
                "avail_pct": avail,
                "model_proj": p.pred_total,
                "vona": vona,
                "next_best_at_pos": next_best[pos]["name"],
                "next_best_proj": next_best[pos]["proj"],
                "opp_cost": opp_cost,
                "opp_cost_pos": opp_cost_pos,
                "net_value": net_value,
            })

    return results


def make_vona_picker(adp_df: pd.DataFrame):
    """Factory: returns a pick function for the draft simulator that
    selects players by dynamic VONA (value over next available minus
    opportunity cost at other positions).

    Uses ``model_rank_value`` (not ``pred_total``) so the picker
    automatically adapts to the ranking mode used to build the board:
    season_sum, week1, vorp, etc.

    The returned callable has signature:
        pick_fn(team, available, pick_number) -> DraftPlayer
    """
    sd_map = _build_sd_map(adp_df)
    order = snake_pick_order(TEAMS, ROUNDS)
    positions = ["QB", "RB", "WR", "TE"]

    def _eba(position: str, pick: int, available: List[DraftPlayer]
             ) -> float:
        """E[best available] at a position at a given pick, using
        model_rank_value weighted by availability probability."""
        best_score = 0.0
        best_val = 0.0
        for p in available:
            if p.position != position or not p.is_modelable:
                continue
            sd = sd_map.get(p.name, 5.0)
            avail = player_availability(p.ecr, sd, pick)
            if avail < 0.10:
                continue
            score = p.model_rank_value * avail
            if score > best_score:
                best_score = score
                best_val = p.model_rank_value
        return best_val

    def _pick(team, available, pick_number):
        # Find this team's next pick
        future = [i + 1 for i, s in enumerate(order)
                  if s == team.slot and i + 1 > pick_number]

        eligible = [p for p in available if team.can_add(p.position)]
        if not eligible:
            eligible = list(available)
        if not eligible:
            return None

        # Last pick for this team — just take best projection
        if not future:
            modelable = [p for p in eligible if p.is_modelable]
            if modelable:
                return max(modelable,
                           key=lambda p: p.model_rank_value)
            return min(eligible, key=lambda p: p.ecr)

        next_pick = future[0]

        # E[best available] at next pick and current pick per position
        next_best = {}
        curr_best = {}
        for pos in positions:
            next_best[pos] = _eba(pos, next_pick, available)
            curr_best[pos] = _eba(pos, pick_number, available)

        # Score each eligible candidate by net value
        best_player = None
        best_net = float("-inf")
        for p in eligible:
            if not p.is_modelable:
                continue
            vona = p.model_rank_value - next_best.get(p.position, 0)
            opp_cost = 0.0
            for other_pos in positions:
                if other_pos == p.position:
                    continue
                dropoff = (curr_best.get(other_pos, 0)
                           - next_best.get(other_pos, 0))
                if dropoff > opp_cost:
                    opp_cost = dropoff
            net = vona - opp_cost
            if net > best_net:
                best_net = net
                best_player = p

        # Fallback to best ECR if no modelable candidates scored
        if best_player is None:
            return min(eligible, key=lambda p: p.ecr)
        return best_player

    return _pick


# ====================================================================
# CLI + Display
# ====================================================================

def print_spread(results: List[SpreadResult], season: int, n: int = 15):
    """Print the biggest model-ADP disagreements."""
    undervalued = [r for r in results if r.rank_spread > 0]
    overvalued = [r for r in results if r.rank_spread < 0]
    undervalued.sort(key=lambda r: -r.rank_spread)
    overvalued.sort(key=lambda r: r.rank_spread)

    print(f"\n{'='*75}")
    print(f"  DRAFT ADVISOR: Model-ADP Spread ({season})")
    print(f"{'='*75}")

    print(f"\n  UNDERVALUED (model ranks higher than ADP):")
    print(f"  {'Player':<25s} {'Pos':>3s} {'ADP':>5s} {'Model':>5s} {'Spread':>7s} {'ModelProj':>9s} {'Actual':>7s}")
    print(f"  {'-'*25} {'-'*3} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*7}")
    for r in undervalued[:n]:
        print(f"  {r.name:<25s} {r.position:>3s} {r.ecr:>5.0f} {r.model_rank:>5d} {r.rank_spread:>+7d} {r.model_projection:>9.1f} {r.actual_total:>7.1f}")

    print(f"\n  OVERVALUED (ADP ranks higher than model):")
    print(f"  {'Player':<25s} {'Pos':>3s} {'ADP':>5s} {'Model':>5s} {'Spread':>7s} {'ModelProj':>9s} {'Actual':>7s}")
    print(f"  {'-'*25} {'-'*3} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*7}")
    for r in overvalued[:n]:
        print(f"  {r.name:<25s} {r.position:>3s} {r.ecr:>5.0f} {r.model_rank:>5d} {r.rank_spread:>+7d} {r.model_projection:>9.1f} {r.actual_total:>7.1f}")

    # Validation
    for threshold in [5, 10, 15, 20]:
        v = validate_spread_direction(results, min_spread=threshold)
        if v["n"] > 0:
            print(f"\n  Validation (|spread| >= {threshold}): "
                  f"{v['model_wins']}-{v['adp_wins']} "
                  f"({v['accuracy']:.1%} model direction accuracy, n={v['n']})")


def print_availability(avail_df: pd.DataFrame, slot: int, season: int, top_n: int = 20):
    """Print availability at the user's first few picks."""
    order = snake_pick_order(TEAMS, ROUNDS)
    my_picks = [i + 1 for i, s in enumerate(order) if s == slot]

    print(f"\n{'='*75}")
    print(f"  DRAFT ADVISOR: Availability at Slot {slot} ({season})")
    print(f"  Your picks: {my_picks[:7]}...")
    print(f"{'='*75}")

    for pick_num in my_picks[:5]:  # first 5 rounds
        round_num = my_picks.index(pick_num) + 1
        col = f"pick_{pick_num}"
        if col not in avail_df.columns:
            continue

        # Sort by availability * model projection (most interesting first)
        subset = avail_df[avail_df[col] > 0.05].copy()
        subset["score"] = subset[col] * subset["model_proj"]
        subset = subset.sort_values("score", ascending=False).head(top_n)

        print(f"\n  Round {round_num} (Pick {pick_num}):")
        print(f"  {'Player':<25s} {'Pos':>3s} {'ECR':>5s} {'SD':>4s} {'P(Avail)':>9s} {'ModelProj':>9s}")
        print(f"  {'-'*25} {'-'*3} {'-'*5} {'-'*4} {'-'*9} {'-'*9}")
        for _, r in subset.iterrows():
            pct = r[col]
            print(f"  {r['name']:<25s} {r['position']:>3s} {r['ecr']:>5.1f} {r['sd']:>4.1f} {pct:>8.0%} {r['model_proj']:>9.1f}")


def print_calibration(cal: pd.DataFrame, max_gap: float, season: int, n_sims: int):
    """Print availability calibration results."""
    print(f"\n{'='*75}")
    print(f"  AVAILABILITY CALIBRATION ({season}, {n_sims} sims)")
    print(f"{'='*75}")
    print(f"\n  {'Decile':<10s} {'Predicted':>10s} {'Empirical':>10s} {'Gap(pp)':>8s} {'N':>8s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for _, r in cal.iterrows():
        print(f"  {r['decile']:<10s} {r['mean_predicted']:>9.1%}"
              f" {r['mean_empirical']:>9.1%}"
              f" {r['gap_pp']:>+7.1f} {int(r['n']):>8d}")

    verdict = "PASS" if max_gap <= 10 else "FAIL"
    print(f"\n  Max |gap|: {max_gap:.1f}pp — {verdict} (target: <=10pp)")


def print_vona(vona_results: List[Dict], slot: int, season: int, top_n: int = 5):
    """Print VONA recommendations per round."""
    print(f"\n{'='*75}")
    print(f"  DRAFT ADVISOR: VONA at Slot {slot} ({season})")
    print(f"{'='*75}")

    df = pd.DataFrame(vona_results)
    if df.empty:
        print("  No VONA data available.")
        return

    for round_num in sorted(df["round"].unique())[:7]:  # first 7 rounds
        round_data = df[df["round"] == round_num].sort_values("net_value", ascending=False)
        pick_num = round_data.iloc[0]["pick"]
        top = round_data.head(top_n)

        print(f"\n  Round {round_num} (Pick {int(pick_num)}):")
        print(f"  {'Player':<22s} {'Pos':>3s} {'Avail':>6s} {'Proj':>7s} {'VONA':>6s} {'OppCost':>8s} {'Net':>7s} {'NextBest':>20s}")
        print(f"  {'-'*22} {'-'*3} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*20}")
        for _, r in top.iterrows():
            rec = " <--" if r["net_value"] == top.iloc[0]["net_value"] else ""
            print(f"  {r['name']:<22s} {r['position']:>3s} {r['avail_pct']:>5.0%} {r['model_proj']:>7.1f} "
                  f"{r['vona']:>+6.1f} {r['opp_cost']:>+8.1f}({r['opp_cost_pos']}) {r['net_value']:>+7.1f}{rec}")


def main():
    parser = argparse.ArgumentParser(
        description="Draft Advisor: decision support for fantasy drafts",
    )
    parser.add_argument("--mode", "-m",
                        choices=["spread", "availability", "validate-avail",
                                 "vona", "all"],
                        default="spread", help="Which advisor feature to run")
    parser.add_argument("--season", "-s", type=int, default=2025)
    parser.add_argument("--slot", type=int, default=6, help="Your draft slot (1-12)")
    parser.add_argument("--ranking", default="week1",
                        help="Model ranking mode: week1 (pre-draft), season_sum (hindsight)")
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="Number of noisy draft sims for availability validation")
    args = parser.parse_args()

    # Load data
    csv_path = Path(args.predictions_csv) if args.predictions_csv else _latest_predictions_csv(args.season)
    if not csv_path or not csv_path.exists():
        print(f"No predictions CSV found for season {args.season}", file=sys.stderr)
        return 1

    print(f"Loading ADP board for {args.season}...")
    adp_df = load_adp_board(args.season)
    print(f"  {len(adp_df)} players, scrape date: {adp_df['scrape_date'].iloc[0]}")

    print(f"Loading model projections ({args.ranking} mode) from {csv_path.name}...")
    projections = load_model_projections(csv_path, ranking=args.ranking, season=args.season)
    print(f"  {len(projections)} players with projections")

    board = build_draft_board(adp_df, projections)
    matched = sum(1 for p in board if p.is_modelable)
    print(f"  Draft board: {len(board)} total, {matched} matched to projections")

    # Run requested mode(s)
    run_all = args.mode == "all"

    if args.mode == "spread" or run_all:
        results = compute_spread(board)
        print_spread(results, args.season)

    if args.mode == "availability" or run_all:
        avail_df = draft_slot_availability(args.slot, board, adp_df)
        print_availability(avail_df, args.slot, args.season)

    if args.mode == "validate-avail" or run_all:
        cal, max_gap = validate_availability_calibration(
            board, adp_df, n_sims=args.n_sims,
        )
        print_calibration(cal, max_gap, args.season, args.n_sims)

    if args.mode == "vona" or run_all:
        vona_results = compute_vona(board, adp_df, args.slot)
        print_vona(vona_results, args.slot, args.season)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

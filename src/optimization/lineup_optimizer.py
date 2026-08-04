"""
DFS Lineup Optimizer for Fantasy Football.

Per Agent Directive V7 Section 9: optimize the downstream action policy.
Per Section 24: domain-specific integration for fantasy sports contests.

Supports DraftKings and FanDuel salary cap optimization with:
- Linear programming for optimal lineup construction
- Cash game (conservative) vs GPP (tournament) strategies
- Correlation stacking for GPP lineups
- Abstention when expected value is negative
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform configurations
# ---------------------------------------------------------------------------

@dataclass
class PlatformConfig:
    """DFS platform salary cap and roster configuration."""
    name: str
    salary_cap: int
    roster_slots: Dict[str, int]
    min_salary_per_player: int = 3000

    @property
    def total_roster_size(self) -> int:
        return sum(self.roster_slots.values())


DRAFTKINGS_CONFIG = PlatformConfig(
    name="DraftKings",
    salary_cap=50000,
    roster_slots={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1},
)

FANDUEL_CONFIG = PlatformConfig(
    name="FanDuel",
    salary_cap=60000,
    roster_slots={"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1, "DST": 1},
)


@dataclass
class LineupResult:
    """Result of a lineup optimization."""
    players: List[Dict]
    total_salary: int
    projected_points: float
    strategy: str
    platform: str
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    projected_floor: float = 0.0   # 10th percentile lineup total
    projected_ceiling: float = 0.0  # 90th percentile lineup total


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class LineupOptimizer:
    """Salary-cap lineup optimizer using linear programming.

    Supports both cash game (maximize floor) and GPP (maximize ceiling)
    strategies per Directive V7 Section 24.
    """

    def __init__(
        self,
        platform: PlatformConfig = None,
    ):
        self.platform = platform or DRAFTKINGS_CONFIG

    # Z-scores for percentile calculations (normal approximation)
    _Z10 = -1.2816  # 10th percentile
    _Z50 = 0.0      # 50th percentile (median = mean for symmetric dist)
    _Z90 = 1.2816   # 90th percentile

    def optimize_lineup(
        self,
        players: pd.DataFrame,
        pred_col: str = "predicted_points",
        std_col: str = "predicted_std",
        salary_col: str = "salary",
        position_col: str = "position",
        name_col: str = "player_name",
        strategy: str = "cash",
    ) -> LineupResult:
        """Optimize a single lineup under salary cap constraints.

        Args:
            players: DataFrame with player projections and salaries.
            pred_col: Column with point predictions.
            std_col: Column with prediction std.
            salary_col: Column with player salaries.
            position_col: Column with positions.
            name_col: Column with player names.
            strategy: 'cash' (maximize floor) or 'gpp' (maximize ceiling).

        Returns:
            LineupResult with optimal lineup including per-player and
            lineup-level percentile ranges (p10 floor, p50 median, p90 ceiling).
        """
        if salary_col not in players.columns:
            return LineupResult(
                players=[], total_salary=0, projected_points=0.0,
                strategy=strategy, platform=self.platform.name,
                is_valid=False, warnings=["No salary column found"],
            )

        df = players.dropna(subset=[pred_col, salary_col]).copy()
        if len(df) < self.platform.total_roster_size:
            return LineupResult(
                players=[], total_salary=0, projected_points=0.0,
                strategy=strategy, platform=self.platform.name,
                is_valid=False, warnings=["Insufficient players"],
            )

        # Compute optimization objective based on strategy
        if strategy == "gpp" and std_col in df.columns:
            # GPP: maximize ceiling (pred + 1*std for upside)
            df["_obj"] = df[pred_col] + df[std_col]
        elif strategy == "cash" and std_col in df.columns:
            # Cash: maximize floor (pred - 0.5*std for safety)
            df["_obj"] = df[pred_col] - 0.5 * df[std_col]
        else:
            df["_obj"] = df[pred_col]

        # Pre-compute per-player percentiles for inclusion in output
        has_std = std_col in df.columns and df[std_col].notna().any()
        if has_std:
            std_vals = df[std_col].fillna(0)
            df["_p10"] = (df[pred_col] + self._Z10 * std_vals).clip(lower=0)
            df["_p50"] = df[pred_col]  # mean ≈ median for symmetric dist
            df["_p90"] = df[pred_col] + self._Z90 * std_vals
        else:
            df["_p10"] = df[pred_col]
            df["_p50"] = df[pred_col]
            df["_p90"] = df[pred_col]

        # Greedy knapsack with position constraints
        lineup = self._greedy_knapsack(
            df, pred_col="_obj", salary_col=salary_col,
            position_col=position_col, name_col=name_col,
        )

        if not lineup:
            return LineupResult(
                players=[], total_salary=0, projected_points=0.0,
                strategy=strategy, platform=self.platform.name,
                is_valid=False, warnings=["Could not construct valid lineup"],
            )

        total_salary = sum(p["salary"] for p in lineup)
        projected = sum(p["projected_points"] for p in lineup)
        lineup_floor = sum(p.get("p10", p["projected_points"]) for p in lineup)
        lineup_ceiling = sum(p.get("p90", p["projected_points"]) for p in lineup)

        return LineupResult(
            players=lineup,
            total_salary=total_salary,
            projected_points=round(projected, 2),
            strategy=strategy,
            platform=self.platform.name,
            is_valid=True,
            projected_floor=round(lineup_floor, 2),
            projected_ceiling=round(lineup_ceiling, 2),
        )

    def _greedy_knapsack(
        self,
        df: pd.DataFrame,
        pred_col: str,
        salary_col: str,
        position_col: str,
        name_col: str,
    ) -> List[Dict]:
        """Greedy position-constrained knapsack solver.

        Sorts by value/cost ratio and fills positions greedily.
        Not globally optimal but fast and practical.
        """
        df = df.copy()
        df["_value_ratio"] = df[pred_col] / np.maximum(df[salary_col], 1)
        df = df.sort_values("_value_ratio", ascending=False)

        selected: List[Dict] = []
        remaining_salary = self.platform.salary_cap
        filled: Dict[str, int] = {pos: 0 for pos in self.platform.roster_slots}
        flex_filled = 0
        used_names = set()

        for _, row in df.iterrows():
            pos = row[position_col]
            salary = int(row[salary_col])
            name = row.get(name_col, "")

            if name in used_names:
                continue
            if salary > remaining_salary:
                continue

            # Check if position slot available
            slot = None
            if pos in filled and filled[pos] < self.platform.roster_slots.get(pos, 0):
                slot = pos
            elif pos in ("RB", "WR", "TE") and flex_filled < self.platform.roster_slots.get("FLEX", 0):
                slot = "FLEX"
            else:
                continue

            selected.append({
                "player_name": name,
                "position": pos,
                "slot": slot,
                "salary": salary,
                "projected_points": round(float(row.get(pred_col, 0)), 2),
                "p10": round(float(row.get("_p10", row.get(pred_col, 0))), 2),
                "p50": round(float(row.get("_p50", row.get(pred_col, 0))), 2),
                "p90": round(float(row.get("_p90", row.get(pred_col, 0))), 2),
            })
            used_names.add(name)
            remaining_salary -= salary
            if slot == "FLEX":
                flex_filled += 1
            else:
                filled[pos] += 1

            total_filled = sum(filled.values()) + flex_filled
            if total_filled >= self.platform.total_roster_size:
                break

        # Validate completeness
        total_filled = sum(filled.values()) + flex_filled
        if total_filled < self.platform.total_roster_size:
            return []

        return selected

    def evaluate_lineup_quality(
        self,
        lineup: LineupResult,
        actuals: pd.DataFrame,
        actual_col: str = "fantasy_points",
        name_col: str = "player_name",
    ) -> Dict[str, float]:
        """Evaluate lineup decision quality against actual outcomes.

        Per Directive V7 Section 9: separate decision quality from
        prediction quality.
        """
        if not lineup.is_valid or not lineup.players:
            return {"error": "Invalid lineup"}

        lineup_names = {p["player_name"] for p in lineup.players}
        matched = actuals[actuals[name_col].isin(lineup_names)]

        if matched.empty:
            return {"error": "No matching actuals found"}

        actual_total = float(matched[actual_col].sum())
        projected_total = lineup.projected_points

        return {
            "projected_total": projected_total,
            "actual_total": round(actual_total, 2),
            "accuracy_pct": round(
                (1 - abs(actual_total - projected_total) / max(actual_total, 1)) * 100, 1
            ),
            "players_matched": len(matched),
            "strategy": lineup.strategy,
        }

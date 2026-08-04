"""
ESPN Fantasy Sync + Normalization Layer

Bridges ESPNFantasyConnector to the internal projection system.
Resolves ESPN players to canonical GSIS IDs, attaches model projections,
and produces a LeagueState object for the optimizer/decision engine.

PRIVACY: No credentials or user data are persisted to disk.
All ESPN data is ephemeral (session-only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from src.data.entity_resolver import EntityResolver, TEAM_CODE_ALIASES
from src.integrations.espn_fantasy import ESPNFantasyConnector
from src.utils.database import DatabaseManager

logger = logging.getLogger(__name__)

# ESPN slot position → internal slot name
SLOT_MAP = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "RB/WR/TE": "FLEX",
    "RB/WR": "FLEX",
    "WR/TE": "FLEX",
    "OP": "SUPERFLEX",
    "BE": "BENCH",
    "IR": "IR",
    "K": "K",
    "D/ST": "DST",
}

# Default roster slot counts (standard ESPN league)
DEFAULT_ROSTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "FLEX": 1,
    "BENCH": 6,
}


@dataclass
class PlayerState:
    """Canonical player representation for the optimizer."""

    player_id: str
    name: str
    team: str
    position: str
    slot: str
    proj_points: float
    espn_proj_points: float
    confidence: float
    volatility: float
    injury_status: str
    percent_owned: float
    edge: float

    def is_resolved(self) -> bool:
        return self.player_id != ""


@dataclass
class LeagueSettings:
    """League configuration for the optimizer."""

    scoring_format: str
    roster_slots: Dict[str, int]
    num_teams: int
    league_name: str


@dataclass
class OpponentState:
    """Opponent roster and projected total."""

    team_name: str
    roster: List[PlayerState]
    projected_total: float


@dataclass
class LeagueState:
    """Complete league state — single input to the optimizer/decision engine."""

    roster: List[PlayerState]
    starters: List[PlayerState]
    bench: List[PlayerState]
    free_agents: List[PlayerState]
    opponent: Optional[OpponentState]
    settings: LeagueSettings
    week: int
    season: int
    sync_timestamp: str
    unresolved_players: List[Dict] = field(default_factory=list)

    @property
    def roster_projected_total(self) -> float:
        return sum(p.proj_points for p in self.starters)

    @property
    def resolution_rate(self) -> float:
        total = len(self.roster)
        if total == 0:
            return 0.0
        resolved = sum(1 for p in self.roster if p.is_resolved())
        return resolved / total


class ESPNSyncService:
    """
    Sync ESPN league data with internal projection system.

    Usage:
        connector = ESPNFantasyConnector(league_id=123, year=2025, espn_s2=..., swid=...)
        connector.connect()

        sync = ESPNSyncService(connector)
        state = sync.sync(team_name="My Team")
    """

    def __init__(
        self,
        connector: ESPNFantasyConnector,
        predictor=None,
        scoring_format: Optional[str] = None,
    ):
        self.connector = connector
        self.predictor = predictor
        self._scoring_format_override = scoring_format
        self._resolver = EntityResolver()
        self._db = DatabaseManager()
        self._player_lookup: Optional[Dict] = None
        self._predictions: Optional[pd.DataFrame] = None

    def sync(
        self,
        team_id: Optional[int] = None,
        team_name: Optional[str] = None,
    ) -> LeagueState:
        """
        Full sync: resolve roster, attach projections, get opponent.

        Args:
            team_id: ESPN team ID (1-indexed)
            team_name: Team name (partial match)

        Returns:
            LeagueState ready for optimizer consumption
        """
        if not self.connector.connected:
            raise RuntimeError("ESPN connector not connected. Call connector.connect() first.")

        # Build player lookup from DB
        self._build_player_lookup()

        # Load model predictions if predictor available
        if self.predictor is not None:
            self._load_predictions()

        # Get user team data
        team_data = self.connector.get_my_team(team_name=team_name, team_id=team_id)
        if "error" in team_data or not team_data.get("roster"):
            raise ValueError(f"Could not load team: {team_data.get('error', 'empty roster')}")

        # Resolve roster players
        roster = self._resolve_players(team_data["roster"], is_roster=True)

        # Split starters vs bench
        starters = [p for p in roster if p.slot not in ("BENCH", "IR")]
        bench = [p for p in roster if p.slot in ("BENCH", "IR")]

        # Get free agents
        free_agents = self._sync_free_agents()

        # Extract settings
        settings = self._extract_settings()

        # Get opponent
        opponent = self._get_opponent(team_data.get("team_id"))

        # Get week/season
        league_info = self.connector.get_league_info()
        week = league_info.get("current_week", 1)
        season = self.connector.year

        # Collect unresolved
        unresolved = [
            {"name": p.name, "team": p.team, "position": p.position, "slot": p.slot}
            for p in roster
            if not p.is_resolved()
        ]

        state = LeagueState(
            roster=roster,
            starters=starters,
            bench=bench,
            free_agents=free_agents,
            opponent=opponent,
            settings=settings,
            week=week,
            season=season,
            sync_timestamp=datetime.now(timezone.utc).isoformat(),
            unresolved_players=unresolved,
        )

        if unresolved:
            logger.warning(
                "ESPN sync: %d/%d players unresolved: %s",
                len(unresolved),
                len(roster),
                [u["name"] for u in unresolved[:5]],
            )

        return state

    def _build_player_lookup(self) -> None:
        """Build name+team → player_id lookup from DB."""
        if self._player_lookup is not None:
            return

        players_df = self._db.get_all_players()
        if players_df is None or players_df.empty:
            self._player_lookup = {}
            return

        lookup = {}
        for _, row in players_df.iterrows():
            pid = str(row.get("player_id", "")).strip()
            name = row.get("name", "")
            team = row.get("team", "")
            if not pid or not name:
                continue

            norm_name = self._resolver.normalize_name(name)
            norm_team = self._resolver.normalize_team_code(team)

            # Primary key: (name, team)
            lookup[(norm_name, norm_team)] = pid
            # Secondary key: name only (for trades/FA where team may differ)
            if norm_name not in lookup:
                lookup[norm_name] = pid

        self._player_lookup = lookup

    def _load_predictions(self) -> None:
        """Load model predictions for all positions."""
        if self._predictions is not None:
            return

        try:
            predictions = self.predictor.predict_next_week(top_n=300)
            if predictions is not None and not predictions.empty:
                self._predictions = predictions
            else:
                self._predictions = pd.DataFrame()
        except Exception as e:
            logger.warning("Failed to load predictions: %s", e)
            self._predictions = pd.DataFrame()

    def _resolve_players(
        self, espn_players: List[Dict], is_roster: bool = True
    ) -> List[PlayerState]:
        """Resolve ESPN player dicts to PlayerState objects."""
        results = []
        for p in espn_players:
            name = p.get("name", "")
            team = p.get("team", "")
            position = p.get("position", "")
            slot_raw = p.get("slot", "BE") if is_roster else "FA"
            espn_proj = float(p.get("projected_points", 0) or 0)
            injury = p.get("injury_status", "Active") or "Active"
            pct_owned = float(p.get("percent_owned", 0) or 0)

            # Skip K and D/ST — not in our model
            if position in ("K", "D/ST"):
                continue

            # Normalize
            norm_name = self._resolver.normalize_name(name)
            norm_team = self._resolver.normalize_team_code(team)
            slot = SLOT_MAP.get(slot_raw, slot_raw)

            # Resolve player_id
            player_id = self._resolve_player_id(norm_name, norm_team)

            # Attach projection
            proj_points, confidence, volatility = self._get_projection(
                player_id, espn_proj
            )

            edge = proj_points - espn_proj if espn_proj > 0 else 0.0

            results.append(
                PlayerState(
                    player_id=player_id,
                    name=name,
                    team=norm_team,
                    position=position,
                    slot=slot,
                    proj_points=proj_points,
                    espn_proj_points=espn_proj,
                    confidence=confidence,
                    volatility=volatility,
                    injury_status=injury,
                    percent_owned=pct_owned,
                    edge=edge,
                )
            )

        return results

    def _resolve_player_id(self, norm_name: str, norm_team: str) -> str:
        """Resolve normalized name+team to canonical player_id."""
        if self._player_lookup is None:
            return ""

        # Try exact name+team match
        pid = self._player_lookup.get((norm_name, norm_team))
        if pid:
            return pid

        # Try name-only match (handles recent trades/FA signings)
        pid = self._player_lookup.get(norm_name)
        if pid:
            return pid

        return ""

    def _get_projection(
        self, player_id: str, espn_fallback: float
    ) -> tuple[float, float, float]:
        """
        Get model projection for a player.

        Returns: (proj_points, confidence, volatility)
        Falls back to ESPN projection if player not in model output.
        """
        if not player_id or self._predictions is None or self._predictions.empty:
            return espn_fallback, 0.5, 5.0

        match = self._predictions[self._predictions["player_id"] == player_id]
        if match.empty:
            return espn_fallback, 0.5, 5.0

        row = match.iloc[0]
        proj = float(row.get("predicted_ppg", row.get("predicted_points", espn_fallback)))

        # Confidence from CI width
        ci_upper = float(row.get("prediction_ci80_upper", proj + 5))
        ci_lower = float(row.get("prediction_ci80_lower", proj - 5))
        ci_width = ci_upper - ci_lower
        volatility = ci_width / 2.0

        if proj > 0:
            confidence = max(0.0, min(1.0, 1.0 - (ci_width / (2.0 * proj))))
        else:
            confidence = 0.5

        return proj, confidence, volatility

    def _sync_free_agents(self, limit: int = 50) -> List[PlayerState]:
        """Fetch and resolve top free agents."""
        fa_list = []
        for pos in ("QB", "RB", "WR", "TE"):
            fas = self.connector.get_free_agents(position=pos, limit=limit // 4)
            fa_list.extend(fas)

        return self._resolve_players(fa_list, is_roster=False)

    def _extract_settings(self) -> LeagueSettings:
        """Extract league settings from ESPN connection."""
        league_info = self.connector.get_league_info()
        scoring_format = self._detect_scoring_format()
        roster_slots = self._detect_roster_slots()

        return LeagueSettings(
            scoring_format=scoring_format,
            roster_slots=roster_slots,
            num_teams=league_info.get("num_teams", 10),
            league_name=league_info.get("name", "Unknown League"),
        )

    def _detect_scoring_format(self) -> str:
        """Detect PPR/Half-PPR/Standard from league settings."""
        if self._scoring_format_override:
            return self._scoring_format_override

        try:
            league = self.connector.league
            if league and hasattr(league, "settings"):
                # espn_api exposes scoring_items on some versions
                scoring_items = getattr(league.settings, "scoring_items", None)
                if scoring_items:
                    for item in scoring_items:
                        # Reception scoring item has statId 53 in ESPN
                        if getattr(item, "stat_id", None) == 53:
                            pts = getattr(item, "points", 0)
                            if pts >= 1.0:
                                return "ppr"
                            elif pts >= 0.5:
                                return "half_ppr"
                            else:
                                return "standard"
        except Exception:
            pass

        # Default to PPR (most common)
        return "ppr"

    def _detect_roster_slots(self) -> Dict[str, int]:
        """Detect roster slot configuration from league."""
        try:
            league = self.connector.league
            if league and hasattr(league, "settings"):
                roster_settings = getattr(league.settings, "roster", None)
                if roster_settings and isinstance(roster_settings, dict):
                    slots = {}
                    # ESPN roster dict maps slot_id → count
                    # Common slot IDs: 0=QB, 2=RB, 4=WR, 6=TE, 23=FLEX, 20=BENCH
                    espn_slot_ids = {
                        0: "QB", 2: "RB", 4: "WR", 6: "TE",
                        23: "FLEX", 20: "BENCH", 17: "IR",
                    }
                    for slot_id, count in roster_settings.items():
                        slot_name = espn_slot_ids.get(int(slot_id))
                        if slot_name and count > 0:
                            slots[slot_name] = int(count)
                    if slots:
                        return slots
        except Exception:
            pass

        return DEFAULT_ROSTER_SLOTS.copy()

    def _get_opponent(self, team_id: Optional[int]) -> Optional[OpponentState]:
        """Get this week's opponent and their roster."""
        if team_id is None:
            return None

        try:
            league = self.connector.league
            if league is None:
                return None

            # Find current matchup for this team
            matchups = getattr(league, "box_scores", None)
            if matchups is None:
                # Try alternate access
                current_week = getattr(league, "current_week", None)
                if current_week and hasattr(league, "box_scores"):
                    matchups = league.box_scores(current_week)

            if not matchups:
                return None

            for matchup in matchups:
                home_team = getattr(matchup, "home_team", None)
                away_team = getattr(matchup, "away_team", None)

                if home_team and home_team.team_id == team_id:
                    opp_team = away_team
                elif away_team and away_team.team_id == team_id:
                    opp_team = home_team
                else:
                    continue

                if opp_team is None:
                    return None

                # Build opponent roster
                opp_roster_raw = []
                for player in opp_team.roster:
                    opp_roster_raw.append({
                        "name": player.name,
                        "position": player.position,
                        "team": player.proTeam,
                        "projected_points": getattr(player, "projected_points", 0),
                        "slot": getattr(player, "slot_position", "BE"),
                        "injury_status": getattr(player, "injuryStatus", "Active"),
                    })

                opp_players = self._resolve_players(opp_roster_raw, is_roster=True)
                opp_starters = [p for p in opp_players if p.slot not in ("BENCH", "IR")]

                return OpponentState(
                    team_name=opp_team.team_name,
                    roster=opp_players,
                    projected_total=sum(p.proj_points for p in opp_starters),
                )

        except Exception as e:
            logger.warning("Failed to get opponent: %s", e)

        return None

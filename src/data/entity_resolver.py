"""Shared entity normalization and resolution utilities for cross-source joins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


TEAM_CODE_ALIASES = {
    "JAC": "JAX",
    "LAR": "LA",
    "STL": "LA",
    "SD": "LAC",
    "OAK": "LV",
}


@dataclass
class ResolutionResult:
    """Container for resolved output and unresolved records for monitoring."""

    dataframe: pd.DataFrame
    unresolved: pd.DataFrame


class EntityResolver:
    """Normalize players/teams and generate canonical join keys."""

    @staticmethod
    def normalize_name(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        return (
            str(value)
            .lower()
            .replace(".", "")
            .replace("'", "")
            .replace("-", " ")
            .strip()
        )

    @staticmethod
    def normalize_team_code(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        code = str(value).strip().upper()
        return TEAM_CODE_ALIASES.get(code, code)

    def build_keys(
        self,
        df: pd.DataFrame,
        *,
        source: str,
        player_id_col: str = "player_id",
        name_col: str = "name",
        team_col: str = "team",
        opponent_col: Optional[str] = "opponent",
        season_col: str = "season",
        week_col: str = "week",
        game_id_col: str = "game_id",
    ) -> ResolutionResult:
        out = df.copy()
        out["entity_source"] = source

        if player_id_col in out.columns:
            out["canonical_player_id"] = out[player_id_col].fillna("").astype(str).str.strip()
        else:
            out["canonical_player_id"] = ""

        if name_col in out.columns:
            out["player_name_key"] = out[name_col].map(self.normalize_name)
        else:
            out["player_name_key"] = ""

        if team_col in out.columns:
            out["team_norm"] = out[team_col].map(self.normalize_team_code)
        else:
            out["team_norm"] = ""

        if opponent_col and opponent_col in out.columns:
            out["opponent_norm"] = out[opponent_col].map(self.normalize_team_code)
        else:
            out["opponent_norm"] = ""

        season_series = pd.to_numeric(out.get(season_col), errors="coerce")
        week_series = pd.to_numeric(out.get(week_col), errors="coerce")
        out["week_key"] = ""
        valid_week = season_series.notna() & week_series.notna()
        out.loc[valid_week, "week_key"] = (
            season_series[valid_week].astype(int).astype(str)
            + ":"
            + week_series[valid_week].astype(int).astype(str)
        )

        out["game_key"] = ""
        if game_id_col in out.columns:
            has_game_id = out[game_id_col].notna() & (out[game_id_col].astype(str).str.strip() != "")
            out.loc[has_game_id, "game_key"] = out.loc[has_game_id, game_id_col].astype(str)

        fallback_game = out["game_key"].eq("") & out["week_key"].ne("") & out["team_norm"].ne("") & out["opponent_norm"].ne("")
        if fallback_game.any():
            home = out.loc[fallback_game, ["team_norm", "opponent_norm"]].min(axis=1)
            away = out.loc[fallback_game, ["team_norm", "opponent_norm"]].max(axis=1)
            out.loc[fallback_game, "game_key"] = out.loc[fallback_game, "week_key"] + ":" + home + "_" + away

        out["resolution_status"] = "resolved"
        out["resolution_reason"] = ""

        missing_player = out["canonical_player_id"].eq("")
        if missing_player.any():
            out.loc[missing_player, ["resolution_status", "resolution_reason"]] = ["unresolved", "missing_player_id"]

        missing_team = out["team_norm"].eq("")
        if missing_team.any():
            out.loc[missing_team, ["resolution_status", "resolution_reason"]] = ["unresolved", "missing_team"]

        # Deterministic fallback for missing IDs when unique by week/team/name.
        key_cols = ["week_key", "team_norm", "player_name_key"]
        have_id = out["canonical_player_id"].ne("")
        lookup = out.loc[have_id, key_cols + ["canonical_player_id"]].drop_duplicates()
        candidate_counts = lookup.groupby(key_cols)["canonical_player_id"].nunique().reset_index(name="candidate_count")
        lookup = lookup.merge(candidate_counts, on=key_cols, how="left")

        try_fill = out["canonical_player_id"].eq("") & out["player_name_key"].ne("") & out["week_key"].ne("")
        if try_fill.any() and not lookup.empty:
            left = out.loc[try_fill, key_cols].copy()
            left["_row_id"] = left.index
            merged = left.merge(lookup, on=key_cols, how="left")
            merged = merged.set_index("_row_id")
            unique_mask = merged["candidate_count"].eq(1) & merged["canonical_player_id"].notna()
            ambiguous_mask = merged["candidate_count"].gt(1)
            if unique_mask.any():
                out.loc[merged.index[unique_mask], "canonical_player_id"] = merged.loc[unique_mask, "canonical_player_id"]
                out.loc[merged.index[unique_mask], ["resolution_status", "resolution_reason"]] = ["resolved", "fallback_name_team_week"]
            if ambiguous_mask.any():
                out.loc[merged.index[ambiguous_mask], ["resolution_status", "resolution_reason"]] = ["unresolved", "ambiguous_name"]

        unresolved = out[out["resolution_status"] != "resolved"].copy()
        return ResolutionResult(dataframe=out, unresolved=unresolved)


resolver = EntityResolver()

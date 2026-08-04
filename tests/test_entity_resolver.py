import pandas as pd

from src.data.entity_resolver import resolver


def test_entity_resolver_team_alias_and_keys_are_deterministic():
    df = pd.DataFrame(
        [
            {"player_id": "P1", "name": "D.J. Moore", "team": "JAX", "opponent": "OAK", "season": 2024, "week": 1},
            {"player_id": "P1", "name": "DJ Moore", "team": "JAC", "opponent": "LV", "season": 2024, "week": 1},
        ]
    )

    first = resolver.build_keys(df, source="t1").dataframe
    second = resolver.build_keys(df, source="t1").dataframe

    assert first["team_norm"].tolist() == ["JAC", "JAC"]
    assert first["opponent_norm"].tolist() == ["LV", "LV"]
    assert first["week_key"].tolist() == ["2024:1", "2024:1"]
    assert first["game_key"].tolist() == ["2024:1:JAC_LV", "2024:1:JAC_LV"]
    pd.testing.assert_frame_equal(first, second)


def test_entity_resolver_marks_ambiguity_and_missing_ids_as_unresolved():
    df = pd.DataFrame(
        [
            {"player_id": "", "name": "John Smith", "team": "KC", "season": 2024, "week": 1},
            {"player_id": "A", "name": "John Smith", "team": "KC", "season": 2024, "week": 1},
            {"player_id": "B", "name": "John Smith", "team": "KC", "season": 2024, "week": 1},
            {"player_id": "", "name": "No Team", "team": "", "season": 2024, "week": 1},
        ]
    )

    resolved = resolver.build_keys(df, source="amb")
    unresolved = resolved.unresolved

    reasons = set(unresolved["resolution_reason"].tolist())
    assert "ambiguous_name" in reasons
    assert "missing_team" in reasons


def test_join_success_rate_improves_with_normalized_team_codes():
    base = pd.DataFrame(
        [
            {"player_id": "P1", "name": "Alpha", "team": "JAX", "season": 2024, "week": 1},
            {"player_id": "P2", "name": "Beta", "team": "OAK", "season": 2024, "week": 1},
            {"player_id": "P3", "name": "Gamma", "team": "KC", "season": 2024, "week": 1},
        ]
    )
    ext = pd.DataFrame(
        [
            {"player_id": "P1", "team": "JAC", "season": 2024, "week": 1, "metric": 1},
            {"player_id": "P2", "team": "LV", "season": 2024, "week": 1, "metric": 2},
            {"player_id": "P3", "team": "KC", "season": 2024, "week": 1, "metric": 3},
        ]
    )

    raw = base.merge(ext, on=["player_id", "team", "season", "week"], how="left")
    raw_rate = raw["metric"].notna().mean()

    base_n = resolver.build_keys(base, source="base").dataframe
    ext_n = resolver.build_keys(ext, source="ext", name_col="player_id").dataframe
    norm = base_n.merge(
        ext_n[["canonical_player_id", "team_norm", "season", "week", "metric"]],
        left_on=["canonical_player_id", "team_norm", "season", "week"],
        right_on=["canonical_player_id", "team_norm", "season", "week"],
        how="left",
    )
    norm_rate = norm["metric"].notna().mean()

    assert raw_rate < norm_rate
    assert norm_rate == 1.0

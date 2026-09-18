import sqlite3

import pandas as pd
import pytest

from scripts.build_canonical_player_weeks import validate_panel


def _base():
    return pd.DataFrame([
        {
            "player_id":"p1","season":2025,"week":1,
            "participation_state":"confirmed_played","offense_snaps":40,
            "has_stats_row":1,"fantasy_points":12.0,
        },
        {
            "player_id":"p2","season":2025,"week":1,
            "participation_state":"confirmed_zero_snaps","offense_snaps":0,
            "has_stats_row":0,"fantasy_points":pd.NA,
        },
        {
            "player_id":"p3","season":2025,"week":1,
            "participation_state":"unknown","offense_snaps":pd.NA,
            "has_stats_row":0,"fantasy_points":pd.NA,
        },
    ])


def test_three_participation_states_validate():
    validate_panel(_base())


def test_missing_stats_row_cannot_receive_fabricated_ppr():
    df = _base()
    df.loc[df.player_id.eq("p2"), "fantasy_points"] = 0.0
    with pytest.raises(ValueError, match="fabricated fantasy_points"):
        validate_panel(df)


def test_confirmed_played_requires_positive_snaps():
    df = _base()
    df.loc[df.player_id.eq("p1"), "offense_snaps"] = 0
    with pytest.raises(ValueError, match="confirmed_played"):
        validate_panel(df)


def test_confirmed_zero_requires_exact_zero():
    df = _base()
    df.loc[df.player_id.eq("p2"), "offense_snaps"] = 1
    with pytest.raises(ValueError, match="confirmed_zero_snaps"):
        validate_panel(df)


def test_player_week_key_is_unique():
    df = pd.concat([_base(), _base().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_panel(df)

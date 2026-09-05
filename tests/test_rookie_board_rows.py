"""First-year players belong on the draft board.

The board's population is built by aggregating the previous season's weekly
stats, so a player whose first season is the upcoming one has nothing to
aggregate and never appears -- even though the season model projects him from
the draft class and PR #96 tuned that cold-start path specifically. These
tests cover the join that puts him back: who he is (draft_picks_v2 stores no
name), which id he carries, and which projection is his.
"""
import importlib.util

import pandas as pd
import pytest

from config.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "generate_draft_data.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("generate_draft_data", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _draft(**overrides):
    row = {"player_id": "MEN516487", "position": "QB", "draft_team": "LVR",
           "pfr_player_id": "MendFe00", "cfb_player_id": "fernando-mendoza-1"}
    row.update(overrides)
    return pd.DataFrame([row])


IDS = pd.DataFrame([{"pfr_id": "MendFe00", "name": "Fernando Mendoza",
                     "gsis_id": "00-0041562"}])
COMBINE = pd.DataFrame([{"pfr_id": "MendFe00",
                         "player_name": "Fernando Mendoza"}])


def test_nflverse_supplies_the_name_and_the_gsis_id(mod):
    out = mod._rookie_identities(_draft(), IDS, COMBINE)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["name"] == "F.Mendoza"
    # GSIS on the board, the draft stub kept only to look the projection up.
    assert row["player_id"] == "00-0041562"
    assert row["draft_id"] == "MEN516487"


def test_combine_names_the_picks_nflverse_misses(mod):
    out = mod._rookie_identities(_draft(), pd.DataFrame(), COMBINE)

    assert out.iloc[0]["name"] == "F.Mendoza"
    # No GSIS id available, so the draft stub has to serve as the board id.
    assert out.iloc[0]["player_id"] == "MEN516487"


def test_college_slug_is_the_last_resort(mod):
    out = mod._rookie_identities(_draft(), pd.DataFrame(), pd.DataFrame())

    assert out.iloc[0]["name"] == "F.Mendoza"


def test_a_pick_nothing_can_name_is_dropped(mod):
    """A board row reading "MEN516487" is worse than no row."""
    draft = _draft(cfb_player_id=None)

    assert mod._rookie_identities(draft, pd.DataFrame(), pd.DataFrame()).empty


def test_pfr_team_codes_become_board_team_codes(mod):
    draft = pd.concat([_draft(), _draft(player_id="X", draft_team="ARI")],
                      ignore_index=True)

    teams = list(mod._rookie_identities(draft, IDS, COMBINE)["team"])
    assert teams == ["LV", "ARI"]


@pytest.mark.parametrize("slug,expected", [
    ("carnell-tate-1", "Carnell Tate"),
    ("omar-cooper-jr-1", "Omar Cooper Jr"),
    (None, None),
    ("", None),
])
def test_slug_names(mod, slug, expected):
    assert mod._name_from_cfb_slug(slug) == expected


IDENTITIES = pd.DataFrame([
    {"draft_id": "MEN516487", "player_id": "00-0041562",
     "name": "Fernando Mendoza", "team": "LV", "position": "QB"},
    {"draft_id": "LOV121782", "player_id": "00-0041501",
     "name": "Jeremiyah Love", "team": "ARI", "position": "RB"},
])

PROJECTIONS = pd.DataFrame([
    {"player_id": "MEN516487", "pred_total": 147.0,
     "confidence_score": 0.05, "support_class": "backup"},
])


@pytest.fixture
def rookie_rows(mod, monkeypatch):
    def build(identities=IDENTITIES, projections=PROJECTIONS, known_ids=()):
        monkeypatch.setattr(mod, "_load_rookie_identities",
                            lambda season: identities)
        return mod._rookie_board_rows(2026, projections, set(known_ids))
    return build


def test_projection_lands_on_the_rookie_row(rookie_rows):
    out = rookie_rows()

    assert len(out) == 1
    row = out.iloc[0]
    assert row["player_id"] == "00-0041562"
    assert row["preseason_projection_total"] == 147.0
    assert row["preseason_confidence"] == 0.05
    assert row["preseason_support_class"] == "backup"
    # The lookup key does not belong on the board.
    assert "draft_id" not in out.columns


def test_prior_season_columns_stay_absent(rookie_rows):
    """Null, not zero: he has no last season, he did not score 0 in it."""
    out = rookie_rows()

    for col in ("ppg", "total_fp", "games_played", "risk_score"):
        assert col not in out.columns
    assert list(out.iloc[0]["key_features"]) == []
    assert dict(out.iloc[0]["feature_importance_rank"]) == {}


def test_an_unprojected_pick_is_not_added(rookie_rows):
    """Jeremiyah Love has no projection, so he gets no row."""
    out = rookie_rows()

    assert list(out["name"]) == ["Fernando Mendoza"]


def test_a_pick_already_on_the_board_is_left_alone(rookie_rows):
    """A UDFA who played last season already has a real, better row."""
    assert rookie_rows(known_ids={"00-0041562"}).empty


def test_no_draft_class_is_not_an_error(rookie_rows):
    assert rookie_rows(identities=pd.DataFrame()).empty

"""The printable page.

It has to be self-contained (no fetch, no CDN -- it gets emailed and printed),
it has to escape whatever ESPN calls a player, and it has to keep saying "no
projection" where there is none rather than printing a zero that would read as
a prediction.
"""
import importlib.util
import re

import pytest

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "render_league_report.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("render_league_report", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _report(**kw):
    base = {
        "season": 2026, "week": 1, "projection_mode": "season_prorated",
        "generated_at": "2026-09-05T23:32:45+00:00",
        "team": {"id": 1, "name": "Baby Back Gibbs"},
        "matchup": {"opponent": "Fig Newtons", "projected_total": 99.49,
                    "opponent_projected_total": 100.78, "edge": -1.29},
        "caveats": ["Every week's projection is the season total over 17."],
        "starters": [
            {"slot": "QB", "name": "Lamar Jackson", "position": "QB",
             "nfl_team": "BAL", "opponent": "IND", "points": 14.07,
             "points_source": "model", "injury_status": "ACTIVE"},
            {"slot": "K", "name": "Cameron Dicker", "position": "K",
             "nfl_team": "LAC", "opponent": None, "points": 9.94,
             "points_source": "espn", "injury_status": "QUESTIONABLE"},
        ],
        "bench": [
            {"name": "Mike Evans", "position": "WR", "points": 9.86,
             "points_source": "model", "injury_status": "OUT",
             "unavailable_reason": "OUT"},
            {"name": "Nobody Knows", "position": "WR", "points": None,
             "points_source": None, "injury_status": "ACTIVE",
             "unavailable_reason": None},
        ],
        "tough_calls": [], "waivers": [],
        "trades": {"basis": "within-position z-score", "horizon_weeks": 14,
                   "proposals": [
            {"with": "Immaculate Concepcion",
             "give": {"name": "Kenneth Walker III", "position": "RB",
                      "nfl_team": "KC", "model_ppg": 9.01, "espn_ppg": 16.06,
                      "injury_status": "ACTIVE"},
             "get": {"name": "Josh Jacobs", "position": "RB",
                     "nfl_team": "GB", "model_ppg": 12.92, "espn_ppg": 15.1,
                     "injury_status": "DAY_TO_DAY"},
             "our_gain_per_week": 3.91, "our_gain_over_horizon": 54.7,
             "their_gain_per_week_espn": 1.12, "horizon_weeks": 14,
             "lineup_change": {"in": [{"name": "Josh Jacobs", "slot": "RB"}],
                               "out": [{"name": "Kenneth Walker III",
                                        "slot": "RB"}], "moved": []}}],
                   },
        "coverage": {"roster": {"matched": 14, "players": 16,
                                "by_reason": {"position not modelled": 2}}},
    }
    base.update(kw)
    return base


def test_the_page_is_self_contained(mod):
    """It gets printed and emailed; it cannot depend on the network."""
    page = mod.render(_report())

    assert not re.findall(r"https?://", page)
    assert "<script" not in page
    assert "src=" not in page


def test_it_prints_to_one_page(mod):
    page = mod.render(_report())

    assert "@page { size: letter portrait" in page
    assert "@media print" in page
    assert "break-inside: avoid" in page


def test_player_names_are_escaped(mod):
    page = mod.render(_report(team={"id": 1, "name": "<script>alert(1)</script>"}))

    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;" in page


def test_an_unprojected_player_says_so_instead_of_showing_zero(mod):
    page = mod.render(_report())

    assert "no projection" in page
    assert re.search(r"Nobody Knows.*?>--<", page, re.S)


def test_a_borrowed_number_is_labelled_espn(mod):
    page = mod.render(_report())

    dicker = re.search(r"Cameron Dicker.*?</tr>", page, re.S).group()
    assert 'class="chip espn">ESPN<' in dicker


def test_injury_status_carries_a_label_not_just_a_colour(mod):
    page = mod.render(_report())

    assert 'class="chip critical">OUT<' in page
    assert 'class="chip warning">Q<' in page


def test_empty_sections_say_nothing_to_report(mod):
    page = mod.render(_report())

    assert page.count("nothing to report") == 2      # tough calls, waivers


def test_the_caveats_and_coverage_reach_the_footer(mod):
    page = mod.render(_report())

    footer = page[page.index("<footer>"):]
    assert "season total over 17" in footer
    assert "14 of" in footer and "position not modelled (2)" in footer


def test_the_report_directory_is_outside_the_published_tree(mod):
    """Same guarantee as the snapshot: private by location, not by discipline."""
    assert mod.REPORT_DIR.resolve().is_relative_to(ESPN_PRIVATE_DIR.resolve())
    assert (PROJECT_ROOT / "docs").resolve() not in mod.REPORT_DIR.resolve().parents


def test_a_proposal_names_the_partner_and_both_sides(mod):
    page = mod.render(_report())

    row = re.search(r"Immaculate.*?</tr>", page, re.S).group()
    assert "Kenneth Walker III" in row and "Josh Jacobs" in row
    assert "+3.9" in row and "+54.7" in row and "+1.1" in row


def test_a_proposal_says_what_it_does_to_the_lineup(mod):
    page = mod.render(_report())

    assert "Josh Jacobs starts at RB" in page
    assert "Kenneth Walker III to the bench" in page


def test_the_two_gains_are_kept_in_separate_columns(mod):
    """Ours and theirs are different currencies -- never summed, never netted."""
    page = mod.render(_report())

    header = re.search(r"<th>With</th>.*?</tr>", page, re.S).group()
    assert "You /wk" in header and "Them /wk" in header and "Over 14" in header


def test_no_workable_trade_says_so(mod):
    trades = dict(_report()["trades"], proposals=[])
    page = mod.render(_report(trades=trades))

    assert page.count("nothing to report") == 3   # tough calls, waivers, trades

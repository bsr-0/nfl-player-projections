"""The co-owner brief: its numbers come from the same rows and intervals as
the detailed review, its sentences follow those intervals, it never reads the
blinded forecasts, and its page stays plain and private."""
import importlib.util
import re

import numpy as np
import pandas as pd
import pytest

from config.settings import ESPN_PRIVATE_DIR, PROJECT_ROOT
from src.evaluation import review_brief as rb
from src.evaluation import weekly_review as wr

BRIEF = PROJECT_ROOT / "src" / "evaluation" / "review_brief.py"
SCRIPT = PROJECT_ROOT / "scripts" / "render_review_brief.py"
CHARTS = PROJECT_ROOT / "src" / "evaluation" / "review_charts.py"
TEAM = PROJECT_ROOT / "src" / "evaluation" / "team_history.py"
BLINDED = re.compile(r"predicted_points_model|plan_a|served_unblended|selection_forward|simulation_|forecasts\.csv", re.I)


def _rows(week=3, n_games=6):
    rows = []
    for g in range(n_games):
        for i in range(5):
            pred = 6.0 + 2 * i
            rows.append({"week": week, "game": f"A{g}@H{g}", "gkey": f"wk{week} A{g}@H{g}", "player_id": f"p{week}{g}{i}", "name": f"P{g}{i}",
                         "team": f"H{g}", "position": "WR", "predicted_points": pred, "rolling3": pred + 2,
                         "prediction_ci80_lower": pred - 5, "prediction_ci80_upper": pred + 5,
                         "actual_points": pred + (1.0 if i % 2 else -1.0), "final": True})
    return pd.DataFrame(rows)


def _ci(e, lo, hi):
    return {"est": e, "lo": lo, "hi": hi, "n": 100, "games": 40}


def test_head_to_head_counts_pairs_by_gap_within_week_and_position():
    d = pd.DataFrame({"week": [1, 1, 1, 1, 2], "position": ["WR", "WR", "WR", "RB", "WR"],
                      "predicted_points": [10.0, 11.0, 17.0, 12.0, 30.0],
                      "actual_points": [8.0, 9.0, 9.0, 1.0, 0.0]})
    out = {(b["lo"], b["hi"]): b for b in rb.head_to_head(d)}
    # WR week 1 pairs: (10,11) gap 1, higher 11 scored 9 > 8 -> win; (10,17) gap 7, 9 > 8 -> win;
    # (11,17) gap 6, tie on 9 -> half. RB and week 2 players have no partner.
    assert out[(0, 2)]["pairs"] == 1 and out[(0, 2)]["share"] == 1.0
    assert out[(6, None)]["pairs"] == 2 and out[(6, None)]["share"] == pytest.approx(0.75)
    assert out[(2, 4)]["pairs"] == 0 and out[(2, 4)]["share"] is None


def test_head_to_head_leaves_out_unstartable_players():
    d = pd.DataFrame({"week": [1, 1], "position": ["WR", "WR"], "predicted_points": [2.0, 9.0],
                      "actual_points": [0.0, 5.0]})
    assert sum(b["pairs"] for b in rb.head_to_head(d)) == 0


def test_weeks_beaten_counts_clear_weeks_from_intervals():
    ps = {"weeks": [1, 2, 3], "by_week": [
        {"week": 1, "partial": False, "vs_rolling3": _ci(-0.3, -0.6, -0.1)},
        {"week": 2, "partial": False, "vs_rolling3": _ci(-0.1, -0.4, 0.2)},
        {"week": 3, "partial": False, "vs_rolling3": _ci(0.2, -0.1, 0.5)},
        {"week": 4, "partial": True, "vs_rolling3": _ci(-2.0, -3.0, -1.0)}]}
    assert rb.weeks_beaten(ps) == {"weeks": 3, "beaten": 2, "clearly": 1}


def _ps(d, b, c, qb=None, te=None, high=None):
    panel = []
    if qb:
        panel.append({"label": "QB", "bias": _ci(*qb), "n_range": 10, "cov80": _ci(.8, .7, .9), "above": _ci(.1, .05, .15)})
    if te:
        panel.append({"label": "TE", "bias": _ci(-1, -2, .5), "n_range": 10, "cov80": _ci(*te), "above": _ci(.23, .18, .28)})
    if high:
        panel.append({"label": "Projected 10+ points", "bias": _ci(*high), "n_range": 10,
                      "cov80": _ci(.8, .75, .85), "above": _ci(.1, .05, .15)})
    return {"weeks": [1, 2, 3, 4, 5], "headline": {"n": 100, "games": 40, "vs_rolling3": _ci(*d), "bias": _ci(*b),
                                                   "cov80": _ci(*c)}, "failure_panel": panel}


def test_summary_follows_the_intervals():
    yes = rb.summary(_ps((-0.3, -0.5, -0.1), (-1.6, -2.0, -1.3), (.78, .76, .80)), {"weeks": 5, "beaten": 5})
    assert "beat a simple average" in yes and "5 of 5 weeks" in yes and "run a little low" in yes
    assert "ranges are about right" in yes
    tie = rb.summary(_ps((-0.1, -0.5, 0.3), (-0.2, -0.6, 0.2), (.70, .65, .75)), {"weeks": 5, "beaten": 3})
    assert "about equally accurate" in tie and "don't lean" in tie and "too tight" in tie


def test_trust_guide_notes_need_their_interval():
    h2h = [{"lo": 0, "hi": 2, "pairs": 500, "share": .53}, {"lo": 2, "hi": 4, "pairs": 400, "share": .6},
           {"lo": 4, "hi": 6, "pairs": 300, "share": .67}, {"lo": 6, "hi": None, "pairs": 300, "share": .78}]
    notes = rb.trust_guide(_ps((-.3, -.5, -.1), (-1, -2, -.5), (.8, .77, .83), qb=(-3.4, -4.8, -2.2),
                               te=(.65, .6, .7), high=(-2.4, -3.5, -1.4)), h2h)
    text = " ".join(notes)
    assert "QBs" in text and "TEs" in text and "projects for 10+" in text and "53%" in text and "78%" in text
    quiet = rb.trust_guide(_ps((-.3, -.5, -.1), (-1, -2, -.5), (.8, .77, .83), qb=(-.5, -1.5, .4),
                               te=(.76, .7, .82), high=(-.4, -1.2, .3)), [])
    assert quiet == []


def test_week_projections_drop_everything_off_the_whitelist():
    weekly = pd.DataFrame({"player_id": ["a"], "predicted_points": [10.0], "predicted_points_model": [12.0],
                           "prediction_ci80_lower": [5.0], "prediction_ci80_upper": [15.0], "pace_prior": [9.0]})
    kept = rb.week_projections(weekly)
    assert list(kept.columns) == ["player_id", "predicted_points", "prediction_ci80_lower", "prediction_ci80_upper"]
    assert not any(BLINDED.search(c) for c in rb.WEEK_COLS)


@pytest.mark.parametrize("path", [BRIEF, SCRIPT, CHARTS, TEAM])
def test_blinded_forecasts_are_never_referenced(path):
    assert not BLINDED.search(path.read_text())


def test_brief_page_is_plain_private_and_self_contained():
    spec = importlib.util.spec_from_file_location("render_review_brief", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = pd.concat([_rows(week=w) for w in (1, 2, 3)], ignore_index=True)
    ps = wr.player_section(rows, 3, {}, weeks=[1, 2, 3])
    wb = rb.weeks_beaten(ps)
    h2h = rb.head_to_head(wr.scored(rows))
    lp = {"week": 4, "team": "Us", "opponent": "Them", "model_total": 50.0, "espn_total": 55.0, "fallback": 1,
          "start": [], "sit": [], "players": [
              {"index": 0, "name": "A Player", "position": "WR", "slot": "WR", "opponent": "KC", "espn": 12.0,
               "model": 11.0, "lo": 5.0, "hi": 18.0, "status": None, "bye": False, "available": True},
              {"index": 1, "name": "A Defense", "position": "D/ST", "slot": "D/ST", "opponent": None, "espn": 6.0,
               "model": None, "lo": None, "hi": None, "status": None, "bye": False, "available": True}]}
    page = mod.page(2026, ps, wb, h2h, rb.trust_guide(ps, h2h), lp, None, "As of now.", True)
    visible = re.sub(r"<style>.*?</style>|<[^>]+>", " ", page, flags=re.S)
    for word in ("signed error", "MAE", "coverage", "bootstrap", "residual"):
        assert word not in visible, word
    # "Our" is the team (lineup, starters, roster); the projections are the model's.
    for phrase in ("our projection", "our range", "our number", "our 80%", "we projected", "we beat", "did we",
                   "we expect", "our top players"):
        assert phrase not in visible.lower(), phrase
    assert "<svg" in page and not re.search(r"https?://|<script|\ssrc=", page)
    assert "@media print" in page and "break-before:page" in page
    out = mod.REPORT_DIR.resolve()
    assert (PROJECT_ROOT / "docs").resolve() not in out.parents and ESPN_PRIVATE_DIR.resolve() in out.parents

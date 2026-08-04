"""Smoke tests for scripts/start_sit_prototype.py — the action #5
paper prototype from council-transcript-20260423-051434.md."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "start_sit_prototype",
    PROJECT_ROOT / "scripts" / "start_sit_prototype.py",
)
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def _preds(rows):
    return [
        {
            "player_id": f"P{i:03d}",
            "name": r["name"],
            "position": r["position"],
            "team": r.get("team", "X"),
            "predicted": r["predicted"],
            "actual": r.get("actual"),
        }
        for i, r in enumerate(rows)
    ]


def test_match_exact_name_and_position():
    preds = _preds([
        {"name": "C.Lamb", "position": "WR", "predicted": 18.0},
        {"name": "C.Lamb", "position": "QB", "predicted": 99.0},  # wrong pos decoy
    ])
    hit = ss._match_roster_entry("C.Lamb", "WR", preds)
    assert hit["position"] == "WR" and hit["predicted"] == 18.0


def test_match_disambiguates_same_last_name_by_first_initial():
    """Bijan Robinson (B) and Keilan Robinson (K) at RB. The matcher
    must use first-initial to disambiguate; the prior (last-name-only)
    matcher silently returned whichever appeared first in the list."""
    preds = _preds([
        # Note: Keilan listed first, so a setdefault-by-last-name would
        # incorrectly map "B.Robinson" → Keilan.
        {"name": "Keilan Robinson", "position": "RB", "predicted": 1.0},
        {"name": "Bijan Robinson", "position": "RB", "predicted": 18.0},
    ])
    hit = ss._match_roster_entry("B.Robinson", "RB", preds)
    assert hit["predicted"] == 18.0, f"Expected Bijan, got {hit}"
    hit = ss._match_roster_entry("K.Robinson", "RB", preds)
    assert hit["predicted"] == 1.0, f"Expected Keilan, got {hit}"


def test_match_falls_back_to_last_name():
    preds = _preds([
        {"name": "Christian McCaffrey", "position": "RB", "predicted": 20.0},
    ])
    hit = ss._match_roster_entry("C.McCaffrey", "RB", preds)
    assert hit["predicted"] == 20.0


def test_match_miss_returns_none():
    preds = _preds([{"name": "P.Mahomes", "position": "QB", "predicted": 24.0}])
    assert ss._match_roster_entry("J.Allen", "QB", preds) is None


def test_best_starters_with_flex_picks_top_remaining_rbwrtb():
    """With flex_slots=1, after the core 6 are filled, the highest
    bench RB/WR/TE should fill the FLEX. K and DST are not eligible."""
    matched = _preds([
        {"name": "QB1", "position": "QB", "predicted": 20},
        {"name": "RB1", "position": "RB", "predicted": 18},
        {"name": "RB2", "position": "RB", "predicted": 16},
        {"name": "RB3", "position": "RB", "predicted": 14},  # FLEX candidate
        {"name": "WR1", "position": "WR", "predicted": 15},
        {"name": "WR2", "position": "WR", "predicted": 14},
        {"name": "WR3", "position": "WR", "predicted": 13},  # also FLEX candidate
        {"name": "TE1", "position": "TE", "predicted": 11},
        {"name": "TE2", "position": "TE", "predicted":  9},  # also FLEX candidate
    ])
    starters, bench = ss._best_starters(matched, flex_slots=1)
    names = {p["name"] for p in starters}
    # Core 6 + FLEX = 7. RB3 (14) is the best bench RB/WR/TE.
    assert names == {"QB1", "RB1", "RB2", "WR1", "WR2", "TE1", "RB3"}
    assert {p["name"] for p in bench} == {"WR3", "TE2"}


def test_best_starters_picks_top_n_per_position():
    matched = _preds([
        {"name": "QB1", "position": "QB", "predicted": 20},
        {"name": "QB2", "position": "QB", "predicted": 10},
        {"name": "RB1", "position": "RB", "predicted": 18},
        {"name": "RB2", "position": "RB", "predicted": 16},
        {"name": "RB3", "position": "RB", "predicted": 5},
        {"name": "WR1", "position": "WR", "predicted": 15},
        {"name": "WR2", "position": "WR", "predicted": 14},
        {"name": "TE1", "position": "TE", "predicted": 11},
    ])
    starters, bench = ss._best_starters(matched)
    names = {p["name"] for p in starters}
    assert names == {"QB1", "RB1", "RB2", "WR1", "WR2", "TE1"}
    assert {p["name"] for p in bench} == {"QB2", "RB3"}


def test_compare_actual_reports_delta_and_swaps():
    user = _preds([
        {"name": "X", "position": "RB", "predicted": 10, "actual": 8},
    ])
    # Give them distinct player_ids so the swap logic sees the diff.
    user[0]["player_id"] = "X"
    model = _preds([
        {"name": "Y", "position": "RB", "predicted": 12, "actual": 20},
    ])
    model[0]["player_id"] = "Y"
    cmp = ss._compare_actual(user, model)
    assert cmp["delta"] == 12.0
    assert cmp["swaps_in"][0]["name"] == "Y"
    assert cmp["swaps_out"][0]["name"] == "X"


def test_compare_actual_no_swap_when_lineups_match():
    same = _preds([{"name": "Z", "position": "QB", "predicted": 20, "actual": 22}])
    same[0]["player_id"] = "Z"
    cmp = ss._compare_actual(same, same)
    assert cmp["delta"] == 0.0
    assert cmp["swaps_in"] == []
    assert cmp["swaps_out"] == []

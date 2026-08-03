#!/usr/bin/env python3
"""
Start/Sit Paper Prototype — action #5 of the 2026-04-23 council's
Critical Next Steps (council-transcript-20260423-051434.md).

The council's success signal:

> User can articulate a decision the tool changed or confirmed, and
> names the format/scoring/league-size constraints you must support.

This is the "paper prototype" — a CLI that, given a real redraft
roster + historical target week, outputs the model's start/sit
recommendation alongside the retrospective truth (what actually
happened that week).  The point is NOT a polished UI; it's the
thinnest possible surface that lets a real fantasy player see what
the model would have recommended for a week they remember.

Usage:
    # Roster from a JSON spec (one player per slot)
    python scripts/start_sit_prototype.py \\
        --roster roster_template.json --season 2024 --week 10

The roster JSON has two sections: ``starters`` (what the user
actually started — for retrospective comparison) and ``bench``
(everyone else on the roster). The tool re-optimizes start/sit
over the full roster using Ridge predictions for that week and
reports the delta.

Constraints for v1:
- PPR scoring only (what the walk-forward was trained on).
- Roster slots: QB:1, RB:2, WR:2, TE:1 (core; extend as league
  formats are measured). No FLEX for v1 — keeps comparison clean.
- Predictions source: most recent walk-forward CSV the repo has
  for the target (season, week).
- Player matching: exact name match, then case-insensitive prefix
  (first-initial period + last-name, e.g. "A.Cooper"); no fuzzy
  matching for v1 to keep misses visible rather than silent.

This is NOT a decision-quality measurement — it's a conversation
starter. Scale = 1 user, 1-3 weeks of their season, 30-minute
session. Notes from the session feed into Phase 4 (calibration +
VORP) and the eventual UI spec.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROSTER_SLOTS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
DEFAULT_FLEX_SLOTS = 0  # default v1: no FLEX (cleanest 6-slot comparison)


# --------------------------------------------------------------------
# Prediction loading
# --------------------------------------------------------------------

def _find_predictions_csv(season: int, explicit: Optional[Path]) -> Path:
    """If --predictions-csv was passed, use it.  Otherwise find the
    newest ``ts_backtest_{season}_*_predictions.csv`` in
    ``data/backtest_results/`` (prefer non-phantom / narrow runs for
    cleaner user demos; wide runs have extra phantom rows that the
    matcher will see as "not on roster" and skip)."""
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"predictions CSV not found: {explicit}")
        return explicit
    results = PROJECT_ROOT / "data" / "backtest_results"
    candidates = sorted(
        results.glob(f"ts_backtest_{season}_*_predictions.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"No ts_backtest_{season}_*_predictions.csv in {results}. "
            f"Run scripts/run_ts_backtest.py --season {season} first."
        )
    return candidates[0]


def load_week_predictions(csv_path: Path, season: int, week: int) -> List[Dict]:
    out: List[Dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            try:
                if int(r.get("season", -1)) != season or int(r.get("week", -1)) != week:
                    continue
            except (TypeError, ValueError):
                continue
            if r.get("is_active", "1") in ("0", 0):
                # Skip phantom rows — player didn't actually play.  For
                # start/sit replay we want to match the user's roster
                # against real observed outcomes.
                continue
            pred = r.get("predicted")
            actual = r.get("actual")
            if not pred:
                continue
            out.append({
                "player_id": r.get("player_id", ""),
                "name": (r.get("name") or "").strip(),
                "position": (r.get("position") or "").strip(),
                "team": (r.get("team") or "").strip(),
                "predicted": float(pred),
                "actual": float(actual) if actual else None,
            })
    return out


# --------------------------------------------------------------------
# Roster matching
# --------------------------------------------------------------------

def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _first_initial(name: str) -> str:
    """First character of the name, lowercased.  Handles 'C.Lamb'
    (→ 'c'), 'CeeDee Lamb' (→ 'c'), 'Christian McCaffrey' (→ 'c')."""
    n = (name or "").strip()
    return n[0].lower() if n else ""


def _last_name(name: str) -> str:
    """Extract the last token, dot-split or whitespace-split."""
    if "." in name:
        return name.split(".")[-1]
    return name.split()[-1] if name.split() else name


def _match_roster_entry(
    roster_name: str, roster_pos: str, predictions: List[Dict]
) -> Optional[Dict]:
    """Match a roster entry to a prediction row. Returns None on miss
    (intentional — silent auto-matching is worse than a loud miss for a
    paper prototype; the user should see what didn't match)."""
    target_norm = _normalize(roster_name)
    target_initial = _first_initial(roster_name)
    target_last_norm = _normalize(_last_name(roster_name))

    # 1. Exact name + position
    for r in predictions:
        if r["position"] == roster_pos and _normalize(r["name"]) == target_norm:
            return r
    # 2. (first_initial, last, position) — disambiguates same-last-name
    #    collisions like B.Robinson / K.Robinson, J.Smith / A.Smith.
    for r in predictions:
        if r["position"] != roster_pos:
            continue
        if (_first_initial(r["name"]) == target_initial
                and _normalize(_last_name(r["name"])) == target_last_norm):
            return r
    return None


# --------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------

def _best_starters(
    matched: List[Dict],
    slots: Dict[str, int] = ROSTER_SLOTS,
    flex_slots: int = DEFAULT_FLEX_SLOTS,
) -> Tuple[List[Dict], List[Dict]]:
    """Return (starters, bench) selected by top-N per position by
    predicted. With ``flex_slots > 0``, additionally fill that many
    FLEX slots with the best remaining RB/WR/TE not already starting.
    Bench is sorted by predicted descending for easy comparison."""
    starters: List[Dict] = []
    bench: List[Dict] = []
    used_ids: set = set()
    for pos, n in slots.items():
        pool = sorted(
            (m for m in matched if m["position"] == pos),
            key=lambda x: x["predicted"],
            reverse=True,
        )
        for p in pool[:n]:
            starters.append(p)
            used_ids.add(p["player_id"])
        bench.extend(pool[n:])
    if flex_slots > 0:
        # Pull best remaining flex-eligible from bench
        flex_pool = sorted(
            (p for p in bench if p["position"] in FLEX_ELIGIBLE),
            key=lambda x: x["predicted"],
            reverse=True,
        )
        for p in flex_pool[:flex_slots]:
            starters.append(p)
            used_ids.add(p["player_id"])
        # Rebuild bench excluding the flex picks
        bench = [b for b in bench if b["player_id"] not in used_ids]
    bench.sort(key=lambda x: x["predicted"], reverse=True)
    return starters, bench


def _compare_actual(user_starters: List[Dict], model_starters: List[Dict]) -> Dict:
    """Compare what the user started vs what the model would have
    started. Returns totals, delta, and per-slot explanation."""
    user_total = sum(p.get("actual") or 0.0 for p in user_starters)
    model_total = sum(p.get("actual") or 0.0 for p in model_starters)
    user_ids = {p["player_id"] for p in user_starters}
    model_ids = {p["player_id"] for p in model_starters}
    swaps_in = [p for p in model_starters if p["player_id"] not in user_ids]
    swaps_out = [p for p in user_starters if p["player_id"] not in model_ids]
    return {
        "user_actual_total": round(user_total, 2),
        "model_actual_total": round(model_total, 2),
        "delta": round(model_total - user_total, 2),
        "swaps_in": swaps_in,
        "swaps_out": swaps_out,
    }


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def _render(
    season: int,
    week: int,
    csv_path: Path,
    matched: List[Tuple[Dict, Dict]],
    unmatched: List[Dict],
    user_starters: List[Dict],
    model_starters: List[Dict],
    bench: List[Dict],
    comparison: Dict,
) -> str:
    lines: List[str] = []
    lines.append(f"Start/Sit paper prototype — season {season}, week {week}")
    lines.append(f"Predictions source: {csv_path.relative_to(PROJECT_ROOT)}")
    lines.append("")

    if unmatched:
        lines.append("UNMATCHED ROSTER ENTRIES (no prediction row — did they play this week?):")
        for entry in unmatched:
            lines.append(f"  - {entry['name']} ({entry['position']}) [{entry.get('where', '?')}]")
        lines.append("")

    lines.append(f"{'slot':<6}  {'model pick':<22}  {'pred':>6}  {'actual':>7}  {'user started':<22}  {'actual':>7}")
    lines.append("-" * 86)

    def fmt_pick(p: Optional[Dict]) -> Tuple[str, str, str]:
        if not p:
            return ("—", "—", "—")
        act = f"{p.get('actual'):.1f}" if p.get("actual") is not None else "—"
        return (p["name"], f"{p['predicted']:.2f}", act)

    # Pair by slot position. After core slots (QB/RB/WR/TE) are
    # filled, any remaining starter is a FLEX (or extra position
    # depth not used in this league).
    def split_core_and_flex(starters: List[Dict]) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
        by_pos: Dict[str, List[Dict]] = {pos: [] for pos in ROSTER_SLOTS}
        flex: List[Dict] = []
        # Sort each position by predicted desc so highest-pred fills
        # the core slot, lowest goes to FLEX (matches how
        # _best_starters allocates).
        per_pos: Dict[str, List[Dict]] = {pos: [] for pos in ROSTER_SLOTS}
        for p in starters:
            per_pos.setdefault(p["position"], []).append(p)
        for pos, plist in per_pos.items():
            plist.sort(key=lambda x: x.get("predicted", 0), reverse=True)
            n_core = ROSTER_SLOTS.get(pos, 0)
            by_pos[pos] = plist[:n_core]
            if pos in FLEX_ELIGIBLE:
                flex.extend(plist[n_core:])
        return by_pos, flex

    model_by_pos, model_flex = split_core_and_flex(model_starters)
    user_by_pos, user_flex = split_core_and_flex(user_starters)

    for pos, n in ROSTER_SLOTS.items():
        for i in range(n):
            m = model_by_pos.get(pos, [])[i] if i < len(model_by_pos.get(pos, [])) else None
            u = user_by_pos.get(pos, [])[i] if i < len(user_by_pos.get(pos, [])) else None
            m_name, m_pred, m_act = fmt_pick(m)
            u_name, _, u_act = fmt_pick(u)
            lines.append(f"{pos:<6}  {m_name:<22}  {m_pred:>6}  {m_act:>7}  {u_name:<22}  {u_act:>7}")
    # Render any FLEX picks (one row per FLEX slot used).
    n_flex = max(len(model_flex), len(user_flex))
    for i in range(n_flex):
        m = model_flex[i] if i < len(model_flex) else None
        u = user_flex[i] if i < len(user_flex) else None
        m_name, m_pred, m_act = fmt_pick(m)
        u_name, _, u_act = fmt_pick(u)
        m_label = f"{m_name} ({m['position']})" if m else m_name
        u_label = f"{u_name} ({u['position']})" if u else u_name
        lines.append(f"{'FLEX':<6}  {m_label:<22}  {m_pred:>6}  {m_act:>7}  {u_label:<22}  {u_act:>7}")
    lines.append("")

    if bench:
        lines.append("Bench (ordered by prediction; higher = near-start candidates):")
        for b in bench[:8]:
            act = f"{b.get('actual'):.1f}" if b.get("actual") is not None else "—"
            lines.append(f"  {b['position']:<3} {b['name']:<24}  pred={b['predicted']:>5.2f}  actual={act:>5}")
        lines.append("")

    lines.append(f"Totals (actual points for this week):")
    lines.append(f"  User-started:   {comparison['user_actual_total']:>6.2f}")
    lines.append(f"  Model-started:  {comparison['model_actual_total']:>6.2f}")
    lines.append(f"  Δ (model − user): {comparison['delta']:+.2f}")
    lines.append("")

    if comparison["swaps_in"] or comparison["swaps_out"]:
        lines.append("Model's recommended swap(s):")
        for inp, outp in zip(comparison["swaps_in"], comparison["swaps_out"]):
            delta = (inp.get("actual") or 0.0) - (outp.get("actual") or 0.0)
            lines.append(
                f"  START {inp['name']} ({inp['position']}, pred {inp['predicted']:.2f})  "
                f"INSTEAD OF  {outp['name']} (pred {outp['predicted']:.2f})  "
                f"→ retrospective Δ = {delta:+.2f}"
            )
    else:
        lines.append("Model's picks match user's — no recommended swaps.")
    lines.append("")

    lines.append("User interview prompts (see docs/USER_INTERVIEW_PROTOCOL.md):")
    lines.append("  - Does this match what you remember deciding?")
    lines.append("  - Any swap that looks wrong?  Why?  What info would have changed the call?")
    lines.append("  - What's missing here that would make you act on this in a live week?")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--roster", type=Path, required=True, help="JSON roster spec (see docs/USER_INTERVIEW_PROTOCOL.md).")
    ap.add_argument("--predictions-csv", type=Path, help="Override prediction source.")
    ap.add_argument("--flex", type=int, default=0,
                    help="Number of FLEX (RB/WR/TE) starting slots in this league. "
                         "Default 0 (matches v1's QB/RB/RB/WR/WR/TE only).")
    args = ap.parse_args()

    if not args.roster.exists():
        ap.error(f"roster file not found: {args.roster}")
    with args.roster.open() as f:
        roster = json.load(f)

    required = {"starters", "bench"}
    if not required.issubset(roster):
        ap.error(f"roster JSON missing keys; expected {required}, got {set(roster)}")

    csv_path = _find_predictions_csv(args.season, args.predictions_csv)
    predictions = load_week_predictions(csv_path, args.season, args.week)
    if not predictions:
        raise SystemExit(
            f"No predictions for (season={args.season}, week={args.week}) in {csv_path}"
        )

    # Match both starters and bench against predictions.
    matched: List[Tuple[Dict, Dict]] = []  # (roster_entry, prediction)
    unmatched: List[Dict] = []
    user_starters: List[Dict] = []
    all_roster_matched: List[Dict] = []
    for entry in roster["starters"]:
        pred = _match_roster_entry(entry["name"], entry["position"], predictions)
        if pred is None:
            unmatched.append({**entry, "where": "starter"})
            continue
        matched.append((entry, pred))
        user_starters.append(pred)
        all_roster_matched.append(pred)
    for entry in roster["bench"]:
        pred = _match_roster_entry(entry["name"], entry["position"], predictions)
        if pred is None:
            unmatched.append({**entry, "where": "bench"})
            continue
        matched.append((entry, pred))
        all_roster_matched.append(pred)

    model_starters, bench = _best_starters(all_roster_matched, flex_slots=args.flex)
    comparison = _compare_actual(user_starters, model_starters)

    print(_render(args.season, args.week, csv_path, matched, unmatched,
                  user_starters, model_starters, bench, comparison))
    return 0


if __name__ == "__main__":
    sys.exit(main())

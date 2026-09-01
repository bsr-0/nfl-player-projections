#!/usr/bin/env python3
"""
Generate Draft Advisor HTML for GitHub Pages.

Reads model projections + ADP data, computes spread/VONA/VORP,
and produces a self-contained _site/index.html.

Usage:
    python scripts/generate_dashboard_html.py
    python scripts/generate_dashboard_html.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.snake_draft_sim import (
    TEAMS,
    ROUNDS,
    build_draft_board,
    load_adp_board,
    load_model_projections,
    load_preseason_projections,
    _apply_vorp,
    _first_initial,
    _last_token,
    _normalize,
)
from scripts.draft_advisor import (
    compute_spread,
    compute_vona,
    validate_spread_direction,
    _latest_predictions_csv,
)

SITE_DIR = PROJECT_ROOT / "_site"
DOCS_DIR = PROJECT_ROOT / "docs"  # GitHub Pages serves from here
DRAFT_PICKS_PATH = PROJECT_ROOT / "data" / "draft_picks.parquet"

# PFR team codes → standard codes used in the rest of the system
PFR_TEAM_MAP = {
    "GNB": "GB", "KAN": "KC", "LVR": "LV", "NOR": "NO",
    "NWE": "NE", "SFO": "SF", "TAM": "TB",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "jr."}

CALIBRATION_POLICY = {
    "blend_model_weight": {"QB": 0.62, "RB": 0.58, "WR": 0.52, "TE": 0.55},
    "min_model_weight": {"QB": 0.42, "RB": 0.38, "WR": 0.35, "TE": 0.38},
    "elite_model_weight_cap": {"QB": 0.58, "RB": 0.48, "WR": 0.44, "TE": 0.48},
    "divergence_soft_pct": 0.10,
    "divergence_hard_pct": 0.45,
    "per_signal_caps_pct": {
        "age": 0.15,       # Raised from 0.05 — allows meaningful penalty for 40+ players
        "team_change": 0.12,
        "usage": 0.04,
        "regression": 0.05,
        "injury": 0.06,
        "breakout": 0.04,
        "manual": 0.25,    # Raised from 0.10 — manual adj needs real leverage for role changes
        "sos": 0.03,
    },
    "global_adjustment_cap_pct": 0.40,  # Raised further — manual overrides for role changes need room
    "market_band_pct": {"QB": 0.22, "RB": 0.18, "WR": 0.16, "TE": 0.18},
    "elite_band_pct": {"QB": 0.18, "RB": 0.12, "WR": 0.12, "TE": 0.14},
    "min_band_points": {"QB": 16.0, "RB": 14.0, "WR": 12.0, "TE": 10.0},
    "large_divergence_pct": 0.20,
    "large_divergence_min_points": {"QB": 16.0, "RB": 14.0, "WR": 12.0, "TE": 10.0},
    "elite_ecr_cutoff": 18,
    "position_bias_tolerance": 12.0,
    "outlier_share_limit": 0.05,
    "displayable_rank_cutoff": 150,
}


def _norm_key(name: str, pos: str | None = None):
    name = (name or "").strip()
    if "." in name and " " not in name:
        parts = name.split(".")
        initial = parts[0].strip()[0].lower() if parts[0].strip() else ""
        last = parts[-1].strip().lower()
    else:
        parts = name.split()
        initial = parts[0][0].lower() if parts else ""
        idx = len(parts) - 1
        while idx > 0 and parts[idx].lower().rstrip(".") in SUFFIXES:
            idx -= 1
        last = parts[idx].lower() if idx >= 0 else ""
    return (initial, "".join(c for c in last if c.isalnum()))


def _mkt_get(lookup: dict, name: str):
    """Look up a market projection by full name first, norm_key second.

    The lookup dict stores both full-name-lower keys (primary, unambiguous) and
    norm_key tuple keys (fallback, for abbreviated names). This avoids collisions
    between players who share a first initial + last name (e.g. Kaytron Allen /
    Keenan Allen both normalize to ('k', 'allen')).
    """
    return lookup.get(name.lower().strip()) or lookup.get(_norm_key(name))


def _spread_identity(sr) -> tuple[str, str, str, str, float]:
    return (
        str(getattr(sr, "player_id", "") or ""),
        getattr(sr, "name", ""),
        getattr(sr, "position", ""),
        getattr(sr, "team", ""),
        round(float(getattr(sr, "ecr", 0.0) or 0.0), 3),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _interpolate_curve(points: list[tuple[int, float]], x: int) -> float | None:
    if not points:
        return None
    pts = sorted(points, key=lambda item: item[0])
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    prev_x, prev_y = pts[0]
    for cur_x, cur_y in pts[1:]:
        if x <= cur_x:
            if cur_x == prev_x:
                return float(cur_y)
            frac = (x - prev_x) / (cur_x - prev_x)
            return float(prev_y + frac * (cur_y - prev_y))
        prev_x, prev_y = cur_x, cur_y
    return float(pts[-1][1])


def _build_market_anchor_context(spread_results, market_lookup: dict) -> dict:
    pos_rank = defaultdict(int)
    overall_curve: list[tuple[int, float]] = []
    pos_curves = defaultdict(list)
    player_context = {}

    for overall_rank, sr in enumerate(sorted(spread_results, key=lambda r: float(r.ecr)), 1):
        pos = getattr(sr, "position", "")
        pos_rank[pos] += 1
        ident = _spread_identity(sr)
        exact_market = _mkt_get(market_lookup, getattr(sr, "name", ""))
        player_context[ident] = {
            "position_rank": pos_rank[pos],
            "overall_rank": overall_rank,
            "exact_market": exact_market,
        }
        if exact_market is not None and exact_market > 0:
            pos_curves[pos].append((pos_rank[pos], float(exact_market)))
            overall_curve.append((overall_rank, float(exact_market)))

    return {
        "player_context": player_context,
        "pos_curves": dict(pos_curves),
        "overall_curve": overall_curve,
    }


def _resolve_market_anchor(sr, anchor_context: dict) -> tuple[float, str]:
    ident = _spread_identity(sr)
    player_context = anchor_context.get("player_context", {}).get(ident, {})
    exact_market = player_context.get("exact_market")
    if exact_market is not None and exact_market > 0:
        return float(exact_market), "exact_market"

    pos = getattr(sr, "position", "")
    pos_rank = int(player_context.get("position_rank") or 0)
    pos_curve = anchor_context.get("pos_curves", {}).get(pos, [])
    pos_curve_estimate = _interpolate_curve(pos_curve, pos_rank) if pos_rank else None
    if pos_curve_estimate is not None and pos_curve_estimate > 0:
        return float(pos_curve_estimate), "position_curve"

    adp_implied = float(getattr(sr, "adp_implied", 0.0) or 0.0)
    if adp_implied > 0:
        return adp_implied, "adp_yield"

    overall_rank = int(player_context.get("overall_rank") or 0)
    overall_curve_estimate = _interpolate_curve(
        anchor_context.get("overall_curve", []), overall_rank
    ) if overall_rank else None
    if overall_curve_estimate is not None and overall_curve_estimate > 0:
        return float(overall_curve_estimate), "overall_curve"

    fallback = float(getattr(sr, "blended_projection", 0.0) or 0.0)
    if fallback > 0:
        return fallback, "legacy_blend"

    return float(getattr(sr, "model_projection", 0.0) or 0.0), "raw_fallback"


def _compute_blend_weight(position: str, ecr: float, raw_proj: float, market_proj: float, team_changed: bool = False) -> float:
    if market_proj <= 0:
        return 1.0

    base_weight = CALIBRATION_POLICY["blend_model_weight"].get(position, 0.5)
    min_weight = CALIBRATION_POLICY["min_model_weight"].get(position, 0.35)

    # When the market consensus ranks a player as effectively undraftable (ECR > 150),
    # reduce model weight. The consensus is almost always right about retirement risk,
    # age cliffs, and major roster uncertainty. Scale from full weight at ECR=150
    # down to 40% of base at ECR=300+. This prevents the model's stale historical
    # data from dominating over the market's forward-looking view.
    if ecr > 150:
        # Steeper falloff: reaches 0.20 floor at ECR~350 (was 0.40 floor near-never reached).
        # At ECR=225: max(0.20, 1-(75/200)) = 0.625. At ECR=250: 0.50. At ECR=300: 0.25.
        # This prevents stale historical data from dominating for backup/unknown players.
        consensus_scale = max(0.20, 1.0 - (ecr - 150) / 200.0)
        base_weight = base_weight * consensus_scale
        min_weight = min_weight * consensus_scale

    divergence_pct = abs(raw_proj - market_proj) / max(abs(market_proj), 1.0)
    soft = CALIBRATION_POLICY["divergence_soft_pct"]
    hard = CALIBRATION_POLICY["divergence_hard_pct"]
    if divergence_pct <= soft:
        weight = base_weight
    else:
        frac = _clamp((divergence_pct - soft) / max(hard - soft, 1e-6), 0.0, 1.0)
        if team_changed:
            frac *= 0.5  # known structural reason for divergence; halve the penalty
        weight = base_weight - frac * (base_weight - min_weight)

    elite_cutoff = CALIBRATION_POLICY["elite_ecr_cutoff"]
    if ecr <= elite_cutoff:
        weight = min(weight, CALIBRATION_POLICY["elite_model_weight_cap"].get(position, weight))

    return _clamp(weight, min_weight, base_weight)


def _is_large_divergence(position: str, projected: float, market_proj: float) -> bool:
    if market_proj <= 0:
        return False
    eps = 1e-9
    abs_delta = abs(projected - market_proj)
    rel_delta = abs_delta / max(abs(market_proj), 1.0)
    min_points = CALIBRATION_POLICY["large_divergence_min_points"].get(
        position,
        CALIBRATION_POLICY["min_band_points"].get(position, 10.0),
    )
    return bool(
        rel_delta > CALIBRATION_POLICY["large_divergence_pct"] + eps
        and abs_delta > min_points + eps
    )


def _is_unresolved_display_divergence(
    position: str,
    projected: float,
    market_proj: float,
    *,
    elite_consensus: bool,
) -> bool:
    if market_proj <= 0:
        return False
    eps = 1e-9
    abs_delta = abs(projected - market_proj)
    rel_delta = abs_delta / max(abs(market_proj), 1.0)
    allowed_pct = (
        CALIBRATION_POLICY["elite_band_pct"].get(position, 0.14)
        if elite_consensus
        else CALIBRATION_POLICY["market_band_pct"].get(position, 0.18)
    )
    rel_threshold = max(CALIBRATION_POLICY["large_divergence_pct"], allowed_pct)
    abs_threshold = max(
        CALIBRATION_POLICY["large_divergence_min_points"].get(
            position,
            CALIBRATION_POLICY["min_band_points"].get(position, 10.0),
        ),
        market_proj * allowed_pct,
    )
    return bool(rel_delta > rel_threshold + eps and abs_delta > abs_threshold + eps)


def _add_adjustment(
    breakdown: list[dict],
    signal: str,
    requested_pct: float,
    cap_pct: float,
    base_projection: float,
    label: str,
) -> bool:
    if abs(requested_pct) < 1e-9:
        return False

    capped_pct = _clamp(requested_pct, -cap_pct, cap_pct)
    delta = base_projection * capped_pct
    breakdown.append({
        "signal": signal,
        "label": label,
        "delta": round(delta, 2),
        "pct": round(capped_pct, 4),
        "requestedPct": round(requested_pct, 4),
        "capHit": abs(capped_pct - requested_pct) > 1e-9,
    })
    return abs(capped_pct - requested_pct) > 1e-9


def _apply_structured_adjustments(
    *,
    sr,
    calibrated_proj: float,
    age_data: dict | None,
    team_change_data: dict | None,
    trend_data: dict | None,
    regression_data: dict | None,
    injury_data: dict | None,
    breakout_data: dict | None,
    manual_data: dict | None,
    sos_label: str,
    sos_mult: float,
) -> tuple[float, float, list[dict], dict]:
    caps = CALIBRATION_POLICY["per_signal_caps_pct"]
    breakdown: list[dict] = []
    flags = {
        "ageCapHit": False,
        "totalAdjustmentCapHit": False,
        "manualCapHit": False,
    }

    # When a player has a major injury signal (expected < 15 games), their
    # usage_trend and regression signals are measuring the injured season, not
    # their true baseline. Suppress those two to avoid triple-counting the same
    # root cause (e.g. ACL mid-season → low late targets → regression penalty).
    major_injury = bool(injury_data and injury_data.get("expected_gp", 17) < 15)

    if age_data and age_data.get("mult", 1.0) != 1.0:
        flags["ageCapHit"] = _add_adjustment(
            breakdown,
            "age",
            float(age_data["mult"]) - 1.0,
            caps["age"],
            calibrated_proj,
            f"Age {age_data['age']:.0f}",
        )

    if team_change_data and team_change_data.get("mult", 1.0) != 1.0:
        from_team = str(team_change_data.get("from", "") or "").strip()
        to_team = str(team_change_data.get("to", "") or "").strip()
        if from_team and to_team:
            team_change_label = f"Team change {from_team}->{to_team}"
        else:
            team_change_label = "Team change"
        _add_adjustment(
            breakdown,
            "team_change",
            float(team_change_data["mult"]) - 1.0,
            caps["team_change"],
            calibrated_proj,
            team_change_label,
        )

    if trend_data and trend_data.get("mult", 1.0) != 1.0 and not major_injury:
        slope = float(trend_data.get("slope", 0.0) or 0.0)
        slope_prefix = "+" if slope > 0 else ""
        _add_adjustment(
            breakdown,
            "usage",
            float(trend_data["mult"]) - 1.0,
            caps["usage"],
            calibrated_proj,
            f"Usage trend {slope_prefix}{slope:.0f}",
        )

    if regression_data and regression_data.get("mult", 1.0) != 1.0 and not major_injury:
        _add_adjustment(
            breakdown,
            "regression",
            float(regression_data["mult"]) - 1.0,
            caps["regression"],
            calibrated_proj,
            f"Regression {int(regression_data.get('pct_above', 0))}%",
        )

    if injury_data and injury_data.get("mult", 1.0) != 1.0:
        _add_adjustment(
            breakdown,
            "injury",
            float(injury_data["mult"]) - 1.0,
            caps["injury"],
            calibrated_proj,
            f"Durability {injury_data['expected_gp']:.0f} expected games",
        )

    if breakout_data and breakout_data.get("mult", 1.0) != 1.0:
        _add_adjustment(
            breakdown,
            "breakout",
            float(breakout_data["mult"]) - 1.0,
            caps["breakout"],
            calibrated_proj,
            f"Breakout +{int(breakout_data.get('eff_change', 0))}% eff",
        )

    if manual_data and manual_data.get("mult", 1.0) != 1.0:
        flags["manualCapHit"] = _add_adjustment(
            breakdown,
            "manual",
            float(manual_data["mult"]) - 1.0,
            caps["manual"],
            calibrated_proj,
            manual_data.get("note", "Manual context"),
        )

    if sos_label and sos_mult != 1.0:
        _add_adjustment(
            breakdown,
            "sos",
            float(sos_mult) - 1.0,
            caps["sos"],
            calibrated_proj,
            sos_label,
        )

    # Separate manual from structural: manual bypasses global cap and market clamp.
    manual_item = next((item for item in breakdown if item["signal"] == "manual"), None)
    manual_delta = round(manual_item["delta"], 2) if manual_item else 0.0
    structural_breakdown = [item for item in breakdown if item["signal"] != "manual"]

    total_structural = sum(item["delta"] for item in structural_breakdown)
    global_cap = calibrated_proj * CALIBRATION_POLICY["global_adjustment_cap_pct"]
    if abs(total_structural) > global_cap > 0:
        flags["totalAdjustmentCapHit"] = True
        scale = global_cap / abs(total_structural)
        for item in structural_breakdown:
            item["delta"] = round(item["delta"] * scale, 2)
            item["pct"] = round(item["pct"] * scale, 4)

    structural_delta = round(sum(item["delta"] for item in structural_breakdown), 2)
    full_breakdown = structural_breakdown + ([manual_item] if manual_item else [])
    return structural_delta, manual_delta, full_breakdown, flags


def _calibrate_projection(
    *,
    sr,
    market_proj: float,
    market_source: str,
    age_data: dict | None,
    team_change_data: dict | None,
    trend_data: dict | None,
    regression_data: dict | None,
    injury_data: dict | None,
    breakout_data: dict | None,
    manual_data: dict | None,
    sos_mult: float,
    sos_label: str,
) -> dict:
    raw_proj = float(getattr(sr, "model_projection", 0.0) or 0.0)
    position = getattr(sr, "position", "")
    ecr = float(getattr(sr, "ecr", 999.0) or 999.0)

    # Pure ML: no blending with market consensus. ML projection is the base.
    calibrated_proj = raw_proj

    structural_delta, manual_delta, adj_breakdown, adj_flags = _apply_structured_adjustments(
        sr=sr,
        calibrated_proj=calibrated_proj,
        age_data=age_data,
        team_change_data=team_change_data,
        trend_data=trend_data,
        regression_data=regression_data,
        injury_data=injury_data,
        breakout_data=breakout_data,
        manual_data=manual_data,
        sos_label=sos_label,
        sos_mult=sos_mult,
    )

    # No market-based clamping — structural adjustments apply directly
    final_display = max(0.0, calibrated_proj + structural_delta + manual_delta)

    # ADP divergence as informational signal (not used for blending)
    divergence_pct = (
        abs(raw_proj - market_proj) / max(abs(market_proj), 1.0)
        if market_proj > 0 else 0.0
    )

    final_adj_delta = round(final_display - calibrated_proj, 2)
    why = [
        f"Our rank #{int(getattr(sr, 'model_rank', 999))} vs consensus #{int(round(ecr))}",
        f"Pure ML projection {round(raw_proj, 1):.1f}",
    ]
    if market_proj > 0:
        direction = "higher" if raw_proj > market_proj else "lower"
        why.append(f"ML is {abs(raw_proj - market_proj):.1f} pts {direction} than ADP consensus ({market_proj:.1f})")
    for item in adj_breakdown:
        if abs(item["delta"]) < 0.05:
            continue
        why.append(f"{item['label']}: {item['delta']:+.1f}")
    why = why[:5]

    flags = {
        "ageCapHit": adj_flags["ageCapHit"],
        "positionBiasCheck": False,
        "totalAdjustmentCapHit": adj_flags["totalAdjustmentCapHit"],
        "manualCapHit": adj_flags["manualCapHit"],
        "marketSource": market_source,
    }

    return {
        "raw_projection": round(raw_proj, 2),
        "market_consensus_projection": round(market_proj, 2),
        "adp_divergence_pct": round(divergence_pct, 4),
        "calibrated_projection": round(calibrated_proj, 2),
        "final_display_projection": round(final_display, 2),
        "adjustment_delta": round(final_adj_delta, 2),
        "adjustment_breakdown": adj_breakdown,
        "calibration_flags": flags,
        "why": why,
    }


def _summarize_board_calibration(players: list[dict]) -> dict:
    if not players:
        return {"summary": {}, "auditPlayers": []}

    displayable_cutoff = CALIBRATION_POLICY["displayable_rank_cutoff"]
    displayable_players = [
        p for p in players
        if p["ecr"] <= displayable_cutoff or p["mr"] <= displayable_cutoff
    ]
    if not displayable_players:
        displayable_players = players

    def _top_by_ecr(position: str, limit: int = 24) -> list[dict]:
        pos_players = [p for p in displayable_players if p["p"] == position]
        return sorted(pos_players, key=lambda p: p["ecr"])[:limit]

    rb_top = _top_by_ecr("RB")
    wr_top = _top_by_ecr("WR")
    rb_avg = sum(p["proj"] for p in rb_top) / len(rb_top) if rb_top else 0.0
    wr_avg = sum(p["proj"] for p in wr_top) / len(wr_top) if wr_top else 0.0
    rb_market_avg = sum(p["marketProj"] for p in rb_top) / len(rb_top) if rb_top else 0.0
    wr_market_avg = sum(p["marketProj"] for p in wr_top) / len(wr_top) if wr_top else 0.0
    actual_gap = round(rb_avg - wr_avg, 2)
    market_gap = round(rb_market_avg - wr_market_avg, 2)

    large_divergence_players = [
        p for p in players if p["calibrationFlags"].get("largeDivergence")
    ]
    unresolved_display_players = [
        p for p in displayable_players
        if p["calibrationFlags"].get("unresolvedDisplayDivergence")
    ]
    clamp_players = [p for p in players if p["calibrationFlags"].get("clampHit")]
    elite_override_players = [
        p for p in players if p["calibrationFlags"].get("eliteConsensusOverrideCheck")
    ]

    audit_players = sorted(
        players,
        key=lambda p: abs(p["rawProj"] - p["marketProj"]),
        reverse=True,
    )[:25]
    audit_players = [
        {
            "name": p["n"],
            "position": p["p"],
            "team": p["t"],
            "ecr": p["ecr"],
            "rawProj": p["rawProj"],
            "marketProj": p["marketProj"],
            "calibratedProj": p["calibratedProj"],
            "proj": p["proj"],
            "adjDelta": p["adjDelta"],
            "flags": p["calibrationFlags"],
        }
        for p in audit_players
    ]

    return {
        "summary": {
            "top24_rb_wr_gap": actual_gap,
            "top24_market_rb_wr_gap": market_gap,
            "rb_wr_gap_excess": round(actual_gap - market_gap, 2),
            "large_divergence_share": round(
                len(unresolved_display_players) / len(displayable_players), 4
            ),
            "raw_large_divergence_share_all": round(
                len(large_divergence_players) / len(players), 4
            ),
            "displayable_players": len(displayable_players),
            "unresolved_display_divergence_count": len(unresolved_display_players),
            "clamp_hits": len(clamp_players),
            "elite_override_checks": len(elite_override_players),
            "position_bias_tolerance": CALIBRATION_POLICY["position_bias_tolerance"],
            "outlier_share_limit": CALIBRATION_POLICY["outlier_share_limit"],
        },
        "auditPlayers": audit_players,
    }


def load_headshot_lookup() -> dict:
    """Load the latest known headshot URLs from ``weekly_rosters_v2``.

    The dashboard only needs a best-effort frontend embellishment, so this
    lookup is intentionally tolerant of missing tables/columns and falls back
    cleanly when the richer roster snapshot is unavailable.
    """
    import sqlite3
    from config.settings import DB_PATH

    lookup = {
        "by_player_id": {},
        "by_name_team": {},
        "by_name_pos": {},
        "by_name": {},
    }

    try:
        conn = sqlite3.connect(str(DB_PATH))
    except Exception:
        return lookup

    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='weekly_rosters_v2'"
        ).fetchone()
        if not table_exists:
            return lookup

        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(weekly_rosters_v2)").fetchall()
        }
        required_cols = {"player_name", "team", "position", "headshot_url"}
        if not required_cols.issubset(columns):
            return lookup

        rows = conn.execute(
            """
            SELECT player_id, player_name, team, position, headshot_url
            FROM weekly_rosters_v2
            WHERE headshot_url IS NOT NULL
              AND TRIM(headshot_url) != ''
            ORDER BY season DESC, week DESC
            """
        ).fetchall()
    except Exception:
        return lookup
    finally:
        conn.close()

    for player_id, player_name, team, position, headshot_url in rows:
        url = str(headshot_url or "").strip()
        if not url:
            continue

        name = str(player_name or "").strip()
        team = str(team or "").strip()
        position = str(position or "").strip()
        pid = str(player_id or "").strip()
        name_key = _norm_key(name, position) if name else None

        if pid:
            lookup["by_player_id"].setdefault(pid, url)
        if name_key and team:
            lookup["by_name_team"].setdefault((name_key, team), url)
        if name_key and position:
            lookup["by_name_pos"].setdefault((name_key, position), url)
        if name_key:
            lookup["by_name"].setdefault(name_key, url)

    return lookup


def resolve_headshot_url(sr, headshot_lookup: dict) -> str:
    """Resolve the best available player headshot URL for a spread row."""
    player_id = str(getattr(sr, "player_id", "") or "").strip()
    if player_id:
        url = headshot_lookup.get("by_player_id", {}).get(player_id)
        if url:
            return url

    name_key = _norm_key(getattr(sr, "name", ""), getattr(sr, "position", ""))
    team = str(getattr(sr, "team", "") or "").strip()
    position = str(getattr(sr, "position", "") or "").strip()

    if name_key and team:
        url = headshot_lookup.get("by_name_team", {}).get((name_key, team))
        if url:
            return url
    if name_key and position:
        url = headshot_lookup.get("by_name_pos", {}).get((name_key, position))
        if url:
            return url
    if name_key:
        return headshot_lookup.get("by_name", {}).get(name_key, "")

    return ""


def load_draft_class(season: int) -> pd.DataFrame:
    """Load draft picks for a season from parquet, normalize team codes."""
    if not DRAFT_PICKS_PATH.exists():
        return pd.DataFrame()
    dp = pd.read_parquet(DRAFT_PICKS_PATH)
    cls = dp[(dp["season"] == season) & (dp["position"].isin(["QB", "RB", "WR", "TE"]))].copy()
    if cls.empty:
        return cls
    cls["team"] = cls["team"].map(lambda t: PFR_TEAM_MAP.get(t, t))
    cls = cls.rename(columns={"pfr_player_name": "name"})
    return cls[["name", "position", "team", "round", "pick"]].reset_index(drop=True)


def build_rookie_projection_curve(seasons: range = range(2020, 2025)) -> dict:
    """Build position+round → expected season FP from historical data."""
    import sqlite3
    from config.settings import DB_PATH

    if not DRAFT_PICKS_PATH.exists():
        return {}

    dp = pd.read_parquet(DRAFT_PICKS_PATH)
    skill = dp[
        (dp["position"].isin(["QB", "RB", "WR", "TE"]))
        & (dp["season"].isin(seasons))
        & (dp["gsis_id"].notna())
    ]

    conn = sqlite3.connect(str(DB_PATH))
    curves: dict = {}  # (pos, round) -> avg_fp

    for pos in ["QB", "RB", "WR", "TE"]:
        for rnd in range(1, 8):
            picks = skill[(skill["position"] == pos) & (skill["round"] == rnd)]
            fps = []
            for _, r in picks.iterrows():
                row = conn.execute(
                    "SELECT SUM(fantasy_points) FROM player_weekly_stats "
                    "WHERE player_id=? AND season=?",
                    (r["gsis_id"], int(r["season"])),
                ).fetchone()
                if row[0] and row[0] > 0:
                    fps.append(row[0])
            curves[(pos, rnd)] = round(sum(fps) / len(fps), 1) if fps else 0

    conn.close()
    return curves


# ------------------------------------------------------------------
# Projection adjustments (age, team change, usage trend)
# ------------------------------------------------------------------

# Position-specific aging: (decline_start_age, pct_per_year)
AGE_CURVES = {
    "QB": (37, 0.03),
    "RB": (27, 0.05),
    "WR": (30, 0.04),
    "TE": (31, 0.03),
}

TEAM_CHANGE_PENALTY = {"QB": 0.10, "RB": 0.05, "WR": 0.12, "TE": 0.08}

# Regression to mean: how much of a career year is retained next season
# From 2020-2025 analysis of 102 career-year player-seasons
REGRESSION_RETAIN = {"QB": 0.833, "RB": 0.748, "WR": 0.763, "TE": 0.774}
CAREER_YEAR_THRESHOLD = 0.30  # 30% above career avg = career year

# Injury availability → games multiplier (from r=0.57 historical analysis)
# Key: (low_avail, high_avail) → multiplier vs baseline 13.85 GP
AVAILABILITY_CURVE = [
    (0.00, 0.40, 0.363),
    (0.40, 0.50, 0.534),
    (0.50, 0.60, 0.651),
    (0.60, 0.70, 0.721),
    (0.70, 0.80, 0.821),
    (0.80, 0.90, 0.935),
    (0.90, 1.01, 1.000),  # healthy baseline
]


def compute_age_adjustments(season: int) -> dict:
    """Return {normalized_name_key: {age, multiplier}} for players with birth dates."""
    from datetime import datetime
    season_start = datetime(season, 9, 1)
    rosters = pd.DataFrame()

    try:
        import nfl_data_py as nfl
        rosters = nfl.import_seasonal_rosters([season - 1])
    except Exception:
        rosters = pd.DataFrame()

    if rosters.empty or "birth_date" not in rosters.columns or "position" not in rosters.columns:
        try:
            import sqlite3
            from config.settings import DB_PATH

            conn = sqlite3.connect(str(DB_PATH))
            rosters = pd.read_sql_query(
                """
                SELECT
                    p.name AS player_name,
                    p.position,
                    p.birth_date
                FROM players p
                JOIN (
                    SELECT player_id, MAX(season) AS season
                    FROM player_weekly_stats
                    WHERE season = ?
                    GROUP BY player_id
                ) s ON p.player_id = s.player_id
                WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                  AND p.birth_date IS NOT NULL
                  AND p.birth_date != ''
                """,
                conn,
                params=(season - 1,),
            )
            conn.close()
        except Exception:
            return {}

    result = {}
    for _, r in rosters.iterrows():
        pos = r.get("position")
        if pos not in AGE_CURVES or pd.isna(r.get("birth_date")):
            continue
        try:
            bd = pd.to_datetime(r["birth_date"])
            age = (season_start - bd).days / 365.25
        except Exception:
            continue

        decline_start, rate = AGE_CURVES[pos]
        if age > decline_start:
            years_over = age - decline_start
            mult = max(0.5, 1.0 - rate * years_over)
        else:
            mult = 1.0

        name = r.get("player_name") or r.get("name") or ""
        if not name:
            continue
        result[(name, pos)] = {"age": round(age, 1), "mult": round(mult, 3)}

    return result


def compute_team_changes(season: int) -> dict:
    """Return {player_name: {from_team, to_team, multiplier}} for players who changed teams."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    prior2 = season - 2
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT p.name, p.position, a.team AS team_old, b.team AS team_new
        FROM (
            SELECT player_id, team, COUNT(*) as gp
            FROM player_weekly_stats WHERE season = ?
            GROUP BY player_id
            HAVING gp >= 4
        ) a
        JOIN (
            SELECT player_id, team, COUNT(*) as gp
            FROM player_weekly_stats WHERE season = ?
            GROUP BY player_id
            HAVING gp >= 4
        ) b ON a.player_id = b.player_id
        JOIN players p ON a.player_id = p.player_id
        WHERE a.team != b.team
          AND a.team != '' AND b.team != ''
          AND p.position IN ('QB','RB','WR','TE')
    """, (prior2, prior)).fetchall()

    conn.close()

    profiles = compute_team_pos_profiles(season)

    result = {}
    for name, pos, old_team, new_team in rows:
        old_p = profiles.get((old_team, pos, 1))
        new_p = profiles.get((new_team, pos, 1))
        if old_p and new_p:
            if pos == "RB":
                old_u = old_p["avg_tgt_share"] * 2.5 + old_p["avg_carry_share"]
                new_u = new_p["avg_tgt_share"] * 2.5 + new_p["avg_carry_share"]
            elif pos in ("WR", "TE"):
                old_u = float(old_p["avg_tgt_share"])
                new_u = float(new_p["avg_tgt_share"])
            else:
                old_u = new_u = 0.0

            if old_u > 0 and new_u > 0:
                raw_mult = new_u / old_u
                mult = round(max(0.80, min(1.25, raw_mult)), 3)
            else:
                mult = round(1.0 - TEAM_CHANGE_PENALTY.get(pos, 0.05), 2)
        else:
            mult = round(1.0 - TEAM_CHANGE_PENALTY.get(pos, 0.05), 2)

        result[(name, pos)] = {
            "from": old_team,
            "to": new_team,
            "mult": mult,
        }
    return result


def compute_usage_trends(season: int) -> dict:
    """Return {player_name: {early, late, slope, multiplier}} based on late-season usage."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT p.name, p.position,
               AVG(CASE WHEN pws.week <= 9 THEN pws.targets END) as early_tgt,
               AVG(CASE WHEN pws.week > 9 THEN pws.targets END) as late_tgt,
               AVG(CASE WHEN pws.week <= 9 THEN pws.rushing_attempts END) as early_car,
               AVG(CASE WHEN pws.week > 9 THEN pws.rushing_attempts END) as late_car
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season = ? AND p.position IN ('QB','RB','WR','TE')
        GROUP BY pws.player_id
        HAVING COUNT(*) >= 8
    """, (prior,)).fetchall()

    conn.close()

    result = {}
    for name, pos, early_tgt, late_tgt, early_car, late_car in rows:
        # RBs use carries, others use targets
        if pos == "RB":
            early = early_car or 0
            late = late_car or 0
            metric = "carries"
        else:
            early = early_tgt or 0
            late = late_tgt or 0
            metric = "targets"

        slope = late - early

        if slope >= 2.0:
            mult = 1.05
        elif slope <= -2.0:
            mult = 0.92
        else:
            mult = 1.0

        if mult != 1.0:
            result[(name, pos)] = {
                "early": round(early, 1),
                "late": round(late, 1),
                "slope": round(slope, 1),
                "metric": metric,
                "mult": mult,
            }
    return result


def compute_injury_risk(season: int) -> dict:
    """Discount projections based on 3-year games-played availability.

    Returns {player_name: {avail_rate, expected_gp, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    prior_seasons = (season - 3, season - 2, season - 1)

    rows = conn.execute("""
        SELECT p.name, p.position,
               COUNT(DISTINCT pws.season || ':' || pws.week) as total_games,
               COUNT(DISTINCT pws.season) as seasons_active
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season IN (?, ?, ?)
          AND p.position IN ('QB','RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id
        HAVING seasons_active >= 2
    """, prior_seasons).fetchall()

    conn.close()

    result = {}
    for name, pos, total_games, seasons_active in rows:
        possible_games = seasons_active * 17
        avail_rate = total_games / possible_games if possible_games > 0 else 1.0

        # Look up multiplier from availability curve
        mult = 1.0
        for low, high, m in AVAILABILITY_CURVE:
            if low <= avail_rate < high:
                mult = m
                break

        if mult >= 0.99:
            continue  # healthy player, no discount

        # Cap at 0.65 — don't discount more than 35% for injury alone
        mult = max(0.65, mult)
        expected_gp = round(mult * 17, 1)
        result[(name, pos)] = {
            "avail_rate": round(avail_rate, 2),
            "expected_gp": expected_gp,
            "mult": round(mult, 3),
        }

    return result


def compute_breakout_candidates(season: int) -> dict:
    """Identify players with efficiency + volume + snap share momentum.

    Only flags players where all three signals align (reduces false positives).
    Returns {player_name: {eff_change, vol_change, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    prior2 = season - 2
    conn = sqlite3.connect(str(DB_PATH))

    # Per-touch efficiency and volume for two seasons
    rows = conn.execute("""
        SELECT p.name, p.position, pws.season,
               SUM(pws.targets) + SUM(pws.rushing_attempts) as touches,
               SUM(pws.fantasy_points) as total_fp,
               COUNT(DISTINCT pws.week) as games,
               AVG(CASE WHEN pws.week > 9 THEN pws.snap_share END) as late_snap
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season IN (?, ?)
          AND p.position IN ('RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id, pws.season
        HAVING games >= 8 AND touches >= 40
    """, (prior2, prior)).fetchall()

    conn.close()

    # Group by player
    from collections import defaultdict
    player_data = defaultdict(dict)
    for name, pos, yr, touches, fp, games, late_snap in rows:
        player_data[name][yr] = {
            "pos": pos,
            "touches": touches,
            "fp": fp,
            "eff": fp / touches if touches > 0 else 0,
            "games": games,
            "late_snap": late_snap or 0,
        }

    result = {}
    for name, seasons in player_data.items():
        if prior not in seasons or prior2 not in seasons:
            continue

        cur = seasons[prior]
        prev = seasons[prior2]

        if prev["eff"] <= 0:
            continue

        eff_change = (cur["eff"] - prev["eff"]) / prev["eff"]
        vol_change = (cur["touches"] - prev["touches"]) / prev["touches"] if prev["touches"] > 0 else 0

        # Both must align:
        # 1. Efficiency improved 15%+
        # 2. Volume stable or growing (>= -10%)
        if eff_change >= 0.15 and vol_change >= -0.10:
            # Conservative boost — only +5% since 75% of efficiency jumps revert
            # But combined signals have higher persistence (~45%)
            mult = 1.05
            pos = cur.get("pos", "")
            result[(name, pos)] = {
                "eff_change": round(eff_change * 100),
                "vol_change": round(vol_change * 100),
                "mult": mult,
            }

    return result


def compute_regression_adjustments(season: int) -> dict:
    """Identify career-year players who should regress.

    Compares most recent season PPG to 3-year career avg.
    If recent season was CAREER_YEAR_THRESHOLD above career avg,
    apply position-specific regression multiplier.

    Returns {player_name: {career_ppg, recent_ppg, pct_above, mult}}.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    # Get per-season PPG for each player over the last 4 seasons
    rows = conn.execute("""
        SELECT p.name, p.position, pws.season,
               AVG(pws.fantasy_points) as ppg,
               COUNT(DISTINCT pws.week) as games
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season BETWEEN ? AND ?
          AND p.position IN ('QB','RB','WR','TE')
          AND pws.fantasy_points > 0
        GROUP BY pws.player_id, pws.season
        HAVING games >= 6
    """, (prior - 3, prior)).fetchall()

    conn.close()

    # Group by player
    from collections import defaultdict
    player_seasons = defaultdict(list)
    player_pos = {}
    for name, pos, season_yr, ppg, games in rows:
        player_seasons[name].append((season_yr, ppg, games))
        player_pos[name] = pos

    result = {}
    for name, seasons_data in player_seasons.items():
        if len(seasons_data) < 2:
            continue

        pos = player_pos[name]
        if pos not in REGRESSION_RETAIN:
            continue

        # Most recent season
        seasons_data.sort(key=lambda x: x[0])
        recent_season, recent_ppg, _ = seasons_data[-1]
        if recent_season != prior:
            continue  # player didn't play last season

        # Career avg from prior seasons (excluding most recent)
        prior_ppgs = [ppg for yr, ppg, _ in seasons_data if yr < recent_season]
        if not prior_ppgs:
            continue
        career_ppg = sum(prior_ppgs) / len(prior_ppgs)

        if career_ppg <= 0:
            continue

        pct_above = (recent_ppg - career_ppg) / career_ppg
        if pct_above < CAREER_YEAR_THRESHOLD:
            continue

        # Apply regression: project next year as blend between
        # career_year * retain_rate and career_avg * 1.17
        retain = REGRESSION_RETAIN[pos]
        # The multiplier represents how much to discount the raw projection
        # Raw proj is based on recent season. Regression pulls it toward career avg.
        mult = retain + (1.0 - retain) * (career_ppg / recent_ppg)
        mult = max(0.6, min(1.0, mult))

        result[(name, pos)] = {
            "career_ppg": round(career_ppg, 1),
            "recent_ppg": round(recent_ppg, 1),
            "pct_above": round(pct_above * 100),
            "mult": round(mult, 3),
        }

    return result


def compute_defense_rankings(season: int) -> dict:
    """Compute FP allowed per game by each defense, per position.

    Returns {team: {pos: rank}} where rank 1 = easiest matchup (most FP allowed).
    Uses schedule join since opponent field is empty in player_weekly_stats.
    """
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rankings = {}  # {pos: {defense_team: avg_fp_allowed}}
    for pos in ["QB", "RB", "WR", "TE"]:
        # FP allowed when team is away defense (opponent scores at home)
        away = conn.execute("""
            SELECT s.away_team as defense, AVG(pws.fantasy_points) as fp
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            JOIN schedule s ON pws.season = s.season AND pws.week = s.week
                AND pws.team = s.home_team
            WHERE pws.season = ? AND p.position = ?
            GROUP BY s.away_team
        """, (prior, pos)).fetchall()

        # FP allowed when team is home defense
        home = conn.execute("""
            SELECT s.home_team as defense, AVG(pws.fantasy_points) as fp
            FROM player_weekly_stats pws
            JOIN players p ON pws.player_id = p.player_id
            JOIN schedule s ON pws.season = s.season AND pws.week = s.week
                AND pws.team = s.away_team
            WHERE pws.season = ? AND p.position = ?
            GROUP BY s.home_team
        """, (prior, pos)).fetchall()

        # Combine home + away
        from collections import defaultdict
        totals = defaultdict(list)
        for team, fp in away + home:
            if team:
                totals[team].append(fp)

        avgs = {team: sum(fps) / len(fps) for team, fps in totals.items() if fps}
        # Rank: 1 = most FP allowed (easiest), 32 = least (hardest)
        sorted_teams = sorted(avgs.items(), key=lambda x: -x[1])
        rankings[pos] = {team: rank + 1 for rank, (team, _) in enumerate(sorted_teams)}

    conn.close()

    # Build per-team summary: {team: {QB: rank, RB: rank, ...}}
    all_teams = set()
    for pos_ranks in rankings.values():
        all_teams.update(pos_ranks.keys())

    result = {}
    for team in all_teams:
        result[team] = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            result[team][pos] = rankings[pos].get(team, 16)

    return result


# NFL divisions (fixed — each team plays 6 games vs division rivals)
NFL_DIVISIONS = {
    "BUF": ["MIA", "NE", "NYJ"], "MIA": ["BUF", "NE", "NYJ"],
    "NE": ["BUF", "MIA", "NYJ"], "NYJ": ["BUF", "MIA", "NE"],
    "BAL": ["CIN", "CLE", "PIT"], "CIN": ["BAL", "CLE", "PIT"],
    "CLE": ["BAL", "CIN", "PIT"], "PIT": ["BAL", "CIN", "CLE"],
    "HOU": ["IND", "JAX", "TEN"], "IND": ["HOU", "JAX", "TEN"],
    "JAX": ["HOU", "IND", "TEN"], "TEN": ["HOU", "IND", "JAX"],
    "DEN": ["KC", "LAC", "LV"], "KC": ["DEN", "LAC", "LV"],
    "LAC": ["DEN", "KC", "LV"], "LV": ["DEN", "KC", "LAC"],
    "DAL": ["NYG", "PHI", "WAS"], "NYG": ["DAL", "PHI", "WAS"],
    "PHI": ["DAL", "NYG", "WAS"], "WAS": ["DAL", "NYG", "PHI"],
    "CHI": ["DET", "GB", "MIN"], "DET": ["CHI", "GB", "MIN"],
    "GB": ["CHI", "DET", "MIN"], "MIN": ["CHI", "DET", "GB"],
    "ATL": ["CAR", "NO", "TB"], "CAR": ["ATL", "NO", "TB"],
    "NO": ["ATL", "CAR", "TB"], "TB": ["ATL", "CAR", "NO"],
    "ARI": ["LAR", "SEA", "SF"], "LAR": ["ARI", "SEA", "SF"],
    "SEA": ["ARI", "LAR", "SF"], "SF": ["ARI", "LAR", "SEA"],
}


def compute_sos_adjustment(
    player_team: str, position: str, defense_rankings: dict
) -> tuple[float, str]:
    """Compute SOS multiplier from division rival defense quality.

    6 of 17 games are against division rivals (known before schedule).
    Average their defense rank at this position → SOS proxy.
    """
    rivals = NFL_DIVISIONS.get(player_team, [])
    if not rivals or not defense_rankings:
        return 1.0, ""

    rival_ranks = []
    for rival in rivals:
        rank = defense_rankings.get(rival, {}).get(position)
        if rank is not None:
            rival_ranks.append(rank)

    if not rival_ranks:
        return 1.0, ""

    avg_rank = sum(rival_ranks) / len(rival_ranks)
    # avg_rank 1-10 = easy division, 11-22 = neutral, 23-32 = tough
    if avg_rank <= 10:
        mult = 1.03
        label = "Easy div"
    elif avg_rank >= 23:
        mult = 0.97
        label = "Hard div"
    else:
        return 1.0, ""

    return mult, label


def check_data_sources(season: int) -> list[dict]:
    """Check which data sources are available for a season.

    Returns a list of {name, status, detail} dicts where status is
    'available', 'unavailable', or 'partial'.
    """
    import sqlite3
    from config.settings import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    sources = []

    def _count(query, params=()):
        return conn.execute(query, params).fetchone()[0]

    # 1. Prior season stats (basis of preseason projections)
    prior = season - 1
    prior_stats = _count(
        "SELECT COUNT(*) FROM player_weekly_stats WHERE season=?", (prior,)
    )
    sources.append({
        "name": f"{prior} Player Stats",
        "status": "available" if prior_stats > 500 else "unavailable",
        "detail": f"{prior_stats:,} player-weeks" if prior_stats else "No data",
    })

    # 2. ADP / Expert Consensus Rankings
    adp_count = _count(
        "SELECT COUNT(*) FROM adp_history WHERE season=?", (season,)
    )
    sources.append({
        "name": "ADP / Expert Rankings",
        "status": "available" if adp_count > 50 else "unavailable",
        "detail": f"{adp_count:,} rankings" if adp_count else "Not yet scraped",
    })

    # 3. NFL Schedule
    sched_count = _count(
        "SELECT COUNT(*) FROM schedule WHERE season=?", (season,)
    )
    sources.append({
        "name": f"{season} NFL Schedule",
        "status": "available" if sched_count >= 256 else "unavailable",
        "detail": f"{sched_count} games" if sched_count else "Released ~May",
    })

    # 4. Vegas lines (spreads + totals)
    vegas_count = _count(
        "SELECT COUNT(*) FROM schedule WHERE season=? AND spread_line IS NOT NULL",
        (season,),
    )
    sources.append({
        "name": "Vegas Lines",
        "status": "available" if vegas_count >= 256 else (
            "partial" if vegas_count > 0 else "unavailable"
        ),
        "detail": f"{vegas_count} games" if vegas_count else "Available ~August",
    })

    # 5. Current season game stats
    curr_stats = _count(
        "SELECT COUNT(*) FROM player_weekly_stats WHERE season=?", (season,)
    )
    sources.append({
        "name": f"{season} Game Stats",
        "status": "available" if curr_stats > 500 else (
            "partial" if curr_stats > 0 else "unavailable"
        ),
        "detail": f"{curr_stats:,} player-weeks" if curr_stats else "Season hasn't started",
    })

    # 6. Rookie / NFL Draft data (from parquet, not DB)
    draft_class = load_draft_class(season)
    rookie_count = len(draft_class)
    sources.append({
        "name": f"{season} NFL Draft Picks",
        "status": "available" if rookie_count > 0 else "unavailable",
        "detail": f"{rookie_count} skill picks" if rookie_count else "Available after NFL Draft (~April)",
    })

    # 7. NGS (Next Gen Stats)
    try:
        ngs_count = _count(
            "SELECT COUNT(*) FROM ngs_passing WHERE season=?", (season,)
        )
    except sqlite3.OperationalError:
        ngs_count = 0
    sources.append({
        "name": "Next Gen Stats",
        "status": "available" if ngs_count > 100 else "unavailable",
        "detail": f"{ngs_count:,} records" if ngs_count else "In-season only (2018+)",
    })

    # 8. Injury reports
    try:
        inj_count = _count(
            "SELECT COUNT(*) FROM injuries WHERE season=?", (season,)
        )
    except sqlite3.OperationalError:
        inj_count = 0
    sources.append({
        "name": "Injury Reports",
        "status": "available" if inj_count > 50 else "unavailable",
        "detail": f"{inj_count:,} reports" if inj_count else "In-season only",
    })

    # 9. Walk-forward backtest predictions
    csv = _latest_predictions_csv(season)
    sources.append({
        "name": "ML Backtest Predictions",
        "status": "available" if csv else "unavailable",
        "detail": str(csv.name) if csv else "Requires full season of data",
    })

    conn.close()
    return sources


def build_team_tendencies(season: int) -> dict:
    """Build team-level rushing/passing tendency vectors from prior season."""
    import sqlite3
    from config.settings import DB_PATH

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT team,
               SUM(rushing_attempts) as rush_att,
               SUM(passing_attempts) as pass_att,
               SUM(rushing_yards) as rush_yd,
               SUM(passing_yards) as pass_yd,
               SUM(targets) as total_targets
        FROM player_weekly_stats
        WHERE season = ? AND team IS NOT NULL AND team != ''
        GROUP BY team
    """, (prior,)).fetchall()

    teams = {}
    for team, rush_att, pass_att, rush_yd, pass_yd, total_tgt in rows:
        rush_att = rush_att or 0
        pass_att = pass_att or 0
        total = rush_att + pass_att
        if total == 0:
            continue
        teams[team] = {
            "rush_pct": round(rush_att / total * 100),
            "pass_pct": round(pass_att / total * 100),
            "rush_yd": round(rush_yd or 0),
            "pass_yd": round(pass_yd or 0),
            "rush_att": round(rush_att),
            "pass_att": round(pass_att),
        }

    # Classify tendency
    for t in teams.values():
        if t["rush_pct"] >= 45:
            t["tendency"] = "Run-heavy"
        elif t["rush_pct"] >= 40:
            t["tendency"] = "Balanced"
        elif t["rush_pct"] >= 35:
            t["tendency"] = "Pass-lean"
        else:
            t["tendency"] = "Pass-heavy"

    conn.close()
    return teams


def build_usage_roles(season: int) -> dict:
    """Build player roles from actual usage data + rookie draft capital."""
    import sqlite3
    from config.settings import DB_PATH
    from collections import defaultdict

    prior = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT pws.team, p.name, p.position,
               SUM(pws.targets) as total_targets,
               SUM(pws.rushing_attempts) as total_carries,
               SUM(pws.snap_count) as total_snaps,
               SUM(pws.receptions) as total_rec,
               COUNT(DISTINCT pws.week) as games
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season = ?
          AND p.position IN ('QB', 'RB', 'WR', 'TE')
          AND pws.team IS NOT NULL AND pws.team != ''
        GROUP BY pws.team, pws.player_id
        HAVING games >= 4
        ORDER BY pws.team, total_snaps DESC
    """, (prior,)).fetchall()

    conn.close()

    team_pos = defaultdict(lambda: defaultdict(list))
    for team, name, pos, tgt, carries, snaps, rec, games in rows:
        team_pos[team][pos].append({
            "name": name,
            "targets": tgt or 0,
            "carries": carries or 0,
            "snaps": snaps or 0,
            "receptions": rec or 0,
            "games": games,
            "is_rookie": False,
            "draft_round": 0,
        })

    # Inject rookies from draft class into their teams
    draft_class = load_draft_class(season)
    rookie_curve = build_rookie_projection_curve()
    for _, rk in draft_class.iterrows():
        team = rk["team"]
        pos = rk["position"]
        rnd = int(rk["round"])
        # Estimate usage from draft capital — Round 1 gets starter-level usage
        est_fp = rookie_curve.get((pos, rnd), 50)
        team_pos[team][pos].append({
            "name": rk["name"],
            "targets": est_fp * 0.3 if pos != "RB" else est_fp * 0.1,
            "carries": est_fp * 0.5 if pos == "RB" else 0,
            "snaps": est_fp * 10,
            "receptions": 0,
            "games": 0,
            "is_rookie": True,
            "draft_round": rnd,
        })

    # Rank within team+position by primary usage metric
    roles = {}
    for team, positions in team_pos.items():
        team_targets = sum(p["targets"] for pos_list in positions.values() for p in pos_list)
        team_carries = sum(p["carries"] for pos_list in positions.values() for p in pos_list)

        for pos, players in positions.items():
            if pos == "RB":
                players.sort(key=lambda x: -(x["carries"] + x["targets"]))
            elif pos in ("WR", "TE"):
                # Weight targets 10x over snaps — snaps have data gaps and rookies
                # get inflated snap estimates that would otherwise bury veterans
                # with missing snap counts (e.g. MHJ: 74 real tgts, 0 recorded snaps).
                players.sort(key=lambda x: -(x["targets"] * 10 + x["snaps"]))
            else:
                # QB: sort by snaps (QBs have 0 targets)
                players.sort(key=lambda x: -x["snaps"])

            for rank, p in enumerate(players, 1):
                tgt_share = round(p["targets"] / team_targets * 100) if team_targets else 0
                carry_share = round(p["carries"] / team_carries * 100) if team_carries else 0

                if p["is_rookie"]:
                    note = f"Rookie R{p['draft_round']}"
                elif pos == "RB":
                    if carry_share >= 60:
                        note = "Bellcow"
                    elif carry_share >= 35:
                        note = "Lead back"
                    elif carry_share >= 20:
                        note = "Committee"
                    else:
                        note = "Backup"
                elif pos in ("WR", "TE"):
                    if tgt_share >= 20:
                        note = "Alpha"
                    elif tgt_share >= 12:
                        note = "Starter"
                    elif tgt_share >= 6:
                        note = "Rotational"
                    else:
                        note = "Depth"
                else:
                    note = "Starter" if rank == 1 else "Backup"

                roles[p["name"]] = {
                    "role": f"{pos}{rank}",
                    "tgt_share": tgt_share,
                    "carry_share": carry_share,
                    "usage": note,
                    "games": p["games"],
                    "team": team,
                }

    return roles


def compute_team_pos_profiles(season: int, lookback: int = 3) -> dict:
    """
    Return {(team, position, rank): {"avg_tgt_share": int, "avg_carry_share": int}}

    rank is 1-based: 1 = lead player (RB1/WR1), 2 = second player, etc.
    Averaged over the past `lookback` seasons so a team's scheme tendencies
    (e.g. KC giving 6% target share to RB1, DET giving 14%) are reflected.

    Keyed with rank so callers can look up the Nth-player profile matching
    the player's projected role on their new team (RB2 gets the historical
    RB2 share, not the RB1 share).
    """
    import sqlite3
    from config.settings import DB_PATH

    start_season = season - lookback
    end_season = season - 1
    conn = sqlite3.connect(str(DB_PATH))

    player_rows = conn.execute("""
        SELECT pws.team, p.position, pws.season, p.name,
               SUM(pws.targets)           AS tgts,
               SUM(pws.rushing_attempts)  AS carries,
               COUNT(DISTINCT pws.week)   AS games
        FROM player_weekly_stats pws
        JOIN players p ON pws.player_id = p.player_id
        WHERE pws.season BETWEEN ? AND ?
          AND p.position IN ('QB','RB','WR','TE')
          AND pws.team IS NOT NULL AND pws.team != ''
        GROUP BY pws.team, p.position, pws.season, p.name
        HAVING games >= 4
    """, (start_season, end_season)).fetchall()

    team_tot_rows = conn.execute("""
        SELECT team, season,
               SUM(targets)           AS team_tgts,
               SUM(rushing_attempts)  AS team_carries
        FROM player_weekly_stats
        WHERE season BETWEEN ? AND ?
          AND team IS NOT NULL AND team != ''
        GROUP BY team, season
    """, (start_season, end_season)).fetchall()

    conn.close()

    team_tot = {
        (t, s): (max(tgts or 0, 1), max(carries or 0, 1))
        for t, s, tgts, carries in team_tot_rows
    }

    from collections import defaultdict
    # (team, pos, season) → list of {tgt_share, carry_share}, unsorted
    tps: dict = defaultdict(list)
    for team, pos, s, _name, tgts, carries, _games in player_rows:
        tot_tgts, tot_carries = team_tot.get((team, s), (1, 1))
        tps[(team, pos, s)].append({
            "tgt_share":   (tgts or 0) * 100.0 / tot_tgts,
            "carry_share": (carries or 0) * 100.0 / tot_carries,
        })

    # Per (team, pos, season) rank players (RB by carries, others by targets)
    # and accumulate each rank's shares across seasons.
    # Key: (team, pos, rank) → list of per-season share dicts
    by_rank: dict = defaultdict(list)
    for (team, pos, _s), players in tps.items():
        if pos == "RB":
            players.sort(key=lambda x: -x["carry_share"])
        else:
            players.sort(key=lambda x: -x["tgt_share"])
        for rank, p in enumerate(players, 1):
            by_rank[(team, pos, rank)].append(p)

    result = {}
    for (team, pos, rank), entries in by_rank.items():
        result[(team, pos, rank)] = {
            "avg_tgt_share":   round(sum(p["tgt_share"]   for p in entries) / len(entries)),
            "avg_carry_share": round(sum(p["carry_share"] for p in entries) / len(entries)),
        }
    return result


def build_board_data(season: int):
    """Build the full draft board with spread, VORP, and projections."""
    adp_df = load_adp_board(season)

    def _player_key(name: str, position: str, team: str = ""):
        return (
            _first_initial(name),
            _normalize(_last_token(name)),
            position,
            team or "",
        )

    def _player_identity(name: str):
        return (
            _first_initial(name),
            _normalize(_last_token(name)),
        )

    csv = _latest_predictions_csv(season)
    if csv:
        projections = load_model_projections(csv, ranking="season_sum", season=season)
    else:
        # No backtest for this season yet — use prior season's ML predictions
        # as the projection basis (much better than raw PPG * 17),
        # then fill in rookies/unmatched with ECR-implied projections
        prior_csv = _latest_predictions_csv(season - 1)
        if prior_csv:
            ml_proj = load_model_projections(
                prior_csv, ranking="season_sum", season=season - 1
            )
            ml_proj["actual_total"] = 0.0
            # Merge with preseason projections for rookies
            fallback = load_preseason_projections(season, adp_df=adp_df)
            if not fallback.empty:
                # Keep ML projections, add only players not already covered by
                # the ML artifact after robust normalized matching.
                ml_player_ids = {
                    str(r.get("player_id") or "")
                    for _, r in ml_proj.iterrows()
                    if str(r.get("player_id") or "")
                }
                ml_keys = {
                    _player_key(r["name"], r["position"], r.get("team", ""))
                    for _, r in ml_proj.iterrows()
                }
                ml_identity_team_keys = {
                    (_player_identity(r["name"]), r.get("team", "") or "")
                    for _, r in ml_proj.iterrows()
                }

                def _should_keep_fallback_row(row) -> bool:
                    row_pid = str(row.get("player_id") or "")
                    row_identity = _player_identity(row["name"])
                    row_identity_team = (row_identity, row.get("team", "") or "")
                    if row_pid:
                        return (
                            row_pid not in ml_player_ids
                            and row_identity_team not in ml_identity_team_keys
                        )

                    row_key = _player_key(
                        row["name"], row["position"], row.get("team", "")
                    )
                    return (
                        row_key not in ml_keys
                        and row_identity_team not in ml_identity_team_keys
                    )

                rookies = fallback[
                    [_should_keep_fallback_row(r) for _, r in fallback.iterrows()]
                ]
                projections = pd.concat([ml_proj, rookies], ignore_index=True)
            else:
                projections = ml_proj
        else:
            projections = load_preseason_projections(season, adp_df=adp_df)

    board = build_draft_board(adp_df, projections)
    spread_results = compute_spread(board)
    validation = validate_spread_direction(spread_results, min_spread=10)

    # Build VORP values
    if not projections.empty:
        vorp_series = _apply_vorp(projections, basis_col="pred_total")
        if "player_id" in projections.columns:
            vorp_map = dict(zip(projections["player_id"].astype(str), vorp_series))
        else:
            vorp_map = dict(zip(projections["name"], vorp_series))
    else:
        vorp_map = {}

    has_actuals = csv is not None

    # Real usage-based roles from prior season play-by-play
    raw_usage_roles = build_usage_roles(season)
    team_tendencies = build_team_tendencies(season)

    usage_roles = {}  # (norm_key, position) -> role data
    for name, data in raw_usage_roles.items():
        pos = data["role"][:2]  # "WR" from "WR3", "RB" from "RB1", etc.
        key = (_norm_key(name), pos)
        # Keep the one with higher usage
        existing = usage_roles.get(key)
        if existing is None or (data["tgt_share"] + data["carry_share"]) > (existing["tgt_share"] + existing["carry_share"]):
            usage_roles[key] = data

    # Fallback: projection-based role for players with no prior-season data
    team_groups = {}
    for sr in spread_results:
        if sr.ecr > 300:
            continue
        team_groups.setdefault(sr.team, []).append(sr)
    proj_role_map = {}
    for team, group in team_groups.items():
        team_pos = {}
        for sr in group:
            team_pos.setdefault(sr.position, []).append(sr)
        for pos, pos_players in team_pos.items():
            pos_players.sort(key=lambda s: -s.model_projection)
            for rank, sr in enumerate(pos_players, 1):
                proj_role_map[sr.name] = f"{pos}{rank}"

    # Build current-roster team lookup to correct stale ADP team labels (e.g. "FA"
    # for players who signed after the last ADP scrape). Keyed by _norm_key(name).
    roster_team_override: dict = {}
    try:
        import nfl_data_py as _nfl_rt
        _rt = _nfl_rt.import_seasonal_rosters([season])
        _rt = _rt[_rt["status"] == "ACT"][["player_name", "team"]].dropna()
        for _, row in _rt.iterrows():
            roster_team_override[_norm_key(str(row["player_name"]))] = str(row["team"])
    except Exception:
        pass

    # Historical team×position usage profiles for team-changer share substitution
    team_pos_profiles = compute_team_pos_profiles(season)

    # Compute projection adjustments
    print("  Computing adjustments (age, team changes, usage trends, SOS)...")
    age_adj = compute_age_adjustments(season)
    team_change_adj = compute_team_changes(season)
    trend_adj = compute_usage_trends(season)
    regression_adj = compute_regression_adjustments(season)
    injury_adj = compute_injury_risk(season)
    breakout_adj = compute_breakout_candidates(season)
    def_rankings = compute_defense_rankings(season)

    # Load manual adjustments (offseason moves, scheme changes, etc.)
    # Matched by case-insensitive full-name substring to avoid norm_key collisions
    # (e.g. "Bijan Robinson" vs "Brian Robinson Jr." share the same norm key).
    manual_adj_path = PROJECT_ROOT / "data" / f"manual_adjustments_{season}.json"
    manual_adjs_raw = []
    if manual_adj_path.exists():
        for entry in json.load(manual_adj_path.open()):
            manual_adjs_raw.append({
                "player": entry["player"].lower(),
                "mult": entry.get("mult", 1.0),
                "note": entry.get("note", ""),
            })

    def _find_manual_adj(name: str):
        name_lower = name.lower()
        for entry in manual_adjs_raw:
            # Match if the entry name appears in the player name or vice versa
            if entry["player"] in name_lower or name_lower in entry["player"]:
                return entry
        return None

    # Load market-implied 2025 projections for edge signal on draft board.
    # Keyed by _norm_key(name) → market_fp float. Missing file = no signal shown.
    mkt_proj_lookup: dict = {}
    market_projection_season = None
    mkt_proj_path = PROJECT_ROOT / "docs" / "data" / "market_projections.json"
    if mkt_proj_path.exists():
        mkt_raw = json.loads(mkt_proj_path.read_text(encoding="utf-8"))
        market_projection_season = mkt_raw.get("season")
        for pname, rec in mkt_raw.get("players", {}).items():
            fp = rec.get("market_fp", 0)
            # Key by full name lowercase — unambiguous, no collisions between
            # players who share first initial + last name (e.g. Kaytron/Keenan Allen).
            # Board player names are always full names so full-name matching is sufficient.
            mkt_proj_lookup[pname.lower().strip()] = fp

    anchor_context = _build_market_anchor_context(spread_results, mkt_proj_lookup)
    headshot_lookup = load_headshot_lookup()

    # Serialize board for JS
    players = []
    for i, sr in enumerate(spread_results):
        if sr.ecr > 300:
            continue
        # Use real usage role if available, else projection-based
        ur = usage_roles.get((_norm_key(sr.name), sr.position))
        if ur:
            team_role = ur["role"]
            usage_note = ur["usage"]
            tgt_share = ur["tgt_share"]
            carry_share = ur["carry_share"]
            # Injury-return override: missed 7+ games + high consensus → incoherent labels
            if ur.get("games", 17) <= 9 and sr.ecr < 100:
                usage_note = "Returning"

            # Team-changer adjustment: prior-season stats reflect old team's scheme.
            # Use ADP team (sr.team) as the primary current-team source — it already
            # reflects offseason moves and avoids norm_key collisions that occur
            # in roster_team_override when two players share first-initial + last-name.
            # Only fall back to roster_team_override when ADP says FA (player recently signed).
            prior_team = ur.get("team", "")
            current_team = (
                sr.team
                if sr.team and sr.team not in ("FA", "", None)
                else roster_team_override.get(_norm_key(sr.name)) or ""
            )
            if prior_team and current_team and prior_team != current_team:
                # Use the player's projected rank on the new team to look up
                # the right rank-bucket (RB2 gets historical RB2 share, not RB1).
                # proj_role_map is keyed by current ADP team so gives new-team rank.
                new_role = proj_role_map.get(sr.name, "")
                try:
                    new_rank = int(new_role[2:]) if len(new_role) > 2 else 1
                except (ValueError, IndexError):
                    new_rank = 1
                profile = (
                    team_pos_profiles.get((current_team, sr.position, new_rank))
                    or team_pos_profiles.get((current_team, sr.position, 1))
                )
                if profile:
                    tgt_share = profile["avg_tgt_share"]
                    carry_share = profile["avg_carry_share"]
                    # Recompute usage note from new team's expected shares
                    if usage_note != "Returning":
                        if sr.position == "RB":
                            if carry_share >= 60:
                                usage_note = "Bellcow"
                            elif carry_share >= 35:
                                usage_note = "Lead back"
                            elif carry_share >= 20:
                                usage_note = "Committee"
                            else:
                                usage_note = "Backup"
                        elif sr.position in ("WR", "TE"):
                            if tgt_share >= 20:
                                usage_note = "Alpha"
                            elif tgt_share >= 12:
                                usage_note = "Starter"
                            elif tgt_share >= 6:
                                usage_note = "Rotational"
                            else:
                                usage_note = "Depth"
        else:
            team_role = proj_role_map.get(sr.name, "")
            usage_note = "Rookie" if "rookie_" in str(sr.name).lower() else "Proj"
            tgt_share = 0
            carry_share = 0

        # Team tendency
        tt = team_tendencies.get(sr.team, {})

        player_age = None
        aa = age_adj.get((sr.name, sr.position))
        if aa is None:
            aa = next(
                (data for (name, pos), data in age_adj.items()
                 if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
                None,
            )
        if aa:
            player_age = aa["age"]
        tc = next(
            (data for (name, pos), data in team_change_adj.items()
             if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
            None,
        )
        # Synthesize direction-aware tc for current-year offseason moves not yet in DB.
        # compute_team_changes() only detects prior2→prior moves; new offseason moves
        # (e.g., SEA→KC in the 2026 offseason) return tc=None. Synthesize from profiles.
        if tc is None and ur is not None and prior_team and current_team and prior_team != current_team:
            old_rank_str = ur.get("role", f"{sr.position}1")
            new_rank_str = proj_role_map.get(sr.name, f"{sr.position}1")
            try:
                old_rank = int(old_rank_str[2:]) if len(old_rank_str) > 2 and old_rank_str[2:].isdigit() else 1
            except (ValueError, IndexError):
                old_rank = 1
            try:
                new_rank = int(new_rank_str[2:]) if len(new_rank_str) > 2 and new_rank_str[2:].isdigit() else 1
            except (ValueError, IndexError):
                new_rank = 1

            old_p = team_pos_profiles.get((prior_team, sr.position, old_rank))
            new_p = team_pos_profiles.get((current_team, sr.position, new_rank))

            if old_p and new_p:
                if sr.position == "RB":
                    old_u = old_p["avg_tgt_share"] * 2.5 + old_p["avg_carry_share"]
                    new_u = new_p["avg_tgt_share"] * 2.5 + new_p["avg_carry_share"]
                elif sr.position in ("WR", "TE"):
                    old_u = float(old_p["avg_tgt_share"])
                    new_u = float(new_p["avg_tgt_share"])
                else:
                    old_u = new_u = 0.0

                if old_u > 0 and new_u > 0:
                    raw_mult = new_u / old_u
                    mult = round(max(0.80, min(1.25, raw_mult)), 3)
                    tc = {"from": prior_team, "to": current_team, "mult": mult}

        tr = next(
            (data for (name, pos), data in trend_adj.items()
             if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
            None,
        )
        ra = next(
            (data for (name, pos), data in regression_adj.items()
             if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
            None,
        )
        ia = next(
            (data for (name, pos), data in injury_adj.items()
             if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
            None,
        )
        ba = next(
            (data for (name, pos), data in breakout_adj.items()
             if _norm_key(name) == _norm_key(sr.name) and pos == sr.position),
            None,
        )

        sos_mult, sos_label = compute_sos_adjustment(
            sr.team, sr.position, def_rankings
        )

        manual = _find_manual_adj(sr.name)
        market_proj, market_source = _resolve_market_anchor(sr, anchor_context)
        calibration = _calibrate_projection(
            sr=sr,
            market_proj=market_proj,
            market_source=market_source,
            age_data=aa,
            team_change_data=tc,
            trend_data=tr,
            regression_data=ra,
            injury_data=ia,
            breakout_data=ba,
            manual_data=manual,
            sos_mult=sos_mult,
            sos_label=sos_label,
        )
        adj_pct = round(
            (
                calibration["final_display_projection"]
                / max(calibration["calibrated_projection"], 1.0) - 1.0
            ) * 100
        )
        adj_reasons = [
            f"{item['label']} {item['delta']:+.1f}"
            for item in calibration["adjustment_breakdown"]
            if item["signal"] != "market_clamp"
        ]

        # Correct stale ADP team labels: if ADP says "FA" but the live roster
        # has the player on a team, use the live roster value.
        board_team = sr.team
        if not board_team or board_team == "FA":
            board_team = roster_team_override.get(_norm_key(sr.name), sr.team)

        players.append({
            "id": i,
            "n": sr.name,
            "p": sr.position,
            "t": board_team,
            "img": resolve_headshot_url(sr, headshot_lookup),
            "ecr": round(sr.ecr, 1),
            "mr": sr.model_rank,
            "sp": sr.rank_spread,
            "proj": round(calibration["final_display_projection"], 1),
            "rawProj": round(calibration["raw_projection"], 1),
            "marketProj": round(calibration["market_consensus_projection"], 1),
            "adpDivergencePct": round(calibration["adp_divergence_pct"] * 100, 1),
            "calibratedProj": round(calibration["calibrated_projection"], 1),
            "adjDelta": round(calibration["adjustment_delta"], 1),
            "adjBreakdown": calibration["adjustment_breakdown"],
            "calibrationFlags": calibration["calibration_flags"],
            "adjPct": adj_pct,
            "adjR": ", ".join(adj_reasons) if adj_reasons else "",
            "age": player_age,
            "vorp": round(
                vorp_map.get(str(getattr(sr, "player_id", "")), vorp_map.get(sr.name, 0)),
                1,
            ),
            "role": team_role,
            "usage": usage_note,
            "ts": tgt_share,
            "cs": carry_share,
            "tt": tt.get("tendency", ""),
            "trp": tt.get("rush_pct", 0),
            "tpp": tt.get("pass_pct", 0),
            "act": round(sr.actual_total, 1) if has_actuals else None,
            "w": sr.model_wins if has_actuals else None,
            "adj_note": manual["note"] if manual else "",
            "why": calibration["why"],
            "mkt25": _mkt_get(mkt_proj_lookup, sr.name),
            "edge": round(calibration["final_display_projection"] - _mkt_get(mkt_proj_lookup, sr.name))
                if _mkt_get(mkt_proj_lookup, sr.name) is not None else None,
        })

    # Recompute vs-experts signal from calibrated projection rank, not raw ML rank.
    # Raw ML rank is pre-calibration and produces misleading signals (e.g. Rodgers
    # at mr=37 showing +206 despite a 103-pt calibrated projection).
    proj_sorted = sorted(players, key=lambda p: -p["proj"])
    cal_rank_map = {p["n"]: i + 1 for i, p in enumerate(proj_sorted)}
    for p in players:
        cal_rank = cal_rank_map.get(p["n"], 999)
        p["sp"] = round(p["ecr"]) - cal_rank

    calibration_summary = _summarize_board_calibration(players)
    rb_wr_gap_excess = calibration_summary["summary"].get("rb_wr_gap_excess", 0.0)
    tolerance = CALIBRATION_POLICY["position_bias_tolerance"]
    if rb_wr_gap_excess > tolerance:
        for player in players:
            if player["p"] == "RB":
                player["calibrationFlags"]["positionBiasCheck"] = True
    elif rb_wr_gap_excess < -tolerance:
        for player in players:
            if player["p"] == "WR":
                player["calibrationFlags"]["positionBiasCheck"] = True

    return {
        "players": players,
        "validation": {
            "n": validation["n"],
            "wins": validation.get("model_wins", 0),
            "acc": round(validation["accuracy"] * 100) if validation["n"] > 0 else 0,
        },
        "has_actuals": has_actuals,
        "season": season,
        "calibration": {
            "policy": CALIBRATION_POLICY,
            "market_projection_season": market_projection_season,
            **calibration_summary,
        },
        "board": board,
        "adp_df": adp_df,
    }


def build_vona_data(board, adp_df, max_slots=14):
    """Pre-compute VONA for each draft slot."""
    vona_all = {}
    for slot in range(1, min(max_slots + 1, TEAMS + 1)):
        raw = compute_vona(board, adp_df, slot, teams=TEAMS, rounds=ROUNDS)
        # Group by round, keep top 5 per round
        by_round = {}
        for r in raw:
            rd = r["round"]
            if rd not in by_round:
                by_round[rd] = []
            by_round[rd].append(r)

        slot_picks = []
        for rd in sorted(by_round.keys()):
            candidates = sorted(by_round[rd], key=lambda x: -x["net_value"])[:5]
            for c in candidates:
                slot_picks.append({
                    "rd": rd,
                    "pk": c["pick"],
                    "n": c["name"],
                    "p": c["position"],
                    "t": c.get("team", ""),
                    "av": round(c["avail_pct"] * 100),
                    "proj": round(c["model_proj"], 1),
                    "vona": round(c["vona"], 1),
                    "oc": round(c["opp_cost"], 1),
                    "ocp": c["opp_cost_pos"],
                    "net": round(c["net_value"], 1),
                })
        vona_all[slot] = slot_picks
    return vona_all


def build_scarcity_data(board_players, teams=10, rounds=15):
    """Compute above-replacement starters remaining by position at each pick.

    Returns:
        by_pick: {pick_str: {pos: {rem, top}}} for picks 1..teams*rounds
        cliffs:  {pos: first_pick_where_rem_drops_below_threshold}
        teams, rounds for the JS to use
    """
    POSITIONS = ["QB", "RB", "WR", "TE"]
    CLIFF_THRESHOLD = {"QB": 3, "RB": 4, "WR": 4, "TE": 3}

    sorted_p = sorted(board_players, key=lambda p: p["ecr"])
    total = teams * rounds

    by_pick = {}
    for pick in range(1, total + 1):
        remaining = sorted_p[pick - 1:]
        counts = {pos: 0 for pos in POSITIONS}
        top = {pos: 0.0 for pos in POSITIONS}
        for p in remaining:
            pos = p["p"]
            if pos in POSITIONS and p.get("vorp", 0) > 0:
                counts[pos] += 1
                if p["proj"] > top[pos]:
                    top[pos] = p["proj"]
        by_pick[str(pick)] = {
            pos: {"rem": counts[pos], "top": round(top[pos], 1)}
            for pos in POSITIONS
        }

    cliffs = {}
    for pos in POSITIONS:
        threshold = CLIFF_THRESHOLD[pos]
        cliffs[pos] = total + 1  # default: never hits cliff
        for pick in range(1, total + 1):
            if by_pick[str(pick)][pos]["rem"] < threshold:
                cliffs[pos] = pick
                break

    return {"by_pick": by_pick, "cliffs": cliffs, "teams": teams, "rounds": rounds}


def build_data_payloads(board_data, vona_data, scarcity_data, data_sources):
    """Return the JSON payloads served to the page at runtime."""
    return {
        "board":    board_data["players"],
        "calibration": board_data.get("calibration", {}),
        "vona":     vona_data,
        "scarcity": scarcity_data,
        "meta":  {
            "season":      board_data["season"],
            "validation":  board_data["validation"],
            "has_actuals": board_data["has_actuals"],
            "teams":       TEAMS,
            "rounds":      ROUNDS,
            "sources":     data_sources,
            "calibration": board_data.get("calibration", {}).get("summary", {}),
        },
    }


def generate_html(season):
    """Generate the HTML shell from template + CSS + JS (no data inlined)."""
    template = (SITE_DIR / "template.html").read_text(encoding="utf-8")
    style    = (SITE_DIR / "style.css").read_text(encoding="utf-8")
    app_js   = (SITE_DIR / "app.js").read_text(encoding="utf-8")

    html = template
    html = html.replace("{{STYLE_CSS}}",   style)
    html = html.replace("{{APP_JS}}",      app_js)
    html = html.replace("{{SEASON}}",      str(season))
    html = html.replace("{{PREV_SEASON}}", str(season - 1))
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate Draft Advisor HTML")
    parser.add_argument("--season", type=int, default=None, help="Season year")
    args = parser.parse_args()

    season = args.season
    if season is None:
        from config.settings import CURRENT_NFL_SEASON
        # Draft advisor targets the upcoming season; nfl_calendar returns the
        # last completed season (e.g. 2025 in May 2026), so default to +1.
        season = CURRENT_NFL_SEASON + 1

    print(f"Building draft advisor for {season}...")

    print("  Loading board data...")
    board_data = build_board_data(season)
    print(f"  {len(board_data['players'])} players loaded")

    print("  Computing VONA for all slots...")
    vona_data = build_vona_data(
        board_data["board"], board_data["adp_df"], max_slots=TEAMS
    )
    print(f"  VONA computed for {len(vona_data)} slots")

    print("  Computing positional scarcity curves...")
    scarcity_data = build_scarcity_data(board_data["players"], teams=TEAMS, rounds=ROUNDS)
    print(f"  Scarcity cliffs: { {k: f'pick {v}' for k, v in scarcity_data['cliffs'].items()} }")

    print("  Checking data sources...")
    data_sources = check_data_sources(season)
    avail = sum(1 for s in data_sources if s["status"] == "available")
    print(f"  {avail}/{len(data_sources)} sources available")

    print("  Generating HTML shell...")
    html = generate_html(season)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # season.html, not index.html. As of 2026-08-31 docs/index.html is a
    # redirect to draft.html -- the draft board absorbed the season view, since
    # it renders the same projections plus VOR, league settings, ADP and byes.
    # Writing this dashboard to index.html would silently clobber that redirect
    # on the next regeneration and resurrect a page the nav no longer links to.
    out_path = DOCS_DIR / "season.html"
    out_path.write_text(html, encoding="utf-8")

    data_dir = DOCS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_data_payloads(board_data, vona_data, scarcity_data, data_sources)
    for name, payload in payloads.items():
        path = data_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        size_kb = path.stat().st_size / 1024
        print(f"  Written to {path} ({size_kb:.0f} KB)")

    size_kb = out_path.stat().st_size / 1024
    print(f"  Written to {out_path} ({size_kb:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()

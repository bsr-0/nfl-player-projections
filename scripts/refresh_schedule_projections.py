#!/usr/bin/env python3
"""
Refresh 2026 schedule data, compute SoS, and update docs/data/board.json.

Run manually:
    python scripts/refresh_schedule_projections.py

Run automatically via GitHub Actions (update_schedule.yml).

Steps:
  1. Load 2025 team defensive stats to compute defensive strength ratings.
  2. Fetch the 2026 NFL schedule via nfl_data_py; fall back to local CSV.
  3. Compute per-team Strength of Schedule (SoS) from opponent def ratings.
  4. Write docs/data/schedule.json (used by the draft board UI).
  5. Patch sos_score / sos_rank / sos_tier onto every player in board.json.
  6. Update data/schedule_impact.json to reflect current status.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_RAW     = ROOT / "data" / "raw"
DATA_DIR     = ROOT / "data"
DOCS_DATA    = ROOT / "docs" / "data"
SEASON       = 2026
TOTAL_WEEKS  = 18

STATS_FILES  = sorted(DATA_RAW.glob("team_weekly_stats_*.csv"))
SCHEDULE_CSV = DATA_RAW / f"schedule_{SEASON}.csv"
TEMPLATE_CSV = DATA_RAW / f"schedule_{SEASON}_template.csv"
BOARD_JSON   = DOCS_DATA / "board.json"
SCHEDULE_OUT = DOCS_DATA / "schedule.json"
IMPACT_JSON  = DATA_DIR / "schedule_impact.json"

# nflverse abbreviations → board.json abbreviations
TEAM_ABBREV_MAP = {
    "LA":  "LAR",   # Los Angeles Rams
    "JAX": "JAC",   # Jacksonville Jaguars
}


def _normalize_abbrevs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["home_team"] = df["home_team"].replace(TEAM_ABBREV_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_ABBREV_MAP)
    return df


# ── 2. Schedule fetch ──────────────────────────────────────────────────────────

NFLVERSE_GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)


def fetch_schedule_nflverse(season: int) -> pd.DataFrame | None:
    """Fetch schedule directly from nflverse/nfldata games.csv on GitHub."""
    try:
        import urllib.request, io
        req = urllib.request.Request(
            NFLVERSE_GAMES_URL, headers={"User-Agent": "nfl-projections/1.0"}
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            df = pd.read_csv(io.BytesIO(r.read()))
        df = df[df["season"] == season].copy()
        df = df[df["game_type"] == "REG"]
        df = df[["season", "week", "home_team", "away_team"]].dropna()
        df["week"] = df["week"].astype(int)
        df = df[df["week"].between(1, TOTAL_WEEKS)]
        df = _normalize_abbrevs(df)
        # Cache locally so subsequent runs are instant
        df.to_csv(SCHEDULE_CSV, index=False)
        print(f"  nflverse: {len(df)} games fetched for {season}  (cached to {SCHEDULE_CSV.name})")
        return df
    except Exception as e:
        print(f"  nflverse fetch failed ({e})")
    return None


def fetch_schedule_nfl_data_py(season: int) -> pd.DataFrame | None:
    """Try nfl_data_py; return None on any failure."""
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
        if df is not None and not df.empty:
            df = df[["season", "week", "home_team", "away_team"]].dropna()
            df["week"] = df["week"].astype(int)
            df = df[df["week"].between(1, TOTAL_WEEKS)]
            df = _normalize_abbrevs(df)
            df.to_csv(SCHEDULE_CSV, index=False)
            print(f"  nfl_data_py: {len(df)} games fetched for {season}  (cached)")
            return df
    except Exception as e:
        print(f"  nfl_data_py unavailable ({e})")
    return None


def load_schedule_csv(path: Path) -> pd.DataFrame | None:
    """Load schedule from a previously cached or template CSV."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[["season", "week", "home_team", "away_team"]].dropna()
        df["week"] = df["week"].astype(int)
        df = _normalize_abbrevs(df)
        print(f"  CSV: {len(df)} games from {path.name}")
        return df
    except Exception as e:
        print(f"  Could not read CSV {path}: {e}")
    return None


def get_schedule(season: int) -> tuple[pd.DataFrame, int]:
    """
    Return (schedule_df, weeks_available).
    schedule_df has columns: season, week, home_team, away_team.
    Tries: cached CSV → nflverse GitHub → nfl_data_py → template CSV.
    """
    # Use cached full schedule if already present
    if SCHEDULE_CSV.exists():
        df = load_schedule_csv(SCHEDULE_CSV)
        if df is not None and df["week"].nunique() >= 17:
            weeks_available = df["week"].nunique()
            return df, weeks_available

    # Live fetch order: nflverse (no auth required) → nfl_data_py
    for fetcher in (fetch_schedule_nflverse, fetch_schedule_nfl_data_py):
        df = fetcher(season)
        if df is not None:
            return df, df["week"].nunique()

    # Last resort: cached partial or template
    for fallback in (SCHEDULE_CSV, TEMPLATE_CSV):
        df = load_schedule_csv(fallback)
        if df is not None:
            return df, df["week"].nunique()

    print("  No schedule data available; SoS will use neutral ratings.")
    return pd.DataFrame(columns=["season", "week", "home_team", "away_team"]), 0


# ── 1. Defensive strength ratings from the most recent complete season ─────────

def load_defensive_ratings(most_recent_seasons: int = 2) -> dict[str, float]:
    """
    Return per-team defensive strength rating from recent season(s).
    Rating = team_avg_pts_allowed / league_avg_pts_allowed.
    < 1.0 = strong defense; > 1.0 = weak defense (allows more than average).
    """
    frames = []
    for path in STATS_FILES:
        try:
            frames.append(pd.read_csv(path))
        except Exception as e:
            print(f"  Warning: could not read {path.name}: {e}")

    if not frames:
        print("  Warning: no team_weekly_stats_*.csv found; using neutral ratings.")
        return {}

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("season", ascending=False)
    recent_seasons = sorted(df["season"].unique(), reverse=True)[:most_recent_seasons]
    df = df[df["season"].isin(recent_seasons)]

    team_avg = df.groupby("team")["points_allowed"].mean()
    league_avg = team_avg.mean()
    ratings = (team_avg / league_avg).to_dict()
    print(f"  Defensive ratings computed from seasons {sorted(recent_seasons)}  "
          f"(league avg = {league_avg:.1f} pts/gm)")
    return ratings

# ── 3. Strength of Schedule ────────────────────────────────────────────────────

def build_team_matchups(schedule_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {team: [opponent, ...]} for all scheduled games."""
    matchups: dict[str, list[str]] = {}
    for _, row in schedule_df.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        matchups.setdefault(home, []).append(away)
        matchups.setdefault(away, []).append(home)
    return matchups


def compute_sos(matchups: dict[str, list[str]],
                def_ratings: dict[str, float]) -> dict[str, dict]:
    """
    Return per-team SoS data.

    sos_raw = avg opponent def_rating (higher = weaker opponents = easier schedule).
    sos_score 0-100: higher score = tougher schedule.
    sos_rank 1-32: 1 = hardest schedule.
    sos_tier: "tough" | "avg" | "easy"
    """
    NEUTRAL = 1.0  # fallback if a team's rating is missing

    raw: dict[str, float] = {}
    for team, opps in matchups.items():
        opp_ratings = [def_ratings.get(opp, NEUTRAL) for opp in opps]
        raw[team] = float(np.mean(opp_ratings)) if opp_ratings else NEUTRAL

    if not raw:
        return {}

    # Rank: lower raw → harder schedule → rank 1
    sorted_teams = sorted(raw, key=raw.get)   # ascending def_rating = strongest opps first
    rank_map = {team: i + 1 for i, team in enumerate(sorted_teams)}

    n = len(raw)
    results = {}
    for team, raw_sos in raw.items():
        rank = rank_map[team]
        # score 100 = hardest (rank 1); score 0 = easiest (rank n)
        score = round((n - rank) / max(n - 1, 1) * 100, 1)
        if score >= 67:
            tier = "tough"
        elif score <= 33:
            tier = "easy"
        else:
            tier = "avg"
        results[team] = {
            "sos_raw":   round(raw_sos, 4),
            "sos_score": score,
            "sos_rank":  rank,
            "sos_tier":  tier,
        }
    return results


# ── 4. Write schedule.json ─────────────────────────────────────────────────────

def write_schedule_json(schedule_df: pd.DataFrame,
                        def_ratings: dict[str, float],
                        sos_data: dict[str, dict],
                        weeks_available: int) -> None:
    """Write docs/data/schedule.json."""

    def_ratings_rounded = {t: round(v, 4) for t, v in sorted(def_ratings.items())}

    teams_out: dict[str, dict] = {}
    for team, sos in sorted(sos_data.items()):
        team_games = schedule_df[
            (schedule_df["home_team"] == team) | (schedule_df["away_team"] == team)
        ].sort_values("week")

        matchups = []
        for _, row in team_games.iterrows():
            opp   = row["away_team"] if row["home_team"] == team else row["home_team"]
            home  = row["home_team"] == team
            d_str = def_ratings.get(opp, 1.0)
            if d_str < 0.90:
                difficulty = "hard"
            elif d_str > 1.10:
                difficulty = "easy"
            else:
                difficulty = "avg"
            matchups.append({
                "week":       int(row["week"]),
                "opp":        opp,
                "home":       bool(home),
                "opp_def_rating": round(d_str, 3),
                "difficulty": difficulty,
            })

        teams_out[team] = {
            "sos_score":  sos["sos_score"],
            "sos_rank":   sos["sos_rank"],
            "sos_tier":   sos["sos_tier"],
            "matchups":   matchups,
        }

    payload = {
        "season":          SEASON,
        "generated_at":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weeks_available": weeks_available,
        "weeks_total":     TOTAL_WEEKS,
        "partial":         weeks_available < TOTAL_WEEKS,
        "def_ratings":     def_ratings_rounded,
        "teams":           teams_out,
    }

    SCHEDULE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_OUT.write_text(json.dumps(payload, indent=2))
    print(f"  Wrote {SCHEDULE_OUT}  ({len(teams_out)} teams, {weeks_available}/{TOTAL_WEEKS} weeks)")


# ── 5. Apply SoS adjustments and rebuild board.json ───────────────────────────

# Replacement-level player count per position (12-team PPR)
REPLACEMENT_LEVELS: dict[str, int] = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
MAX_SOS_ADJ = 0.08   # cap adjustment at ±8%
MIN_DELTA_WHY = 1.0  # only add a why-bullet if |delta| >= 1 pt


def _sos_multiplier(sos_score: float) -> float:
    """
    Convert a 0-100 SoS score to a proj multiplier.
    score=0 (easiest) → +8%;  score=50 (avg) → 0%;  score=100 (hardest) → -8%.
    """
    return 1.0 - (sos_score - 50.0) / 50.0 * MAX_SOS_ADJ


def _compute_vorp(board: list, pos_key: str = "p", proj_key: str = "proj") -> dict:
    """
    Return {player_id: vorp} using replacement-level proj per position.
    Replacement level = proj of the Nth player at that position.
    """
    by_pos: dict[str, list] = {}
    for p in board:
        by_pos.setdefault(p[pos_key], []).append(p)

    vorp_map: dict[int, float] = {}
    for pos, players in by_pos.items():
        n = REPLACEMENT_LEVELS.get(pos, 12)
        sorted_p = sorted(players, key=lambda x: x[proj_key], reverse=True)
        repl = sorted_p[n - 1][proj_key] if len(sorted_p) >= n else 0.0
        for p in players:
            vorp_map[p["id"]] = round(p[proj_key] - repl, 1)
    return vorp_map


def patch_board(sos_data: dict[str, dict], weeks_available: int) -> None:
    """
    Apply SoS multiplier to proj, recalculate vorp/mr/sp, add why bullets.
    Also stamps sos_score/sos_rank/sos_tier on every player.
    """
    with open(BOARD_JSON) as f:
        board = json.load(f)

    # ── Step A: apply SoS multiplier to proj ──
    adjusted = 0
    for p in board:
        team = p.get("t", "")
        sos  = sos_data.get(team, {"sos_score": 50.0, "sos_rank": 16, "sos_tier": "avg",
                                   "sos_raw": 1.0})
        p["sos_score"] = sos["sos_score"]
        p["sos_rank"]  = sos["sos_rank"]
        p["sos_tier"]  = sos["sos_tier"]

        if team not in sos_data:
            continue

        mult  = _sos_multiplier(sos["sos_score"])
        delta = round(p["proj"] * (mult - 1.0), 1)
        p["proj"] = round(p["proj"] * mult, 1)
        adjusted += 1

        # Remove any stale schedule why-bullet, then add updated one
        why = [w for w in (p.get("why") or [])
               if "schedule" not in w.lower() and "sos" not in w.lower()]
        if abs(delta) >= MIN_DELTA_WHY:
            delta_pgm = round(delta / 17, 2)
            sign  = "+" if delta_pgm > 0 else ""
            tier  = sos["sos_tier"]
            rank  = sos["sos_rank"]
            n_wks = weeks_available
            why.append(
                f"2026 schedule: {sign}{delta_pgm} pts/gm ({tier} slate, "
                f"SoS rank {rank}/32, {n_wks}-wk data)"
            )
        p["why"] = why

    # ── Step B: recalculate VORP with adjusted proj ──
    vorp_map = _compute_vorp(board)
    for p in board:
        p["vorp"] = vorp_map.get(p["id"], p.get("vorp", 0))

    # ── Step C: re-rank globally by adjusted proj → update mr and sp ──
    ranked = sorted(board, key=lambda p: p["proj"], reverse=True)
    rank_map = {p["id"]: i + 1 for i, p in enumerate(ranked)}
    for p in board:
        p["mr"] = rank_map[p["id"]]
        # sp = how many spots we rank a player above/below consensus
        # positive = we like them more than consensus (sleeper signal)
        ecr = p.get("ecr")
        if ecr is not None:
            p["sp"] = round(ecr - p["mr"])

    BOARD_JSON.write_text(json.dumps(board, separators=(",", ":")))
    print(f"  Adjusted proj for {adjusted}/{len(board)} players")
    print(f"  Recalculated vorp, mr, sp for all {len(board)} players")


# ── 6. Update schedule_impact.json ────────────────────────────────────────────

def update_impact_json(weeks_available: int, sos_data: dict) -> None:
    impact = {
        "schedule_incorporated": weeks_available > 0,
        "season":                SEASON,
        "weeks_available":       weeks_available,
        "weeks_total":           TOTAL_WEEKS,
        "partial":               weeks_available < TOTAL_WEEKS,
        "teams_with_sos":        len(sos_data),
        "generated_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reason": (
            f"Schedule incorporated from {weeks_available}/{TOTAL_WEEKS} weeks."
            if weeks_available > 0
            else "Schedule not yet available; re-run when nfl_data_py has 2026 data."
        ),
    }
    IMPACT_JSON.write_text(json.dumps(impact, indent=2))
    print(f"  Updated {IMPACT_JSON}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  NFL {SEASON} Schedule + SoS Refresh  —  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'='*60}\n")

    print("1/5  Loading defensive ratings from 2025 team stats…")
    def_ratings = load_defensive_ratings()

    print("\n2/5  Fetching 2026 schedule…")
    schedule_df, weeks_available = get_schedule(SEASON)

    print(f"\n3/5  Computing Strength of Schedule ({weeks_available} weeks available)…")
    matchups  = build_team_matchups(schedule_df)
    sos_data  = compute_sos(matchups, def_ratings)
    if sos_data:
        tiers = {t: v["sos_tier"] for t, v in sos_data.items()}
        easy  = [t for t, tier in tiers.items() if tier == "easy"]
        tough = [t for t, tier in tiers.items() if tier == "tough"]
        print(f"  Easy schedules : {', '.join(sorted(easy))}")
        print(f"  Tough schedules: {', '.join(sorted(tough))}")
    else:
        print("  No schedule data — all teams get neutral SoS.")
        # Seed with neutral values for all teams that appear in board.json
        try:
            with open(BOARD_JSON) as f:
                board = json.load(f)
            for p in board:
                t = p.get("t", "")
                if t and t not in sos_data:
                    sos_data[t] = {"sos_score": 50.0, "sos_rank": 16, "sos_tier": "avg"}
        except Exception:
            pass

    print("\n4/5  Writing docs/data/schedule.json…")
    write_schedule_json(schedule_df, def_ratings, sos_data, weeks_available)

    print("\n5/5  Applying SoS adjustments + rebuilding docs/data/board.json…")
    patch_board(sos_data, weeks_available)
    update_impact_json(weeks_available, sos_data)

    print(f"\n{'='*60}")
    status = "PARTIAL" if weeks_available < TOTAL_WEEKS else "COMPLETE"
    print(f"  Done — schedule status: {status} ({weeks_available}/{TOTAL_WEEKS} weeks)")
    if weeks_available < TOTAL_WEEKS:
        print("  Re-run once nfl_data_py publishes the full 2026 schedule.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

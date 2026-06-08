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
TEMPLATE_CSV = DATA_RAW / f"schedule_{SEASON}_template.csv"
BOARD_JSON   = DOCS_DATA / "board.json"
SCHEDULE_OUT = DOCS_DATA / "schedule.json"
IMPACT_JSON  = DATA_DIR / "schedule_impact.json"


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


# ── 2. Schedule fetch ──────────────────────────────────────────────────────────

def fetch_schedule_nfl_data_py(season: int) -> pd.DataFrame | None:
    """Try nfl_data_py first; return None on any failure."""
    try:
        import nfl_data_py as nfl
        df = nfl.import_schedules([season])
        if df is not None and not df.empty:
            df = df[["season", "week", "home_team", "away_team"]].dropna()
            df = df[df["week"].between(1, TOTAL_WEEKS)]
            print(f"  nfl_data_py: {len(df)} games fetched for {season}")
            return df
    except Exception as e:
        print(f"  nfl_data_py unavailable ({e}); trying local CSV fallback.")
    return None


def load_schedule_csv(path: Path) -> pd.DataFrame | None:
    """Load schedule from local CSV template."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[["season", "week", "home_team", "away_team"]].dropna()
        print(f"  CSV fallback: {len(df)} games from {path.name}")
        return df
    except Exception as e:
        print(f"  Could not read CSV {path}: {e}")
    return None


def get_schedule(season: int) -> tuple[pd.DataFrame, int]:
    """
    Return (schedule_df, weeks_available).
    schedule_df has columns: season, week, home_team, away_team.
    """
    df = fetch_schedule_nfl_data_py(season)
    if df is None:
        df = load_schedule_csv(TEMPLATE_CSV)
    if df is None:
        print("  No schedule data available; SoS will use neutral ratings.")
        return pd.DataFrame(columns=["season", "week", "home_team", "away_team"]), 0

    weeks_available = df["week"].nunique()
    return df, weeks_available


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


# ── 5. Patch board.json ────────────────────────────────────────────────────────

def patch_board(sos_data: dict[str, dict]) -> None:
    """Add sos_score, sos_rank, sos_tier to every player in board.json."""
    with open(BOARD_JSON) as f:
        board = json.load(f)

    patched = 0
    for player in board:
        team = player.get("t", "")
        if team in sos_data:
            player["sos_score"] = sos_data[team]["sos_score"]
            player["sos_rank"]  = sos_data[team]["sos_rank"]
            player["sos_tier"]  = sos_data[team]["sos_tier"]
            patched += 1
        else:
            player["sos_score"] = 50.0
            player["sos_rank"]  = 16
            player["sos_tier"]  = "avg"

    BOARD_JSON.write_text(json.dumps(board, separators=(",", ":")))
    print(f"  Patched {patched}/{len(board)} players in board.json")


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

    print("\n5/5  Patching docs/data/board.json + schedule_impact.json…")
    patch_board(sos_data)
    update_impact_json(weeks_available, sos_data)

    print(f"\n{'='*60}")
    status = "PARTIAL" if weeks_available < TOTAL_WEEKS else "COMPLETE"
    print(f"  Done — schedule status: {status} ({weeks_available}/{TOTAL_WEEKS} weeks)")
    if weeks_available < TOTAL_WEEKS:
        print("  Re-run once nfl_data_py publishes the full 2026 schedule.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

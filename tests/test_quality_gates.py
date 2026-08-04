import pandas as pd

from src.data.quality_gates import run_quality_gates, EXPECTED_TEAMS


def _build_dataset(weeks: int = 6) -> pd.DataFrame:
    records = []
    positions = ["QB", "RB", "WR", "TE"]
    for week in range(1, weeks + 1):
        for team in sorted(EXPECTED_TEAMS):
            for pos in positions:
                player_id = f"{team}_{pos}_{week}"
                records.append(
                    {
                        "player_id": player_id,
                        "season": 2025,
                        "week": week,
                        "team": team,
                        "position": pos,
                        "snap_count": 30,
                        "targets": 5 if pos in {"WR", "TE"} else 0,
                        "carries": 8 if pos == "RB" else 0,
                        "passing_attempts": 30 if pos == "QB" else 0,
                        "fantasy_points": 10.0,
                    }
                )
    return pd.DataFrame(records)


def test_quality_gates_pass_for_healthy_dataset():
    df = _build_dataset()
    result = run_quality_gates(df, expected_season=2025, expected_week=6)
    assert result.passed is True
    assert result.report["status"] == "pass"


def test_quality_gates_fail_completeness_when_team_missing():
    df = _build_dataset()
    df = df[~((df["week"] == 6) & (df["team"] == "KC"))]
    result = run_quality_gates(df, expected_season=2025, expected_week=6)
    assert result.passed is False
    assert "KC" in result.report["checks"]["completeness"]["missing_teams"]


def test_quality_gates_fail_on_anomaly_drop():
    df = _build_dataset()
    # Severe row drop in the latest week should fail anomaly bounds.
    df = df[(df["week"] < 6) | ((df["week"] == 6) & (df["team"].isin(["BUF", "MIA"])))]
    result = run_quality_gates(df, expected_season=2025, expected_week=6)
    assert result.passed is False
    assert result.report["checks"]["anomalies"]["passed"] is False


def test_quality_gates_fail_freshness_for_stale_week():
    df = _build_dataset(weeks=5)
    result = run_quality_gates(df, expected_season=2025, expected_week=6)
    assert result.passed is False
    assert result.report["checks"]["freshness"]["passed"] is False

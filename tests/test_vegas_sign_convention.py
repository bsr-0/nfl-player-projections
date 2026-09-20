"""Pin the sign convention of the player-level Vegas features.

nflverse `schedule.spread_line` is the HOME team's expected margin (positive
= home favoured; corr(home margin, spread_line) = +0.43 on 2006-2025). The
player-level features derived from it use ONE convention everywhere:

    spread              negative = this team is favoured
    implied_team_total  (game_total - spread) / 2
    is_favorite         spread < 0
    win_probability     0.5 - spread / 28 (clipped)

Found 2026-09-19: `external_data.get_vegas_features` (the production path)
assigned every team its OPPONENT's implied total and inverted `spread`, so
`is_favorite`/`win_probability` were backwards and `implied_team_total`
correlated -0.14..-0.19 with the team's real points instead of +0.42. The
`feature_engineering` lookup fallback got the home side right and the away
side wrong, and `backtester._evaluation_frame` used a third sign for
`spread`. These tests exercise all producers on one game and require them
to agree.
"""
import pandas as pd
import pytest

# Home favoured by 3, total 45 -> home implied 24, away implied 21.
SCHED = pd.DataFrame({
    "game_id": ["2024_01_BBB_AAA"], "season": [2024], "week": [1],
    "home_team": ["AAA"], "away_team": ["BBB"],
    "spread_line": [3.0], "total_line": [45.0],
})
TEAMS = pd.DataFrame({
    "player_id": ["h", "a"], "position": ["RB", "RB"],
    "season": [2024, 2024], "week": [1, 1], "team": ["AAA", "BBB"],
})


def _check(frame: pd.DataFrame):
    by_team = frame.set_index("team")
    h, a = by_team.loc["AAA"], by_team.loc["BBB"]
    assert h.spread == -3.0 and a.spread == 3.0
    assert h.implied_team_total == 24.0 and a.implied_team_total == 21.0
    assert h.game_total == 45.0 and a.game_total == 45.0
    assert h.implied_team_total + a.implied_team_total == h.game_total
    if "is_favorite" in frame.columns:
        assert h.is_favorite == 1 and a.is_favorite == 0
    if "win_probability" in frame.columns:
        assert h.win_probability > 0.5 > a.win_probability
        assert h.win_probability + a.win_probability == pytest.approx(1.0)


def test_external_data_production_path():
    from src.data.external_data import VegasLinesLoader
    out = VegasLinesLoader().get_vegas_features(TEAMS.copy(), SCHED.copy())
    _check(out)


def test_feature_engineering_early_return_derives_from_existing_spread():
    from src.data.external_data import VegasLinesLoader
    from src.features.feature_engineering import FeatureEngineer
    base = VegasLinesLoader().get_vegas_features(TEAMS.copy(), SCHED.copy())
    base = base.drop(columns=["implied_team_total", "is_favorite"])
    out = FeatureEngineer()._create_vegas_game_script_features(base)
    _check(out)


def test_feature_engineering_lookup_fallback(monkeypatch):
    import nfl_data_py as nfl
    from src.features import feature_engineering as fe_mod
    from src.features.feature_engineering import FeatureEngineer

    class _NoCache:
        def _get_connection(self):
            raise RuntimeError("no schedule cache in this test")

    import src.utils.database as db_mod
    monkeypatch.setattr(db_mod, "DatabaseManager", _NoCache)
    monkeypatch.setattr(nfl, "import_schedules", lambda seasons: SCHED.copy())
    out = FeatureEngineer()._create_vegas_game_script_features(TEAMS.copy())
    _check(out)


def test_calculate_implied_totals_matches_per_team_features():
    from src.data.external_data import VegasLinesLoader
    lines = SCHED.rename(columns={"total_line": "total"})
    out = VegasLinesLoader().calculate_implied_totals(lines)
    assert out.home_implied_total.iloc[0] == 24.0
    assert out.away_implied_total.iloc[0] == 21.0


def test_baselines_fallback_from_spread_matches_convention():
    import numpy as np
    from src.evaluation.baselines import vegas_implied_baseline
    df = pd.DataFrame({
        "player_id": ["h"] * 3 + ["a"] * 3, "position": "RB",
        "season": 2024, "week": [1, 2, 3] * 2,
        "team": ["AAA"] * 3 + ["BBB"] * 3,
        "spread": [-3.0] * 3 + [3.0] * 3, "game_total": 45.0,
        "fantasy_points": [10.0, 12.0, 14.0, 8.0, 9.0, 7.0],
    })
    explicit = df.assign(implied_team_total=[24.0] * 3 + [21.0] * 3)
    # Without implied_team_total the baseline recomputes it from spread;
    # that must reproduce the explicit-column result exactly.
    from_spread = vegas_implied_baseline(df, target_col="fantasy_points")
    from_explicit = vegas_implied_baseline(explicit, target_col="fantasy_points")
    assert np.allclose(from_spread.values, from_explicit.values, equal_nan=True)
    assert from_spread.notna().any()

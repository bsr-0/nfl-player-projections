from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "generate_dashboard_html",
    PROJECT_ROOT / "scripts" / "generate_dashboard_html.py",
)
dashboard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dashboard
spec.loader.exec_module(dashboard)


def test_build_board_data_uses_season_total_projection_for_ui(monkeypatch):
    monkeypatch.setattr(dashboard, "load_adp_board", lambda season: pd.DataFrame({"name": ["Player A"]}))
    monkeypatch.setattr(dashboard, "_latest_predictions_csv", lambda season: Path("fake.csv"))
    monkeypatch.setattr(
        dashboard,
        "load_headshot_lookup",
        lambda: {
            "by_player_id": {"p1": "https://example.com/player-a.png"},
            "by_name_team": {},
            "by_name_pos": {},
            "by_name": {},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame({
            "name": ["Player A"],
            "pred_total": [240.0],
        }),
    )
    monkeypatch.setattr(dashboard, "build_draft_board", lambda adp_df, projections: ["board"])
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([12.3]))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {"KC": {"tendency": "Pass-lean", "rush_pct": 40, "pass_pct": 60}})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))
    monkeypatch.setattr(
        dashboard,
        "compute_age_adjustments",
        lambda season: {"Player A": {"age": 31.0, "mult": 0.8}},
    )
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Player A",
                position="WR",
                team="KC",
                ecr=15.0,
                model_rank=8,
                rank_spread=7,
                model_projection=240.0,
                blended_projection=90.0,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )

    data = dashboard.build_board_data(2026)

    assert len(data["players"]) == 1
    player = data["players"][0]
    assert player["img"] == "https://example.com/player-a.png"
    assert player["rawProj"] == 240.0
    assert player["marketProj"] == 90.0
    assert player["calibratedProj"] == 142.5
    assert player["blendProj"] == 142.5
    assert player["proj"] == 102.0
    assert player["adjDelta"] == -40.5
    assert player["calibrationFlags"]["ageCapHit"] is True
    assert player["calibrationFlags"]["clampHit"] is True
    assert player["vorp"] == 12.3
    assert player["why"][0] == "Our rank #8 vs consensus #15"
    assert "Base projection 142.5 from 35% model 240.0 and 65% market 90.0" == player["why"][1]
    assert any("Age 31" in reason for reason in player["why"])


def test_build_board_data_dedupes_fallback_rows_when_ml_artifact_already_has_player(monkeypatch):
    captured = {}

    def _capture_board(adp_df, projections):
        captured["projections"] = projections.copy()
        return ["board"]

    monkeypatch.setattr(dashboard, "load_adp_board", lambda season: pd.DataFrame({"name": ["Trey McBride"]}))
    monkeypatch.setattr(dashboard, "_latest_predictions_csv", lambda season: None if season == 2026 else Path("fake-2025.csv"))
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame([
            {
                "name": "T.McBride",
                "position": "WR",
                "team": "ARI",
                "pred_total": 178.4,
                "actual_total": 0.0,
                "weeks": 17,
                "model_rank_value": 178.4,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "load_preseason_projections",
        lambda season, adp_df=None: pd.DataFrame([
            {
                "player_id": "rookie_Trey McBride_TE",
                "name": "Trey McBride",
                "position": "TE",
                "team": "ARI",
                "pred_total": 90.0,
                "actual_total": 0.0,
                "weeks": 0,
                "model_rank_value": 90.0,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "build_draft_board",
        _capture_board,
    )
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Trey McBride",
                position="TE",
                team="ARI",
                ecr=20.0,
                model_rank=4,
                rank_spread=8,
                model_projection=178.4,
                blended_projection=70.0,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([20.0]))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_age_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))

    dashboard.build_board_data(2026)

    projections = captured["projections"]
    assert len(projections) == 1
    assert projections.iloc[0]["name"] == "T.McBride"


def test_build_board_data_keeps_distinct_fallback_row_when_player_id_differs(monkeypatch):
    captured = {}

    def _capture_board(adp_df, projections):
        captured["projections"] = projections.copy()
        return ["board"]

    monkeypatch.setattr(
        dashboard,
        "load_adp_board",
        lambda season: pd.DataFrame({"name": ["Bijan Robinson", "Brian Robinson Jr."]}),
    )
    monkeypatch.setattr(
        dashboard,
        "_latest_predictions_csv",
        lambda season: None if season == 2026 else Path("fake-2025.csv"),
    )
    monkeypatch.setattr(
        dashboard,
        "load_model_projections",
        lambda csv, ranking, season: pd.DataFrame([
            {
                "player_id": "BIJAN_PID",
                "name": "B.Robinson",
                "position": "RB",
                "team": "ATL",
                "pred_total": 305.2,
                "actual_total": 0.0,
                "weeks": 17,
                "model_rank_value": 305.2,
            }
        ]),
    )
    monkeypatch.setattr(
        dashboard,
        "load_preseason_projections",
        lambda season, adp_df=None: pd.DataFrame([
            {
                "player_id": "BRIAN_PID",
                "name": "Brian Robinson Jr.",
                "position": "RB",
                "team": "WAS",
                "pred_total": 181.4,
                "actual_total": 0.0,
                "weeks": 0,
                "model_rank_value": 181.4,
            }
        ]),
    )
    monkeypatch.setattr(dashboard, "build_draft_board", _capture_board)
    monkeypatch.setattr(
        dashboard,
        "compute_spread",
        lambda board: [
            SimpleNamespace(
                player_id="p1",
                name="Bijan Robinson",
                position="RB",
                team="ATL",
                ecr=1.0,
                model_rank=1,
                rank_spread=0,
                model_projection=305.2,
                blended_projection=305.2,
                actual_total=0.0,
                model_wins=False,
            )
        ],
    )
    monkeypatch.setattr(dashboard, "validate_spread_direction", lambda results, min_spread=10: {"n": 0, "accuracy": 0.0})
    monkeypatch.setattr(dashboard, "_apply_vorp", lambda projections, basis_col="pred_total": pd.Series([20.0] * len(projections)))
    monkeypatch.setattr(dashboard, "build_usage_roles", lambda season: {})
    monkeypatch.setattr(dashboard, "build_team_tendencies", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_age_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_team_changes", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_usage_trends", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_regression_adjustments", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_injury_risk", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_breakout_candidates", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_defense_rankings", lambda season: {})
    monkeypatch.setattr(dashboard, "compute_sos_adjustment", lambda team, position, def_rankings: (1.0, ""))

    dashboard.build_board_data(2026)

    projections = captured["projections"]
    assert set(projections["player_id"]) == {"BIJAN_PID", "BRIAN_PID"}


def test_site_source_includes_position_scarcity_strip():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'id="scarcityStrip"' in template
    assert "function renderScarcityStrip()" in app_js
    assert 'document.getElementById("scarcityStrip").innerHTML = renderScarcityStrip()' in app_js
    assert "How much starter-level talent is still on the board" in app_js
    assert ".scarcity-strip" in style


def test_site_source_includes_hover_definitions_for_key_terms():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert "Projected full-season fantasy points in PPR scoring" in template
    assert "Projected full-season fantasy points in PPR scoring" in app_js
    assert "How many more points this player is worth than a typical backup-level option at the same position" in template
    assert "How many more points this player is worth than a typical backup-level option at the same position" in app_js
    assert "How many spots higher or lower our rankings are than the public draft market. Positive means better value" in template
    assert "How many spots higher or lower our rankings are than the public draft market. Positive means better value" in app_js
    assert ".def-term" in style


def test_site_source_includes_why_expander():
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'Array.isArray(p.why)' in app_js
    assert 'class="why-expander"' in app_js
    assert "Why we like or fade him" in app_js
    assert ".why-expander" in style
    assert ".why-list" in style


def test_site_source_includes_player_photos():
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert "function renderPlayerPhoto(p)" in app_js
    assert 'class="player-photo"' in app_js
    assert ".player-photo" in style
    assert ".player-summary" in style


def test_site_source_includes_draft_mode_watchlist():
    template = (PROJECT_ROOT / "_site" / "template.html").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "_site" / "style.css").read_text(encoding="utf-8")

    assert 'id="draftModeBtn"' in template
    assert 'id="draftPanel"' in template
    assert "function toggleDraftMode()" in app_js
    assert "function toggleWatch(playerId)" in app_js
    assert 'class="queue-btn' in app_js
    assert "renderDraftPanel()" in app_js
    assert ".draft-panel" in style
    assert ".watch-chip" in style
    assert "body.draft-mode .search-box" in style


def test_site_source_uses_plain_language_for_round_guide():
    app_js = (PROJECT_ROOT / "_site" / "app.js").read_text(encoding="utf-8")

    assert "Round-by-round targets" in app_js
    assert "Best fits for round" in app_js
    assert "Next round preview" in app_js


def _spread_stub(**overrides):
    base = {
        "player_id": "pid",
        "name": "Player A",
        "position": "WR",
        "team": "KC",
        "ecr": 50.0,
        "model_rank": 20,
        "rank_spread": 0,
        "model_projection": 200.0,
        "blended_projection": 140.0,
        "actual_total": 0.0,
        "model_wins": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_calibration_caps_age_only_adjustment_and_flags_cap_hit():
    result = dashboard._calibrate_projection(
        sr=_spread_stub(position="RB", ecr=42.0, model_projection=234.4),
        market_proj=219.7,
        market_source="exact_market",
        age_data={"age": 32.7, "mult": 0.72},
        team_change_data=None,
        trend_data=None,
        regression_data=None,
        injury_data=None,
        breakout_data=None,
        manual_data=None,
        sos_mult=1.0,
        sos_label="",
    )

    assert result["final_display_projection"] >= 215.0
    assert result["adjustment_delta"] >= -12.0
    assert result["calibration_flags"]["ageCapHit"] is True
    assert any(item["signal"] == "age" for item in result["adjustment_breakdown"])


def test_calibration_clamps_speculative_mid_tier_receiver_boosts():
    result = dashboard._calibrate_projection(
        sr=_spread_stub(
            name="Michael Pittman Jr.",
            position="WR",
            ecr=62.0,
            model_rank=52,
            model_projection=212.0,
            blended_projection=150.0,
        ),
        market_proj=148.0,
        market_source="exact_market",
        age_data=None,
        team_change_data=None,
        trend_data=None,
        regression_data=None,
        injury_data=None,
        breakout_data=None,
        manual_data={"player": "michael pittman jr.", "mult": 1.15, "note": "PIT: WR1 with Rodgers returning"},
        sos_mult=1.0,
        sos_label="",
    )

    assert result["final_display_projection"] <= 171.7
    assert result["calibration_flags"]["clampHit"] is True
    assert any(item["signal"] == "manual" for item in result["adjustment_breakdown"])


def test_calibration_preserves_credible_elite_wr_range():
    for name, raw_proj, market_proj in [
        ("Ja'Marr Chase", 266.2, 261.1),
        ("Justin Jefferson", 258.0, 252.0),
    ]:
        result = dashboard._calibrate_projection(
            sr=_spread_stub(
                name=name,
                position="WR",
                ecr=2.0,
                model_rank=10,
                model_projection=raw_proj,
                blended_projection=market_proj,
            ),
            market_proj=market_proj,
            market_source="exact_market",
            age_data=None,
            team_change_data=None,
            trend_data=None,
            regression_data=None,
            injury_data=None,
            breakout_data=None,
            manual_data=None,
            sos_mult=1.0,
            sos_label="",
        )

        assert result["final_display_projection"] >= market_proj * 0.95
        assert result["calibration_flags"]["eliteConsensusOverrideCheck"] is False
        assert result["calibration_flags"]["clampHit"] is False


def test_calibration_does_not_flag_tiny_market_anchor_on_relative_pct_alone():
    result = dashboard._calibrate_projection(
        sr=_spread_stub(
            name="Deep RB",
            position="RB",
            ecr=260.0,
            model_rank=300,
            model_projection=12.0,
            blended_projection=3.0,
        ),
        market_proj=3.0,
        market_source="exact_market",
        age_data=None,
        team_change_data=None,
        trend_data=None,
        regression_data=None,
        injury_data=None,
        breakout_data=None,
        manual_data=None,
        sos_mult=1.0,
        sos_label="",
    )

    assert result["calibration_flags"]["largeDivergence"] is False
    assert result["calibration_flags"]["unresolvedDisplayDivergence"] is False


def test_calibration_respects_qb_allowed_band_for_unresolved_flag():
    result = dashboard._calibrate_projection(
        sr=_spread_stub(
            name="QB Example",
            position="QB",
            ecr=80.0,
            model_rank=200,
            model_projection=58.2,
            blended_projection=155.8,
        ),
        market_proj=155.8,
        market_source="exact_market",
        age_data=None,
        team_change_data=None,
        trend_data=None,
        regression_data=None,
        injury_data=None,
        breakout_data=None,
        manual_data=None,
        sos_mult=1.0,
        sos_label="",
    )

    assert result["calibration_flags"]["largeDivergence"] is True
    assert result["calibration_flags"]["clampHit"] is True
    assert result["calibration_flags"]["unresolvedDisplayDivergence"] is False


def test_compute_age_adjustments_falls_back_to_local_db(monkeypatch):
    class _FakeNfl:
        @staticmethod
        def import_seasonal_rosters(_seasons):
            return pd.DataFrame()

    class _FakeConn:
        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "nfl_data_py", _FakeNfl)
    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: _FakeConn())
    monkeypatch.setattr(
        dashboard.pd,
        "read_sql_query",
        lambda query, conn, params=(): pd.DataFrame(
            [
                {"player_name": "Derrick Henry", "position": "RB", "birth_date": "1994-01-04"},
            ]
        ),
    )

    ages = dashboard.compute_age_adjustments(2026)

    assert "Derrick Henry" in ages
    assert ages["Derrick Henry"]["age"] > 32
    assert ages["Derrick Henry"]["mult"] < 1.0


def test_calibration_summary_uses_unresolved_display_divergence_rate():
    players = [
        {
            "n": "A",
            "p": "RB",
            "t": "X",
            "ecr": 10.0,
            "mr": 10,
            "proj": 200.0,
            "marketProj": 190.0,
            "rawProj": 260.0,
            "calibratedProj": 210.0,
            "adjDelta": -10.0,
            "calibrationFlags": {"largeDivergence": True, "unresolvedDisplayDivergence": False, "clampHit": True, "eliteConsensusOverrideCheck": False},
        },
        {
            "n": "B",
            "p": "WR",
            "t": "Y",
            "ecr": 20.0,
            "mr": 20,
            "proj": 120.0,
            "marketProj": 90.0,
            "rawProj": 150.0,
            "calibratedProj": 130.0,
            "adjDelta": -10.0,
            "calibrationFlags": {"largeDivergence": True, "unresolvedDisplayDivergence": True, "clampHit": False, "eliteConsensusOverrideCheck": False},
        },
        {
            "n": "C",
            "p": "RB",
            "t": "Z",
            "ecr": 250.0,
            "mr": 250,
            "proj": 30.0,
            "marketProj": 10.0,
            "rawProj": 80.0,
            "calibratedProj": 40.0,
            "adjDelta": -10.0,
            "calibrationFlags": {"largeDivergence": True, "unresolvedDisplayDivergence": True, "clampHit": True, "eliteConsensusOverrideCheck": False},
        },
    ]

    summary = dashboard._summarize_board_calibration(players)["summary"]

    assert summary["displayable_players"] == 2
    assert summary["unresolved_display_divergence_count"] == 1
    assert summary["large_divergence_share"] == 0.5
    assert summary["raw_large_divergence_share_all"] == 1.0

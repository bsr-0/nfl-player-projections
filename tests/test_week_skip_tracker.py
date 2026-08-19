"""Player-weeks dropped from a season projection must be recorded, not silent.

`compute_player_week_predictions` skips a week when the synthetic row can't
be built (no prior history) or when predict() raises. Both used to vanish --
the first logged nothing at all -- which let `weeks_synthetic == 0` quietly
mean "or we failed to project the weeks they missed". That corrupted the
reference bucket of the v33 availability experiment (see GAPS.md).
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.season_projection import (
    WeekSkipTracker, compute_player_week_predictions,
)

FEATS = ["f1", "f2"]


def _real_row(week, fp=12.0):
    return pd.DataFrame([{"f1": 1.0, "f2": 2.0, "fantasy_points": fp,
                          "week": week, "season": 2023}])


class _Model:
    def predict(self, X):
        return np.array([10.0] * len(X))


class _ExplodingModel:
    def predict(self, X):
        raise ValueError("boom")


class TestWeekSkipTrackerRecording:
    def test_no_prior_history_is_recorded(self, monkeypatch):
        """A week whose synthetic row cannot be built must leave a trace."""
        import src.models.single_week_ppr.season_projection as sp
        monkeypatch.setattr(sp, "build_synthetic_week_row",
                            lambda *a, **k: None)

        t = WeekSkipTracker("test")
        out = compute_player_week_predictions(
            "p1", {5: _real_row(5)}, {5}, {5: "KC", 6: "KC", 7: "KC"}, [5, 6, 7], _Model(),
            FEATS, pd.DataFrame(), MagicMock(), MagicMock(), 2023,
            skip_tracker=t)

        assert len(out) == 1, "only the real week should be predicted"
        assert len(t.skips) == 2, "weeks 6 and 7 must both be recorded"
        assert {s["week"] for s in t.skips} == {6, 7}
        assert all(s["reason"] == WeekSkipTracker.NO_PRIOR_HISTORY for s in t.skips)

    def test_predict_failure_is_recorded_with_the_error(self):
        t = WeekSkipTracker("test")
        out = compute_player_week_predictions(
            "p1", {5: _real_row(5)}, {5}, {5: "KC"}, [5], _ExplodingModel(),
            FEATS, pd.DataFrame(), MagicMock(), MagicMock(), 2023,
            skip_tracker=t)

        assert out == []
        assert len(t.skips) == 1
        assert t.skips[0]["reason"] == WeekSkipTracker.PREDICT_FAILED
        assert "ValueError" in t.skips[0]["error"]

    def test_tracker_is_optional(self, monkeypatch):
        """Existing callers that pass no tracker must keep working."""
        import src.models.single_week_ppr.season_projection as sp
        monkeypatch.setattr(sp, "build_synthetic_week_row", lambda *a, **k: None)
        out = compute_player_week_predictions(
            "p1", {5: _real_row(5)}, {5}, {5: "KC", 6: "KC"}, [5, 6], _Model(),
            FEATS, pd.DataFrame(), MagicMock(), MagicMock(), 2023)
        assert len(out) == 1

    def test_accounting_identity_is_detectable(self, monkeypatch):
        """The point of the tracker: real + synthetic + skipped == possible.
        Without it, a caller reading only the returned list would conclude
        this player had 1 possible week."""
        import src.models.single_week_ppr.season_projection as sp
        monkeypatch.setattr(sp, "build_synthetic_week_row", lambda *a, **k: None)

        possible = [5, 6, 7, 8]
        t = WeekSkipTracker("test")
        out = compute_player_week_predictions(
            "p1", {5: _real_row(5)}, {5}, {w: "KC" for w in possible}, possible, _Model(),
            FEATS, pd.DataFrame(), MagicMock(), MagicMock(), 2023,
            skip_tracker=t)

        n_real = sum(1 for r in out if r["is_real"])
        n_synth = sum(1 for r in out if not r["is_real"])
        assert n_real + n_synth + len(t.skips) == len(possible)


class TestWeekSkipTrackerReport:
    def test_clean_run_says_so(self, capsys):
        WeekSkipTracker("phase X").report()
        assert "no silently-dropped weeks" in capsys.readouterr().out

    def test_report_is_loud_and_counts_by_reason(self, capsys):
        t = WeekSkipTracker("phase X")
        t.record("p1", 2023, 6, WeekSkipTracker.NO_PRIOR_HISTORY)
        t.record("p1", 2023, 7, WeekSkipTracker.NO_PRIOR_HISTORY)
        t.record("p2", 2023, 3, WeekSkipTracker.PREDICT_FAILED, error=ValueError("x"))
        t.report()
        out = capsys.readouterr().out
        assert "3 PLAYER-WEEK(S) SKIPPED" in out
        assert "across 2 player(s)" in out
        assert f"{WeekSkipTracker.NO_PRIOR_HISTORY}: 2" in out
        assert f"{WeekSkipTracker.PREDICT_FAILED}: 1" in out
        assert "!!" in out

    def test_sidecar_json_is_written(self, tmp_path):
        import json
        t = WeekSkipTracker("phase X")
        t.record("p1", 2023, 6, WeekSkipTracker.NO_PRIOR_HISTORY)
        out_csv = tmp_path / "results.csv"
        with redirect_stdout(io.StringIO()):
            t.report(out_csv)
        sidecar = Path(str(out_csv) + ".weekskips.json")
        assert sidecar.exists()
        rec = json.loads(sidecar.read_text())
        assert rec[0]["week"] == 6 and rec[0]["player_id"] == "p1"

    def test_reporting_never_breaks_a_completed_run(self):
        """A bad output path must not raise after a run has finished."""
        t = WeekSkipTracker("phase X")
        t.record("p1", 2023, 6, WeekSkipTracker.NO_PRIOR_HISTORY)
        with redirect_stdout(io.StringIO()):
            t.report(Path("/nonexistent-root-dir/x/y/results.csv"))


class TestDefenseRankingGuard:
    def test_single_team_frame_skips_ranking(self, capsys):
        from src.data.external_data import DefenseRankingsLoader
        df = pd.DataFrame(dict(player_id=['a'], position=['QB'], team=['KC'],
                               opponent=['DEN'], season=[2023], week=[5],
                               fantasy_points=[20.0]))
        out = DefenseRankingsLoader().calculate_defense_rankings(df)
        assert out.empty
        assert "Skipping defense rankings" in capsys.readouterr().out

    def test_full_league_frame_still_ranks(self):
        from src.data.external_data import DefenseRankingsLoader
        teams = [f"T{i:02d}" for i in range(32)]
        rows = []
        for wk in (1, 2):
            for i in range(0, 32, 2):
                rows.append(dict(player_id=f"{teams[i]}QB", position='QB',
                                 team=teams[i], opponent=teams[i+1],
                                 season=2023, week=wk, fantasy_points=15.0))
        with redirect_stdout(io.StringIO()):
            out = DefenseRankingsLoader().calculate_defense_rankings(pd.DataFrame(rows))
        assert not out.empty
        assert out['team'].nunique() == 16

"""Tests for FoldFailureTracker -- makes silently-dropped folds detectable.

Every phase wraps per-fold loading in a broad `except: continue`. That's
deliberate (one bad fold shouldn't kill a multi-hour grid) but the original
`logger.warning`-only trace vanished whenever output was piped through
`tail`. A real train/test column-mismatch bug silently deleted QB/2025/'all'
from a Phase 3 grid that way, and the resulting 2-of-3-season average was
nearly promoted into FINAL_CONFIG.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.single_week_ppr.evaluate import FoldFailureTracker


class TestFoldFailureTracker:
    def test_clean_run_states_no_folds_dropped(self, capsys):
        t = FoldFailureTracker("Phase X")
        t.report()
        assert "no silently-dropped folds" in capsys.readouterr().out

    def test_failure_summary_is_loud_and_names_the_fold(self, capsys):
        t = FoldFailureTracker("Phase 3 (window/weighting grid)")
        t.record("QB", 2025, KeyError("air_yards_share_pct_roll3_mean_missing"), window="all")
        t.report()
        out = capsys.readouterr().out
        assert "1 FOLD(S) FAILED" in out
        assert "QB/2025" in out
        assert "INCOMPLETE" in out          # warns against comparing the aggregate
        assert "air_yards_share" in out      # surfaces the real cause
        assert "!!!!" in out                  # visually unmissable even when tailed

    def test_writes_sidecar_json_next_to_output(self, tmp_path):
        out_csv = tmp_path / "phase3.csv"
        t = FoldFailureTracker("Phase 3")
        t.record("QB", 2025, ValueError("boom"), window="all")
        t.report(out_csv)

        sidecar = Path(str(out_csv) + ".failures.json")
        assert sidecar.exists()          # survives terminal scrollback
        rec = json.loads(sidecar.read_text())
        assert rec[0]["position"] == "QB"
        assert rec[0]["season"] == 2025
        assert rec[0]["window"] == "all"
        assert rec[0]["error_type"] == "ValueError"

    def test_no_sidecar_written_when_nothing_failed(self, tmp_path):
        out_csv = tmp_path / "phase3.csv"
        FoldFailureTracker("Phase 3").report(out_csv)
        assert not Path(str(out_csv) + ".failures.json").exists()

    def test_reporting_never_raises_on_unwritable_path(self, capsys):
        """A completed multi-hour run must not die in its reporting step."""
        t = FoldFailureTracker("Phase 3")
        t.record("QB", 2025, ValueError("boom"))
        t.report(Path("/nonexistent-root-dir-xyz/out.csv"))  # must not raise
        assert "1 FOLD(S) FAILED" in capsys.readouterr().out

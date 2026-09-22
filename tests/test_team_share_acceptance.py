"""scripts/check_team_share_acceptance.py -- acceptance-gate correctness."""
import json

import pytest

import scripts.check_team_share_acceptance as gate


def _write_metadata(models_dir, target, pooled):
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"team_share_{target}_model_metadata.json"
    path.write_text(json.dumps({"backtest": {"pooled": pooled}}))
    return path


def _arm(mae, ci_low, ci_high, significant):
    return {
        "mae": mae,
        "vs_rolling3_bootstrap": {
            "point_estimate": mae, "ci_low": ci_low, "ci_high": ci_high,
            "n_bootstrap": 500, "significant_improvement": significant,
        },
    }


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setattr(gate, "MODELS_DIR", d)
    return d


def test_passes_when_an_arm_is_significantly_better(models_dir):
    pooled = {
        "rolling3": {"mae": 0.10},
        "ridge": _arm(0.08, -0.03, -0.01, significant=True),
        "xgboost": _arm(0.11, -0.01, 0.02, significant=False),
    }
    _write_metadata(models_dir, "targets", pooled)
    assert gate.check_target("targets") is True


def test_fails_when_no_arm_is_significantly_better(models_dir):
    pooled = {
        "rolling3": {"mae": 0.10},
        "ridge": _arm(0.095, -0.01, 0.005, significant=False),
        "xgboost": _arm(0.11, -0.005, 0.02, significant=False),
    }
    _write_metadata(models_dir, "targets", pooled)
    assert gate.check_target("targets") is False


def test_fails_on_missing_metadata_file(models_dir):
    assert gate.check_target("targets") is False


def test_fails_on_missing_rolling3_baseline(models_dir):
    _write_metadata(models_dir, "targets", {"ridge": _arm(0.08, -0.03, -0.01, significant=True)})
    assert gate.check_target("targets") is False


def test_fails_gracefully_on_metadata_predating_bootstrap_field(models_dir):
    """A metadata file written before the bootstrap-CI change has no
    vs_rolling3_bootstrap key -- must fail closed, not crash or silently pass."""
    pooled = {"rolling3": {"mae": 0.10}, "ridge": {"mae": 0.08}}
    _write_metadata(models_dir, "targets", pooled)
    assert gate.check_target("targets") is False


def test_main_exit_code_reflects_pass_fail(models_dir, monkeypatch):
    pooled_pass = {
        "rolling3": {"mae": 0.10},
        "ridge": _arm(0.08, -0.03, -0.01, significant=True),
    }
    pooled_fail = {
        "rolling3": {"mae": 0.10},
        "ridge": _arm(0.099, -0.005, 0.003, significant=False),
    }
    _write_metadata(models_dir, "targets", pooled_pass)
    _write_metadata(models_dir, "rushing_attempts", pooled_fail)

    monkeypatch.setattr("sys.argv", ["check_team_share_acceptance.py", "--target", "targets"])
    assert gate.main() == 0

    monkeypatch.setattr("sys.argv", ["check_team_share_acceptance.py", "--target", "rushing_attempts"])
    assert gate.main() == 1

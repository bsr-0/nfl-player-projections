from argparse import Namespace
from pathlib import Path

from scripts.run_participation_pipeline import stage_commands


def _args(tmp_path, stages):
    return Namespace(
        output_root=tmp_path / "participation", db=tmp_path / "nfl.db", stages=stages,
        seasons=[2013, 2026], min_train_seasons=3, include_pregame_injury=True,
        phase3_seasons=[2023, 2024, 2025], n_bootstrap=2000, phase2_model="logistic",
    )


def test_main_pipeline_orders_all_stages_and_handoffs(tmp_path):
    commands = stage_commands(_args(tmp_path, ["phase1", "phase2", "phase3"]))
    assert [stage for stage, _ in commands] == ["phase1", "phase2", "phase3"]
    phase2 = commands[1][1]
    phase3 = commands[2][1]
    phase1 = commands[0][1]
    assert phase1[phase1.index("--db") + 1] == str(tmp_path / "nfl.db")
    assert "--include-pregame-injury" in phase2
    assert phase3[phase3.index("--phase2-model") + 1] == "logistic"
    assert str(tmp_path / "participation" / "phase2" / "oof_predictions.csv") in phase3


def test_partial_phase3_uses_same_durable_phase2_location(tmp_path):
    command = stage_commands(_args(tmp_path, ["phase3"]))[0][1]
    assert command[command.index("--phase2-oof") + 1] == str(
        tmp_path / "participation" / "phase2" / "oof_predictions.csv"
    )

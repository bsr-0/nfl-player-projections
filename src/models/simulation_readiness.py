"""Explicit production-readiness gate for game/player simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class SimulationReadiness:
    ready: bool
    missing: tuple[str, ...]

def production_readiness(player_predictions: pd.DataFrame,
                         *, role_correlation_artifact: str | Path | None,
                         calibration_artifact: str | Path | None) -> SimulationReadiness:
    """Require data artifacts that turn an exploratory simulator into a model."""
    missing = []
    # Must match simulation_adapter.player_inputs_from_predictions's required
    # columns ("opponent", "season", "week" are needed there to match a player
    # row to its scheduled game) plus "participation_prob", which that adapter
    # treats as optional with a fallback but production readiness should not.
    required = {"player_id", "team", "opponent", "position", "predicted_points",
                "participation_prob", "season", "week"}
    missing_columns = required - set(player_predictions.columns)
    if missing_columns:
        missing.append(f"player serving data lacks {sorted(missing_columns)}")
    share_columns = {"target_share", "rush_share", "pass_share"}
    if not (share_columns & set(player_predictions.columns)):
        missing.append("no causal target/carry/pass share columns")
    elif player_predictions[list(share_columns & set(player_predictions.columns))].isna().all().all():
        missing.append("causal usage share columns contain no values")
    for label, artifact in (
        ("role-correlation artifact", role_correlation_artifact),
        ("calibration artifact", calibration_artifact),
    ):
        if artifact is None or not Path(artifact).exists():
            missing.append(f"missing {label}")
    return SimulationReadiness(not missing, tuple(missing))

def require_production_ready(player_predictions: pd.DataFrame, **kwargs) -> None:
    report = production_readiness(player_predictions, **kwargs)
    if not report.ready:
        raise RuntimeError("simulation is not production-ready: " + "; ".join(report.missing))

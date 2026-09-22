import pandas as pd
import pytest

from src.models.simulation_readiness import production_readiness, require_production_ready

def test_readiness_rejects_exploratory_serving_rows():
    rows = pd.DataFrame([{"player_id": "p", "team": "H", "position": "RB", "predicted_points": 10.}])
    report = production_readiness(rows, role_correlation_artifact=None, calibration_artifact=None)
    assert not report.ready and len(report.missing) >= 3
    with pytest.raises(RuntimeError, match="not production-ready"):
        require_production_ready(rows, role_correlation_artifact=None, calibration_artifact=None)

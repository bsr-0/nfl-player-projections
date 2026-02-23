"""Rubric compliance checker for fantasy football prediction requirements.

This script validates that the codebase still satisfies key architectural and
evaluation requirements. It is intended for CI/pre-release checks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CheckResult:
    """One rubric check result."""

    name: str
    passed: bool
    details: str
    severity: str = "error"  # error | warning


def _import(path: str) -> Any:
    """Import module by dotted path."""
    return importlib.import_module(path)


def _has_all_keys(mapping: Dict[str, Any], keys: List[str]) -> Tuple[bool, List[str]]:
    missing = [k for k in keys if k not in mapping]
    return len(missing) == 0, missing


def check_config_contract() -> List[CheckResult]:
    """Validate core config required by rubric."""
    out: List[CheckResult] = []
    settings = _import("config.settings")

    # Position-specific modeling contract
    positions = list(getattr(settings, "POSITIONS", []))
    expected_positions = ["QB", "RB", "WR", "TE"]
    out.append(
        CheckResult(
            name="positions_all",
            passed=set(expected_positions).issubset(set(positions)),
            details=f"POSITIONS={positions}",
        )
    )

    # Temporal windows contract
    rolling_windows = list(getattr(settings, "ROLLING_WINDOWS", []))
    required_windows = {3, 4, 5, 8}
    out.append(
        CheckResult(
            name="rolling_windows_3_4_5_8",
            passed=required_windows.issubset(set(rolling_windows)),
            details=f"ROLLING_WINDOWS={rolling_windows}",
        )
    )

    # Multi-horizon model config contract
    model_cfg = getattr(settings, "MODEL_CONFIG", {})
    has_cfg, missing_cfg = _has_all_keys(
        model_cfg,
        [
            "use_4w_hybrid",
            "use_18w_deep",
            "lstm_weight",
            "arima_weight",
            "deep_blend_traditional",
        ],
    )
    out.append(
        CheckResult(
            name="model_config_multi_horizon_keys",
            passed=has_cfg,
            details=f"missing={missing_cfg}" if not has_cfg else "all keys present",
        )
    )

    # Production retraining config contract
    retraining_cfg = getattr(settings, "RETRAINING_CONFIG", {})
    has_ret, missing_ret = _has_all_keys(
        retraining_cfg,
        ["auto_retrain", "retrain_day", "degradation_threshold_pct"],
    )
    out.append(
        CheckResult(
            name="retraining_config_keys",
            passed=has_ret,
            details=f"missing={missing_ret}" if not has_ret else "all keys present",
        )
    )

    return out


def check_model_architecture_contract() -> List[CheckResult]:
    """Validate core model classes and conversion layers exist."""
    out: List[CheckResult] = []

    position_models = _import("src.models.position_models")
    horizon_models = _import("src.models.horizon_models")
    util_converter = _import("src.models.utilization_to_fp")

    out.append(
        CheckResult(
            name="position_specific_model_classes",
            passed=all(
                hasattr(position_models, cls_name)
                for cls_name in ["PositionModel", "MultiWeekModel"]
            ),
            details="PositionModel + MultiWeekModel must exist",
        )
    )
    out.append(
        CheckResult(
            name="horizon_model_classes",
            passed=all(
                hasattr(horizon_models, cls_name)
                for cls_name in ["Hybrid4WeekModel", "DeepSeasonLongModel"]
            ),
            details="Hybrid4WeekModel + DeepSeasonLongModel must exist",
        )
    )
    out.append(
        CheckResult(
            name="utilization_to_fp_converter",
            passed=hasattr(util_converter, "UtilizationToFPConverter"),
            details="UtilizationToFPConverter must exist",
        )
    )

    return out


def check_feature_and_eval_contract() -> List[CheckResult]:
    """Validate feature engineering and evaluation surface contracts."""
    out: List[CheckResult] = []
    features = _import("src.features.feature_engineering")
    metrics = _import("src.evaluation.metrics")

    feature_engineer = getattr(features, "FeatureEngineer", None)
    required_feature_methods = [
        "_create_rolling_features",
        "_create_trend_features",
        "_create_vegas_game_script_features",
        "_create_advanced_requirement_features",
    ]
    feature_ok = feature_engineer is not None and all(
        hasattr(feature_engineer, m) for m in required_feature_methods
    )
    out.append(
        CheckResult(
            name="feature_engineering_required_methods",
            passed=feature_ok,
            details=f"missing={[m for m in required_feature_methods if feature_engineer is None or not hasattr(feature_engineer, m)]}",
        )
    )

    evaluator = getattr(metrics, "ModelEvaluator", None)
    eval_ok = evaluator is not None and hasattr(evaluator, "evaluate_model")
    out.append(
        CheckResult(
            name="model_evaluator_exists",
            passed=eval_ok,
            details="ModelEvaluator.evaluate_model required",
        )
    )

    for fn_name in [
        "spearman_rank_correlation",
        "tier_classification_accuracy",
        "boom_bust_metrics",
        "vor_accuracy",
        "compare_to_expert_consensus",
    ]:
        out.append(
            CheckResult(
                name=f"metric_function_{fn_name}",
                passed=hasattr(metrics, fn_name),
                details=f"{fn_name} must exist",
            )
        )

    return out


def check_monitoring_artifacts(require_artifacts: bool) -> List[CheckResult]:
    """Validate model monitoring artifacts (optional strict mode)."""
    out: List[CheckResult] = []
    settings = _import("config.settings")
    models_dir = Path(getattr(settings, "MODELS_DIR"))

    required_files = [
        "model_metadata.json",
        "model_monitoring_report.json",
        "top10_features_per_position.json",
    ]
    for fname in required_files:
        exists = (models_dir / fname).exists()
        out.append(
            CheckResult(
                name=f"artifact_{fname}",
                passed=exists or not require_artifacts,
                details=f"{models_dir / fname}",
                severity="error" if require_artifacts else "warning",
            )
        )
    return out


def run_checks(require_artifacts: bool = False) -> List[CheckResult]:
    """Run all rubric checks."""
    checks: List[CheckResult] = []
    checks.extend(check_config_contract())
    checks.extend(check_model_architecture_contract())
    checks.extend(check_feature_and_eval_contract())
    checks.extend(check_monitoring_artifacts(require_artifacts=require_artifacts))
    return checks


def summarize(results: List[CheckResult]) -> Dict[str, Any]:
    """Build summary payload with pass/fail stats."""
    errors = [r for r in results if r.severity == "error"]
    warnings = [r for r in results if r.severity == "warning"]
    failed_errors = [r for r in errors if not r.passed]
    failed_warnings = [r for r in warnings if not r.passed]
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "failed_errors": len(failed_errors),
        "failed_warnings": len(failed_warnings),
        "checks": [asdict(r) for r in results],
    }


def _print_human(summary: Dict[str, Any]) -> None:
    """Print human-readable report."""
    print("\nRubric Compliance Report")
    print("=" * 30)
    for c in summary["checks"]:
        icon = "PASS" if c["passed"] else ("WARN" if c["severity"] == "warning" else "FAIL")
        print(f"[{icon}] {c['name']}: {c['details']}")
    print("-" * 30)
    print(
        f"Total={summary['total']}  Passed={summary['passed']}  "
        f"Failed={summary['failed']}  Error-failures={summary['failed_errors']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check rubric compliance of codebase.")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail when monitoring artifacts are missing in MODELS_DIR.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    args = parser.parse_args()

    results = run_checks(require_artifacts=args.require_artifacts)
    summary = summarize(results)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_human(summary)

    return 1 if summary["failed_errors"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

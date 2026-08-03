#!/usr/bin/env python3
"""
Directive V7 Deliverables Checker.

Per Sections 16 and 25: programmatically verify that all required
artifacts exist in the repository.

Usage:
    python scripts/check_deliverables.py [--json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def check_deliverables() -> dict:
    """Check all required deliverables and return results."""
    checks = {}

    # Part I deliverables (Sections 1-17)
    checks["experiment_ledger"] = {
        "description": "Experiment tracking ledger (§3)",
        "path": "src/evaluation/experiment_tracker.py",
        "exists": (ROOT / "src/evaluation/experiment_tracker.py").exists(),
    }
    checks["problem_definition"] = {
        "description": "Problem definition document (§4)",
        "path": "docs/PROBLEM_DEFINITION.md",
        "exists": (ROOT / "docs/PROBLEM_DEFINITION.md").exists(),
    }
    checks["decision_objectives"] = {
        "description": "Decision objectives document (§1, §9)",
        "path": "docs/DECISION_OBJECTIVES.md",
        "exists": (ROOT / "docs/DECISION_OBJECTIVES.md").exists(),
    }
    checks["data_lineage"] = {
        "description": "Data lineage documentation (§5)",
        "path": "docs/DATA_LINEAGE.md",
        "exists": (ROOT / "docs/DATA_LINEAGE.md").exists(),
    }
    checks["feature_pipeline"] = {
        "description": "Feature engineering pipeline (§6)",
        "path": "src/features/feature_engineering.py",
        "exists": (ROOT / "src/features/feature_engineering.py").exists(),
    }
    checks["calibration_metrics"] = {
        "description": "Calibration metrics (ECE, reliability curves) (§8)",
        "path": "src/evaluation/metrics.py",
        "exists": (ROOT / "src/evaluation/metrics.py").exists(),
    }
    checks["decision_optimizer"] = {
        "description": "Decision optimization layer (§9)",
        "path": "src/evaluation/decision_optimizer.py",
        "exists": (ROOT / "src/evaluation/decision_optimizer.py").exists(),
    }
    checks["lineup_optimizer"] = {
        "description": "DFS lineup optimizer (§9, §24)",
        "path": "src/optimization/lineup_optimizer.py",
        "exists": (ROOT / "src/optimization/lineup_optimizer.py").exists(),
    }
    checks["backtester"] = {
        "description": "Time-series backtester with drawdown + scenarios (§10)",
        "path": "src/evaluation/ts_backtester.py",
        "exists": (ROOT / "src/evaluation/ts_backtester.py").exists(),
    }
    checks["leakage_guards"] = {
        "description": "Leakage detection guards (§11, §15)",
        "path": "src/utils/leakage.py",
        "exists": (ROOT / "src/utils/leakage.py").exists(),
    }
    checks["evaluation_matrix"] = {
        "description": "Evaluation matrix generator (§13)",
        "path": "src/evaluation/metrics.py",
        "exists": (ROOT / "src/evaluation/metrics.py").exists(),
    }
    checks["model_artifacts"] = {
        "description": "Trained model artifacts directory",
        "path": "data/models/",
        "exists": (ROOT / "data/models").is_dir(),
    }

    # Part II deliverables (Sections 18-25)
    checks["monitoring"] = {
        "description": "Production monitoring with circuit breaker (§18)",
        "path": "src/evaluation/monitoring.py",
        "exists": (ROOT / "src/evaluation/monitoring.py").exists(),
    }
    checks["ab_testing"] = {
        "description": "A/B testing with pre-promotion checks (§18)",
        "path": "src/evaluation/ab_testing.py",
        "exists": (ROOT / "src/evaluation/ab_testing.py").exists(),
    }
    checks["schema_validator"] = {
        "description": "Schema validation + freshness SLA (§19)",
        "path": "src/data/schema_validator.py",
        "exists": (ROOT / "src/data/schema_validator.py").exists(),
    }
    checks["ci_pipeline"] = {
        "description": "CI/CD pipeline (§23)",
        "path": ".github/workflows/rubric-compliance.yml",
        "exists": (ROOT / ".github/workflows/rubric-compliance.yml").exists(),
    }
    checks["calibration_tests"] = {
        "description": "Calibration quality tests (§23)",
        "path": "tests/test_calibration_quality.py",
        "exists": (ROOT / "tests/test_calibration_quality.py").exists(),
    }
    checks["determinism_tests"] = {
        "description": "Pipeline determinism tests (§23)",
        "path": "tests/test_pipeline_determinism.py",
        "exists": (ROOT / "tests/test_pipeline_determinism.py").exists(),
    }
    checks["yahoo_integration"] = {
        "description": "Yahoo Fantasy integration (§24)",
        "path": "src/integrations/yahoo_fantasy.py",
        "exists": (ROOT / "src/integrations/yahoo_fantasy.py").exists(),
    }
    checks["sleeper_integration"] = {
        "description": "Sleeper Fantasy integration (§24)",
        "path": "src/integrations/sleeper_fantasy.py",
        "exists": (ROOT / "src/integrations/sleeper_fantasy.py").exists(),
    }

    # Documentation deliverables
    checks["architecture_doc"] = {
        "description": "Architecture documentation (§2, §12)",
        "path": "docs/ARCHITECTURE.md",
        "exists": (ROOT / "docs/ARCHITECTURE.md").exists(),
    }
    checks["operating_summary"] = {
        "description": "Operating summary (§17)",
        "path": "docs/OPERATING_SUMMARY.md",
        "exists": (ROOT / "docs/OPERATING_SUMMARY.md").exists(),
    }
    checks["runbook"] = {
        "description": "Operational runbook (§16)",
        "path": "docs/RUNBOOK.md",
        "exists": (ROOT / "docs/RUNBOOK.md").exists(),
    }
    checks["deployment_config"] = {
        "description": "Deployment configuration",
        "path": "Dockerfile",
        "exists": (ROOT / "Dockerfile").exists(),
    }

    # Summary
    total = len(checks)
    passed = sum(1 for c in checks.values() if c["exists"])
    failed = total - passed

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Check Directive V7 deliverables")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = check_deliverables()

    if args.json:
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["failed"] == 0 else 1)

    print("=" * 60)
    print("DIRECTIVE V7 DELIVERABLES CHECK")
    print("=" * 60)
    print()

    for name, check in results["checks"].items():
        status = "PASS" if check["exists"] else "FAIL"
        icon = "+" if check["exists"] else "-"
        print(f"  [{icon}] {check['description']}")
        if not check["exists"]:
            print(f"      Missing: {check['path']}")

    print()
    print(f"  Total: {results['total']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Pass rate: {results['pass_rate']}%")
    print()

    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

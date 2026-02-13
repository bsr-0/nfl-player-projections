#!/usr/bin/env python3
"""
Environment Check Script for NFL Predictor

This script validates that all required dependencies are installed
and the environment is properly configured before running the application.

Usage:
    python scripts/check_environment.py
    
    # Verbose mode with version details
    python scripts/check_environment.py --verbose
    
    # Auto-fix missing dependencies
    python scripts/check_environment.py --fix
"""

import sys
import subprocess
import importlib
from pathlib import Path


# Minimum Python version required
MIN_PYTHON_VERSION = (3, 9)

# Required packages with minimum versions (package_name, import_name, min_version)
REQUIRED_PACKAGES = [
    # Core data processing
    ("numpy", "numpy", "1.20.0"),
    ("pandas", "pandas", "2.0.0"),
    
    # Machine Learning
    ("scikit-learn", "sklearn", "1.0.0"),
    ("xgboost", "xgboost", "1.5.0"),
    ("lightgbm", "lightgbm", "3.0.0"),
    ("optuna", "optuna", "3.0.0"),
    ("joblib", "joblib", "1.0.0"),
    
    # Data collection
    ("requests", "requests", "2.25.0"),
    ("beautifulsoup4", "bs4", "4.9.0"),
    ("lxml", "lxml", "4.6.0"),
    
    # Database
    ("sqlalchemy", "sqlalchemy", "2.0.0"),
    
    # Web application
    ("streamlit", "streamlit", "1.20.0"),
    ("plotly", "plotly", "5.0.0"),
    
    # NFL data
    ("nfl-data-py", "nfl_data_py", "0.3.0"),
    
    # Testing
    ("pytest", "pytest", "7.0.0"),
    
    # Utilities
    ("tqdm", "tqdm", "4.60.0"),
    ("python-dateutil", "dateutil", "2.8.0"),
]

# Optional packages (nice to have but not required)
OPTIONAL_PACKAGES = [
    ("fastapi", "fastapi", "0.100.0"),
    ("uvicorn", "uvicorn", "0.20.0"),
    ("matplotlib", "matplotlib", "3.5.0"),
    ("seaborn", "seaborn", "0.12.0"),
    ("pytest-cov", "pytest_cov", "4.0.0"),
]


def parse_version(version_str: str) -> tuple:
    """Parse version string into tuple for comparison."""
    try:
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])
    except (ValueError, AttributeError):
        return (0, 0, 0)


def check_python_version() -> tuple:
    """Check Python version meets minimum requirements."""
    current = sys.version_info[:2]
    meets_requirement = current >= MIN_PYTHON_VERSION
    return meets_requirement, current, MIN_PYTHON_VERSION


def check_package(package_name: str, import_name: str, min_version: str) -> dict:
    """Check if a package is installed and meets version requirements."""
    result = {
        "package": package_name,
        "import_name": import_name,
        "required_version": min_version,
        "installed": False,
        "version": None,
        "meets_version": False,
        "error": None,
    }
    
    try:
        module = importlib.import_module(import_name)
        result["installed"] = True
        
        # Try to get version
        version = getattr(module, "__version__", None)
        if version is None:
            # Try alternative version attributes
            version = getattr(module, "version", None)
            if version is None and hasattr(module, "VERSION"):
                version = str(module.VERSION)
        
        result["version"] = version
        
        if version:
            result["meets_version"] = parse_version(version) >= parse_version(min_version)
        else:
            # If we can't determine version, assume it's okay
            result["meets_version"] = True
            
    except ImportError as e:
        result["error"] = str(e)
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
    
    return result


def check_data_files() -> dict:
    """Check if required data files exist."""
    project_root = Path(__file__).parent.parent
    
    checks = {
        "database": project_root / "data" / "nfl_data.db",
        "models_dir": project_root / "data" / "models",
        "config": project_root / "config" / "settings.py",
    }
    
    results = {}
    for name, path in checks.items():
        results[name] = {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else None,
            "is_dir": path.is_dir() if path.exists() else None,
        }
    
    # Check for trained models
    models_dir = project_root / "data" / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.joblib"))
        results["trained_models"] = {
            "path": str(models_dir),
            "exists": True,
            "count": len(model_files),
            "files": [f.name for f in model_files],
        }
    else:
        results["trained_models"] = {"exists": False, "count": 0, "files": []}
    
    return results


def install_missing_packages(packages: list) -> bool:
    """Attempt to install missing packages using pip."""
    if not packages:
        return True
    
    print(f"\nAttempting to install {len(packages)} missing packages...")
    
    for package in packages:
        print(f"  Installing {package}...", end=" ")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("OK")
        except subprocess.CalledProcessError:
            print("FAILED")
            return False
    
    return True


def main():
    """Main environment check function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check NFL Predictor environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix missing dependencies")
    args = parser.parse_args()
    
    print("=" * 60)
    print("NFL Predictor Environment Check")
    print("=" * 60)
    
    all_ok = True
    missing_packages = []
    
    # Check Python version
    print("\n1. Python Version")
    print("-" * 40)
    meets_req, current, required = check_python_version()
    status = "OK" if meets_req else "FAIL"
    print(f"   Python {current[0]}.{current[1]} (required: {required[0]}.{required[1]}+) [{status}]")
    
    if not meets_req:
        all_ok = False
        print(f"   ERROR: Python {required[0]}.{required[1]}+ is required")
    
    # Check required packages
    print("\n2. Required Packages")
    print("-" * 40)
    
    for package_name, import_name, min_version in REQUIRED_PACKAGES:
        result = check_package(package_name, import_name, min_version)
        
        if result["installed"] and result["meets_version"]:
            status = "OK"
            version_str = result["version"] or "unknown"
        elif result["installed"]:
            status = "OUTDATED"
            version_str = f"{result['version']} (need {min_version}+)"
            all_ok = False
        else:
            status = "MISSING"
            version_str = "not installed"
            all_ok = False
            missing_packages.append(package_name)
        
        if args.verbose or status != "OK":
            print(f"   {package_name:20} {version_str:20} [{status}]")
    
    if not args.verbose and all_ok:
        print(f"   All {len(REQUIRED_PACKAGES)} required packages installed")
    
    # Check optional packages
    if args.verbose:
        print("\n3. Optional Packages")
        print("-" * 40)
        
        for package_name, import_name, min_version in OPTIONAL_PACKAGES:
            result = check_package(package_name, import_name, min_version)
            
            if result["installed"]:
                status = "OK"
                version_str = result["version"] or "unknown"
            else:
                status = "MISSING"
                version_str = "not installed"
            
            print(f"   {package_name:20} {version_str:20} [{status}]")
    
    # Check data files
    print("\n4. Data Files")
    print("-" * 40)
    
    data_results = check_data_files()
    
    for name, info in data_results.items():
        if name == "trained_models":
            count = info.get("count", 0)
            status = "OK" if count > 0 else "EMPTY"
            print(f"   {name:20} {count} model files [{status}]")
            if args.verbose and info.get("files"):
                for f in info["files"][:5]:
                    print(f"      - {f}")
                if count > 5:
                    print(f"      ... and {count - 5} more")
        else:
            exists = info.get("exists", False)
            status = "OK" if exists else "MISSING"
            print(f"   {name:20} {'found' if exists else 'not found':20} [{status}]")
    
    # Attempt to fix if requested
    if args.fix and missing_packages:
        success = install_missing_packages(missing_packages)
        if success:
            print("\nPackages installed successfully. Please re-run the check.")
        else:
            print("\nSome packages failed to install. Please install manually.")
    
    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("Environment check PASSED - ready to run!")
        print("=" * 60)
        return 0
    else:
        print("Environment check FAILED - please fix the issues above")
        if missing_packages and not args.fix:
            print(f"\nTo install missing packages, run:")
            print(f"  pip install {' '.join(missing_packages)}")
            print(f"\nOr run: python scripts/check_environment.py --fix")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Pytest configuration and shared fixtures."""
import warnings

# Apply filters early so they're in effect when tests import (e.g. scipy at import time)
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*", category=RuntimeWarning)
# SciPy/NumPy version mismatch (environment); can fire when a plugin imports scipy before conftest
warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy.*")


def pytest_configure(config):
    """Pytest hook; additional config if needed."""
    pass

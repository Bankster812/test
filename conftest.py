"""Root conftest — ensures the repo root is importable so tests can
`import wholesale` / `import neuromorphic` regardless of pytest's rootdir
insertion behaviour."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    """Marker registrieren — sonst warnt pytest, und `-m "not slow"` geht nicht."""
    config.addinivalue_line(
        "markers",
        "slow: baut das CAD-Modell und misst es nach (rund eine Minute)")

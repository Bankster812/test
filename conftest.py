"""Pytest bootstrap.

Ensures the repository root is importable so ``import neuromorphic`` works when
the suite is run as a bare ``pytest`` (which, unlike ``python -m pytest``, does
not add the current directory to ``sys.path``).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

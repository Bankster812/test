"""Bindet die Selbstpruefung des Maskenmodells in pytest ein.

Die eigentlichen Pruefungen stehen in cad/ktm_exc_mask/selftest.py und
laufen auch ohne pytest:  python3 cad/ktm_exc_mask/selftest.py
"""

import sys
from pathlib import Path

import pytest

CAD_DIR = Path(__file__).resolve().parents[1] / "cad" / "ktm_exc_mask"


@pytest.fixture(scope="module")
def selftest():
    pytest.importorskip("cadquery", reason="cadquery nicht installiert")
    pytest.importorskip("trimesh", reason="trimesh nicht installiert")
    sys.path.insert(0, str(CAD_DIR))
    import selftest as module
    return module


@pytest.mark.slow
def test_modell_besteht_alle_pruefungen(selftest):
    from ktm_mask.params import MaskParams

    params = CAD_DIR / "params.json"
    p = MaskParams.load(params) if params.exists() else MaskParams()
    p.split = True
    result = selftest.run(p)
    assert not result.failed, [row[0] for row in result.failed]


def test_parameter_sind_plausibel(selftest):
    from ktm_mask.params import MaskParams

    params = CAD_DIR / "params.json"
    p = MaskParams.load(params) if params.exists() else MaskParams()
    assert p.validate() == []

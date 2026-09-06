"""Parametrisches Modell einer modifizierten KTM-EXC-Scheinwerfermaske.

Einstiegspunkte liegen eine Ebene hoeher: `build.py` erzeugt die Dateien,
`selftest.py` misst das Ergebnis nach, `apply_to_scan.py` uebertraegt die
Merkmale auf ein reales Netz.

Alle Masse in Millimetern. Koordinaten: X quer (+ = rechts), Y hoch,
Z laengs (0 = Maskenvorderkante, negativ nach hinten).
"""

__all__ = ["assembly", "blinker", "features", "gauges", "geometry",
           "params", "rearmount", "shell"]

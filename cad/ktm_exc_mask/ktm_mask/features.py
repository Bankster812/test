"""Die uebertragenen Merkmale als Liste — eine Quelle fuer beide Wege.

`build.py` arbeitet damit auf der parametrischen Grundmaske,
`apply_to_scan.py` mit denselben Koerpern auf einem echten Scan. Was hier
steht, gilt also fuer beide.

Ein Merkmal ist entweder anzuformen (Adder) oder abzuziehen (Cutter).
`clip=True` heisst: der Koerper wird mit der Aussenhaut verschnitten, damit
er buendig anschliesst statt aus der Flanke zu ragen.
"""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq

from . import blinker, rearmount


@dataclass
class Feature:
    name: str
    solid: cq.Workplane
    clip: bool = False


def adders(p) -> list:
    """Anzuformende Koerper, in Reihenfolge."""
    out = []
    if p.blinker:
        for side, label in ((+1, "rechts"), (-1, "links")):
            out.append(Feature(f"blinker_verstaerkung_{label}",
                               blinker.reinforcement(p, side), clip=True))
    kind = p.rear_mount_type
    if kind == "plate":
        out.append(Feature("hinterer_rahmen", rearmount.plate(p)))
    if kind in ("plate", "tabs"):
        t = rearmount.tabs(p)
        if t is not None:
            out.append(Feature("befestigungslaschen", t, clip=(kind == "tabs")))
    if kind == "strap":
        pads, _ = rearmount.straps(p)
        if pads is not None:
            out.append(Feature("gummiband_verstaerkung", pads, clip=True))
    return out


def cutters(p) -> list:
    """Abzuziehende Koerper, in Reihenfolge."""
    out = []
    if p.blinker:
        for side, label in ((+1, "rechts"), (-1, "links")):
            out.append(Feature(f"blinker_aussparung_{label}",
                               blinker.pocket_cutter(p, side)))
            out.append(Feature(f"blinker_bohrungen_{label}",
                               blinker.hole_cutter(p, side)))
    if p.rear_mount_type == "strap":
        _, slots = rearmount.straps(p)
        if slots is not None:
            out.append(Feature("gummiband_durchbrueche", slots))
    return out

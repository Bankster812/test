"""Die uebertragenen Merkmale als Liste — eine Quelle fuer beide Wege.

`build.py` arbeitet damit auf dem parametrischen Grundkoerper,
`apply_to_scan.py` mit denselben Koerpern auf einem echten Netz.

Ein Merkmal ist entweder anzuformen (Adder) oder abzuziehen (Cutter).
`clip=True` heisst: beim Scan-Weg wird der Koerper zusaetzlich an der
Aussenhaut des Scans beschnitten, damit er buendig anschliesst.
"""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq

from . import blinker, rearmount, slots
from .shell import outer_solid


@dataclass
class Feature:
    name: str
    solid: cq.Workplane
    clip: bool = False


def adders(p, outer=None) -> list:
    """Anzuformende Koerper, in Reihenfolge."""
    if outer is None:
        outer = outer_solid(p)
    out = []

    pad, _ = rearmount.top_bracket(p, outer)
    if pad is not None:
        out.append(Feature("oberer_befestigungsbock", pad, clip=True))

    pads, _ = rearmount.light_bosses(p, outer)
    if pads is not None:
        out.append(Feature("scheinwerferdome", pads, clip=True))

    pads, _ = rearmount.tabs(p, outer)
    if pads is not None:
        out.append(Feature("befestigungslaschen", pads, clip=True))

    if p.stalk:
        for side, label in ((+1, "rechts"), (-1, "links")):
            out.append(Feature(f"schaftverstaerkung_{label}",
                               blinker.reinforcement(p, side), clip=True))
    return out


def cutters(p, outer=None) -> list:
    """Abzuziehende Koerper, in Reihenfolge."""
    if outer is None:
        outer = outer_solid(p)
    out = []

    for name, cutter in slots.cutters(p):
        out.append(Feature(name, cutter))

    _, hole = rearmount.top_bracket(p, outer)
    if hole is not None:
        out.append(Feature("bohrung_befestigungsbock", hole))

    _, holes = rearmount.light_bosses(p, outer)
    if holes is not None:
        out.append(Feature("bohrungen_scheinwerferdome", holes))

    _, holes = rearmount.tabs(p, outer)
    if holes is not None:
        out.append(Feature("bohrungen_laschen", holes))

    if p.stalk:
        for side, label in ((+1, "rechts"), (-1, "links")):
            out.append(Feature(f"schafttasche_{label}",
                               blinker.pocket_cutter(p, side)))
            out.append(Feature(f"schaftbohrung_{label}",
                               blinker.hole_cutter(p, side)))
    return out

"""Passproben — kleine Ausschnitte der Maske zum schnellen Testdruck.

Statt Stunden fuer die ganze Maske druckt man Minuten den Ausschnitt um ein
Merkmal herum, prueft am realen Teil und korrigiert die Parameter. Erst
danach die komplette Maske.
"""

from __future__ import annotations

import cadquery as cq

from .geometry import surface_x
from . import slots


def _box_at(center, size):
    return (cq.Workplane("XY").box(size[0], size[1], size[2])
            .translate(center))


def slot_gauge(p, body: cq.Workplane, side: int = +1) -> cq.Workplane:
    """Flankenausschnitt mit allen Blinkerhalter-Aussparungen."""
    ys = slots.slot_heights(p) or [p.slot_y]
    y_mid = (max(ys) + min(ys)) / 2.0
    span = (max(ys) - min(ys)) + p.slot_h + 26.0
    z_mid = p.slot_z0 + p.slot_z_len / 2.0
    x_edge = surface_x(p.outline_ccw, p.width, p.height, p.level_objs,
                       y_mid, max(z_mid, p.z_rear))
    depth = p.slot_depth + 28.0
    return body.intersect(_box_at(
        (side * (x_edge - depth / 2.0 + 6.0), y_mid, z_mid),
        (depth, span, p.slot_z_len + 30.0)))


def top_gauge(p, body: cq.Workplane) -> cq.Workplane:
    """Ausschnitt mit dem oberen Befestigungsbock."""
    return body.intersect(_box_at(
        (0.0, p.top_bracket_y, p.z_front / 2.0),
        (p.top_bracket_w + 44.0, p.top_bracket_h + 40.0, p.depth + 20.0)))


def light_gauge(p, body: cq.Workplane) -> cq.Workplane:
    """Ausschnitt mit einem Scheinwerferdom und der Kante des Ausschnitts."""
    cx = p.light_boss_dx
    cy = p.light_y + p.light_boss_dy
    return body.intersect(_box_at(
        (cx * 0.55, cy, p.z_front / 2.0),
        (cx + p.light_boss_d + 20.0, p.light_boss_d + 40.0, p.depth + 20.0)))


ALL = {"aussparungen_rechts": lambda p, b: slot_gauge(p, b, +1),
       "aussparungen_links": lambda p, b: slot_gauge(p, b, -1),
       "befestigungsbock": top_gauge,
       "scheinwerferdom": light_gauge}

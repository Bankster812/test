"""Passproben — kleine Ausschnitte der Maske zum schnellen Testdruck.

Statt 8 Stunden fuer die ganze Maske druckt man 20 Minuten den Ausschnitt
um ein Merkmal herum, prueft die Passung am realen Teil und korrigiert die
Parameter. Erst danach die komplette Maske.
"""

from __future__ import annotations

import math

import cadquery as cq

from . import blinker as blinker_mod
from . import rearmount


def _box_at(center, size):
    return (cq.Workplane("XY")
            .box(size[0], size[1], size[2])
            .translate(center))


def blinker_gauge(p, body: cq.Workplane, side: int = +1) -> cq.Workplane:
    """Flankenausschnitt mit kompletter Blinkeraufnahme."""
    cx, cy, cz = blinker_mod.mount_point(p, side)
    size = (max(60.0, p.boss_d + p.pocket_depth + 40.0),
            p.pocket_h + 34.0, p.pocket_w + 34.0)
    return body.intersect(_box_at((cx - side * size[0] / 2.0 + side * 12.0,
                                   cy, cz), size))


def rear_gauge(p, body: cq.Workplane) -> cq.Workplane:
    """Ausschnitt der Rueckseite mit einer Befestigungslasche."""
    if p.rear_mount_type in ("none", "strap") or not p.tab_angles_deg:
        z = p.z_rear
        return body.intersect(_box_at((0.0, 0.0, z + 12.0), (120.0, 120.0, 30.0)))
    ang = math.radians(p.tab_angles_deg[0])
    rear = p.section_objs[-1]
    anchor = (rearmount._scaled_section(rear, p.plate_aperture)
              if p.rear_mount_type == "plate" else rear)
    r = rearmount._contour_radius(anchor, p.tab_angles_deg[0])
    cx, cy = r * math.cos(ang), r * math.sin(ang)
    return body.intersect(_box_at((cx * 0.85, cy * 0.85, p.z_rear + 15.0),
                                  (p.tab_len + 60.0, p.tab_w + 50.0, 34.0)))


def rim_gauge(p, body: cq.Workplane) -> cq.Workplane:
    """Stueck der Vorderkante — prueft Wandstaerke und Randradius."""
    sec = p.section_objs[0]
    return body.intersect(_box_at((0.0, sec.y_offset + sec.height / 2.0 - 15.0,
                                   -14.0), (90.0, 40.0, 32.0)))


ALL = {"blinker_rechts": lambda p, b: blinker_gauge(p, b, +1),
       "blinker_links": lambda p, b: blinker_gauge(p, b, -1),
       "rueckseite": rear_gauge,
       "vorderkante": rim_gauge}

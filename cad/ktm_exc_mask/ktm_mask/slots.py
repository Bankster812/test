"""Blinkerhalter-Aussparungen — die Uebernahme von der Stark-EX-Maske.

Offene Schlitze in der Seitenkante, je drei uebereinander pro Seite. Der
Halter des Blinkers schiebt sich von aussen ein; die drei Positionen sind
die Hoehenverstellung.

Beide Seiten sind gespiegelt gleich ausgefuehrt. Auf dem Kunststoff-Render
der Spendermaske weicht eine Seite ab (zusaetzlicher flacher Halter mit
Doppelloch); massgeblich ist die symmetrische Ausfuehrung der
Carbon-Version, und die ist hier umgesetzt.

Die Tiefe wird an jeder Stelle von der OERTLICHEN Aussenkante gemessen.
Weil die Kontur nach vorn hin kleiner wird, folgt der Schnittkoerper dieser
Neigung — sonst wuerde der Schlitz vorn tiefer schneiden als hinten.
"""

from __future__ import annotations

import math

import cadquery as cq

from .geometry import surface_x

OVERHANG = 10.0   # wie weit der Schnittkoerper vor der Kante beginnt


def slot_heights(p):
    """Hoehenlagen der Aussparungen, mittig um `slot_y` verteilt."""
    n = max(0, p.slot_count)
    return [p.slot_y + (i - (n - 1) / 2.0) * p.slot_pitch for i in range(n)]


def _edge_x(p, y: float, z: float) -> float:
    return surface_x(p.outline_ccw, p.width, p.height, p.level_objs, y, z)


def _rrect_wire(cx: float, cy: float, z: float, w: float, h: float, r: float):
    """Abgerundetes Rechteck als geschlossener Draht auf Tiefe z.

    Bewusst aus Strecken und Drei-Punkt-Boegen zusammengesetzt: die
    Sketch-Verrundung laesst sich nicht frei positionieren, und aus einem
    platzierten Sketch bekommt man keinen Draht fuer den Loft heraus.
    """
    hw, hh = w / 2.0, h / 2.0
    r = max(0.0, min(r, min(hw, hh) - 1e-6))
    k = r * (1.0 - math.sqrt(0.5))          # Bogenmitte, 45 Grad
    return (
        cq.Workplane("XY").workplane(offset=z)
        .moveTo(cx - hw + r, cy - hh).lineTo(cx + hw - r, cy - hh)
        .threePointArc((cx + hw - k, cy - hh + k), (cx + hw, cy - hh + r))
        .lineTo(cx + hw, cy + hh - r)
        .threePointArc((cx + hw - k, cy + hh - k), (cx + hw - r, cy + hh))
        .lineTo(cx - hw + r, cy + hh)
        .threePointArc((cx - hw + k, cy + hh - k), (cx - hw, cy + hh - r))
        .lineTo(cx - hw, cy - hh + r)
        .threePointArc((cx - hw + k, cy - hh + k), (cx - hw + r, cy - hh))
        .close().wire().val()
    )


def _profile_wire(p, y: float, z: float, side: int):
    """Querschnitt des Schnittkoerpers auf Tiefe z, an die oertliche Kante gelegt."""
    x_edge = _edge_x(p, y, z)
    length = p.slot_depth + OVERHANG
    x_center = side * (x_edge + OVERHANG / 2.0 - p.slot_depth / 2.0)
    return _rrect_wire(x_center, y, z, length, p.slot_h, p.slot_fillet)


def slot_cutter(p, y: float, side: int) -> cq.Workplane:
    """Ein Schnittkoerper, der der Neigung der Flanke folgt.

    Die Kontur wird nach vorn hin kleiner, die Kante wandert also nach
    innen. Ein gerades Prisma wuerde deshalb hinten deutlich tiefer
    schneiden als vorn; der Loft zwischen zwei an die oertliche Kante
    gelegten Profilen haelt die Tiefe konstant.
    """
    z0 = p.slot_z0
    z1 = p.slot_z0 + p.slot_z_len
    w0 = _profile_wire(p, y, max(z0, p.z_rear), side)
    w1 = _profile_wire(p, y, min(z1, p.z_front), side)
    return cq.Workplane(obj=cq.Solid.makeLoft([w0, w1], ruled=True))


def cutters(p) -> list:
    """Alle Aussparungen, links wie rechts."""
    if not p.slots or p.slot_count <= 0:
        return []
    out = []
    for i, y in enumerate(slot_heights(p), start=1):
        for side, label in ((+1, "rechts"), (-1, "links")):
            out.append((f"aussparung_{label}_{i}", slot_cutter(p, y, side)))
    return out

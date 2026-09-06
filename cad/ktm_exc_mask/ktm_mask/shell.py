"""Grundkoerper der EXC-Maske: Aussenhaut, Kavitaet, Scheinwerferausschnitt."""

from __future__ import annotations

import cadquery as cq

from .geometry import Section, inner_points, superellipse_points


def section_wire(sec: Section, num_points: int, z: float | None = None):
    """Geschlossener Spline-Draht eines Querschnitts als CadQuery-Wire."""
    pts = superellipse_points(sec, num_points)
    zz = sec.z if z is None else z
    return (
        cq.Workplane("XY")
        .workplane(offset=zz)
        .spline(pts, periodic=True)
        .wire()
        .val()
    )


def outer_solid(p) -> cq.Workplane:
    """Massiver Aussenkoerper (noch ohne Hohlraum)."""
    wires = [section_wire(s, p.section_points) for s in p.section_objs]
    return cq.Workplane(obj=cq.Solid.makeLoft(wires, ruled=False))


def cavity_solid(p) -> cq.Workplane:
    """Innenkoerper — wird vom Aussenkoerper abgezogen.

    Beginnt `wall` hinter der Vorderkante (die Stirnflaeche bleibt also
    stehen) und laeuft hinten ueber die Maske hinaus, damit die Rueckseite
    offen bleibt. Die Konturen sind echte Parallelkurven zur Aussenhaut.
    """
    secs = p.section_objs
    z_front = secs[0].z - p.wall
    z_back = secs[-1].z - 10.0

    zs = [z_front]
    zs += [s.z for s in secs if z_back < s.z < z_front]
    zs.append(z_back)

    wires = []
    for z in zs:
        pts = inner_points(secs, z, p.wall, p.section_points)
        wires.append(cq.Workplane("XY").workplane(offset=z)
                     .spline(pts, periodic=True).wire().val())
    return cq.Workplane(obj=cq.Solid.makeLoft(wires, ruled=False))


def _light_sketch(p):
    """2D-Umriss des Scheinwerferausschnitts in der XY-Ebene."""
    if p.light_shape == "round":
        return cq.Sketch().circle(p.light_d / 2.0)
    r = min(p.light_r, min(p.light_w, p.light_h) / 2.0 - 1e-6)
    sk = cq.Sketch().rect(p.light_w, p.light_h)
    if r > 0:
        sk = sk.vertices().fillet(r)
    return sk


def light_cutter(p) -> cq.Workplane:
    """Durchbruch fuer den Scheinwerfer, durch die ganze Stirnwand."""
    return (
        cq.Workplane("XY")
        .workplane(offset=20.0)
        .center(0.0, p.light_y)
        .placeSketch(_light_sketch(p))
        .extrude(-(20.0 + p.wall + 10.0))
    )


def light_lip(p) -> cq.Workplane:
    """Auflagekragen innen: der Scheinwerfer legt sich von hinten dagegen."""
    if p.light_lip <= 0.0 or p.light_lip_t <= 0.0:
        return None
    if p.light_shape == "round":
        outer = cq.Sketch().circle(p.light_d / 2.0 + p.light_lip)
    else:
        r = min(p.light_r + p.light_lip,
                min(p.light_w, p.light_h) / 2.0 + p.light_lip - 1e-6)
        outer = cq.Sketch().rect(p.light_w + 2 * p.light_lip,
                                 p.light_h + 2 * p.light_lip)
        if r > 0:
            outer = outer.vertices().fillet(r)
    # ab der Stirnflaeche aufbauen, nicht ab der Wandrueckseite: eine
    # Vereinigung auf exakt aufeinanderliegenden Flaechen ist unzuverlaessig
    return (
        cq.Workplane("XY")
        .workplane(offset=0.0)
        .center(0.0, p.light_y)
        .placeSketch(outer)
        .extrude(-(p.wall + p.light_lip_t))
    )


def light_screw_bosses(p):
    """Schraubdome fuer die Scheinwerferbefestigung: (Material, Bohrungen)."""
    if p.light_screws <= 0:
        return None, None
    import math

    r = p.light_screw_bcd / 2.0
    bosses, holes = None, None
    for i in range(p.light_screws):
        # gleichmaessig verteilt, erster Dom oben
        a = math.radians(90.0 + 360.0 * i / p.light_screws)
        x, y = r * math.cos(a), r * math.sin(a) + p.light_y
        boss = (
            cq.Workplane("XY").workplane(offset=0.0)
            .center(x, y).circle(p.light_screw_boss_d / 2.0)
            .extrude(-(p.wall + p.light_screw_boss_t))
        )
        hole = (
            cq.Workplane("XY").workplane(offset=5.0)
            .center(x, y).circle(p.light_screw_d / 2.0)
            .extrude(-(5.0 + p.wall + p.light_screw_boss_t + 2.0))
        )
        bosses = boss if bosses is None else bosses.union(boss)
        holes = hole if holes is None else holes.union(hole)
    return bosses, holes


def hollow(p, outer: cq.Workplane) -> cq.Workplane:
    """Aushoehlen, Rueckseite offen.

    Bewusst ueber Loft-und-Abziehen statt ueber das Flaechen-Offset des
    Kernels: dessen Ergebnis meldet sich zwar als gueltig, liefert bei
    dieser Geometrie aber falsche Volumina und zerstoert nachfolgende
    boolesche Operationen (Teilung, Verschneidungen).
    """
    return outer.cut(cavity_solid(p))


def build_shell(p, outer: cq.Workplane | None = None) -> cq.Workplane:
    """Fertige Grundmaske: Hohlkoerper mit Scheinwerferausschnitt."""
    if outer is None:
        outer = outer_solid(p)
    body = hollow(p, outer)

    if p.light_cut:
        lip = light_lip(p)
        if lip is not None:
            # Kragen nur innerhalb der Maske stehen lassen
            body = body.union(lip.intersect(outer))
        bosses, holes = light_screw_bosses(p)
        if bosses is not None:
            body = body.union(bosses.intersect(outer))
        body = body.cut(light_cutter(p))
        if holes is not None:
            body = body.cut(holes)
    return body

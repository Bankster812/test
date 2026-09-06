"""Grundkoerper: gewoelbtes Schild, hinten offen, mit Scheinwerferausschnitt."""

from __future__ import annotations

import cadquery as cq

from .geometry import (cavity_level, level_points, offset_folds,
                       polygon_area)

MIN_CAVITY_AREA = 40.0    # mm^2 — darunter lohnt keine weitere Ebene

# Die Kavitaet ist ein Loft ueber gut zwanzig Konturen und wird je
# Wandstaerke mehrfach gebraucht: einmal fuer den Hohlraum, dann noch einmal
# fuer jede oertliche Aufdickung. Ohne diesen Zwischenspeicher baut sie sich
# ein halbes Dutzend Mal neu auf.
_CAVITY_CACHE: dict = {}


def _wire(pts, z: float, spline: bool):
    wp = cq.Workplane("XY").workplane(offset=z)
    wp = wp.spline(pts, periodic=True) if spline else wp.polyline(pts).close()
    return wp.wire().val()


def _level_zs(p, extra_per_gap: int = 2):
    """Tiefen fuer den Loft — Zwischenebenen glaetten die Woelbung."""
    zs = []
    lv = p.level_objs
    for a, b in zip(lv, lv[1:]):
        zs.append(a.z)
        for k in range(1, extra_per_gap + 1):
            zs.append(a.z + (b.z - a.z) * k / (extra_per_gap + 1))
    zs.append(lv[-1].z)
    return zs


def outer_solid(p) -> cq.Workplane:
    """Massiver Aussenkoerper — noch ohne Hohlraum."""
    wires = [
        _wire(level_points(p.outline_ccw, p.width, p.height, p.level_objs, z),
              z, p.spline_outline)
        for z in _level_zs(p)
    ]
    # ruled=True, nicht der glatte Loft: der glatte kommt mit der winzigen
    # Scheitelkontur nicht zurecht und liefert einen Koerper, der sich als
    # gueltig meldet, bei jeder booleschen Operation aber nichts mehr
    # zurueckgibt. Die Woelbung entsteht stattdessen ueber die
    # Zwischenebenen in _level_zs.
    return cq.Workplane(obj=cq.Solid.makeLoft(wires, ruled=True))


def _cavity_key(p, wall: float):
    return (tuple(map(tuple, p.outline)), tuple(map(tuple, p.levels)),
            p.width, p.height, p.spline_outline, round(wall, 6))


def cavity_solid(p, wall: float | None = None) -> cq.Workplane:
    """Innenkoerper — wird vom Aussenkoerper abgezogen.

    Die Konturen sind echte Parallelflaechen zur Aussenhaut: sie wandern
    nicht nur nach innen, sondern auch nach hinten (siehe
    geometry.cavity_level). Hinten laeuft der Koerper ueber die Maske
    hinaus, damit die Rueckseite offen bleibt.

    Mit `wall` groesser als der Wandstaerke entsteht ein kleinerer
    Innenkoerper — davon leben alle oertlichen Aufdickungen (Doeme, Boecke).
    """
    wall = p.wall if wall is None else wall
    key = _cavity_key(p, wall)
    cached = _CAVITY_CACHE.get(key)
    if cached is not None:
        return cached

    z_back, back_pts = cavity_level(p.outline_ccw, p.width, p.height,
                                    p.level_objs, p.z_rear, wall)
    wires = [_wire(back_pts, z_back - 12.0, p.spline_outline)]
    last_z = z_back - 12.0

    for z in _level_zs(p):
        z_new, pts = cavity_level(p.outline_ccw, p.width, p.height,
                                  p.level_objs, z, wall)
        if z_new <= last_z + 1e-6:
            continue
        outer_pts = level_points(p.outline_ccw, p.width, p.height,
                                 p.level_objs, z)
        if offset_folds(outer_pts, pts) > 0:
            # Sollte mit der Parallelflaechen-Rechnung nicht mehr vorkommen;
            # eine gefaltete Kontur wuerde jeden Loft darauf unbrauchbar
            # machen, deshalb hier abbrechen statt sie einzubauen.
            break
        if polygon_area(pts) < MIN_CAVITY_AREA:
            break
        wires.append(_wire(pts, z_new, p.spline_outline))
        last_z = z_new

    if len(wires) < 2:
        raise ValueError("Kavitaet laesst sich nicht bilden — Wandstaerke "
                         "zu gross fuer diese Kontur")
    # dasselbe Verfahren wie beim Aussenkoerper — sonst laufen Aussenhaut
    # und Kavitaet unterschiedlich und die Wandstaerke schwankt
    result = cq.Workplane(obj=cq.Solid.makeLoft(wires, ruled=True))
    _CAVITY_CACHE[key] = result
    return result


def _rounded_rect(w: float, h: float, r: float) -> cq.Sketch:
    r = max(0.0, min(r, min(w, h) / 2.0 - 1e-6))
    sk = cq.Sketch().rect(w, h)
    return sk.vertices().fillet(r) if r > 0 else sk


def light_cutter(p) -> cq.Workplane:
    """Durchbruch fuer den Scheinwerfer — hohes Oval im unteren Drittel."""
    return (
        cq.Workplane("XY")
        .workplane(offset=p.z_front + 30.0)
        .center(0.0, p.light_y)
        .placeSketch(_rounded_rect(p.light_w, p.light_h, p.light_r))
        .extrude(-(p.depth + 60.0))
    )


def light_lip(p) -> cq.Workplane:
    """Auflagekragen: der Scheinwerfer legt sich von hinten dagegen."""
    if p.light_lip <= 0.0 or p.light_lip_t <= 0.0:
        return None
    return (
        cq.Workplane("XY")
        .workplane(offset=p.z_front + 30.0)
        .center(0.0, p.light_y)
        .placeSketch(_rounded_rect(p.light_w + 2 * p.light_lip,
                                   p.light_h + 2 * p.light_lip,
                                   p.light_r + p.light_lip))
        .extrude(-(p.depth + 60.0 - p.light_lip_t))
    )


def build_shell(p, outer: cq.Workplane | None = None) -> cq.Workplane:
    """Hohlkoerper mit Scheinwerferausschnitt und Auflagekragen."""
    if outer is None:
        outer = outer_solid(p)
    body = outer.cut(cavity_solid(p))

    if p.light_cut:
        lip = light_lip(p)
        if lip is not None:
            # Der Kragen wird aus dem Vollen genommen und auf die Aussenhaut
            # beschnitten; danach schneidet der Durchbruch das Loch hinein.
            body = body.union(lip.intersect(outer))
        body = body.cut(light_cutter(p))
    return body

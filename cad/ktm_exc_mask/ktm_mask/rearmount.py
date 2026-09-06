"""Rueckseite der Maske — die Uebernahme von der Spendermaske.

Aus den Referenzbildern uebernommen:

  * oberer Befestigungsbock mittig, als erhabener Sockel an der Innenseite
    mit Durchgangsloch — der obere Anbindungspunkt der Maske
  * vier Doeme rings um den Scheinwerferausschnitt, zwei oben, zwei unten,
    an denen der Scheinwerfer verschraubt wird
  * optionale weitere Befestigungslaschen ueber `tab_positions`

Alle drei entstehen nach demselben Muster: ein Prisma durch die Maske wird
mit dem Aussenkoerper verschnitten und davon eine um `t` staerker versetzte
Kavitaet abgezogen. Uebrig bleibt ein Feld, das der Woelbung folgt und die
Wand oertlich von `wall` auf `wall + t` aufdickt — genau das, was ein
angespritzter Dom ist.
"""

from __future__ import annotations

import cadquery as cq

from .shell import cavity_solid


def _pads(p, outer: cq.Workplane, items, thickness: float):
    """Oertliche Aufdickungen der Wand in den Umrissen von `items`.

    `items` ist eine Liste (sketch, (cx, cy)). Alle Umrisse gleicher Dicke
    werden zusammen verrechnet: der Abzug der Kavitaet ist die teure
    Operation und faellt so einmal an statt je Dom.
    """
    prism = None
    for sketch, center in items:
        part = (
            cq.Workplane("XY")
            .workplane(offset=p.z_front + 30.0)
            .center(center[0], center[1])
            .placeSketch(sketch)
            .extrude(-(p.depth + 60.0))
        )
        prism = part if prism is None else prism.union(part)
    if prism is None:
        return None
    return prism.intersect(outer).cut(cavity_solid(p, p.wall + thickness))


def _hole(p, center, diameter: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .workplane(offset=p.z_front + 30.0)
        .center(center[0], center[1])
        .circle(diameter / 2.0)
        .extrude(-(p.depth + 60.0))
    )


def top_bracket(p, outer):
    """Oberer Befestigungsbock: (Aufdickung, Bohrung)."""
    if not p.top_bracket:
        return None, None
    center = (0.0, p.top_bracket_y)
    r = min(p.top_bracket_w, p.top_bracket_h) / 4.0
    sk = cq.Sketch().rect(p.top_bracket_w, p.top_bracket_h).vertices().fillet(r)
    return (_pads(p, outer, [(sk, center)], p.top_bracket_t),
            _hole(p, center, p.top_bracket_hole_d))


def light_bosses(p, outer):
    """Vier Schraubdome um den Scheinwerferausschnitt: (Aufdickung, Bohrungen)."""
    if not p.light_boss:
        return None, None
    items, holes = [], None
    for sx in (+1, -1):
        for sy in (+1, -1):
            center = (sx * p.light_boss_dx, p.light_y + sy * p.light_boss_dy)
            items.append((cq.Sketch().circle(p.light_boss_d / 2.0), center))
            hole = _hole(p, center, p.light_boss_hole_d)
            holes = hole if holes is None else holes.union(hole)
    return _pads(p, outer, items, p.light_boss_t), holes


def tabs(p, outer):
    """Zusaetzliche Befestigungslaschen an frei gesetzten Positionen."""
    if p.rear_mount_type != "tabs" or not p.tab_positions:
        return None, None
    items, holes = [], None
    r = min(p.tab_w, p.tab_len) / 3.0
    for x, y in p.tab_positions:
        center = (float(x), float(y))
        items.append((cq.Sketch().rect(p.tab_len, p.tab_w).vertices().fillet(r),
                      center))
        hole = _hole(p, center, p.tab_hole_d)
        holes = hole if holes is None else holes.union(hole)
    return _pads(p, outer, items, p.tab_t), holes

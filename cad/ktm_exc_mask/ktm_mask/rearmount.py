"""Rueckseite der Maske — die Uebernahme von der Spendermaske.

Nach den Referenzbildern nachgebildet:

  * **Verrippung** der Innenseite — das praegende Merkmal: duenne Waende,
    die als Netz ueber die Innenflaeche laufen, dazu ein Kamm kurzer Finnen
    im oberen Bereich
  * **oberer Befestigungsbock** als erhabener Sattel mit Mulde,
    Durchgangsloch und zwei flankierenden Stuetzrippen
  * **Halterstreifen** fuer den Scheinwerfer: laenglich und radial zur
    Ausschnittmitte ausgerichtet, nicht rund
  * **Zapfen** an der Innenseite als Aufnahme fuer Clips oder Kabelfuehrung
  * optionale Befestigungslaschen ueber `tab_positions`

Zwei Bauweisen liegen dem zugrunde:

  Aufdickung  Ein Prisma wird mit dem Aussenkoerper verschnitten, davon eine
              um `t` staerker versetzte Kavitaet abgezogen. Uebrig bleibt
              ein Feld, das der Woelbung folgt und die Wand oertlich von
              `wall` auf `wall + t` aufdickt — ein angespritzter Dom.

  Aufbau      Ein Prisma wird mit der SCHICHT zwischen der normalen und
              einer um `t` tieferen Kavitaet verschnitten. Uebrig bleibt
              ein Koerper, der auf der Innenflaeche steht und ihrer
              Woelbung folgt — eine Rippe oder ein Zapfen.
"""

from __future__ import annotations

import math

import cadquery as cq

from .shell import cavity_solid


def _prism(p, sketch: cq.Sketch, center):
    """Prisma im Umriss von `sketch`, laengs durch die ganze Maske."""
    return (
        cq.Workplane("XY")
        .workplane(offset=p.z_front + 30.0)
        .center(center[0], center[1])
        .placeSketch(sketch)
        .extrude(-(p.depth + 60.0))
    )


def _pads(p, outer: cq.Workplane, items, thickness: float):
    """Oertliche Aufdickungen der Wand in den Umrissen von `items`.

    `items` ist eine Liste (sketch, (cx, cy), winkel_grad). Alles mit
    gleicher Dicke wird zusammen verrechnet: der Abzug der Kavitaet ist die
    teure Operation und faellt so einmal an statt je Merkmal.
    """
    prism = None
    for sketch, center, angle in items:
        part = _prism(p, sketch, center)
        if angle:
            part = part.rotate((center[0], center[1], 0.0),
                               (center[0], center[1], 1.0), angle)
        prism = part if prism is None else prism.union(part)
    if prism is None:
        return None
    return prism.intersect(outer).cut(cavity_solid(p, p.wall + thickness))


_LAYER_CACHE: dict = {}

# Wie weit aufstehende Koerper in die Wand hineinreichen.
# Ohne diese Ueberlappung faellt ihre Aussenflaeche exakt mit der
# Innenflaeche der Schale zusammen, und die Vereinigung zweier Koerper auf
# genau aufeinanderliegenden Flaechen ist unzuverlaessig: das Ergebnis kam
# mit offenen Kanten heraus und war nicht mehr druckbar.
STAND_OVERLAP = 0.6

# Wie weit vor dem Scheitel aufstehende Koerper enden.
# Zum Scheitel hin legt sich die Woelbung nach vorn um und die Schicht
# duennt aus; eine Rippe laeuft dort als Messerschneide aus, die sich nicht
# mehr sauber vernetzen laesst — das Ergebnis kam mit offenen Kanten heraus.
# Rippen haben in der Scheitelkalotte ohnehin nichts zu suchen, also enden
# sie an einer sauberen Ebene davor.
STAND_Z_MARGIN = 14.0


def _inner_layer(p, thickness: float):
    """Schicht der Dicke `thickness` auf der Innenflaeche, mit Ueberlappung.

    Alles, was hierin liegt, steht auf der Innenwand und folgt ihrer
    Woelbung — die Grundlage fuer Rippen und Zapfen. Der Abzug zweier
    grosser Lofts ist teuer und wird je Hoehe nur einmal gerechnet.
    """
    key = (tuple(map(tuple, p.outline)), tuple(map(tuple, p.levels)),
           p.width, p.height, p.spline_outline, p.wall, round(thickness, 6))
    cached = _LAYER_CACHE.get(key)
    if cached is None:
        inner = cavity_solid(p, max(0.2, p.wall - STAND_OVERLAP))
        cached = inner.cut(cavity_solid(p, p.wall + thickness))
        _LAYER_CACHE[key] = cached
    return cached


def _standing(p, items, height: float):
    """Koerper, die auf der Innenflaeche stehen (Rippen, Zapfen)."""
    prism = None
    for sketch, center, angle in items:
        part = _prism(p, sketch, center)
        if angle:
            part = part.rotate((center[0], center[1], 0.0),
                               (center[0], center[1], 1.0), angle)
        prism = part if prism is None else prism.union(part)
    if prism is None:
        return None
    cap = (
        cq.Workplane("XY")
        .workplane(offset=p.z_rear - 20.0)
        .rect(4.0 * p.width, 4.0 * p.height)
        .extrude(p.depth + 20.0 - STAND_Z_MARGIN)
    )
    return prism.intersect(_inner_layer(p, height)).intersect(cap)


def _hole(p, center, diameter: float) -> cq.Workplane:
    return (
        cq.Workplane("XY")
        .workplane(offset=p.z_front + 30.0)
        .center(center[0], center[1])
        .circle(diameter / 2.0)
        .extrude(-(p.depth + 60.0))
    )


def _rounded(w: float, h: float, r: float | None = None) -> cq.Sketch:
    r = min(w, h) / 3.0 if r is None else r
    r = max(0.0, min(r, min(w, h) / 2.0 - 1e-6))
    sk = cq.Sketch().rect(w, h)
    return sk.vertices().fillet(r) if r > 0 else sk


def ribs(p):
    """Rippennetz auf der Innenseite."""
    if not p.ribs or not p.rib_segments:
        return None
    items = []
    for x1, y1, x2, y2 in p.rib_segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        items.append((cq.Sketch().rect(length, p.rib_t), center,
                      math.degrees(math.atan2(dy, dx))))
    return _standing(p, items, p.rib_h)


def posts(p):
    """Zapfen an der Innenseite: (Aufbau, Bohrungen)."""
    if not p.post_positions:
        return None, None
    items, bores = [], None
    for x, y in p.post_positions:
        center = (float(x), float(y))
        items.append((cq.Sketch().circle(p.post_d / 2.0), center, 0.0))
        if p.post_bore > 0:
            bore = _hole(p, center, p.post_bore)
            bores = bore if bores is None else bores.union(bore)
    return _standing(p, items, p.post_len), bores


def top_bracket(p, outer):
    """Oberer Befestigungsbock als Sattel: (Aufbau, Abzuege).

    Der Sattel selbst ist eine Aufdickung, die Mulde und das Durchgangsloch
    sind Abzuege, die beiden Stuetzrippen stehen als Rippen daneben.
    """
    if not p.top_bracket:
        return None, None
    center = (0.0, p.top_bracket_y)
    saddle = _pads(p, outer,
                   [(_rounded(p.top_bracket_w, p.top_bracket_h), center, 0.0)],
                   p.top_bracket_t)

    wings = None
    if p.top_bracket_wing > 0:
        items = []
        for side in (+1, -1):
            x = side * (p.top_bracket_w / 2.0 + p.top_bracket_wing / 2.0 - 1.0)
            items.append((cq.Sketch().rect(p.top_bracket_wing, p.rib_t),
                          (x, p.top_bracket_y), 0.0))
        wings = _standing(p, items, p.top_bracket_t * 0.8)
    if wings is not None:
        saddle = saddle.union(wings)

    cuts = _hole(p, center, p.top_bracket_hole_d)
    if p.top_bracket_recess_t > 0:
        recess = (
            cq.Workplane("XY")
            .workplane(offset=p.z_rear - p.top_bracket_t - 20.0)
            .center(center[0], center[1])
            .placeSketch(_rounded(p.top_bracket_recess_w,
                                  p.top_bracket_recess_h))
            .extrude(20.0 + p.top_bracket_recess_t)
        )
        cuts = cuts.union(recess)
    return saddle, cuts


def light_bosses(p, outer):
    """Halterstreifen des Scheinwerfers: (Aufdickung, Bohrungen).

    Radial zur Ausschnittmitte ausgerichtet — auf der Carbon-Version sind
    das aufgeschraubte Streifen, auf dem Kunststoffteil angespritzte
    Stege. Beides laeuft in dieselbe Richtung.
    """
    if not p.light_boss:
        return None, None
    items, holes = [], None
    for sx in (+1, -1):
        for sy in (+1, -1):
            cx = sx * p.light_boss_dx
            cy = p.light_y + sy * p.light_boss_dy
            angle = math.degrees(math.atan2(cy - p.light_y, cx))
            items.append((_rounded(p.light_boss_len, p.light_boss_w,
                                   p.light_boss_w / 2.0), (cx, cy), angle))
            # Bohrung am aeusseren Ende des Streifens
            reach = (p.light_boss_len / 2.0 - p.light_boss_w / 2.0) * 0.75
            hx = cx + reach * math.cos(math.radians(angle))
            hy = cy + reach * math.sin(math.radians(angle))
            hole = _hole(p, (hx, hy), p.light_boss_hole_d)
            holes = hole if holes is None else holes.union(hole)
    return _pads(p, outer, items, p.light_boss_t), holes


def tabs(p, outer):
    """Zusaetzliche Befestigungslaschen an frei gesetzten Positionen."""
    if p.rear_mount_type != "tabs" or not p.tab_positions:
        return None, None
    items, holes = [], None
    for x, y in p.tab_positions:
        center = (float(x), float(y))
        items.append((_rounded(p.tab_len, p.tab_w), center, 0.0))
        hole = _hole(p, center, p.tab_hole_d)
        holes = hole if holes is None else holes.union(hole)
    return _pads(p, outer, items, p.tab_t), holes

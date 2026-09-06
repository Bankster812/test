"""Querschnitts-Geometrie der Maske — reine Mathematik, kein CAD-Kernel.

Koordinatensystem (durchgaengig im ganzen Paket, Einheit Millimeter):

    X   quer zur Fahrtrichtung, + = rechte Fahrzeugseite
    Y   hoch,                   + = oben
    Z   laengs,                 0 = vordere Maskenebene, negativ = nach hinten

Die Maske wird als Loft durch mehrere geschlossene Querschnitte beschrieben.
Jeder Querschnitt ist eine Supserellipse (Exponent `n`), zusaetzlich unten
eingezogen (`narrow`), damit die typische Enduro-Schildform entsteht:
oben breit, nach unten zulaufend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Section:
    """Ein Querschnitt der Maske auf Hoehe z."""

    z: float          # Tiefe (0 = Maskenvorderkante, negativ nach hinten)
    width: float      # Gesamtbreite
    height: float     # Gesamthoehe
    exponent: float   # Superellipsen-Exponent: 2 = Ellipse, 4+ = kastiger
    y_offset: float   # vertikale Verschiebung der Querschnittsmitte
    narrow: float     # Einzug unten, 0 = keiner, 0.4 = stark zulaufend

    @classmethod
    def from_list(cls, v) -> "Section":
        return cls(*(float(x) for x in v))

    def as_list(self):
        return [self.z, self.width, self.height, self.exponent,
                self.y_offset, self.narrow]


def _narrow_factor(y_local: float, half_h: float, narrow: float) -> float:
    """Breiten-Skalierung: unterhalb der Mitte laeuft der Querschnitt zu."""
    if narrow <= 0.0 or y_local >= 0.0 or half_h <= 0.0:
        return 1.0
    t = min(1.0, abs(y_local) / half_h)
    return max(0.05, 1.0 - narrow * t)


def superellipse_points(sec: Section, num_points: int = 72):
    """Punkte auf dem Querschnitt, gegen den Uhrzeigersinn, geschlossen."""
    half_w, half_h = sec.width / 2.0, sec.height / 2.0
    inv = 2.0 / sec.exponent
    pts = []
    for i in range(num_points):
        a = 2.0 * math.pi * i / num_points
        ca, sa = math.cos(a), math.sin(a)
        x = half_w * math.copysign(abs(ca) ** inv, ca)
        y_local = half_h * math.copysign(abs(sa) ** inv, sa)
        x *= _narrow_factor(y_local, half_h, sec.narrow)
        pts.append((x, y_local + sec.y_offset))
    return pts


def interp_section(sections, z: float) -> Section:
    """Querschnitt an beliebiger Tiefe z (linear interpoliert bzw. geklemmt).

    `sections` muss nach z absteigend sortiert sein (0, -20, -45, ...).
    """
    if not sections:
        raise ValueError("keine Querschnitte definiert")
    if z >= sections[0].z:
        return sections[0]
    if z <= sections[-1].z:
        return sections[-1]
    for a, b in zip(sections, sections[1:]):
        if b.z <= z <= a.z:
            span = a.z - b.z
            t = 0.0 if span == 0 else (a.z - z) / span
            return Section(
                z=z,
                width=a.width + t * (b.width - a.width),
                height=a.height + t * (b.height - a.height),
                exponent=a.exponent + t * (b.exponent - a.exponent),
                y_offset=a.y_offset + t * (b.y_offset - a.y_offset),
                narrow=a.narrow + t * (b.narrow - a.narrow),
            )
    return sections[-1]


def surface_x(sections, y: float, z: float) -> float:
    """Halbe Aussenbreite der Maske an der Stelle (y, z), also die
    X-Koordinate der Aussenhaut auf der rechten Seite.

    Wird gebraucht, um die Blinker-Aussparung exakt auf die Flanke zu setzen.
    """
    sec = interp_section(sections, z)
    half_w, half_h = sec.width / 2.0, sec.height / 2.0
    y_local = y - sec.y_offset
    if half_h <= 0.0:
        return 0.0
    ratio = min(1.0, abs(y_local) / half_h)
    # implizite Superellipse: |x/a|^n + |y/b|^n = 1
    x = half_w * (max(0.0, 1.0 - ratio ** sec.exponent)) ** (1.0 / sec.exponent)
    return x * _narrow_factor(y_local, half_h, sec.narrow)


def offset_points(pts, distances):
    """Geschlossenen Polygonzug nach innen versetzen.

    Jeder Punkt wandert entlang seiner eigenen Normalen — anders als beim
    blossen Verkleinern von Breite und Hoehe bleibt die Wandstaerke damit
    auch in den Ecken des Querschnitts konstant.
    """
    n = len(pts)
    out = []
    for i in range(n):
        px, py = pts[i]
        ax, ay = pts[(i - 1) % n]
        bx, by = pts[(i + 1) % n]
        tx, ty = bx - ax, by - ay
        length = math.hypot(tx, ty)
        if length < 1e-12:
            out.append((px, py))
            continue
        # Umlauf gegen den Uhrzeigersinn -> (ty, -tx) zeigt nach aussen
        nx, ny = ty / length, -tx / length
        d = distances[i] if hasattr(distances, "__len__") else distances
        out.append((px - nx * d, py - ny * d))
    return out


def taper_factors(sections, z: float, num_points: int, delta: float = 0.5):
    """Korrektur der Versatzweite wegen der Formschraege in Laengsrichtung.

    Die Flanke steht schraeg zur Querschnittsebene. Ein Versatz von `w` in
    der Ebene ergibt daher nur `w * cos(theta)` senkrecht zur Haut. Der
    Faktor 1/cos(theta) gleicht das aus.
    """
    lo = superellipse_points(interp_section(sections, z - delta), num_points)
    hi = superellipse_points(interp_section(sections, z + delta), num_points)
    factors = []
    for (x0, y0), (x1, y1) in zip(lo, hi):
        ds = math.hypot(x1 - x0, y1 - y0) / (2.0 * delta)
        factors.append(math.sqrt(1.0 + ds * ds))
    return factors


def inner_points(sections, z: float, wall: float, num_points: int):
    """Punkte der Kavitaetskontur auf Hoehe z."""
    pts = superellipse_points(interp_section(sections, z), num_points)
    factors = taper_factors(sections, z, num_points)
    return offset_points(pts, [wall * f for f in factors])

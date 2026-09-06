"""Kontur- und Flaechengeometrie der Maske — reine Mathematik, kein CAD-Kernel.

Koordinatensystem (durchgaengig, Einheit Millimeter):

    X   quer zur Fahrtrichtung, + = rechte Fahrzeugseite
    Y   hoch,                   + = oben
    Z   laengs,                 0 = hintere Kante (Rand), + = nach vorn

Aufbau der Maske: eine geschlossene Umrisskontur (`outline`, normiert auf
eine Einheitsbox) wird auf mehreren Tiefen `z` unterschiedlich stark
verkleinert und daraus ein Loft gebildet. So entsteht das gewoelbte Schild:
aussen am Rand die volle Kontur, nach vorn hin zunehmend kleiner.

Die Wandstaerke entsteht ueber echte Parallelkurven (`offset_points`), nicht
ueber blosses Skalieren — nur so bleibt sie auch in den Ecken konstant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Level:
    """Eine Ebene des Lofts: verkleinerte Kontur auf Tiefe z."""

    z: float          # Tiefe (0 = hintere Kante, positiv nach vorn)
    scale: float      # Verkleinerung der Kontur, 1.0 = voller Umriss
    y_offset: float   # Verschiebung nach oben, verlagert den Scheitel

    @classmethod
    def from_list(cls, v) -> "Level":
        return cls(*(float(x) for x in v))

    def as_list(self):
        return [self.z, self.scale, self.y_offset]


def outline_points(outline, width: float, height: float,
                   scale: float = 1.0, y_offset: float = 0.0):
    """Normierte Kontur auf Millimeter bringen und skalieren.

    `outline` ist eine Liste [x, y] im Bereich -0.5 … +0.5. Die Skalierung
    wirkt um den Ursprung, `y_offset` verschiebt anschliessend.
    """
    return [(px * width * scale, py * height * scale + y_offset)
            for px, py in outline]


def polygon_area(pts) -> float:
    """Vorzeichenbehaftete Flaeche — positiv bei Umlauf gegen den Uhrzeigersinn."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def ensure_ccw(pts):
    """Kontur gegen den Uhrzeigersinn orientieren.

    `offset_points` setzt diese Orientierung voraus, damit die berechnete
    Normale nach aussen zeigt.
    """
    return list(pts) if polygon_area(pts) > 0 else list(reversed(pts))


def offset_points(pts, distances):
    """Geschlossenen Polygonzug nach innen versetzen.

    Jeder Punkt wandert entlang seiner eigenen Normalen. Anders als beim
    blossen Verkleinern bleibt die Wandstaerke damit auch dort konstant, wo
    die Kontur ihre Richtung stark aendert.
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
        nx, ny = ty / length, -tx / length      # Umlauf CCW -> zeigt nach aussen
        d = distances[i] if hasattr(distances, "__len__") else distances
        out.append((px - nx * d, py - ny * d))
    return out


def interp_level(levels, z: float) -> Level:
    """Ebene auf beliebiger Tiefe z (linear interpoliert bzw. geklemmt).

    `levels` muss nach z aufsteigend sortiert sein (0, 18, 34, …).
    """
    if not levels:
        raise ValueError("keine Ebenen definiert")
    if z <= levels[0].z:
        return Level(z, levels[0].scale, levels[0].y_offset)
    if z >= levels[-1].z:
        return Level(z, levels[-1].scale, levels[-1].y_offset)
    for a, b in zip(levels, levels[1:]):
        if a.z <= z <= b.z:
            span = b.z - a.z
            t = 0.0 if span == 0 else (z - a.z) / span
            return Level(z,
                         a.scale + t * (b.scale - a.scale),
                         a.y_offset + t * (b.y_offset - a.y_offset))
    return levels[-1]


def _slopes(outline, width, height, levels, z: float, delta: float = 0.5):
    """Neigung der Haut je Konturpunkt: Weg in der Ebene pro Weg in Z.

    Negativ, wo die Kontur nach vorn hin kleiner wird.
    """
    lo_lvl = interp_level(levels, z - delta)
    hi_lvl = interp_level(levels, z + delta)
    lo = outline_points(outline, width, height, lo_lvl.scale, lo_lvl.y_offset)
    hi = outline_points(outline, width, height, hi_lvl.scale, hi_lvl.y_offset)
    out = []
    for (x0, y0), (x1, y1) in zip(lo, hi):
        ds = math.hypot(x1 - x0, y1 - y0) / (2.0 * delta)
        sign = -1.0 if math.hypot(x1, y1) < math.hypot(x0, y0) else 1.0
        out.append(sign * ds)
    return out


def offset_folds(original, offset) -> int:
    """Zahl der Kanten, die durch den Versatz ihre Richtung umkehren.

    Groesser als null heisst: die versetzte Kontur hat sich selbst
    ueberholt. Solche Konturen haben oft noch eine positive Flaeche, fallen
    einer reinen Flaechenpruefung also nicht auf — sie zerstoeren aber jeden
    nachfolgenden Loft und jede boolesche Operation darauf.
    """
    n = len(original)
    folds = 0
    for i in range(n):
        j = (i + 1) % n
        ax = original[j][0] - original[i][0]
        ay = original[j][1] - original[i][1]
        bx = offset[j][0] - offset[i][0]
        by = offset[j][1] - offset[i][1]
        if ax * bx + ay * by < 0.0:
            folds += 1
    return folds


def level_points(outline, width, height, levels, z: float):
    """Aussenkontur auf Tiefe z."""
    lvl = interp_level(levels, z)
    return outline_points(outline, width, height, lvl.scale, lvl.y_offset)


def cavity_level(outline, width, height, levels, z: float, wall: float):
    """Ebene der Kavitaet als echte Parallelflaeche: (z_neu, punkte).

    Die Haut steht schraeg zur Querschnittsebene. Wer nur INNERHALB der
    Ebene versetzt, muss das mit 1/cos(theta) ausgleichen — und das waechst
    nahe am Scheitel ueber alle Grenzen, bis die Kontur sich in sich selbst
    faltet.

    Die Flaechennormale hat aber auch einen Anteil in Z. Richtig ist daher:
    in der Ebene um wall*cos(theta) versetzen (also WENIGER als die
    Wandstaerke) und die Ebene um wall*sin(theta) nach hinten schieben. Am
    Scheitel laeuft das sauber gegen "kein Versatz in der Ebene, dafuer die
    volle Wandstaerke nach hinten".
    """
    pts = level_points(outline, width, height, levels, z)
    slopes = _slopes(outline, width, height, levels, z)
    cos_t = [1.0 / math.sqrt(1.0 + m * m) for m in slopes]
    mean_m = sum(slopes) / len(slopes)
    sin_mean = abs(mean_m) / math.sqrt(1.0 + mean_m * mean_m)
    return z - wall * sin_mean, offset_points(pts, [wall * c for c in cos_t])


def surface_x(outline, width, height, levels, y: float, z: float) -> float:
    """X-Koordinate der Aussenhaut auf der rechten Seite bei (y, z).

    Strahlschnitt von (0, y) nach +X mit dem Konturpolygon. Wird gebraucht,
    um Anbauten exakt auf die Flanke zu setzen.
    """
    pts = level_points(outline, width, height, levels, z)
    best = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        if (y0 - y) * (y1 - y) > 0:            # Kante schneidet die Hoehe nicht
            continue
        if abs(y1 - y0) < 1e-12:
            continue
        t = (y - y0) / (y1 - y0)
        x = x0 + t * (x1 - x0)
        if x > best:
            best = x
    return best


def bounds(outline, width, height, levels):
    """Aussenmasse des Lofts: (breite, hoehe, tiefe)."""
    xs, ys = [], []
    for lvl in levels:
        for x, y in outline_points(outline, width, height, lvl.scale, lvl.y_offset):
            xs.append(x)
            ys.append(y)
    return (max(xs) - min(xs), max(ys) - min(ys),
            levels[-1].z - levels[0].z)

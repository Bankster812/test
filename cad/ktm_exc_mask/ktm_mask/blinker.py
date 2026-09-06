"""Seitliche Blinkeraufnahme — die Uebernahme von der Stark-EX-Maske.

Aufgebaut aus vier Elementen je Seite:

  1. Aussparung  flachgefraeste Tasche in der Flanke (der Blinkerfuss taucht ein)
  2. Verstaerkung  Materialaufdickung innen hinter dem Taschenboden
  3. Schaftbohrung  Durchgang fuer den Blinkerschaft
  4. Verdrehsicherung  zweite kleine Bohrung daneben

Die Achse steht nicht senkrecht auf der Flanke, sondern wird ueber
`blinker_yaw_deg` / `blinker_pitch_deg` frei eingestellt — so zeigt der
Blinker nach aussen/vorn statt quer zur Fahrtrichtung.
"""

from __future__ import annotations

import math

import cadquery as cq

from .geometry import surface_x

OUT = 40.0  # wie weit die Schnittkoerper vor der Flanke starten


def axis(p, side: int):
    """Einheitsvektor der Blinkerachse (nach aussen), side = +1 rechts / -1 links."""
    yaw = math.radians(p.blinker_yaw_deg)
    pitch = math.radians(p.blinker_pitch_deg)
    return (
        side * math.cos(pitch) * math.cos(yaw),
        math.sin(pitch),
        math.cos(pitch) * math.sin(yaw),
    )


def mount_point(p, side: int):
    """Punkt auf der Aussenhaut, an dem der Blinker sitzt."""
    x = surface_x(p.section_objs, p.blinker_y, p.blinker_z)
    return (side * x, p.blinker_y, p.blinker_z)


def _plane(p, side: int) -> cq.Plane:
    """Lokale Ebene: Ursprung auf der Flanke, Normale = Blinkerachse."""
    n = cq.Vector(*axis(p, side))
    x_dir = cq.Vector(0, 1, 0).cross(n)
    if x_dir.Length < 1e-9:                     # Achse zeigt senkrecht nach oben
        x_dir = cq.Vector(1, 0, 0)
    return cq.Plane(origin=cq.Vector(*mount_point(p, side)),
                    xDir=x_dir.normalized(), normal=n)


def _pocket_sketch(p):
    r = min(p.pocket_r, min(p.pocket_w, p.pocket_h) / 2.0 - 1e-6)
    sk = cq.Sketch().rect(p.pocket_w, p.pocket_h)
    if r > 0:
        sk = sk.vertices().fillet(r)
    return sk


def pocket_cutter(p, side: int) -> cq.Workplane:
    """Die Aussparung selbst: von aussen bis `pocket_depth` unter die Haut."""
    return (
        cq.Workplane(_plane(p, side))
        .workplane(offset=OUT)
        .placeSketch(_pocket_sketch(p))
        .extrude(-(OUT + p.pocket_depth))
    )


def reinforcement(p, side: int, outer=None) -> cq.Workplane:
    """Verstaerkung hinter der Tasche.

    Zwei Teile: ein Feld ueber die ganze Taschenflaeche (damit der
    Taschenboden ueberhaupt Material hat — die Tasche ist tiefer als die
    Wand stark ist) und darin der Zylinder fuer den Blinkerschaft.

    Das Feld wird von ausserhalb der Haut aufgebaut. Ohne `outer` bleibt es
    ungekuerzt — der Aufrufer verschneidet es dann selbst mit der Aussenhaut
    (beim Scan mit dessen konvexer Huelle), damit es buendig anschliesst.
    """
    plane = _plane(p, side)
    r = min(p.pocket_r + p.pocket_margin,
            min(p.pocket_w + 2 * p.pocket_margin,
                p.pocket_h + 2 * p.pocket_margin) / 2.0 - 1e-6)
    sk = cq.Sketch().rect(p.pocket_w + 2 * p.pocket_margin,
                          p.pocket_h + 2 * p.pocket_margin)
    if r > 0:
        sk = sk.vertices().fillet(r)
    field = (
        cq.Workplane(plane)
        .workplane(offset=OUT)
        .placeSketch(sk)
        .extrude(-(OUT + p.pocket_depth + p.pocket_floor_t))
    )
    if outer is not None:
        field = field.intersect(outer)
    boss = (
        cq.Workplane(plane)
        .workplane(offset=-p.pocket_depth)
        .circle(p.boss_d / 2.0)
        .extrude(-p.boss_depth)
    )
    return field.union(boss)


def hole_cutter(p, side: int) -> cq.Workplane:
    """Schaftbohrung, Verdrehsicherung und optionale Mutterntasche."""
    through = OUT + p.pocket_depth + p.boss_depth + 20.0
    cut = (
        cq.Workplane(_plane(p, side))
        .workplane(offset=OUT)
        .circle(p.bolt_hole_d / 2.0)
        .extrude(-through)
    )
    if p.antirot and p.antirot_d > 0:
        pin = (
            cq.Workplane(_plane(p, side))
            .workplane(offset=OUT)
            .center(0.0, -p.antirot_offset)
            .circle(p.antirot_d / 2.0)
            .extrude(-through)
        )
        cut = cut.union(pin)
    if p.nut_pocket:
        nut = (
            cq.Workplane(_plane(p, side))
            .workplane(offset=-(p.pocket_depth + p.boss_depth - p.nut_depth))
            .polygon(6, p.nut_af / math.cos(math.pi / 6.0))
            .extrude(-(p.nut_depth + 20.0))
        )
        cut = cut.union(nut)
    return cut


def ece_check(p) -> dict:
    """Abstand der leuchtenden Flaechen — Richtwert fuer die ECE-Vorgabe.

    Fuer Krafraeder wird ueblicherweise ein Mindestabstand von 240 mm
    zwischen den Innenkanten der beiden vorderen Fahrtrichtungsanzeiger
    verlangt. Der Wert hier ist eine Rechenhilfe, keine Zulassungsaussage —
    mit der Pruefstelle abstimmen.
    """
    n = axis(p, +1)
    x_surface = surface_x(p.section_objs, p.blinker_y, p.blinker_z)
    x_lens = x_surface + p.blinker_stalk_len * n[0]
    inner = 2.0 * x_lens - p.blinker_lens_d
    needed = p.ece_min_inner_spacing
    missing = max(0.0, needed - inner)
    return {
        "flanke_ab_mitte_mm": round(x_surface, 1),
        "lichtmitte_ab_mitte_mm": round(x_lens, 1),
        "innenkanten_abstand_mm": round(inner, 1),
        "richtwert_mm": needed,
        "erfuellt": inner >= needed,
        "fehlende_schaftlaenge_je_seite_mm": round(
            missing / (2.0 * n[0]) if n[0] > 1e-6 else float("inf"), 1),
    }

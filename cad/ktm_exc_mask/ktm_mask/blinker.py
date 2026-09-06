"""Optionale Schaftaufnahme fuer einen Blinker mit Gewindeschaft.

Die Serienloesung der Spendermaske sind die Schlitze in der Seitenkante
(siehe slots.py). Wer stattdessen einen Blinker mit Gewindeschaft direkt
anschrauben will, schaltet ueber `stalk = true` diese Aufnahme dazu:
flache Tasche in der Flanke, vollflaechig hinterlegt, mit Durchgangsloch
und Verdrehsicherung.
"""

from __future__ import annotations

import math

import cadquery as cq

from .geometry import surface_x

OUT = 40.0   # wie weit die Schnittkoerper vor der Flanke starten


def axis(p, side: int):
    """Einheitsvektor der Blinkerachse nach aussen, side = +1 rechts."""
    yaw = math.radians(p.stalk_yaw_deg)
    pitch = math.radians(p.stalk_pitch_deg)
    return (side * math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.sin(yaw))


def mount_point(p, side: int):
    """Punkt auf der Aussenhaut, an dem der Blinker sitzt."""
    x = surface_x(p.outline_ccw, p.width, p.height, p.level_objs,
                  p.stalk_y, p.stalk_z)
    return (side * x, p.stalk_y, p.stalk_z)


def _plane(p, side: int) -> cq.Plane:
    n = cq.Vector(*axis(p, side))
    x_dir = cq.Vector(0, 1, 0).cross(n)
    if x_dir.Length < 1e-9:
        x_dir = cq.Vector(1, 0, 0)
    return cq.Plane(origin=cq.Vector(*mount_point(p, side)),
                    xDir=x_dir.normalized(), normal=n)


def _rounded(w, h, r) -> cq.Sketch:
    r = max(0.0, min(r, min(w, h) / 2.0 - 1e-6))
    sk = cq.Sketch().rect(w, h)
    return sk.vertices().fillet(r) if r > 0 else sk


def reinforcement(p, side: int, outer=None) -> cq.Workplane:
    """Hinterlegung der Tasche plus Zylinder fuer den Schaft.

    Die Tasche ist tiefer als die Wand stark ist; ohne ein Feld ueber ihre
    ganze Flaeche stuende sie rings um die Schraube offen. Ohne `outer`
    bleibt das Feld ungekuerzt — der Aufrufer verschneidet dann selbst.
    """
    plane = _plane(p, side)
    field = (
        cq.Workplane(plane).workplane(offset=OUT)
        .placeSketch(_rounded(p.pocket_w + 2 * p.pocket_margin,
                              p.pocket_h + 2 * p.pocket_margin,
                              p.pocket_r + p.pocket_margin))
        .extrude(-(OUT + p.pocket_depth + p.pocket_floor_t))
    )
    if outer is not None:
        field = field.intersect(outer)
    boss = (
        cq.Workplane(plane).workplane(offset=-p.pocket_depth)
        .circle(p.boss_d / 2.0).extrude(-p.boss_depth)
    )
    return field.union(boss)


def pocket_cutter(p, side: int) -> cq.Workplane:
    """Die Tasche selbst: von aussen bis `pocket_depth` unter die Haut."""
    return (
        cq.Workplane(_plane(p, side)).workplane(offset=OUT)
        .placeSketch(_rounded(p.pocket_w, p.pocket_h, p.pocket_r))
        .extrude(-(OUT + p.pocket_depth))
    )


def hole_cutter(p, side: int) -> cq.Workplane:
    """Schaftbohrung und Verdrehsicherung."""
    through = OUT + p.pocket_depth + p.boss_depth + 20.0
    cut = (cq.Workplane(_plane(p, side)).workplane(offset=OUT)
           .circle(p.bolt_hole_d / 2.0).extrude(-through))
    if p.antirot and p.antirot_d > 0:
        cut = cut.union(
            cq.Workplane(_plane(p, side)).workplane(offset=OUT)
            .center(0.0, -p.antirot_offset)
            .circle(p.antirot_d / 2.0).extrude(-through))
    return cut


def ece_check(p) -> dict:
    """Abstand der leuchtenden Flaechen — Richtwert fuer die ECE-Vorgabe.

    Fuer vordere Fahrtrichtungsanzeiger am Krad werden ueblicherweise
    240 mm zwischen den Innenkanten verlangt. Rechenhilfe, keine
    Zulassungsaussage — mit der Pruefstelle abstimmen.
    """
    n = axis(p, +1)
    x_surface = surface_x(p.outline_ccw, p.width, p.height, p.level_objs,
                          p.stalk_y, p.stalk_z)
    x_lens = x_surface + p.blinker_stalk_len * n[0]
    inner = 2.0 * x_lens - p.blinker_lens_d
    missing = max(0.0, p.ece_min_inner_spacing - inner)
    return {
        "flanke_ab_mitte_mm": round(x_surface, 1),
        "lichtmitte_ab_mitte_mm": round(x_lens, 1),
        "innenkanten_abstand_mm": round(inner, 1),
        "richtwert_mm": p.ece_min_inner_spacing,
        "erfuellt": inner >= p.ece_min_inner_spacing,
        "fehlende_schaftlaenge_je_seite_mm": round(
            missing / (2.0 * n[0]) if n[0] > 1e-6 else float("inf"), 1),
    }

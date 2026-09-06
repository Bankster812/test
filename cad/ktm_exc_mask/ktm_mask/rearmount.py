"""Rueckseite der Maske — die Uebernahme von der Spendermaske.

Drei austauschbare Varianten ueber `rear_mount_type`:

  "plate"  hinterer Rahmen (Ring) mit angeformten Befestigungslaschen und
           Kabeldurchfuehrung. Das ist die Variante, die einer uebernommenen
           Rueckseite am naechsten kommt.
  "tabs"   nur die Laschen, direkt in der Flanke verankert — weniger Material.
  "strap"  Zapfen fuer Gummibaender, wie bei der EXC-Maske ab Werk.
  "none"   Rueckseite offen lassen.

Alle angeformten Teile werden mit dem massiven Aussenkoerper verschnitten
(`outer`), damit sie buendig in der Aussenhaut sitzen und nicht durchstossen.
"""

from __future__ import annotations

import math

import cadquery as cq

from .geometry import Section, _narrow_factor, interp_section, superellipse_points

OVERLAP = 4.0  # wie weit angeformte Teile in die Wand hineinragen


def _scaled_section(sec: Section, factor: float) -> Section:
    return Section(sec.z, sec.width * factor, sec.height * factor,
                   sec.exponent, sec.y_offset * factor, sec.narrow)


def _prism(sec: Section, num_points: int, z0: float, thickness: float):
    """Prisma aus einem Querschnitt, ab z0 um `thickness` nach vorn."""
    pts = superellipse_points(sec, num_points)
    return (
        cq.Workplane("XY")
        .workplane(offset=z0)
        .spline(pts, periodic=True)
        .close()
        .extrude(thickness)
    )


def _contour_radius(sec: Section, angle_deg: float) -> float:
    """Radius der Querschnittskontur unter einem Winkel (Bisektion)."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    half_w, half_h = sec.width / 2.0, sec.height / 2.0
    lo, hi = 0.0, max(sec.width, sec.height)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        x, y_local = dx * mid, dy * mid - sec.y_offset
        f = _narrow_factor(y_local, half_h, sec.narrow)
        v = (abs(x / (half_w * f)) ** sec.exponent
             + abs(y_local / half_h) ** sec.exponent)
        if v < 1.0:
            lo = mid
        else:
            hi = mid
    return lo


def plate(p) -> cq.Workplane:
    """Hinterer Rahmen mit Innenausschnitt und Kabeldurchfuehrung."""
    rear = p.section_objs[-1]
    z0 = rear.z
    ring = _prism(rear, p.section_points, z0, p.plate_t).cut(
        _prism(_scaled_section(rear, p.plate_aperture), p.section_points,
               z0 - 5.0, p.plate_t + 10.0))

    if p.cable_hole_d > 0:
        y = rear.y_offset + p.cable_hole_y_frac * (rear.height / 2.0)
        ring = ring.cut(
            cq.Workplane("XY").workplane(offset=z0 - 5.0)
            .center(0.0, y).circle(p.cable_hole_d / 2.0)
            .extrude(p.plate_t + 10.0))
    return ring


def tabs(p) -> cq.Workplane:
    """Befestigungslaschen in der hinteren Ebene, nach innen zeigend.

    Bei "plate" setzen sie an der Rahmenoeffnung an, bei "tabs" direkt an der
    Aussenhaut — in beiden Faellen mit Ueberlappung, damit sie verschmelzen.
    """
    rear = p.section_objs[-1]
    z0 = rear.z
    anchor = (_scaled_section(rear, p.plate_aperture)
              if p.rear_mount_type == "plate" else rear)
    result = None
    for ang in (p.tab_angles_deg or [])[: max(0, p.tab_count)]:
        r_out = _contour_radius(anchor, ang) + OVERLAP
        length = p.tab_len + OVERLAP
        r_mid = max(2.0, r_out - length / 2.0)
        cx, cy = r_mid * math.cos(math.radians(ang)), r_mid * math.sin(math.radians(ang))
        pad = (
            cq.Workplane("XY").workplane(offset=z0)
            .center(cx, cy)
            .placeSketch(cq.Sketch().rect(length, p.tab_w)
                         .vertices().fillet(min(p.tab_w, length) / 2.5))
            .extrude(p.tab_t)
            .rotate((cx, cy, z0), (cx, cy, z0 + 1), ang)
        )
        hole_r = max(2.0, r_out - OVERLAP - p.tab_len * 0.45)
        hx, hy = (hole_r * math.cos(math.radians(ang)),
                  hole_r * math.sin(math.radians(ang)))
        pad = pad.cut(
            cq.Workplane("XY").workplane(offset=z0 - 5.0)
            .center(hx, hy).circle(p.tab_hole_d / 2.0)
            .extrude(p.tab_t + 10.0))
        result = pad if result is None else result.union(pad)
    return result


def straps(p):
    """Gummiband-Aufnahme: verstaerktes Feld in der Flanke mit Durchbruch.

    Der Haken des Gummibands greift durch den Schlitz — das ist die
    druckbare und bruchsichere Variante, ein freistehender Zapfen waere
    in Schichtrichtung die Sollbruchstelle.

    Gibt (Verstaerkungen, Durchbrueche) zurueck.
    """
    sec = interp_section(p.section_objs, p.strap_z)
    pads, slots = None, None
    for ang in p.strap_angles_deg:
        a = math.radians(ang)
        r = _contour_radius(sec, ang)
        origin = cq.Vector(r * math.cos(a), r * math.sin(a), p.strap_z)
        normal = cq.Vector(math.cos(a), math.sin(a), 0.0)
        x_dir = cq.Vector(0, 0, 1).cross(normal).normalized()
        plane = cq.Plane(origin=origin, xDir=x_dir, normal=normal)

        pad = (cq.Workplane(plane).workplane(offset=OVERLAP)
               .placeSketch(cq.Sketch().rect(p.strap_pad_w, p.strap_pad_h)
                            .vertices().fillet(min(p.strap_pad_w,
                                                   p.strap_pad_h) / 4.0))
               .extrude(-(OVERLAP + p.strap_pad_depth)))
        through = 20.0 + p.strap_pad_depth + p.wall + 10.0
        slot = (cq.Workplane(plane).workplane(offset=20.0)
                .placeSketch(cq.Sketch().rect(p.strap_slot_w, p.strap_slot_h)
                             .vertices().fillet(p.strap_slot_h / 2.5))
                .extrude(-through))
        pads = pad if pads is None else pads.union(pad)
        slots = slot if slots is None else slots.union(slot)
    return pads, slots
